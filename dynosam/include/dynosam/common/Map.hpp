/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/common/MapNodes.hpp"
#include "dynosam/backend/BackendDefinitions.hpp" //for all the chr's used in the keys
#include "dynosam/common/DynamicObjects.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/logger/Logger.hpp"

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <boost/concept/usage.hpp>
#include <boost/concept/assert.hpp>

#include <type_traits>

namespace dyno {



//Implemented as Best from https://en.cppreference.com/w/cpp/memory/enable_shared_from_this
//So that constructor is only usable by this class. Instead, use create
//Want to use std::enable_shared_from_this<Map>, so that the nodes carry a weak_ptr
//to the Map
template<typename MEASUREMENT = Keypoint>
class Map : public std::enable_shared_from_this<Map<MEASUREMENT>> {
    struct Private{};
public:
    using Measurement = MEASUREMENT;
    using This = Map<Measurement>;


    /// @brief Alias to a GenericTrackedStatusVector using the templated Measurement type,
    /// specifying that StatusVector must contain the desired measurement type
    /// @tparam DERIVEDSTATUS
    template<typename DERIVEDSTATUS>
    using MeasurementStatusVector = GenericTrackedStatusVector<DERIVEDSTATUS, Measurement>;

    using ObjectNodeM = ObjectNode<Measurement>;
    using FrameNodeM = FrameNode<Measurement>;
    using LandmarkNodeM = LandmarkNode<Measurement>;


    DYNO_POINTER_TYPEDEFS(This)

    // Constructor is only usable by this class
    Map(Private) {}

    static std::shared_ptr<This> create()
    {
        return std::make_shared<This>(Private());
    }

    std::shared_ptr<This> getptr()
    {
        //dependant base so need to qualify name lookup with this
        return this->shared_from_this();
    }

    template<typename DERIVEDSTATUS>
    void updateObservations(const MeasurementStatusVector<DERIVEDSTATUS>& measurements) {
        for(const DERIVEDSTATUS& status_measurement : measurements) {
            const TrackedValueStatus<MEASUREMENT>& status = static_cast<const TrackedValueStatus<MEASUREMENT>&>(status_measurement);
            const MEASUREMENT& measurement = status_measurement.value();
            const TrackletId tracklet_id = status.trackletId();
            const FrameId frame_id = status.frameId();
            const ObjectId object_id = status.objectId();
            const bool is_static = status.isStatic();
            addOrUpdateMapStructures(measurement, tracklet_id, frame_id, object_id, is_static);
        }
    }


    void updateEstimates(const gtsam::Values& values, const gtsam::NonlinearFactorGraph& graph, FrameId frame_id) {
        values_.insert_or_assign(values);
        graph_ = graph;
        last_estimate_update_ = frame_id;

        //loop over values and add new ones to initial
        for(const auto& [key, new_value] : values) {
            if(!initial_.exists(key)) {
                initial_.insert(key, new_value);
            }
        }
        CHECK_EQ(values_.size(), initial_.size());
    }

    inline bool frameExists(FrameId frame_id) const { return frames_.exists(frame_id); }
    inline bool landmarkExists(TrackletId tracklet_id) const {return landmarks_.exists(tracklet_id); }
    inline bool objectExists(ObjectId object_id) const {return objects_.exists(object_id); }

    //should these not all be const?!!! dont want to modify the values
    typename ObjectNodeM::Ptr getObject(ObjectId object_id) const {
        if(objectExists(object_id)) { return objects_.at(object_id);}
        return nullptr;
    }

    typename FrameNodeM::Ptr getFrame(FrameId frame_id) const {
        if(frameExists(frame_id)) { return frames_.at(frame_id);}
        return nullptr;
    }
    typename LandmarkNodeM::Ptr getLandmark(TrackletId tracklet_id) const {
        if(landmarkExists(tracklet_id)) { return landmarks_.at(tracklet_id);}
        return nullptr;
    }

    // const typename ObjectNodeM::Ptr getObject(ObjectId object_id) const;
    // const typename FrameNodeM::Ptr getFrame(FrameId frame_id) const;
    // const typename LandmarkNodeM::Ptr getLandmark(TrackletId tracklet_id) const;

    const gtsam::FastMap<TrackletId, typename LandmarkNodeM::Ptr>& getLandmarks() const {
        return landmarks_;
    }

    TrackletIds getStaticTrackletsByFrame(FrameId frame_id) const {
        //if frame does not exist?
        TrackletIds tracklet_ids;
        auto frame_node = frames_.at(frame_id);
        for(const auto& landmark_node : frame_node->static_landmarks) {
            tracklet_ids.push_back(landmark_node->tracklet_id);
        }

        return tracklet_ids;
    }

    //object related queries
    inline size_t numObjectsSeen() const {
        return objects_.size();
    }

    //newest frame
    const typename FrameNodeM::Ptr lastFrame() const { return frames_.crbegin()->second; }
    const typename FrameNodeM::Ptr firstFrame() const { return frames_.cbegin()->second; }

    FrameId lastFrameId() const { return this->lastFrame()->frame_id; }
    FrameId firstFrameId() const { return this->firstFrame()->frame_id; }

    inline FrameId lastEstimateUpdate() const { return last_estimate_update_; }

    StateQuery<gtsam::Pose3> getPoseEstimate(FrameId frame_id) {
        auto frame_node = getFrame(frame_id);
        if(frame_node) {
            return frame_node->getPoseEstimate();
        }
        else {
            return StateQuery<gtsam::Pose3>::NotInMap(CameraPoseSymbol(frame_id));
        }
    }

    //TODo: test
    bool getLandmarkObjectId(ObjectId& object_id, TrackletId tracklet_id) const {
        const auto lmk = getLandmark(tracklet_id);
        if(!lmk) { return false; }

        object_id = lmk->getObjectId();
        return true;
    }

    //landmark related queries
    //TODO: test
    StatusLandmarkEstimates getFullStaticMap() const {
         //dont go over the frames as this contains references to the landmarks multiple times
        //e.g. the ones seen in that frame
        StatusLandmarkEstimates estimates;
        for(const auto&[tracklet_id, landmark_node] : landmarks_) {
            (void)tracklet_id;
            if(landmark_node->isStatic()) {
                landmark_node->appendStaticLandmarkEstimate(estimates);
            }
        }
        return estimates;
    }

    StatusLandmarkEstimates getDynamicMap(FrameId frame_id) const {
        return this->getFrame(frame_id)->getAllDynamicLandmarkEstimates();
    }

    StatusLandmarkEstimates getStaticMap(FrameId frame_id) const {
        return CHECK_NOTNULL(this->getFrame(frame_id))->getAllStaticLandmarkEstimates();
    }

    MotionEstimateMap getMotionEstimates(FrameId frame_id) const {
        MotionEstimateMap motion_estimates;
        const auto frame_k_node = this->getFrame(frame_id);

        if(!frame_k_node) {
            return motion_estimates;
        }
        for(const auto& object_node : frame_k_node->objects_seen) {
            StateQuery<gtsam::Pose3> motion_query = object_node->getMotionEstimate(frame_id);

            //hardcoded motion reference frame
            if(motion_query) {
                motion_estimates.insert2(
                    object_node->getId(),
                    ReferenceFrameValue<gtsam::Pose3>(
                        motion_query.get(),
                        ReferenceFrame::GLOBAL
                    ));
            }
        }
        return motion_estimates;
    }

    ObjectPoseMap composeEstimatedObjectPoseMap() const {
        ObjectPoseMap object_poses;
        auto frame_itr = frames_.begin();

        for(auto itr = frame_itr; itr != frames_.end(); itr++) {
            const auto[frame_id_k, frame_k_ptr] = *itr;
            const auto pose_estimates = frame_k_ptr->getPoseEstimates();

            for(const auto& [object_id, pose] : pose_estimates) {

                if(!object_poses.exists(object_id)) {
                    object_poses.insert2(object_id, gtsam::FastMap<FrameId, gtsam::Pose3>{});
                }
                object_poses[object_id].insert2(frame_id_k, pose);

            }
        }
        return object_poses;

    }


    //recomputes every time
    ObjectPoseMap computeComposedObjectPoseMap(std::optional<GroundTruthPacketMap> gt_packet_map = {}) const {
        ObjectPoseMap object_poses;
        auto frame_itr = frames_.begin();
        //advance itr one so we're now at the second frame
        std::advance(frame_itr, 1);
        for(auto itr = frame_itr; itr != frames_.end(); itr++) {
            auto prev_itr = itr;
            std::advance(prev_itr, -1);
            CHECK(prev_itr != frames_.end());

            const auto[frame_id_k, frame_k_ptr] = *itr;
            const auto[frame_id_k_1, frame_k_1_ptr] = *prev_itr;
            CHECK_EQ(frame_id_k_1 + 1, frame_id_k);

            //collect all object centoids from the latest estimate
            gtsam::FastMap<ObjectId, gtsam::Point3> centroids_k = frame_k_ptr->computeObjectCentroids();
            gtsam::FastMap<ObjectId, gtsam::Point3> centroids_k_1 = frame_k_1_ptr->computeObjectCentroids();
            //collect motions
            MotionEstimateMap motion_estimates = frame_k_ptr->getMotionEstimates();

            //construct centroid vectors in object id order
            gtsam::Point3Vector object_centroids_k_1, object_centroids_k;
            //we may not have a pose for every motion, e.g. if there is a new object at frame k,
            //it wont have a motion yet!
            //if we have a motion we MUST have a pose at the previous frame!
            for(const auto& [object_id, _] : motion_estimates) {
                CHECK(centroids_k.exists(object_id));
                CHECK(centroids_k_1.exists(object_id));

               object_centroids_k.push_back(centroids_k.at(object_id));
               object_centroids_k_1.push_back(centroids_k_1.at(object_id));
            }

            if(FLAGS_init_object_pose_from_gt) {
                if(!gt_packet_map) LOG(WARNING) << "FLAGS_init_object_pose_from_gt is true but gt_packet map not provided!";
                dyno::propogateObjectPoses(
                    object_poses,
                    motion_estimates,
                    object_centroids_k_1,
                    object_centroids_k,
                    frame_id_k,
                    gt_packet_map);
            }
            else {
                dyno::propogateObjectPoses(
                    object_poses,
                    motion_estimates,
                    object_centroids_k_1,
                    object_centroids_k,
                    frame_id_k);
            }

        }

        return object_poses;

    }


    template<typename ValueType>
    StateQuery<ValueType> query(gtsam::Key key) const {
        if(values_.exists(key)) {
            return StateQuery<ValueType>(key, values_.at<ValueType>(key));
        }
        else {
            return StateQuery<ValueType>::NotInMap(key);
        }
    }

    inline const gtsam::Values& getValues() const { return values_; }
    //TODO: this should be the FULL graph but need to verify how we update this (ie.e when optimization happens and then how we update the estimates)
    inline const gtsam::NonlinearFactorGraph& getGraph() const { return graph_; }
    inline const gtsam::Values& getInitialValues() const { return initial_; }

    bool exists(gtsam::Key key, const gtsam::Values& new_values = gtsam::Values()) const {
        return (values_.exists(key) || new_values.exists(key));
    }

    template<typename ValueType>
    ValueType at(gtsam::Key key, const gtsam::Values& new_values = gtsam::Values()) const {
        StateQuery<ValueType> state_query = this->query<ValueType>(key);
        if(state_query) {
            return state_query.get();
        }

        if(new_values.exists(key)) {
            return new_values.at<ValueType>(key);
        }

        throw gtsam::ValuesKeyDoesNotExist("Requesting value from dyno::Map::at", key);
    }

    template<typename ValueType>
    bool safeGet(ValueType& value, gtsam::Key key, const gtsam::Values& new_values = gtsam::Values()) {
        if(!this->exists(key, new_values)) {
            return false;
        }

        value = this->at<ValueType>(key, new_values);
        return true;
    }

    ObjectIds getObjectIds() const {
        ObjectIds object_ids;
        for(const auto& [object_id, _] : objects_) {
            object_ids.push_back(object_id);
        }
        return object_ids;
    }

    FrameIds getFrameIds() const {
        FrameIds frame_ids;
        for(const auto& [frame_id, _] : frames_) {
            frame_ids.push_back(frame_id);
        }
        return frame_ids;
    }


private:
    void addOrUpdateMapStructures(const Measurement& measurement, TrackletId tracklet_id, FrameId frame_id, ObjectId object_id, bool is_static) {
        typename LandmarkNodeM::Ptr  landmark_node = nullptr;
        typename FrameNodeM::Ptr frame_node = nullptr;

        CHECK((is_static && object_id == background_label) || (!is_static && object_id != background_label));

        if(!landmarkExists(tracklet_id)) {
            landmark_node = std::make_shared<LandmarkNodeM>(getptr());
            CHECK_NOTNULL(landmark_node);
            landmark_node->tracklet_id = tracklet_id;
            landmark_node->object_id = object_id;
            landmarks_.insert2(tracklet_id, landmark_node);
        }

        if(!frameExists(frame_id)) {
            frame_node = std::make_shared<FrameNodeM>(getptr());
            frame_node->frame_id = frame_id;
            frames_.insert2(frame_id, frame_node);
        }

        landmark_node = getLandmark(tracklet_id);
        frame_node = getFrame(frame_id);

        CHECK(landmark_node);
        CHECK(frame_node);

        CHECK_EQ(landmark_node->tracklet_id, tracklet_id);
        //this might fail of a tracklet get associated with a different object
        CHECK_EQ(landmark_node->object_id, object_id);
        CHECK_EQ(frame_node->frame_id, frame_id);

        landmark_node->add(frame_node, measurement);

        //add to frame
        if(is_static) {
            frame_node->static_landmarks.insert(landmark_node);
        }
        else {
            CHECK(object_id != background_label);

            typename ObjectNodeM::Ptr object_node = nullptr;
            if(!objectExists(object_id)) {
                object_node = std::make_shared<ObjectNodeM>(getptr());
                object_node->object_id = object_id;
                objects_.insert2(object_id, object_node);
            }

            object_node = getObject(object_id);
            CHECK(object_node);

            object_node->dynamic_landmarks.insert(landmark_node);

            frame_node->dynamic_landmarks.insert(landmark_node);
            frame_node->objects_seen.insert(object_node);

        }
    }

private:
    //nodes
    gtsam::FastMap<FrameId, typename FrameNodeM::Ptr> frames_;
    gtsam::FastMap<TrackletId, typename LandmarkNodeM::Ptr> landmarks_;
    gtsam::FastMap<ObjectId, typename ObjectNodeM::Ptr> objects_;

    //estimates
    gtsam::Values values_;
    gtsam::NonlinearFactorGraph graph_;
    //initial estimate for each value
    gtsam::Values initial_;
    FrameId last_estimate_update_{0};
};



using Map3d = Map<Landmark>;
using ObjectNode3d = Map3d::ObjectNodeM;
using LandmarkNode3d = Map3d::LandmarkNodeM;
using FrameNode3d = Map3d::FrameNodeM;

using Map2d = Map<Keypoint>;
using ObjectNode2d = Map2d::ObjectNodeM;
using LandmarkNode2d = Map2d::LandmarkNodeM;
using FrameNode2d = Map2d::FrameNodeM;

} //dyno
