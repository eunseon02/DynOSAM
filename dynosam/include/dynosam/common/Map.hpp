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

    //TODo: test
    bool getLandmarkObjectId(ObjectId& object_id, TrackletId tracklet_id) const {
        const auto lmk = getLandmark(tracklet_id);
        if(!lmk) { return false; }

        object_id = lmk->getObjectId();
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

    const auto getFrames() const { return frames_; }


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

    // //estimates
    // gtsam::Values values_;
    // gtsam::NonlinearFactorGraph graph_;
    // //initial estimate for each value
    // gtsam::Values initial_;
    FrameId last_estimate_update_{0};
};

using Map2dDepth = Map<KeypointDepth>;
using ObjectNode2dDepth = Map2dDepth::ObjectNodeM;
using LandmarkNode2dDepth = Map2dDepth::LandmarkNodeM;
using FrameNode2dDepth = Map2dDepth::FrameNodeM;


using Map3d2d = Map<LandmarkKeypoint>;
using ObjectNode3d2d = Map3d2d::ObjectNodeM;
using LandmarkNode3d2d = Map3d2d::LandmarkNodeM;
using FrameNode3d2d = Map3d2d::FrameNodeM;


using Map3d = Map<Landmark>;
using ObjectNode3d = Map3d::ObjectNodeM;
using LandmarkNode3d = Map3d::LandmarkNodeM;
using FrameNode3d = Map3d::FrameNodeM;

using Map2d = Map<Keypoint>;
using ObjectNode2d = Map2d::ObjectNodeM;
using LandmarkNode2d = Map2d::LandmarkNodeM;
using FrameNode2d = Map2d::FrameNodeM;

} //dyno
