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
    // using StaticLandmarkNodeM = StaticLandmarkNode<Measurement>;
    // using DynamicLandmarkNodeM = DynamicLandmarkNode<Measurement>;


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
    void updateObservations(const MeasurementStatusVector<DERIVEDSTATUS>& measurements);


    void updateEstimates(const gtsam::Values& values, const gtsam::NonlinearFactorGraph& graph, FrameId frame_id);

    bool frameExists(FrameId frame_id) const;
    bool landmarkExists(TrackletId tracklet_id) const;
    bool objectExists(ObjectId object_id) const;

    //should these not all be const?!!! dont want to modify the values
    typename ObjectNodeM::Ptr getObject(ObjectId object_id);
    typename FrameNodeM::Ptr getFrame(FrameId frame_id);
    typename LandmarkNodeM::Ptr getLandmark(TrackletId tracklet_id);

    const typename ObjectNodeM::Ptr getObject(ObjectId object_id) const;
    const typename FrameNodeM::Ptr getFrame(FrameId frame_id) const;
    const typename LandmarkNodeM::Ptr getLandmark(TrackletId tracklet_id) const;

    const gtsam::FastMap<TrackletId, typename LandmarkNodeM::Ptr>& getLandmarks() const {
        return landmarks_;
    }

    TrackletIds getStaticTrackletsByFrame(FrameId frame_id) const;

    //object related queries
    size_t numObjectsSeen() const;

    const typename FrameNodeM::Ptr lastFrame() const;

    FrameId lastEstimateUpdate() const;

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
    bool getLandmarkObjectId(ObjectId& object_id, TrackletId tracklet_id) const;

    //landmark related queries
    //TODO: test
    StatusLandmarkEstimates getFullStaticMap() const;
    StatusLandmarkEstimates getDynamicMap(FrameId frame_id) const;
    StatusLandmarkEstimates getStaticMap(FrameId frame_id) const;


    MotionEstimateMap getMotionEstimates(FrameId frame_id) const;


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
        for(const auto& [object_id, object_node] : objects_) {
            (void)object_node;
            object_ids.push_back(object_id);
        }
        return object_ids;
    }


private:
    void addOrUpdateMapStructures(const Measurement& measurement, TrackletId tracklet_id, FrameId frame_id, ObjectId object_id, bool is_static);

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


#include "dynosam/common/Map-inl.hpp"
