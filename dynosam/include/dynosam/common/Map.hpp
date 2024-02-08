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

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <boost/concept/usage.hpp>
#include <boost/concept/assert.hpp>

#include <type_traits>

namespace dyno {



//TODO: state node

//Implemented as Best from https://en.cppreference.com/w/cpp/memory/enable_shared_from_this
//So that constructor is only usable by this class. Instead, use create
//Want to use std::enable_shared_from_this<Map>, so that the nodes carry a weak_ptr
//to the Map
class Map : public std::enable_shared_from_this<Map> {
    struct Private{};
public:
    DYNO_POINTER_TYPEDEFS(Map)

    // Constructor is only usable by this class
    Map(Private) {}

    static std::shared_ptr<Map> create()
    {
        return std::make_shared<Map>(Private());
    }

    std::shared_ptr<Map> getptr()
    {
        return shared_from_this();
    }

    void updateObservations(const StatusKeypointMeasurements& keypoint_measurements);
    void updateEstimates(const gtsam::Values& values, const gtsam::NonlinearFactorGraph& graph, FrameId frame_id);

    bool frameExists(FrameId frame_id) const;
    bool landmarkExists(TrackletId tracklet_id) const;
    bool objectExists(ObjectId object_id) const;

    //should these not all be const?!!! dont want to modify the values
    ObjectNode::Ptr getObject(ObjectId object_id);
    FrameNode::Ptr getFrame(FrameId frame_id);
    LandmarkNode::Ptr getLandmark(TrackletId tracklet_id);

    const ObjectNode::Ptr getObject(ObjectId object_id) const;
    const FrameNode::Ptr getFrame(FrameId frame_id) const;
    const LandmarkNode::Ptr getLandmark(TrackletId tracklet_id) const;

    TrackletIds getStaticTrackletsByFrame(FrameId frame_id) const;

    //object related queries
    size_t numObjectsSeen() const;


    //TODO:test
    const FrameNode::Ptr lastFrame() const;

    FrameId lastEstimateUpdate() const;


    //TODo: test
    bool getLandmarkObjectId(ObjectId& object_id, TrackletId tracklet_id) const;

    template<typename ValueType>
    StateQuery<ValueType> query(gtsam::Key key) const {
        if(values_.exists(key)) {
            return StateQuery<ValueType>(key, values_.at<ValueType>(key));
        }
        else {
            return StateQuery<ValueType>::NotInMap(key);
        }
    }

    const gtsam::Values& getValues() const;
    const gtsam::NonlinearFactorGraph& getGraph() const;




private:
    void addOrUpdateMapStructures(TrackletId tracklet_id, FrameId frame_id, ObjectId object_id, bool is_static);

//TODO: for now
private:
    //nodes
    gtsam::FastMap<FrameId, FrameNode::Ptr> frames_;
    gtsam::FastMap<TrackletId, LandmarkNode::Ptr> landmarks_;
    gtsam::FastMap<ObjectId, ObjectNode::Ptr> objects_;

    //estimates
    gtsam::Values values_;
    gtsam::NonlinearFactorGraph graph_;
    FrameId last_estimate_update_{0};
};


/**
 * @brief Free helper function to query the Map using a weak_ptr.
 *
 * Useful as the MapNodes carry a weak_ptr to the map.
 *
 * @tparam ValueType
 * @param map_ptr
 * @param key
 * @return std::optional<ValueType>
 */
template<typename ValueType>
StateQuery<ValueType> queryWeakMap(const std::weak_ptr<Map> map_ptr, gtsam::Key key) {
    if(auto map = map_ptr.lock()) {
        return map->query<ValueType>(key);
    }
    return StateQuery<ValueType>::InvalidMap();
}


}
