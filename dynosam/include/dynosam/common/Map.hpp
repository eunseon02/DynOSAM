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
#include "dynosam/backend/BackendDefinitions.hpp" //for all the chr's used in the keys
#include "dynosam/utils/GtsamUtils.hpp"

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace dyno {

class Map {

public:
    DYNO_POINTER_TYPEDEFS(Map)

    void updateObservations(const StatusKeypointMeasurements& keypoint_measurements);

    bool frameExists(FrameId frame_id) const;
    bool trackletExists(TrackletId tracklet_id) const;

    TrackletIds getStaticTrackletsByFrame(FrameId frame_id) const;


private:
    void addOrUpdateMapStructures(TrackletId tracklet_id, FrameId frame_id, ObjectId object_id, bool is_static);

private:
    struct FrameNode;
    struct ObjectNode;
    struct TrackletNode;

    using FrameNodePtr = std::shared_ptr<FrameNode>;
    using ObjectNodePtr = std::shared_ptr<ObjectNode>;
    using TrackletNodePtr = std::shared_ptr<TrackletNode>;

    //TODO: using std::set, prevents the same thign being added multiple times
    //while this is what we want, it may hide a bug, as we never want this situation toa ctually occur

    struct FrameNode {
        DYNO_POINTER_TYPEDEFS(FrameNode)

        FrameId frame_id;
        //set prevents the same feature being added multiple times
        std::set<TrackletNodePtr> dynamic_tracklets;
        std::set<TrackletNodePtr> static_tracklets;
        std::set<ObjectNodePtr> objects_seen;

        bool operator<(const FrameNodePtr& rhs) const {
            return frame_id < rhs->frame_id;
        }
    };

    struct ObjectNode {
        DYNO_POINTER_TYPEDEFS(ObjectNode)

        ObjectId object_id;
        //all tracklets
        std::set<TrackletNodePtr> dynamic_tracklets;

        bool operator<(const ObjectNodePtr& rhs) const {
            return object_id < rhs->object_id;
        }
    };

    struct TrackletNode {
        DYNO_POINTER_TYPEDEFS(TrackletNode)

        TrackletId tracklet_id;
        ObjectId object_id; //will this change ever?
        std::set<FrameNodePtr> frames_seen;

        // bool operator<(const TrackletNodePtr& rhs) const {
        //     return tracklet_id < rhs->tracklet_id;
        // }
    };

    gtsam::FastMap<FrameId, FrameNode::Ptr> frames_;
    gtsam::FastMap<TrackletId, TrackletNode::Ptr> tracklets_;
    gtsam::FastMap<ObjectId, ObjectNode::Ptr> objects_;


};

}
