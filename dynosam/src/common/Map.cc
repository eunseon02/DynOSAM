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

#include "dynosam/common/Map.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/backend/DynamicPointSymbol.hpp"

#include <glog/logging.h>

namespace dyno {

void Map::updateObservations(const StatusKeypointMeasurements& keypoint_measurements) {
    for(const StatusKeypointMeasurement& status_measurement : keypoint_measurements) {
        const KeypointStatus& status = status_measurement.first;
        const auto& measurement = status_measurement.second;

        const TrackletId tracklet_id = measurement.first;
        const FrameId frame_id = status.frame_id_;
        const ObjectId object_id = status.label_;
        const bool is_static = status.isStatic();

        addOrUpdateMapStructures(tracklet_id, frame_id, object_id, is_static);
    }

}

bool Map::frameExists(FrameId frame_id) const {
    return frames_.exists(frame_id);
}
bool Map::trackletExists(TrackletId tracklet_id) const {
    return tracklets_.exists(tracklet_id);
}

TrackletIds Map::getStaticTrackletsByFrame(FrameId frame_id) const {
    //if frame does not exist?
    TrackletIds tracklet_ids;
    auto frame_node = frames_.at(frame_id);
    for(const auto& tracklet_node : frame_node->static_tracklets) {
        tracklet_ids.push_back(tracklet_node->tracklet_id);
    }

    return tracklet_ids;
}

void Map::addOrUpdateMapStructures(TrackletId tracklet_id, FrameId frame_id, ObjectId object_id, bool is_static) {
    TrackletNodePtr tracklet_node = nullptr;
    FrameNodePtr frame_node = nullptr;

    CHECK((is_static && object_id == background_label) || (!is_static && object_id != background_label));


    if(!tracklets_.exists(tracklet_id)) {
        tracklet_node = std::make_shared<TrackletNode>();
        tracklet_node->tracklet_id = tracklet_id;
        tracklet_node->object_id = object_id;
        tracklets_.insert2(tracklet_id, tracklet_node);
    }

    if(!frames_.exists(frame_id)) {
        frame_node = std::make_shared<FrameNode>();
        frame_node->frame_id = frame_id;

        frames_.insert2(frame_id, frame_node);
    }

    tracklet_node = tracklets_.at(tracklet_id);
    frame_node = frames_.at(frame_id);

    CHECK(tracklet_node);
    CHECK(frame_node);

    CHECK_EQ(tracklet_node->tracklet_id, tracklet_id);
    //this might fail of a tracklet get associated with a different object
    CHECK_EQ(tracklet_node->object_id, object_id);
    CHECK_EQ(frame_node->frame_id, frame_id);

    tracklet_node->frames_seen.insert(frame_node);

    //add to frame
    if(is_static) {
        frame_node->static_tracklets.insert(tracklet_node);
    }
    else {
        CHECK(object_id != background_label);

        ObjectNode::Ptr object_node = nullptr;
        if(!objects_.exists(object_id)) {
            object_node = std::make_shared<ObjectNode>();
            object_node->object_id = object_id;
            objects_.insert2(object_id, object_node);
        }

        object_node = objects_.at(object_id);
        CHECK(object_node);

        object_node->dynamic_tracklets.insert(tracklet_node);
        frame_node->dynamic_tracklets.insert(tracklet_node);

    }

}


} //dyno
