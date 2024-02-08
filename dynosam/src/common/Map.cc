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

void Map::updateEstimates(const gtsam::Values& values, const gtsam::NonlinearFactorGraph& graph, FrameId frame_id) {
    values_.insert_or_assign(values);
    graph_ = graph;
    last_estimate_update_ = frame_id;
}

bool Map::frameExists(FrameId frame_id) const {
    return frames_.exists(frame_id);
}
bool Map::landmarkExists(TrackletId tracklet_id) const {
    return landmarks_.exists(tracklet_id);
}

bool Map::objectExists(ObjectId object_id) const {
    return objects_.exists(object_id);
}

ObjectNode::Ptr Map::getObject(ObjectId object_id) {
    if(objectExists(object_id)) { return objects_.at(object_id);}
    return nullptr;
}

FrameNode::Ptr Map::getFrame(FrameId frame_id) {
    if(frameExists(frame_id)) { return frames_.at(frame_id);}
    return nullptr;
}
LandmarkNode::Ptr Map::getLandmark(TrackletId tracklet_id) {
    if(landmarkExists(tracklet_id)) { return landmarks_.at(tracklet_id);}
    return nullptr;
}

const ObjectNode::Ptr Map::getObject(ObjectId object_id) const {
    if(objectExists(object_id)) { return objects_.at(object_id);}
    return nullptr;
}
const FrameNode::Ptr Map::getFrame(FrameId frame_id) const {
    if(frameExists(frame_id)) { return frames_.at(frame_id);}
    return nullptr;
}
const LandmarkNode::Ptr Map::getLandmark(TrackletId tracklet_id) const {
    if(landmarkExists(tracklet_id)) { return landmarks_.at(tracklet_id);}
    return nullptr;
}

TrackletIds Map::getStaticTrackletsByFrame(FrameId frame_id) const {
    //if frame does not exist?
    TrackletIds tracklet_ids;
    auto frame_node = frames_.at(frame_id);
    for(const auto& landmark_node : frame_node->static_landmarks) {
        tracklet_ids.push_back(landmark_node->tracklet_id);
    }

    return tracklet_ids;
}

size_t Map::numObjectsSeen() const {
    return objects_.size();
}

const FrameNode::Ptr Map::lastFrame() const {
    return frames_.crbegin()->second;
}

FrameId Map::lastEstimateUpdate() const {
    return last_estimate_update_;
}

bool Map::getLandmarkObjectId(ObjectId& object_id, TrackletId tracklet_id) const {
    const auto lmk = getLandmark(tracklet_id);

    if(!lmk) { return false; }

    object_id = lmk->getObjectId();
    return true;
}

void Map::addOrUpdateMapStructures(TrackletId tracklet_id, FrameId frame_id, ObjectId object_id, bool is_static) {
    LandmarkNodePtr landmark_node = nullptr;
    FrameNodePtr frame_node = nullptr;

    CHECK((is_static && object_id == background_label) || (!is_static && object_id != background_label));


    if(!landmarkExists(tracklet_id)) {
        landmark_node = std::make_shared<LandmarkNode>(getptr());
        landmark_node->tracklet_id = tracklet_id;
        landmark_node->object_id = object_id;
        landmarks_.insert2(tracklet_id, landmark_node);
    }

    if(!frameExists(frame_id)) {
        frame_node = std::make_shared<FrameNode>(getptr());
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

    landmark_node->frames_seen.insert(frame_node);

    //add to frame
    if(is_static) {
        frame_node->static_landmarks.insert(landmark_node);
    }
    else {
        CHECK(object_id != background_label);

        ObjectNode::Ptr object_node = nullptr;
        if(!objectExists(object_id)) {
            object_node = std::make_shared<ObjectNode>(getptr());
            object_node->object_id = object_id;
            objects_.insert2(object_id, object_node);
        }

        object_node = getObject(object_id);
        CHECK(object_node);

        object_node->dynamic_landmarks.insert(landmark_node);
        frame_node->dynamic_landmarks.insert(landmark_node);

    }

}


} //dyno
