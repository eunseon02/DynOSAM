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

#include "dynosam/common/Map.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/backend/DynamicPointSymbol.hpp"
#include "dynosam/frontend/vision/Frame.hpp"

#include "dynosam/utils/SafeCast.hpp"

#include <glog/logging.h>

namespace dyno {

template<typename MEASUREMENT>
template<typename DERIVEDSTATUS>
void Map<MEASUREMENT>::updateObservations(const MeasurementStatusVector<DERIVEDSTATUS>& measurements) {
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

template<typename MEASUREMENT>
void Map<MEASUREMENT>::updateEstimates(const gtsam::Values& values, const gtsam::NonlinearFactorGraph& graph, FrameId frame_id) {
    values_.insert_or_assign(values);
    graph_ = graph;
    last_estimate_update_ = frame_id;
}


template<typename MEASUREMENT>
bool Map<MEASUREMENT>::frameExists(FrameId frame_id) const {
    return frames_.exists(frame_id);
}

template<typename MEASUREMENT>
bool Map<MEASUREMENT>::landmarkExists(TrackletId tracklet_id) const {
    return landmarks_.exists(tracklet_id);
}

template<typename MEASUREMENT>
bool Map<MEASUREMENT>::objectExists(ObjectId object_id) const {
    return objects_.exists(object_id);
}

template<typename MEASUREMENT>
typename Map<MEASUREMENT>::ObjectNodeM::Ptr Map<MEASUREMENT>::getObject(ObjectId object_id) {
    if(objectExists(object_id)) { return objects_.at(object_id);}
    return nullptr;
}

template<typename MEASUREMENT>
typename Map<MEASUREMENT>::FrameNodeM::Ptr Map<MEASUREMENT>::getFrame(FrameId frame_id) {
    if(frameExists(frame_id)) { return frames_.at(frame_id);}
    return nullptr;
}

template<typename MEASUREMENT>
typename Map<MEASUREMENT>::LandmarkNodeM::Ptr Map<MEASUREMENT>::getLandmark(TrackletId tracklet_id) {
    if(landmarkExists(tracklet_id)) { return landmarks_.at(tracklet_id);}
    return nullptr;
}

template<typename MEASUREMENT>
const typename Map<MEASUREMENT>::ObjectNodeM::Ptr Map<MEASUREMENT>::getObject(ObjectId object_id) const {
    if(objectExists(object_id)) { return objects_.at(object_id);}
    return nullptr;
}

template<typename MEASUREMENT>
const typename Map<MEASUREMENT>::FrameNodeM::Ptr Map<MEASUREMENT>::getFrame(FrameId frame_id) const {
    if(frameExists(frame_id)) { return frames_.at(frame_id);}
    return nullptr;
}

template<typename MEASUREMENT>
const typename Map<MEASUREMENT>::LandmarkNodeM::Ptr Map<MEASUREMENT>::getLandmark(TrackletId tracklet_id) const {
    if(landmarkExists(tracklet_id)) { return landmarks_.at(tracklet_id);}
    return nullptr;
}

template<typename MEASUREMENT>
TrackletIds Map<MEASUREMENT>::getStaticTrackletsByFrame(FrameId frame_id) const {
    //if frame does not exist?
    TrackletIds tracklet_ids;
    auto frame_node = frames_.at(frame_id);
    for(const auto& landmark_node : frame_node->static_landmarks) {
        tracklet_ids.push_back(landmark_node->tracklet_id);
    }

    return tracklet_ids;
}

template<typename MEASUREMENT>
size_t Map<MEASUREMENT>::numObjectsSeen() const {
    return objects_.size();
}

template<typename MEASUREMENT>
const typename Map<MEASUREMENT>::FrameNodeM::Ptr Map<MEASUREMENT>::lastFrame() const {
    return frames_.crbegin()->second;
}

template<typename MEASUREMENT>
FrameId Map<MEASUREMENT>::lastEstimateUpdate() const {
    return last_estimate_update_;
}

template<typename MEASUREMENT>
bool Map<MEASUREMENT>::getLandmarkObjectId(ObjectId& object_id, TrackletId tracklet_id) const {
    const auto lmk = getLandmark(tracklet_id);

    if(!lmk) { return false; }

    object_id = lmk->getObjectId();
    return true;
}

template<typename MEASUREMENT>
StatusLandmarkEstimates Map<MEASUREMENT>::getFullStaticMap() const {
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

template<typename MEASUREMENT>
StatusLandmarkEstimates Map<MEASUREMENT>::getDynamicMap(FrameId frame_id) const {
    return this->getFrame(frame_id)->getAllDynamicLandmarkEstimates();
}

template<typename MEASUREMENT>
void Map<MEASUREMENT>::addOrUpdateMapStructures(const Measurement& measurement, TrackletId tracklet_id, FrameId frame_id, ObjectId object_id, bool is_static) {
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


} //dyno
