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

#include "dynosam/common/MapNodes.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"


namespace dyno {


template<typename MEASUREMENT>
int FrameNode<MEASUREMENT>::getId() const {
    return (int)frame_id;
}

template<typename MEASUREMENT>
bool FrameNode<MEASUREMENT>::objectObserved(ObjectId object_id) const {
    return objects_seen.exists(object_id);
}

template<typename MEASUREMENT>
StateQuery<gtsam::Pose3> FrameNode<MEASUREMENT>::getPoseEstimate() const {
    const gtsam::Key pose_key = CameraPoseSymbol(frame_id);
    return this->map_ptr_->template query<gtsam::Pose3>(pose_key);

}

template<typename MEASUREMENT>
StateQuery<gtsam::Pose3> FrameNode<MEASUREMENT>::getObjectMotionEstimate(ObjectId object_id) const {
    const gtsam::Key motion_key = ObjectMotionSymbol(object_id, frame_id);
    return this->map_ptr_->template query<gtsam::Pose3>(motion_key);
}

template<typename MEASUREMENT>
StateQuery<Landmark> FrameNode<MEASUREMENT>::getDynamicLandmarkEstimate(TrackletId tracklet_id) const {
    auto iter = dynamic_landmarks.find(tracklet_id);
    CHECK(iter != dynamic_landmarks.end()) << "Requested dynamic landmark with id " << tracklet_id << " does not exist at frame " << frame_id;

    const LandmarkNodePtr<MEASUREMENT> lmk_node = *iter;

    return lmk_node->getDynamicLandmarkEstimate(frame_id);

}

template<typename MEASUREMENT>
StatusLandmarkEstimates FrameNode<MEASUREMENT>::getAllDynamicLandmarkEstimates() const {
    //quicker to go through the dynamic_landmarks than the objects as we already know they exist
    //in this frame
    StatusLandmarkEstimates estimates;
    for(const auto& lmk_ptr : dynamic_landmarks) {
        StateQuery<Landmark> lmk_status_query = lmk_ptr->getDynamicLandmarkEstimate(frame_id);
        if(lmk_status_query) {
            estimates.push_back(
                LandmarkStatus::Dynamic(
                    lmk_status_query.get(), //estimate
                    frame_id,
                    lmk_ptr->getId(), //tracklet id
                    lmk_ptr->getObjectId(),
                    LandmarkStatus::Method::OPTIMIZED
                ) //status
            );

        }
    }

    return estimates;
}

template<typename MEASUREMENT>
StatusLandmarkEstimates FrameNode<MEASUREMENT>::getDynamicLandmarkEstimates(ObjectId object_id) const {
    CHECK(objectObserved(object_id)); //TODO: handle exception?

    //not sure what will be faster? going through all landmarks and finding object id
    //or going to this object and iterating over lmks there.
    //Doing the former requires iterating over all the seen lmks (which should be in this frame)
    //and then just checking if the lmk has the right object
    //Doing the latter requires a call to getSeenFrames() and THEN finding the right frame in that set. This means
    //iterating over all the lmks for that dynamic object...
    StatusLandmarkEstimates estimates;
    for(const auto& lmk_ptr : dynamic_landmarks) {
        if(lmk_ptr->getObjectId() == object_id) {
            StateQuery<Landmark> lmk_status_query = lmk_ptr->getDynamicLandmarkEstimate(frame_id);
            if(lmk_status_query) {
                estimates.push_back(
                    LandmarkStatus::Dynamic(
                        lmk_status_query.get(), //estimate
                        frame_id,
                        lmk_ptr->getId(), //tracklet id
                        lmk_ptr->getObjectId(),
                        LandmarkStatus::Method::OPTIMIZED
                    ) //status
                );
            }
        }
    }
    return estimates;
}

template<typename MEASUREMENT>
StateQuery<Landmark> FrameNode<MEASUREMENT>::getStaticLandmarkEstimate(TrackletId tracklet_id) const {
    auto iter = static_landmarks.find(tracklet_id);
    CHECK(iter != static_landmarks.end()) << "Requested static landmark with id " << tracklet_id << " does not exist at frame " << frame_id;

    const LandmarkNodePtr<MEASUREMENT> lmk_node = *iter;

    return lmk_node->getStaticLandmarkEstimate();
}


/// LandmarkNode
template<typename MEASUREMENT>
int LandmarkNode<MEASUREMENT>::getId() const {
    return (int)tracklet_id;
}

template<typename MEASUREMENT>
ObjectId LandmarkNode<MEASUREMENT>::getObjectId() const {
    return object_id;
}

template<typename MEASUREMENT>
bool LandmarkNode<MEASUREMENT>::isStatic() const {
    return object_id == background_label;
}

template<typename MEASUREMENT>
size_t LandmarkNode<MEASUREMENT>::numObservations() const {
    return frames_seen_.size();
}

template<typename MEASUREMENT>
void LandmarkNode<MEASUREMENT>::add(FrameNodePtr<MEASUREMENT> frame_node, const MEASUREMENT& measurement) {
    frames_seen_.insert(frame_node);

    //add measurement to map
    //first check that we dont already have a measurement at this frame
    if(measurements_.exists(frame_node)) {
        throw DynosamException("Unable to add new measurement to landmark node " + std::to_string(this->getId()) + " at frame " + std::to_string(frame_node->getId()) + " as a measurement already exists at this frame!");
    }

    measurements_.insert2(frame_node, measurement);

    CHECK_EQ(frames_seen_.size(), measurements_.size());
}


template<typename MEASUREMENT>
StateQuery<Landmark> LandmarkNode<MEASUREMENT>::getStaticLandmarkEstimate() const {
    const gtsam::Key lmk_key = StaticLandmarkSymbol(this->tracklet_id);

    if(!this->isStatic()) {
        throw InvalidLandmarkQuery(lmk_key, "Static estimate requested but landmark is dynamic!");
    }

    return this->map_ptr_->template query<Landmark>(lmk_key);
}

template<typename MEASUREMENT>
StateQuery<Landmark> LandmarkNode<MEASUREMENT>::getDynamicLandmarkEstimate(FrameId frame_id) const {
    auto iter = frames_seen_.find(frame_id);
    CHECK(iter != frames_seen_.end()) << "Requested dynamic landmark with id " << this->tracklet_id << " does not exist at frame " << frame_id;
    const gtsam::Key lmk_key = DynamicLandmarkSymbol(frame_id, this->tracklet_id);

    if(this->isStatic()) {
        throw InvalidLandmarkQuery(lmk_key, "Dynamic estimate requested but landmark is static!");
    }

    return this->map_ptr_->template query<Landmark>(lmk_key);
}

template<typename MEASUREMENT>
bool LandmarkNode<MEASUREMENT>::hasMeasurement(FrameId frame_id) const {
    return measurements_.exists(frame_id);
}

template<typename MEASUREMENT>
const MEASUREMENT& LandmarkNode<MEASUREMENT>::getMeasurement(FrameId frame_id) const {
    if(!hasMeasurement(frame_id)) {
        throw DynosamException("Missing measurement in landmark node with id " + std::to_string(tracklet_id) + " at frame " +  std::to_string(frame_id));
    }
    return measurements_.at(frame_id);
}

/// ObjectNode
template<typename MEASUREMENT>
int ObjectNode<MEASUREMENT>::getId() const {
    return (int)object_id;
}

template<typename MEASUREMENT>
FrameNodePtrSet<MEASUREMENT> ObjectNode<MEASUREMENT>::getSeenFrames() const {
    FrameNodePtrSet<MEASUREMENT> seen_frames;
    for(const auto& lmks : dynamic_landmarks) {
        seen_frames.merge(lmks->getSeenFrames());
    }
    return seen_frames;
}

template<typename MEASUREMENT>
FrameIds ObjectNode<MEASUREMENT>::getSeenFrameIds() const {
    return getSeenFrames().template collectIds<FrameId>();
}

template<typename MEASUREMENT>
LandmarkNodePtrSet<MEASUREMENT> ObjectNode<MEASUREMENT>::getLandmarksSeenAtFrame(FrameId frame_id) const {
    LandmarkNodePtrSet<MEASUREMENT> seen_lmks;

    for(const auto& lmk : dynamic_landmarks) {
        //all frames this lmk was seen in
        const FrameNodePtrSet<MEASUREMENT>& frames = lmk->getSeenFrames();
        //lmk was observed at this frame
        if(frames.find(frame_id) != frames.end()) {
            seen_lmks.insert(lmk);
        }
    }
    return seen_lmks;
}


} //dyno
