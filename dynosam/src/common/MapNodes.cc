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

#include "dynosam/common/MapNodes.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"


namespace dyno {



int FrameNode::getId() const {
    return (int)frame_id;
}

bool FrameNode::objectObserved(ObjectId object_id) const {
    return objects_seen.find(object_id) != objects_seen.end();
}

StateQuery<gtsam::Pose3> FrameNode::getPoseEstimate() const {
    const gtsam::Key pose_key = CameraPoseSymbol(frame_id);
    return queryWeakMap<gtsam::Pose3>(map_ptr_, pose_key);

}

StateQuery<gtsam::Pose3> FrameNode::getObjectMotionEstimate(ObjectId object_id) const {
    const gtsam::Key motion_key = ObjectMotionSymbol(object_id, frame_id);
    return queryWeakMap<gtsam::Pose3>(map_ptr_, motion_key);
}
StateQuery<Landmark> FrameNode::getDynamicLandmarkEstimate(TrackletId tracklet_id) const {
    const gtsam::Key lmk_key = DynamicLandmarkSymbol(frame_id, tracklet_id);
    return queryWeakMap<Landmark>(map_ptr_, lmk_key);
}

StatusLandmarkEstimates FrameNode::getAllDynamicLandmarkEstimates() const {
    //quicker to go through the dynamic_landmarks than the objects as we already know they exist
    //in this frame
    StatusLandmarkEstimates estimates;
    for(const auto& lmk_ptr : dynamic_landmarks) {
        StateQuery<Landmark> lmk_status_query = getDynamicLandmarkEstimate(lmk_ptr->getId());
        if(lmk_status_query) {
            // LandmarkStatus lmk_status = LandmarkStatus::Dynamic(
            //     LandmarkStatus::Method::OPTIMIZED,
            //     lmk_ptr->getObjectId()
            // );

            // auto estimate = std::make_pair(lmk_ptr->getId(), lmk_status_query.get());
            // estimates.push_back(std::make_pair(lmk_status, estimate));
            appendStatusEstimate(
                estimates,
                LandmarkStatus::Dynamic(
                    LandmarkStatus::Method::OPTIMIZED,
                    lmk_ptr->getObjectId()
                ), //status
                lmk_ptr->getId(), //tracklet id
                lmk_status_query.get() //estimate
            );

        }
    }

    return estimates;
}

StatusLandmarkEstimates FrameNode::getDynamicLandmarkEstimates(ObjectId object_id) const {
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
            StateQuery<Landmark> lmk_status_query = getDynamicLandmarkEstimate(lmk_ptr->getId());
            if(lmk_status_query) {
                appendStatusEstimate(
                    estimates,
                    LandmarkStatus::Dynamic(
                        LandmarkStatus::Method::OPTIMIZED,
                        lmk_ptr->getObjectId()
                    ), //status
                    lmk_ptr->getId(), //tracklet id
                    lmk_status_query.get() //estimate
                );
            }
        }
    }
    return estimates;
}


/// LandmarkNode

int LandmarkNode::getId() const {
    return (int)tracklet_id;
}

ObjectId LandmarkNode::getObjectId() const {
    return object_id;
}

 bool LandmarkNode::isStatic() const {
    return object_id == background_label;
}

size_t LandmarkNode::numObservations() const {
    return frames_seen.size();
}

StateQuery<Landmark> LandmarkNode::getStaticLandmarkEstimate() const {
    const gtsam::Key lmk_key = StaticLandmarkSymbol(tracklet_id);

    if(!isStatic()) {
        throw InvalidLandmarkQuery(lmk_key, "static estimate requested but landmark is dynamic!");
    }

    return queryWeakMap<Landmark>(map_ptr_, lmk_key);
}

/// ObjectNode

int ObjectNode::getId() const {
    return (int)object_id;
}

FrameNodePtrSet ObjectNode::getSeenFrames() const {
    FrameNodePtrSet seen_frames;
    for(const auto& lmks : dynamic_landmarks) {
        seen_frames.merge(lmks->frames_seen);
    }
    return seen_frames;
}

FrameIds ObjectNode::getSeenFrameIds() const {
    return getSeenFrames().collectIds<FrameId>();
}

LandmarkNodePtrSet ObjectNode::getLandmarksSeenAtFrame(FrameId frame_id) const {
    LandmarkNodePtrSet seen_lmks;

    for(const auto& lmk : dynamic_landmarks) {
        //all frames this lmk was seen in
        const FrameNodePtrSet& frames = lmk->frames_seen;
        //lmk was observed at this frame
        if(frames.find(frame_id) != frames.end()) {
            seen_lmks.insert(lmk);
        }
    }
    return seen_lmks;
}


} //dyno
