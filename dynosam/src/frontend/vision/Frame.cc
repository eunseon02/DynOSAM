/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"

namespace dyno {

ObjectId Frame::global_object_id{1};

Frame::Frame(
        FrameId frame_id,
        Timestamp timestamp,
        Camera::Ptr camera,
        const TrackingInputImages& tracking_images,
        const FeatureContainer& static_features,
        const FeatureContainer& dynamic_features)
        :   frame_id_(frame_id),
            timestamp_(timestamp),
            camera_(camera),
            tracking_images_(tracking_images),
            static_features_(static_features),
            dynamic_features_(dynamic_features)
        {
            constructDynamicObservations();
        }

bool Frame::exists(TrackletId tracklet_id) const {
    const bool result = static_features_.exists(tracklet_id) || dynamic_features_.exists(tracklet_id);

    //debug checking -> should only be in one feature container
    if(result) {
        CHECK(!(static_features_.exists(tracklet_id) &&  dynamic_features_.exists(tracklet_id)))
            << "Tracklet Id " <<  tracklet_id << " exists in both static and dynamic feature sets. Should be unique!";
    }
    return result;
}

Feature::Ptr Frame::at(TrackletId tracklet_id) const {
    if(!exists(tracklet_id)) {
        return nullptr;
    }

    if(static_features_.exists(tracklet_id)) {
        CHECK(!dynamic_features_.exists(tracklet_id));
        return static_features_.getByTrackletId(tracklet_id);
    }
    else {
        CHECK(dynamic_features_.exists(tracklet_id));
        return dynamic_features_.getByTrackletId(tracklet_id);
    }
}


FeaturePtrs Frame::collectFeatures(TrackletIds tracklet_ids) const {
    FeaturePtrs features;
    for(const auto tracklet_id : tracklet_ids) {
        Feature::Ptr feature = at(tracklet_id);
        if(!feature) {
            throw std::runtime_error("Failed to collectFeatures - tracklet id " + std::to_string(tracklet_id) + " does not exist");
        }

        features.push_back(feature);
    }

    return features;
}


Landmark Frame::backProjectToCamera(TrackletId tracklet_id) const {
    Feature::Ptr feature = at(tracklet_id);
    if(!feature) {
        throw std::runtime_error("Failed to backProjectToCamera - tracklet id " + std::to_string(tracklet_id) + " does not exist");
    }

    //if no depth, project to unitsphere?
    CHECK(feature->hasDepth());

    Landmark lmk;
    camera_->backProject(feature->keypoint_, feature->depth_, &lmk);
    return lmk;
}

Landmark Frame::backProjectToWorld(TrackletId tracklet_id) const {
    Feature::Ptr feature = at(tracklet_id);
    if(!feature) {
        throw std::runtime_error("Failed to backProjectToWorld - tracklet id " + std::to_string(tracklet_id) + " does not exist");
    }

    //if no depth, project to unitsphere?
    CHECK(feature->hasDepth());

    Landmark lmk;
    camera_->backProject(feature->keypoint_, feature->depth_, &lmk, T_world_camera_);
    return lmk;
}

void Frame::updateDepths(const ImageWrapper<ImageType::Depth>& depth, double max_static_depth, double max_dynamic_depth) {
    updateDepthsFeatureContainer(static_features_, depth, max_static_depth);
    updateDepthsFeatureContainer(dynamic_features_, depth, max_dynamic_depth);
}


void Frame::getCorrespondences(AbsolutePoseCorrespondences& correspondences, const Frame& previous_frame, KeyPointType kp_type) const {
  correspondences.clear();
  FeaturePairs feature_correspondences;
  getCorrespondences(feature_correspondences, previous_frame, kp_type);

  //this will also take a point in the camera frame and put into world frame
  for(const auto& pair : feature_correspondences) {
    const Feature::Ptr& prev_feature = pair.first;
    const Feature::Ptr& curr_feature = pair.second;

    CHECK(prev_feature);
    CHECK(curr_feature);

    CHECK_EQ(prev_feature->tracklet_id_, curr_feature->tracklet_id_);


    //this will not work for monocular or some other system that never has depth but will eventually have a point?
    if(!prev_feature->hasDepth()) {
      throw std::runtime_error("Error in FrameProcessor::getCorrespondences for AbsolutePoseCorrespondences - previous feature does not have depth!");
    }

    //eventuall map?
    Landmark lmk_w = previous_frame.backProjectToWorld(prev_feature->tracklet_id_);
    correspondences.push_back(TrackletCorrespondance(prev_feature->tracklet_id_, lmk_w, curr_feature->keypoint_));
  }

}

void Frame::getCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, KeyPointType kp_type) const {
    if(kp_type == KeyPointType::STATIC) {
        getStaticCorrespondences(correspondences, previous_frame);
    }
    else {
        getDynamicCorrespondences(correspondences, previous_frame);
    }
}
void Frame::getStaticCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame) const {
    vision_tools::getCorrespondences(correspondences, previous_frame.static_features_, static_features_, true);
}

void Frame::getDynamicCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame) const {
    vision_tools::getCorrespondences(correspondences, previous_frame.dynamic_features_, dynamic_features_, true);
}


void Frame::updateDepthsFeatureContainer(FeatureContainer& container, const ImageWrapper<ImageType::Depth>& depth, double max_depth) {
    const cv::Mat& depth_mat = depth;
    // FeatureFilterIterator iter(container, [&](const Feature::Ptr& f) -> bool { return f->usable();});
    auto iter = container.usableIterator();

    for(Feature::Ptr feature : iter) {
        CHECK(feature->usable());
        // const Feature::Ptr& feature = *iter;
        const int x = functional_keypoint::u(feature->keypoint_);
        const int y = functional_keypoint::v(feature->keypoint_);
        const Depth d = depth_mat.at<Depth>(y, x);

        if(d > max_depth || d <= 0) {
            feature->markInvalid();
        }

         //if now invalid or happens to be invalid from a previous frame, make depth invalid too
        if(!feature->usable()) {
            feature->depth_ = Feature::invalid_depth;
        }
        else {
            feature->depth_ = d;
        }
    }

}


void Frame::constructDynamicObservations() {
    CHECK_GT(dynamic_features_.size(), 0u);
    object_observations_.clear();


    const ObjectIds instance_labels = vision_tools::getObjectLabels(tracking_images_.get<ImageType::MotionMask>());

    auto inlier_iterator = dynamic_features_.usableIterator();
    for(const Feature::Ptr& dynamic_feature : inlier_iterator) {
        CHECK(!dynamic_feature->isStatic());
        CHECK(dynamic_feature->usable());

        const ObjectId instance_label = dynamic_feature->instance_label_;
        //this check is just for sanity!
        CHECK(std::find(instance_labels.begin(), instance_labels.end(), instance_label) != instance_labels.end());

        if(object_observations_.find(instance_label) == object_observations_.end()) {
            DynamicObjectObservation observation;
            observation.tracking_label_ = -1;
            observation.instance_label_ = instance_label;
            object_observations_[instance_label] = observation;
        }

        object_observations_[instance_label].object_features_.push_back(dynamic_feature->tracklet_id_);
    }
}

void Frame::moveObjectToStatic(ObjectId instance_label) {
    auto it = object_observations_.find(instance_label);
    CHECK(it != object_observations_.end());


    DynamicObjectObservation& observation = it->second;
    observation.marked_as_moving_ = false;
    CHECK(observation.instance_label_ == instance_label);
    //go through all features, move them to from dynamic structure and add them to static
    for(TrackletId tracklet_id : observation.object_features_) {
        CHECK(dynamic_features_.exists(tracklet_id));
        Feature::Ptr dynamic_feature = dynamic_features_.getByTrackletId(tracklet_id);

        if(!dynamic_feature->usable()) {continue;}


        CHECK(!dynamic_feature->isStatic());
        CHECK_EQ(dynamic_feature->tracklet_id_, tracklet_id);
        CHECK_EQ(dynamic_feature->instance_label_, instance_label);
        dynamic_feature->type_ = KeyPointType::STATIC;
        dynamic_feature->instance_label_ = background_label;
        dynamic_feature->tracking_label_ = background_label;

        dynamic_features_.remove(tracklet_id);
        static_features_.add(dynamic_feature);
    }

    object_observations_.erase(it);

}

void Frame::updateObjectTrackingLabel(const DynamicObjectObservation& observation, ObjectId new_tracking_label) {
    auto it = object_observations_.find(observation.instance_label_);
    CHECK(it != object_observations_.end());

    auto& obs = it->second;
    obs.tracking_label_ = new_tracking_label;
    //update all features
    for(TrackletId tracklet_id : obs.object_features_) {
        Feature::Ptr feature = dynamic_features_.getByTrackletId(tracklet_id);
        CHECK(feature);
        feature->tracking_label_ = new_tracking_label;
    }
}


FeatureFilterIterator Frame::usableStaticFeaturesBegin() {
    return FeatureFilterIterator(static_features_, [&](const Feature::Ptr& f) -> bool
        {
            return f->usable();
        }
    );
}

FeatureFilterIterator Frame::usableDynamicFeaturesBegin() {
    return FeatureFilterIterator(dynamic_features_, [&](const Feature::Ptr& f) -> bool
        {
            return f->usable();
        }
    );
}


// Frame::FeatureFilterIterator Frame::dynamicUsableBegin() {
//     return FeatureFilterIterator(dynamic_features_, [&](const Feature::Ptr& f) -> bool
//         {
//             return f->usable();
//         }
//     );
// }

} //dyno
