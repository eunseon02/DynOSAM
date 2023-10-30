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

#include "dynosam/frontend/vision/VisionTools.hpp"

#include <algorithm>  // std::set_difference, std::sort
#include <vector>     // std::vector



namespace dyno {

FrameProcessor::FrameProcessor(const FrontendParams& params, Camera::Ptr camera) : params_(params), camera_(CHECK_NOTNULL(camera)) {}

void FrameProcessor::getCorrespondences(AbsolutePoseCorrespondences& correspondences, const Frame& previous_frame, const Frame& current_frame, KeyPointType kp_type) const {
  correspondences.clear();
  FeaturePairs feature_correspondences;
  getCorrespondences(feature_correspondences, previous_frame, current_frame, kp_type);

  LOG(INFO) << "Found correspondences " << feature_correspondences.size();

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
    Landmark lmk_w;
    camera_->backProject(prev_feature->keypoint_, prev_feature->depth_, &lmk_w, previous_frame.T_world_camera_);
    correspondences.push_back(TrackletCorrespondance(prev_feature->tracklet_id_, lmk_w, curr_feature->keypoint_));
  }
}

void FrameProcessor::getCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, const Frame& current_frame, KeyPointType kp_type) const {
  if(kp_type == KeyPointType::STATIC) {
    getStaticCorrespondences(correspondences, previous_frame, current_frame);
  }
  else {
    getDynamicCorrespondences(correspondences, previous_frame, current_frame);
  }
}


ObjectIds FrameProcessor::getObjectLabels(const ImageWrapper<ImageType::MotionMask>& image) {
  return getObjectLabels(image.image);
}

ObjectIds FrameProcessor::getObjectLabels(const ImageWrapper<ImageType::SemanticMask>& image) {
  return getObjectLabels(image.image);
}


void FrameProcessor::getStaticCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, const Frame& current_frame) const {
  getCorrespondencesFromContainer(correspondences, previous_frame.static_features_, current_frame.static_features_);
}

void FrameProcessor::getDynamicCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, const Frame& current_frame) const {
  getCorrespondencesFromContainer(correspondences, previous_frame.dynamic_features_, current_frame.dynamic_features_);
}

void FrameProcessor::getCorrespondencesFromContainer(FeaturePairs& correspondences, const FeatureContainer& previous_features, const FeatureContainer& current_features) const {
  correspondences.clear();

  for(const auto& curr_feature : current_features) {
    //check if previous feature and is valid
    if(previous_features.exists(curr_feature->tracklet_id_)) {
      const auto prev_feature = previous_features.getByTrackletId(curr_feature->tracklet_id_);
      CHECK(prev_feature);

      //only if valid
      if(prev_feature->usable()) {
        correspondences.push_back({prev_feature, curr_feature});
      }
    }
  }
}

ObjectIds FrameProcessor::getObjectLabels(const cv::Mat& image) {
  std::set<ObjectId> unique_labels;
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      const ObjectId label = image.at<ObjectId>(i, j);

      if(label != background_label) {
        unique_labels.insert(label);
      }
    }
  }

  return ObjectIds(unique_labels.begin(), unique_labels.end());
}

RGBDProcessor::RGBDProcessor(const FrontendParams& params, Camera::Ptr camera)
    : FrameProcessor(params, camera) {}


void RGBDProcessor::updateDepth(Frame::Ptr frame, ImageWrapper<ImageType::Depth> disparity) {
    cv::Mat depth;
    disparityToDepth(disparity.image, depth);

    setDepths(frame->static_features_, depth, params_.depth_background_thresh);
    setDepths(frame->dynamic_features_, depth, params_.depth_obj_thresh);
}

void RGBDProcessor::disparityToDepth(const cv::Mat& disparity, cv::Mat& depth) {
    disparity.copyTo(depth);
    for (int i = 0; i < disparity.rows; i++)
     {
    for (int j = 0; j < disparity.cols; j++)
    {
      if (disparity.at<double>(i, j) < 0)
      {
        depth.at<double>(i, j) = 0;
      }
      else
      {
        depth.at<double>(i, j) = params_.base_line / (disparity.at<double>(i, j) / params_.depth_scale_factor);
      }
    }
  }
}

void RGBDProcessor::setDepths(FeatureContainer features, const cv::Mat& depth, double max_threshold) {
    for(Feature::Ptr feature : features) {

        if(!feature) continue;

        const int x = functional_keypoint::u(feature->keypoint_);
        const int y = functional_keypoint::v(feature->keypoint_);
        const Depth d = depth.at<Depth>(y, x);


        if(d > max_threshold || d <= 0) {
            feature->markInvalid();
        }

        //if now invalid or happens to be invalid from a previous frame, make depth invalud too
        if(!feature->usable()) {
            feature->depth_ = Feature::invalid_depth;
        }
        else {
            feature->depth_ = d;
        }
    }
}

void RGBDProcessor::updateMovingObjects(const Frame& previous_frame, Frame::Ptr current_frame,  cv::Mat& debug) const {
  const cv::Mat& rgb = current_frame->tracking_images_.get<ImageType::RGBMono>();

  rgb.copyTo(debug);

  const gtsam::Pose3& previous_pose = previous_frame.T_world_camera_;
  const gtsam::Pose3& current_pose = current_frame->T_world_camera_;

  const auto previous_dynamic_feature_container = previous_frame.dynamic_features_;
  const auto current_dynamic_feature_container = current_frame->dynamic_features_;

  //iterate over each object seen in the previous frame and collect features in current and previous frames to determine scene flow
  for(auto& [object_id, current_object_observation] : current_frame->object_observations_) {

    int object_track_count = 0; //number of tracked points on the object
    int sf_count = 0; //number of points on the object with a sufficient scene flow thresh

    const TrackletIds& object_features = current_object_observation.object_features_;
    for(const auto tracklet_id : object_features) {
      if(previous_dynamic_feature_container.exists(tracklet_id)) {
        CHECK(current_dynamic_feature_container.exists(tracklet_id));

        Feature::Ptr current_feature = current_dynamic_feature_container.getByTrackletId(tracklet_id);
        Feature::Ptr previous_feature = previous_dynamic_feature_container.getByTrackletId(tracklet_id);

        if(!previous_feature->usable()) {
          current_feature->markInvalid();
          continue;
        }

        Landmark lmk_previous, lmk_current;
        camera_->backProject(previous_feature->keypoint_, previous_feature->depth_, &lmk_previous, previous_pose);
        camera_->backProject(current_feature->keypoint_, current_feature->depth_, &lmk_current, current_pose);

        Landmark flow_world = lmk_previous - lmk_current;
        double sf_norm = flow_world.norm();

        if(sf_norm > params_.scene_flow_magnitude) {
          sf_count++;
        }

        object_track_count++;
      }
    }

    if(sf_count < 50) {
      continue;
    }
    double average_flow_count = (double)sf_count / (double)object_track_count;

    LOG(INFO) << "Num points that are dynamic " << average_flow_count << "/" << params_.scene_flow_percentage << " for object " << object_id;
    if(average_flow_count > params_.scene_flow_percentage) {
      current_object_observation.marked_as_moving_ = true;

      static const cv::Scalar blue(255, 0, 0);

      for(TrackletId track : object_features) {
        Feature::Ptr current_feature = current_dynamic_feature_container.getByTrackletId(track);
        const Keypoint& px = current_feature->keypoint_;
        cv::circle(debug, utils::gtsamPointToCV(px), 6, blue, 1);
      }


      //only debug stuff

    }

  }

}





void determineOutlierIds(const TrackletIds& inliers, const TrackletIds& tracklets, TrackletIds& outliers)
{
  VLOG_IF(1, inliers.size() > tracklets.size())
      << "Usage warning: inlier size (" << inliers.size() << ") > tracklets size (" << tracklets.size()
      << "). Are you parsing inliers as tracklets incorrectly?";
  outliers.clear();
  TrackletIds inliers_sorted(inliers.size()), tracklets_sorted(tracklets.size());
  std::copy(inliers.begin(), inliers.end(), inliers_sorted.begin());
  std::copy(tracklets.begin(), tracklets.end(), tracklets_sorted.begin());

  std::sort(inliers_sorted.begin(), inliers_sorted.end());
  std::sort(tracklets_sorted.begin(), tracklets_sorted.end());

  // full set A (tracklets) must be first and inliers MUST be a subset of A for the set_difference function to work
  std::set_difference(tracklets_sorted.begin(), tracklets_sorted.end(), inliers_sorted.begin(), inliers_sorted.end(),
                      std::inserter(outliers, outliers.begin()));
}


} //dyno
