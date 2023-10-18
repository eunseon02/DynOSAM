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

#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {

FeatureTracker::FeatureTracker(const FrontendParams& params)
    : params_(params) {
    feature_detector_ = std::make_unique<ORBextractor>(
      params_.n_features, static_cast<float>(params_.scale_factor), params_.n_levels,
      params_.init_threshold_fast, params_.min_threshold_fast);
}

Frame::Ptr FeatureTracker::track(FrameId frame_id, Timestamp timestamp, const InputImages& input_images, size_t& n_optical_flow, size_t& n_new_tracks) {

    if(initial_computation_) {
        //intitial computation
        CHECK(!previous_frame_);
        img_size_ = input_images.img_.size();
        computeImageBounds(img_size_, min_x_, max_x_, min_y_, max_y_);

        grid_elements_width_inv_ = static_cast<double>(FRAME_GRID_COLS) / static_cast<double>(max_x_ - min_x_);
        grid_elements_height_inv_ = static_cast<double>(FRAME_GRID_ROWS) / static_cast<double>(max_y_ - min_y_);

        LOG(INFO) << grid_elements_width_inv_ << " " << grid_elements_height_inv_;
        initial_computation_ = false;
    }
    else {
        //TODO:: update frame mask
        CHECK(previous_frame_);
        CHECK_EQ(previous_frame_->frame_id_, frame_id - 1u) << "Incoming frame id must be consequative";
    }

    FeaturePtrs static_features;
    trackStatic(frame_id, input_images, static_features, n_optical_flow, n_new_tracks);

    // FeaturePtrs dynamic_features;
    // return nullptr;
    auto new_frame = std::make_shared<Frame>(frame_id, timestamp, input_images);
    new_frame->static_features_ = static_features;

    previous_frame_ = new_frame;
    return new_frame;
    // trackDynamic(input_images, dynamic_features);

}


void FeatureTracker::trackStatic(FrameId frame_id, const InputImages& input_packet, FeaturePtrs& static_features, size_t& n_optical_flow,
                                 size_t& n_new_tracks)
{
  const cv::Mat& rgb = input_packet.img_;
  const cv::Mat& motion_mask = input_packet.motion_mask_;

  cv::Mat mono;
  CHECK(!rgb.empty());
  PLOG_IF(ERROR, rgb.channels() == 1) << "Input image should be RGB (channels == 3), not 1";
  // Transfer color image to grey image
  rgb.copyTo(mono);

  cv::Mat viz;
  rgb.copyTo(viz);

  if (mono.channels() == 3)
  {
    cv::cvtColor(mono, mono, cv::COLOR_RGB2GRAY);
  }
  else if (rgb.channels() == 4)
  {
    cv::cvtColor(mono, mono, cv::COLOR_RGBA2GRAY);
  }

  cv::Mat descriptors;
  KeypointsCV detected_keypoints;
  (*feature_detector_)(mono, cv::Mat(), detected_keypoints, descriptors);
  // save detections
  // orb_detections_.insert({ frame_id, detected_keypoints });
  // cv::drawKeypoints(viz, detected_keypoints, viz, cv::Scalar(0, 0, 255));
  VLOG(20) << "detected - " << detected_keypoints.size();

  // static_features = feature_correspondences;
  FeaturePtrs features_tracked;
  std::vector<bool> detections_tracked(detected_keypoints.size(), false);

  const int& min_tracks = params_.max_tracking_points_bg;
  std::vector<std::size_t> grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

  bool use_optical_tracking = true;

  // appy tracking (ie get correspondences)
  if (previous_frame_)
  {
    for (Feature::Ptr previous_feature : previous_frame_->static_features_)
    {
      const size_t tracklet_id = previous_feature->tracklet_id_;
      const size_t age = previous_feature->age_;
      KeypointCV kp = previous_feature->predicted_keypoint_;
      //TODO: is contained

      // if (camera_.isKeypointContained(kp, previous_feature->depth) && previous_feature->inlier)
      // {
      //   size_t new_age = age + 1;
      //   Feature::Ptr feature = constructStaticFeature(images, kp, new_age, tracklet_id, frame_id);
      //   if (feature)
      //   {
      //     cv::arrowedLine(viz, previous_feature->keypoint.pt, kp.pt, cv::Scalar(255, 0, 0));
      //     utils::DrawCircleInPlace(viz, kp.pt, cv::Scalar(0, 0, 255));
      //     features_tracked.push_back(feature);
      //   }
      // }
    }
  }

  n_optical_flow = features_tracked.size();
  VLOG(20) << "tracked with optical flow - " << n_optical_flow;

  // Assign Features to Grid Cells
  // int n_reserve = (FRAME_GRID_COLS * FRAME_GRID_ROWS) / (0.5 * min_tracks);
  int n_reserve = (FRAME_GRID_COLS * FRAME_GRID_ROWS) / (0.5 * n_optical_flow);
  VLOG(20) << "reserving - " << n_reserve;

  // assign tracked features to grid
  FeaturePtrs features_assigned;
  for (Feature::Ptr feature : features_tracked)
  {
    const cv::KeyPoint& kp = feature->keypoint_;
    int grid_x, grid_y;
    if (posInGrid(kp, grid_x, grid_y))
    {
      grid[grid_x][grid_y].push_back(feature->tracklet_id_);
      features_assigned.push_back(feature);
    }
  }


   // container used just for sanity check
  FeaturePtrs new_features;
  VLOG(20) << "assigned grid features - " << features_assigned.size();
  // only add new features if tracks drop below min tracks
  if (features_assigned.size() < min_tracks)
  {
    // iterate over new observations
    for (size_t i = 0; i < detected_keypoints.size(); i++)
    {
      if (features_assigned.size() >= min_tracks)
      {
        break;
      }

      // TODO: if not object etc etc
      const KeypointCV& kp = detected_keypoints[i];
      const int& x = kp.pt.x;
      const int& y = kp.pt.y;

      // if not already tracked with optical flow
      if (motion_mask.at<int>(y, x) != background_label)
      {
        continue;
      }

      int grid_x, grid_y;
      if (posInGrid(kp, grid_x, grid_y))
      {
        // only add of we have less than n_reserve ammount
        // if (grid[grid_x][grid_y].size() < n_reserve)
        if (grid[grid_x][grid_y].empty())
        {
          const size_t age = 0;
          size_t tracklet_id = tracklet_count;
          Feature::Ptr feature = constructStaticFeature(input_packet, kp, age, tracklet_id, frame_id);
          if (feature)
          {

            tracklet_count++;
            grid[grid_x][grid_y].push_back(feature->tracklet_id_);
            features_assigned.push_back(feature);
            new_features.push_back(feature);
            // draw an untracked kp
            utils::drawCircleInPlace(viz, feature->keypoint_.pt, cv::Scalar(0, 255, 0));
          }
        }
      }
    }
  }

  size_t total_tracks = features_assigned.size();
  n_new_tracks = total_tracks - n_optical_flow;
  VLOG(1) << "new tracks - " << n_new_tracks;
  VLOG(1) << "total tracks - " << total_tracks;

  static_features.clear();
  static_features = features_assigned;

  cv::imshow("Optical flow", viz);
  cv::waitKey(1);



}


bool FeatureTracker::posInGrid(const cv::KeyPoint& kp, int& pos_x, int& pos_y) const {
    pos_x = round((kp.pt.x - min_x_) * grid_elements_width_inv_);
  pos_y = round((kp.pt.y - min_y_) * grid_elements_height_inv_);

  // Keypoint's coordinates are undistorted, which could cause to go out of the image
  if (pos_x < 0 || pos_x >= FRAME_GRID_COLS || pos_y < 0 || pos_y >= FRAME_GRID_ROWS)
    return false;

  return true;

}


void FeatureTracker::computeImageBounds(const cv::Size& size, int& min_x, int& max_x, int& min_y, int& max_y) const {
    // TODO: distortion
    //see orbslam
    min_x = 0;
    max_x = size.width;
    min_y = 0;
    max_y = size.height;
}


Feature::Ptr FeatureTracker::constructStaticFeature(const InputImages& input_packet, const cv::KeyPoint& kp, size_t age, TrackletId tracklet_id, FrameId frame_id) const {
  const int& x = kp.pt.x;
  const int& y = kp.pt.y;
  const int& rows = input_packet.img_.rows;
  const int& cols = input_packet.img_.cols;
  if (input_packet.motion_mask_.at<int>(y, x) != background_label)
  {
    return nullptr;
  }

  // check flow
  double flow_xe = static_cast<double>(input_packet.optical_flow_.at<cv::Vec2f>(y, x)[0]);
  double flow_ye = static_cast<double>(input_packet.optical_flow_.at<cv::Vec2f>(y, x)[1]);

  if (!(flow_xe != 0 && flow_ye != 0))
  {
    return nullptr;
  }

  // check predicted flow is within image
  cv::KeyPoint predicted_kp(static_cast<double>(x) + flow_xe, static_cast<double>(y) + flow_ye, 0, 0, 0, kp.octave, -1);
  if (predicted_kp.pt.x >= cols || predicted_kp.pt.y >= rows || predicted_kp.pt.x <= 0 || predicted_kp.pt.y <= 0)
  {
    return nullptr;
  }

  Feature::Ptr feature = std::make_shared<Feature>();
  feature->keypoint_ = kp;
  feature->age_ = age;
  feature->tracklet_id_ = tracklet_id;
  feature->frame_id_ = frame_id;
  feature->type_ = KeyPointType::STATIC;
  feature->inlier_ = true;
  feature->label_ = background_label;

  // the flow is not actually what
  feature->predicted_keypoint_ = predicted_kp;
  return feature;
}

}
