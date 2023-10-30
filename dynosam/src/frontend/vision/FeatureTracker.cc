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
#include "dynosam/frontend/vision/VisionTools.hpp"

#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>



DEFINE_int32(semantic_mask_step_size, 3, "The step sized used across the semantic mask when sampling points");

namespace dyno {

FeatureTracker::FeatureTracker(const FrontendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : params_(params),
      camera_(camera),
      display_queue_(display_queue) {
    feature_detector_ = std::make_unique<ORBextractor>(
      params_.n_features, static_cast<float>(params_.scale_factor), params_.n_levels,
      params_.init_threshold_fast, params_.min_threshold_fast);

    img_size_ = camera_->getParams().imageSize();
    CHECK(!img_size_.empty());
}

Frame::Ptr FeatureTracker::track(FrameId frame_id, Timestamp timestamp, const TrackingInputImages& tracking_images, size_t& n_optical_flow, size_t& n_new_tracks) {

    if(initial_computation_) {
        //intitial computation
        const cv::Size& other_size = tracking_images.get<ImageType::RGBMono>().size();
        CHECK(!previous_frame_);
        CHECK(img_size_.width == other_size.width && img_size_.height == other_size.height);
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
    trackStatic(frame_id, tracking_images, static_features, n_optical_flow, n_new_tracks);

    FeaturePtrs dynamic_features;
    trackDynamic(frame_id, tracking_images, dynamic_features);

    auto new_frame = std::make_shared<Frame>(frame_id, timestamp, tracking_images);
    new_frame->static_features_ = FeatureContainer(static_features);
    new_frame->dynamic_features_ = FeatureContainer(dynamic_features);
    new_frame->initial_object_labels_ = FrameProcessor::getObjectLabels(tracking_images.get<ImageType::MotionMask>());

    previous_frame_ = new_frame;
    return new_frame;

}


cv::Mat FeatureTracker::computeImageTracks(const Frame& previous_frame, const Frame& current_frame) const {
  cv::Mat img_rgb;

  const cv::Mat& rgb = current_frame.tracking_images_.get<ImageType::RGBMono>();
  rgb.copyTo(img_rgb);


  static const cv::Scalar gray(0, 255, 255);
  static const cv::Scalar red(0, 0, 255);
  static const cv::Scalar green(0, 255, 0);
  static const cv::Scalar blue(255, 0, 0);

  // // Add extra corners if desired.
  // for (const auto& px : extra_corners_gray) {
  //   cv::circle(img_rgb, px, 4, gray, 2);
  // }
  // for (const auto& px : extra_corners_blue) {
  //   cv::circle(img_rgb, px, 4, blue, 2);
  // }

  // Add all keypoints in cur_frame with the tracks.
  for (size_t i = 0; i < current_frame.static_features_.size(); ++i) {
    const Feature::Ptr& feature = current_frame.static_features_.at(i);
    const Keypoint& px_cur = feature->keypoint_;
    if (!feature->usable()) {  // Untracked landmarks are red.
      cv::circle(img_rgb,  utils::gtsamPointToCV(px_cur), 4, red, 2);
    } else {

      const Feature::Ptr& prev_feature = previous_frame.static_features_.getByTrackletId(feature->tracklet_id_);
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        cv::circle(img_rgb,  utils::gtsamPointToCV(px_cur), 6, green, 1);
        const Keypoint& px_prev = prev_feature->keypoint_;
        cv::arrowedLine(img_rgb, utils::gtsamPointToCV(px_prev), utils::gtsamPointToCV(px_cur), green, 1);
      } else {  // New feature tracks are blue.
        cv::circle(img_rgb, utils::gtsamPointToCV(px_cur), 6, blue, 1);
      }
    }
  }


  // for (size_t i = 0; i < current_frame.dynamic_features_.size(); ++i) {
  //   const Feature::Ptr& feature = current_frame.dynamic_features_.at(i);
  //   const Keypoint& px_cur = feature->keypoint_;
  //   if (!feature->usable()) {  // Untracked landmarks are red.
  //     // cv::circle(img_rgb,  utils::gtsamPointToCV(px_cur), 1, red, 2);
  //   } else {


  //     const Feature::Ptr& prev_feature = previous_frame.dynamic_features_.getByTrackletId(feature->tracklet_id_);
  //     if (prev_feature) {
  //       // If feature was in previous frame, display tracked feature with
  //       // green circle/line:
  //       // cv::circle(img_rgb,  utils::gtsamPointToCV(px_cur), 6, green, 1);
  //       const Keypoint& px_prev = prev_feature->keypoint_;
  //       cv::arrowedLine(img_rgb, utils::gtsamPointToCV(px_prev), utils::gtsamPointToCV(px_cur), green, 1);
  //     } else {  // New feature tracks are blue.
  //       cv::circle(img_rgb, utils::gtsamPointToCV(px_cur), 1, blue, 1);
  //     }
  //   }
  // }


  return img_rgb;

}


void FeatureTracker::trackStatic(FrameId frame_id, const TrackingInputImages& tracking_images, FeaturePtrs& static_features, size_t& n_optical_flow,
                                 size_t& n_new_tracks)
{
  const cv::Mat& rgb = tracking_images.get<ImageType::RGBMono>();
  const cv::Mat& motion_mask = tracking_images.get<ImageType::MotionMask>();

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
      Keypoint kp = previous_feature->predicted_keypoint_;
      const int x = functional_keypoint::u(kp);
      const int y = functional_keypoint::v(kp);

      ObjectId instance_label = motion_mask.at<ObjectId>(y, x);

      if (camera_->isKeypointContained(kp) && previous_feature->usable() && instance_label == background_label)
      {
        size_t new_age = age + 1;
        Feature::Ptr feature = constructStaticFeature(tracking_images, kp, new_age, tracklet_id, frame_id);
        if (feature)
        {

          // cv::arrowedLine(viz,
          //   utils::gtsamPointToCV(previous_feature->keypoint_),
          //   utils::gtsamPointToCV(kp),
          //   cv::Scalar(255, 0, 0));

          // utils::drawCircleInPlace(viz, utils::gtsamPointToCV(kp), cv::Scalar(0, 0, 255));
          features_tracked.push_back(feature);
        }
      }
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
    const Keypoint& kp = feature->keypoint_;
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
      const KeypointCV& kp_cv = detected_keypoints[i];
      const int& x = kp_cv.pt.x;
      const int& y = kp_cv.pt.y;

      // if not already tracked with optical flow
      if (motion_mask.at<int>(y, x) != background_label)
      {
        continue;
      }

      Keypoint kp(x, y);
      int grid_x, grid_y;
      if (posInGrid(kp, grid_x, grid_y))
      {
        // only add of we have less than n_reserve ammount
        // if (grid[grid_x][grid_y].size() < n_reserve)
        if (grid[grid_x][grid_y].empty())
        {
          const size_t age = 0;
          size_t tracklet_id = tracklet_count;
          Feature::Ptr feature = constructStaticFeature(tracking_images, kp, age, tracklet_id, frame_id);
          if (feature)
          {

            tracklet_count++;
            grid[grid_x][grid_y].push_back(feature->tracklet_id_);
            features_assigned.push_back(feature);
            new_features.push_back(feature);
            // draw an untracked kp
            utils::drawCircleInPlace(viz, utils::gtsamPointToCV(feature->keypoint_), cv::Scalar(0, 255, 0));
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

  // if(display_queue_) display_queue_->push(ImageToDisplay("static tracks", viz));


}


void FeatureTracker::trackDynamic(FrameId frame_id, const TrackingInputImages& tracking_images, FeaturePtrs& dynamic_features) {
  // first dectect dynamic points
  const cv::Mat& rgb = tracking_images.get<ImageType::RGBMono>();
  const cv::Mat& flow = tracking_images.get<ImageType::OpticalFlow>();
  const cv::Mat& motion_mask = tracking_images.get<ImageType::MotionMask>();

  FeaturePtrs sampled_features, tracked_features;
  ObjectIds instance_labels;


  // cv::Mat viz;
  // rgb.copyTo(viz);

  // a mask to show which feature have been with optical flow from the previous frame
  // zeros mean have not been tracked
  cv::Mat tracked_feature_mask = cv::Mat::zeros(rgb.size(), CV_8UC1);

  if (previous_frame_)
  {
    for (Feature::Ptr previous_dynamic_feature : previous_frame_->dynamic_features_)
    {
      const TrackletId tracklet_id = previous_dynamic_feature->tracklet_id_;
      const size_t age = previous_dynamic_feature->age_;
      const Keypoint kp = previous_dynamic_feature->predicted_keypoint_;

      const int x = functional_keypoint::u(kp);
      const int y = functional_keypoint::v(kp);

      ObjectId predicted_label = motion_mask.at<ObjectId>(y, x);

      const Keypoint previous_point = previous_dynamic_feature->keypoint_;
      // InstanceLabel previous_label = previous_frame_->images_.semantic_mask.at<InstanceLabel>(previous_point.pt.y,
      // previous_point.pt.x); bool same_propogated_label = previous_label == predicted_label;
      if(camera_->isKeypointContained(kp) && previous_dynamic_feature->usable() && predicted_label != background_label) {
        size_t new_age = age + 1;
        double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(y, x)[0]);
        double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(y, x)[1]);

        // // save correspondences
        Feature::Ptr feature = std::make_shared<Feature>();
        // feature->instance_label = images.semantic_mask.at<InstanceLabel>(y, x);

        //this should be up to date after we update the mask with the propogate mask
        feature->instance_label_ = predicted_label;
        feature->tracking_label_ = predicted_label;
        feature->frame_id_ = frame_id;
        feature->type_ = KeyPointType::DYNAMIC;
        feature->age_ = new_age;
        feature->tracklet_id_ = tracklet_id;
        feature->keypoint_ = kp;
        feature->predicted_keypoint_ = Keypoint(kp(0) + flow_xe, kp(1) + flow_ye);

        // propogate dynamic object label??
        // feature->label_ = previous_dynamic_feature->label_;

        // TODO: change to tracked
        sampled_features.push_back(feature);
        tracked_feature_mask.at<uchar>(y, x) = 1;
        instance_labels.push_back(feature->instance_label_);
        // calculate scene flow
        // Landmark lmk_current, lmk_previous;
        // camera_.backProject(feature->keypoint, feature->depth, &lmk_current);
        // camera_.backProject(previous_point, previous_dynamic_feature->depth, &lmk_previous);

        // //put both into the world frame
        // lmk_previous = previous_frame_->pose_.transformFrom(lmk_previous);

        //  cv::arrowedLine(viz,
        //     utils::gtsamPointToCV(previous_feature->keypoint_),
        //     utils::gtsamPointToCV(kp),
        //     cv::Scalar(255, 0, 0));

        //   utils::drawCircleInPlace(viz, utils::gtsamPointToCV(kp), cv::Scalar(0, 0, 255));

        // cv::arrowedLine(viz, previous_dynamic_feature->keypoint.pt, feature->keypoint.pt,
        //                 Display::getObjectColour(feature->instance_label));
        // utils::DrawCircleInPlace(viz, feature->keypoint.pt, cv::Scalar(0, 0, 255));
      }
    }
  }

  LOG(INFO) << "Tracked dynamic points - " << sampled_features.size();

  int step = FLAGS_semantic_mask_step_size;
  for (int i = 0; i < rgb.rows - step; i = i + step)
  {
    for (int j = 0; j < rgb.cols - step; j = j + step)
    {
      if (motion_mask.at<ObjectId>(i, j) == background_label)
      {
        continue;
      }

      // if (images.depth.at<double>(i, j) >= tracking_params_.depth_obj_thresh || images.depth.at<double>(i, j) <= 0)
      // {
      //   continue;
      // }

      double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(i, j)[0]);
      double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(i, j)[1]);

      // int occupied = static_cast<int>(tracked_feature_mask.at<uchar>(i, j));

      // we are within the image bounds?
      if (j + flow_xe < rgb.cols && j + flow_xe > 0 && i + flow_ye < rgb.rows && i + flow_ye > 0)
      {
        // // save correspondences
        Feature::Ptr feature = std::make_shared<Feature>();

        // Depth d = images.depth.at<double>(i, j);  // be careful with the order  !!!
        // if (d > 0)
        // {
        //   feature->depth = d;
        // }
        // else
        // {
        //   // log warning?
        //   continue;
        // }

        feature->instance_label_ = motion_mask.at<ObjectId>(i, j);
        feature->tracking_label_ = motion_mask.at<ObjectId>(i, j);
        feature->frame_id_ = frame_id;
        feature->type_ = KeyPointType::DYNAMIC;
        feature->age_ = 0;
        feature->tracklet_id_ = tracklet_count;
        tracklet_count++;
        // the flow is not actually what
        // feature->optical_flow = cv::Point2d(flow_xe, flow_ye);
        feature->predicted_keypoint_ = Keypoint(j + flow_xe, i+ flow_ye);
        feature->keypoint_ = Keypoint(j, i);

        sampled_features.push_back(feature);
        instance_labels.push_back(feature->instance_label_);
      }
    }
  }
  dynamic_features = sampled_features;

  // cv::imshow("Dynamic flow", viz);
  // cv::waitKey(1);

  LOG(INFO) << "Dynamic Sampled " << sampled_features.size();

  // std::sort(instance_labels.begin(), instance_labels.end());
  // std::vector<InstanceLabel> unique_instance_labels = instance_labels;
  // unique_instance_labels.erase(std::unique(unique_instance_labels.begin(), unique_instance_labels.end()),
  //                              unique_instance_labels.end());
}


bool FeatureTracker::posInGrid(const Keypoint& kp, int& pos_x, int& pos_y) const {
  const int u = functional_keypoint::u(kp);
  const int v = functional_keypoint::v(kp);
  pos_x = round((u - min_x_) * grid_elements_width_inv_);
  pos_y = round((v - min_y_) * grid_elements_height_inv_);

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


Feature::Ptr FeatureTracker::constructStaticFeature(const TrackingInputImages& tracking_images, const Keypoint& kp, size_t age, TrackletId tracklet_id, FrameId frame_id) const {
  //implicit double -> int cast for pixel location
  const int x = functional_keypoint::u(kp);
  const int y = functional_keypoint::v(kp);

  const cv::Mat& rgb = tracking_images.get<ImageType::RGBMono>();
  const cv::Mat& motion_mask = tracking_images.get<ImageType::MotionMask>();
  const cv::Mat& optical_flow = tracking_images.get<ImageType::OpticalFlow>();

  if (motion_mask.at<int>(y, x) != background_label)
  {
    return nullptr;
  }

  // check flow
  double flow_xe = static_cast<double>(optical_flow.at<cv::Vec2f>(y, x)[0]);
  double flow_ye = static_cast<double>(optical_flow.at<cv::Vec2f>(y, x)[1]);

  if (!(flow_xe != 0 && flow_ye != 0))
  {
    return nullptr;
  }

  // check predicted flow is within image
  Keypoint predicted_kp(static_cast<double>(x) + flow_xe,   static_cast<double>(y) + flow_ye);
  if(!camera_->isKeypointContained(predicted_kp)) {
    return nullptr;
  }

  Feature::Ptr feature = std::make_shared<Feature>();
  feature->keypoint_ = kp;
  feature->age_ = age;
  feature->tracklet_id_ = tracklet_id;
  feature->frame_id_ = frame_id;
  feature->type_ = KeyPointType::STATIC;
  feature->inlier_ = true;
  feature->instance_label_ = background_label;
  feature->tracking_label_ = background_label;
  feature->predicted_keypoint_ = predicted_kp;
  return feature;
}

}
