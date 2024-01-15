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
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/frontend/vision/OccupancyGrid2D.hpp"

#include "dynosam/utils/TimingStats.hpp"

#include "dynosam/visualizer/ColourMap.hpp"

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
    //take "copy" of tracking_images which is then given to the frame
    //this will mean that the tracking images (input) are not necessarily the same as the ones inside the returned frame
    TrackingInputImages input_images = tracking_images;

    if(initial_computation_) {
        //intitial computation
        const cv::Size& other_size = input_images.get<ImageType::RGBMono>().size();
        CHECK(!previous_frame_);
        CHECK(img_size_.width == other_size.width && img_size_.height == other_size.height);
        initial_computation_ = false;
    }
    else {
        propogateMask(input_images);
        CHECK(previous_frame_);
        CHECK_EQ(previous_frame_->frame_id_, frame_id - 1u) << "Incoming frame id must be consequative";
    }

    FeatureContainer static_features;
    {
      utils::TimingStatsCollector static_track_timer("static_feature_track");
      trackStatic(frame_id, input_images, static_features, n_optical_flow, n_new_tracks);
    }

    FeatureContainer dynamic_features;
    {
      utils::TimingStatsCollector dynamic_track_timer("dynamic_feature_track");
      trackDynamic(frame_id, input_images, dynamic_features);
    }

    previous_tracked_frame_ = previous_frame_; // Update previous frame (previous to the newly created frame)


    auto new_frame = std::make_shared<Frame>(
      frame_id,
      timestamp,
      camera_,
      tracking_images,
      static_features,
      dynamic_features);

    LOG(INFO) << "Tracked on frame " << frame_id << " t= " << timestamp << ", object ids " << container_to_string(new_frame->getObjectIds());
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
  for (const Feature::Ptr& feature : current_frame.static_features_) {
    const Keypoint& px_cur = feature->keypoint_;
    if (!feature->usable()) {  // Untracked landmarks are red.
      cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 4, red, 2);
    } else {

      const Feature::Ptr& prev_feature = previous_frame.static_features_.getByTrackletId(feature->tracklet_id_);
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 6, green, 1);
        const Keypoint& px_prev = prev_feature->keypoint_;
        cv::arrowedLine(img_rgb, utils::gtsamPointToCv(px_prev), utils::gtsamPointToCv(px_cur), green, 1);
      } else {  // New feature tracks are blue.
        cv::circle(img_rgb, utils::gtsamPointToCv(px_cur), 6, blue, 1);
      }
    }
  }

  for(const auto& [instance_label, object_observation] : current_frame.object_observations_) {
    // CHECK(object_observation.marked_as_moving_);
    //get average center of 2 object
    FeaturePtrs features = current_frame.collectFeatures(object_observation.object_features_);

    size_t count = 0;
    int center_x = 0, center_y = 0;
    auto usable_iterator = internal::filter_const_iterator<FeaturePtrs>(features, [](const Feature::Ptr& f) { return Feature::IsUsable(f); });
    for(const Feature::Ptr& feature : usable_iterator) {
      center_x += functional_keypoint::u(feature->keypoint_);
      center_y += functional_keypoint::v(feature->keypoint_);
      count++;
    }

    center_x /= features.size();
    center_y /= features.size();

    cv::putText(
      img_rgb,
      std::to_string(object_observation.tracking_label_),
      cv::Point(center_x, center_y),
      cv::FONT_HERSHEY_DUPLEX,
      1.0,
      CV_RGB(118, 185, 0),  // font color
      3);

  }

  for ( const Feature::Ptr& feature : current_frame.dynamic_features_) {
    const Keypoint& px_cur = feature->keypoint_;
    if (!feature->usable()) {  // Untracked landmarks are red.
      // cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 1, red, 2);
    } else {


      const Feature::Ptr& prev_feature = previous_frame.dynamic_features_.getByTrackletId(feature->tracklet_id_);
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        // cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 6, green, 1);
        const Keypoint& px_prev = prev_feature->keypoint_;
        const cv::Scalar colour = ColourMap::getObjectColour(feature->instance_label_, true);
        cv::arrowedLine(img_rgb, utils::gtsamPointToCv(px_prev), utils::gtsamPointToCv(px_cur), colour, 1);
      } else {  // New feature tracks are blue.
        // cv::circle(img_rgb, utils::gtsamPointToCv(px_cur), 1, blue, 1);
      }
    }
  }


  return img_rgb;

}


void FeatureTracker::trackStatic(FrameId frame_id, const TrackingInputImages& tracking_images, FeatureContainer& static_features, size_t& n_optical_flow,
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

 // assign tracked features to grid and add to static features
  static_features.clear();
  std::vector<bool> detections_tracked(detected_keypoints.size(), false);

  const int& min_tracks = params_.max_tracking_points_bg;
  // std::vector<std::size_t> grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
  OccupandyGrid2D grid(params_.cell_size,
          std::ceil(static_cast<double>(img_size_.width)/params_.cell_size),
          std::ceil(static_cast<double>(img_size_.height)/params_.cell_size));

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
          static_features.add(feature);
          const size_t cell_idx = grid.getCellIndex(kp);
          grid.occupancy_[cell_idx] = true;
        }
      }
    }
  }

  n_optical_flow = static_features.size();
  LOG(INFO) << "tracked with optical flow - " << n_optical_flow;


  if (static_features.size() < min_tracks)
  {
    // iterate over new observations
    for (size_t i = 0; i < detected_keypoints.size(); i++)
    {
      if (static_features.size() >= min_tracks)
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
      const size_t cell_idx = grid.getCellIndex(kp);
      // if (posInGrid(kp, grid_x, grid_y))
      // {
        // only add of we have less than n_reserve ammount
        // if (grid[grid_x][grid_y].size() < n_reserve)
        if (!grid.isOccupied(cell_idx))
        {
          const size_t age = 0;
          size_t tracklet_id = tracklet_count;
          Feature::Ptr feature = constructStaticFeature(tracking_images, kp, age, tracklet_id, frame_id);
          if (feature)
          {

            tracklet_count++;
            grid.occupancy_[cell_idx] = true;
            static_features.add(feature);
            // new_features.push_back(feature);
            // draw an untracked kp
            // utils::drawCircleInPlace(viz, utils::gtsamPointToCv(feature->keypoint_), cv::Scalar(0, 255, 0));
          }
        }
      // }
    }
  }

  size_t total_tracks = static_features.size();
  n_new_tracks = total_tracks - n_optical_flow;

}


void FeatureTracker::trackDynamic(FrameId frame_id, const TrackingInputImages& tracking_images, FeatureContainer& dynamic_features) {
  // first dectect dynamic points
  const cv::Mat& rgb = tracking_images.get<ImageType::RGBMono>();
  const cv::Mat& flow = tracking_images.get<ImageType::OpticalFlow>();
  const cv::Mat& motion_mask = tracking_images.get<ImageType::MotionMask>();

  ObjectIds instance_labels;
  dynamic_features.clear();

  OccupandyGrid2D grid(FLAGS_semantic_mask_step_size,
          std::ceil(static_cast<double>(img_size_.width)/FLAGS_semantic_mask_step_size),
          std::ceil(static_cast<double>(img_size_.height)/FLAGS_semantic_mask_step_size));

  // cv::Mat viz;
  // rgb.copyTo(viz);

  // a mask to show which feature have been with optical flow from the previous frame
  // zeros mean have not been tracked
  cv::Mat tracked_feature_mask = cv::Mat::zeros(rgb.size(), CV_8UC1);

  if (previous_frame_)
  {
    const cv::Mat& previous_motion_mask = previous_frame_->tracking_images_.get<ImageType::MotionMask>();
    utils::TimingStatsCollector tracked_dynamic_features("tracked_dynamic_features");
    for (Feature::Ptr previous_dynamic_feature : previous_frame_->usableDynamicFeaturesBegin())
    {
      const TrackletId tracklet_id = previous_dynamic_feature->tracklet_id_;
      const size_t age = previous_dynamic_feature->age_;

      const Keypoint kp = previous_dynamic_feature->predicted_keypoint_;
      const int x = functional_keypoint::u(kp);
      const int y = functional_keypoint::v(kp);
      ObjectId predicted_label = motion_mask.at<ObjectId>(y, x);


      const Keypoint previous_kp = previous_dynamic_feature->keypoint_;
      const int prev_x = functional_keypoint::u(previous_kp);
      const int prev_y = functional_keypoint::v(previous_kp);
      ObjectId previous_label = previous_motion_mask.at<ObjectId>(prev_y, prev_x);

      //only include point if it is contained, it is not static and the previous label is the same as the predicted label
      if(camera_->isKeypointContained(kp) && predicted_label != background_label && predicted_label == previous_label) {
        size_t new_age = age + 1;
        double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(y, x)[0]);
        double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(y, x)[1]);

        OpticalFlow flow(flow_xe, flow_ye);

        // // save correspondences
        Feature::Ptr feature = std::make_shared<Feature>();
        feature->instance_label_ = predicted_label;
        feature->tracking_label_ = predicted_label;
        feature->frame_id_ = frame_id;
        feature->type_ = KeyPointType::DYNAMIC;
        feature->age_ = new_age;
        feature->tracklet_id_ = tracklet_id;
        feature->keypoint_ = kp;
        feature->measured_flow_ = flow;
        feature->predicted_keypoint_ = Feature::CalculatePredictedKeypoint(kp, flow);


        dynamic_features.add(feature);
        tracked_feature_mask.at<uchar>(y, x) = 1;
        instance_labels.push_back(feature->instance_label_);

        const size_t cell_idx = grid.getCellIndex(kp);
        grid.occupancy_[cell_idx] = true;
      }
    }
  }

  LOG(INFO) << "Tracked dynamic points - " << dynamic_features.size();

  int step = FLAGS_semantic_mask_step_size;
  for (int i = 0; i < rgb.rows - step; i = i + step)
  {
    for (int j = 0; j < rgb.cols - step; j = j + step)
    {
      if (motion_mask.at<ObjectId>(i, j) == background_label)
      {
        continue;
      }


      double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(i, j)[0]);
      double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(i, j)[1]);

      OpticalFlow flow(flow_xe, flow_ye);
      Keypoint keypoint(j, i);
      const size_t cell_idx = grid.getCellIndex(keypoint);

      if(grid.isOccupied(cell_idx)) {continue;}

      // we are within the image bounds?
      if (j + flow_xe < rgb.cols && j + flow_xe > 0 && i + flow_ye < rgb.rows && i + flow_ye > 0)
      {
        // save correspondences
        Feature::Ptr feature = std::make_shared<Feature>();

        feature->instance_label_ = motion_mask.at<ObjectId>(i, j);
        feature->tracking_label_ = motion_mask.at<ObjectId>(i, j);
        feature->frame_id_ = frame_id;
        feature->type_ = KeyPointType::DYNAMIC;
        feature->age_ = 0;
        feature->tracklet_id_ = tracklet_count;
        tracklet_count++;
        feature->predicted_keypoint_ = Feature::CalculatePredictedKeypoint(keypoint, flow);
        feature->keypoint_ = keypoint;

        dynamic_features.add(feature);
        instance_labels.push_back(feature->instance_label_);
      }
    }
  }

}


void FeatureTracker::propogateMask(TrackingInputImages& tracking_images) {
  if(!previous_frame_) return;


  const cv::Mat& previous_rgb = previous_frame_->tracking_images_.get<ImageType::RGBMono>();
  const cv::Mat& previous_mask = previous_frame_->tracking_images_.get<ImageType::MotionMask>();
  const cv::Mat& previous_flow = previous_frame_->tracking_images_.get<ImageType::OpticalFlow>();

  // note reference
  cv::Mat& current_mask = tracking_images.get<ImageType::MotionMask>();

  ObjectIds instance_labels;
  for(const Feature::Ptr& dynamic_feature : previous_frame_->usableDynamicFeaturesBegin()) {
    CHECK(dynamic_feature->instance_label_ != background_label);
    instance_labels.push_back(dynamic_feature->instance_label_);
  }

  CHECK_EQ(instance_labels.size(), previous_frame_->numDynamicUsableFeatures());
  std::sort(instance_labels.begin(), instance_labels.end());
  instance_labels.erase(std::unique(instance_labels.begin(), instance_labels.end()), instance_labels.end());
  //each row is correlated with a specific instance label and each column is the tracklet id associated with that label
  std::vector<TrackletIds> object_features(instance_labels.size());

  // collect the predicted labels and semantic labels in vector

  //TODO: inliers?
  for (const Feature::Ptr& dynamic_feature : previous_frame_->usableDynamicFeaturesBegin())
  {
    CHECK(Feature::IsNotNull(dynamic_feature));
    for (size_t j = 0; j < instance_labels.size(); j++)
    {
      // save object label for object j with feature i
      if (dynamic_feature->instance_label_ == instance_labels[j])
      {
        object_features[j].push_back(dynamic_feature->tracklet_id_);
        CHECK(dynamic_feature->instance_label_ != background_label);
        break;
      }
    }
  }

   // check each object label distribution in the coming frame
  for (size_t i = 0; i < object_features.size(); i++)
  {
    //labels at the current mask using the predicted keypoint from the previous frame
    //each iteration is per label so temp_label should correspond to features within the same object
    ObjectIds temp_label;
    for (size_t j = 0; j < object_features[i].size(); j++)
    {
      Feature::Ptr feature = previous_frame_->dynamic_features_.getByTrackletId(object_features[i][j]);
      CHECK(Feature::IsNotNull(feature));
      const Keypoint& predicted_kp = feature->predicted_keypoint_;
      const int u = functional_keypoint::u(predicted_kp);
      const int v = functional_keypoint::v(predicted_kp);
      // ensure u and v are sitll inside the CURRENT frame
      if (u < previous_rgb.cols && u > 0 && v < previous_rgb.rows && v > 0)
      {
        //add instance label at predicted keypoint
        temp_label.push_back(current_mask.at<ObjectId>(v, u));
      }
    }

    if (temp_label.size() < 100)
    {
      LOG(WARNING) << "not enoug points to track object " << instance_labels[i] << " points size - "
                   << temp_label.size();
      //TODO:mark has static!!
      continue;
    }

    // find label that appears most in LabTmp()
    // (1) count duplicates
    std::map<int, int> label_duplicates;
    //k is object label
    for (int k : temp_label)
    {
      if (label_duplicates.find(k) == label_duplicates.end())
      {
        label_duplicates.insert({ k, 0 });
      }
      else
      {
        label_duplicates.at(k)++;
      }
    }
    // (2) and sort them by descending order by number of times an object appeared (ie. by pair.second)
    std::vector<std::pair<int, int>> sorted;
    for (auto k : label_duplicates)
    {
      sorted.push_back(std::make_pair(k.first, k.second));
    }

    auto sort_pair_int = [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool {
      return (a.second > b.second);
    };
    std::sort(sorted.begin(), sorted.end(), sort_pair_int);

    // recover the missing mask (time consuming!)
    // LOG(INFO) << sorted[0].first << " " << sorted[0].second << " " << instance_labels[i];
    //  if (sorted[0].second < 30)
    // {
    //   LOG(WARNING) << "not enoug points to track object " << instance_labels[i] << " points size - "
    //                << sorted[0].second;
    //   //TODO:mark has static!!
    //   continue;
    // }
    if (sorted[0].first == 0)  //?
    // if (sorted[0].first == instance_labels[i])  //?
    {
      for (int j = 0; j < previous_rgb.rows; j++)
      {
        for (int k = 0; k < previous_rgb.cols; k++)
        {
          if (previous_mask.at<ObjectId>(j, k) == instance_labels[i])
          {
            const int flow_x = previous_flow.at<cv::Vec2f>(j, k)[0];
            const int flow_y = previous_flow.at<cv::Vec2f>(j, k)[1];

            if (k + flow_x < previous_rgb.cols && k + flow_x > 0 && j + flow_y < previous_rgb.rows && j + flow_y > 0)
            {
              current_mask.at<ObjectId>(j + flow_y, k + flow_x) = instance_labels[i];
              // updated_mask_points++;
            }
          }
        }
      }
    }
  }
}


// bool FeatureTracker::posInGrid(const Keypoint& kp, int& pos_x, int& pos_y) const {
//   const int u = functional_keypoint::u(kp);
//   const int v = functional_keypoint::v(kp);
//   pos_x = round((u - min_x_) * grid_elements_width_inv_);
//   pos_y = round((v - min_y_) * grid_elements_height_inv_);

//   // Keypoint's coordinates are undistorted, which could cause to go out of the image
//   if (pos_x < 0 || pos_x >= FRAME_GRID_COLS || pos_y < 0 || pos_y >= FRAME_GRID_ROWS)
//     return false;

//   return true;

// }


// void FeatureTracker::computeImageBounds(const cv::Size& size, int& min_x, int& max_x, int& min_y, int& max_y) const {
//     // TODO: distortion
//     //see orbslam
//     min_x = 0;
//     max_x = size.width;
//     min_y = 0;
//     max_y = size.height;
// }


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

  OpticalFlow flow(flow_xe, flow_ye);

  // check predicted flow is within image
  Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(kp, flow);
  if(!camera_->isKeypointContained(predicted_kp)) {
    return nullptr;
  }

  Feature::Ptr feature = std::make_shared<Feature>();
  feature->keypoint_ = kp;
  feature->measured_flow_ = flow;
  feature->predicted_keypoint_ = predicted_kp;
  feature->age_ = age;
  feature->tracklet_id_ = tracklet_id;
  feature->frame_id_ = frame_id;
  feature->type_ = KeyPointType::STATIC;
  feature->inlier_ = true;
  feature->instance_label_ = background_label;
  feature->tracking_label_ = background_label;
  return feature;
}

}
