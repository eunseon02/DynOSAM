/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/frontend/vision/FeatureTracker.hpp"

#include <glog/logging.h>

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/anms/NonMaximumSuppression.h"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/TimingStats.hpp"

namespace dyno {

FeatureTracker::FeatureTracker(const FrontendParams& params, Camera::Ptr camera,
                               ImageDisplayQueue* display_queue)
    : FeatureTrackerBase(params.tracker_params, camera, display_queue),
      frontend_params_(params) {
  static_feature_tracker_ = std::make_unique<KltFeatureTracker>(
      params.tracker_params, camera, display_queue);
  CHECK(!img_size_.empty());
}

Frame::Ptr FeatureTracker::track(FrameId frame_id, Timestamp timestamp,
                                 const ImageContainer& image_container) {
  // take "copy" of tracking_images which is then given to the frame
  // this will mean that the tracking images (input) are not necessarily the
  // same as the ones inside the returned frame
  ImageContainer input_images = image_container;

  info_ = FeatureTrackerInfo();  // clear the info
  info_.frame_id = frame_id;
  info_.timestamp = timestamp;

  if (initial_computation_) {
    // intitial computation
    const cv::Size& other_size = input_images.get<ImageType::RGBMono>().size();
    CHECK(!previous_frame_);
    CHECK_EQ(img_size_.width, other_size.width);
    CHECK_EQ(img_size_.height, other_size.height);
    initial_computation_ = false;
  } else {
    if (params_.use_propogate_mask) {
      propogateMask(input_images);
    }
    CHECK(previous_frame_);
    CHECK_EQ(previous_frame_->frame_id_, frame_id - 1u)
        << "Incoming frame id must be consequative";
  }

  // TODO: figure out some better way of scaling this as it scales with the size
  // of the image...
  // TODO: and make parameter...
  static constexpr auto kDetectionBoaderThickness = 10;  // in pixels
  // create detection mask around the boarder of each dynamic object with some
  // thickness this prevents static and dynamic points being detected around the
  // edge of the dynamic object as there are lots of inconsistencies here the
  // detection mask is in the opencv mask form: CV_8UC1 where white pixels (255)
  // are valid and black pixels (0) should not be detected on
  cv::Mat boarder_detection_mask;
  vision_tools::computeObjectMaskBoundaryMask(
      input_images.get<ImageType::MotionMask>(), boarder_detection_mask,
      kDetectionBoaderThickness, true);

  FeatureContainer static_features;
  {
    utils::TimingStatsCollector static_track_timer("static_feature_track");
    static_features = static_feature_tracker_->trackStatic(
        previous_frame_, input_images, info_, boarder_detection_mask);
  }

  FeatureContainer dynamic_features;
  {
    utils::TimingStatsCollector dynamic_track_timer("dynamic_feature_track");
    trackDynamic(frame_id, input_images, dynamic_features,
                 boarder_detection_mask);
  }

  previous_tracked_frame_ = previous_frame_;  // Update previous frame (previous
                                              // to the newly created frame)

  auto new_frame =
      std::make_shared<Frame>(frame_id, timestamp, camera_, input_images,
                              static_features, dynamic_features, info_);

  // update depth threshold information
  new_frame->setMaxBackgroundDepth(frontend_params_.max_background_depth);
  new_frame->setMaxObjectDepth(frontend_params_.max_object_depth);

  VLOG(1) << "Tracked on frame " << frame_id << " t= " << std::setprecision(15)
          << timestamp << ", object ids "
          << container_to_string(new_frame->getObjectIds());
  previous_frame_ = new_frame;

  return new_frame;
}

void FeatureTracker::trackDynamic(FrameId frame_id,
                                  const ImageContainer& image_container,
                                  FeatureContainer& dynamic_features,
                                  const cv::Mat& detection_mask) {
  // first dectect dynamic points
  const cv::Mat& rgb = image_container.get<ImageType::RGBMono>();
  // flow is going to take us from THIS frame to the next frame (which does not
  // make sense for a realtime system)
  const cv::Mat& flow = image_container.get<ImageType::OpticalFlow>();
  const cv::Mat& motion_mask = image_container.get<ImageType::MotionMask>();

  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();

  ObjectIds instance_labels;
  dynamic_features.clear();

  const auto step_size = params_.semantic_mask_step_size;

  OccupandyGrid2D grid(
      step_size, std::ceil(static_cast<double>(img_size_.width) / step_size),
      std::ceil(static_cast<double>(img_size_.height) / step_size));

  if (previous_frame_) {
    const cv::Mat& previous_motion_mask =
        previous_frame_->image_container_.get<ImageType::MotionMask>();
    utils::TimingStatsCollector tracked_dynamic_features(
        "tracked_dynamic_features");
    for (Feature::Ptr previous_dynamic_feature :
         previous_frame_->usableDynamicFeaturesBegin()) {
      const TrackletId tracklet_id = previous_dynamic_feature->trackletId();
      const size_t age = previous_dynamic_feature->age();

      const Keypoint kp = previous_dynamic_feature->predictedKeypoint();
      const int x = functional_keypoint::u(kp);
      const int y = functional_keypoint::v(kp);
      // ObjectId predicted_label = functional_keypoint::at<ObjectId>(kp,
      // motion_mask);
      const ObjectId predicted_label = motion_mask.at<ObjectId>(y, x);

      if (!detection_mask.empty()) {
        const unsigned char valid_detection =
            detection_mask.at<unsigned char>(y, x);
        if (valid_detection == 0) {
          continue;
        }
      }

      // const Keypoint previous_kp = previous_dynamic_feature->keypoint();
      // ObjectId previous_label =
      // functional_keypoint::at<ObjectId>(previous_kp, previous_motion_mask);
      ObjectId previous_label = previous_dynamic_feature->objectId();
      CHECK_NE(previous_label, background_label);
      CHECK_GT(previous_label, 0);

      PerObjectStatus& object_tracking_info =
          info_.getObjectStatus(predicted_label);
      object_tracking_info.num_previous_track++;

      // true if predicted label not on the background
      const bool is_predicted_object_label =
          predicted_label != background_label;
      // true if predicted label the same as the previous label of the tracked
      // point
      const bool is_precited_same_as_previous =
          predicted_label == previous_label;

      // update stats
      if (!is_predicted_object_label)
        object_tracking_info.num_tracked_with_background_label++;
      if (!is_precited_same_as_previous)
        object_tracking_info.num_tracked_with_different_label++;

      // only include point if it is contained, it is not static and the
      // previous label is the same as the predicted label
      if (camera_->isKeypointContained(kp) && is_predicted_object_label &&
          is_precited_same_as_previous) {
        size_t new_age = age + 1;
        double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(y, x)[0]);
        double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(y, x)[1]);

        OpticalFlow flow(flow_xe, flow_ye);
        const Keypoint predicted_kp =
            Feature::CalculatePredictedKeypoint(kp, flow);

        if (!isWithinShrunkenImage(predicted_kp)) {
          object_tracking_info.num_outside_shrunken_image++;
          continue;
        }

        if (flow_xe == 0 || flow_ye == 0) {
          object_tracking_info.num_zero_flow++;
          continue;
        }

        // limit point tracking of a certain age
        TrackletId tracklet_to_use = tracklet_id;
        if (new_age > 25u) {
          tracklet_to_use = tracked_id_manager.getTrackletIdCount();
          tracked_id_manager.incrementTrackletIdCount();
          new_age = 0;
        }

        // // save correspondences
        Feature::Ptr feature = std::make_shared<Feature>();
        (*feature)
            .objectId(predicted_label)
            .frameId(frame_id)
            .keypointType(KeyPointType::DYNAMIC)
            .age(new_age)
            .trackletId(tracklet_to_use)
            .keypoint(kp)
            .measuredFlow(flow)
            .predictedKeypoint(predicted_kp);

        dynamic_features.add(feature);
        instance_labels.push_back(feature->objectId());

        object_tracking_info.num_track++;

        const size_t cell_idx = grid.getCellIndex(kp);
        grid.occupancy_[cell_idx] = true;
      }
    }
  }

  // struct KepointData {
  //   OpticalFlow flow;
  //   Keypoint predicted_kp;
  // };

  // gtsam::FastMap<ObjectId, KeypointsCV> sampled_keypoints;
  // gtsam::FastMap<ObjectId, KepointData> sampled_keypoint_data;

  // constexpr static auto step = 4u;

  // for (int i = 0; i < rgb.rows - step; i = i + step) {
  //     for (int j = 0; j < rgb.cols - step; j = j + step) {
  //       if (!detection_mask.empty()) {
  //         const unsigned char valid_detection =
  //             detection_mask.at<unsigned char>(i, j);
  //         // //marked invalid by the detection mask
  //         if (valid_detection == 0) {
  //           continue;
  //         }
  //       }

  //       const ObjectId object_id = motion_mask.at<ObjectId>(i, j);

  //       if (object_id == background_label) {
  //         continue;
  //       }

  //       PerObjectStatus& object_tracking_info =
  //       info_.getObjectStatus(object_id);

  //       double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(i, j)[0]);
  //       double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(i, j)[1]);

  //       // TODO: close to zero?
  //       if (flow_xe == 0 || flow_ye == 0) {
  //         // object_tracking_info.num_zero_flow++;
  //         continue;
  //       }

  //       OpticalFlow flow(flow_xe, flow_ye);
  //       Keypoint keypoint(j, i);
  //       const Keypoint predicted_kp =
  //           Feature::CalculatePredictedKeypoint(keypoint, flow);
  //       const size_t cell_idx = grid.getCellIndex(keypoint);

  //       if (isWithinShrunkenImage(keypoint)) {
  //         // save correspondences
  //         auto opencv_keypoint = utils::gtsamPointToKeyPoint(keypoint);
  //         if(!sampled_keypoints.exists(object_id)) {
  //           sampled_keypoints.insert2(object_id, KeypointsCV{});
  //         }
  //         KeypointsCV& keypoints = sampled_keypoints.at(object_id);
  //         keypoints.push_back(opencv_keypoint);

  //       } else {
  //         object_tracking_info.num_outside_shrunken_image++;
  //       }
  //     }
  //   }

  //   const int max_features_to_track = 50;
  //   static constexpr float tolerance = 0.1;
  //   Eigen::MatrixXd binning_mask;

  //   AdaptiveNonMaximumSuppression
  //   non_maximum_supression(AnmsAlgorithmType::RangeTree); for(auto&
  //   [object_id, opencv_keypoints] : sampled_keypoints) {
  //     const PerObjectStatus& object_tracking_info =
  //     info_.getObjectStatus(object_id); const int& number_tracked =
  //     object_tracking_info.num_track; int nr_corners_needed = std::max(
  //       max_features_to_track - number_tracked, 0);

  //     std::vector<cv::KeyPoint>& max_keypoints = opencv_keypoints;
  //     const size_t sampled_size = max_keypoints.size();
  //     max_keypoints = non_maximum_supression.suppressNonMax(
  //       opencv_keypoints,
  //       nr_corners_needed,
  //       tolerance,
  //       img_size_.width,
  //       img_size_.height,
  //       5,
  //       5,
  //       binning_mask);

  //     VLOG(10) << "Kps: " << max_keypoints.size() << " for j=" << object_id
  //     << " after ANMS (originally " << sampled_size << ")";
  //     std::vector<cv::Point2f> points;
  //     cv::KeyPoint::convert(max_keypoints, points);

  //     for(const auto& p : points) {
  //       Keypoint kp = utils::cvPointToGtsam(p);
  //       CHECK(isWithinShrunkenImage(kp));
  //       auto tracklet_id = tracked_id_manager.getTrackletIdCount();
  //       tracked_id_manager.incrementTrackletIdCount();
  //       // Feature::Ptr feature = std::make_shared<Feature>();
  //       // (*feature)
  //       //     .objectId(label)
  //       //     .frameId(frame_id)
  //       //     .keypointType(KeyPointType::DYNAMIC)
  //       //     .age(0)
  //       //     .trackletId(tracklet_id)
  //       //     .keypoint(keypoint)
  //       //     .measuredFlow(flow)
  //       //     .predictedKeypoint(predicted_kp);

  //       // dynamic_features.add(feature);
  //       // instance_labels.push_back(feature->objectId());
  //     }
  //   }

  int step = params_.semantic_mask_step_size;
  for (int i = 0; i < rgb.rows - step; i = i + step) {
    for (int j = 0; j < rgb.cols - step; j = j + step) {
      if (!detection_mask.empty()) {
        const unsigned char valid_detection =
            detection_mask.at<unsigned char>(i, j);
        // //marked invalid by the detection mask
        if (valid_detection == 0) {
          continue;
        }
      }

      const ObjectId label = motion_mask.at<ObjectId>(i, j);

      if (label == background_label) {
        continue;
      }

      PerObjectStatus& object_tracking_info = info_.getObjectStatus(label);

      double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(i, j)[0]);
      double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(i, j)[1]);

      // TODO: close to zero?
      if (flow_xe == 0 || flow_ye == 0) {
        object_tracking_info.num_zero_flow++;
        continue;
      }

      OpticalFlow flow(flow_xe, flow_ye);
      Keypoint keypoint(j, i);
      const Keypoint predicted_kp =
          Feature::CalculatePredictedKeypoint(keypoint, flow);
      const size_t cell_idx = grid.getCellIndex(keypoint);

      if (isWithinShrunkenImage(keypoint) && !grid.isOccupied(cell_idx)) {
        // save correspondences
        auto tracklet_id = tracked_id_manager.getTrackletIdCount();
        tracked_id_manager.incrementTrackletIdCount();
        Feature::Ptr feature = std::make_shared<Feature>();
        (*feature)
            .objectId(label)
            .frameId(frame_id)
            .keypointType(KeyPointType::DYNAMIC)
            .age(0)
            .trackletId(tracklet_id)
            .keypoint(keypoint)
            .measuredFlow(flow)
            .predictedKeypoint(predicted_kp);

        dynamic_features.add(feature);
        instance_labels.push_back(feature->objectId());

        object_tracking_info.num_sampled++;
      } else {
        object_tracking_info.num_outside_shrunken_image++;
      }
    }
  }
}

void FeatureTracker::propogateMask(ImageContainer& image_container) {
  if (!previous_frame_) return;

  const cv::Mat& previous_rgb =
      previous_frame_->image_container_.get<ImageType::RGBMono>();
  const cv::Mat& previous_mask =
      previous_frame_->image_container_.get<ImageType::MotionMask>();
  const cv::Mat& previous_flow =
      previous_frame_->image_container_.get<ImageType::OpticalFlow>();

  // note reference
  cv::Mat& current_mask = image_container.get<ImageType::MotionMask>();

  ObjectIds instance_labels;
  for (const Feature::Ptr& dynamic_feature :
       previous_frame_->usableDynamicFeaturesBegin()) {
    CHECK(dynamic_feature->objectId() != background_label);
    instance_labels.push_back(dynamic_feature->objectId());
  }

  CHECK_EQ(instance_labels.size(), previous_frame_->numDynamicUsableFeatures());
  std::sort(instance_labels.begin(), instance_labels.end());
  instance_labels.erase(
      std::unique(instance_labels.begin(), instance_labels.end()),
      instance_labels.end());
  // each row is correlated with a specific instance label and each column is
  // the tracklet id associated with that label
  std::vector<TrackletIds> object_features(instance_labels.size());

  // collect the predicted labels and semantic labels in vector

  // TODO: inliers?
  for (const Feature::Ptr& dynamic_feature :
       previous_frame_->usableDynamicFeaturesBegin()) {
    CHECK(Feature::IsNotNull(dynamic_feature));
    for (size_t j = 0; j < instance_labels.size(); j++) {
      // save object label for object j with feature i
      if (dynamic_feature->objectId() == instance_labels[j]) {
        object_features[j].push_back(dynamic_feature->trackletId());
        CHECK(dynamic_feature->objectId() != background_label);
        break;
      }
    }
  }

  // check each object label distribution in the coming frame
  for (size_t i = 0; i < object_features.size(); i++) {
    // labels at the current mask using the predicted keypoint from the previous
    // frame each iteration is per label so temp_label should correspond to
    // features within the same object
    ObjectIds temp_label;
    for (size_t j = 0; j < object_features[i].size(); j++) {
      // feature at k-1
      Feature::Ptr feature = previous_frame_->dynamic_features_.getByTrackletId(
          object_features[i][j]);
      CHECK(Feature::IsNotNull(feature));
      // kp at k
      const Keypoint& predicted_kp = feature->predictedKeypoint();
      const int u = functional_keypoint::u(predicted_kp);
      const int v = functional_keypoint::v(predicted_kp);
      // ensure u and v are sitll inside the CURRENT frame
      if (u < previous_rgb.cols && u > 0 && v < previous_rgb.rows && v > 0) {
        // add instance label at predicted keypoint
        temp_label.push_back(current_mask.at<ObjectId>(v, u));
      }
    }

    // this is a lovely magic number inherited from some old code :)
    if (temp_label.size() < 150) {
      LOG(WARNING) << "not enoug points to track object " << instance_labels[i]
                   << " points size - " << temp_label.size();
      // TODO:mark has static!!???
      continue;
    }

    // find label that appears most in LabTmp()
    // (1) count duplicates
    std::map<int, int> label_duplicates;
    // k is object label
    for (int k : temp_label) {
      if (label_duplicates.find(k) == label_duplicates.end()) {
        label_duplicates.insert({k, 0});
      } else {
        label_duplicates.at(k)++;
      }
    }
    // (2) and sort them by descending order by number of times an object
    // appeared (ie. by pair.second)
    std::vector<std::pair<int, int>> sorted;
    for (auto k : label_duplicates) {
      sorted.push_back(std::make_pair(k.first, k.second));
    }

    auto sort_pair_int = [](const std::pair<int, int>& a,
                            const std::pair<int, int>& b) -> bool {
      return (a.second > b.second);
    };
    std::sort(sorted.begin(), sorted.end(), sort_pair_int);

    // recover the missing mask (time consuming!)
    // LOG(INFO) << sorted[0].first << " " << sorted[0].second << " " <<
    // instance_labels[i];
    //  if (sorted[0].second < 30)
    // {
    //   LOG(WARNING) << "not enoug points to track object " <<
    //   instance_labels[i] << " points size - "
    //                << sorted[0].second;
    //   //TODO:mark has static!!
    //   continue;
    // }
    if (sorted[0].first == 0)  //?
    // if (sorted[0].first == instance_labels[i])  //?
    {
      for (int j = 0; j < previous_rgb.rows; j++) {
        for (int k = 0; k < previous_rgb.cols; k++) {
          if (previous_mask.at<ObjectId>(j, k) == instance_labels[i]) {
            const double flow_xe =
                static_cast<double>(previous_flow.at<cv::Vec2f>(j, k)[0]);
            const double flow_ye =
                static_cast<double>(previous_flow.at<cv::Vec2f>(j, k)[1]);

            if (flow_xe == 0 || flow_ye == 0) {
              continue;
            }

            OpticalFlow flow(flow_xe, flow_ye);
            // x, y
            Keypoint kp(k, j);
            const Keypoint predicted_kp =
                Feature::CalculatePredictedKeypoint(kp, flow);

            if (!isWithinShrunkenImage(predicted_kp)) {
              continue;
            }

            if ((predicted_kp(0) < previous_rgb.cols && predicted_kp(0) > 0 &&
                 predicted_kp(1) < previous_rgb.rows && predicted_kp(1) > 0)) {
              current_mask.at<ObjectId>(functional_keypoint::v(predicted_kp),
                                        functional_keypoint::u(predicted_kp)) =
                  instance_labels[i];
              //  current_rgb
              // updated_mask_points++;
            }
          }
        }
      }
    }
  }
}

}  // namespace dyno
