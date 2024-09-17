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

#include "dynosam/utils/TimingStats.hpp"

#include "dynosam/visualizer/ColourMap.hpp"

#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>



DEFINE_int32(semantic_mask_step_size, 3, "The step sized used across the semantic mask when sampling points");
DEFINE_bool(use_propogate_mask, true, "If true, the semantic mask will be propogated with optical flow");

namespace dyno {



FeatureTracker::FeatureTracker(const FrontendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : FeatureTrackerBase(params, camera, display_queue) {
    static_feature_tracker_ = std::make_unique<KltFeatureTracker>(
      params, camera, display_queue
    );
    CHECK(!img_size_.empty());
}

Frame::Ptr FeatureTracker::track(FrameId frame_id, Timestamp timestamp, const ImageContainer& image_container) {
    //take "copy" of tracking_images which is then given to the frame
    //this will mean that the tracking images (input) are not necessarily the same as the ones inside the returned frame
    ImageContainer input_images = image_container;

    info_ = FeatureTrackerInfo(); //clear the info
    info_.frame_id = frame_id;
    info_.timestamp = timestamp;


    if(initial_computation_) {
        //intitial computation
        const cv::Size& other_size = input_images.get<ImageType::RGBMono>().size();
        CHECK(!previous_frame_);
        CHECK_EQ(img_size_.width, other_size.width);
        CHECK_EQ(img_size_.height, other_size.height);
        initial_computation_ = false;
    }
    else {

        if(FLAGS_use_propogate_mask) {
          propogateMask(input_images);
        }
        CHECK(previous_frame_);
        CHECK_EQ(previous_frame_->frame_id_, frame_id - 1u) << "Incoming frame id must be consequative";
    }

    //TODO: figure out some better way of scaling this as it scales with the size of the image...
    //TODO: and make parameter...
    static constexpr auto kDetectionBoaderThickness = 20; //in pixels
    //create detection mask around the boarder of each dynamic object with some thickness
    //this prevents static and dynamic points being detected around the edge of the dynamic object
    //as there are lots of inconsistencies here
    //the detection mask is in the opencv mask form: CV_8UC1 where white pixels (255) are valid
    //and black pixels (0) should not be detected on
    cv::Mat boarder_detection_mask;
    vision_tools::computeObjectMaskBoundaryMask(
        input_images.get<ImageType::MotionMask>(),
        boarder_detection_mask,
        kDetectionBoaderThickness,
        true
    );

    FeatureContainer static_features;
    {
      utils::TimingStatsCollector static_track_timer("static_feature_track");
      static_features = static_feature_tracker_->trackStatic(previous_frame_, input_images, info_, boarder_detection_mask);
    }

    FeatureContainer dynamic_features;
    {
      utils::TimingStatsCollector dynamic_track_timer("dynamic_feature_track");
      trackDynamic(
        frame_id,
        input_images,
        dynamic_features,
        boarder_detection_mask);
    }

    previous_tracked_frame_ = previous_frame_; // Update previous frame (previous to the newly created frame)


    auto new_frame = std::make_shared<Frame>(
      frame_id,
      timestamp,
      camera_,
      input_images,
      static_features,
      dynamic_features,
      info_);

    LOG(INFO) << "Tracked on frame " << frame_id << " t= " << std::setprecision(15) << timestamp << ", object ids " << container_to_string(new_frame->getObjectIds());
    previous_frame_ = new_frame;

    return new_frame;

}

//doesnt make any sense for this function to be here?
//Debug could be part of a global config singleton?
cv::Mat FeatureTracker::computeImageTracks(const Frame& previous_frame, const Frame& current_frame, bool debug) const {
  cv::Mat img_rgb;

  const cv::Mat& rgb = current_frame.image_container_.get<ImageType::RGBMono>();
  rgb.copyTo(img_rgb);


  static const cv::Scalar gray(0, 255, 255);
  static const cv::Scalar red(0, 0, 255);
  static const cv::Scalar green(0, 255, 0);
  static const cv::Scalar blue(255, 0, 0);

  constexpr static int kFeatureThicknessDebug = 5;
  constexpr static int kFeatureThickness = 4;
  int static_point_thickness = debug ? kFeatureThicknessDebug : kFeatureThickness;

  int num_static_tracks = 0;
  // Add all keypoints in cur_frame with the tracks.
  for (const Feature::Ptr& feature : current_frame.static_features_) {
    const Keypoint& px_cur = feature->keypoint();
    const auto pc_cur = utils::gtsamPointToCv(px_cur);
    if (!feature->usable() && debug) {  // Untracked landmarks are red.
      cv::circle(img_rgb,  pc_cur, static_point_thickness, red, 2);
    } else {

      const Feature::Ptr& prev_feature = previous_frame.static_features_.getByTrackletId(feature->trackletId());
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        cv::circle(img_rgb,  pc_cur, static_point_thickness, green, 1);

          // draw the optical flow arrow
        const auto pc_prev = utils::gtsamPointToCv(prev_feature->keypoint());
        cv::arrowedLine(img_rgb, pc_prev, pc_cur, green, 1);

        num_static_tracks++;

      } else if(debug) {  // New feature tracks are blue.
        cv::circle(img_rgb, pc_cur, 6, blue, 1);
      }
    }
  }

  // for(const auto& [instance_label, object_observation] : current_frame.object_observations_) {
  //   // CHECK(object_observation.marked_as_moving_);
  //   //get average center of 2 object
  //   FeaturePtrs features = current_frame.collectFeatures(object_observation.object_features_);

  //   size_t count = 0;
  //   int center_x = 0, center_y = 0;
  //   auto usable_iterator = internal::filter_const_iterator<FeaturePtrs>(features, [](const Feature::Ptr& f) { return Feature::IsUsable(f); });
  //   for(const Feature::Ptr& feature : usable_iterator) {
  //     center_x += functional_keypoint::u(feature->keypoint_);
  //     center_y += functional_keypoint::v(feature->keypoint_);
  //     count++;
  //   }

  //   center_x /= features.size();
  //   center_y /= features.size();

  //   cv::putText(
  //     img_rgb,
  //     std::to_string(object_observation.tracking_label_),
  //     cv::Point(center_x, center_y),
  //     cv::FONT_HERSHEY_DUPLEX,
  //     1.0,
  //     CV_RGB(118, 185, 0),  // font color
  //     3);

  // }

  for ( const Feature::Ptr& feature : current_frame.dynamic_features_) {
    const Keypoint& px_cur = feature->keypoint();
    if (!feature->usable()) {  // Untracked landmarks are red.
      // cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 1, red, 2);
    } else {


      const Feature::Ptr& prev_feature = previous_frame.dynamic_features_.getByTrackletId(feature->trackletId());
      if (prev_feature) {
        // If feature was in previous frame, display tracked feature with
        // green circle/line:
        // cv::circle(img_rgb,  utils::gtsamPointToCv(px_cur), 6, green, 1);
        const Keypoint& px_prev = prev_feature->keypoint();
        const cv::Scalar colour = Color::uniqueId(feature->objectId()).bgra();
        cv::arrowedLine(img_rgb, utils::gtsamPointToCv(px_prev), utils::gtsamPointToCv(px_cur), colour, 1);
      } else {  // New feature tracks are blue.
        // cv::circle(img_rgb, utils::gtsamPointToCv(px_cur), 1, blue, 1);
      }
    }
  }


  std::vector<ObjectId> objects_to_print;
  for(const auto& object_observation_pair : current_frame.object_observations_) {
      const ObjectId object_id = object_observation_pair.first;
      const cv::Rect& bb = object_observation_pair.second.bounding_box_;

      //TODO: if its marked as moving!!

      if(bb.empty()) { continue; }


      objects_to_print.push_back(object_id);
      const cv::Scalar colour = Color::uniqueId(object_id).bgra();
      const std::string label = "object " + std::to_string(object_id);
      utils::drawLabeledBoundingBox(img_rgb, label, colour, bb);

  }

  //draw text info
  std::stringstream ss;
  ss << "Frame ID: " << current_frame.getFrameId() << " | ";
  ss << "VO tracks: " << num_static_tracks << " | ";
  ss << "Objects: ";

  if(objects_to_print.empty()) {
    ss << "None";
  }
  else {
    ss << "[";
    for (size_t i = 0; i < objects_to_print.size(); ++i) {
        ss << objects_to_print[i];
        if (i != objects_to_print.size() - 1) {
            ss << ", "; // Add comma between elements
        }
    }
    ss << "]";
  }

  constexpr static double kFontScale = 0.6;
  constexpr static int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  constexpr static int kThickness = 1;


  //taken from ORB-SLAM2 ;)
  int base_line;
  cv::Size text_size = cv::getTextSize(ss.str(),kFontFace, kFontScale, kThickness, &base_line);
  cv::Mat image_text = cv::Mat(img_rgb.rows+text_size.height+10,img_rgb.cols,img_rgb.type());
  img_rgb.copyTo(image_text.rowRange(0,img_rgb.rows).colRange(0,img_rgb.cols));
  image_text.rowRange(img_rgb.rows,image_text.rows) = cv::Mat::zeros(text_size.height+10,img_rgb.cols,img_rgb.type());
  cv::putText(image_text, ss.str(), cv::Point(5,image_text.rows-5), kFontFace, kFontScale, cv::Scalar(255, 255, 255), kThickness);



  return image_text;

}




void FeatureTracker::trackDynamic(FrameId frame_id, const ImageContainer& image_container, FeatureContainer& dynamic_features, const cv::Mat& detection_mask) {
  // first dectect dynamic points
  const cv::Mat& rgb = image_container.get<ImageType::RGBMono>();
  //flow is going to take us from THIS frame to the next frame (which does not make sense for a realtime system)
  const cv::Mat& flow = image_container.get<ImageType::OpticalFlow>();
  const cv::Mat& motion_mask = image_container.get<ImageType::MotionMask>();


  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();

  ObjectIds instance_labels;
  dynamic_features.clear();

  OccupandyGrid2D grid(FLAGS_semantic_mask_step_size,
          std::ceil(static_cast<double>(img_size_.width)/FLAGS_semantic_mask_step_size),
          std::ceil(static_cast<double>(img_size_.height)/FLAGS_semantic_mask_step_size));


  if (previous_frame_)
  {
    const cv::Mat& previous_motion_mask = previous_frame_->image_container_.get<ImageType::MotionMask>();
    utils::TimingStatsCollector tracked_dynamic_features("tracked_dynamic_features");
    for (Feature::Ptr previous_dynamic_feature : previous_frame_->usableDynamicFeaturesBegin())
    {
      const TrackletId tracklet_id = previous_dynamic_feature->trackletId();
      const size_t age = previous_dynamic_feature->age();

      const Keypoint kp = previous_dynamic_feature->predictedKeypoint();
      ObjectId predicted_label = functional_keypoint::at<ObjectId>(kp, motion_mask);
      // CHECK_NE(predicted_label, background_label);
      const int x = functional_keypoint::u(kp);
      const int y = functional_keypoint::v(kp);

      const Keypoint previous_kp = previous_dynamic_feature->keypoint();
      // ObjectId previous_label = functional_keypoint::at<ObjectId>(previous_kp, previous_motion_mask);
      ObjectId previous_label = previous_dynamic_feature->objectId();
      CHECK_NE(previous_label, background_label);


      PerObjectStatus& object_tracking_info = info_.getObjectStatus(predicted_label);
      object_tracking_info.num_previous_track++;

      //true if predicted label not on the background
      const bool is_predicted_object_label = predicted_label != background_label;
      //true if predicted label the same as the previous label of the tracked point
      const bool is_precited_same_as_previous = predicted_label == previous_label;

      //update stats
      if(!is_predicted_object_label) object_tracking_info.num_tracked_with_background_label++;
      if(!is_precited_same_as_previous) object_tracking_info.num_tracked_with_different_label++;


      //only include point if it is contained, it is not static and the previous label is the same as the predicted label
      if(camera_->isKeypointContained(kp) && is_predicted_object_label && is_precited_same_as_previous) {
        size_t new_age = age + 1;
        double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(y, x)[0]);
        double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(y, x)[1]);

        OpticalFlow flow(flow_xe, flow_ye);
        const Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(kp, flow);

        if(!isWithinShrunkenImage(predicted_kp)) {
          object_tracking_info.num_outside_shrunken_image++;
          continue;
        }

        if(flow_xe == 0 || flow_ye == 0) {
          object_tracking_info.num_zero_flow++;
          continue;
        }

        //limit point tracking of a certain age
        TrackletId tracklet_to_use = tracklet_id;
        if(new_age > 25u) {
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
        // feature->instance_label_ = predicted_label;
        // feature->tracking_label_ = predicted_label;
        // feature->frame_id_ = frame_id;
        // feature->type_ = KeyPointType::DYNAMIC;
        // feature->age_ = new_age;
        // feature->tracklet_id_ = tracklet_to_use;
        // feature->keypoint_ = kp;
        // feature->measured_flow_ = flow;
        // feature->predicted_keypoint_ = predicted_kp;

        dynamic_features.add(feature);
        instance_labels.push_back(feature->objectId());

        object_tracking_info.num_track++;

        const size_t cell_idx = grid.getCellIndex(kp);
        grid.occupancy_[cell_idx] = true;
      }
    }
  }

  int step = FLAGS_semantic_mask_step_size;
  for (int i = 0; i < rgb.rows - step; i = i + step)
  {
    for (int j = 0; j < rgb.cols - step; j = j + step)
    {

      if(!detection_mask.empty()) {
        const unsigned char valid_detection = detection_mask.at<unsigned char>(i, j);
        // //marked invalid by the detection mask
        if(valid_detection == 0) {
          continue;
        }
      }

      const ObjectId label = motion_mask.at<ObjectId>(i, j);
      PerObjectStatus& object_tracking_info = info_.getObjectStatus(label);

      if (label == background_label)
      {
        continue;
      }

      double flow_xe = static_cast<double>(flow.at<cv::Vec2f>(i, j)[0]);
      double flow_ye = static_cast<double>(flow.at<cv::Vec2f>(i, j)[1]);

      //TODO: close to zero?
      if(flow_xe == 0 || flow_ye == 0) {
        object_tracking_info.num_zero_flow++;
        continue;
      }

      OpticalFlow flow(flow_xe, flow_ye);
      Keypoint keypoint(j, i);
      const Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(keypoint, flow);
      const size_t cell_idx = grid.getCellIndex(keypoint);

      // //TODO: this is a problem for the omd dataset?
      // if(grid.isOccupied(cell_idx)) {continue;}

      // if ((predicted_kp(0) < rgb.cols && predicted_kp(0) > 0 && predicted_kp(1) < rgb.rows && predicted_kp(1) > 0))
      if(isWithinShrunkenImage(keypoint) && !grid.isOccupied(cell_idx))
      {
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
        // Feature::Ptr feature = std::make_shared<Feature>();

        // feature->instance_label_ = label;
        // feature->tracking_label_ = label;
        // feature->frame_id_ = frame_id;
        // feature->type_ = KeyPointType::DYNAMIC;
        // feature->age_ = 0;
        // feature->tracklet_id_ = tracked_id_manager.getTrackletIdCount();
        // tracked_id_manager.incrementTrackletIdCount();
        // feature->predicted_keypoint_ = predicted_kp;
        // feature->measured_flow_ = flow;
        // feature->keypoint_ = keypoint;

        dynamic_features.add(feature);
        instance_labels.push_back(feature->objectId());

        object_tracking_info.num_sampled++;
      }
      else {
        object_tracking_info.num_outside_shrunken_image++;
      }
    }
  }

}


void FeatureTracker::propogateMask(ImageContainer& image_container) {
  if(!previous_frame_) return;


  const cv::Mat& previous_rgb = previous_frame_->image_container_.get<ImageType::RGBMono>();
  const cv::Mat& previous_mask = previous_frame_->image_container_.get<ImageType::MotionMask>();
  const cv::Mat& previous_flow = previous_frame_->image_container_.get<ImageType::OpticalFlow>();

  // note reference
  cv::Mat& current_mask = image_container.get<ImageType::MotionMask>();

  ObjectIds instance_labels;
  for(const Feature::Ptr& dynamic_feature : previous_frame_->usableDynamicFeaturesBegin()) {
    CHECK(dynamic_feature->objectId() != background_label);
    instance_labels.push_back(dynamic_feature->objectId());
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
      if (dynamic_feature->objectId() == instance_labels[j])
      {
        object_features[j].push_back(dynamic_feature->trackletId());
        CHECK(dynamic_feature->objectId() != background_label);
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
      //feature at k-1
      Feature::Ptr feature = previous_frame_->dynamic_features_.getByTrackletId(object_features[i][j]);
      CHECK(Feature::IsNotNull(feature));
      //kp at k
      const Keypoint& predicted_kp = feature->predictedKeypoint();
      const int u = functional_keypoint::u(predicted_kp);
      const int v = functional_keypoint::v(predicted_kp);
      // ensure u and v are sitll inside the CURRENT frame
      if (u < previous_rgb.cols && u > 0 && v < previous_rgb.rows && v > 0)
      {
        //add instance label at predicted keypoint
        temp_label.push_back(current_mask.at<ObjectId>(v, u));
      }
    }

    //this is a lovely magic number inherited from some old code :)
    if (temp_label.size() < 30)
    {
      LOG(WARNING) << "not enoug points to track object " << instance_labels[i] << " points size - "
                   << temp_label.size();
      //TODO:mark has static!!???
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
            const double flow_xe = static_cast<double>(previous_flow.at<cv::Vec2f>(j, k)[0]);
            const double flow_ye = static_cast<double>(previous_flow.at<cv::Vec2f>(j, k)[1]);

            if(flow_xe == 0 || flow_ye == 0) {
              continue;
            }

            OpticalFlow flow(flow_xe, flow_ye);
            //x, y
            Keypoint kp(k, j);
            const Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(kp, flow);

            if(!isWithinShrunkenImage(predicted_kp)) {
              continue;
            }

            if ((predicted_kp(0) < previous_rgb.cols && predicted_kp(0) > 0 && predicted_kp(1) < previous_rgb.rows && predicted_kp(1) > 0))
            {
              current_mask.at<ObjectId>(functional_keypoint::v(predicted_kp), functional_keypoint::u(predicted_kp)) = instance_labels[i];
              //  current_rgb
              // updated_mask_points++;
            }
          }
        }
      }
    }
  }
}

}
