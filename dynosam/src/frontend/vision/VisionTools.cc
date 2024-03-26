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


namespace vision_tools {



void getCorrespondences(FeaturePairs& correspondences, const FeatureFilterIterator& previous_features, const FeatureFilterIterator& current_features) {
  correspondences.clear();

  const FeatureContainer& previous_feature_container = previous_features.getContainer();

  for(const auto& curr_feature : current_features) {
    //check if previous feature and is valid
    if(previous_feature_container.exists(curr_feature->tracklet_id_)) {
      const auto prev_feature = previous_feature_container.getByTrackletId(curr_feature->tracklet_id_);
      CHECK(prev_feature);

      //having checked that feature is in the previous set, also check that it ahderes to the filter
      if(!previous_features(prev_feature)) {
        continue;
      }
      correspondences.push_back({prev_feature, curr_feature});
    }
  }
}


ObjectIds getObjectLabels(const cv::Mat& image) {
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



std::vector<std::vector<int> > trackDynamic(const FrontendParams& params, const Frame& previous_frame, Frame::Ptr current_frame) {
  auto& objects_by_instance_label = current_frame->object_observations_;

  auto& previous_dynamic_feature_container = previous_frame.dynamic_features_;
  auto& current_dynamic_feature_container = current_frame->dynamic_features_;

  ObjectIds instance_labels_to_remove;

  //TODO: shrink object boundary?
  for(auto& [instance_label, object_observation] : objects_by_instance_label) {
    double obj_center_depth = 0, sf_min=100, sf_max=0, sf_mean=0, sf_count=0;
    std::vector<int> sf_range(10,0);

    const size_t num_object_features = object_observation.object_features_.size();
    // LOG(INFO) << "tracking object observation with instance label " << instance_label << " and " << num_object_features << " features";

    for(const TrackletId tracklet_id : object_observation.object_features_) {
      if(previous_dynamic_feature_container.exists(tracklet_id)) {
        CHECK(current_dynamic_feature_container.exists(tracklet_id));

        Feature::Ptr current_feature = current_dynamic_feature_container.getByTrackletId(tracklet_id);
        Feature::Ptr previous_feature = previous_dynamic_feature_container.getByTrackletId(tracklet_id);

        if(!previous_feature->usable()) {
          current_feature->markInvalid();
          continue;
        }

        //this can happen in situations such as the updateDepths when depths > thresh are marked invalud
        if(!current_feature->usable()) { continue;}

        CHECK(!previous_feature->isStatic());
        CHECK(!current_feature->isStatic());

        Landmark lmk_previous = previous_frame.backProjectToWorld(tracklet_id);
        Landmark lmk_current = current_frame->backProjectToWorld(tracklet_id);

        Landmark flow_world = lmk_current - lmk_previous ;
        double sf_norm = flow_world.norm();

        if (sf_norm<params.scene_flow_magnitude)
            sf_count = sf_count+1;
        if(sf_norm<sf_min)
            sf_min = sf_norm;
        if(sf_norm>sf_max)
            sf_max = sf_norm;
        sf_mean = sf_mean + sf_norm;

        {
          if (0.0<=sf_norm && sf_norm<0.05)
              sf_range[0] = sf_range[0] + 1;
          else if (0.05<=sf_norm && sf_norm<0.1)
              sf_range[1] = sf_range[1] + 1;
          else if (0.1<=sf_norm && sf_norm<0.2)
              sf_range[2] = sf_range[2] + 1;
          else if (0.2<=sf_norm && sf_norm<0.4)
              sf_range[3] = sf_range[3] + 1;
          else if (0.4<=sf_norm && sf_norm<0.8)
              sf_range[4] = sf_range[4] + 1;
          else if (0.8<=sf_norm && sf_norm<1.6)
              sf_range[5] = sf_range[5] + 1;
          else if (1.6<=sf_norm && sf_norm<3.2)
              sf_range[6] = sf_range[6] + 1;
          else if (3.2<=sf_norm && sf_norm<6.4)
              sf_range[7] = sf_range[7] + 1;
          else if (6.4<=sf_norm && sf_norm<12.8)
              sf_range[8] = sf_range[8] + 1;
          else if (12.8<=sf_norm && sf_norm<25.6)
              sf_range[9] = sf_range[9] + 1;
        }

      }

    }


    if (sf_count/num_object_features>params.scene_flow_percentage || num_object_features < 150u)
    {
      // label this object as static background
      // LOG(INFO) << "Instance object " << instance_label << " to static for frame " << current_frame->frame_id_;
      instance_labels_to_remove.push_back(instance_label);
    }
    else {
      // LOG(INFO) << "Instance object " << instance_label << " marked as dynamic";
      object_observation.marked_as_moving_ = true;
    }
  }

  //we do the removal after the iteration so as not to mess up the loop
  for(const auto label : instance_labels_to_remove) {
    VLOG(30) << "Removing label " << label;
    //TODO: this is really really slow!!
    current_frame->moveObjectToStatic(label);
    // LOG(INFO) << "Done Removing label " << label;
  }


  // Relabel the objects that associate with the objects in last frame
  objects_by_instance_label = current_frame->object_observations_; //get iterator gain as the observations have changed

  // per object iteration
  for(auto& [instance_label, object_observation] : objects_by_instance_label) {
    CHECK(object_observation.marked_as_moving_) << "Object with instance label "
      << instance_label << " is not marked as moving, but should be";
    // LOG(INFO) << "Assigning tracking label for instance " << instance_label;

    const size_t num_object_features = object_observation.object_features_.size();
    // save semantic labels in last frame
    ObjectIds instance_labels_prev;
    //feature in each object iteration
    for(const TrackletId tracklet_id : object_observation.object_features_) {
      if(previous_dynamic_feature_container.exists(tracklet_id)) {
        CHECK(current_dynamic_feature_container.exists(tracklet_id));

        //TODO: inliers outliers!@!

        Feature::Ptr previous_feature = previous_dynamic_feature_container.getByTrackletId(tracklet_id);
        instance_labels_prev.push_back(previous_feature->instance_label_);
      }
    }

    const bool tracked_in_previous_frame = instance_labels_prev.size() > 0u;
    //since we only call this function after boostrapping (frame > 1) this condition only works when an object appears from frame 2 onwards
    if(!tracked_in_previous_frame) {
      // LOG(INFO) << "New object with instance label " << instance_label << " assigned tracking label " << Frame::global_object_id;
      current_frame->updateObjectTrackingLabel(object_observation, Frame::global_object_id);
      Frame::global_object_id++;
      continue;
     }

    // find label that appears most in instance_labels_prev()
    // (1) count duplicates
    std::map<int, int> dups;
    for(int k : instance_labels_prev) ++dups[k];
    // (2) and sort them by descending order
    std::vector<std::pair<int, int> > sorted;
    for (auto k : dups)
        sorted.push_back(std::make_pair(k.first,k.second));

    //copied from FeatureTracker.cc
    auto sort_pair_int = [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool {
      return (a.second > b.second);
    };
    std::sort(sorted.begin(), sorted.end(), sort_pair_int);

    // label the (instance) object in current frame
    ObjectId new_label = sorted[0].first;

    //TODO: make global_object_id access a function?
    if(Frame::global_object_id == 1) {
      current_frame->updateObjectTrackingLabel(object_observation, Frame::global_object_id);
      Frame::global_object_id++;
    }
    else {
      //here we need to propogate the tracking label (if exists) from the previous frame
      //this means we expect all the features in the previous frame to have the same tracking label
      //(can this even not happen?)

      //find previous observation to see if one has the same instance label
      auto& previous_object_observations = previous_frame.object_observations_;
      auto it = std::find_if(
        previous_object_observations.begin(),
        previous_object_observations.end(),
        /*pair = <ObjectId, DynamicObjectObservation>*/
        [&](const auto& pair) { return pair.second.instance_label_ == new_label; });

      //object tracking label was in the previous frame
      //tracking label has not been updated from previous frame? will still be -1?
      if(it != previous_object_observations.end() && it->second.tracking_label_ != -1) {
        ObjectId propogated_tracking_label = it->second.tracking_label_;
        // LOG(INFO) << "Propogating tracking label " << propogated_tracking_label << " from frames " << previous_frame.frame_id_ << " -> " << current_frame->frame_id_;
        current_frame->updateObjectTrackingLabel(object_observation, propogated_tracking_label);
      }
      else {
        // new object (or at least not tracked)
        // LOG(INFO) << "New object with instance label " << instance_label << " assigned tracking label " << Frame::global_object_id;

        current_frame->updateObjectTrackingLabel(object_observation, Frame::global_object_id);
        Frame::global_object_id++;
      }

    }

    VLOG(20) << "Done tracking for instance label " << instance_label;
  }
  return std::vector<std::vector<int> >();
}


bool findObjectBoundingBox(const cv::Mat& mask, ObjectId object_id, cv::Rect& rect) {
  cv::Mat obj_mask = (mask == object_id);
  cv::Mat dilated_obj_mask;
  cv::Mat dilate_element = cv::getStructuringElement(cv::MORPH_RECT,
                                                      cv::Size(1, 11)); // a rectangle of 1*5
  cv::dilate(obj_mask, dilated_obj_mask, dilate_element, cv::Point(0, 10)); // defining anchor point so it only erode down

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(dilated_obj_mask, contours,hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

  if(contours.empty()) {
      rect = cv::Rect();
      return false;
  }
  else if(contours.size() == 1u) {
      rect = cv::boundingRect(contours.at(0));
  }
  else {
      std::vector<cv::Rect> rectangles;
      for(auto it : contours) {
          rectangles.push_back(cv::boundingRect(it));
      }
      cv::Rect merged_rect = rectangles[0];
      for(const auto& r : rectangles) { merged_rect |= r; }
      rect = merged_rect;
  }
  return true;
}

} //vision_tools

// void RGBDProcessor::updateMovingObjects(const Frame& previous_frame, Frame::Ptr current_frame,  cv::Mat& debug) const {
//   const cv::Mat& rgb = current_frame->tracking_images_.get<ImageType::RGBMono>();

//   rgb.copyTo(debug);

//   const gtsam::Pose3& previous_pose = previous_frame.T_world_camera_;
//   const gtsam::Pose3& current_pose = current_frame->T_world_camera_;

//   const auto previous_dynamic_feature_container = previous_frame.dynamic_features_;
//   const auto current_dynamic_feature_container = current_frame->dynamic_features_;

//   //iterate over each object seen in the previous frame and collect features in current and previous frames to determine scene flow
//   for(auto& [object_id, current_object_observation] : current_frame->object_observations_) {

//     int object_track_count = 0; //number of tracked points on the object
//     int sf_count = 0; //number of points on the object with a sufficient scene flow thresh

//     const TrackletIds& object_features = current_object_observation.object_features_;
//     for(const auto tracklet_id : object_features) {
//       if(previous_dynamic_feature_container.exists(tracklet_id)) {
//         CHECK(current_dynamic_feature_container.exists(tracklet_id));

//         Feature::Ptr current_feature = current_dynamic_feature_container.getByTrackletId(tracklet_id);
//         Feature::Ptr previous_feature = previous_dynamic_feature_container.getByTrackletId(tracklet_id);

//         if(!previous_feature->usable()) {
//           current_feature->markInvalid();
//           continue;
//         }

//         Landmark lmk_previous, lmk_current;
//         camera_->backProject(previous_feature->keypoint_, previous_feature->depth_, &lmk_previous, previous_pose);
//         camera_->backProject(current_feature->keypoint_, current_feature->depth_, &lmk_current, current_pose);

//         Landmark flow_world = lmk_previous - lmk_current;
//         double sf_norm = flow_world.norm();

//         if(sf_norm > params_.scene_flow_magnitude) {
//           sf_count++;
//         }

//         object_track_count++;
//       }
//     }

//     if(sf_count < 50) {
//       continue;
//     }
//     double average_flow_count = (double)sf_count / (double)object_track_count;

//     LOG(INFO) << "Num points that are dynamic " << average_flow_count << "/" << params_.scene_flow_percentage << " for object " << object_id;
//     if(average_flow_count > params_.scene_flow_percentage) {
//       current_object_observation.marked_as_moving_ = true;

//       static const cv::Scalar blue(255, 0, 0);

//       for(TrackletId track : object_features) {
//         Feature::Ptr current_feature = current_dynamic_feature_container.getByTrackletId(track);
//         const Keypoint& px = current_feature->keypoint_;
//         cv::circle(debug, utils::gtsamPointToCV(px), 6, blue, 1);
//       }


//       //only debug stuff

//     }

//   }

// }





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
