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

#include "dynosam/frontend/vision/Frame.hpp"

#include <tbb/parallel_for.h>
#include <glog/logging.h>

#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_common/viz/Colour.hpp"

namespace dyno {

Frame::Frame(
    FrameId frame_id, Timestamp timestamp, Camera::Ptr camera,
    const ImageContainer& image_container,
    const FeatureContainer& static_features,
    const FeatureContainer& dynamic_features,
    const std::vector<Edge>& static_edges,
    const std::map<ObjectId, SingleDetectionResult>& object_observations,
    std::optional<FeatureTrackerInfo> tracking_info)
    : frame_id_(frame_id),
      timestamp_(timestamp),
      camera_(camera),
      image_container_(image_container.clone()),  // Deep copy for thread safety
      static_features_(static_features),
      dynamic_features_(dynamic_features),
      static_edges_(static_edges),
      object_observations_(object_observations),
      tracking_info_(tracking_info) {
  // NOTE: no rectification, use camera matrix as P for cv::undistortPoints
  // see
  // https://stackoverflow.com/questions/22027419/bad-results-when-undistorting-points-using-opencv-in-python
  // i mean, this could just be shared between frames?
  const CameraParams& cam_params = camera->getParams();
  cv::Mat P = cam_params.getCameraMatrix();
  cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
  undistorter_ = std::make_shared<UndistorterRectifier>(P, cam_params, R);

  // Cache image dimensions and camera parameters for convenience
  const cv::Size img_size = cam_params.imageSize();
  img_width_ = img_size.width;
  img_height_ = img_size.height;
  cam_fx_ = cam_params.fx();
  cam_fy_ = cam_params.fy();
  cam_cx_ = cam_params.cu();
  cam_cy_ = cam_params.cv();


  //-- Depth-related preprocessing: remove inconsistent edge features
  // Validate image_container_ and static_edges_ before processing
  if (!image_container_.hasDepth()) {
    LOG(WARNING) << "image_container_ has no depth, skipping edge depth processing";
    // Still need to initialize edge_point_lookup_map_ even without depth
    if (!static_edges_.empty()) {
      assignPropertyIdx();
      constructSearchPlainParallel();
    }
  } else if (static_edges_.empty()) {
    LOG(WARNING) << "static_edges_ is empty, skipping edge processing";
  } else {
    try {
      assignProperty3D(image_container_.depth()); //-- Assign depth to edges
      // edgeCullingDepth();         //-- Remove all edges with invalid depth and invalid edge points within valid edges
      edgeCullingDepthParallel();
      //-- Now all edge points in the remaining edges have valid depth
      edgeCullingContinuity();    //-- Ensure 3D point depth continuity for each ordered edge

      //-- Update frame_edge_ID and frame_point_index for each edge point in this frame
      assignPropertyIdx();
      
      //constructSearchPlain();
      constructSearchPlainParallel();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Exception in edge processing: " << e.what();
      // Try to at least initialize edge_point_lookup_map_ even if processing failed
      if (!static_edges_.empty()) {
        try {
          assignPropertyIdx();
          constructSearchPlainParallel();
        } catch (...) {
          LOG(ERROR) << "Failed to initialize edge_point_lookup_map_";
        }
      }
    } catch (...) {
      LOG(ERROR) << "Unknown exception in edge processing";
      // Try to at least initialize edge_point_lookup_map_ even if processing failed
      if (!static_edges_.empty()) {
        try {
          assignPropertyIdx();
          constructSearchPlainParallel();
        } catch (...) {
          LOG(ERROR) << "Failed to initialize edge_point_lookup_map_";
        }
      }
    }
  }
  
}

Frame::Frame(FrameId frame_id, Timestamp timestamp, Camera::Ptr camera,
             const ImageContainer& image_container,
             const FeatureContainer& static_features,
             const FeatureContainer& dynamic_features,
             const std::vector<Edge>& static_edges,
             std::optional<FeatureTrackerInfo> tracking_info)
    : Frame(frame_id, timestamp, camera, image_container, static_features,
            dynamic_features, static_edges, {}, tracking_info) {
  // dynamic info is not pre-calculated so calculate it here! SLOW!
  constructDynamicObservations();
}

bool Frame::exists(TrackletId tracklet_id) const {
  const bool result = static_features_.exists(tracklet_id) ||
                      dynamic_features_.exists(tracklet_id);

  // debug checking -> should only be in one feature container
  if (result) {
    CHECK(!(static_features_.exists(tracklet_id) &&
            dynamic_features_.exists(tracklet_id)))
        << "Tracklet Id " << tracklet_id
        << " exists in both static and dynamic feature sets. Should be unique!";
  }
  return result;
}

Feature::Ptr Frame::at(TrackletId tracklet_id) const {
  if (!exists(tracklet_id)) {
    return nullptr;
  }

  if (static_features_.exists(tracklet_id)) {
    CHECK(!dynamic_features_.exists(tracklet_id));
    return static_features_.getByTrackletId(tracklet_id);
  } else {
    CHECK(dynamic_features_.exists(tracklet_id));
    return dynamic_features_.getByTrackletId(tracklet_id);
  }
}

bool Frame::isFeatureUsable(TrackletId tracklet_id) const {
  const auto& feature = at(tracklet_id);
  if (!feature) {
    throw std::runtime_error(
        "Failed to check feature usability - tracklet id " +
        std::to_string(tracklet_id) + " does not exist");
  }

  return feature->usable();
}

FeaturePtrs Frame::collectFeatures(TrackletIds tracklet_ids) const {
  FeaturePtrs features;
  for (const auto tracklet_id : tracklet_ids) {
    Feature::Ptr feature = at(tracklet_id);
    if (!feature) {
      throw std::runtime_error("Failed to collectFeatures - tracklet id " +
                               std::to_string(tracklet_id) + " does not exist");
    }

    features.push_back(feature);
  }

  return features;
}

Landmark Frame::backProjectToCamera(TrackletId tracklet_id) const {
  Feature::Ptr feature = at(tracklet_id);
  if (!feature) {
    throw std::runtime_error("Failed to backProjectToCamera - tracklet id " +
                             std::to_string(tracklet_id) + " does not exist");
  }

  // if no depth, project to unitsphere?
  CHECK(feature->hasDepth());

  // Landmark lmk;
  return getLandmarkFromCache(landmark_in_camera_cache_, feature,
                              gtsam::Pose3::Identity());
  // return lmk;
}

Landmark Frame::backProjectToWorld(TrackletId tracklet_id) const {
  Feature::Ptr feature = at(tracklet_id);
  if (!feature) {
    throw std::runtime_error("Failed to backProjectToWorld - tracklet id " +
                             std::to_string(tracklet_id) + " does not exist");
  }

  // if no depth, project to unitsphere?
  CHECK(feature->hasDepth());

  // Landmark lmk;
  // camera_->backProject(feature->keypoint_, feature->depth_, &lmk,
  // T_world_camera_);
  return getLandmarkFromCache(landmark_in_world_cache_, feature,
                              T_world_camera_);
}

Camera::CameraImpl Frame::getFrameCamera() const {
  const CameraParams& camera_params = camera_->getParams();
  return Camera::CameraImpl(
      T_world_camera_,
      camera_params.constructGtsamCalibration<Camera::CalibrationType>());
}

PointCloudLabelRGB::Ptr Frame::projectToDenseCloud(
    const cv::Mat* detection_mask) const {
  if (!image_container_.hasDepth()) {
    return nullptr;
  }

  PointCloudLabelRGB::Ptr cloud = pcl::make_shared<PointCloudLabelRGB>();
  const cv::Mat& depth_image = image_container_.depth();
  const cv::Mat& motion_mask = image_container_.objectMotionMask();

  if (detection_mask) {
    CHECK(utils::cvSizeEqual(detection_mask->size(), depth_image.size()));
  }

  const int rows = depth_image.rows;
  const int cols = depth_image.cols;

  // Reserve memory for the cloud (approximate max size)
  // cloud->points.reserve(rows * cols);
  cloud->points.resize(rows * cols);

  // api needs image ref
  // in this function we will not actually change the depth image
  cv::Mat& depth_image_ref = const_cast<cv::Mat&>(depth_image);
  FunctionalParallelOpenCVMat process(
      depth_image_ref, [&](cv::Mat& depth_image, int i, int j) {
        const unsigned char* detection_ptr =
            detection_mask ? detection_mask->ptr<unsigned char>(i) : nullptr;
        const ObjectId* motion_mask_ptr = motion_mask.ptr<ObjectId>(i);
        const Depth* depth_ptr = depth_image.ptr<Depth>(i);

        if (detection_mask && detection_ptr[j] == 0) return;

        const ObjectId object_id = motion_mask_ptr[j];
        const Depth depth = depth_ptr[j];

        double depth_thresh;
        Color colour;
        if (object_id == background_label) {
          depth_thresh = max_background_threshold_;
          colour = Color::black();
        } else {
          depth_thresh = max_object_threshold_;
          colour = Color::uniqueId(object_id);
        }

        if (depth > depth_thresh || depth <= 0 || !std::isfinite(depth)) return;

        // Back-projection
        const Keypoint kp(j, i);
        Landmark point;
        // this call is probably very slow
        camera_->backProject(kp, depth, &point);

        cloud->points[i * cols + j] = PointLabelRGB(
            static_cast<float>(point(0)), static_cast<float>(point(1)),
            static_cast<float>(point(2)), static_cast<std::uint8_t>(colour.r),
            static_cast<std::uint8_t>(colour.g),
            static_cast<std::uint8_t>(colour.b),
            static_cast<std::uint32_t>(object_id));
      });
  process.run();

  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = false;
  return cloud;
}

bool Frame::updateDepths() {
  if (!image_container_.hasDepth()) {
    return false;
  }

  const ImageWrapper<ImageType::Depth>& depth = image_container_.depth();
  updateDepthsFeatureContainer(static_features_, depth,
                               max_background_threshold_);
  updateDepthsFeatureContainer(dynamic_features_, depth, max_object_threshold_);

  // Update depths for edge features
  const cv::Mat& matDepth = depth;
  tbb::parallel_for(0, (int)static_edges_.size(), [&](int i) {
    // Inner loop remains serial
    for(int j = 0; j < static_edges_[i].mvPoints.size(); ++j) {
        assignProperty3DEach(static_edges_[i].mvPoints[j], matDepth);
    }
  });

  return true;
}

Frame& Frame::setMaxBackgroundDepth(double thresh) {
  CHECK_GT(thresh, 0);
  max_background_threshold_ = thresh;
  return *this;
}

Frame& Frame::setMaxObjectDepth(double thresh) {
  CHECK_GT(thresh, 0);
  max_object_threshold_ = thresh;
  return *this;
}

bool Frame::getCorrespondences(FeaturePairs& correspondences,
                               const Frame& previous_frame,
                               KeyPointType kp_type) const {
  if (kp_type == KeyPointType::STATIC) {
    return getStaticCorrespondences(correspondences, previous_frame);
  } else {
    return getDynamicCorrespondences(correspondences, previous_frame);
  }
}

Frame::ConstructCorrespondanceFunc<Landmark, Keypoint>
Frame::landmarkWorldKeypointCorrespondance() const {
  auto func = [&](const Frame& previous_frame,
                  const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    if (!previous_feature->hasDepth()) {
      throw std::runtime_error(
          "Error in constructing Landmark (w) -> keypoint correspondences - "
          "previous feature does not have depth!");
    }

    // eventuall map?
    Landmark lmk_w =
        previous_frame.backProjectToWorld(previous_feature->trackletId());
    return TrackletCorrespondance(previous_feature->trackletId(), lmk_w,
                                  current_feature->keypoint());
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

Frame::ConstructCorrespondanceFunc<Keypoint, Keypoint>
Frame::imageKeypointCorrespondance() const {
  auto func = [&](const Frame&, const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    return TrackletCorrespondance(previous_feature->trackletId(),
                                  previous_feature->keypoint(),
                                  current_feature->keypoint());
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

Frame::ConstructCorrespondanceFunc<Landmark, gtsam::Vector3>
Frame::landmarkWorldProjectedBearingCorrespondance() const {
  auto func = [&](const Frame& previous_frame,
                  const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    if (!previous_feature->hasDepth()) {
      throw std::runtime_error(
          "Error in constructing Landmark (w) -> keypoint correspondences - "
          "previous feature does not have depth!");
    }

    // eventuall map?
    Landmark lmk_w =
        previous_frame.backProjectToWorld(previous_feature->trackletId());

    // TODO: what if image is already undistorted
    gtsam::Vector3 projected_versor =
        undistorter_->undistortKeypointAndGetProjectedVersor(
            current_feature->keypoint());
    return TrackletCorrespondance(previous_feature->trackletId(), lmk_w,
                                  projected_versor);
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

Frame::ConstructCorrespondanceFunc<Landmark, Landmark>
Frame::landmarkWorldPointCloudCorrespondance() const {
  auto func = [&](const Frame& previous_frame,
                  const Feature::Ptr& previous_feature,
                  const Feature::Ptr& current_feature) {
    if (!previous_feature->hasDepth()) {
      throw std::runtime_error(
          "Error in constructing Landmark (w) -> keypoint correspondences - "
          "previous feature does not have depth!");
    }

    // eventuall map?
    Landmark lmk_w_k_1 =
        previous_frame.backProjectToWorld(previous_feature->trackletId());
    // eventuall map?
    Landmark lmk_w_k = backProjectToWorld(current_feature->trackletId());

    return TrackletCorrespondance(previous_feature->trackletId(), lmk_w_k_1,
                                  lmk_w_k);
  };

  return std::bind(func, std::placeholders::_1, std::placeholders::_2,
                   std::placeholders::_3);
}

bool Frame::getDynamicCorrespondences(FeaturePairs& correspondences,
                                      const Frame& previous_frame,
                                      ObjectId object_id) const {
  if (object_observations_.find(object_id) == object_observations_.end()) {
    LOG(WARNING) << "Object object instance id " << object_id
                 << " not found for frame " << frame_id_;
    return false;
  }

  // const SingleDetectionResult& observation =
  // object_observations_.at(object_id);
  // // TODO: need to put back on - if we have motion mask, we should just mark
  // all objects as moving CHECK(observation.marked_as_moving_); const
  // TrackletIds& tracklets = observation.object_features_;

  // FeatureContainer feature_container;
  // for(const TrackletId tracklet : tracklets) {
  //     if(isFeatureUsable(tracklet)) {
  //         feature_container.add(this->at(tracklet));
  //     }
  // }

  auto current_dynamic_features_iterator = FeatureFilterIterator(
      const_cast<FeatureContainer&>(this->dynamic_features_),
      [object_id](const Feature::Ptr& f) -> bool {
        return Feature::IsUsable(f) && f->objectId() == object_id;
      });

  // make iterator for the previous dynamic features that ensure each feature is
  // usable and has a matching instance label
  auto previous_dynamic_features_iterator = FeatureFilterIterator(
      const_cast<FeatureContainer&>(previous_frame.dynamic_features_),
      [object_id](const Feature::Ptr& f) -> bool {
        return Feature::IsUsable(f) && f->objectId() == object_id;
      });

  // get the correspondences from these two iterators
  // we iterate over the current feature container which should only contain
  // features on the object and compare against the container
  vision_tools::getCorrespondences(
      correspondences, previous_dynamic_features_iterator,
      // we iterate over the current feature container which should only contain
      // features on the object
      current_dynamic_features_iterator);

  // LOG(INFO) << "Found " << correspondences.size() << " correspondences for
  // object instance " << object_id << " " << (correspondences.size() > 0u);

  return correspondences.size() > 0u;
}

bool Frame::getStaticCorrespondences(FeaturePairs& correspondences,
                                     const Frame& previous_frame) const {
  vision_tools::getCorrespondences(
      correspondences, previous_frame.static_features_.beginUsable(),
      static_features_.beginUsable());

  return correspondences.size() > 0u;
}

bool Frame::getDynamicCorrespondences(FeaturePairs& correspondences,
                                      const Frame& previous_frame) const {
  vision_tools::getCorrespondences(
      correspondences, previous_frame.dynamic_features_.beginUsable(),
      dynamic_features_.beginUsable());
  return correspondences.size() > 0u;
}

void Frame::updateDepthsFeatureContainer(
    FeatureContainer& container, const ImageWrapper<ImageType::Depth>& depth,
    double max_depth) {
  // auto iter = container.beginUsable();
  // auto iter = container.begin();

  int count = 0;

  // iterate over all features
  for (Feature::Ptr feature : container) {
    // CHECK(feature->usable());
    // const Feature::Ptr& feature = *iter;
    // const int x = functional_keypoint::u(feature->keypoint_);
    // const int y = functional_keypoint::v(feature->keypoint_);
    // const Depth d = depth_mat.at<Depth>(y, x);
    const Depth d = functional_keypoint::at<Depth>(feature->keypoint(), depth);

    if (d > max_depth || d <= 0) {
      feature->markInvalid();
      feature->depth(Feature::invalid_depth);
      count++;
    } else {
      feature->depth(d);
    }

    //  //if now invalid or happens to be invalid from a previous frame, make
    //  depth invalid too
    // if(!feature->usable()) {
    //     feature->depth_ = Feature::invalid_depth;
    // }
    // else {
    //     feature->depth_ = d;
    // }
  }

  // LOG(INFO) << count << " features marked invalud due to depth out of " <<
  // container.size() << " with max depth " << max_depth;
}

void Frame::constructDynamicObservations() {
  object_observations_.clear();
  
  // Early return if no dynamic features
  if (dynamic_features_.empty()) {
    return;
  }
  
  // assumes that the mask gets updated with the tracking label
  const ObjectIds instance_labels =
      vision_tools::getObjectLabels(image_container_.objectMotionMask());

  auto inlier_iterator = dynamic_features_.beginUsable();
  for (const Feature::Ptr& dynamic_feature : inlier_iterator) {
    CHECK(!dynamic_feature->isStatic());
    CHECK(dynamic_feature->usable());

    const ObjectId object_id = dynamic_feature->objectId();
    // this check is just for sanity!
    CHECK(std::find(instance_labels.begin(), instance_labels.end(),
                    object_id) != instance_labels.end())
        << "Missing " << object_id << " in "
        << container_to_string(instance_labels);

    if (object_observations_.find(object_id) == object_observations_.end()) {
      SingleDetectionResult observation;
      observation.object_id = object_id;
      object_observations_[object_id] = observation;
    }

    // object_observations_[object_id].object_features.push_back(
    //     dynamic_feature->trackletId());
  }

  // Early return if no object observations were created
  if (object_observations_.empty()) {
    return;
  }

  // now construct image masks from tracking mask
  // For each tracked object, find its id in the mask
  // and draw it.
  // We apply some eroding/dilation on it to make the resulting submask smoother
  // so that we can more easily fit an rectangle to it
  const cv::Mat& mask = image_container_.objectMotionMask();
  for (auto& object_observation_pair : object_observations_) {
    const ObjectId object_id = object_observation_pair.first;
    SingleDetectionResult& obs = object_observation_pair.second;
    // findObjectBoundingBox may fail if object_id is not in mask, handle gracefully
    if (!vision_tools::findObjectBoundingBox(mask, object_id, obs.bounding_box)) {
      // If bounding box not found, set empty rect and log warning
      obs.bounding_box = cv::Rect();
      VLOG(10) << "Failed to find bounding box for object_id=" << object_id 
               << " in mask (object may have been removed)";
    }
  }
}


// Edge feature assignment
// void Frame::assignProperty3D(const cv::Mat& matDepth)
// {
//     // Parallel process outer loop
//     tbb::parallel_for(0, (int)static_features_.size(), [&](int i) {
//         // Inner loop remains serial
//         for(int j = 0; j < static_features_[i].mvPoints.size(); ++j) {
//             assignProperty3DEach(static_features_[i].mvPoints[j], matDepth);
//         }
//     });

//     // for(int i = 0; i < static_features_.size(); ++i)
//     // {
//     //     for(int j = 0; j < static_features_[i].mvPoints.size(); ++j)
//     //     {
//     //         assignProperty3DEach(static_features_[i].mvPoints[j], matDepth);
//     //     }
//     // }
// }

// assignProperty3DEach implementation moved to line 1103 to avoid duplication


// void Frame::moveObjectToStatic(ObjectId instance_label) {
//   auto it = object_observations_.find(instance_label);
//   CHECK(it != object_observations_.end());

//   SingleDetectionResult& observation = it->second;
//   observation.marked_as_moving_ = false;
//   CHECK(observation.instance_label_ == instance_label);
//   // go through all features, move them to from dynamic structure and add
//   them
//   // to static
//   for (TrackletId tracklet_id : observation.object_features_) {
//     CHECK(dynamic_features_.exists(tracklet_id));
//     Feature::Ptr dynamic_feature =
//         dynamic_features_.getByTrackletId(tracklet_id);

//     if (!dynamic_feature->usable()) {
//       continue;
//     }

//     CHECK(!dynamic_feature->isStatic());
//     CHECK_EQ(dynamic_feature->trackletId(), tracklet_id);
//     CHECK_EQ(dynamic_feature->objectId(), instance_label);
//     dynamic_feature->keypointType(KeyPointType::STATIC);
//     dynamic_feature->objectId(background_label);
//     // dynamic_feature->tracking_label_ = background_label;

//     dynamic_features_.remove(tracklet_id);
//     // Jesse: no, do not move points (these are dense) to static - instrad we
//     // need to mark the AREA around the object as static and then retrack all
//     // points in there!!
//     //  static_features_.add(dynamic_feature);
//   }

//   object_observations_.erase(it);
// }

// void Frame::updateObjectTrackingLabel(
//     const SingleDetectionResult& observation, ObjectId new_tracking_label)
//     {
//   auto it = object_observations_.find(observation.instance_label_);
//   CHECK(it != object_observations_.end());

//   auto& obs = it->second;
//   obs.tracking_label_ = new_tracking_label;
//   // update all features
//   for (TrackletId tracklet_id : obs.object_features_) {
//     Feature::Ptr feature = dynamic_features_.getByTrackletId(tracklet_id);
//     CHECK(feature);
//     feature->objectId(new_tracking_label);
//   }
// }

FeatureFilterIterator Frame::usableStaticFeaturesBegin() {
  return static_features_.beginUsable();
}

FeatureFilterIterator Frame::usableStaticFeaturesBegin() const {
  return static_features_.beginUsable();
}

FeatureFilterIterator Frame::usableDynamicFeaturesBegin() {
  return dynamic_features_.beginUsable();
}

FeatureFilterIterator Frame::usableDynamicFeaturesBegin() const {
  return dynamic_features_.beginUsable();
}

Landmark Frame::getLandmarkFromCache(LandmarkMap& /* cache */, Feature::Ptr feature,
                                     const gtsam::Pose3& X_world) const {
  // TODO: dont cache as we now update the optical flow and the depth in the
  // frontend and cacheing it will not use the right values!!!
  //  const auto& it = cache.find(feature->tracklet_id_);
  //  if(it != cache.end()) {
  //      return it->second;
  //  }

  Landmark lmk;
  camera_->backProject(feature->keypoint(), feature->depth(), &lmk, X_world);
  // cache.insert({feature->tracklet_id_, lmk});
  return lmk;
}

// Frame::FeatureFilterIterator Frame::dynamicUsableBegin() {
//     return FeatureFilterIterator(dynamic_features_, [&](const Feature::Ptr&
//     f) -> bool
//         {
//             return f->usable();
//         }
//     );
// }

void Frame::searchRadius(float x, float y, double radius, std::vector<orderedEdgePoint>& result)
{
    result.clear();

    // Validate edge_point_lookup_map_ before accessing
    if (edge_point_lookup_map_.empty()) {
        LOG(WARNING) << "edge_point_lookup_map_ is empty in searchRadius";
        return;
    }

    //-- 定义搜索区域的矩形边界（整数像素坐标）
    int minX = static_cast<int>(std::max(0.0, x - radius));
    int maxX = static_cast<int>(std::min(edge_point_lookup_map_.cols - 1.0, x + radius));
    int minY = static_cast<int>(std::max(0.0, y - radius));
    int maxY = static_cast<int>(std::min(edge_point_lookup_map_.rows - 1.0, y + radius));

    //-- 搜索到的点距 (x,y) 的距离
    std::vector<float> list_distance;

    //-- 遍历搜索区域内的所有像素, 寻找半径内的非（-1，-1）的边缘点
    for(int py = minY; py <= maxY; ++py)
    {
        for(int px = minX; px <= maxX; ++px)
        {
            const cv::Vec2i& pixel = edge_point_lookup_map_.at<cv::Vec2i>(py, px);
            int edgeID = pixel[0];     //-- frame_edge_ID
            int pointIdx = pixel[1];   //-- frame_point_index

            //-- 跳过无效点
            if (edgeID == -1 || pointIdx == -1) continue;

            //-- 计算距离（欧几里得距离）
            float dx = px - x;
            float dy = py - y;
            float distance = std::sqrt(dx * dx + dy * dy);

            //-- 如果距离在半径内，添加到结果
            if (distance <= radius)
            {
                //-- 获取原始点数据
                // Validate edge_id_to_index_map_ before accessing
                auto it = edge_id_to_index_map_.find(edgeID);
                if (it == edge_id_to_index_map_.end()) {
                    continue;  // Skip if edgeID not found in map
                }
                const auto& edge = static_edges_[it->second];
                orderedEdgePoint point = edge.mvPoints[pointIdx];

                //-- 确保原始点数据的frame_edge_ID 和 frame_point_index 确实是搜到的结果
                assert(point.frame_edge_ID == edgeID && point.frame_point_index == pointIdx);

                result.push_back(point);
                float distance = sqrt((point.x-x)*(point.x-x) + (point.y-y)*(point.y-y));
                list_distance.push_back(distance);
            }
        }
    }

    assert(list_distance.size() == result.size());

    // 创建索引数组
    std::vector<size_t> indices(result.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 根据相邻点相对(x,y)的距离对索引数组排序
    std::sort(indices.begin(), indices.end(), 
              [&list_distance](size_t i, size_t j) { return list_distance[i] < list_distance[j]; });

    // 根据排序后的索引重新排列 result
    std::vector<orderedEdgePoint> sorted_result;
    sorted_result.reserve(result.size()); 
    for (size_t i : indices) {
        sorted_result.push_back(result[i]);
    }
    result = std::move(sorted_result);
}

bool isPointsAssociated(const orderedEdgePoint& pt1, const orderedEdgePoint& pt2)
{
    float res = fabs(pt1.imgGradAngle-pt2.imgGradAngle);
    if(res > 180) res = 360 - res;
    //-- 梯度方向一致性关联
    if(res<10.0){
        return true;
    }else{
        return false;
    }
}

std::vector<int> Frame::edgeWiseCorrespondenceReproject(Edge& query_edge, const Sophus::SE3d& T2curr)
{
    // Validate edge_point_lookup_map_ before processing
    if (edge_point_lookup_map_.empty()) {
        LOG(WARNING) << "edge_point_lookup_map_ is empty in edgeWiseCorrespondenceReproject, cannot process";
        return std::vector<int>();
    }

    // * STEP 1. 得到参考帧边缘重投影到当前帧的坐标
    std::vector<orderedEdgePoint>& queryList = query_edge.mvPoints;
    std::vector<cv::Point> warped_queryList;
    const size_t num_points = queryList.size();
    warped_queryList.reserve(num_points);

    // 使用缓存的相机内参
    const double fx = cam_fx_;
    const double fy = cam_fy_;
    const double cx = cam_cx_;  // cu is the principal point x
    const double cy = cam_cy_;  // cv is the principal point y
    const float inv_fx = 1.0f / static_cast<float>(fx);
    const float inv_fy = 1.0f / static_cast<float>(fy);

    for(size_t i = 0; i < num_points; ++i)
    {
        //-- 由像素与深度值恢复的3D点
        const auto& pt = queryList[i];
        float z = pt.depth;
        float x = (static_cast<float>(pt.x) - static_cast<float>(cx)) * inv_fx * z;
        float y = (static_cast<float>(pt.y) - static_cast<float>(cy)) * inv_fy * z;
        
        //-- 重投影得到新的投影点
        Eigen::Vector3d point = T2curr * Eigen::Vector3d(x, y, z);
        warped_queryList.emplace_back(
            static_cast<int>(fx * point.x() / point.z() + cx),
            static_cast<int>(fy * point.y() / point.z() + cy)
        );
    }

    // * STEP 2. 半径邻域搜索，并投票得到 query edge 的每个点最想关联的边缘

    //-- first:当前帧的边的ID   second: 该条当前帧边有几个query edge的点意愿关联
    std::map<int, int> edgeVoteMapTotal;
    const float radius = 6.0f;
    const int threshold_value = std::min(static_cast<int>(num_points * 0.3f), 5);

    for(size_t i = 0; i < num_points; ++i)
    {
        orderedEdgePoint& pt = queryList[i];
        float x = warped_queryList[i].x;
        float y = warped_queryList[i].y;
        std::vector<orderedEdgePoint> neighbors_points;
        searchRadius(x, y, radius, neighbors_points);

        //-- 预存该点的近邻匹配关系
        pt.mvAssoFrameEdgeIDs.clear();
        pt.mvAssoFramePointIndices.clear();
        pt.mvAssoFrameEdgeIDs.reserve(neighbors_points.size());
        pt.mvAssoFramePointIndices.reserve(neighbors_points.size());

        //-- 对于一个点，建立一个投票，得到这个点最倾向关联的边
        std::unordered_map<int, int> edgeVoteMap;
        for (const auto& neighbor : neighbors_points) 
        {
            if (isPointsAssociated(pt, neighbor)) 
            {
                //-- 直接递增，避免find检查
                edgeVoteMap[neighbor.frame_edge_ID]++;
                //-- 确认可以关联后，更新关联的缓存
                pt.mvAssoFrameEdgeIDs.push_back(neighbor.frame_edge_ID);
                pt.mvAssoFramePointIndices.push_back(neighbor.frame_point_index);
            }
        }

        if (!edgeVoteMap.empty()) 
        {
            // 找出票数最多的边，此时max_pair.first 就是当前 query point 最想关联的边缘
            const auto max_pair = *std::max_element(
                edgeVoteMap.begin(), edgeVoteMap.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; }
            );
            // 每个点只有一个最想关联的边缘
            edgeVoteMapTotal[max_pair.first] += 1;
        }
    }

    // * STEP 3. 整理投票，确认当前query edge 能与哪些 current edges 关联
    //-- 现在得到的edgeVoteMapTotal包含了query edge与 candidate edge关联的投票关系
    
    std::vector<int> result;
    result.reserve(edgeVoteMapTotal.size());  // 预分配内存
    
    //-- 找出满足阈值要求的当前帧可关联边缘
    int max_votes = 0;
    for (const auto& [edge_id, votes] : edgeVoteMapTotal) 
    {
        if(votes > max_votes) max_votes = votes;
        if(votes > threshold_value) result.push_back(edge_id);
    }

    if (result.empty()) {
        return result;
    }

    // * STEP 4: 更新关联关系（使用哈希表加速查找）
    const std::unordered_set<int> validAssociation(result.begin(), result.end());
    for(auto& pt : query_edge.mvPoints) 
    {
        // 直接修改原数据，避免拷贝
        for(size_t j = 0; j < pt.mvAssoFrameEdgeIDs.size(); ++j) 
        {
            if(validAssociation.count(pt.mvAssoFrameEdgeIDs[j])) 
            {
                pt.asso_edge_ID = pt.mvAssoFrameEdgeIDs[j];
                pt.asso_point_index = pt.mvAssoFramePointIndices[j];
                pt.mbAssociated = true;
                break;
            }
        }
        // 清空内存（使用swap确保内存释放）
        std::vector<int>().swap(pt.mvAssoFrameEdgeIDs);
        std::vector<int>().swap(pt.mvAssoFramePointIndices);
    }
    
    return result;
}



void Frame::assignPropertyIdx()
{
    //-- 根据edges的ID构造ID与索引的映射
    for(size_t i = 0; i < static_edges_.size(); ++i)
    {
        //-- 更新edge_id与edge在static_edges_中的index的映射关系
        const int edge_id = static_edges_[i].edge_ID;

        if(edge_id_to_index_map_.find(edge_id) != edge_id_to_index_map_.end())
        {
            std::cout<<"\033[31m"<<"[ERROR]"<<"\033[0m"<<
            " WRONG EDGE POINT INDEX "<<edge_id<<", INDICES SHOULD BE DIFFERENT!"<<std::endl;
            continue;
        }else{
            edge_id_to_index_map_[edge_id] = i;
        }

        auto& edge = static_edges_[i];

        //-- 对于边缘中的每个边缘点，更新其对帧中所有边缘的索引
        for(int j = 0; j < edge.mvPoints.size(); ++j)
        {
            auto& point = edge.mvPoints[j];
            //-- 更新边缘id索引
            point.frame_edge_ID = edge_id;
            //-- 更新边缘点列表索引
            point.frame_point_index = static_cast<int>(j);
        }
    }
}

//-- 构建搜索阵列，把所有的 static_edges_ 里的所有点怼到一个 cv::Mat 里
void Frame::constructSearchPlain()
{
    // 创建一个 CV_32SC2 类型的 Mat，初始值设为 (-1, -1) 表示无效位置
    edge_point_lookup_map_ = cv::Mat(img_height_, img_width_, CV_32SC2, cv::Scalar(-1, -1));

    for (size_t i = 0; i < static_edges_.size(); ++i) 
    {
        const auto& edge = static_edges_[i];
        for (size_t j = 0; j < edge.mvPoints.size(); ++j) 
        {
            const auto& point = edge.mvPoints[j];
            
            // 确保坐标在图像范围内
            if(point.x >= 0 && point.x < img_width_ && point.y >= 0 && point.y < img_height_){
                // 访问指定位置并赋值
                auto& pixel = edge_point_lookup_map_.at<cv::Vec2i>(point.y, point.x);
                pixel[0] = point.frame_edge_ID;      // 第一个通道存储 edge ID
                pixel[1] = point.frame_point_index;  // 第二个通道存储 point index
            }else{
                std::cerr << "Point (" << point.x << ", " << point.y 
                          << ") out of bounds!" << std::endl;
            }
        }
    }
}

void Frame::constructSearchPlainParallel()
{
    //-- 创建并初始化矩阵
    edge_point_lookup_map_ = cv::Mat(img_height_, img_width_, CV_32SC2, cv::Scalar(-1, -1));
    
    // Validate image dimensions before processing
    if (img_height_ <= 0 || img_width_ <= 0) {
        LOG(ERROR) << "Invalid image dimensions in constructSearchPlainParallel: " 
                   << img_height_ << "x" << img_width_;
        return;
    }
    
    // 使用 parallel_for_each 并行处理所有边
    tbb::parallel_for_each(static_edges_.begin(), static_edges_.end(),
        [&](const auto& edge) {
            // 遍历当前边的所有点
            for (const auto& point : edge.mvPoints) {
                // Validate point coordinates before accessing
                if (point.x < 0 || point.x >= img_width_ || 
                    point.y < 0 || point.y >= img_height_) {
                    continue;  // Skip invalid points
                }
                // 直接写入矩阵
                auto& pixel = edge_point_lookup_map_.at<cv::Vec2i>(point.y, point.x);
                pixel[0] = point.frame_edge_ID;      // 存储 edge ID
                pixel[1] = point.frame_point_index; // 存储 point index
            }
        });
}

cv::Mat Frame::visualizeSearchPlain()
{
    std::map<int, std::vector<cv::Point>> edgeMap;

    for (int y = 0; y < edge_point_lookup_map_.rows; ++y) {
        for (int x = 0; x < edge_point_lookup_map_.cols; ++x) {
            const cv::Vec2i& pixel = edge_point_lookup_map_.at<cv::Vec2i>(y, x);
            int edgeID = pixel[0];  // 通道1: frame_edge_ID
            if (edgeID != -1) {     // 忽略 (-1,-1) 的无效点
                edgeMap[edgeID].emplace_back(x, y);
            }
        }
    }
    // Step 2: 创建彩色图像 (3通道 BGR)
    cv::Mat colorMat(edge_point_lookup_map_.size(), CV_8UC3, cv::Scalar(0, 0, 0)); // 默认黑色

    // Step 3: 为每个 edgeID 生成随机颜色
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(50, 255); // 避免太暗的颜色

    for (const auto& [edgeID, points] : edgeMap) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen)); // 随机 BGR 颜色

        // 绘制该 edgeID 的所有点
        for (const auto& pt : points) {
            colorMat.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(
                static_cast<uchar>(color[0]),
                static_cast<uchar>(color[1]),
                static_cast<uchar>(color[2])
            );
        }
    }

    return colorMat;
}

void Frame::assignProperty3D(const cv::Mat& matDepth)
{
    // 并行处理外循环
    tbb::parallel_for(0, (int)static_edges_.size(), [&](int i) {
        // 内循环保持串行
        for(size_t j = 0; j < static_edges_[i].mvPoints.size(); ++j) {
            assignProperty3DEach(static_edges_[i].mvPoints[j], matDepth);
        }
    });

    // for(int i = 0; i < static_edges_.size(); ++i)
    // {
    //     for(int j = 0; j < static_edges_[i].mvPoints.size(); ++j)
    //     {
    //         assignProperty3DEach(static_edges_[i].mvPoints[j], matDepth);
    //     }
    // }
}

void Frame::assignProperty3DEach(orderedEdgePoint& pt, const cv::Mat& matDepth)
{
    int x_idx = pt.x;
    int y_idx = pt.y;

    //-- 原本点的真实深度
    float depth_orig = matDepth.at<float>(y_idx, x_idx);

    //-- 在5x5的patch中计算修正深度以及可见性分数
    std::vector<float> validDepthList; //-- 所有深度不为0的点的深度列表
    int patch_total = 0;               //-- 当前的patch一共有多少像素
    for(int x_bias = -2; x_bias <= 2; ++x_bias){
        for(int y_bias = -2; y_bias <= 2; ++y_bias){
            int curr_x_idx = x_idx + x_bias;
            int curr_y_idx = y_idx + y_bias;
            
            //-- 判断该位置在不在图像区域里
            if(curr_x_idx < 0 || curr_x_idx >= img_width_ ||
               curr_y_idx < 0 || curr_y_idx >= img_height_) continue;

            patch_total += 1; //-- 累积总体像素
            //-- 在图像区域里的话判断深度值是否有效
            float depth = matDepth.at<float>(curr_y_idx, curr_x_idx);
            if(depth > 0.2) validDepthList.push_back(depth);

        }
    }
    std::sort(validDepthList.begin(), validDepthList.end());//-- 从小到大排序
    int size = validDepthList.size();
    float adjusted_depth = 0;     //-- 矫正后的深度

    //-- 计算前景深度
    if(size >= 8){
        //-- 检查跳变并返回第一组连续数据
        std::vector<size_t> jump_indices;
        float rel_thres = 0.05;
        for (size_t i = 1; i < validDepthList.size(); ++i){
            float dx = validDepthList[i] - validDepthList[i-1];
            float x = validDepthList[i-1];
            float relative_change = dx/x; // validDepthList 中均为大于0的数，不担心除0

            if (std::fabs(relative_change) > rel_thres) {
                jump_indices.push_back(i); // 记录跳变位置
                break;
            }
        }
        
        //-- 如果一个 patch 内深度值有不连续，则抠出最小的那一部分区域
        std::vector<float> adjustDepthList;
        if(jump_indices.empty())
        {
            adjustDepthList = validDepthList;
        }else{
            size_t first_jump = jump_indices[0];
            adjustDepthList = std::vector<float>(validDepthList.begin(), validDepthList.begin() + first_jump);
        }

        int partitionSize = adjustDepthList.size();
        //-- 取最小的部分的深度的中位数作为深度值
        float medianValue = (partitionSize%2==0) ? 
                            (adjustDepthList[partitionSize/2-1] + adjustDepthList[partitionSize/2])/2.0 : 
                            adjustDepthList[partitionSize/2];
        if(depth_orig >= adjustDepthList.front() && depth_orig <= adjustDepthList.back()){
            //-- 如果真实深度在这个区间之间，就取真实深度（真实深度本身是前景）
            adjusted_depth = depth_orig;
        }else{
            //-- 如果真实深度不在前景区间，则修改深度为前景区间
            adjusted_depth = medianValue;
        }
    }

    pt.depth = adjusted_depth; //-- 为点特征的深度进行赋值

    //-- 计算远近分数
    if(pt.depth > 0.2){
        Eigen::Vector3d pt_3d;
        pt_3d.x() = (pt.x - cam_cx_)/cam_fx_ * pt.depth;
        pt_3d.y() = (pt.y - cam_cy_)/cam_fy_ * pt.depth;
        pt_3d.z() = pt.depth;
        double range = pt_3d.norm();
        //-- 使用反sigmoid函数计算远近分数
        pt.score_depth = 1.0 / (std::exp((range - 2.5) * 1.0) + 1);
        //-- 更新类中的3D点
        pt.x_3d = pt_3d.x();
        pt.y_3d = pt_3d.y();
        pt.z_3d = pt_3d.z();
    }else{
        pt.score_depth = 0;
    }
    
}

//-- 剔除所有深度无效的边缘以及有效边缘中的无效边缘点
void Frame::edgeCullingDepth()
{
    //-- 遍历所有的边缘，祛除深度大量无效的边缘
    for(auto edgeIter = static_edges_.begin(); edgeIter != static_edges_.end(); )
    {
        //-- 获取当前边缘的引用，避免拷贝
        Edge& currentEdge = *edgeIter;

        int validPointCount = 0;
        int totalPointCount = currentEdge.mvPoints.size();
        
        //-- 先统计有效点的数量
        for(const auto& point : currentEdge.mvPoints)
        {
            if(point.depth > 0.2f && point.depth < 5.0f)
            {
                validPointCount++;
            }
        }

        //-- 计算有效点比例
        float validRatio = static_cast<float>(validPointCount) / totalPointCount;

        if(validRatio >= 0.3f)
        {
            //-- 如果边缘保留，则移除其中的无效点
            auto newEnd = std::remove_if(currentEdge.mvPoints.begin(), 
                                        currentEdge.mvPoints.end(),
                                        [](const auto& point) {
                                            return point.depth <= 0.2f || point.depth >= 5.0f;
                                        });
            currentEdge.mvPoints.erase(newEnd, currentEdge.mvPoints.end());
            edgeIter++;  // 保留这个边缘，移动到下一个
        }
        else
        {
            //-- 如果边缘无效（70%以上都是无效点），则移除整个边缘
            edgeIter = static_edges_.erase(edgeIter);
        }
    }
}

void Frame::edgeCullingDepthParallel()
{
    // 使用 char 代替 atomic<bool>，并用 memory_order_relaxed 保证基本线程安全
    std::vector<char> retainFlags(static_edges_.size());

    tbb::parallel_for(0, (int)static_edges_.size(), [&](int i) {
        Edge& currentEdge = static_edges_[i];
        int validPointCount = 0;
        const int totalPointCount = currentEdge.mvPoints.size();
        
        // 统计有效点数量
        for(const auto& point : currentEdge.mvPoints) {
            if(point.depth > 0.2f && point.depth < 5.0f) {
                validPointCount++;
            }
        }

        // 计算有效比例并决定是否保留
        float validRatio = static_cast<float>(validPointCount) / totalPointCount;
        retainFlags[i] = (validRatio >= 0.3f) ? 1 : 0;

        // 如果是保留的边缘，先过滤掉无效点
        if(retainFlags[i]) {
            auto newEnd = std::remove_if(currentEdge.mvPoints.begin(), 
                                        currentEdge.mvPoints.end(),
                                        [](const auto& point) {
                                            return point.depth <= 0.2f || point.depth >= 5.0f;
                                        });
            currentEdge.mvPoints.erase(newEnd, currentEdge.mvPoints.end());
        }
    });

    // 第二阶段：串行执行实际删除操作
    auto newEnd = std::remove_if(static_edges_.begin(), static_edges_.end(),
        [&retainFlags, &static_edges_ = this->static_edges_](const Edge& edge) {
            size_t index = &edge - &static_edges_[0];
            return retainFlags[index] == 0;
        });
    static_edges_.erase(newEnd, static_edges_.end());
}

//-- 确保每条有序边缘的3D点深度连续一致
void Frame::edgeCullingContinuity()
{
    std::vector<bool> isEdgeValid(static_edges_.size(), true);
    //-- 在CullingDepth之后调用，此时认为Edge中每个点都含有有效的深度
    tbb::parallel_for(0, (int)static_edges_.size(), [&](int cnt) {
    //for(int cnt = 0; cnt < static_edges_.size(); ++cnt)
        Edge& edge = static_edges_[cnt];
        
        //* STEP 1. 检索边缘的深度跳变
        std::vector<bool> jumpFlags(edge.mvPoints.size(), false);
        float lastDepth = edge.mvPoints[0].depth;
        for (size_t i = 1; i < edge.mvPoints.size(); ++i) 
        {
            //-- 当前点的深度
            float currentDepth = edge.mvPoints[i].depth;
            //-- 比较深度判断是否连续
            jumpFlags[i] = (std::fabs(currentDepth - lastDepth) > 0.05f);
            lastDepth = currentDepth;
        }
        //-- 跳变次数
        int jump_num = std::count(jumpFlags.begin(), jumpFlags.end(), true);
        float jump_avg =  static_cast<float>(edge.mvPoints.size())/static_cast<float>(jump_num);
        if(jump_avg < 5){
            //-- 如果跳变的比较多，说明该边缘正处于前后景模糊的位置
            isEdgeValid[cnt] = false;
            //-- 对于这样的边缘，考虑直接删而不重新拼,故而跳过
            return;
        }

        //*STEP 2. 对于跳变的不多的边缘，先根据 jumpFlags 进行切片
        int start_ptr = 0;
        //-- 一个 edge 被拆出的 segment, first是首 index，second 是末 index
        std::vector<std::pair<int, int>> segment;

        for(size_t i = 1; i < jumpFlags.size(); ++i)
        {
            if(jumpFlags[i])
            {
                //-- 第 i 个位置跳变了说明 start -- i-1 这一段是连续的
                segment.push_back(std::make_pair(start_ptr, i-1));
                start_ptr = i;
            }
        }

        //-- 最后一截拼入, 此时segment中包含所有的边缘切片（包括不连续的单个点的切片）
        segment.push_back(std::make_pair(start_ptr, jumpFlags.size()-1));
        
        // * STEP 3. 使用并查集合并连续的片段
        //-- 表示并查集
        DisjointSet mergeSet(segment.size());

        for(int i = 0; i < segment.size(); ++i)
        {
            float depth_end = edge.mvPoints[segment[i].second].depth; //-- 片段末端点的深度
            //-- 判断后续的片段能不能和第i段相拼接
            for(int j = i+2; j < segment.size(); ++j)
            {
                float depth_front = edge.mvPoints[segment[j].first].depth; //-- 片段首端点的深度
                //-- 两个片段首末端点深度连续说明可以拼
                if(std::fabs(depth_front - depth_end) < 0.01)
                {
                    //-- 如果能拼上就在并查集上合并这两个节点
                    mergeSet.to_union(i,j);
                }
            }
        }

        // * STEP 4. 挑选最大连续片段以进行后续拼接
        //-- 整理并查集，得到每个集合的总点数
        mergeSet.pruningSet();
        std::map<int, std::vector<int>> cluster; //-- first是root_idx, second是集合中的所有片段的索引
        for(int i = 0; i < segment.size(); ++i)
        {
            int root_idx = mergeSet.find(i);
            cluster[root_idx].push_back(i);
        }
        //-- 此时cluster.second中这些片段本身也是有序的
        int max_length = -1;
        std::vector<int> max_cluster;
        for(const auto& pair : cluster)
        {
            std::vector<int> current_cluster = pair.second;
            int current_length = 0;
            
            //-- 计算当前cluster的长度
            for(int i = 0; i < current_cluster.size(); ++i)
            {
                const auto& seg = segment[current_cluster[i]];
                current_length += seg.second - seg.first + 1;
            }

            if(current_length > max_length){
                max_length = current_length;
                max_cluster = current_cluster;
            }
        }

        // * STEP 5. 根据 max_cluster 拼接
        //-- 重新捏一个mvPoints出来
        std::vector<orderedEdgePoint> new_mvPoints;
        for(int i = 0; i < max_cluster.size(); ++i)
        {
            //-- 每个切片的首尾index
            int start_idx = segment[max_cluster[i]].first;
            int end_index = segment[max_cluster[i]].second;
            for(int j = start_idx; j <= end_index; ++j)
            {
                new_mvPoints.push_back(edge.mvPoints[j]);
            }
        }
        if(new_mvPoints.size() >= 5)
        {
            edge.mvPoints = new_mvPoints;
        }else{
            isEdgeValid[cnt] = false;
        }
    });

    // * STEP 6. 移除拼接完成后过短的边缘以及跳变过多的边缘
    int cnt_idx = 0;
    for(auto iter = static_edges_.begin(); iter != static_edges_.end(); )
    {
        if(isEdgeValid[cnt_idx] == true){
            iter ++;
        }else{
            iter = static_edges_.erase(iter);
        }
        cnt_idx += 1;
    }
}


std::vector<orderedEdgePoint> Frame::getCoarseSampledPoints(int bias, int maximum_point)
{
    std::vector<orderedEdgePoint> selectedPoints;
    for(int i = 0; i < static_edges_.size(); ++i)
    {
        const Edge& edge = static_edges_[i];
        //-- 获取采样的序列
        for(int j = 0; j < edge.mvPoints.size(); ++j)
        {
            const orderedEdgePoint& pt = edge.mvPoints[j];
            selectedPoints.push_back(pt);
        }
    }
    //-- 根据空间均匀的原则进行采样，分数高的点优先
    //-- 对点按分数进行排序
    std::sort(selectedPoints.begin(),selectedPoints.end(),
              [](const orderedEdgePoint& a, const orderedEdgePoint& b){ 
                 return a.score_depth > b.score_depth; 
              });

    //-- 创建全黑的图像作为掩膜
    cv::Mat mask(img_height_, img_width_, CV_8U, cv::Scalar(0));
    //-- 根据掩膜进行采样排序
    std::vector<orderedEdgePoint> sampledPoints;
    sampledPoints.reserve(std::min(maximum_point, static_cast<int>(selectedPoints.size())));

    for(const auto& pt : selectedPoints)
    {
        if(mask.at<uint8_t>(pt.y, pt.x) == 255) continue;
        //-- 当前点可以选择
        sampledPoints.push_back(pt);
        cv::circle(mask, cv::Point(pt.x, pt.y), bias, 255, -1);
        if(sampledPoints.size() >= maximum_point) break;
    }
    //-- 目前是完全采样完成的所有点
    return sampledPoints;
}



}  // namespace dyno
