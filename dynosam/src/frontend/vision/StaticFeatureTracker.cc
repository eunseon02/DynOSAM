/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/frontend/vision/StaticFeatureTracker.hpp"

#include <gflags/gflags.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

DECLARE_bool(use_edge_feature);

namespace dyno {

StaticFeatureTracker::StaticFeatureTracker(const TrackerParams& params,
                                           Camera::Ptr camera,
                                           ImageDisplayQueue* display_queue)
    : FeatureTrackerBase(params, camera, display_queue) {}

ExternalFlowFeatureTracker::ExternalFlowFeatureTracker(
    const TrackerParams& params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : StaticFeatureTracker(params, camera, display_queue),
      static_grid_(
          static_cell_size,
          std::ceil(static_cast<double>(camera->getParams().ImageWidth()) /
                    static_cell_size),
          std::ceil(static_cast<double>(camera->getParams().ImageHeight()) /
                    static_cell_size)) {
  const auto orb_params = params.orb_params;
  orb_detector_ = std::make_unique<ORBextractor>(
      params.max_nr_keypoints_before_anms,
      static_cast<float>(orb_params.scale_factor), orb_params.n_levels,
      orb_params.init_threshold_fast, orb_params.min_threshold_fast);

  CHECK(!img_size_.empty());
}

// if previous frame is null, assume that it is the first frame... and that
// frames are always processed in order!
FeatureContainer ExternalFlowFeatureTracker::trackStatic(
    Frame::Ptr previous_frame, const ImageContainer& image_container,
    FeatureTrackerInfo& tracker_info, const cv::Mat&,
    std::vector<Edge>& detected_edges,
    const std::optional<gtsam::Rot3>&) {
  VLOG(10) << "ExternalFlowFeatureTracker::trackStatic";
  // ExternalFlowFeatureTracker doesn't detect edges, so clear output
  detected_edges.clear();
  const ImageWrapper<ImageType::RGBMono>& rgb_wrapper = image_container.rgb();
  const cv::Mat& rgb = rgb_wrapper.toRGB();
  cv::Mat mono = ImageType::RGBMono::toMono(rgb_wrapper);
  CHECK(!mono.empty());

  const cv::Mat& motion_mask = image_container.objectMotionMask();
  CHECK(!motion_mask.empty());

  cv::Mat descriptors;
  KeypointsCV detected_keypoints;
  (*orb_detector_)(mono, cv::Mat(), detected_keypoints, descriptors);

  // assign tracked features to grid and add to static features
  FeatureContainer static_features;

  const size_t& min_tracks =
      static_cast<size_t>(params_.max_features_per_frame);
  const FrameId frame_k = image_container.frameId();

  // appy tracking (ie get correspondences)
  // TODO: only track frames that have been tracked for some time?
  if (previous_frame) {
    // TODO: for now assume consequative frames
    const FrameId frame_k_1 = previous_frame->getFrameId();
    CHECK_EQ(frame_k_1 + 1u, frame_k);

    for (Feature::Ptr previous_feature : previous_frame->static_features_) {
      const size_t tracklet_id = previous_feature->trackletId();
      const size_t age = previous_feature->age();
      const Keypoint kp = previous_feature->predictedKeypoint();

      // check kp contained before we do a static grid look up to ensure we
      // don't go out of bounds
      if (!camera_->isKeypointContained(kp)) {
        continue;
      }

      const int x = functional_keypoint::u(kp);
      const int y = functional_keypoint::v(kp);
      const size_t cell_idx = static_grid_.getCellIndex(kp);
      const ObjectId instance_label = motion_mask.at<ObjectId>(y, x);

      if (static_grid_.isOccupied(cell_idx)) continue;

      if (previous_feature->usable() && instance_label == background_label) {
        size_t new_age = age + 1;
        Feature::Ptr feature = constructStaticFeature(
            image_container, kp, new_age, tracklet_id, frame_k);
        if (feature) {
          static_features.add(feature);
          static_grid_.occupancy_[cell_idx] = true;
        }
      }
    }
  }

  // number features tracked with optical flow
  const auto n_optical_flow = static_features.size();
  tracker_info.static_track_optical_flow = n_optical_flow;

  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();

  if (static_features.size() < min_tracks) {
    // iterate over new observations
    for (size_t i = 0; i < detected_keypoints.size(); i++) {
      if (static_features.size() >= min_tracks) {
        break;
      }

      const KeypointCV& kp_cv = detected_keypoints[i];
      const int& x = kp_cv.pt.x;
      const int& y = kp_cv.pt.y;

      // if not already tracked with optical flow
      if (motion_mask.at<int>(y, x) != background_label) {
        continue;
      }

      Keypoint kp(x, y);
      const size_t cell_idx = static_grid_.getCellIndex(kp);
      if (!static_grid_.isOccupied(cell_idx)) {
        const size_t age = 0;
        size_t tracklet_id = tracked_id_manager.getTrackletIdCount();
        Feature::Ptr feature = constructStaticFeature(image_container, kp, age,
                                                      tracklet_id, frame_k);
        if (feature) {
          tracked_id_manager.incrementTrackletIdCount();
          static_grid_.occupancy_[cell_idx] = true;
          static_features.add(feature);
        }
      }
    }
  }

  static_grid_.reset();

  size_t total_tracks = static_features.size();
  tracker_info.static_track_detections = total_tracks - n_optical_flow;
  return static_features;
}

Feature::Ptr ExternalFlowFeatureTracker::constructStaticFeature(
    const ImageContainer& image_container, const Keypoint& kp, size_t age,
    TrackletId tracklet_id, FrameId frame_id) const {
  // implicit double -> int cast for pixel location
  const int x = functional_keypoint::u(kp);
  const int y = functional_keypoint::v(kp);

  const cv::Mat& rgb = image_container.rgb();
  const cv::Mat& motion_mask = image_container.objectMotionMask();
  const cv::Mat& optical_flow = image_container.opticalFlow();

  CHECK(!optical_flow.empty());
  CHECK(!motion_mask.empty());

  if (motion_mask.at<int>(y, x) != background_label) {
    return nullptr;
  }

  // check flow
  double flow_xe = static_cast<double>(optical_flow.at<cv::Vec2f>(y, x)[0]);
  double flow_ye = static_cast<double>(optical_flow.at<cv::Vec2f>(y, x)[1]);

  if (!(flow_xe != 0 && flow_ye != 0)) {
    return nullptr;
  }

  OpticalFlow flow(flow_xe, flow_ye);

  // check predicted flow is within image
  Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(kp, flow);
  if (!camera_->isKeypointContained(predicted_kp)) {
    return nullptr;
  }

  Feature::Ptr feature = std::make_shared<Feature>();
  (*feature)
      .objectId(background_label)
      .frameId(frame_id)
      .keypointType(KeyPointType::STATIC)
      .age(age)
      .trackletId(tracklet_id)
      .keypoint(kp)
      .measuredFlow(flow)
      .predictedKeypoint(predicted_kp);
  return feature;
}

KltFeatureTracker::KltFeatureTracker(const TrackerParams& params,
                                     Camera::Ptr camera,
                                     ImageDisplayQueue* display_queue)
    : StaticFeatureTracker(params, camera, display_queue) {
  detector_ = std::make_shared<SparseFeatureDetector>(
      params, FunctionalDetector::FactoryCreate(params));

  static const cv::Size klt_window_size(21, 21);  // Window size for KLT
  static const int klt_max_level = 3;             // Max pyramid levels for KLT
  static const cv::TermCriteria klt_criteria = cv::TermCriteria(
      cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.03);

  // used as flags argument for calcOpticalFlowPyrLK - initially starts as
  // default (0) flag
  int klt_flags = 0;

  lk_cuda_tracker_ = cv::cuda::SparsePyrLKOpticalFlow::create(
      klt_window_size, klt_max_level, klt_criteria.maxCount);

  CHECK_NOTNULL(detector_);
}

FeatureContainer KltFeatureTracker::trackStatic(
    Frame::Ptr previous_frame, const ImageContainer& image_container,
    FeatureTrackerInfo& tracker_info, const cv::Mat& detection_mask,
    std::vector<Edge>& detected_edges,
    const std::optional<gtsam::Rot3>& R_km1_k) {
  VLOG(10) << "KltFeatureTracker::trackStatic called, FLAGS_use_edge_feature="
           << FLAGS_use_edge_feature;
  // tracked features and new features
  FeatureContainer new_tracks_and_detections;
  EdgeContainer new_edges;
  
  // Clear output detected_edges
  detected_edges.clear();

  // Validate image_container before processing
  if (!image_container.hasRgb()) {
    LOG_EVERY_N(ERROR, 50)
        << "image_container has no RGB, cannot track static features";
    return new_tracks_and_detections;
  }

  cv::Mat current_equialized_greyscale;
  VLOG(15) << "KltFeatureTracker::trackStatic: calling equalizeImage for current frame";
  equalizeImage(image_container, current_equialized_greyscale);
  VLOG(15) << "KltFeatureTracker::trackStatic: equalizeImage completed, image size="
           << current_equialized_greyscale.size();
  
  if (current_equialized_greyscale.empty()) {
    LOG_EVERY_N(ERROR, 50)
        << "current_equialized_greyscale is empty after equalizeImage, cannot track static features";
    return new_tracks_and_detections;
  }

  if (!previous_frame) {
    VLOG(10) << "KltFeatureTracker::trackStatic: no previous frame, detecting features";
    FeatureContainer previous_inliers;
    detectFeatures(current_equialized_greyscale, image_container,
                   previous_inliers, new_tracks_and_detections, new_edges,
                   detection_mask);
    
    // Copy detected edges from new_edges to output parameter
    VLOG(15) << "KltFeatureTracker::trackStatic (no previous frame): new_edges.size()="
             << new_edges.size();
    for (const Edge& edge : new_edges) {
      detected_edges.push_back(edge);
    }
    VLOG(15) << "KltFeatureTracker::trackStatic (no previous frame): detected_edges.size()="
             << detected_edges.size();

    tracker_info.static_track_detections = new_tracks_and_detections.size();

    return new_tracks_and_detections;
  } else {
    VLOG(10) << "KltFeatureTracker::trackStatic: has previous frame, processing";
    // we have previous tracks
    // we should have already calculated the processed rgb image from the
    // previous frame
    cv::Mat previous_equialized_greyscale;
    VLOG(15) << "KltFeatureTracker::trackStatic: calling equalizeImage for previous frame";
    equalizeImage(previous_frame->image_container_,
                  previous_equialized_greyscale);
    VLOG(15) << "KltFeatureTracker::trackStatic: equalizeImage for previous frame completed, image size="
             << previous_equialized_greyscale.size();
    
    if (previous_equialized_greyscale.empty()) {
      LOG_EVERY_N(ERROR, 50)
          << "previous_equialized_greyscale is empty after equalizeImage, cannot track static features";
      return new_tracks_and_detections;
    }

    VLOG(15) << "KltFeatureTracker::trackStatic: collecting previous inliers";
    FeatureContainer previous_inliers;
    auto iter = previous_frame->static_features_.beginUsable();
    for (const auto& inlier_feature : iter) {
      previous_inliers.add(inlier_feature);
    }
    VLOG(15) << "KltFeatureTracker::trackStatic: collected " << previous_inliers.size()
             << " previous inliers";

    // if we dont actually have any previous tracks
    // this may be in cases where we have an IMU and so we have some odometry
    // but no feature tracks from the previous frame!
    if (previous_inliers.empty()) {
      VLOG(10) << "KltFeatureTracker::trackStatic: previous_inliers is empty, detecting new features";
      FeatureContainer previous_inliers;
      detectFeatures(current_equialized_greyscale, image_container,
                     previous_inliers, new_tracks_and_detections, new_edges,
                     detection_mask);
      
      // Copy detected edges from new_edges to output parameter
      for (const Edge& edge : new_edges) {
        detected_edges.push_back(edge);
      }
      
      tracker_info.static_track_detections = new_tracks_and_detections.size();
      return new_tracks_and_detections;
    }

    VLOG(10) << "KltFeatureTracker::trackStatic: previous_inliers not empty, proceeding to trackPoints";
    // Tracklet ids associated with the set of previous inliers that are now
    // outliers
    TrackletIds previous_outliers;
    VLOG(15) << "KltFeatureTracker::trackStatic: created previous_outliers";
    VLOG(15) << "KltFeatureTracker::trackStatic sizes"
             << " curr_img=" << current_equialized_greyscale.size()
             << " prev_img=" << previous_equialized_greyscale.size()
             << " prev_inliers=" << previous_inliers.size();

    // track features from the previous frame and detect new ones if necessary
    bool track_result = trackPoints(
        current_equialized_greyscale, previous_equialized_greyscale,
        image_container, previous_inliers, new_tracks_and_detections,
        previous_outliers, tracker_info, detection_mask, R_km1_k, new_edges);
    
    VLOG(15) << "KltFeatureTracker::trackStatic: trackPoints returned " << track_result;
    
    if (!track_result) {
      LOG_EVERY_N(ERROR, 50) << "KltFeatureTracker::trackStatic: trackPoints failed!";
      return new_tracks_and_detections;
    }
    // CHECK(trackEdges(current_equialized_greyscale, previous_equialized_greyscale, image_container, previous_inliers, new_tracks_and_detections, previous_outliers, tracker_info, detection_mask, R_km1_k));

    // Copy detected edges from new_edges to output parameter
    VLOG(15) << "KltFeatureTracker::trackStatic (with previous frame): new_edges.size()="
             << new_edges.size();
    for (const Edge& edge : new_edges) {
      detected_edges.push_back(edge);
    }
    VLOG(15) << "KltFeatureTracker::trackStatic (with previous frame): detected_edges.size()="
             << detected_edges.size();

    // after tracking, mark features in the older frame as outliers
    // TODO: (jesse) actually not sure we HAVE to do this, but better to keep
    // things as consisent as possible
    previous_frame->static_features_.markOutliers(previous_outliers);

    return new_tracks_and_detections;
  }
}

std::vector<Edge> KltFeatureTracker::getDetectedEdges() const {
  // Return detected edges stored during detectFeatures()
  return detected_edges_;
}

void KltFeatureTracker::equalizeImage(const ImageContainer& image_container,
                                      cv::Mat& equialized_greyscale) const {
  try {
    if (!image_container.hasRgb()) {
      LOG_EVERY_N(ERROR, 100) << "image_container has no RGB in equalizeImage";
      equialized_greyscale = cv::Mat();
      return;
    }
    
    const ImageWrapper<ImageType::RGBMono>& rgb_wrapper = image_container.rgb();
    
    if (!rgb_wrapper.exists()) {
      LOG_EVERY_N(ERROR, 100) << "rgb_wrapper does not exist in equalizeImage";
      equialized_greyscale = cv::Mat();
      return;
    }
    
    const cv::Mat& rgb = rgb_wrapper.toRGB();
    if (rgb.empty()) {
      LOG_EVERY_N(ERROR, 100) << "rgb image is empty in equalizeImage";
      equialized_greyscale = cv::Mat();
      return;
    }
    
    cv::Mat mono = ImageType::RGBMono::toMono(rgb_wrapper);
    if (mono.empty()) {
      LOG_EVERY_N(ERROR, 100) << "mono image is empty in equalizeImage";
      equialized_greyscale = cv::Mat();
      return;
    }

    mono.copyTo(equialized_greyscale);
    // CHECK(clahe_);

    // clahe_->apply(mono, equialized_greyscale);
  } catch (const std::exception& e) {
    LOG_EVERY_N(ERROR, 100) << "Exception in equalizeImage: " << e.what();
    equialized_greyscale = cv::Mat();
  } catch (...) {
    LOG_EVERY_N(ERROR, 100) << "Unknown exception in equalizeImage";
    equialized_greyscale = cv::Mat();
  }
}

std::vector<cv::Point2f> KltFeatureTracker::detectRawFeatures(
    const cv::Mat& processed_img, int number_tracked, const cv::Mat& mask) {
  KeypointsCV keypoints;
  detector_->detect(processed_img, keypoints, number_tracked, mask);

  std::vector<cv::Point2f> points;
  cv::KeyPoint::convert(keypoints, points);
  return points;
}

std::vector<Edge> KltFeatureTracker::detectEdgeFeatures(
  const cv::Mat& processed_img, int number_tracked, const cv::Mat& mask,
  const cv::Mat& depth_img) {
  std::vector<Edge> edges;
  detector_->detectEdge(processed_img, edges, mask, depth_img);
  return edges;
}

std::vector<Edge> KltFeatureTracker::detectEdges(const cv::Mat& processed_img, int number_tracked) {
  std::vector<Edge> detected_edges;
  
  // Validate input image
  if (processed_img.empty()) {
    VLOG(5) << "processed_img is empty, skipping edge detection";
    detected_edges_.clear();
    return detected_edges;
  }
  
  // Follow the same pattern as trackStatic -> detectFeatures -> detectEdgeFeatures
  {
    utils::ChronoTimingStats edge_timer("static_feature_track.detect_edges");
    // Use empty mask to detect edges on all pixels (static + dynamic regions)
    // This matches the behavior in detectFeatures (line 607)
    cv::Mat empty_mask;
    try {
      detected_edges = detectEdgeFeatures(processed_img, number_tracked, empty_mask);
      VLOG(15) << "Edge detection: detected " << detected_edges.size()
               << " edges (FLAGS_use_edge_feature=" << FLAGS_use_edge_feature << ")";
    } catch (const std::exception& e) {
      LOG_EVERY_N(ERROR, 50) << "Exception in detectEdgeFeatures: " << e.what();
      detected_edges.clear();
    } catch (...) {
      LOG_EVERY_N(ERROR, 50) << "Unknown exception in detectEdgeFeatures";
      detected_edges.clear();
    }
  }
  
  // Validate and store detected edges (same validation as in detectFeatures)
  // Always update detected_edges_ even if empty
  // This ensures that if edge detection fails in one frame, we can retry in the next frame
  try {
    // Validation disabled to match edgeSelector behavior
    // std::vector<Edge> valid_edges;  // Store only valid edges
    // if (!detected_edges.empty()) {
    //   for (const Edge& edge : detected_edges) {
    //     // Validate edge before adding
    //     if (edge.mvPoints.empty()) {
    //       VLOG(5) << "Skipping edge with empty mvPoints";
    //       continue;
    //     }
    //     
    //     // Validate edge points for invalid values
    //     bool edge_valid = true;
    //     for (const auto& pt : edge.mvPoints) {
    //       // Check for NaN or invalid coordinates
    //       if (std::isnan(pt.x) || std::isnan(pt.y) || 
    //           std::isinf(pt.x) || std::isinf(pt.y) ||
    //           pt.x < 0 || pt.y < 0) {
    //         VLOG(5) << "Skipping edge with invalid point coordinates (x=" << pt.x << ", y=" << pt.y << ")";
    //         edge_valid = false;
    //         break;
    //       }
    //       // Check for invalid gradient angle
    //       if (std::isnan(pt.imgGradAngle) || std::isinf(pt.imgGradAngle)) {
    //         VLOG(5) << "Skipping edge with invalid gradient angle (angle=" << pt.imgGradAngle << ")";
    //         edge_valid = false;
    //         break;
    //       }
    //     }
    //     
    //     if (!edge_valid) {
    //       continue;
    //     }
    //     
    //     valid_edges.push_back(edge);  // Store valid edge
    //   }
    // }
    // Always update detected_edges_ with all edges (no validation)
    // This matches the behavior in edgeSelector
    detected_edges_ = detected_edges;
  } catch (const std::exception& e) {
    LOG_EVERY_N(ERROR, 50) << "Exception while storing detected edges: " << e.what();
    detected_edges_.clear();
  } catch (...) {
    LOG_EVERY_N(ERROR, 50) << "Unknown exception while storing detected edges";
    detected_edges_.clear();
  }
  
  return detected_edges_;
}

std::vector<Edge> KltFeatureTracker::detectEdges(const ImageContainer& image_container, int number_tracked) {
  // Follow the same pattern as trackStatic (lines 272-274)
  // First equalize the image, then detect edges
  std::vector<Edge> detected_edges;
  
  // Validate image_container before processing
  if (!image_container.hasRgb()) {
    LOG_EVERY_N(ERROR, 50)
        << "image_container has no RGB, cannot detect edges";
    detected_edges_.clear();
    return detected_edges;
  }

  // Equalize image (same as trackStatic line 274)
  cv::Mat equalized_greyscale;
  VLOG(15) << "KltFeatureTracker::detectEdges: calling equalizeImage";
  equalizeImage(image_container, equalized_greyscale);
  VLOG(15) << "KltFeatureTracker::detectEdges: equalizeImage completed, image size="
           << equalized_greyscale.size();
  
  if (equalized_greyscale.empty()) {
    LOG_EVERY_N(ERROR, 50)
        << "equalized_greyscale is empty after equalizeImage, cannot detect edges";
    detected_edges_.clear();
    return detected_edges;
  }

  // Call the processed image version (same pattern as trackStatic -> detectFeatures -> detectEdgeFeatures)
  // Prefer depth-aware edge detection in this path.
  return detectEdgeFeatures(equalized_greyscale, number_tracked, cv::Mat(),
                            image_container.depth());
}

bool KltFeatureTracker::detectFeatures(const cv::Mat& processed_img,
                                       const ImageContainer& image_container,
                                       const FeatureContainer& current_features,
                                       FeatureContainer& new_features,
                                       EdgeContainer& new_edges,
                                       const cv::Mat& detection_mask) {
  const FrameId frame_k = image_container.frameId();
  const cv::Mat& motion_mask = image_container.objectMotionMask();

  // internal detection mask that is appended with new invalid pixels
  // this builds the static detection mask over the existing input mask
  cv::Mat detection_mask_impl;
  // If we are provided with an external detection/feature mask, initalise the
  // detection mask with this and add more invalid sections to it
  if (!detection_mask.empty()) {
    CHECK_EQ(motion_mask.rows, detection_mask.rows);
    CHECK_EQ(motion_mask.cols, detection_mask.cols);
    detection_mask_impl = detection_mask.clone();
  } else {
    detection_mask_impl = cv::Mat(motion_mask.size(), CV_8U, cv::Scalar(255));
  }
  CHECK_EQ(detection_mask_impl.type(), CV_8U);
  // create mask from object mask so that all pixels > 0 are ignored (by setting
  // the value in the new mask to 0 at these locations) start with an invalid
  // mask
  cv::Mat object_feature_mask = cv::Mat::zeros(motion_mask.size(), CV_8U);
  // Set background pixels (0 in motion_mask) -> valid (1 in
  // object_feature_mask)
  object_feature_mask.setTo(255, motion_mask == 0);
  // combine with existing mask information (from current features)
  cv::bitwise_and(detection_mask_impl, object_feature_mask,
                  detection_mask_impl);

  // slow
  // add mask over objects detected in the scene
  // TODO: should just be a masking operation but treating all non-zero pixels
  // as 1 (ie make binary) and then inveverting the mask so that object pixels
  // (originally 1) become 0, indicating they should not be used! for (int i =
  // 0; i < motion_mask.rows; i++) {
  //   for (int j = 0; j < motion_mask.cols; j++) {
  //     const ObjectId label = motion_mask.at<ObjectId>(i, j);

  //     if (label != background_label) {
  //       cv::circle(
  //           detection_mask_impl, cv::Point2f(j, i),
  //           params_.min_distance_btw_tracked_and_detected_static_features,
  //           cv::Scalar(0), cv::FILLED);
  //     }
  //   }
  // }

  // add mask over current static features
  for (const auto& feature : current_features) {
    const Keypoint kp = feature->keypoint();
    CHECK(feature->usable());
    cv::circle(detection_mask_impl, cv::Point2f(kp(0), kp(1)),
               params_.min_distance_btw_tracked_and_detected_static_features,
               cv::Scalar(0), cv::FILLED);
  }

  std::vector<cv::Point2f> detected_points;
  {
    utils::ChronoTimingStats timer("static_feature_track.detect_raw");
    detected_points = detectRawFeatures(processed_img, current_features.size(),
                                        detection_mask_impl);
  }
  {
    utils::ChronoTimingStats timer("static_feature_track.detect_edges");
    if (FLAGS_use_edge_feature) {
      VLOG(10) << "KltFeatureTracker::detectFeatures: FLAGS_use_edge_feature=true, detecting edges";
      // Use empty mask to detect edges on all pixels (static + dynamic regions)
      cv::Mat empty_mask;
      std::vector<Edge> edges =
          detectEdgeFeatures(processed_img, current_features.size(), empty_mask,
                             image_container.depth());
      VLOG(10) << "KltFeatureTracker::detectFeatures: detectEdgeFeatures returned " << edges.size() << " edges";
      // Add detected edges to output container and internal storage
      for (const Edge& edge : edges) {
        new_edges.add(edge);
      }
      detected_edges_ = edges;
      VLOG(10) << "KltFeatureTracker::detectFeatures: after adding edges, new_edges.size()=" << new_edges.size();
    } else {
      VLOG(10) << "KltFeatureTracker::detectFeatures: FLAGS_use_edge_feature=false, skipping edge detection";
    }
  }

  for (const cv::Point2f& detected_point : detected_points) {
    Keypoint kp(static_cast<double>(detected_point.x),
                static_cast<double>(detected_point.y));
    const int x = functional_keypoint::u(kp);
    const int y = functional_keypoint::v(kp);

    if (!(camera_->isKeypointContained(kp) && isWithinShrunkenImage(kp))) {
      continue;
    }

    // with the detection mask this should never happen
    if (motion_mask.at<int>(y, x) != background_label) {
      continue;
    }

    Feature::Ptr feature = constructNewStaticFeature(kp, frame_k);
    if (feature) {
      new_features.add(feature);
    }
  }

  return true;
}

bool KltFeatureTracker::trackPoints(const cv::Mat& current_processed_img,
                                    const cv::Mat& previous_processed_img,
                                    const ImageContainer& image_container,
                                    const FeatureContainer& previous_features,
                                    FeatureContainer& tracked_features,
                                    TrackletIds& outlier_previous_features,
                                    FeatureTrackerInfo& tracker_info,
                                    const cv::Mat& detection_mask,
                                    const std::optional<gtsam::Rot3>& R_km1_k,
                                    EdgeContainer& new_edges) {
  VLOG(20) << "KltFeatureTracker::trackPoints: entered function";
  if (current_processed_img.empty() || previous_processed_img.empty() ||
      previous_features.empty()) {
    VLOG(5) << "KltFeatureTracker::trackPoints: input validation failed";
    return false;
  }
  VLOG(20) << "KltFeatureTracker::trackPoints: input validation passed";

  outlier_previous_features.clear();

  const cv::Mat& motion_mask = image_container.objectMotionMask();
  
  const FrameId frame_k = image_container.frameId();

  std::vector<uchar> klt_status;
  std::vector<float> err;
  // All tracklet ids from the set of previous features to track
  TrackletIds tracklet_ids;

  // cannot just get inliers (becuase in reality this is)
  VLOG(20) << "KltFeatureTracker::trackPoints: previous_features.size()="
           << previous_features.size();
  std::vector<cv::Point2f> previous_pts =
      previous_features.toOpenCV(&tracklet_ids, true);
  CHECK_EQ(previous_pts.size(), previous_features.size());
  CHECK_EQ(previous_pts.size(), tracklet_ids.size());

  static const cv::Size klt_window_size(21, 21);  // Window size for KLT
  static const int klt_max_level = 3;             // Max pyramid levels for KLT
  static const cv::TermCriteria klt_criteria = cv::TermCriteria(
      cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.03);

  // used as flags argument for calcOpticalFlowPyrLK - initially starts as
  // default (0) flag
  int klt_flags = 0;
  std::vector<cv::Point2f> current_points;
  if (R_km1_k) {
    predictKeypointsGivenRotation(current_points, previous_pts, *R_km1_k);
    klt_flags = cv::OPTFLOW_USE_INITIAL_FLOW;
  } else {
    // as per documentation the vector must have the same size as the input
    current_points.resize(previous_pts.size());
  }
  CHECK_EQ(current_points.size(), previous_pts.size());

  {
    // utils::ChronoTimingStats timer("static_feature_track.calc_LK");
    // cv::cuda::GpuMat gpu_prev_img(previous_processed_img);
    // cv::cuda::GpuMat gpu_current_img(current_processed_img);

    // cv::cuda::GpuMat d_points1(previous_pts);    // upload points
    // cv::cuda::GpuMat d_points2(current_points);  // output points
    // cv::cuda::GpuMat d_status;                   // status of each point
    // cv::cuda::GpuMat d_err;                      // error for each point

    // lk_cuda_tracker_->calc(gpu_prev_img, gpu_current_img, d_points1,
    // d_points2,
    //                        d_status, d_err);

    // // Download results back to CPU
    // d_points2.download(current_points);
    // d_status.download(status);

    cv::calcOpticalFlowPyrLK(previous_processed_img, current_processed_img,
                             previous_pts, current_points, klt_status, err,
                             klt_window_size, klt_max_level, klt_criteria,
                             klt_flags);

    // if we used OPTFLOW_USE_INITIAL_FLOW check that we actually got good flow
    if (klt_flags == cv::OPTFLOW_USE_INITIAL_FLOW) {
      static constexpr int kMinSuccessTracks = 10;
      int succ_num = 0;
      for (size_t i = 0; i < klt_status.size(); i++) {
        if (klt_status[i]) succ_num++;
      }
      if (succ_num < kMinSuccessTracks) {
        LOG_EVERY_N(WARNING, 50) << "Using initial flow for KLT tracking failed: only "
                                 << succ_num << " tracked! Retrying without initial flow.";
        cv::calcOpticalFlowPyrLK(previous_processed_img, current_processed_img,
                                 previous_pts, current_points, klt_status, err,
                                 klt_window_size, klt_max_level, klt_criteria);
      }
    }

    // check flow back
    std::vector<cv::Point2f> reverse_previous_feature_points = current_points;
    std::vector<uchar> klt_reverse_status;
    cv::calcOpticalFlowPyrLK(current_processed_img, previous_processed_img,
                             current_points, reverse_previous_feature_points,
                             klt_reverse_status, err, cv::Size(21, 21), 5);
    CHECK_EQ(klt_reverse_status.size(), tracklet_ids.size());

    auto distance = [](const cv::Point2f& pt1,
                       const cv::Point2f& pt2) -> float {
      float dx = pt1.x - pt2.x;
      float dy = pt1.y - pt2.y;
      return std::sqrt(dx * dx + dy * dy);
    };
    
    // update klt status based on result from flow
    for (size_t i = 0; i < klt_status.size(); i++) {
      const bool both_status_good =
          klt_status.at(i) && klt_reverse_status.at(i);
      const bool within_distance =
          distance(previous_pts.at(i), reverse_previous_feature_points.at(i)) <=
          0.5;

      if (both_status_good && within_distance) {
        klt_status.at(i) = 1;
      } else {
        klt_status.at(i) = 0;
      }
    }
  }

  CHECK_EQ(previous_pts.size(), current_points.size());
  CHECK_EQ(klt_status.size(), current_points.size());

  std::vector<cv::Point2f> good_current, good_previous;
  TrackletIds good_tracklets;
  
  // can also look at the err?
  for (size_t i = 0; i < klt_status.size(); i++) {
    if (klt_status[i]) {
      good_current.push_back(current_points.at(i));
      good_previous.push_back(previous_pts.at(i));
      good_tracklets.push_back(tracklet_ids.at(i));
    }
  }
  VLOG(20) << "KltFeatureTracker::trackPoints: good_current.size()=" << good_current.size();

  // Geometric verification using RANSAC
  VLOG(20) << "KltFeatureTracker::trackPoints: calling geometricVerification, good_previous.size()=" << good_previous.size() 
            << ", good_current.size()=" << good_current.size();
  const cv::Mat geometric_verification_mask =
      geometricVerification(good_previous, good_current);
  VLOG(20) << "KltFeatureTracker::trackPoints: geometricVerification completed, mask.rows=" << geometric_verification_mask.rows;
  
  VLOG(20) << "KltFeatureTracker::trackPoints: creating verified vectors";
  std::vector<cv::Point2f> verified_current, verified_previous;
  TrackletIds verified_tracklets;
  VLOG(20) << "KltFeatureTracker::trackPoints: iterating over geometric_verification_mask, rows=" << geometric_verification_mask.rows;
  for (int i = 0; i < geometric_verification_mask.rows; ++i) {
    if (geometric_verification_mask.at<uchar>(i)) {
      verified_current.push_back(good_current.at(i));
      verified_previous.push_back(good_previous.at(i));
      verified_tracklets.push_back(good_tracklets.at(i));
    }
  }
  VLOG(20) << "KltFeatureTracker::trackPoints: verified vectors populated, verified_current.size()=" << verified_current.size();

  CHECK_EQ(verified_tracklets.size(), verified_current.size());
  VLOG(20) << "KltFeatureTracker::trackPoints: verified size check passed";

  // add to tracked features
  VLOG(20) << "KltFeatureTracker::trackPoints: starting to add tracked features, verified_tracklets.size()=" << verified_tracklets.size();
  for (size_t i = 0; i < verified_tracklets.size(); i++) {
    VLOG(25) << "KltFeatureTracker::trackPoints: processing verified feature " << i << " of " << verified_tracklets.size();
    TrackletId tracklet_id = verified_tracklets.at(i);
    VLOG(25) << "KltFeatureTracker::trackPoints: tracklet_id=" << tracklet_id;

    VLOG(25) << "KltFeatureTracker::trackPoints: getting previous_feature by tracklet_id";
    const Feature::Ptr previous_feature =
        previous_features.getByTrackletId(tracklet_id);
    VLOG(25) << "KltFeatureTracker::trackPoints: got previous_feature, ptr=" << (void*)previous_feature.get();
    
    if (!previous_feature) {
      LOG_EVERY_N(ERROR, 100) << "KltFeatureTracker::trackPoints: previous_feature is null for tracklet_id=" << tracklet_id;
      continue;
    }
    
    // TODO: check this is the same as the previos kp to guarnatee order?
    VLOG(25) << "KltFeatureTracker::trackPoints: checking if previous_feature is usable";
    if (!previous_feature->usable()) {
      VLOG(25) << "KltFeatureTracker::trackPoints: previous_feature is not usable for tracklet_id=" << tracklet_id;
      continue;
    }
    VLOG(25) << "KltFeatureTracker::trackPoints: previous_feature is usable";

    VLOG(25) << "KltFeatureTracker::trackPoints: getting verified_current point at index " << i;
    const cv::Point2f kp_cv = verified_current.at(i);
    VLOG(25) << "KltFeatureTracker::trackPoints: kp_cv=(" << kp_cv.x << ", " << kp_cv.y << ")";
    
    Keypoint kp(static_cast<double>(kp_cv.x), static_cast<double>(kp_cv.y));
    VLOG(25) << "KltFeatureTracker::trackPoints: created Keypoint";

    const int x = functional_keypoint::u(kp);
    const int y = functional_keypoint::v(kp);
    VLOG(25) << "KltFeatureTracker::trackPoints: x=" << x << ", y=" << y;

    VLOG(25) << "KltFeatureTracker::trackPoints: checking motion_mask, size=" << motion_mask.size();
    // Bounds check before accessing motion_mask
    if (y < 0 || y >= motion_mask.rows || x < 0 || x >= motion_mask.cols) {
      VLOG(25) << "KltFeatureTracker::trackPoints: keypoint out of bounds (x=" << x << ", y=" << y 
                   << "), motion_mask size=" << motion_mask.size() << ", skipping";
      continue;
    }
    const ObjectId instance_label = motion_mask.at<ObjectId>(y, x);
    if (instance_label != background_label) {
      VLOG(25) << "KltFeatureTracker::trackPoints: motion_mask label is not background, skipping";
      continue;
    }
    VLOG(25) << "KltFeatureTracker::trackPoints: motion_mask label is background";

    VLOG(25) << "KltFeatureTracker::trackPoints: checking if keypoint is contained and within shrunken image";
    if (!(camera_->isKeypointContained(kp) && isWithinShrunkenImage(kp))) {
      VLOG(25) << "KltFeatureTracker::trackPoints: keypoint not contained or not within shrunken image, skipping";
      continue;
    }
    VLOG(25) << "KltFeatureTracker::trackPoints: keypoint is valid";
    
    VLOG(25) << "KltFeatureTracker::trackPoints: calling constructStaticFeatureFromPrevious";
    Feature::Ptr feature = constructStaticFeatureFromPrevious(
        kp, previous_feature, tracklet_id, frame_k);
    VLOG(25) << "KltFeatureTracker::trackPoints: constructStaticFeatureFromPrevious returned, ptr=" << (void*)feature.get();
    
    if (feature) {
      VLOG(25) << "KltFeatureTracker::trackPoints: adding feature to tracked_features";
      tracked_features.add(feature);
      VLOG(25) << "KltFeatureTracker::trackPoints: feature added";
    } else {
      VLOG(25) << "KltFeatureTracker::trackPoints: constructStaticFeatureFromPrevious returned nullptr";
    }
  }
  VLOG(20) << "KltFeatureTracker::trackPoints: finished adding tracked features";

  // Get the outliers associated with the previous_features container by taking
  // the set difference between the verified and total tracklets NOTE: verified
  // tracklets are not necessary the same as the tracklets in tracked_features
  // as tracked features may excluse some features (e.g. if not in the shrunken
  // image) or (will eventually) have new tracklets after a new detection takes
  // place we just want the set difference between the original features and
  // ones we KNOW are outliers
  {
    utils::ChronoTimingStats timer("static_feature_track.find_outliers");
    determineOutlierIds(verified_tracklets, tracklet_ids,
                        outlier_previous_features);
  }

  const auto& n_tracked = tracked_features.size();
  tracker_info.static_track_optical_flow = n_tracked;

  // Always detect edges if edge features are enabled, even if we have enough point features
  // FineTracker needs edge features every frame for pose refinement
  bool need_feature_detection = tracked_features.size() <
      static_cast<size_t>(params_.min_features_per_frame);
  bool need_edge_detection = FLAGS_use_edge_feature;
  
  VLOG(10) << "KltFeatureTracker::trackPoints: need_feature_detection=" << need_feature_detection 
            << ", need_edge_detection=" << need_edge_detection 
            << ", FLAGS_use_edge_feature=" << FLAGS_use_edge_feature
            << ", tracked_features.size()=" << tracked_features.size()
            << ", min_features_per_frame=" << params_.min_features_per_frame;

  if (need_feature_detection || need_edge_detection) {
    utils::ChronoTimingStats timer("static_feature_track.detect");
    
    if (need_feature_detection) {
      // if we do not have enough features, detect more on the current image
      VLOG(10) << "KltFeatureTracker::trackPoints: calling detectFeatures";
      detectFeatures(current_processed_img, image_container, tracked_features,
                     tracked_features, new_edges, detection_mask);
      VLOG(10) << "KltFeatureTracker::trackPoints: after detectFeatures, new_edges.size()=" << new_edges.size();
      tracker_info.new_static_detections = true;
      const auto n_detected = tracked_features.size() - n_tracked;
      tracker_info.static_track_detections += n_detected;
    } else {
      // Only detect edges without detecting new point features
      // This ensures edges are detected every frame for FineTracker
      VLOG(10) << "KltFeatureTracker::trackPoints: calling detectEdges";
      std::vector<Edge> detected_edges =
          detectEdgeFeatures(current_processed_img, tracked_features.size(),
                             cv::Mat(), image_container.depth());
      VLOG(10) << "KltFeatureTracker::trackPoints: detectEdges returned " << detected_edges.size() << " edges";
      // Add detected edges to new_edges container
      for (const Edge& edge : detected_edges) {
        new_edges.add(edge);
      }
      VLOG(10) << "KltFeatureTracker::trackPoints: after adding edges, new_edges.size()=" << new_edges.size();
    }
  } else {
    VLOG(10) << "KltFeatureTracker::trackPoints: skipping edge detection (need_feature_detection=" 
              << need_feature_detection << ", need_edge_detection=" << need_edge_detection << ")";
  }

  return true;
}

cv::Mat KltFeatureTracker::geometricVerification(
    const std::vector<cv::Point2f>& good_old,
    const std::vector<cv::Point2f>& good_new) const {
  if (good_old.size() >= 4) {  // Minimum number of points required for RANSAC
    cv::Mat mask;
    cv::findHomography(good_old, good_new, cv::RANSAC, 5.0, mask);
    return mask;
  } else {
    return cv::Mat::ones(
        good_old.size(), 1,
        CV_8U);  // If not enough points, assume all are inliers
  }
}

Feature::Ptr KltFeatureTracker::constructStaticFeatureFromPrevious(
    const Keypoint& kp_current, Feature::Ptr previous_feature,
    const TrackletId tracklet_id, const FrameId frame_id) const {
  CHECK(previous_feature);
  CHECK_EQ(previous_feature->trackletId(), tracklet_id);

  size_t age = previous_feature->age();
  age++;

  TrackletId tracklet_to_use = tracklet_id;
  // if age is too large, or age is zero, retrieve new tracklet id
  if (age > params_.max_feature_track_age) {
    // TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();
    // tracklet_to_use = tracked_id_manager.getTrackletIdCount();
    // tracked_id_manager.incrementTrackletIdCount();
    // age = 0u;
    return nullptr;
  }

  // update previous keypoint
  previous_feature->measuredFlow(kp_current - previous_feature->keypoint());
  // This is so awful, but happens becuase the way the code was originally
  // written, we expect flow from k to k+1 (grrrr)
  previous_feature->predictedKeypoint(kp_current);

  Feature::Ptr feature = std::make_shared<Feature>();
  (*feature)
      .objectId(background_label)
      .frameId(frame_id)
      .keypointType(KeyPointType::STATIC)
      .age(age)
      .markInlier()
      .trackletId(tracklet_to_use)
      .keypoint(kp_current);

  return feature;
}

Feature::Ptr KltFeatureTracker::constructNewStaticFeature(
    const Keypoint& kp_current, const FrameId frame_id) const {
  static const auto kAge = 0u;

  TrackletIdManager& tracked_id_manager = TrackletIdManager::instance();
  TrackletId tracklet_to_use = tracked_id_manager.getAndIncrementTrackletId();

  Feature::Ptr feature = std::make_shared<Feature>();
  (*feature)
      .objectId(background_label)
      .frameId(frame_id)
      .keypointType(KeyPointType::STATIC)
      .age(kAge)
      .markInlier()
      .trackletId(tracklet_to_use)
      .keypoint(kp_current);
  return feature;
}

// EdgeFeatureTracker class is currently commented out in the header
// Uncomment the following code when EdgeFeatureTracker is re-enabled
/*
EdgeFeatureTracker::EdgeFeatureTracker(const TrackerParams& params,
                                         Camera::Ptr camera,
                                         ImageDisplayQueue* display_queue)
    : StaticFeatureTracker(params, camera, display_queue) {
  detector_ = std::make_shared<SparseFeatureDetector>(
      params, FunctionalDetector::FactoryCreate(params));

  // Initialize FineTracker with camera parameters
  const auto& cam_params = camera->getParams();

  CHECK_NOTNULL(detector_);
  CHECK_NOTNULL(fine_tracker_);
}



std::vector<Edge> EdgeFeatureTracker::getDetectedEdges() const {
  return detected_edges_;
}

void EdgeFeatureTracker::equalizeImage(const ImageContainer& image_container,
                                       cv::Mat& equialized_greyscale) const {
  const ImageWrapper<ImageType::RGBMono>& rgb_wrapper = image_container.rgb();
  const cv::Mat& rgb = rgb_wrapper.toRGB();
  cv::Mat mono = ImageType::RGBMono::toMono(rgb_wrapper);
  CHECK(!mono.empty());

  mono.copyTo(equialized_greyscale);
}

std::vector<Edge> EdgeFeatureTracker::detectEdgeFeatures(
    const cv::Mat& processed_img, int number_tracked, const cv::Mat& mask) {
  std::vector<Edge> edges;
  detector_->detectEdge(processed_img, edges, mask);
  return edges;
}
*/

// bool EdgeFeatureTracker::trackEdges(
//     const cv::Mat& current_processed_img,
//     const cv::Mat& previous_processed_img, const ImageContainer& image_container,
//     const EdgeContainer& previous_edges, EdgeContainer& tracked_edges,
//     // TrackletIds& outlier_previous_features, FeatureTrackerInfo& tracker_info,
//     const cv::Mat& detection_mask, const std::optional<gtsam::Rot3>& inital_pose) {
//   if (current_processed_img.empty() || previous_processed_img.empty() ||
//       previous_edges.empty()) {
//     return false;
//   }

//   outlier_previous_features.clear();

//   const cv::Mat& motion_mask = image_container.objectMotionMask();
//   const FrameId frame_k = image_container.frameId();
//   const Timestamp timestamp = image_container.timestamp();
  
  
//   getFineSampledPoints(previous_edges, params_.fine.geo_photo_ratio);
//   // Get edges from previous frame 
//   std::vector<Edge> prev_edges;
//   prev_edges.reserve(previous_edges.size());
//   for (const Edge& edge : previous_edges) {
//     prev_edges.push_back(edge);
//   }
//   CHECK_EQ(prev_edges.size(), previous_edges.size());



//   // FineTracker::estimate implementation
//   // This replaces the estimate() function call with its logic directly in trackEdges
//   if (fine_tracker_ && !prev_edges.empty() && !current_edges.empty()) {
//     // Create temporary Frame objects for FineTracker
//     // Note: FineTracker requires FramePtr, so we need to create minimal Frame objects
//     // with edges and images
    
//     // Create reference frame (previous frame) with edges
//     FeatureContainer empty_features_ref, empty_features_curr;
//     std::map<ObjectId, SingleDetectionResult> empty_observations;
    
//     // Convert edges to Frame-compatible format
//     // Note: FineTracker expects Frame objects with mvEdges and mMatGray
//     // We need to create Frame objects that contain the edges
    
//     // For now, we'll use FineTracker's estimate logic directly
//     // Set reference and current frames if we have previous frame info
//     // This is a simplified version - full implementation would require
//     // proper Frame object creation with edges
    
//     // FineTracker::estimate logic (from FineTracker::estimate):
//     //   1. assert(mpF_ref != nullptr && mpF_cur != nullptr);
//     //   2. associationRef2CurParallel();
//     //   3. if(geo_photo_ratio > 0) {
//     //          RegistrationCombinedParallel();
//     //      } else {
//     //          RegistrationGeometricParallel();
//     //      }
//     //   4. T21 = T_cur_ref.inverse();
    
//     // Since FineTracker requires FramePtr, we need to create Frame objects
//     // with edges and images. For now, we'll call FineTracker::estimate
//     // if we have the necessary frame information.
    
//     // TODO: Create proper Frame objects with edges for FineTracker
//     // Once Frame objects are created, we can call:
//     //   fine_tracker_->setReference(frame_ref);
//     //   fine_tracker_->setCurrent(frame_curr);
//     //   Sophus::SE3d T_ref_cur;
//     //   fine_tracker_->estimate(T_ref_cur, true);
    
//     // For now, this is a placeholder that shows where estimate() logic would go
//     // The actual implementation would require creating Frame objects with edges
//   }

//   // Update tracked edges based on edge associations
//   // This would use the results from FineTracker's estimate
  
//   tracker_info.static_track_optical_flow = tracked_edges.size();


//   return true;
// }

}  // namespace dyno
