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

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"

#include <glog/logging.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <unordered_set>

#include <opencv4/opencv2/opencv.hpp>
#include "dynosam/frontend/FrontendModuleAccessor.hpp"
#include "dynosam/visualizer/VoViewer.hpp"

#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam_common/Flags.hpp"  //for common flags
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/utils/SafeCast.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
// #include "dynosam_common/EdgeSelector.hpp"
#include "dynosam_cv/RGBDCamera.hpp"
#include "dynosam/visualizer/EdgeVizUtils.hpp"
#include "dynosam/frontend/vision/Object.hpp"


DEFINE_bool(use_frontend_logger, false,
            "If true, the frontend logger will be used");

DEFINE_bool(use_edge_feature, true,
            "If true, the edge detection will be used");

DEFINE_bool(use_dynamic_track, true,
            "If true, the dynamic tracking will be used");

DEFINE_bool(use_static_track, true,
            "If true, the static feature tracking will be used");

DEFINE_bool(log_projected_masks, false,
            "If true, projected masks will be saved at every frame");

DEFINE_bool(set_dense_labelled_cloud, false,
            "If true, the dense labelled point cloud will be set");

DEFINE_bool(use_object_motion_filtering, false, "For testing!");

DEFINE_bool(use_object, false,
            "If true, object detection will be enabled");

DEFINE_bool(use_edge_selector_track, false,
            "If true, use edgeSelector.processImage() + direct KeyFrame creation "
            "instead of tracker_->track() (localmapping.cc style)");

DEFINE_string(output_edge_kf_trajectory, "",
              "If non-empty, append edge keyframe poses to this file in TUM format "
              "(timestamp tx ty tz qx qy qz qw). This logs keyframes across the entire run, "
              "not just the current sliding window.");

namespace {
inline void ensureParentDirExists(const std::string& filename) {
  try {
    const std::filesystem::path out_path(filename);
    const auto parent = out_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to create parent directory for: " << filename
                 << " error=" << e.what();
  }
}

inline void writeTumPoseLine(std::ostream& os, double timestamp,
                            const Eigen::Matrix4d& T_world_cam) {
  const double tx = T_world_cam(0, 3);
  const double ty = T_world_cam(1, 3);
  const double tz = T_world_cam(2, 3);
  const Eigen::Matrix3d R = T_world_cam.block<3, 3>(0, 0);
  const Eigen::Quaterniond q(R);
  os << std::fixed << std::setprecision(6) << timestamp << " "
     << tx << " " << ty << " " << tz << " "
     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
}
}  // namespace

namespace dyno {

// Helper function matching coarseTracking.cpp generateSrcPixelsSampled
static double degreesToPiRange(double degrees) {
    assert(degrees >= 0 && degrees < 360.0);
    // Map to [-π, π]
    if (degrees <= 180) {
        return degrees * M_PI / 180.0; // 0~180 → 0~π
    } else {
        return (degrees - 360) * M_PI / 180.0; // 180~360 → -π~0
    }
}

static void generateSrcPixelsSampled(const Frame::Ptr& frame_cur,
                                     std::vector<float>& edge_point_total_x,
                                     std::vector<float>& edge_point_total_y,
                                     std::vector<float>& edge_depth_total,
                                     std::vector<float>& edge_weight_total,
                                     std::vector<cv::Point3f>& cloud_total,
                                     std::vector<float>& edge_point_total_theta,
                                     int sample_bias,
                                     int maximum_point) {
    edge_point_total_x.clear();
    edge_point_total_y.clear();
    edge_depth_total.clear();
    edge_weight_total.clear();
    cloud_total.clear();
    edge_point_total_theta.clear();
    
    std::vector<orderedEdgePoint> sampledPoints = frame_cur->getCoarseSampledPoints(sample_bias, maximum_point);
    for(size_t i = 0; i < sampledPoints.size(); ++i) {
        const orderedEdgePoint& pt = sampledPoints[i];
        float angle = pt.imgGradAngle;
        angle = static_cast<float>(degreesToPiRange(angle));
        float depth = pt.depth;
        float weight = pt.score_depth;
        cv::Point3f cloud_pt = cv::Point3f(pt.x_3d, pt.y_3d, pt.z_3d);
        edge_depth_total.push_back(depth);
        edge_weight_total.push_back(weight);
        edge_point_total_x.push_back(pt.x);
        edge_point_total_y.push_back(pt.y);
        cloud_total.push_back(cloud_pt);
        edge_point_total_theta.push_back(angle);
    }
}

RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(
    const DynoParams& params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : FrontendModule(params, display_queue),
      camera_(camera),
      motion_solver_(params.frontend_params_.ego_motion_solver_params,
                     camera->getParams()),
      imu_frontend_(params.frontend_params_.imu_params) {
  CHECK_NOTNULL(camera_);
  // Log loaded Canny parameters from YAML
  const auto& tracker_params = getFrontendParams().tracker_params;
  // LOG(INFO) << "Loaded Canny params from YAML: coarse.cannyLow=" 
  //           << tracker_params.edge_coarse.cannyLow 
  //           << ", coarse.cannyHigh=" << tracker_params.edge_coarse.cannyHigh;
  
  tracker_ = std::make_unique<FeatureTracker>(getFrontendParams(), camera_,
                                              display_queue);
  fine_tracker_ = std::make_unique<fine::FineTracker>(camera->getParams().fx(), camera->getParams().fy(), camera->getParams().cu(), camera->getParams().cv(), getFrontendParams().tracker_params.edge_fine.geo_photo_ratio);
  // LOG(INFO) << "height: " << camera->getParams().ImageHeight() << " width: " << camera->getParams().ImageWidth();
  // LOG(INFO) << "fx: " << camera->getParams().fx() << " fy: " << camera->getParams().fy() << " cx: " << camera->getParams().cu() << " cy: " << camera->getParams().cv();
  direct_tracker_ = std::make_unique<direct::DirectTracker>(camera->getParams().ImageWidth(), camera->getParams().ImageHeight(), camera->getParams().fx(), camera->getParams().fy(), camera->getParams().cu(), camera->getParams().cv());
  // edge_selector_ = std::make_unique<edgeSelector>(20.0, tracker_params.edge_coarse.cannyLow, tracker_params.edge_coarse.cannyHigh);

  object_matcher_ = std::make_unique<ObjectMatcher>(camera_->getParams());


  // Initialize edge-based local mapping (from localmapping.cc)
  local_map_.reset(new dyno::localMap());
  map_.reset(new dyno::EdgeMap());

  // Initialize CLIP feature client (connects to Python server on tcp://localhost:5555)
  // NOTE: CLIP-based tracking/re-association is currently disabled in this build.
  // clip_client_ = std::make_shared<dyno::ClipFeatureClient>("tcp://localhost:5555", 200);
  // TODO: Load canny parameters from config
  // edge_selector_ = std::make_unique<edgeSelector>(20.0, 50, 150);
  
  // Load sliding window parameters from config
  const auto& win_params = getFrontendParams().tracker_params.edge_win;
  window_size_ = win_params.window_size;
  window_step_ = win_params.window_step;
  kf_trans_thres_ = static_cast<float>(win_params.kf_trans_thres);
  kf_rot_thres_ = static_cast<float>(win_params.kf_rot_thres);
  
  // Load object detection confidence threshold from config
  kMinConfidenceScore_ = getFrontendParams().min_confidence_score;

  // Set camera intrinsics for edge_viz utility (copied from dyno_sam.cc)
  edge_viz::setCameraParams(camera->getParams().fx(), camera->getParams().fy(),
                            camera->getParams().cu(), camera->getParams().cv());
  
  if (FLAGS_use_frontend_logger) {
    LOG(INFO) << "Using front-end logger!";
    logger_ = std::make_unique<RGBDFrontendLogger>();
  }

  if (FLAGS_use_object_motion_filtering) {
    ObjectMotionSolverFilter::Params filter_params;
    object_motion_solver_ = std::make_shared<ObjectMotionSolverFilter>(
        filter_params, camera->getParams());
  } else {
    ObjectMotionSovlerF2F::Params object_motion_solver_params =
        getFrontendParams().object_motion_solver_params;
    // add ground truth hook
    object_motion_solver_params.ground_truth_packets_request = [&]() {
      return this->shared_module_info.getGroundTruthPackets();
    };
    object_motion_solver_params.refine_motion_with_3d = false;
    object_motion_solver_ = std::make_shared<ObjectMotionSovlerF2F>(
        object_motion_solver_params, camera->getParams());
  }
  
  // Start background processing thread for keyframe queue
  processing_running_ = true;
  processing_thread_ = std::thread(&RGBDInstanceFrontendModule::processingThreadFunction, this);

  // Initialize visualization caches
  {
    std::lock_guard<std::mutex> lock(local_map_mutex_);
    cluster_clouds_cache_ =
        std::make_shared<const std::vector<std::vector<cv::Point3d>>>(
            std::vector<std::vector<cv::Point3d>>{});
    cluster_colors_cache_ =
        std::make_shared<const std::vector<cv::Vec3b>>(
            std::vector<cv::Vec3b>{});
    local_map_clouds_cache_ =
        std::make_shared<const std::vector<std::vector<cv::Point3d>>>(
            std::vector<std::vector<cv::Point3d>>{});
    environment_cloud_cache_ =
        std::make_shared<const std::vector<EnvironmentCloudFrame>>(
            std::vector<EnvironmentCloudFrame>{});
    environment_frames_.clear();
    // Initialize visualization data (will be updated every frame)
    latest_visualization_data_ = nullptr;
  }
  
  // Initialize last keyframe cache
  {
    std::lock_guard<std::mutex> lock(last_kf_mutex_);
    last_keyframe_ = nullptr;
  }
}

RGBDInstanceFrontendModule::~RGBDInstanceFrontendModule() {
  // Stop processing thread
  processing_running_ = false;
  // Push nullptr to wake up thread
  KeyFramePtr null_kf = nullptr;
  optimization_queue_.push(null_kf);
  
  if (processing_thread_.joinable()) {
    processing_thread_.join();
  }
  
  if (FLAGS_save_frontend_json) {
    LOG(INFO) << "Saving frontend output as json";
    const std::string file_path =
        getOutputFilePath(kRgbdFrontendOutputJsonFile);
    // JsonConverter::WriteOutJson(output_packet_record_, file_path,
    //                             JsonConverter::Format::BSON);
  }

}

FrontendModule::ImageValidationResult
RGBDInstanceFrontendModule::validateImageContainer(
    const ImageContainer::Ptr& image_container) const {
  return ImageValidationResult(image_container->hasDepth(),
                               "Depth is required");
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::boostrapSpin(
    FrontendInputPacketBase::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;


  // Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
  //                                    *image_container);

  // Process image with edgeSelector
  const ImageWrapper<ImageType::RGBMono>& rgb_wrapper = image_container->rgb();
  cv::Mat grayscale_image = ImageType::RGBMono::toMono(rgb_wrapper);
  // edge_selector_->processImage(grayscale_image);
  
  // Still need to track frame for other processing
  Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                                     *image_container);
  CHECK(frame->updateDepths());
  // LOG(INFO) << "frame edge features: " << frame->static_edges_.size()
  //           << ", edgeSelector edges: " << edge_selector_->mvEdges.size();

  // Initialize first frame pose to identity
  frame->T_world_camera_ = gtsam::Pose3::Identity();

  // Create first keyframe if edge features are enabled
  bool created_edge_kf_this_frame = false;
  if (FLAGS_use_edge_feature && !frame->static_edges_.empty()) {
    const gtsam::Pose3& pose_curr = frame->T_world_camera_;
    
    // Get fine sampled points
    const int sample_bias = static_cast<int>(getFrontendParams().tracker_params.edge_fine.sample_bias);
    const int maximum_point = getFrontendParams().tracker_params.edge_coarse.maximum_point;

    frame->getFineSampledPoints(sample_bias);

    std::vector<float> x_list, y_list, depth_list, weight_list, theta_list;
    std::vector<cv::Point3f> frame_cloud_ref;  // Not used but required by function signature
    
    // Get coarse sampling parameters from config
    generateSrcPixelsSampled(frame, x_list, y_list, depth_list, weight_list, 
                            frame_cloud_ref, theta_list, sample_bias, maximum_point);
    if(x_list.size()>200) is_data_valid_ = true;

    // Set reference frame
    const ImageWrapper<ImageType::RGBMono>& rgb_wrapper_ref = frame->image_container_.rgb();
    cv::Mat mono_ref = ImageType::RGBMono::toMono(rgb_wrapper_ref);
    direct_tracker_->setReference(mono_ref, x_list, y_list, depth_list, weight_list, theta_list);
    direct_tracker_->setReferenceFrameId(frame->getFrameId());
    fine_tracker_->setCurrent(frame);

    
    // Create first keyframe
    KeyFramePtr pKF = createKeyFrameFromFrame(frame, pose_curr);
    
    if (pKF) {
      // Initialize objects after the first KeyFrame exists so we can gate with edge-point support.
      ObjectsInitialization(frame, pKF.get());

      // Add to local map
      {
        std::lock_guard<std::mutex> lock(local_map_mutex_);
        local_map_->addFrame2LocalMap(pKF);
        
        // Update cached last keyframe Frame (store Frame object, not KeyFrame)
        {
          std::lock_guard<std::mutex> kf_lock(last_kf_mutex_);
          last_keyframe_ = frame;
        }
      }
      
      // Push to queue for async processing (though it won't optimize until window is full)
      optimization_queue_.push(pKF);
      
      pose_last_edge_kf_ = pose_curr;
      is_edge_initialized_ = true;
      
      // LOG(INFO) << "Created first edge keyframe: kf_id=" << pKF->KF_ID 
      //           << ", frame_id=" << frame->getFrameId();
    }
  }

  return {State::Nominal, nullptr};
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {
  const auto t_nominal_start = std::chrono::steady_clock::now();
  ImageContainer::Ptr image_container = input->image_container_;

  const bool has_imu = input->imu_measurements.has_value();
  const bool has_stereo = image_container->hasRightRgb();

  //! Rotation from k-1 to k in k-1
  std::optional<gtsam::Rot3> R_curr_ref;
  ImuFrontend::PimPtr pim;
  if (has_imu) {
    pim = imu_frontend_.preintegrateImuMeasurements(
        input->imu_measurements.value());

    nav_state_curr_ =
        pim->predict(nav_state_prev_, gtsam::imuBias::ConstantBias{});
    // nav_state_curr_ =
    //     pim->predict(nav_state_last_kf_, gtsam::imuBias::ConstantBias{});
    last_imu_k_ = input->getFrameId();

    // relative rotation
    R_curr_ref =
        nav_state_prev_.attitude().inverse() * nav_state_curr_.attitude();
  }

  const auto ms = [](const auto& a, const auto& b) -> double {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  const auto t_track_start = std::chrono::steady_clock::now();
  Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                                     *image_container, R_curr_ref);
  const auto t_track_end = std::chrono::steady_clock::now();

  Frame::Ptr previous_frame = tracker_->getPreviousFrame();
  CHECK(previous_frame);
  // const FeatureTrackerInfo& tracker_info = tracker_->getTrackerInfo();
  // VLOG(1) << to_string(tracker_info);

  const auto t_update_depths_start = std::chrono::steady_clock::now();
  {
    // this will mark some points as invalid if they are out of depth range
    utils::ChronoTimingStats update_depths_timer("depth_updater");
    frame->updateDepths();
    
    // Also update depths for previous frame if it doesn't have them
    // This is important when using ApproximateTime sync where timestamps may differ
    // updateDepths() internally checks if depth is available, so we can call it safely
    if (previous_frame) {
      previous_frame->updateDepths();
    }
  }
  const auto t_update_depths_end = std::chrono::steady_clock::now();


  const auto t_sample0 = std::chrono::steady_clock::now();
  const int sample_bias = static_cast<int>(getFrontendParams().tracker_params.edge_fine.sample_bias);
  frame->getFineSampledPoints(sample_bias);
  const auto t_sample1 = std::chrono::steady_clock::now();
  

  bool stereo_result = false;
  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();

  if (has_stereo) {
    CHECK(rgbd_camera) << "Stereo imagery provided but rgbd camera is null!";
  }

  const auto t_stereo1_start = std::chrono::steady_clock::now();
  if (has_stereo) {
    const cv::Mat& left_rgb = image_container->rgb();
    const cv::Mat& right_rgb = image_container->rightRgb();

    FeaturePtrs stereo_features_1;
    stereo_result =
        tracker_->stereoTrack(stereo_features_1, frame->static_features_,
                              left_rgb, right_rgb, rgbd_camera->baseline());
  }
  const auto t_stereo1_end = std::chrono::steady_clock::now();
  //  // this includes the refine correspondances with joint optical flow
  // // TODO: lots of internal logic around how the actual pose gets predicted.
  // // should streamline this and tell backend how pose was selected!!
  // if (!solveCameraMotion(frame, previous_frame, R_curr_ref)) {
  //   LOG(ERROR) << "Could not solve for camera";
  // }

  
  // TODO: Uncomment below to use IMU prediction or constant velocity model as initial pose
  // gtsam::Pose3 T_k_1_k_initial;
  // if (last_imu_k_ == frame->getFrameId() && has_imu) {
  //   // Use IMU prediction if available
  //   const gtsam::Pose3 T_world_k_1 = nav_state_prev_.pose();
  //   const gtsam::Pose3 T_world_k = nav_state_curr_.pose();
  //   T_k_1_k_initial = T_world_k_1.inverse() * T_world_k;
  //   VLOG(10) << "RGBDInstanceFrontendModule::nominalSpin: using IMU prediction for initial pose";
  // } else {
  //   // Use constant velocity model
  //   T_k_1_k_initial = vo_velocity_;
  //   VLOG(10) << "RGBDInstanceFrontendModule::nominalSpin: using constant velocity model for initial pose";
  // }
  
  // Get last keyframe pose (if available) to use as reference instead of previous_frame
  // Use cached value to avoid frequent lock
  gtsam::Pose3 pose_ref;  // Reference pose for computing current frame pose
  {
    std::lock_guard<std::mutex> lock(last_kf_mutex_);
    if (last_keyframe_) {
        // Use last keyframe Frame pose as reference
        pose_ref = last_keyframe_->T_world_camera_;
        
    } else {
      // No keyframe yet, use previous_frame pose
      LOG(ERROR) << "No keyframe yet, using previous_frame pose";
    }
  }
  
  gtsam::Pose3 pose_cur_initial = gtsam::Pose3();  // Identity pose
  
  // Perform DirectTrack
  const auto t_direct_start = std::chrono::steady_clock::now();
  gtsam::Pose3 pose_cur_refined;
  if (!DirectTrack(frame, previous_frame, pose_cur_initial, pose_cur_refined)) {
    LOG(ERROR) << "DirectTrack failed, using identity pose";
  }
  const auto t_direct_end = std::chrono::steady_clock::now();
  
  // Log pose_ref in TUM format
  const gtsam::Point3& t = pose_ref.translation();
  const gtsam::Rot3& R = pose_ref.rotation();
  const gtsam::Quaternion q = R.toQuaternion();
  const double timestamp = last_keyframe_->getTimestamp();
  
  // Update frame pose with DirectTrack result
  frame->T_world_camera_ = pose_ref * pose_cur_refined;
  
  // Calculate relative pose from DirectTrack result for FineTrack (if edge features enabled)
  const auto t_fine_start = std::chrono::steady_clock::now();
  if (FLAGS_use_edge_feature) {
    gtsam::Pose3 pose_cur_fine_refined;
    if (!FineTrack(frame, pose_cur_refined, pose_cur_fine_refined)) {
      VLOG(5) << "Could not fine track";
    } else {
      // Check for pose jump 
      // Reject fine tracking result if it shows unreasonable motion (prevents trajectory flips)
      const gtsam::Matrix3& R = pose_cur_fine_refined.rotation().matrix();
      const gtsam::Point3& t = pose_cur_fine_refined.translation();
      const double t_thres = 0.10;  // meters
      const double angle_thres = 15.0;  // degrees
      
      Eigen::AngleAxisd rotation_vector(R);
      const double angle_deg = rotation_vector.angle() * 180.0 / M_PI;
      const double translation_norm = t.norm();
      
      if (translation_norm > t_thres || angle_deg > angle_thres) {
        LOG_EVERY_N(WARNING, 50) << "FineTrack pose jump detected: trans=" << translation_norm 
                                  << "m, angle=" << angle_deg << "deg, using DirectTrack result";
        // Use DirectTrack result with reference pose (T_ref_prev already computed above)
        frame->T_world_camera_ = pose_ref * pose_cur_refined;
      } else {
        // Update frame pose with refined result from FineTrack
        frame->T_world_camera_ = pose_ref * pose_cur_fine_refined;
        // T_w_ref * T_ref_prev * T_prev_cur = T_w_cur
      }

      frame->T_world_camera_ = pose_ref * pose_cur_fine_refined;
    }
  }
  const auto t_fine_end = std::chrono::steady_clock::now();

  // Run object association only on edge keyframe timing.
  // If edge feature is disabled, keep the original per-frame behavior.
  bool run_object_association_this_frame = true;
  if (FLAGS_use_edge_feature) {
    bool local_map_empty_for_assoc = false;
    {
      std::lock_guard<std::mutex> lock(local_map_mutex_);
      local_map_empty_for_assoc = local_map_->mvKeyFrames.empty();
    }
    if (local_map_empty_for_assoc || !is_edge_initialized_) {
      run_object_association_this_frame = true;
    } else {
      run_object_association_this_frame =
          shouldAddEdgeKeyFrame(frame->T_world_camera_, pose_last_edge_kf_);
    }
  }

  if (FLAGS_use_object && run_object_association_this_frame) {
    const int img_width = camera_->getParams().ImageWidth();
    const int img_height = camera_->getParams().ImageHeight();
    dyno::BBox2 img_bbox(0.0, 0.0,
                         static_cast<double>(img_width),
                         static_cast<double>(img_height));
    // Use dyno::Object* for projections (matches ObjectMatcher API)
    std::unordered_map<dyno::Object*, Ellipse> proj_bboxes;

    // Compute projection matrix P = K * [R_cw | t_cw]
    // NOTE: T_world_camera_ is a pose that takes camera coords -> world coords (T_wc),
    // but OA-SLAM's ellipsoid math expects a world->camera transform [R_cw | t_cw].
    // So we must invert T_world_camera_ before building Rt and P.
    const gtsam::Pose3& T_wc = frame->T_world_camera_;
    const gtsam::Pose3  T_cw = T_wc.inverse();

    const gtsam::Matrix3& R_cw = T_cw.rotation().matrix();
    const gtsam::Point3&  t_cw = T_cw.translation();
    
    // Get camera intrinsic matrix K (Eigen 3x3)
    const gtsam::Matrix3& K = camera_->getParams().getCameraMatrixEigen();
    
    // Construct [R_cw | t_cw] (3x4 matrix, world -> camera)
    Eigen::Matrix<double, 3, 4> Rt;
    Rt.block<3, 3>(0, 0) = R_cw;
    Rt.block<3, 1>(0, 3) = t_cw;
    
    // Compute projection matrix P = K * [R_cw | t_cw]
    Eigen::Matrix<double, 3, 4> P = K * Rt;

    // Debug: Log projection matrix components
    // VLOG(1) << "[RGBDFrontend] Frame " << frame->getFrameId() 
    //         << " - T_world_camera translation: [" << t.transpose() << "]";
    // VLOG(1) << "[RGBDFrontend] Frame " << frame->getFrameId()
    //         << " - T_world_camera rotation (euler ZYX): [" 
    //         << frame->T_world_camera_.rotation().yaw() << ", "
    //         << frame->T_world_camera_.rotation().pitch() << ", "
    //         << frame->T_world_camera_.rotation().roll() << "]";
    // VLOG(1) << "[RGBDFrontend] Frame " << frame->getFrameId()
    //         << " - K matrix: fx=" << K(0,0) << ", fy=" << K(1,1) 
    //         << ", cx=" << K(0,2) << ", cy=" << K(1,2);

    // ── Object association: edge-projection (primary) + bbox IoU (fallback) ──
    //
    // Primary  : project object's associated 3D edge points into current frame,
    //            count overlap with detection mask.  ratio >= 60% → match.
    // Fallback : if object has too few edge points (sparse early on), use
    //            bbox IoU between object's last observed bbox and current
    //            detection bbox.  iou >= 0.3 → match.
    //
    // Whichever method scores highest wins (per detection node).
    constexpr float kEdgeRatioTh  = 0.5f;  // edge-projection mask overlap threshold
    constexpr int   kMinEdgePts   = 10;    // min projected inliers for edge method
    constexpr float kBboxIoUTh    = 0.3f;  // bbox IoU threshold for fallback
    // score encoding: edge ratio stored in [0,1], bbox iou stored as -iou so
    // edge match always preferred over bbox match
    // (we use a single best_score per node; edge wins if > 0, bbox if < 0)

    const cv::Mat& motion_mask = frame->image_container_.objectMotionMask();
    const bool has_mask = !motion_mask.empty() &&
                          motion_mask.type() == CV_32SC1 &&
                          motion_mask.size() == cv::Size(img_width, img_height);

    if (map_ && frame->graph) {
      // Pre-compute target mask value per detection (packed BGR from label)
      struct DetInfo { int mask_val; int label; BBox2 bbox; };
      std::map<int, DetInfo> det_info;
      for (const auto& [nid, attr] : frame->graph->attributes) {
        const int lbl = attr.label;
        const unsigned char tb = static_cast<unsigned char>((lbl * 37) % 256);
        const unsigned char tg = static_cast<unsigned char>((lbl * 17) % 256);
        const unsigned char tr = static_cast<unsigned char>((lbl * 97) % 256);
        det_info[nid] = {
          (static_cast<int>(tb) << 16) | (static_cast<int>(tg) << 8) | static_cast<int>(tr),
          lbl, attr.bbox
        };
      }

      // best_match[node_id] = {Object*, score}
      // score > 0  → edge-projection match (ratio)
      // score < 0  → bbox IoU fallback (-iou), only used if no edge match
      std::map<int, std::pair<dyno::Object*, float>> best_match;

      const std::vector<dyno::Object*> all_objs = map_->GetAllObjects();
      for (dyno::Object* obj : all_objs) {
        if (!obj || obj->isBad()) continue;
        const int obj_cat = static_cast<int>(obj->GetCategoryId());

        // ── Primary: edge-projection ────────────────────────────────────────
        const auto world_pts = obj->GetAssociatedMapPoints();
        bool edge_method_used = false;

        if (has_mask && static_cast<int>(world_pts.size()) >= kMinEdgePts) {
          // Project into current frame
          std::vector<cv::Point2i> proj_pts;
          proj_pts.reserve(world_pts.size());
          for (const auto& pw : world_pts) {
            const Eigen::Vector4d ph(pw.x(), pw.y(), pw.z(), 1.0);
            const Eigen::Vector3d pc = P * ph;
            if (pc.z() <= 0.1) continue;
            const int u = static_cast<int>(pc.x() / pc.z());
            const int v = static_cast<int>(pc.y() / pc.z());
            if (u < 0 || v < 0 || u >= img_width || v >= img_height) continue;
            proj_pts.emplace_back(u, v);
          }

          if (static_cast<int>(proj_pts.size()) >= kMinEdgePts) {
            edge_method_used = true;
            for (const auto& [nid, dinfo] : det_info) {
              if (dinfo.label != obj_cat) continue;
              int inside = 0;
              for (const auto& pt : proj_pts) {
                if (motion_mask.at<int>(pt.y, pt.x) == dinfo.mask_val) ++inside;
              }
              const float ratio = static_cast<float>(inside) /
                                  static_cast<float>(proj_pts.size());
              if (inside >= kMinEdgePts && ratio >= kEdgeRatioTh) {
                auto it = best_match.find(nid);
                // edge match (score > 0) always beats bbox match (score < 0)
                if (it == best_match.end() ||
                    it->second.second < 0 ||
                    ratio > it->second.second) {
                  best_match[nid] = {obj, ratio};
              }
              }
          }
      }
        }

        // ── Fallback: bbox IoU ───────────────────────────────────────────────
        // Only used if edge method couldn't run OR produced no match for this obj
        if (!edge_method_used) {
          // Use last observed bboxes to compute IoU with current detections
          const auto obs_bboxes = obj->GetObservedBboxes();  // most-recent first
          if (obs_bboxes.empty()) continue;
          const BBox2& last_bb = obs_bboxes.back();

          for (const auto& [nid, dinfo] : det_info) {
            if (dinfo.label != obj_cat) continue;
            // skip if already have a positive edge match for this node
            auto it = best_match.find(nid);
            if (it != best_match.end() && it->second.second > 0) continue;

            const float iou = static_cast<float>(bboxes_iou(last_bb, dinfo.bbox));
            if (iou >= kBboxIoUTh) {
              // Store as negative score so edge match can override later
              if (it == best_match.end() || iou > -it->second.second) {
                best_match[nid] = {obj, -iou};
              }
            }
          }
        }
      }

      // ── CLIP fallback for unmatched detections (disabled) ───────────────

      // Apply best matches to frame graph
      int n_edge = 0, n_bbox = 0;
      // CLIP fallback is disabled in this build.
      int n_clip = 0;
      for (auto& [nid, match] : best_match) {
        frame->graph->attributes[nid].obj = match.first;
        if (match.second > 0) ++n_edge;
        else if (match.second > -2.f) ++n_bbox;
        // else: CLIP match (counted separately)
        VLOG(3) << "[ObjAssoc] node=" << nid
                << " obj_id=" << match.first->GetId()
                << (match.second > 0 ? " edge_ratio=" :
                    match.second > -2.f ? " bbox_iou=" : " clip_sim=")
                << std::abs(match.second > -2.f ? match.second : match.second + 2.f);
      }
      VLOG(2) << "[ObjAssoc] edge=" << n_edge << " bbox=" << n_bbox
              << " clip=" << n_clip
              << " total=" << (n_edge + n_bbox + n_clip)
              << " / " << frame->graph->attributes.size();
    } else {
      // No map or no graph — fall back to Wasserstein with empty proj_bboxes
      object_matcher_->MatchObjectsWasserDistance(*frame, proj_bboxes, P);
    }




  }
  // Update visualization snapshot every frame
  {
    EdgeVisualizationDataPtr snap = std::make_shared<EdgeVisualizationData>();
    snap->currentFramePose = frame->T_world_camera_.matrix();

    // Read caches + sliding window under the local_map_mutex_
    std::lock_guard<std::mutex> lock(local_map_mutex_);

    // Update cluster cache if it's empty but clusters exist in local_map_
    // This handles the case where clusters were created after the last cache update
    if ((!cluster_clouds_cache_ || cluster_clouds_cache_->empty()) && 
        local_map_ && local_map_->mvEleEdgeClusters.size() > 0) {
      std::vector<std::vector<cv::Point3d>> clusterClouds;
      std::vector<cv::Vec3b> clusterCloudColors;
      edge_viz::visualizeAssociationResult(local_map_, clusterClouds, clusterCloudColors);
      
      if (!clusterClouds.empty()) {
        cluster_clouds_cache_ =
            std::make_shared<const std::vector<std::vector<cv::Point3d>>>(
                std::move(clusterClouds));
        cluster_colors_cache_ =
            std::make_shared<const std::vector<cv::Vec3b>>(
                std::move(clusterCloudColors));
        VLOG(2) << "Updated cluster cache in snapshot: " << cluster_clouds_cache_->size() << " clusters";
      }
    }
    
    snap->clusterClouds = cluster_clouds_cache_;
    snap->clusterCloudColors = cluster_colors_cache_;
    snap->localMapClouds = local_map_clouds_cache_;
    snap->localMapCloudColors = local_map_colors_cache_;
    snap->environment_cloud = environment_cloud_cache_;

    auto window = std::make_shared<std::vector<Eigen::Matrix4d>>();
    if (local_map_ && local_map_->mvKeyFrames.size() > 0) {
      window->reserve(local_map_->mvKeyFrames.size());
      for (size_t i = 0; i < local_map_->mvKeyFrames.size(); ++i) {
        window->push_back(local_map_->mvKeyFrames[i]->KF_pose_g.matrix());
      }
    }
    snap->slidingWindow = window;

    // Update latest visualization data (thread-safe, visualization thread can read anytime)
    {
      std::lock_guard<std::mutex> viz_lock(viz_mutex_);
      latest_visualization_data_ = snap;
    }
  }



  const auto t_stereo2_start = std::chrono::steady_clock::now();
  if (has_stereo && stereo_result) {
    // need to match aagain after optical flow used to update the keypoints
    // wow this seems to make a pretty big difference!!
    const cv::Mat& left_rgb = image_container->rgb();
    const cv::Mat& right_rgb = image_container->rightRgb();
    FeaturePtrs stereo_features_2;
    stereo_result &=
        tracker_->stereoTrack(stereo_features_2, frame->static_features_,
                              left_rgb, right_rgb, rgbd_camera->baseline());
  }
  const auto t_stereo2_end = std::chrono::steady_clock::now();

  // VERY important calculation
  const gtsam::Pose3 T_k_1_k =
      nav_state_prev_.pose().inverse() * frame->T_world_camera_;
  vo_velocity_ = T_k_1_k;

  // we currently use the frame pose as the nav state - this value can come from
  // either the VO OR the IMU, depending on the result from the
  // solveCameraMotion this is only relevant since we dont solve incremental so
  // the backend is not immediately updating the frontend at which point we can
  // just use the best estimate in the case of the VO, the nav_state velocity
  const gtsam::NavState best_nav_state(frame->T_world_camera_,
                                       nav_state_curr_.velocity());
  // will be wrong (currently!!)
  nav_state_prev_ = best_nav_state;

  // if (R_curr_ref) {
  //   imu_frontend_.resetIntegration();
  // }

  const auto t_object_motion_start = std::chrono::steady_clock::now();
  const auto [object_motions, object_poses] =
      object_motion_solver_->solve(frame, previous_frame);
  const auto t_object_motion_end = std::chrono::steady_clock::now();

  // True only when an edge KeyFrame is created in this nominalSpin iteration.
  bool created_edge_kf_this_frame = false;
  std::vector<ObjectEdgeSnapshotItem> pre_kf_object_edge_snapshot;

  // const FeatureTrackerInfo& tracker_info = *frame->getTrackingInfo();
  // VLOG(1) << to_string(tracker_info);

  // Edge-based keyframe processing 
  if (FLAGS_use_edge_feature && !frame->static_edges_.empty()) {
    const gtsam::Pose3& pose_curr_coarse = frame->T_world_camera_;
    
    if (!is_edge_initialized_) {
      pose_last_edge_kf_ = pose_curr_coarse;
      is_edge_initialized_ = true;
    }
    
    // Check if should add edge keyframe
    bool select_edge_kf = shouldAddEdgeKeyFrame(pose_curr_coarse, pose_last_edge_kf_);
    
    // Check if local map is empty and get last keyframe pose (thread-safe)
    // Minimize lock time by reading values first, then computing pose
    bool local_map_empty = false;
    gtsam::Pose3 pose_last_kf;
    {
      std::lock_guard<std::mutex> lock(local_map_mutex_);
      local_map_empty = local_map_->mvKeyFrames.empty();
      if (!local_map_empty) {
        // Copy the pose outside the lock
        Sophus::SE3d kf_pose_sophus = local_map_->mvKeyFrames.back()->KF_pose_g;
        pose_last_kf = gtsam::Pose3(kf_pose_sophus.matrix());
      }
    }
    
    bool should_add_kf = select_edge_kf || local_map_empty;
    
    if (should_add_kf) {

      std::vector<float> x_list, y_list, depth_list, weight_list, theta_list;
      std::vector<cv::Point3f> frame_cloud_ref;  // Not used but required by function signature
      const int sample_bias = getFrontendParams().tracker_params.edge_coarse.sample_bias;
      const int maximum_point = getFrontendParams().tracker_params.edge_coarse.maximum_point;
      generateSrcPixelsSampled(frame, x_list, y_list, depth_list, weight_list, 
                              frame_cloud_ref, theta_list, sample_bias, maximum_point);
      if(x_list.size()>200) is_data_valid_ = true;

      // Set reference frame
      const ImageWrapper<ImageType::RGBMono>& rgb_wrapper_ref = frame->image_container_.rgb();
      cv::Mat mono_ref = ImageType::RGBMono::toMono(rgb_wrapper_ref);
      direct_tracker_->setReference(mono_ref, x_list, y_list, depth_list, weight_list, theta_list);
      direct_tracker_->setReferenceFrameId(frame->getFrameId());
      fine_tracker_->setCurrent(frame);
      const auto t_create_kf_start = std::chrono::steady_clock::now();
      // Calculate pose for keyframe
      gtsam::Pose3 pose_curr;
      if (local_map_empty) {
        pose_curr = pose_curr_coarse;
      } else {
        gtsam::Pose3 pose_last_prior = pose_last_edge_kf_;
        gtsam::Pose3 pose_bias = pose_last_prior.inverse() * pose_curr_coarse;
        pose_curr = pose_last_kf * pose_bias;
      }
      
      KeyFramePtr pKF = createKeyFrameFromFrame(frame, pose_curr);
      const auto t_create_kf_end = std::chrono::steady_clock::now();

      // Keep KeyFrame alive by registering it in the global EdgeMap (shared ownership).
      // This prevents Object::observed_kfs raw pointers from becoming dangling when the
      // sliding-window local_map_ is reset.
      if (map_ && pKF) {
        map_->AddKeyFrame(pKF);
      }

      // Get depth data per detection from frame
      const auto& depth_data_per_det = frame->getDepthDataPerDetection();
      
      // Safety check: graph and map must be initialized before processing objects
      if (pKF->graph && map_) {
        // Snapshot object-edge map BEFORE updating/creating objects at this KF.
        // This is used to visualize "previous-KF-only" projections.
        pre_kf_object_edge_snapshot.clear();
        // Use only objects referenced by this KF's detection graph to reduce
        // race risk with concurrent map culling.
        if (pKF->graph) {
          pre_kf_object_edge_snapshot.reserve(pKF->graph->attributes.size());
          for (const auto& kv : pKF->graph->attributes) {
            const auto& attr = kv.second;
            dyno::Object* obj = attr.obj;
            if (!obj || obj->isBad()) continue;
            const auto pts = obj->GetAssociatedMapPoints();
            if (pts.empty()) continue;
            ObjectEdgeSnapshotItem item;
            item.color_bgr = obj->GetColor();
            item.world_points.reserve(pts.size());
            for (const auto& p : pts) {
              if (!p.allFinite()) continue;
              item.world_points.emplace_back(p.x(), p.y(), p.z());
            }
            if (!item.world_points.empty()) {
              pre_kf_object_edge_snapshot.emplace_back(std::move(item));
            }
          }
        }

        // Construct Rt [R_cw | t_cw] from T_world_camera_ (used for both new objects and AddDetection)
        Eigen::Matrix3d K_eigen = camera_->getParams().getCameraMatrixEigen();
        const gtsam::Pose3& T_wc = frame->T_world_camera_;
        const gtsam::Pose3  T_cw = T_wc.inverse();
        Matrix34d Rt;
        Rt.block<3, 3>(0, 0) = T_cw.rotation().matrix();
        Rt.block<3, 1>(0, 3) = T_cw.translation();

        Eigen::Matrix<double, 3, 4> P = K_eigen * Rt;
        const cv::Mat& motion_mask = frame->image_container_.objectMotionMask();

        // Optional visualization image to debug object handling on this frame.
        cv::Mat object_viz =
            ImageType::RGBMono::toRGB(frame->image_container_.rgb()).clone();
        
        for(auto [node_id, attribute] : pKF->graph->attributes){
          const auto& bb = attribute.bbox;  // [xmin, ymin, xmax, ymax]
          cv::Rect rect(static_cast<int>(bb[0]),
                        static_cast<int>(bb[1]),
                        static_cast<int>(bb[2] - bb[0]),
                        static_cast<int>(bb[3] - bb[1]));

          if(attribute.obj){
            // Matched by edge-projection: update existing object with new observation
              attribute.obj->AddDetection(
                  attribute.label, 
                  attribute.bbox, 
                  attribute.ell, 
                  attribute.confidence, 
                  Rt, 
                  static_cast<unsigned int>(frame->getFrameId()), 
                  pKF.get()
              );
          } else {
            // Filter by confidence score
            if (attribute.confidence < kMinConfidenceScore_) {
                continue;
            }

            // Check if node_id is valid index for depth_data_per_det
            if (node_id >= depth_data_per_det.size()) {
                continue;
            }
            
            const auto& depth_data = depth_data_per_det[node_id];
          
            //create new object
            Object* obj = new Object(
                static_cast<unsigned int>(attribute.label),
                attribute.bbox,  // BBox2 is typedef of Eigen::Vector4d
                attribute.ell,
                static_cast<double>(attribute.confidence),
                depth_data,
                K_eigen,
                Rt,
                static_cast<long unsigned int>(frame->getFrameId()),
                pKF.get()  // Convert shared_ptr to raw pointer
            );
            if(obj->GetAssociatedMapPoints().size()<5){
                delete obj;
                continue;
            }
            // Extract CLIP feature for the new object (disabled)
            // if (clip_client_) { ... }

            map_->AddObject(obj);
            local_map_->mlpRecentAddedObjects.push_back(obj);
            pKF->graph->attributes[node_id].obj = obj;
          }

          // Draw bbox for this detection on object_viz (for debugging):
          cv::Scalar draw_col(0, 255, 255);      // default yellow for bbox
          cv::Scalar edge_col(150, 150, 150);    // fixed gray for edges
          if (pKF->graph->attributes[node_id].obj) {
            cv::Scalar c = pKF->graph->attributes[node_id].obj->GetColor();
            draw_col = c;      // bbox는 object 색으로
          }

          // Overlay the detection mask in the same object color inside bbox.
          // The motion mask stores packed BGR as 32-bit int.
          const cv::Mat& motion_mask = frame->image_container_.objectMotionMask();
          if (!motion_mask.empty() &&
              motion_mask.type() == CV_32SC1 &&
              motion_mask.size() == object_viz.size()) {
            const int cls_label = attribute.label;
            const unsigned char target_b =
                static_cast<unsigned char>((cls_label * 37) % 256);
            const unsigned char target_g =
                static_cast<unsigned char>((cls_label * 17) % 256);
            const unsigned char target_r =
                static_cast<unsigned char>((cls_label * 97) % 256);
            const int target_mask_val =
                (static_cast<int>(target_b) << 16) |
                (static_cast<int>(target_g) << 8) |
                static_cast<int>(target_r);

            const int x0 = std::max(0, rect.x);
            const int y0 = std::max(0, rect.y);
            const int x1 = std::min(object_viz.cols, rect.x + rect.width);
            const int y1 = std::min(object_viz.rows, rect.y + rect.height);
            constexpr float kMaskAlpha = 0.35f;
            for (int y = y0; y < y1; ++y) {
              const int* mask_row = motion_mask.ptr<int>(y);
              cv::Vec3b* viz_row = object_viz.ptr<cv::Vec3b>(y);
              for (int x = x0; x < x1; ++x) {
                if (mask_row[x] != target_mask_val) continue;
                for (int c = 0; c < 3; ++c) {
                  const float blended =
                      (1.0f - kMaskAlpha) * static_cast<float>(viz_row[x][c]) +
                      kMaskAlpha * static_cast<float>(draw_col[c]);
                  viz_row[x][c] = static_cast<uchar>(
                      std::max(0.0f, std::min(255.0f, blended)));
                }
              }
            }
          }

          // Draw bbox for this detection
          cv::rectangle(object_viz, rect, draw_col, 2);
          std::string text = std::to_string(attribute.label);
          cv::putText(object_viz, text, rect.tl() + cv::Point(0, -3),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, draw_col, 1, cv::LINE_AA);

          // Additionally, draw edge points inside this detection bbox,
          // coloring only those edges that are actually associated with this
          // object (mmEdgeIndex2ObjectId[edge_idx] == obj_id), and drawing the
          // rest in gray.
          std::vector<std::size_t> enc_list =
              pKF->GetEdgeIndicesInBox(static_cast<float>(bb[0]),
                                       static_cast<float>(bb[2]),
                                       static_cast<float>(bb[1]),
                                       static_cast<float>(bb[3]));
          int obj_id_for_viz = -1;
          if (pKF->graph->attributes[node_id].obj) {
            obj_id_for_viz =
                static_cast<int>(pKF->graph->attributes[node_id].obj->GetId());
          }
          for (std::size_t enc : enc_list) {
            const int edge_id = static_cast<int>(enc / 100000);
            const int pt_idx  = static_cast<int>(enc % 100000);
            auto itEdge = pKF->mmIndexMap.find(edge_id);
            if (itEdge == pKF->mmIndexMap.end()) continue;
            const int edge_idx = itEdge->second;
            const Edge& e = pKF->mvEdges[edge_idx];
            if (pt_idx < 0 || pt_idx >= static_cast<int>(e.mvPoints.size()))
              continue;

            // Decide color based on association
            cv::Scalar pt_col = edge_col;  // default gray
            if (obj_id_for_viz >= 0) {
              auto it_obj =
                  pKF->mmEdgeIndex2ObjectId.find(static_cast<int>(edge_idx));
              if (it_obj != pKF->mmEdgeIndex2ObjectId.end() &&
                  it_obj->second == obj_id_for_viz) {
                // This edge is associated with this object -> use object color
                pt_col = draw_col;
              }
            }

            const auto& pt = e.mvPoints[pt_idx];
            int u = static_cast<int>(std::round(pt.x));
            int v = static_cast<int>(std::round(pt.y));
            if (u < 0 || v < 0 || u >= object_viz.cols || v >= object_viz.rows)
              continue;
            cv::circle(object_viz, cv::Point(u, v), 1, pt_col, -1,
                       cv::LINE_AA);
          }
        }

        // Send visualization to display queue (optional debug window)
        if (display_queue_) {
          display_queue_->push(ImageToDisplay("Detections (KF only)", object_viz));
        }
      } else {
        if (!pKF->graph) {
          LOG(ERROR) << "processEdgeKeyFrame: pKF->graph is nullptr, skipping object creation";
        }
        if (!map_) {
          LOG(ERROR) << "processEdgeKeyFrame: map_ is nullptr, skipping object creation";
        }
      }

      ObjectCulling(pKF);


      
      if (pKF) {
        created_edge_kf_this_frame = true;
        {
          std::lock_guard<std::mutex> lock(last_kf_mutex_);
          last_keyframe_ = frame;
        }

        
        
        optimization_queue_.push(pKF);
        VLOG(10) << "\033[32m[QUEUE PUSH]\033[0m kf_id=" << pKF->KF_ID
                  << ", frame_id=" << frame->getFrameId();
      }
      
      pose_last_edge_kf_ = pose_curr_coarse;
      
      // LOG_EVERY_N(INFO, 30) << "CreateKeyFrame frame=" << frame->getFrameId()
      //                       << " ms=" << std::chrono::duration<double, std::milli>(
      //                           t_create_kf_end - t_create_kf_start).count();
    }
  }

  const auto t_fill_packet_start = std::chrono::steady_clock::now();
  VisionImuPacket::Ptr vision_imu_packet = std::make_shared<VisionImuPacket>();
  vision_imu_packet->frameId(frame->getFrameId());
  vision_imu_packet->timestamp(frame->getTimestamp());
  vision_imu_packet->pim(pim);
  vision_imu_packet->groundTruthPacket(input->optional_gt_);
  fillOutputPacketWithTracks(vision_imu_packet, *frame, T_k_1_k, object_motions,
                             object_poses);
  const auto t_fill_packet_end = std::chrono::steady_clock::now();

  if (R_curr_ref) {
    imu_frontend_.resetIntegration();
  }

  const auto t_create_image_start = std::chrono::steady_clock::now();
  DebugImagery debug_imagery;
  debug_imagery.tracking_image =
      createTrackingImage(frame, previous_frame, object_poses);
  static cv::Mat cached_object_edge_proj_image;
  if (created_edge_kf_this_frame || cached_object_edge_proj_image.empty()) {
    if (!pre_kf_object_edge_snapshot.empty()) {
      VLOG(1) << "[ObjEdgeProjSelect] frame_id=" << frame->getFrameId()
              << " mode=SNAPSHOT"
              << " snapshot_objs=" << pre_kf_object_edge_snapshot.size();
      cached_object_edge_proj_image =
          createObjectEdgeProjectionImageFromSnapshot(
              frame, pre_kf_object_edge_snapshot);
    } else {
      VLOG(1) << "[ObjEdgeProjSelect] frame_id=" << frame->getFrameId()
              << " mode=LIVE_FALLBACK";
      cached_object_edge_proj_image =
          createObjectEdgeProjectionImage(frame, object_poses);
    }
  }
  const ImageContainer& processed_image_container = frame->image_container_;
  debug_imagery.rgb_viz =
      ImageType::RGBMono::toRGB(processed_image_container.rgb());
  // debug_imagery.flow_viz =
  //     ImageType::OpticalFlow::toRGB(processed_image_container.opticalFlow());
  // debug_imagery.mask_viz = ImageType::MotionMask::toRGB(
  //     processed_image_container.objectMotionMask());
  debug_imagery.depth_viz =
      ImageType::Depth::toRGB(processed_image_container.depth());
  const auto t_create_image_end = std::chrono::steady_clock::now();

  if (display_queue_) {
    display_queue_->push(
        ImageToDisplay("Tracks", debug_imagery.tracking_image));
    if (created_edge_kf_this_frame && !cached_object_edge_proj_image.empty()) {
      display_queue_->push(
          ImageToDisplay("Tracks Object Edge", cached_object_edge_proj_image));
    }

    cv::Mat stereo_matches;
    if (tracker_->drawStereoMatches(stereo_matches, *frame)) {
      display_queue_->push(ImageToDisplay("Stereo Matches", stereo_matches));
    }
  }

  vision_imu_packet->debugImagery(debug_imagery);

  if (FLAGS_set_dense_labelled_cloud) {
    VLOG(30) << "Setting dense labelled cloud";
    utils::ChronoTimingStats labelled_clout_timer(
        "frontend.dense_labelled_cloud");
    const cv::Mat& board_detection_mask = tracker_->getBoarderDetectionMask();
    PointCloudLabelRGB::Ptr dense_labelled_cloud =
        frame->projectToDenseCloud(&board_detection_mask);
    vision_imu_packet->denseLabelledCloud(dense_labelled_cloud);
  }

  // if (FLAGS_save_frontend_json)
  //   output_packet_record_.insert({output->getFrameId(), output});

  const auto t_send_logger_start = std::chrono::steady_clock::now();
  sendToFrontendLogger(frame, vision_imu_packet);
  const auto t_send_logger_end = std::chrono::steady_clock::now();

  const auto t_nominal_end = std::chrono::steady_clock::now();
  

  return {State::Nominal, vision_imu_packet};
}

bool RGBDInstanceFrontendModule::solveCameraMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,
    std::optional<gtsam::Rot3> R_curr_ref) {
  utils::ChronoTimingStats timer("frontend.solve_camera_motion");
  Pose3SolverResult result;

  const auto& frontend_params = getFrontendParams();
  if (frontend_params.use_ego_motion_pnp) {
    result = motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k,
                                                          R_curr_ref);
  } else {
    // TODO: untested
    LOG(FATAL) << "Not tested";
    // result = motion_solver_.geometricOutlierRejection3d3d(frame_k_1,frame_k);
  }

  VLOG(15) << (frontend_params.use_ego_motion_pnp ? "3D2D" : "3D3D")
           << "camera pose estimate at frame " << frame_k->frame_id_
           << (result.status == TrackingStatus::VALID ? " success "
                                                      : " failure ")
           << ":\n"
           << "- Tracking Status: " << to_string(result.status) << '\n'
           << "- Total Correspondences: "
           << result.inliers.size() + result.outliers.size() << '\n'
           << "\t- # inliers: " << result.inliers.size() << '\n'
           << "\t- # outliers: " << result.outliers.size() << '\n';

  // collect all usable tracklets
  TrackletIds tracklets = frame_k->static_features_.collectTracklets();
  CHECK_GE(tracklets.size(),
           result.inliers.size() +
               result.outliers.size());  // tracklets shoudl be more (or same
                                         // as) correspondances as there will
                                         // be new points untracked
  frame_k->static_features_.markOutliers(result.outliers);

  // was 60!
  if (result.status != TrackingStatus::VALID || result.inliers.size() < 30) {
    // TODO: fix code structure - nav state should be passed in?
    // use nav state which we assume is updated by IMU
    std::stringstream ss;
    ss << "Number usable static feature < 30 or status is invalid. ";

    // check if we have a nav state update from the IMU (this is also a cheap
    // way of checking that we HAVE an imu). If we do we can use the nav state
    // directly to update the current pose as the nav state is the forward
    // prediction from the IMU
    if (last_imu_k_ == frame_k->getFrameId()) {
      frame_k->T_world_camera_ = nav_state_curr_.pose();
      ss << "Nav state was previous updated with IMU. Using predicted pose to "
            "set camera transform; k"
         << frame_k->getFrameId();
    } else {
      // no IMU for forward prediction, use constant velocity model to propogate
      // pose expect nav_state_prev_ to always be updated with the best
      // pose!
      frame_k->T_world_camera_ = nav_state_prev_.pose() * vo_velocity_;
      ss << "Nav state has no information from imu. Using constant velocity "
            "model to propofate pose; k"
         << frame_k->getFrameId();
    }

    VLOG(10) << ss.str();

    // if fails should we mark current inliers as outliers?

    // TODO: should almost definitely do this in future, but right now we use
    // measurements to construct a framenode in the backend so if there are no
    // measurements we get a frame_node null.... for now... make hack and set
    // all ages of inliers to 1!!! since we need n measurements in the backend
    // this will ensure that they dont get added to the
    //  optimisation problem but will get added to the map...
    for (const auto& inlier : result.inliers) {
      frame_k->static_features_.getByTrackletId(inlier)->age(1u);
    }
    // frame_k->static_features_.markOutliers(result.inliers);

    // for some reason using tracklets to mark all features gives error as a
    // tracklet id
    // seems to be not actually in static features. Dont know why
    // maybe remeber that tracklets (fromc collectTracklets) is actually just
    // the usable tracklets...?
    // frame_k->static_features_.markOutliers(tracklets);
    return false;
  } else {
    frame_k->T_world_camera_ = result.best_result;

    if (frontend_params.refine_camera_pose_with_joint_of) {
      VLOG(10) << "Refining camera pose with joint of";
      utils::ChronoTimingStats timer("frontend.solve_camera_motion.of_refine");
      OpticalFlowAndPoseOptimizer flow_optimizer(
          frontend_params.object_motion_solver_params.joint_of_params);

      auto flow_opt_result = flow_optimizer.optimizeAndUpdate<CalibrationType>(
          frame_k_1, frame_k, result.inliers, result.best_result);
      frame_k->T_world_camera_ = flow_opt_result.best_result.refined_pose;
      VLOG(15) << "Refined camera pose with optical flow - error before: "
               << flow_opt_result.error_before.value_or(NaN)
               << " error_after: " << flow_opt_result.error_after.value_or(NaN);
    }
    return true;
  }
}
bool RGBDInstanceFrontendModule::DirectTrack(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,
                                           const gtsam::Pose3& T_k_1_k_initial, 
                                           gtsam::Pose3& T_k_1_k_refined) {
  utils::ChronoTimingStats timer("frontend.direct_track");
  
  // // Get grayscale images
  // const ImageWrapper<ImageType::RGBMono>& rgb_wrapper_ref = frame_k_1->image_container_.rgb();
  // cv::Mat mono_ref = ImageType::RGBMono::toMono(rgb_wrapper_ref);
  const ImageWrapper<ImageType::RGBMono>& rgb_wrapper_cur = frame_k->image_container_.rgb();
  cv::Mat mono_cur = ImageType::RGBMono::toMono(rgb_wrapper_cur);
  
  // // Collect static features with depth from previous frame
  // TrackletIds tracklets = frame_k_1->static_features_.collectTracklets();
  // if (tracklets.empty()) {
  //   // LOG(WARNING) << "DirectTrack: no tracklets available";
  //   T_k_1_k_refined = T_k_1_k_initial;
  //   return false;
  // }
  
  if(is_data_valid_) {
    // Set current frame
    direct_tracker_->setCurrent(mono_cur);
    direct_tracker_->setCurrentFrameId(frame_k->getFrameId());
    const gtsam::Pose3 T_k_k_1_initial = T_k_1_k_initial.inverse();
    const gtsam::Matrix4& T_matrix = T_k_k_1_initial.matrix();
    Sophus::SE3d T21(Sophus::SO3d(T_matrix.topLeftCorner<3, 3>()), 
                     T_matrix.topRightCorner<3, 1>());
    // std::cout<<"DirectTrack: reference frame id=" << direct_tracker_->getReference()
    //           << ", current frame id=" << direct_tracker_->getCurrent()<<std::endl;
         
    direct_tracker_->estimatePyramid(T21, true);
    // std::cout<<"DirectTrack: result pose is valid"<<T21.matrix().allFinite()<<std::endl;

    if(checkPoseJump(T21)){
      LOG(WARNING) << "\033[33m [WARNING] \033[0m"<<frame_k->getFrameId()<<" : pose jumped, remain with initial pose!";
      // Convert gtsam::Pose3 back to Sophus::SE3d
      const gtsam::Pose3 T_k_k_1_initial = T_k_1_k_initial.inverse();
      const gtsam::Matrix4& T_matrix = T_k_k_1_initial.matrix();
      T21 = Sophus::SE3d(Sophus::SO3d(T_matrix.topLeftCorner<3, 3>()), 
                         T_matrix.topRightCorner<3, 1>());
    }

    // Check if tracking was successful
    if (!T21.matrix().allFinite()) {
      LOG(WARNING) << "DirectTracker returned invalid pose matrix";
      T_k_1_k_refined = T_k_1_k_initial;
      return false;
    }
    
    // Log T21 in TUM format
    const Eigen::Matrix4d& T = T21.matrix();
    Eigen::Vector3d translation = T.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation = T.block<3, 3>(0, 0);
    Eigen::Quaterniond q(rotation);
    const double timestamp = frame_k->getTimestamp();
    
    // LOG(INFO) << "T21 TUM: " << std::fixed << std::setprecision(6)
    //           << timestamp << " "
    //           << translation.x() << " " << translation.y() << " " << translation.z() << " "
    //           << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
    
    const Eigen::Matrix4d T_result = T21.inverse().matrix();
    T_k_1_k_refined = gtsam::Pose3(T_result);
  
  }
    
  return true;
}

bool RGBDInstanceFrontendModule::FineTrack(Frame::Ptr frame,
                                           const gtsam::Pose3& pose_cur_refined, 
                                           gtsam::Pose3& pose_cur_fine_refined) {
  utils::ChronoTimingStats timer("frontend.fine_track");
  
  // Get reference frame (last keyframe Frame)
  Frame::Ptr ref_frame;
  {
    std::lock_guard<std::mutex> lock(last_kf_mutex_);
    ref_frame = last_keyframe_;
  }
  
  if (!ref_frame) {
    VLOG(5) << "FineTrack: no reference keyframe available";
    return false;
  }
  
  // Convert gtsam::Pose3 to Sophus::SE3d for FineTracker
  // FineTracker expects T_cur_ref (current to reference), which is pose_cur_refined.inverse()
  const gtsam::Pose3 T_cur_ref_initial = pose_cur_refined;
  const gtsam::Matrix4& T_matrix = T_cur_ref_initial.matrix();
  Sophus::SE3d T21(Sophus::SO3d(T_matrix.topLeftCorner<3, 3>()), 
                   T_matrix.topRightCorner<3, 1>());

  fine_tracker_->setReference(frame);
  fine_tracker_->setPosePriorCur2Ref(T21);
  Sophus::SE3d pose_final;
  
  // LOG(INFO) << "FineTrack: reference frame id=" << fine_tracker_->getReference()
  //           << ", current frame id=" << fine_tracker_->getCurrent();

  fine_tracker_->estimate(pose_final, true);
  if(checkPoseJump(pose_final)){
    LOG(WARNING) << "\033[33m [WARNING] \033[0m"<<frame->getFrameId()<<" : pose jumped, remain with coarse result!";
    pose_final = T21;
  }
  pose_cur_fine_refined = gtsam::Pose3(pose_final.inverse().matrix());
  
  return true;
}

bool RGBDInstanceFrontendModule::checkPoseJump(Sophus::SE3d pose)
{
    Eigen::Matrix3d R = pose.rotationMatrix();
    Eigen::Vector3d t = pose.translation();
    double t_thres = 0.10;
    double angle_thres = 15.0;
    Eigen::AngleAxisd rotationVector(R);  
    Eigen::Vector3d axis = rotationVector.axis();  
    double angle = rotationVector.angle() * 180.0 / M_PI;
    if(t_thres < t.norm() || angle > angle_thres){
        return true;
    }else{
        return false;
    }

}

void RGBDInstanceFrontendModule::fillOutputPacketWithTracks(
    VisionImuPacket::Ptr vision_imu_packet, const Frame& frame,
    const gtsam::Pose3& T_k_1_k, const ObjectMotionMap& object_motions,
    const ObjectPoseMap& object_poses) const {
  CHECK(vision_imu_packet);
  const auto frame_id = frame.getFrameId();
  // construct image tracks
  const double& static_pixel_sigma =
      params_.backend_params_.static_pixel_noise_sigma;
  const double& static_point_sigma =
      params_.backend_params_.static_point_noise_sigma;

  const double& dynamic_pixel_sigma =
      params_.backend_params_.dynamic_pixel_noise_sigma;
  const double& dynamic_point_sigma =
      params_.backend_params_.dynamic_point_noise_sigma;

  gtsam::Vector2 static_pixel_sigmas;
  static_pixel_sigmas << static_pixel_sigma, static_pixel_sigma;

  gtsam::Vector2 dynamic_pixel_sigmas;
  dynamic_pixel_sigmas << dynamic_pixel_sigma, dynamic_pixel_sigma;

  auto& camera = *this->camera_;
  auto fill_camera_measurements =
      [&camera](FeatureFilterIterator it,
                CameraMeasurementStatusVector* measurements, FrameId frame_id,
                const gtsam::Vector2& pixel_sigmas, double depth_sigma) {
        std::shared_ptr<RGBDCamera> rgbd_camera = camera.safeGetRGBDCamera();

        for (const Feature::Ptr& f : it) {
          const TrackletId tracklet_id = f->trackletId();
          const Keypoint& kp = f->keypoint();
          const ObjectId object_id = f->objectId();
          CHECK_EQ(f->objectId(), object_id);
          CHECK(Feature::IsUsable(f));

          MeasurementWithCovariance<Keypoint> kp_measurement =
              MeasurementWithCovariance<Keypoint>::FromSigmas(kp, pixel_sigmas);
          CameraMeasurement camera_measurement(kp_measurement);

          // This can come from either stereo or rgbd
          if (f->hasDepth()) {
            // MeasurementWithCovariance<Landmark> landmark_measurement(
            //     // assume sigma_u and sigma_v are identical
            //     vision_tools::backProjectAndCovariance(
            //         *f, camera, pixel_sigmas(0), depth_sigma));
            // camera_measurement.landmark(landmark_measurement);
            Landmark landmark;
            camera.backProject(kp, f->depth(), &landmark);

            gtsam::Vector3 sigmas;
            sigmas << depth_sigma, depth_sigma, depth_sigma;

            MeasurementWithCovariance<Landmark> landmark_measurement =
                MeasurementWithCovariance<Landmark>::FromSigmas(landmark,
                                                                sigmas);
            camera_measurement.landmark(landmark_measurement);
          }

          if (f->hasRightKeypoint()) {
            CHECK(f->hasDepth())
                << "Right keypoint set for feature but no depth!";
            MeasurementWithCovariance<Keypoint> right_kp_measurement =
                MeasurementWithCovariance<Keypoint>::FromSigmas(
                    f->rightKeypoint(), pixel_sigmas);
            camera_measurement.rightKeypoint(right_kp_measurement);
          }
          // no right keypoint and has rgbd camera and has depth, project
          // keypoint into right camera
          else if (rgbd_camera && f->hasDepth()) {
            bool right_projection_result = rgbd_camera->projectRight(f);
            if (!right_projection_result) {
              // TODO: for now mark as outlier and ignore point
              f->markOutlier();
              continue;
            }

            CHECK(f->hasRightKeypoint());
            MeasurementWithCovariance<Keypoint> right_kp_measurement =
                MeasurementWithCovariance<Keypoint>::FromSigmas(
                    f->rightKeypoint(), pixel_sigmas);
            camera_measurement.rightKeypoint(right_kp_measurement);
          }

          if (f->keypointType() == KeyPointType::STATIC) {
            CHECK_EQ(object_id, background_label);
          } else {
            CHECK_NE(object_id, background_label);
          }

          measurements->push_back(
              CameraMeasurementStatus(camera_measurement, frame_id, tracklet_id,
                                      object_id, ReferenceFrame::LOCAL));
        }
      };

  // TODO: fill ttracking status?
  VisionImuPacket::CameraTracks camera_tracks;
  auto* static_measurements = &camera_tracks.measurements;
  fill_camera_measurements(frame.usableStaticFeaturesBegin(),
                           static_measurements, frame_id, static_pixel_sigmas,
                           static_point_sigma);
  camera_tracks.X_W_k = frame.getPose();
  camera_tracks.T_k_1_k = T_k_1_k;
  vision_imu_packet->cameraTracks(camera_tracks);

  // First collect all dynamic measurements then split them by object
  // This is a bit silly
  CameraMeasurementStatusVector dynamic_measurements;
  fill_camera_measurements(frame.usableDynamicFeaturesBegin(),
                           &dynamic_measurements, frame_id,
                           dynamic_pixel_sigmas, dynamic_point_sigma);

  VisionImuPacket::ObjectTrackMap object_tracks;
  // motions in this frame (ie. new motions!!)
  MotionEstimateMap motion_estimates = object_motions.toEstimateMap(frame_id);
  auto pose_estimates = object_poses.toEstimateMap(frame_id);

  // fill object tracks based on valid motions
  for (const auto& [object_id, motion_reference_estimate] : motion_estimates) {
    CHECK(pose_estimates.exists(object_id))
        << "Object pose missing " << info_string(frame_id, object_id)
        << " but frontend motion available";
    const auto& L_W_k = pose_estimates.at(object_id);

    VisionImuPacket::ObjectTracks object_track;
    object_track.H_W_k_1_k = motion_reference_estimate;
    object_track.L_W_k = L_W_k;
    object_tracks.insert2(object_id, object_track);
  }

  for (const auto& dm : dynamic_measurements) {
    const auto& object_id = dm.objectId();
    // throw out features detected on objects where the tracking failed
    if (object_tracks.exists(object_id)) {
      VisionImuPacket::ObjectTracks& object_track = object_tracks.at(object_id);
      object_track.measurements.push_back(dm);
    }
  }
  vision_imu_packet->objectTracks(object_tracks);
}

void RGBDInstanceFrontendModule::sendToFrontendLogger(
    const Frame::Ptr& frame, const VisionImuPacket::Ptr& vision_imu_packet) {
  if (logger_) {
    auto ground_truths = this->shared_module_info.getGroundTruthPackets();
    logger_->logCameraPose(frame->getFrameId(), vision_imu_packet->cameraPose(),
                           ground_truths);
    logger_->logObjectMotion(frame->getFrameId(),
                             vision_imu_packet->objectMotions(), ground_truths);
    logger_->logObjectPose(frame->getFrameId(),
                           vision_imu_packet->objectPoses(), ground_truths);
    logger_->logTrackingLengthHistogram(frame);
    logger_->logPoints(frame->getFrameId(), vision_imu_packet->cameraPose(),
                       vision_imu_packet->dynamicLandmarkMeasurements());
    logger_->logFrameIdToTimestamp(frame->getFrameId(), frame->getTimestamp());
  }
}

cv::Mat RGBDInstanceFrontendModule::createTrackingImage(
    const Frame::Ptr& frame_k, const Frame::Ptr& frame_k_1,
    const ObjectPoseMap& object_poses) const {
  // ── Commented out: default feature-track visualization ───────────────────
  // cv::Mat tracking_image = tracker_->computeImageTracks(
  //     *frame_k_1, *frame_k, getFrontendParams().image_tracks_vis_params);

  // Start from a blank RGB copy of the current frame image.
  cv::Mat tracking_image =
      ImageType::RGBMono::toRGB(frame_k->image_container_.rgb()).clone();

  // ── Edge visualization: per-point depth continuity via normal-side check ──
  // Since frame_k->static_edges_ has already passed edgeCullingContinuity(),
  // consecutive-point checks tend to be all continuous.
  // Instead, classify each point by depth jump across +/- edge normal:
  //   if one side is much deeper, mark as discontinuous (BLUE).
  {
    constexpr float kOffsetPx = 4.0f;
    // Depth-jump threshold should be larger at near range (avoid false BLUE)
    // and smaller at far range (avoid false RED).
    // thr(d) = clamp(kInvScale / d, kThrMin, kThrMax)
    constexpr float kInvScale = 0.24f;  // meter^2
    constexpr float kThrMin = 0.04f;    // m
    constexpr float kThrMax = 0.4f;    // m
    const cv::Vec3b kContCol(0,   0, 255);  // RED  (BGR)
    const cv::Vec3b kDiscCol(255, 0,   0);  // BLUE (BGR)

    // Use raw depth image for side-depth probing; edge lookup map is biased
    // toward edge pixels and tends to classify everything as continuous.
    const cv::Mat depth_img = frame_k->image_container_.depth();
    auto lookupDepthFrame = [&](int px, int py) -> float {
      if (depth_img.empty()) return -1.f;
      if (px < 0 || py < 0 ||
          px >= frame_k->img_width_ || py >= frame_k->img_height_) return -1.f;
      // 3x3 local median-like robust sample around (px,py)
      float vals[9];
      int n = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          int x = px + dx, y = py + dy;
          if (x < 0 || y < 0 || x >= frame_k->img_width_ || y >= frame_k->img_height_) continue;
          float d = depth_img.at<float>(y, x);
          if (d > 0.2f && std::isfinite(d)) vals[n++] = d;
        }
      }
      if (n == 0) return -1.f;
      std::nth_element(vals, vals + n / 2, vals + n);
      return vals[n / 2];
    };

    for (const auto& edge : frame_k->static_edges_) {
      if (edge.mvPoints.empty()) continue;
      for (const auto& pt : edge.mvPoints) {
        const int u = static_cast<int>(std::round(pt.x));
        const int v = static_cast<int>(std::round(pt.y));
        if (u < 0 || v < 0 ||
            u >= tracking_image.cols || v >= tracking_image.rows) continue;
        if (pt.depth <= 0.2f) continue;

        const float th = pt.imgGradAngle * static_cast<float>(M_PI) / 180.f;
        const float nx = std::cos(th), ny = std::sin(th);
        const int xp = static_cast<int>(std::round(pt.x + kOffsetPx * nx));
        const int yp = static_cast<int>(std::round(pt.y + kOffsetPx * ny));
        const int xn = static_cast<int>(std::round(pt.x - kOffsetPx * nx));
        const int yn = static_cast<int>(std::round(pt.y - kOffsetPx * ny));

        const float dp = lookupDepthFrame(xp, yp);
        const float dn = lookupDepthFrame(xn, yn);
        const float d_edge = pt.depth;
        const float thr_unclamped = kInvScale / std::max(0.2f, d_edge);
        const float thr = std::max(kThrMin, std::min(kThrMax, thr_unclamped));
        // Balanced rule:
        // 1) both valid: discontinuous if either side is much deeper
        // 2) only one valid: discontinuous if that valid side is much deeper
        // 3) none valid: keep continuous (insufficient evidence)
        bool discontinuous = false;
        if (dp > 0.f && dn > 0.f) {
          discontinuous = ((dp - d_edge) > thr) || ((dn - d_edge) > thr);
        } else if (dp > 0.f || dn > 0.f) {
          const float d_valid = (dp > 0.f) ? dp : dn;
          discontinuous = ((d_valid - d_edge) > thr);
        }

        tracking_image.at<cv::Vec3b>(v, u) =
            discontinuous ? kDiscCol : kContCol;
      }
    }
  }

  // ── Edge fitting visualization on Tracks window ──────────────────────
  // For each object pose active in this frame, project the object's
  // associated 3D edge points into the current image and visualize:
  //   - in-mask projected points (mask overlap in association) in object color
  //   - out-of-mask projected points in gray
  if (false && map_) {
    const auto objects_in_frame = object_poses.collectByFrame(frame_k->getFrameId());
    if (!objects_in_frame.empty()) {
  const auto& camera_params = camera_->getParams();
  const auto& K = camera_params.getCameraMatrix();
      const double fx = K.at<double>(0, 0);
      const double fy = K.at<double>(1, 1);
      const double cx = K.at<double>(0, 2);
      const double cy = K.at<double>(1, 2);

      const gtsam::Pose3& X_k = frame_k->getPose();  // world -> camera handled below
      const gtsam::Pose3 T_cw = X_k.inverse();      // camera <- world
      const Eigen::Matrix3d R_cw = T_cw.rotation().matrix();
      const Eigen::Vector3d t_cw = T_cw.translation();

      const cv::Mat motion_mask = frame_k->image_container_.objectMotionMask();
      const bool has_motion_mask =
          !motion_mask.empty() && motion_mask.type() == CV_32SC1 &&
          motion_mask.size() == cv::Size(frame_k->img_width_, frame_k->img_height_);

      constexpr float kMinProjZ = 0.1f;
      constexpr int kStride = 10;         // visualize subset for speed
      constexpr int kMaxProjPts = 2000;   // safety cap
      constexpr int kMinEdgePts = 10;    // same as association
      constexpr float kEdgeRatioTh = 0.5f;

      for (const auto& [obj_id, pose_w_o] : objects_in_frame) {
        (void)pose_w_o;  // associated_world_points_ are already in world; only camera pose is needed.
        dyno::Object* obj = map_->GetObject(static_cast<int>(obj_id));
        if (!obj || obj->isBad()) continue;

        const auto assoc_pts = obj->GetAssociatedMapPoints();
        if (assoc_pts.empty()) continue;

        const int lbl = static_cast<int>(obj->GetCategoryId());
        const unsigned char target_b =
            static_cast<unsigned char>((lbl * 37) % 256);
        const unsigned char target_g =
            static_cast<unsigned char>((lbl * 17) % 256);
        const unsigned char target_r =
            static_cast<unsigned char>((lbl * 97) % 256);
        const int target_mask_val =
            (static_cast<int>(target_b) << 16) |
            (static_cast<int>(target_g) << 8) |
            static_cast<int>(target_r);

        const cv::Scalar obj_col = obj->GetColor();  // BGR
        const cv::Scalar out_col(180, 180, 180);      // gray for outliers

        int in_cnt = 0;
        int tot_cnt = 0;
        int drawn = 0;

        for (size_t i = 0; i < assoc_pts.size(); i += kStride) {
          const Eigen::Vector3d& pw = assoc_pts[i];
          if (!pw.allFinite()) continue;
          const Eigen::Vector3d pc = R_cw * pw + t_cw;
          if (pc.z() <= kMinProjZ) continue;

          const int u = static_cast<int>(std::round(fx * pc.x() / pc.z() + cx));
          const int v = static_cast<int>(std::round(fy * pc.y() / pc.z() + cy));
          if (u < 0 || v < 0 || u >= tracking_image.cols || v >= tracking_image.rows) continue;

          ++tot_cnt;
          const bool in_mask = has_motion_mask && (motion_mask.at<int>(v, u) == target_mask_val);
          if (in_mask) ++in_cnt;

          const cv::Scalar& col = in_mask ? obj_col : out_col;
          cv::circle(tracking_image, cv::Point(u, v), 1, col, -1, cv::LINE_AA);
          ++drawn;
          if (drawn >= kMaxProjPts) break;
        }

        if (tot_cnt >= kMinEdgePts && (static_cast<float>(in_cnt) / static_cast<float>(tot_cnt)) >= kEdgeRatioTh) {
          // Draw a small ratio tag near the last observed bbox (pixel space).
          const auto bboxes = obj->GetObservedBboxes();
          if (!bboxes.empty()) {
            const auto& bb = bboxes.back();
            int x = static_cast<int>(std::round(bb[0]));
            int y = static_cast<int>(std::round(bb[1]));
            x = std::max(0, std::min(x, tracking_image.cols - 1));
            y = std::max(0, std::min(y, tracking_image.rows - 1));
            const float ratio = tot_cnt > 0 ? (static_cast<float>(in_cnt) / static_cast<float>(tot_cnt)) : 0.f;
            const std::string txt = "edge " + std::to_string(ratio).substr(0, 4);
            cv::putText(tracking_image, txt, cv::Point(x, std::max(0, y - 3)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, obj_col, 1, cv::LINE_AA);
          }
        }
      }
    }
  }

  // ── BBox IoU visualization (Tracks window only) ──────────────────────
  // Draw IoU between object's last observed bbox and the bbox of the
  // corresponding target instance-mask pixels in current frame.
  if (map_) {
    const cv::Mat motion_mask = frame_k->image_container_.objectMotionMask();
    const bool has_motion_mask =
        !motion_mask.empty() && motion_mask.type() == CV_32SC1;

    if (has_motion_mask) {
      constexpr float kBboxIoUTh = 0.3f;
      constexpr double kPadRatio = 0.20;

      auto bboxIoU = [&](const dyno::BBox2& a, const dyno::BBox2& b) -> float {
        const double ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
        const double bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];
        const double ix1 = std::max(ax1, bx1);
        const double iy1 = std::max(ay1, by1);
        const double ix2 = std::min(ax2, bx2);
        const double iy2 = std::min(ay2, by2);
        const double iw = std::max(0.0, ix2 - ix1);
        const double ih = std::max(0.0, iy2 - iy1);
        const double inter = iw * ih;
        const double aw = std::max(0.0, ax2 - ax1);
        const double ah = std::max(0.0, ay2 - ay1);
        const double bw = std::max(0.0, bx2 - bx1);
        const double bh = std::max(0.0, by2 - by1);
        const double uni = aw * ah + bw * bh - inter;
        return uni > 0.0 ? static_cast<float>(inter / uni) : 0.f;
      };

      auto clampRect = [&](const dyno::BBox2& bb) -> cv::Rect {
        const int x1 = std::clamp(static_cast<int>(std::floor(bb[0])), 0,
                                  tracking_image.cols - 1);
        const int y1 = std::clamp(static_cast<int>(std::floor(bb[1])), 0,
                                  tracking_image.rows - 1);
        const int x2 = std::clamp(static_cast<int>(std::ceil(bb[2])), 0,
                                  tracking_image.cols - 1);
        const int y2 = std::clamp(static_cast<int>(std::ceil(bb[3])), 0,
                                  tracking_image.rows - 1);
        const int w = std::max(0, x2 - x1);
        const int h = std::max(0, y2 - y1);
        return cv::Rect(x1, y1, w, h);
      };

      const auto all_objs = map_->GetAllObjects();
      for (dyno::Object* obj : all_objs) {
        if (!obj || obj->isBad()) continue;

        const auto obs_bboxes = obj->GetObservedBboxes();
        if (obs_bboxes.empty()) continue;
        const dyno::BBox2 last_bb = obs_bboxes.back();  // [xmin,ymin,xmax,ymax]

        const int lbl = static_cast<int>(obj->GetCategoryId());
        const unsigned char target_b = static_cast<unsigned char>((lbl * 37) % 256);
        const unsigned char target_g = static_cast<unsigned char>((lbl * 17) % 256);
        const unsigned char target_r = static_cast<unsigned char>((lbl * 97) % 256);
        const int target_mask_val =
            (static_cast<int>(target_b) << 16) |
            (static_cast<int>(target_g) << 8) |
            (static_cast<int>(target_r));

        const double bb_w = std::max(1e-6, last_bb[2] - last_bb[0]);
        const double bb_h = std::max(1e-6, last_bb[3] - last_bb[1]);
        const double sx1 = last_bb[0] - kPadRatio * bb_w;
        const double sy1 = last_bb[1] - kPadRatio * bb_h;
        const double sx2 = last_bb[2] + kPadRatio * bb_w;
        const double sy2 = last_bb[3] + kPadRatio * bb_h;

        const int x0 = std::max(0, static_cast<int>(std::floor(sx1)));
        const int y0 = std::max(0, static_cast<int>(std::floor(sy1)));
        const int x1 = std::min(tracking_image.cols - 1,
                                 static_cast<int>(std::ceil(sx2)));
        const int y1 = std::min(tracking_image.rows - 1,
                                 static_cast<int>(std::ceil(sy2)));
        if (x1 <= x0 || y1 <= y0) continue;

        // mask bbox extraction (target label only)
        bool found_target = false;
        double mx1_t = 0.0, my1_t = 0.0, mx2_t = 0.0, my2_t = 0.0;

        for (int y = y0; y <= y1; ++y) {
          const int* row = motion_mask.ptr<int>(y);
          for (int x = x0; x <= x1; ++x) {
            const int mv = row[x];
            if (mv == target_mask_val) {
              if (!found_target) {
                mx1_t = mx2_t = x;
                my1_t = my2_t = y;
                found_target = true;
              } else {
                mx1_t = std::min(mx1_t, static_cast<double>(x));
                my1_t = std::min(my1_t, static_cast<double>(y));
                mx2_t = std::max(mx2_t, static_cast<double>(x));
                my2_t = std::max(my2_t, static_cast<double>(y));
              }
            }
          }
        }

        if (!found_target) continue;

        dyno::BBox2 mask_bb;
        mask_bb << mx1_t, my1_t, mx2_t + 1.0, my2_t + 1.0;

        const float iou = bboxIoU(last_bb, mask_bb);
        const cv::Scalar obj_col = obj->GetColor();  // BGR
        // Requested: visualize mask bbox with the same object color.
        // Encode pass/fail only with line thickness.
        const bool pass = (iou >= kBboxIoUTh);
        const int mask_thickness = pass ? 2 : 1;

        const cv::Rect last_rect = clampRect(last_bb);
        const cv::Rect mask_rect = clampRect(mask_bb);
        if (last_rect.area() > 0) cv::rectangle(tracking_image, last_rect, obj_col, 2);
        if (mask_rect.area() > 0) cv::rectangle(tracking_image, mask_rect, obj_col, mask_thickness);

        const std::string txt = "IoU=" + std::to_string(iou).substr(0, 6);
        const cv::Point txt_pt(last_rect.x, std::max(0, last_rect.y - 4));
        cv::putText(tracking_image, txt, txt_pt, cv::FONT_HERSHEY_SIMPLEX,
                    0.45, obj_col, 1, cv::LINE_AA);
      }
    }
  }

  return tracking_image;
}

cv::Mat RGBDInstanceFrontendModule::createObjectEdgeProjectionImage(
    const Frame::Ptr& frame_k, const ObjectPoseMap& object_poses) const {
  (void)object_poses;
  cv::Mat img =
      ImageType::RGBMono::toRGB(frame_k->image_container_.rgb()).clone();

  if (!map_) return img;

  const std::vector<dyno::Object*> all_objs = map_->GetAllObjects();
  if (all_objs.empty()) return img;
  const auto cur_frame_id = static_cast<size_t>(frame_k->getFrameId());

  const auto& camera_params = camera_->getParams();
  const auto& K = camera_params.getCameraMatrix();
  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);

  const gtsam::Pose3 T_cw = frame_k->T_world_camera_.inverse();  // camera <- world
  const Eigen::Matrix3d R_cw = T_cw.rotation().matrix();
  const Eigen::Vector3d t_cw = T_cw.translation();

  constexpr float kMinProjZ = 0.1f;
  constexpr int kStride = 1;        // dense projection for visibility
  constexpr int kMaxProjPts = 5000; // keep UI responsive
  size_t total_pts = 0;
  size_t z_valid_pts = 0;
  size_t in_image_pts = 0;
  size_t drawn_pts = 0;
  std::unordered_set<uint64_t> unique_pixels;

  for (dyno::Object* obj : all_objs) {
    if (!obj || obj->isBad()) continue;
    // Visualize only edges from objects observed before the current frame
    // (exclude objects newly created/updated at this frame).
    if (obj->GetLastObsFrameId() >= cur_frame_id) continue;

    const auto assoc_pts = obj->GetAssociatedMapPoints();
    if (assoc_pts.empty()) continue;
    const cv::Scalar obj_col = obj->GetColor();  // BGR

    int drawn = 0;
    for (size_t i = 0; i < assoc_pts.size(); i += kStride) {
      ++total_pts;
      const Eigen::Vector3d& pw = assoc_pts[i];
      if (!pw.allFinite()) continue;
      const Eigen::Vector3d pc = R_cw * pw + t_cw;
      if (pc.z() <= kMinProjZ) continue;
      ++z_valid_pts;

      const int u = static_cast<int>(std::round(fx * pc.x() / pc.z() + cx));
      const int v = static_cast<int>(std::round(fy * pc.y() / pc.z() + cy));
      if (u < 0 || v < 0 || u >= img.cols || v >= img.rows) continue;
      ++in_image_pts;
      unique_pixels.insert((static_cast<uint64_t>(v) << 32) |
                           static_cast<uint32_t>(u));

      cv::circle(img, cv::Point(u, v), 2, obj_col, -1, cv::LINE_AA);
      ++drawn_pts;
      if (++drawn >= kMaxProjPts) break;
    }
  }

  VLOG(1) << "[ObjEdgeProjLive] frame_id=" << frame_k->getFrameId()
          << " total_pts=" << total_pts
          << " z_valid=" << z_valid_pts
          << " in_image=" << in_image_pts
          << " drawn=" << drawn_pts
          << " unique_pix=" << unique_pixels.size();

  const std::string dbg_txt = "LIVE fid=" + std::to_string(frame_k->getFrameId()) +
                              " drawn=" + std::to_string(drawn_pts) +
                              " uniq=" + std::to_string(unique_pixels.size());
  cv::putText(img, dbg_txt, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX,
              0.65, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

  return img;
}

cv::Mat RGBDInstanceFrontendModule::createObjectEdgeProjectionImageFromSnapshot(
    const Frame::Ptr& frame_k,
    const std::vector<ObjectEdgeSnapshotItem>& snapshot) const {
  cv::Mat img =
      ImageType::RGBMono::toRGB(frame_k->image_container_.rgb()).clone();
  if (snapshot.empty()) return img;

  const auto& camera_params = camera_->getParams();
  const auto& K = camera_params.getCameraMatrix();
  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);

  const gtsam::Pose3 T_cw = frame_k->T_world_camera_.inverse();  // camera <- world
  const Eigen::Matrix3d R_cw = T_cw.rotation().matrix();
  const Eigen::Vector3d t_cw = T_cw.translation();

  constexpr float kMinProjZ = 0.1f;
  constexpr int kStride = 1;
  constexpr int kMaxProjPts = 5000;

  size_t total_pts = 0;
  size_t z_valid_pts = 0;
  size_t in_image_pts = 0;
  size_t drawn_pts = 0;
  std::unordered_set<uint64_t> unique_pixels;

  for (const auto& item : snapshot) {
    const cv::Scalar obj_col = item.color_bgr;  // BGR for OpenCV image overlay
    int drawn = 0;
    for (size_t i = 0; i < item.world_points.size(); i += kStride) {
      ++total_pts;
      const auto& pw_cv = item.world_points[i];
      const Eigen::Vector3d pw(pw_cv.x, pw_cv.y, pw_cv.z);
      const Eigen::Vector3d pc = R_cw * pw + t_cw;
      if (pc.z() <= kMinProjZ) continue;
      ++z_valid_pts;

      const int u = static_cast<int>(std::round(fx * pc.x() / pc.z() + cx));
      const int v = static_cast<int>(std::round(fy * pc.y() / pc.z() + cy));
      if (u < 0 || v < 0 || u >= img.cols || v >= img.rows) continue;
      ++in_image_pts;
      unique_pixels.insert((static_cast<uint64_t>(v) << 32) |
                           static_cast<uint32_t>(u));

      cv::circle(img, cv::Point(u, v), 2, obj_col, -1, cv::LINE_AA);
      ++drawn_pts;
      if (++drawn >= kMaxProjPts) break;
    }
  }

  VLOG(1) << "[ObjEdgeProjSnap] frame_id=" << frame_k->getFrameId()
          << " snapshot_objs=" << snapshot.size()
          << " total_pts=" << total_pts
          << " z_valid=" << z_valid_pts
          << " in_image=" << in_image_pts
          << " drawn=" << drawn_pts
          << " unique_pix=" << unique_pixels.size();

  // Overlay frame id + unique pixel count on the image so display/log alignment is obvious.
  const std::string dbg_txt = "SNAP fid=" + std::to_string(frame_k->getFrameId()) +
                              " drawn=" + std::to_string(drawn_pts) +
                              " uniq=" + std::to_string(unique_pixels.size());
  cv::putText(img, dbg_txt, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX,
              0.65, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

  return img;
}

// Edge-based local mapping functions
bool RGBDInstanceFrontendModule::shouldAddEdgeKeyFrame(
    const gtsam::Pose3& pose_curr, const gtsam::Pose3& pose_last) const {
  gtsam::Pose3 trans = pose_last.inverse() * pose_curr;
  
  gtsam::Matrix3 R_bias = trans.rotation().matrix();
  Eigen::AngleAxisd rotation_vector(R_bias);
  double theta = rotation_vector.angle() * 180.0 / M_PI;
  double translation = trans.translation().norm();
  
  return (theta > kf_rot_thres_ || translation > kf_trans_thres_);
}

void RGBDInstanceFrontendModule::processingThreadFunction() {
  int last_popped_kf_id = -1;
  while (processing_running_) {
    KeyFramePtr kf;
    optimization_queue_.pop(kf);  // Blocking pop from queue
    
    if (!kf) {
      // nullptr means shutdown signal
      break;
    }
    
    // Check if keyframes are popped in order
    if (last_popped_kf_id >= 0 && kf->KF_ID <= last_popped_kf_id) {
      LOG(WARNING) << "\033[31m[QUEUE ORDER ERROR]\033[0m last_kf_id=" << last_popped_kf_id
                   << ", current_kf_id=" << kf->KF_ID
                   << " (queue may not maintain FIFO order)";
    }
    last_popped_kf_id = kf->KF_ID;
    
    VLOG(1) << "[KF PROCESS] kf_id=" << kf->KF_ID
            << " frame_id=" << kf->KF_ID;
    VLOG(10) << "\033[34m[QUEUE POP]\033[0m kf_id=" << kf->KF_ID;
    
    // Process sliding window keyframe
    processSlidingWindowKeyFrame(kf);
  }
}

void RGBDInstanceFrontendModule::processSlidingWindowKeyFrame(KeyFramePtr kf) {
  if (!kf) {
    return;
  }
  
  const auto t0 = std::chrono::steady_clock::now();
  
  // Add keyframe to local map and perform optimization if window is full
  // Keep lock held throughout to match localmapping.cc behavior (addFrame2LocalMap -> clusterFittingProjection)
  size_t kf_count = 0;
  {
    std::lock_guard<std::mutex> lock(local_map_mutex_);
    
    // Log before adding keyframe
    size_t clusters_before = local_map_->mvEleEdgeClusters.size();
    size_t edges_before = 0;
    for(const auto& kf : local_map_->mvKeyFrames) {
      edges_before += kf->mvEdges.size();
    }
    
    // Log keyframe edges before adding to local map
    if (kf->mvEdges.empty()) {
      VLOG(1) << "WARNING: processSlidingWindowKeyFrame: kf(id=" << kf->KF_ID 
              << ") has 0 edges before adding to local_map!";
    }
    
    local_map_->addFrame2LocalMap(kf);
    kf_count = local_map_->mvKeyFrames.size();
    
    // Log keyframe edges after adding to local map
    if (kf->mvEdges.empty()) {
      VLOG(1) << "WARNING: processSlidingWindowKeyFrame: kf(id=" << kf->KF_ID 
              << ") has 0 edges after adding to local_map!";
    }
    
    // Note: processSlidingWindowKeyFrame receives KeyFrame from queue, not Frame
    // Frame object is already cached when keyframe was created in nominalSpin
    // So we don't update last_keyframe_ here
    
    // Log after adding keyframe
    size_t clusters_after = local_map_->mvEleEdgeClusters.size();
    size_t edges_after = 0;
    for(const auto& kf : local_map_->mvKeyFrames) {
      edges_after += kf->mvEdges.size();
    }
    
    // VLOG(1) << "After addFrame2LocalMap: kf_id=" << kf->KF_ID
    //         << " kf_count=" << kf_count
    //         << " clusters: " << clusters_before << " -> " << clusters_after
    //         << " (+" << (clusters_after - clusters_before) << ")"
    //         << " edges: " << edges_before << " -> " << edges_after
    //         << " (+" << (edges_after - edges_before) << ")";
    
    // This allows visualization of clusters before optimization
    if (local_map_ && local_map_->mvEleEdgeClusters.size() > 0) {
      std::vector<std::vector<cv::Point3d>> clusterClouds;
      std::vector<cv::Vec3b> clusterCloudColors;
      edge_viz::visualizeAssociationResult(local_map_, clusterClouds, clusterCloudColors);

      if (!clusterClouds.empty()) {
        cluster_clouds_cache_ =
            std::make_shared<const std::vector<std::vector<cv::Point3d>>>(
                std::move(clusterClouds));
        cluster_colors_cache_ =
            std::make_shared<const std::vector<cv::Vec3b>>(
                std::move(clusterCloudColors));
        VLOG(2) << "Updated cluster cache in processSlidingWindowKeyFrame: " 
                << cluster_clouds_cache_->size() << " clusters";
      }
    }
    
    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = [](const auto& a, const auto& b) -> double {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    
    VLOG(10) << "EdgeKF processed from queue kf_id=" << kf->KF_ID
            << " add_to_map_ms=" << ms(t0, t1)
            << " local_map_kfs=" << kf_count;
    
    // Check if window is full and perform optimization
    if (kf_count == window_size_) {
      // Perform optimization directly (already in processing thread)
      if (!optimization_in_progress_.load()) {
        optimization_in_progress_ = true;
      
        LOG(INFO) << "Starting edge sliding-window optimization";
        
        const auto t_opt_start = std::chrono::steady_clock::now();
        
        local_map_->clusterFittingProjection();
        const auto t1 = std::chrono::steady_clock::now();
        
        // LOG merged clusters after clusterFittingProjection
        int merged_after_fit = 0;
        for(const auto& cluster : local_map_->mvEleEdgeClusters) {
          if(cluster.mbMerged) merged_after_fit++;
        }
        // LOG(INFO) << "After clusterFittingProjection: merged_clusters=" << merged_after_fit
        //           << " / " << local_map_->mvEleEdgeClusters.size();
        
        dyno::Optimizer::optimizeAllInvolvedKFs(local_map_);

        // ── Per-object edge pipeline (tbb parallel) ──────────────────────────
        // Each object runs its own association → cluster → fitting → BA
        // pipeline using only the edges associated with that object.
        if (map_) {
          // Use ALL keyframes ever registered (not just sliding window)
          // so that per-object optimization uses every observation of that object.
          const std::vector<dyno::KeyFramePtr> all_kfs = map_->GetAllKeyFrames();
          std::vector<dyno::Object*> all_objs = map_->GetAllObjects();

          // Filter to objects that have at least one edge in the CURRENT window.
          // Objects not visible in the current window don't need re-optimization.
          const std::vector<dyno::KeyFramePtr>& window_kfs = local_map_->mvKeyFrames;
          std::unordered_set<int> active_obj_ids;
          for (const auto& kf : window_kfs) {
            if (!kf) continue;
            for (const auto& [edge_idx, obj_id] : kf->mmEdgeIndex2ObjectId) {
              if (obj_id >= 0) active_obj_ids.insert(obj_id);
            }
            // Also check edge.object_id as fallback
            for (size_t i = 0; i < kf->mvEdges.size(); ++i) {
              int oid = kf->mvEdges[i].object_id;
              if (oid >= 0) active_obj_ids.insert(oid);
            }
          }

          std::vector<dyno::Object*> active_objs;
          for (dyno::Object* obj : all_objs) {
            if (!obj || obj->isBad()) continue;
            if (active_obj_ids.count(static_cast<int>(obj->GetId())) == 0) continue;
            active_objs.push_back(obj);
          }

          if (!active_objs.empty()) {
            const auto t_obj_start = std::chrono::steady_clock::now();
            tbb::parallel_for_each(
                active_objs.begin(), active_objs.end(),
                [&all_kfs](dyno::Object* obj) {
                  obj->OptimizeWithEdgePipeline(all_kfs);
                });
            const auto t_obj_end = std::chrono::steady_clock::now();
            LOG(INFO) << "[PerObjectEdgePipeline] " << active_objs.size()
                      << " objects in "
                      << std::chrono::duration<double, std::milli>(
                             t_obj_end - t_obj_start).count()
                      << " ms (tbb parallel)";
          }
        }

        // ── Periodic CLIP-based object merge (disabled) ────────────────

        // Update merged local map cache (heavy data) for visualization snapshots
        {
          std::vector<std::vector<cv::Point3d>> mergedClouds;
          std::vector<cv::Vec3b> mergedCloudColors;
          if (local_map_ && local_map_->mvEleEdgeClusters.size() > 0) {
            edge_viz::visualizeMergedLocalMap(local_map_, mergedClouds, mergedCloudColors);
            if (!mergedClouds.empty()) {
              local_map_clouds_cache_ =
                  std::make_shared<const std::vector<std::vector<cv::Point3d>>>(
                      std::move(mergedClouds));
              local_map_colors_cache_ =
                  std::make_shared<const std::vector<cv::Vec3b>>(
                      std::move(mergedCloudColors));
              VLOG(2) << "Updated localMapClouds cache: " << local_map_clouds_cache_->size() << " merged clusters";
            }
          }

          // Accumulate environment cloud from merged local map
          if (local_map_clouds_cache_ && !local_map_clouds_cache_->empty()) {
            EnvironmentCloudFrame current;
            // Flatten merged local map clouds into a single frame cloud,
            // propagating per-cluster colors to each point.
            if (local_map_clouds_cache_ && local_map_colors_cache_ &&
                !local_map_colors_cache_->empty()) {
              const auto& clouds = *local_map_clouds_cache_;
              const auto& colors = *local_map_colors_cache_;
              const size_t n = std::min(clouds.size(), colors.size());
              for (size_t i = 0; i < n; ++i) {
                const auto& c = clouds[i];
                const cv::Vec3b& col = colors[i];
                for (const auto& pt : c) {
                  current.points.push_back(pt);
                  current.colors.push_back(col);
                }
              }
            }
            if (!current.points.empty()) {
              environment_frames_.push_back(std::move(current));
              while (environment_frames_.size() > 150) {
                environment_frames_.pop_front();
              }
              auto env = std::make_shared<std::vector<EnvironmentCloudFrame>>();
              env->reserve(environment_frames_.size());
              for (const auto& f : environment_frames_) {
                env->push_back(f);
              }
              environment_cloud_cache_ =
                  std::make_shared<const std::vector<EnvironmentCloudFrame>>(
                      std::move(*env));
              VLOG(2) << "Updated environment_cloud cache: " << environment_cloud_cache_->size() << " frames";
            }
          }
        }
        
        const auto t2 = std::chrono::steady_clock::now();
        
        VLOG(5) << "Edge sliding-window optimized (async)"
                << " cluster_fit_ms="
                << std::chrono::duration<double, std::milli>(t1 - t_opt_start).count()
                << " optimize_ms="
                << std::chrono::duration<double, std::milli>(t2 - t1).count()
                << " total_ms="
                << std::chrono::duration<double, std::milli>(t2 - t_opt_start).count();
        
        // Update sliding window after optimization (this will reset local map)
        // Keep lock held to match localmapping.cc behavior
        updateEdgeSlidingWindow();
        
        optimization_in_progress_ = false;
      } else {
        VLOG(1) << "Optimization already in progress, skipping";
      }
    } else {
      // Log when window is not full yet
      static int log_count = 0;
      if (++log_count % 50 == 0) {
        LOG(INFO) << "Edge sliding-window not full yet: kf_count=" << kf_count
                  << " < window_size=" << window_size_
                  << " (need " << (window_size_ - kf_count) << " more keyframes)";
      }
    }
  }
}

// void RGBDInstanceFrontendModule::processEdgeKeyFrame(
//     const Frame::Ptr& frame, const gtsam::Pose3& pose_curr) {
//   const auto t0 = std::chrono::steady_clock::now();
//   // Convert Frame to KeyFrame format (from localmapping.cc line 222)
//   const auto t_kf0 = std::chrono::steady_clock::now();
//   KeyFramePtr pKF = createKeyFrameFromFrame(frame, pose_curr);
//   const auto t_kf1 = std::chrono::steady_clock::now();
  
//   if (pKF) {
//     // Add to local map (from localmapping.cc line 226)
//     // Use mutex to protect local_map_ from concurrent access
//     // Lock only for the minimal time needed
//     const auto t_add0 = std::chrono::steady_clock::now();
//     size_t kf_count = 0;
//     {
//       std::lock_guard<std::mutex> lock(local_map_mutex_);
//       local_map_->addFrame2LocalMap(pKF);
//       kf_count = local_map_->mvKeyFrames.size();

//       // Update covisibility cluster cache (heavy data) for snapshots
//       // Always update if clusters exist, even if cache is not empty (clusters may have changed)
//       if (local_map_ && local_map_->mvEleEdgeClusters.size() > 0) {
//         std::vector<std::vector<cv::Point3d>> clusterClouds;
//         std::vector<cv::Vec3b> clusterCloudColors;
//         edge_viz::visualizeAssociationResult(local_map_, clusterClouds, clusterCloudColors);

//         if (!clusterClouds.empty()) {
//           cluster_clouds_cache_ =
//               std::make_shared<const std::vector<std::vector<cv::Point3d>>>(
//                   std::move(clusterClouds));
//           cluster_colors_cache_ =
//               std::make_shared<const std::vector<cv::Vec3b>>(
//                   std::move(clusterCloudColors));
//           // VLOG(2) << "Updated cluster cache in processEdgeKeyFrame: " 
//           //         << cluster_clouds_cache_->size() << " clusters (kf_id=" << pKF->KF_ID << ")";
//         } else {
//           // VLOG(2) << "processEdgeKeyFrame: clusters exist but visualizeAssociationResult returned empty (kf_id=" << pKF->KF_ID << ")";
//         }
//       } else {
//         // VLOG(2) << "processEdgeKeyFrame: no clusters to visualize (kf_id=" << pKF->KF_ID 
//                 // << ", clusters=" << (local_map_ ? local_map_->mvEleEdgeClusters.size() : 0) << ")";
//       }
//     }
//     const auto t_add1 = std::chrono::steady_clock::now();
    
//     const auto t1 = std::chrono::steady_clock::now();
//     const auto ms =
//         [](const auto& a, const auto& b) -> double {
//       return std::chrono::duration<double, std::milli>(b - a).count();
//     };
    
//     VLOG(1) << "EdgeKF frame=" << frame->getFrameId()
//             << " create_kf_ms=" << ms(t_kf0, t_kf1)
//             << " add_to_map_ms=" << ms(t_add0, t_add1)
//             << " total_ms=" << ms(t0, t1)
//             << " local_map_kfs=" << kf_count;


//     // Get depth data per detection from frame
//     const auto& depth_data_per_det = frame->getDepthDataPerDetection();
    
//     for(auto [node_id, attribute] : pKF->graph->attributes){
//       if(!attribute.obj){
//           //std::cout<<"not asscociated node id:"<<node_id<<std::endl;
//           //TODO check if match new

//           // Check if node_id is valid index for depth_data_per_det
//           if (node_id >= depth_data_per_det.size()) {
//               // VLOG(1) << "processEdgeKeyFrame: node_id=" << node_id 
//               //         << " is out of bounds for depth_data_per_det (size=" 
//               //         << depth_data_per_det.size() << ")";
//               continue;
//           }
          
//           const auto& depth_data = depth_data_per_det[node_id];
          
//           // Check depth validity (same as ObjectsInitialization)
//           if (depth_data.first <= 0.01f || depth_data.first >= 15.0f) {
//               VLOG(1) << "processEdgeKeyFrame: node_id=" << node_id 
//                       << " has invalid depth=" << depth_data.first;
//               continue;
//           }

//           Eigen::Matrix3d K_eigen = camera_->getParams().getCameraMatrixEigen();
//           // Construct Rt [R | t] from T_world_camera_
//           Matrix34d Rt;
//           Rt.block<3, 3>(0, 0) = frame->T_world_camera_.rotation().matrix();
//           Rt.block<3, 1>(0, 3) = frame->T_world_camera_.translation();
          
        
//           //create new object
//           Object* obj = new Object(
//               static_cast<unsigned int>(attribute.label),
//               attribute.bbox,  // BBox2 is typedef of Eigen::Vector4d
//               attribute.ell,
//               static_cast<double>(attribute.confidence),
//               depth_data,
//               K_eigen,
//               Rt,
//               static_cast<long unsigned int>(frame->getFrameId()),
//               pKF.get()  // Convert shared_ptr to raw pointer
//           );
//           // if(obj->GetAssociatedMapPoints().size()<5){
//           //     delete obj;
//           //     continue;
//           // }
//           map_->AddObject(obj);
//           pKF->graph->attributes[node_id].obj = obj;
//           //auto proj = obj->GetEllipsoid().project(P);
//           //auto c = proj.GetCenter();
//           //auto axes = proj.GetAxes();
//           //double angle = proj.GetAngle();
//           //cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, cv::Scalar(0, 255, 255), 2);
//           //if(axes[0] <= 0.001 || axes[1] <= 0.001)
//           //    continue;
//           //cv::ellipse(im_rgb_, cv::Point2f(c[0], c[1]), cv::Size2f(axes[0], axes[1]), TO_DEG(angle), 0, 360, obj->GetColor(), 2);
//       }
//     }

//   }
// }

KeyFramePtr RGBDInstanceFrontendModule::createKeyFrameFromFrame(
  const Frame::Ptr& frame, const gtsam::Pose3& pose_curr) {
// Convert Frame to KeyFrame (from localmapping.cc line 222)
// KeyFramePtr pKF(new KeyFrame(i, pose_curr, rgb_stamp_seq[i], 
//                              selector.mvEdges, imgRGB, imgDepth, 
//                              ph.fx, ph.fy, ph.cx, ph.cy));

  // Get images from Frame
const ImageContainer& img_container = frame->image_container_;
cv::Mat imgRGB = ImageType::RGBMono::toRGB(img_container.rgb());

cv::Mat imgDepth;
if (img_container.hasDepth()) {
  // ImageWrapper has implicit conversion to cv::Mat
  imgDepth = img_container.depth();
  // Convert depth scale if needed
  const CameraParams& cam_params = camera_->getParams();
  
  // Check if depth is already in meters (float/double type with reasonable range)
  bool already_in_meters = false;
  if (imgDepth.type() == CV_32F || imgDepth.type() == CV_64F) {
    double min_val, max_val;
    cv::minMaxLoc(imgDepth, &min_val, &max_val);
    // If depth values are in reasonable range (0.1m to 50m), assume already in meters
    if (min_val >= 0.0 && max_val > 0.1 && max_val < 50.0) {
      already_in_meters = true;
    }
  }
  
  if (already_in_meters) {
    // Already in meters, just ensure it's CV_32F
    if (imgDepth.type() == CV_64F) {
      imgDepth.convertTo(imgDepth, CV_32F);
    } else if (imgDepth.type() != CV_32F) {
      imgDepth.convertTo(imgDepth, CV_32F);
    }
  } else if (cam_params.hasDepthParams()) {
    // Need to convert from raw depth units to meters
    const double depth_scale = cam_params.depthParams().depth_to_meters;
    if (depth_scale != 1.0) {
      imgDepth.convertTo(imgDepth, CV_32F, depth_scale);
    } else if (imgDepth.type() != CV_32F) {
      // Ensure downstream code gets float depth even if scale == 1
      imgDepth.convertTo(imgDepth, CV_32F);
    }
  } else if (imgDepth.type() != CV_32F) {
    imgDepth.convertTo(imgDepth, CV_32F);
  }
  
} else {
  LOG_EVERY_N(WARNING, 100) << "Frame " << frame->getFrameId()
                            << " has no depth, cannot create edge KeyFrame";
  return nullptr;
}

// Use edges from Frame (already processed in tracker_->track())
// No need to process again with edgeSelector - frame->static_edges_ already contains the edges
VLOG(2) << "Using edges from Frame frame=" << frame->getFrameId()
        << " edges=" << frame->static_edges_.size();

// Validate images
CHECK(!imgRGB.empty()) << "RGB image is empty!";
CHECK(!imgDepth.empty()) << "Depth image is empty!";

// Convert gtsam::Pose3 to Sophus::SE3d for KeyFrame
const gtsam::Matrix4& T_matrix = pose_curr.matrix();
Sophus::SE3d pose_sophus(
    Sophus::SO3d(T_matrix.topLeftCorner<3, 3>()), 
    T_matrix.topRightCorner<3, 1>());

// Create KeyFrame
FrameId frame_id = frame->getFrameId();
double timestamp = frame->getTimestamp();
const CameraParams& cam_params = camera_->getParams();

const auto t_kf_ctor0 = std::chrono::steady_clock::now();
KeyFramePtr pKF(new KeyFrame(
    frame_id, pose_sophus, timestamp, 
    frame->static_edges_, imgRGB, imgDepth,
    cam_params.fx(), cam_params.fy(), 
    cam_params.cu(), cam_params.cv()));
const auto t_kf_ctor1 = std::chrono::steady_clock::now();

// Copy edge grid from Frame to KeyFrame (same pattern as ORB-SLAM2 F.mGrid -> mGrid)
pKF->mGrid.clear();
pKF->mGrid.resize(dyno::FRAME_GRID_COLS);
for (int i = 0; i < dyno::FRAME_GRID_COLS; ++i) {
  pKF->mGrid[i].resize(dyno::FRAME_GRID_ROWS);
  for (int j = 0; j < dyno::FRAME_GRID_ROWS; ++j) {
    pKF->mGrid[i][j] = frame->mGrid[i][j];
  }
}

// Copy graph from Frame to KeyFrame (graph contains object detection attributes)
if (frame->graph) {
  pKF->graph = frame->graph;
}

// Assign object ids to edges in KeyFrame based on segmentation mask (not just bbox)
if (pKF->graph) {
  // Initialize all edge indices to background (-1)
  pKF->mmEdgeIndex2ObjectId.clear();

  // Use motion/instance mask from the originating Frame
  const cv::Mat& motion_mask = frame->image_container_.objectMotionMask();
  if (!motion_mask.empty()) {
    // For each node (detection) in the graph, use its bbox as a coarse window,
    // then check membership using the mask value at each edge point.
    for (const auto& kv : pKF->graph->attributes) {
      const auto& attr = kv.second;
      // Use obj->GetId() if object is already associated, otherwise use attr.object_id
      int obj_id = -1;
      if (attr.obj) {
        obj_id = static_cast<int>(attr.obj->GetId());
      } else {
        obj_id = attr.object_id;  // fallback to detection's object_id
      }
      const int cls_label = attr.label;    // category id from detection
      // Compute expected BGR color for this category_id (matching generate_detection_files.py)
      // B = (category_id * 37) % 256
      // G = (category_id * 17) % 256
      // R = (category_id * 97) % 256
      const unsigned char target_b = static_cast<unsigned char>((cls_label * 37) % 256);
      const unsigned char target_g = static_cast<unsigned char>((cls_label * 17) % 256);
      const unsigned char target_r = static_cast<unsigned char>((cls_label * 97) % 256);
      // Pack into 32-bit int: (B << 16) | (G << 8) | R
      const int target_mask_val = (static_cast<int>(target_b) << 16) | 
                                  (static_cast<int>(target_g) << 8) | 
                                  static_cast<int>(target_r);
      const Eigen::Vector4d& bb = attr.bbox;      // [xmin, ymin, xmax, ymax]

      // Get edge/point indices whose points fall inside this bbox (coarse)
      std::vector<std::size_t> encoded_indices =
          pKF->GetEdgeIndicesInBox(static_cast<float>(bb[0]),
                                   static_cast<float>(bb[2]),
                                   static_cast<float>(bb[1]),
                                   static_cast<float>(bb[3]));

      for (std::size_t enc : encoded_indices) {
        const int edge_id = static_cast<int>(enc / 100000);  // encoded edge id
        const int pt_idx  = static_cast<int>(enc % 100000);  // encoded point index
        auto itEdge = pKF->mmIndexMap.find(edge_id);
        if (itEdge == pKF->mmIndexMap.end()) {
          continue;
        }
        const int edge_idx = itEdge->second;
        if (edge_idx < 0 || edge_idx >= static_cast<int>(pKF->mvEdges.size())) {
          continue;
        }
        Edge& edge = pKF->mvEdges[edge_idx];
        if (pt_idx < 0 || pt_idx >= static_cast<int>(edge.mvPoints.size())) {
          continue;
        }

        // Sample mask at this edge point
        const orderedEdgePoint& pt = edge.mvPoints[pt_idx];
        int u = static_cast<int>(std::round(pt.x));
        int v = static_cast<int>(std::round(pt.y));
        if (u < 0 || v < 0 || u >= motion_mask.cols || v >= motion_mask.rows) {
          continue;
        }

        // Read packed BGR value from mask (stored as 32-bit int)
        int mask_val = motion_mask.at<int>(v, u);
        if (mask_val != target_mask_val) {
          continue;  // point not inside this object's mask (color doesn't match)
        }

        // Prefer keeping existing non-background assignment if already set
        auto it = pKF->mmEdgeIndex2ObjectId.find(edge_idx);
        if (it != pKF->mmEdgeIndex2ObjectId.end() && it->second >= 0) {
          continue;
        }

        pKF->mmEdgeIndex2ObjectId[edge_idx] = obj_id;
        // Also store on the Edge itself for convenience
        edge.object_id = obj_id;
        edge.color = (attr.obj) ? attr.obj->GetColor() : cv::Scalar(150, 150, 150);
      }
    }
  }
}

return pKF;
}

void RGBDInstanceFrontendModule::optimizeEdgeSlidingWindow() {
  // This function is kept for backward compatibility
  // Optimization is now handled directly in processSlidingWindowKeyFrame()
  LOG(WARNING) << "optimizeEdgeSlidingWindow() called directly - optimization is handled in processSlidingWindowKeyFrame()";
}

void RGBDInstanceFrontendModule::updateEdgeSlidingWindow() {
  // From localmapping.cc lines 251-269
  // Note: This function should be called from optimizationThreadFunction
  // with local_map_mutex_ already locked
  const auto t0 = std::chrono::steady_clock::now();
  
  size_t current_kf_count = local_map_->mvKeyFrames.size();
  
  // // Log keyframes before removal
  // LOG(INFO) << "\033[33m[SLIDING WINDOW UPDATE]\033[0m before: kf_count=" << current_kf_count
  //           << ", window_size=" << window_size_ << ", window_step=" << window_step_;
  // if (current_kf_count > 0) {
  //   std::string kf_ids_before = "kf_ids=[";
  //   for (size_t i = 0; i < std::min(current_kf_count, size_t(10)); ++i) {
  //     if (i > 0) kf_ids_before += ", ";
  //     kf_ids_before += std::to_string(local_map_->mvKeyFrames[i]->KF_ID);
  //   }
  //   if (current_kf_count > 10) kf_ids_before += ", ...";
  //   kf_ids_before += "]";
  //   LOG(INFO) << "  " << kf_ids_before;
  // }
  
  // Save poses and timestamps of removed keyframes
  std::vector<Eigen::Matrix4d> removed_poses;
  std::vector<double> removed_stamps;
  std::vector<int> removed_kf_ids;
  
  for (int j = 0; j < window_step_; ++j) {
    removed_kf_ids.push_back(local_map_->mvKeyFrames[j]->KF_ID);
    double removed_stamp = local_map_->mvKeyFrames[j]->KF_stamp;
    Eigen::Matrix4d removed_pose = 
        local_map_->mvKeyFrames[j]->KF_pose_g.matrix();
    
    removed_poses.push_back(removed_pose);
    removed_stamps.push_back(removed_stamp);
  }
  
  // LOG(INFO) << "  Removing " << window_step_ << " keyframes: kf_ids=["
  //           << (removed_kf_ids.empty() ? "" : std::to_string(removed_kf_ids[0]));
  // for (size_t i = 1; i < removed_kf_ids.size(); ++i) {
  //   LOG(INFO) << ", " << removed_kf_ids[i];
  // }
  // LOG(INFO) << "]";
  
  // Keep overlapping keyframes
  std::vector<KeyFramePtr> newKFs(
      local_map_->mvKeyFrames.begin() + window_step_,
      local_map_->mvKeyFrames.begin() + window_size_);
  
  // std::string kept_kf_ids = "kf_ids=[";
  // for (size_t i = 0; i < newKFs.size(); ++i) {
  //   if (i > 0) kept_kf_ids += ", ";
  //   kept_kf_ids += std::to_string(newKFs[i]->KF_ID);
  // }
  // kept_kf_ids += "]";
  // LOG(INFO) << "  Keeping " << newKFs.size() << " keyframes: " << kept_kf_ids;
  
  // Reset local map and add overlapping keyframes
  local_map_.reset(new dyno::localMap());
  
  // Log keyframe edges before adding
  size_t total_edges_before = 0;
  for (int j = 0; j < window_size_ - window_step_; ++j) {
    size_t kf_edges = newKFs[j]->mvEdges.size();
    total_edges_before += kf_edges;
    if (kf_edges == 0) {
      LOG(WARNING) << "updateEdgeSlidingWindow: newKFs[" << j << "] (id=" 
              << newKFs[j]->KF_ID << ") has 0 edges!";
    }
  }
  for (int j = 0; j < window_size_ - window_step_; ++j) {
    // Log keyframe edges before clearing and re-adding
    if (newKFs[j]->mvEdges.empty()) {
      LOG(WARNING) << "updateEdgeSlidingWindow: newKFs[" << j << "] (id=" 
              << newKFs[j]->KF_ID << ") already has 0 edges before re-adding!";
    }
    
    newKFs[j]->mmEdgeIndex2ElementEdgeID.clear();
    newKFs[j]->mmMapAssociations.clear();
    local_map_->addFrame2LocalMap(newKFs[j]);
    
    // Log keyframe edges after re-adding
    if (newKFs[j]->mvEdges.empty()) {
      LOG(WARNING) << "updateEdgeSlidingWindow: newKFs[" << j << "] (id=" 
              << newKFs[j]->KF_ID << ") has 0 edges after re-adding!";
    }
    
    // Log after each keyframe addition
    if (j == 0) {
      VLOG(2) << "  After adding KF[0] (id=" << newKFs[j]->KF_ID 
              << ", edges=" << newKFs[j]->mvEdges.size() 
              << "): local_map kfs=" << local_map_->mvKeyFrames.size()
              << ", element_edges=" << local_map_->mvElementEdges.size()
              << ", clusters=" << local_map_->mvEleEdgeClusters.size()
              << ", state=" << (local_map_->msState == dyno::localMap::State::NOT_INITIALIZED ? "NOT_INIT" : "INIT");
    } else if (j == 1) {
      VLOG(2) << "  After adding KF[1] (id=" << newKFs[j]->KF_ID 
              << ", edges=" << newKFs[j]->mvEdges.size() 
              << "): local_map kfs=" << local_map_->mvKeyFrames.size()
              << ", element_edges=" << local_map_->mvElementEdges.size()
              << ", clusters=" << local_map_->mvEleEdgeClusters.size()
              << ", state=" << (local_map_->msState == dyno::localMap::State::NOT_INITIALIZED ? "NOT_INIT" : "INIT");
    }
  }
  
  // After adding keyframes, if we have 2 or more keyframes and state is still NOT_INITIALIZED,
  // initLocalMap() should have been called by addFrame2LocalMap. But if it wasn't (e.g., 
  // we added exactly 2 keyframes in the loop above), we need to ensure initialization.
  // Actually, addFrame2LocalMap handles this automatically when mvKeyFrames.size() == 2,
  // so we just need to make sure clusters are created. Let's verify the state.
  if (local_map_->mvKeyFrames.size() >= 2 && local_map_->msState == dyno::localMap::State::NOT_INITIALIZED) {
    // This shouldn't happen if addFrame2LocalMap worked correctly, but let's be safe
    LOG(WARNING) << "updateEdgeSlidingWindow: WARNING - local_map has " << local_map_->mvKeyFrames.size() 
            << " keyframes but state is NOT_INITIALIZED";
  }
  
  // Update cluster cache after sliding window update
  // Log detailed state for debugging
  if (local_map_) {
    size_t element_edges_count = local_map_->mvElementEdges.size();
    size_t clusters_count = local_map_->mvEleEdgeClusters.size();
    size_t kf_count = local_map_->mvKeyFrames.size();
    VLOG(2) << "updateEdgeSlidingWindow: after update - kf_count=" << kf_count
            << ", element_edges=" << element_edges_count
            << ", clusters=" << clusters_count
            << ", state=" << (local_map_->msState == dyno::localMap::State::NOT_INITIALIZED ? "NOT_INIT" : 
                              local_map_->msState == dyno::localMap::State::INITIALIZED ? "INIT" : "LOST");
    
    if (clusters_count > 0) {
      std::vector<std::vector<cv::Point3d>> clusterClouds;
      std::vector<cv::Vec3b> clusterCloudColors;
      edge_viz::visualizeAssociationResult(local_map_, clusterClouds, clusterCloudColors);
      
      if (!clusterClouds.empty()) {
        cluster_clouds_cache_ =
            std::make_shared<const std::vector<std::vector<cv::Point3d>>>(
                std::move(clusterClouds));
        cluster_colors_cache_ =
            std::make_shared<const std::vector<cv::Vec3b>>(
                std::move(clusterCloudColors));
      } else {
        LOG(WARNING) << "updateEdgeSlidingWindow: clusters exist but visualizeAssociationResult returned empty";
      }
    } else {
      LOG(WARNING) << "updateEdgeSlidingWindow: no clusters after update (element_edges=" 
              << element_edges_count << ")";
    }
  }
  
  // Note: updateEdgeSlidingWindow only has KeyFrame objects, not Frame objects
  // Frame object is already cached when keyframe was created in nominalSpin
  // So we don't update last_keyframe_ here - it will be updated when next keyframe is created
  
  const auto t1 = std::chrono::steady_clock::now();
  // LOG(INFO) << "  After update: kf_count=" << local_map_->mvKeyFrames.size()
  //           << " rebuild_ms="
  //           << std::chrono::duration<double, std::milli>(t1 - t0).count();
}


void RGBDInstanceFrontendModule::ObjectsInitialization(const Frame::Ptr& frame, dyno::KeyFrame* kf){
  // Get static_detection_result from frame (set by FeatureTracker from ImageContainer)
  if (!frame->image_container_.hasStaticDetectionResult() ||
      frame->image_container_.staticDetectionResult().detections.empty()) {
    LOG(WARNING) << "WARNING: NO DETECTION IN THE INITIALIZATION FRAME";
    return;
  }
  const auto& static_detection_result = frame->image_container_.staticDetectionResult();
  const auto& depth_data_per_det = frame->getDepthDataPerDetection();
  
  if (frame->graph == nullptr || depth_data_per_det.size() != static_detection_result.detections.size()) {
    LOG(WARNING) << "ObjectsInitialization: graph or depth_data size mismatch";
    return;
  }
  
  // Get K from camera params
  Eigen::Matrix3d K_eigen = camera_->getParams().getCameraMatrixEigen();
  // Construct Rt [R_cw | t_cw] from T_world_camera_
  const gtsam::Pose3& T_wc = frame->T_world_camera_;
  const gtsam::Pose3  T_cw = T_wc.inverse();
  Matrix34d Rt;
  Rt.block<3, 3>(0, 0) = T_cw.rotation().matrix();
  Rt.block<3, 1>(0, 3) = T_cw.translation();
  
  int count = 0;
  for (size_t di = 0; di < static_detection_result.detections.size(); ++di) {
      const auto& det = static_detection_result.detections[di];
      if (di >= depth_data_per_det.size()) continue;
      
      // Filter by confidence score
      if (det.score < kMinConfidenceScore_) {
          continue;
      }
      
      const auto& depth_data = depth_data_per_det[di];
      
      if(depth_data.first > 0.01f && depth_data.first < 15.0f){
          dyno::Object* obj = new dyno::Object(det.category_id, det.bbox, det.ell, det.score, depth_data, K_eigen,
                        Rt, 0, kf);
          map_->AddObject(obj);
          local_map_->mlpRecentAddedObjects.push_back(obj);
          count += 1;
      }
  }
  // std::vector<dyno::Object*> objects = map_->GetAllObjects();
  // LOG(INFO) << "ObjectsInitialization: created " << count << " objects, Map has " << objects.size() << " objects";
}

void RGBDInstanceFrontendModule::ObjectCulling(const KeyFramePtr& pKF) {
  std::list<Object*>::iterator lit = local_map_->mlpRecentAddedObjects.begin();
  const unsigned long int nCurrentKFid = pKF->KF_ID;
  
  while(lit!=local_map_->mlpRecentAddedObjects.end())
  {
      Object* obj = *lit;
      if(obj->isBad())
      {
          lit = local_map_->mlpRecentAddedObjects.erase(lit);
      }
      else if(((int)nCurrentKFid-(int)obj->mnLastKFid)>=5 && obj->GetObservationNumber()<=2){
          obj->SetBadFlag();
          lit = local_map_->mlpRecentAddedObjects.erase(lit);
      }
      else if(((int)nCurrentKFid-(int)obj->mnLastKFid)>=6)
          lit = local_map_->mlpRecentAddedObjects.erase(lit);
      else
          lit++;
  }
}

}  // namespace dyno
