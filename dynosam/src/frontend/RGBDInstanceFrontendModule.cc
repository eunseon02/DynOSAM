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

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam_common/Flags.hpp"  //for common flags
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/utils/SafeCast.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_cv/RGBDCamera.hpp"


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

DEFINE_bool(use_edge_selector_track, false,
            "If true, use edgeSelector.processImage() + direct KeyFrame creation "
            "instead of tracker_->track() (localmapping.cc style)");

namespace dyno {

RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(
    const DynoParams& params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : FrontendModule(params, display_queue),
      camera_(camera),
      motion_solver_(params.frontend_params_.ego_motion_solver_params,
                     camera->getParams()),
      imu_frontend_(params.frontend_params_.imu_params) {
  CHECK_NOTNULL(camera_);
  tracker_ = std::make_unique<FeatureTracker>(getFrontendParams(), camera_,
                                              display_queue);
  fine_tracker_ = std::make_unique<fine::FineTracker>(camera->getParams().fx(), camera->getParams().fy(), camera->getParams().cu(), camera->getParams().cv(), getFrontendParams().tracker_params.edge_fine.geo_photo_ratio);
  direct_tracker_ = std::make_unique<direct::DirectTracker>(camera->getParams().ImageWidth(), camera->getParams().ImageHeight(), camera->getParams().fx(), camera->getParams().fy(), camera->getParams().cu(), camera->getParams().cv());
  
  // Initialize edge-based local mapping (from localmapping.cc)
  local_map_.reset(new edge_map::localMap());
  // TODO: Load canny parameters from config
  edge_selector_ = std::make_unique<edgeSelector>(20.0, 50, 150);
  
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

  Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                                     *image_container);
  CHECK(frame->updateDepths());

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

  // Use DirectTrack instead of solveCameraMotion
  // Calculate initial relative pose estimate (can use IMU prediction or constant velocity)
  VLOG(10) << "RGBDInstanceFrontendModule::nominalSpin: about to call DirectTrack";
  gtsam::Pose3 T_k_1_k_initial;
  if (last_imu_k_ == frame->getFrameId() && has_imu) {
    // Use IMU prediction if available
    const gtsam::Pose3 T_world_k_1 = nav_state_prev_.pose();
    const gtsam::Pose3 T_world_k = nav_state_curr_.pose();
    T_k_1_k_initial = T_world_k_1.inverse() * T_world_k;
    VLOG(10) << "RGBDInstanceFrontendModule::nominalSpin: using IMU prediction for initial pose";
  } else {
    // Use constant velocity model
    T_k_1_k_initial = vo_velocity_;
    VLOG(10) << "RGBDInstanceFrontendModule::nominalSpin: using constant velocity model for initial pose";
  }
  
  // Perform DirectTrack
  VLOG(10) << "RGBDInstanceFrontendModule::nominalSpin: calling DirectTrack";
  const auto t_direct_start = std::chrono::steady_clock::now();
  gtsam::Pose3 T_k_1_k_refined;
  if (!DirectTrack(frame, previous_frame, T_k_1_k_initial, T_k_1_k_refined)) {
    VLOG(5) << "DirectTrack failed, using initial pose estimate";
    T_k_1_k_refined = T_k_1_k_initial;
  }
  const auto t_direct_end = std::chrono::steady_clock::now();
  VLOG(10) << "RGBDInstanceFrontendModule::nominalSpin: DirectTrack completed";
  
  // Update frame pose with DirectTrack result
  frame->T_world_camera_ = previous_frame->T_world_camera_ * T_k_1_k_refined;
  
  // Calculate relative pose from DirectTrack result for FineTrack (if edge features enabled)
  const auto t_fine_start = std::chrono::steady_clock::now();
  if (FLAGS_use_edge_feature) {
    gtsam::Pose3 T_k_1_k_fine_refined;
    if (!FineTrack(frame, previous_frame, T_k_1_k_refined, T_k_1_k_fine_refined)) {
      VLOG(5) << "Could not fine track";
    } else {
      // Update frame pose with refined result from FineTrack
      frame->T_world_camera_ = previous_frame->T_world_camera_ * T_k_1_k_fine_refined;
    }
  }
  const auto t_fine_end = std::chrono::steady_clock::now();



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
      const auto t_create_kf_start = std::chrono::steady_clock::now();
      // Calculate pose for keyframe
      gtsam::Pose3 pose_curr;
      if (local_map_empty) {
        pose_curr = pose_curr_coarse;
      } else {
        // Use last optimized pose and relative motion
        gtsam::Pose3 pose_last_prior = pose_curr_coarse; // Simplified
        gtsam::Pose3 pose_bias = pose_last_prior.inverse() * pose_curr_coarse;
        pose_curr = pose_last_kf * pose_bias;
      }
      
      // Create keyframe in nominalSpin (synchronous)
      KeyFramePtr pKF = createKeyFrameFromFrame(frame, pose_curr);
      const auto t_create_kf_end = std::chrono::steady_clock::now();
      
      if (pKF) {
        // Push to queue for async processing
        optimization_queue_.push(pKF);
      }
      
      pose_last_edge_kf_ = pose_curr_coarse;
      
      LOG_EVERY_N(INFO, 30) << "CreateKeyFrame frame=" << frame->getFrameId()
                            << " ms=" << std::chrono::duration<double, std::milli>(
                                t_create_kf_end - t_create_kf_start).count();
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
  
  // Log processing times
  const char* CYAN = "\033[36m";
  const char* RESET = "\033[0m";
  
  static int log_counter = 0;
  if (++log_counter % 10 == 0) {
    std::ostringstream oss;
    oss << CYAN << "nominalSpin frame=" << frame->getFrameId() << "\n"
        << "  ├─ track_ms: " << ms(t_track_start, t_track_end) << "\n"
        << "  ├─ update_depths_ms: " << ms(t_update_depths_start, t_update_depths_end) << "\n"
        << "  ├─ stereo1_ms: " << ms(t_stereo1_start, t_stereo1_end) << "\n"
        << "  ├─ direct_ms: " << ms(t_direct_start, t_direct_end) << "\n"
        << "  ├─ fine_ms: " << ms(t_fine_start, t_fine_end) << "\n"
        << "  ├─ stereo2_ms: " << ms(t_stereo2_start, t_stereo2_end) << "\n"
        << "  ├─ object_motion_ms: " << ms(t_object_motion_start, t_object_motion_end) << "\n"
        << "  ├─ fill_packet_ms: " << ms(t_fill_packet_start, t_fill_packet_end) << "\n"
        << "  ├─ create_image_ms: " << ms(t_create_image_start, t_create_image_end) << "\n"
        << "  ├─ send_logger_ms: " << ms(t_send_logger_start, t_send_logger_end) << "\n"
        << "  └─ total_ms: " << ms(t_nominal_start, t_nominal_end) << RESET;
    LOG(INFO) << oss.str();
  }

  // if (FLAGS_log_projected_masks)
  //   vision_tools::writeOutProjectMaskAndDepthMap(
  //       frame->image_container_.depth(),
  //       frame->image_container_.objectMotionMask(), *frame->getCamera(),
  //       frame->getFrameId());

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
  // LOG(INFO) << "DirectTrack: entered function";
  
  // Check if direct_tracker_ is initialized
  if (!direct_tracker_) {
    // LOG(ERROR) << "DirectTrack: direct_tracker_ is null!";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  
  // Validate frame image containers
  // LOG(INFO) << "DirectTrack: validating frame image containers";
  if (!frame_k_1->image_container_.hasRgb() || !frame_k->image_container_.hasRgb()) {
    // LOG(WARNING) << "DirectTrack: frame image containers missing RGB, skipping";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  
  // Get grayscale images
  // LOG(INFO) << "DirectTrack: converting RGB to mono";
  const ImageWrapper<ImageType::RGBMono>& rgb_wrapper_ref = frame_k_1->image_container_.rgb();
  const ImageWrapper<ImageType::RGBMono>& rgb_wrapper_cur = frame_k->image_container_.rgb();
  cv::Mat mono_ref = ImageType::RGBMono::toMono(rgb_wrapper_ref);
  cv::Mat mono_cur = ImageType::RGBMono::toMono(rgb_wrapper_cur);
  
  if (mono_ref.empty() || mono_cur.empty()) {
    // LOG(WARNING) << "DirectTrack: failed to convert RGB to mono";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  // LOG(INFO) << "DirectTrack: mono images converted, ref.size()=" << mono_ref.size() << ", cur.size()=" << mono_cur.size();
  
  // Collect static features with depth from previous frame
  // LOG(INFO) << "DirectTrack: collecting tracklets";
  TrackletIds tracklets = frame_k_1->static_features_.collectTracklets();
  if (tracklets.empty()) {
    // LOG(WARNING) << "DirectTrack: no tracklets available";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  // LOG(INFO) << "DirectTrack: collected " << tracklets.size() << " tracklets";
  
  std::vector<float> x_list, y_list, depth_list, weight_list, theta_list;
  // LOG(INFO) << "DirectTrack: filtering features with depth";
  for (const auto& tracklet_id : tracklets) {
    Feature::Ptr feature = frame_k_1->static_features_.getByTrackletId(tracklet_id);
    if (!feature || !feature->usable() || !feature->hasDepth()) {
      continue;
    }
    
    const Keypoint& kp = feature->keypoint();
    x_list.push_back(kp(0));
    y_list.push_back(kp(1));
    depth_list.push_back(feature->depth());
    weight_list.push_back(1.0f);  // Default weight
    theta_list.push_back(0.0f);  // Default theta (gradient angle)
  }
  
  if (x_list.empty()) {
    // LOG(WARNING) << "DirectTrack: no valid features with depth";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  // LOG(INFO) << "DirectTrack: found " << x_list.size() << " features with depth";
  
  // Set reference frame
  // LOG(INFO) << "DirectTrack: calling setReference";
  try {
    direct_tracker_->setReference(mono_ref, x_list, y_list, depth_list, weight_list, theta_list);
  } catch (const std::exception& e) {
    // LOG(ERROR) << "DirectTrack: setReference failed with exception: " << e.what();
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  // LOG(INFO) << "DirectTrack: setReference completed";
  
  // Set current frame
  // LOG(INFO) << "DirectTrack: calling setCurrent";
  try {
    direct_tracker_->setCurrent(mono_cur);
  } catch (const std::exception& e) {
    // LOG(ERROR) << "DirectTrack: setCurrent failed with exception: " << e.what();
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  // LOG(INFO) << "DirectTrack: setCurrent completed";
  
  // Convert gtsam::Pose3 to Sophus::SE3d for DirectTracker
  // DirectTracker expects T_cur_ref (current to reference), which is T_k_k_1 = T_k_1_k^-1
  // LOG(INFO) << "DirectTrack: converting pose to Sophus::SE3d";
  const gtsam::Pose3 T_k_k_1_initial = T_k_1_k_initial.inverse();
  const gtsam::Matrix4& T_matrix = T_k_k_1_initial.matrix();
  Sophus::SE3d T21(Sophus::SO3d(T_matrix.topLeftCorner<3, 3>()), 
                   T_matrix.topRightCorner<3, 1>());
  // LOG(INFO) << "DirectTrack: pose conversion completed";
  
  // Run DirectTracker estimation
  // LOG(INFO) << "DirectTrack: calling estimatePyramid";
  try {
    direct_tracker_->estimatePyramid(T21, true);
    // LOG(INFO) << "DirectTrack: estimatePyramid completed";
  } catch (const std::runtime_error& e) {
    // LOG(WARNING) << "DirectTracker failed with error: " << e.what() 
    //              << ". Falling back to initial pose.";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  } catch (const std::exception& e) {
    // LOG(WARNING) << "DirectTracker failed with exception: " << e.what() 
    //              << ". Falling back to initial pose.";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  } catch (...) {
    LOG_EVERY_N(WARNING, 50)
        << "DirectTracker failed with unknown exception. Falling back to initial pose.";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  
  // Check if tracking was successful
  if (!T21.matrix().allFinite()) {
    LOG_EVERY_N(WARNING, 50) << "DirectTracker returned invalid pose matrix";
    T_k_1_k_refined = T_k_1_k_initial;
    return false;
  }
  
  // Convert Sophus::SE3d result back to gtsam::Pose3
  // DirectTracker returns T_cur_ref, so we need to invert to get T_k_1_k
  VLOG(10) << "DirectTrack: converting result back to gtsam::Pose3";
  const Eigen::Matrix4d T_result = T21.inverse().matrix();
  T_k_1_k_refined = gtsam::Pose3(T_result);
  VLOG(10) << "DirectTrack: completed successfully";
  
  return true;
}

bool RGBDInstanceFrontendModule::FineTrack(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,
                                           const gtsam::Pose3& T_k_1_k_initial, 
                                           gtsam::Pose3& T_k_1_k_refined) {
  utils::ChronoTimingStats timer("frontend.fine_track");
  
  // Count edge points
  size_t ref_edge_points = 0;
  size_t cur_edge_points = 0;
  size_t ref_sampled_points = 0;
  
  for (const auto& edge : frame_k_1->static_edges_) {
    ref_edge_points += edge.mvPoints.size();
    ref_sampled_points += edge.mvSampledEdgeIndex.size();
  }
  
  for (const auto& edge : frame_k->static_edges_) {
    cur_edge_points += edge.mvPoints.size();
  }
  
  if (ref_edge_points == 0 || cur_edge_points == 0) {
    VLOG(5) << "FineTrack: edges have no points (ref: " << ref_edge_points
             << ", cur: " << cur_edge_points << "), skipping";
    return false;
  }

  VLOG(5) << "FineTrack: edge features available (ref edges: " 
           << frame_k_1->static_edges_.size() << " with " << ref_edge_points 
           << " points (" << ref_sampled_points << " sampled), cur edges: " 
           << frame_k->static_edges_.size() << " with " << cur_edge_points 
           << " points)";
  
  // Validate frame image containers before FineTracker estimation
  if (!frame_k_1->image_container_.hasRgb() || !frame_k->image_container_.hasRgb()) {
    VLOG(5) << "FineTrack: frame image containers missing RGB, skipping";
    return false;
  }
  
  // Convert gtsam::Pose3 to Sophus::SE3d for FineTracker
  // FineTracker expects T_cur_ref (current to reference), which is T_k_k_1 = T_k_1_k^-1
  const gtsam::Pose3 T_k_k_1_initial = T_k_1_k_initial.inverse();
  const gtsam::Matrix4& T_matrix = T_k_k_1_initial.matrix();
  Sophus::SE3d T21(Sophus::SO3d(T_matrix.topLeftCorner<3, 3>()), 
                   T_matrix.topRightCorner<3, 1>());
  
  // Run FineTracker estimation with exception handling
  try {
    fine_tracker_->estimate(frame_k_1, frame_k, T21);
  } catch (const std::runtime_error& e) {
    LOG_EVERY_N(WARNING, 50) << "FineTracker failed with error: " << e.what()
                             << ". Falling back to initial pose.";
    return false;
  } catch (const std::exception& e) {
    LOG_EVERY_N(WARNING, 50) << "FineTracker failed with exception: " << e.what()
                             << ". Falling back to initial pose.";
    return false;
  } catch (...) {
    LOG_EVERY_N(WARNING, 50)
        << "FineTracker failed with unknown exception. Falling back to initial pose.";
    return false;
  }
  
  // Check if tracking was successful
  if (!T21.matrix().allFinite()) {
    LOG_EVERY_N(WARNING, 50) << "FineTracker returned invalid pose matrix";
    return false;
  }
  
  // Convert Sophus::SE3d result back to gtsam::Pose3
  // FineTracker returns T_ref_cur (T_k_1_k) after inverting T_cur_ref internally
  // So T21 is already T_k_1_k, no need to invert again
  const Eigen::Matrix4d T_result = T21.matrix();
  T_k_1_k_refined = gtsam::Pose3(T_result);
  
  return true;
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
  cv::Mat tracking_image = tracker_->computeImageTracks(
      *frame_k_1, *frame_k, getFrontendParams().image_tracks_vis_params);

  const auto& camera_params = camera_->getParams();
  const auto& K = camera_params.getCameraMatrix();
  const auto& D = camera_params.getDistortionCoeffs();

  const gtsam::Pose3& X_k = frame_k->getPose();

  // poses are expected to be in the world frame
  gtsam::FastMap<ObjectId, gtsam::Pose3> poses_k_map =
      object_poses.collectByFrame(frame_k->getFrameId());
  std::vector<gtsam::Pose3> poses_k_vec;
  std::transform(poses_k_map.begin(), poses_k_map.end(),
                 std::back_inserter(poses_k_vec),
                 [&X_k](const std::pair<ObjectId, gtsam::Pose3>& pair) {
                   // put object pose into the camera frame so it can be
                   // projected into the image
                   return X_k.inverse() * pair.second;
                 });

  // TODO: bring back when visualisation is unified with incremental solver!!
  //  utils::drawObjectPoseAxes(tracking_image, K, D, poses_k_vec);
  return tracking_image;
}

// Edge-based local mapping functions (from localmapping.cc)

bool RGBDInstanceFrontendModule::shouldAddEdgeKeyFrame(
    const gtsam::Pose3& pose_curr, const gtsam::Pose3& pose_last) const {
  // From localmapping.cc shouldAddKeyFrame function
  gtsam::Pose3 trans = pose_last.inverse() * pose_curr;
  
  gtsam::Matrix3 R_bias = trans.rotation().matrix();
  Eigen::AngleAxisd rotation_vector(R_bias);
  double theta = rotation_vector.angle() * 180.0 / M_PI;
  double translation = trans.translation().norm();
  
  return (theta > kf_rot_thres_ || translation > kf_trans_thres_);
}

void RGBDInstanceFrontendModule::processingThreadFunction() {
  while (processing_running_) {
    KeyFramePtr kf;
    optimization_queue_.pop(kf);  // Blocking pop from queue
    
    if (!kf) {
      // nullptr means shutdown signal
      break;
    }
    
    // Process sliding window keyframe
    processSlidingWindowKeyFrame(kf);
  }
}

void RGBDInstanceFrontendModule::processSlidingWindowKeyFrame(KeyFramePtr kf) {
  if (!kf) {
    return;
  }
  
  const auto t0 = std::chrono::steady_clock::now();
  
  // Add keyframe to local map
  size_t kf_count = 0;
  {
    std::lock_guard<std::mutex> lock(local_map_mutex_);
    local_map_->addFrame2LocalMap(kf);
    kf_count = local_map_->mvKeyFrames.size();
  }
  
  const auto t1 = std::chrono::steady_clock::now();
  const auto ms = [](const auto& a, const auto& b) -> double {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  
  VLOG(1) << "EdgeKF processed from queue kf_id=" << kf->KF_ID
          << " add_to_map_ms=" << ms(t0, t1)
          << " local_map_kfs=" << kf_count;
  
  // Check if window is full and perform optimization
  if (kf_count >= window_size_) {
    // Perform optimization directly (already in processing thread)
    if (!optimization_in_progress_.load()) {
      optimization_in_progress_ = true;
      
      const auto t_opt_start = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::mutex> map_lock(local_map_mutex_);
        
        // From localmapping.cc lines 233-236
        local_map_->clusterFittingProjection();
        const auto t1 = std::chrono::steady_clock::now();
        edge_map::Optimizer::optimizeAllInvolvedKFs(local_map_);
        const auto t2 = std::chrono::steady_clock::now();
        
        VLOG(1) << "Edge sliding-window optimized (async)"
                << " cluster_fit_ms="
                << std::chrono::duration<double, std::milli>(t1 - t_opt_start).count()
                << " optimize_ms="
                << std::chrono::duration<double, std::milli>(t2 - t1).count()
                << " total_ms="
                << std::chrono::duration<double, std::milli>(t2 - t_opt_start).count();
        
        // Update sliding window after optimization
        updateEdgeSlidingWindow();
      }
      
      optimization_in_progress_ = false;
    }
  }
}

void RGBDInstanceFrontendModule::processEdgeKeyFrame(
    const Frame::Ptr& frame, const gtsam::Pose3& pose_curr) {
  const auto t0 = std::chrono::steady_clock::now();
  // Convert Frame to KeyFrame format (from localmapping.cc line 222)
  const auto t_kf0 = std::chrono::steady_clock::now();
  KeyFramePtr pKF = createKeyFrameFromFrame(frame, pose_curr);
  const auto t_kf1 = std::chrono::steady_clock::now();
  
  if (pKF) {
    // Add to local map (from localmapping.cc line 226)
    // Use mutex to protect local_map_ from concurrent access
    // Lock only for the minimal time needed
    const auto t_add0 = std::chrono::steady_clock::now();
    size_t kf_count = 0;
    {
      std::lock_guard<std::mutex> lock(local_map_mutex_);
      local_map_->addFrame2LocalMap(pKF);
      kf_count = local_map_->mvKeyFrames.size();
    }
    const auto t_add1 = std::chrono::steady_clock::now();
    
    const auto t1 = std::chrono::steady_clock::now();
    const auto ms =
        [](const auto& a, const auto& b) -> double {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    
    VLOG(1) << "EdgeKF frame=" << frame->getFrameId()
            << " create_kf_ms=" << ms(t_kf0, t_kf1)
            << " add_to_map_ms=" << ms(t_add0, t_add1)
            << " total_ms=" << ms(t0, t1)
            << " local_map_kfs=" << kf_count;
  }
}

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
    if (cam_params.hasDepthParams()) {
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
  VLOG(2) << "KeyFrame ctor frame=" << frame_id
          << " ms="
          << std::chrono::duration<double, std::milli>(t_kf_ctor1 - t_kf_ctor0).count();
  
  // Get fine sampled points (from localmapping.cc line 223)
  // TODO: Get sample_bias from params
  // Note: bias must be >= 1 (integer), using 2 as default (similar to FineTracker which uses 4)
  const auto t_sample0 = std::chrono::steady_clock::now();
  pKF->getFineSampledPoints(2);  // Default value, should come from params
  const auto t_sample1 = std::chrono::steady_clock::now();
  VLOG(2) << "Fine sample frame=" << frame_id
          << " ms="
          << std::chrono::duration<double, std::milli>(t_sample1 - t_sample0).count();
  
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
  // Save poses and timestamps of removed keyframes
  std::vector<Eigen::Matrix4d> removed_poses;
  std::vector<double> removed_stamps;
  
  for (int j = 0; j < window_step_; ++j) {
    double removed_stamp = local_map_->mvKeyFrames[j]->KF_stamp;
    Eigen::Matrix4d removed_pose = 
        local_map_->mvKeyFrames[j]->KF_pose_g.matrix();
    
    removed_poses.push_back(removed_pose);
    removed_stamps.push_back(removed_stamp);
  }
  
  // Keep overlapping keyframes
  std::vector<KeyFramePtr> newKFs(
      local_map_->mvKeyFrames.begin() + window_step_,
      local_map_->mvKeyFrames.begin() + window_size_);
  
  // Reset local map and add overlapping keyframes
  local_map_.reset(new edge_map::localMap());
  for (int j = 0; j < window_size_ - window_step_; ++j) {
    newKFs[j]->mmEdgeIndex2ElementEdgeID.clear();
    newKFs[j]->mmMapAssociations.clear();
    local_map_->addFrame2LocalMap(newKFs[j]);
  }
  
  const auto t1 = std::chrono::steady_clock::now();
  VLOG(1) << "Updated edge sliding window, kept "
          << (window_size_ - window_step_) << " keyframes"
          << " rebuild_ms="
          << std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace dyno
