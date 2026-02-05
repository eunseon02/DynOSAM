/*
 * Edge Backend Module Implementation
 * Converts localmapping.cc logic into BackendModule framework
 */

#include "dynosam/backend/EdgeBackendModule.hpp"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

// Edge mapping includes (assumed to be in edge_map namespace)
#include "edge_map/edgeSelector.h"
#include "edge_map/KeyFrame.h"
#include "edge_map/localMap.h"
#include "edge_map/Optimizer.h"

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendOutputPacket.hpp"
#include "dynosam/backend/Accessor.hpp"
#include "dynosam_opt/Map.hpp"

namespace dyno {

EdgeBackendModule::EdgeBackendModule(const BackendParams& backend_params,
                                     Camera::Ptr camera,
                                     ImageDisplayQueue* display_queue)
    : Base(backend_params, display_queue),
      camera_(CHECK_NOTNULL(camera)),
      is_initialized_(false),
      window_size_(10),  // TODO: load from params
      window_step_(4),   // TODO: load from params
      kf_rot_thres_(5.0f),    // TODO: load from params
      kf_trans_thres_(0.1f) { // TODO: load from params

  CHECK_NOTNULL(map_);
  
  // Get camera parameters
  const CameraParams& camera_params = camera_->getParams();
  fx_ = camera_params.fx();
  fy_ = camera_params.fy();
  cx_ = camera_params.cu();
  cy_ = camera_params.cv();
  depth_scale_ = camera_params.hasDepthParams() 
                     ? static_cast<float>(camera_params.depthParams().depth_to_meters)
                     : 1.0f;
  
  // Initialize edge mapping components
  initializeEdgeMapping();
  
  LOG(INFO) << "EdgeBackendModule initialized";
}

EdgeBackendModule::~EdgeBackendModule() {
  LOG(INFO) << "Destructing EdgeBackendModule";
  // Save trajectory if needed
  // tum_file::saveEvaluationFiles(...);
}

void EdgeBackendModule::initializeEdgeMapping() {
  // Initialize local map
  pLocalMap_.reset(new edge_map::localMap());
  
  // Initialize edge selector (parameters from localmapping.cc)
  // edgeSelector selector(20.0, canny_low, canny_high);
  // TODO: Load canny parameters from config
  edge_selector_ = std::make_unique<edgeSelector>(20.0, 50, 150);
  
  LOG(INFO) << "Edge mapping components initialized";
}

EdgeBackendModule::SpinReturn EdgeBackendModule::boostrapSpinImpl(
    VisionImuPacket::ConstPtr input) {
  const FrameId frame_k = input->frameId();
  const Timestamp timestamp = input->timestamp();
  
  LOG(INFO) << "EdgeBackend bootstrap spin for frame " << frame_k;
  
  // Get initial pose from frontend
  const gtsam::Pose3& X_k_w = input->cameraPose();
  
  // Initialize pose tracking
  pose_last_ = X_k_w;
  is_initialized_ = true;
  
  // Process first frame as keyframe
  processKeyFrame(input, X_k_w);
  
  // Construct output packet
  BackendOutputPacket::Ptr backend_output =
      constructOutputPacket(frame_k, timestamp);
  
  return {State::Nominal, backend_output};
}

EdgeBackendModule::SpinReturn EdgeBackendModule::nominalSpinImpl(
    VisionImuPacket::ConstPtr input) {
  const FrameId frame_k = input->frameId();
  const Timestamp timestamp = input->timestamp();
  
  LOG(INFO) << "EdgeBackend nominal spin for frame " << frame_k;
  CHECK_EQ(spin_state_.frame_id, frame_k);
  
  // Get current pose from frontend
  const gtsam::Pose3& pose_curr_coarse = input->cameraPose();
  
  // Check if should add keyframe (from localmapping.cc logic)
  if (!is_initialized_) {
    pose_last_ = pose_curr_coarse;
    is_initialized_ = true;
  }
  
  // Calculate relative transformation
  gtsam::Pose3 trans = pose_last_.inverse() * pose_curr_coarse;
  bool select = shouldAddKeyFrame(pose_curr_coarse, pose_last_);
  
  if (select) {
    pose_last_ = pose_curr_coarse;
    
    // Get prior pose (similar to localmapping.cc lines 203-212)
    gtsam::Pose3 pose_curr;
    if (pLocalMap_->mvKeyFrames.empty()) {
      pose_curr = pose_curr_coarse;
    } else {
      // Use last optimized pose and relative motion
      gtsam::Pose3 pose_last_prior = pose_curr_coarse; // Simplified
      gtsam::Pose3 pose_bias = pose_last_prior.inverse() * pose_curr_coarse;
      gtsam::Pose3 pose_last_adjust = 
          pLocalMap_->mvKeyFrames.back()->KF_pose_g;
      pose_curr = pose_last_adjust * pose_bias;
    }
    
    // Process keyframe (from localmapping.cc lines 216-226)
    processKeyFrame(input, pose_curr);
    
    // Check if window is full and optimize (from localmapping.cc lines 230-270)
    if (pLocalMap_->mvKeyFrames.size() == window_size_) {
      optimizeSlidingWindow();
      updateSlidingWindow();
    }
  }
  
  // Construct output packet
  BackendOutputPacket::Ptr backend_output =
      constructOutputPacket(frame_k, timestamp);
  
  return {State::Nominal, backend_output};
}

bool EdgeBackendModule::shouldAddKeyFrame(const gtsam::Pose3& pose_curr,
                                          const gtsam::Pose3& pose_last) const {
  // From localmapping.cc shouldAddKeyFrame function
  gtsam::Pose3 trans = pose_last.inverse() * pose_curr;
  
  gtsam::Matrix3 R_bias = trans.rotation().matrix();
  Eigen::AngleAxisd rotation_vector(R_bias);
  double theta = rotation_vector.angle() * 180.0 / M_PI;
  double translation = trans.translation().norm();
  
  return (theta > kf_rot_thres_ || translation > kf_trans_thres_);
}

void EdgeBackendModule::processKeyFrame(VisionImuPacket::ConstPtr input,
                                        const gtsam::Pose3& pose_curr) {
  // Convert VisionImuPacket to KeyFrame format
  // This is the key conversion from localmapping.cc
  
  // Get image from VisionImuPacket
  // Note: VisionImuPacket might not have raw images directly
  // You may need to access them differently or store them separately
  
  // For now, assuming we can get RGB and depth images
  // TODO: Check how to get images from VisionImuPacket or store them separately
  
  // Create KeyFrame (similar to localmapping.cc line 222)
  edge_map::KeyFramePtr pKF = createKeyFrameFromPacket(
      input, pose_curr, input->frameId());
  
  // Add to local map (from localmapping.cc line 226)
  pLocalMap_->addFrame2LocalMap(pKF);
  
  LOG(INFO) << "Added keyframe " << input->frameId() 
            << " to local map (total: " << pLocalMap_->mvKeyFrames.size() << ")";
}

edge_map::KeyFramePtr EdgeBackendModule::createKeyFrameFromPacket(
    VisionImuPacket::ConstPtr input, const gtsam::Pose3& pose_curr,
    FrameId frame_id) {
  // Convert VisionImuPacket to KeyFrame
  // This requires accessing RGB and depth images from the packet
  
  // TODO: Get RGB and depth images from VisionImuPacket
  // For now, this is a placeholder - you'll need to adapt based on
  // how images are stored in VisionImuPacket
  
  // From localmapping.cc:
  // KeyFramePtr pKF(new KeyFrame(i, pose_curr, rgb_stamp_seq[i], 
  //                              selector.mvEdges, imgRGB, imgDepth, 
  //                              ph.fx, ph.fy, ph.cx, ph.cy));
  
  // Process image with edge selector
  // cv::Mat imgRGB = ...; // Get from input
  // cv::Mat imgDepth = ...; // Get from input
  // edge_selector_->processImage(imgRGB);
  
  // Create KeyFrame
  // Note: You'll need to adapt the KeyFrame constructor call
  // based on the actual KeyFrame class definition
  
  // Placeholder - needs actual implementation
  return nullptr;
}

void EdgeBackendModule::optimizeSlidingWindow() {
  // From localmapping.cc lines 233-236
  pLocalMap_->clusterFittingProjection();
  edge_map::Optimizer::optimizeAllInvolvedKFs(pLocalMap_);
  
  LOG(INFO) << "Optimized sliding window";
}

void EdgeBackendModule::updateSlidingWindow() {
  // From localmapping.cc lines 251-269
  // Save poses and timestamps of removed keyframes
  for (int j = 0; j < window_step_; ++j) {
    double removed_stamp = pLocalMap_->mvKeyFrames[j]->KF_stamp;
    Eigen::Matrix4d removed_pose = 
        pLocalMap_->mvKeyFrames[j]->KF_pose_g.matrix();
    
    list_kf_poses_.push_back(removed_pose);
    list_kf_stamps_.push_back(removed_stamp);
  }
  
  // Keep overlapping keyframes
  std::vector<edge_map::KeyFramePtr> newKFs(
      pLocalMap_->mvKeyFrames.begin() + window_step_,
      pLocalMap_->mvKeyFrames.begin() + window_size_);
  
  // Reset local map and add overlapping keyframes
  pLocalMap_.reset(new edge_map::localMap());
  for (int j = 0; j < window_size_ - window_step_; ++j) {
    newKFs[j]->mmEdgeIndex2ElementEdgeID.clear();
    newKFs[j]->mmMapAssociations.clear();
    pLocalMap_->addFrame2LocalMap(newKFs[j]);
  }
  
  LOG(INFO) << "Updated sliding window, kept " 
            << (window_size_ - window_step_) << " keyframes";
}

std::pair<gtsam::Values, gtsam::NonlinearFactorGraph>
EdgeBackendModule::getActiveOptimisation() const {
  // Return current optimization state
  // For edge mapping, this might be different from GTSAM formulation
  // You may need to convert edge_map optimization to GTSAM format
  // or return empty if not using GTSAM
  
  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;
  
  // TODO: Convert edge_map optimization state to GTSAM format
  // For now, return empty
  return {values, graph};
}

Accessor::Ptr EdgeBackendModule::getAccessor() {
  // Create accessor to get optimized poses
  // This allows frontend to access backend-optimized poses
  
  // TODO: Implement EdgeAccessor that wraps edge_map poses
  // For now, return a simple accessor or nullptr
  // You might need to create a custom Accessor implementation
  
  return nullptr; // Placeholder
}

BackendOutputPacket::Ptr EdgeBackendModule::constructOutputPacket(
    FrameId frame_k, Timestamp timestamp) const {
  // Construct BackendOutputPacket from current state
  // Similar to RegularBackendModule::constructOutputPacket
  
  BackendOutputPacket::Ptr backend_output =
      std::make_shared<BackendOutputPacket>();
  
  backend_output->frame_id = frame_k;
  backend_output->timestamp = timestamp;
  
  // Get optimized pose from local map
  if (!pLocalMap_->mvKeyFrames.empty()) {
    const auto& last_kf = pLocalMap_->mvKeyFrames.back();
    backend_output->T_world_camera = last_kf->KF_pose_g;
  } else {
    // Fallback to identity if no keyframes
    backend_output->T_world_camera = gtsam::Pose3();
  }
  
  return backend_output;
}

}  // namespace dyno
