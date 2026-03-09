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

#pragma once

#include <fstream>
#include <unordered_set>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <tbb/concurrent_queue.h>

#include <gtsam/navigation/NavState.h>

#include "dynosam/frontend/FrontendModule.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_cv/Camera.hpp"


#include "dynosam/frontend/vision/FineTracker.hpp"
#include "dynosam/frontend/vision/DirectTracker.hpp"

// Edge-based local mapping includes
#include "dynosam_common/EdgeSelector.hpp"
#include "dynosam/backend/edge_map/KeyFrame.hpp"
#include "dynosam/backend/edge_map/localMap.hpp"
#include "dynosam/backend/edge_map/Optimizer.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"
#include "dynosam/backend/edge_map/Map.hpp"
#include "dynosam/frontend/vision/ObjectMatcher.hpp"

#include <deque>

namespace dyno {

class RGBDInstanceFrontendModule : public FrontendModule {
 public:
  RGBDInstanceFrontendModule(const DynoParams& params, Camera::Ptr camera,
                             ImageDisplayQueue* display_queue);
  ~RGBDInstanceFrontendModule();

  using SpinReturn = FrontendModule::SpinReturn;

 private:
  Camera::Ptr camera_;
  EgoMotionSolver motion_solver_;
  // TODO: shared pointer for now during debig phase!
  ObjectMotionSolver::Ptr object_motion_solver_;
  FeatureTracker::UniquePtr tracker_;
  fine::FineTracker::UniquePtr fine_tracker_;
  direct::DirectTracker::UniquePtr direct_tracker_;
  ObjectMatcher::UniquePtr object_matcher_;
  RGBDFrontendLogger::UniquePtr logger_;

 private:
  ImageValidationResult validateImageContainer(
      const ImageContainer::Ptr& image_container) const override;
  SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
  SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;

  /**
   * @brief Solves PnP between frame_k-1 and frame_k using the tracked
   * correspondances to estimate the frame of the current camera
   *
   * the pose of the Frame::Ptr (frame_k) is updated, and the features marked as
   * outliers by PnP are set as outliers.
   *
   * Depending on FrontendParams::use_ego_motion_pnp, a differnet solver will be
   * used to estimate the pose
   *
   * @param frame_k
   * @param frame_k_1
   * @return true
   * @return false
   */
  bool solveCameraMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,
                         std::optional<gtsam::Rot3> R_curr_ref = {});

  bool DirectTrack(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,
                   const gtsam::Pose3& T_k_1_k_initial, gtsam::Pose3& T_k_1_k_refined);
  
  bool FineTrack(Frame::Ptr frame,
                 const gtsam::Pose3& T_k_1_k_initial, gtsam::Pose3& T_k_1_k_refined);

  bool checkPoseJump(Sophus::SE3d pose);
  
  void fillOutputPacketWithTracks(VisionImuPacket::Ptr vision_imu_packet,
                                  const Frame& frame,
                                  const gtsam::Pose3& T_k_1_k,
                                  const ObjectMotionMap& object_motions,
                                  const ObjectPoseMap& object_poses) const;

  void sendToFrontendLogger(const Frame::Ptr& frame,
                            const VisionImuPacket::Ptr& vision_imu_packet);

  cv::Mat createTrackingImage(const Frame::Ptr& frame_k,
                              const Frame::Ptr& frame_k_1,
                              const ObjectPoseMap& object_poses) const;

  void ObjectCulling(const KeyFramePtr& pKF);

  // used when we want to seralize the output to json via the
  // FLAGS_save_frontend_json flag
  //   std::map<FrameId, RGBDInstanceOutputPacket::Ptr> output_packet_record_;

  //! Imu frontend - mantains pre-integration from last kf to current k
  ImuFrontend imu_frontend_;
  //! Nav state at k
  gtsam::NavState nav_state_curr_;
  // this is always udpated with the best X_k pose but the velocity may be wrong
  // if no IMU...
  //! Nav state at k-1
  gtsam::NavState nav_state_prev_;

  //! Nav state of the last key-frame
  gtsam::NavState nav_state_last_kf_;

  //! Tracks when the nav state was updated using IMU, else VO
  //! in the front-end, when updating with VO, the velocity will (currently) be
  //! wrong!!
  FrameId last_imu_k_{0};

  //! The relative camera pose (T_k_1_k) from the previous frame
  //! this is used as a constant velocity model when VO tracking fails and the
  //! IMU is not available!
  gtsam::Pose3 vo_velocity_;

  // DEBUGGING FOR KF
  mutable gtsam::Pose3 T_lkf_;

  //! Last keyframe
  Frame::Ptr frame_lkf_;

  // Edge-based local mapping components (from localmapping.cc)
  dyno::localMapPtr local_map_;
  std::unique_ptr<edgeSelector> edge_selector_;
  gtsam::Pose3 pose_last_edge_kf_;  // Last edge keyframe pose
  bool is_edge_initialized_{false};
  
  // Cached last keyframe (updated when keyframe is added, avoids frequent lock)
  mutable std::mutex last_kf_mutex_;
  Frame::Ptr last_keyframe_{nullptr};  // Last keyframe Frame (from when keyframe was created)
  
  // Sliding window parameters
  int window_size_{10};
  int window_step_{4};
  float kf_rot_thres_{5.0f};      // degrees
  float kf_trans_thres_{0.1f};    // meters
  double kMinConfidenceScore_{0.2};  // Minimum confidence score threshold for object detection
  
  // Threading for async keyframe processing
  mutable std::mutex local_map_mutex_;  // Protects local_map_ during optimization (mutable for const getLocalMap)
  tbb::concurrent_bounded_queue<KeyFramePtr> optimization_queue_;
  std::thread processing_thread_;
  std::atomic<bool> processing_running_{false};
  std::atomic<bool> optimization_in_progress_{false};  // Track if optimization is currently running

  // Optional: log edge keyframe poses across the *entire* run (not just sliding window).
  // If enabled, we append keyframes as they leave the sliding window and dump remaining on shutdown.
  std::string edge_kf_trajectory_file_;
  std::ofstream edge_kf_trajectory_stream_;
  std::unordered_set<int> edge_kf_logged_ids_;

  // Latest visualization data (always available, updated every frame)
  // Protected by viz_mutex_ for thread-safe read/write
  mutable std::mutex viz_mutex_;
  EdgeVisualizationDataPtr latest_visualization_data_;

  // Cached visualization state (heavy data) protected by local_map_mutex_
  std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> cluster_clouds_cache_;
  std::shared_ptr<const std::vector<cv::Vec3b>> cluster_colors_cache_;
  std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> local_map_clouds_cache_;
  std::shared_ptr<const std::vector<cv::Vec3b>> local_map_colors_cache_;
  std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> environment_cloud_cache_;
  std::deque<std::vector<cv::Point3d>> environment_frames_;  // <=150 frames

  // Global object map (ellipsoid/object-level SLAM map)
  std::shared_ptr<dyno::EdgeMap> map_;


  bool is_data_valid_{false};
  

  
  // Processing thread function
  void processingThreadFunction();
  
  // Edge-based keyframe management
  bool shouldAddEdgeKeyFrame(const gtsam::Pose3& pose_curr, 
                             const gtsam::Pose3& pose_last) const;
  // void processEdgeKeyFrame(const Frame::Ptr& frame, const gtsam::Pose3& pose_curr);
  KeyFramePtr createKeyFrameFromFrame(const Frame::Ptr& frame, 
                                                 const gtsam::Pose3& pose_curr);
  void optimizeEdgeSlidingWindow();
  void updateEdgeSlidingWindow();
  
  // Sliding window keyframe processing and optimization trigger
  // Called by processing thread after popping from queue
  void processSlidingWindowKeyFrame(KeyFramePtr kf);
  
 public:
  void ObjectsInitialization(const Frame::Ptr& frame, dyno::KeyFrame* kf);
  // Getter for local map (for visualization)
  // Thread-safe: returns a copy of the pointer (shared_ptr is thread-safe for reading)
  dyno::localMapPtr getLocalMap() const {
    std::lock_guard<std::mutex> lock(local_map_mutex_);
    return local_map_;
  }
  
  // Getter for current frame (for visualization)
  // Returns the most recently tracked frame
  Frame::Ptr getCurrentFrame() const {
    if (tracker_) {
      return tracker_->getCurrentFrame();
    }
    return nullptr;
  }
  
  // Get latest visualization data (thread-safe, always returns current snapshot)
  // Returns nullptr if no data is available yet
  EdgeVisualizationDataPtr getLatestVisualizationData() const {
    std::lock_guard<std::mutex> lock(viz_mutex_);
    return latest_visualization_data_;
  }
  
  // Getter for global object map (for visualization)
  // Thread-safe: returns a copy of the pointer (shared_ptr is thread-safe for reading)
  std::shared_ptr<dyno::EdgeMap> getMap() const {
    return map_;
  }

};

}  // namespace dyno
