/*
 * Edge Backend Module
 * Edge-based mapping backend module for DynoSAM
 */

#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/BackendOutputPacket.hpp"
#include "dynosam/backend/EdgeBackendDefinitions.hpp"
#include "dynosam/backend/VisionImuBackendModule.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"
#include "dynosam_common/Flags.hpp"
#include "dynosam_opt/Map.hpp"

// Forward declarations for edge_map components
namespace edge_map {
class localMap;
using localMapPtr = std::shared_ptr<localMap>;
class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;
}  // namespace edge_map

namespace dyno {

class EdgeBackendModule
    : public VisionImuBackendModule<EdgeBackendModuleTraits> {
 public:
  DYNO_POINTER_TYPEDEFS(EdgeBackendModule)

  using Base = VisionImuBackendModule<EdgeBackendModuleTraits>;
  using RGBDMap = Base::MapType;

  /**
   * @brief Constructor for EdgeBackendModule
   * @param backend_params Backend parameters
   * @param camera Camera parameters
   * @param display_queue Optional display queue for visualization
   */
  EdgeBackendModule(const BackendParams& backend_params, Camera::Ptr camera,
                    ImageDisplayQueue* display_queue = nullptr);

  ~EdgeBackendModule();

  using SpinReturn = Base::SpinReturn;

  /**
   * @brief Get active optimization graph and values
   * Required by BackendModuleType
   */
  std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> getActiveOptimisation()
      const override;

  /**
   * @brief Get accessor to optimized values
   * Required by BackendModule
   */
  Accessor::Ptr getAccessor() override;

  /**
   * @brief Get local map for visualization or external access
   */
  edge_map::localMapPtr getLocalMap() const { return pLocalMap_; }

 protected:
  /**
   * @brief Bootstrap spin - called for initial frames
   * Required by VisionImuBackendModule
   */
  SpinReturn boostrapSpinImpl(VisionImuPacket::ConstPtr input) override;

  /**
   * @brief Nominal spin - called for regular processing
   * Required by VisionImuBackendModule
   * This is where localmapping.cc의 main loop logic goes
   */
  SpinReturn nominalSpinImpl(VisionImuPacket::ConstPtr input) override;

 private:
  /**
   * @brief Initialize local map and edge selector
   */
  void initializeEdgeMapping();

  /**
   * @brief Check if current frame should be added as keyframe
   */
  bool shouldAddKeyFrame(const gtsam::Pose3& pose_curr,
                         const gtsam::Pose3& pose_last) const;

  /**
   * @brief Process keyframe - extract edges and add to local map
   */
  void processKeyFrame(VisionImuPacket::ConstPtr input,
                       const gtsam::Pose3& pose_curr);

  /**
   * @brief Optimize sliding window when window is full
   */
  void optimizeSlidingWindow();

  /**
   * @brief Update sliding window - remove old keyframes
   */
  void updateSlidingWindow();

  /**
   * @brief Construct output packet from current state
   */
  BackendOutputPacket::Ptr constructOutputPacket(FrameId frame_k,
                                                 Timestamp timestamp) const;

  /**
   * @brief Convert VisionImuPacket to KeyFrame format
   */
  edge_map::KeyFramePtr createKeyFrameFromPacket(
      VisionImuPacket::ConstPtr input, const gtsam::Pose3& pose_curr,
      FrameId frame_id);

  // Member variables
  Camera::Ptr camera_;
  edge_map::localMapPtr pLocalMap_;
  
  // Edge selector (from localmapping.cc)
  // Note: edgeSelector type needs to be defined in edge_map namespace
  std::unique_ptr<class edgeSelector> edge_selector_;
  
  // Sliding window parameters
  int window_size_;
  int window_step_;
  float kf_rot_thres_;
  float kf_trans_thres_;
  
  // State tracking
  gtsam::Pose3 pose_last_;
  bool is_initialized_;
  
  // Output tracking (for trajectory saving)
  std::vector<Eigen::Matrix4d> list_kf_poses_;
  std::vector<double> list_kf_stamps_;
  
  // Camera parameters (from localmapping.cc)
  float fx_, fy_, cx_, cy_;
  float depth_scale_;
};

}  // namespace dyno
