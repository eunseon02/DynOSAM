#pragma once

#include "dynosam/visualizer/VisualizerPipelines.hpp"
#include <pangolin/pangolin.h>


#include <fstream>
#include <mutex>
#include <string>
#include <vector>
// #include <pangolin/pangolin.h>
#include <thread>
#include <atomic>

namespace dyno {

/**
 * @brief BackendModuleDisplay that logs trajectory to file in TUM format
 * and displays it in real-time using Pangolin 3D viewer (Viewer style)
 */
class TrajectoryLoggerDisplay : public BackendModuleDisplay {
 public:
  DYNO_POINTER_TYPEDEFS(TrajectoryLoggerDisplay)

  TrajectoryLoggerDisplay(const std::string& output_file, bool show_window = true);
  ~TrajectoryLoggerDisplay() override;

  void spin(const BackendOutputPacket::ConstPtr& output) override;

  // Get current trajectory (thread-safe)
  std::vector<std::pair<double, gtsam::Pose3>> getTrajectory() const;

 private:
  void runPangolinViewer();  // Pangolin viewer thread function
  void drawTrajectory3D();   // Draw trajectory in 3D
  
  std::string output_file_;
  std::ofstream trajectory_file_;
  mutable std::mutex trajectory_mutex_;
  std::vector<std::pair<double, gtsam::Pose3>> trajectory_;
  
  // Pangolin visualization
  bool show_window_;
  std::atomic<bool> viewer_running_;
  std::thread viewer_thread_;
  
  // Viewer parameters (similar to Viewer.hpp)
  float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
  bool mbFollowCamera;
  bool mbShowTrajectory;
  bool mbShowKeyFrames;
  bool mbShowGraph;
};

}  // namespace dyno
