#pragma once

#include "dynosam/visualizer/VisualizerPipelines.hpp"
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace dyno {

/**
 * @brief BackendModuleDisplay that logs trajectory to file in TUM format
 * and displays it in real-time in an OpenCV window
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
  void updateVisualization();
  cv::Point2i projectTo2D(const gtsam::Point3& pt_3d) const;
  
  std::string output_file_;
  std::ofstream trajectory_file_;
  mutable std::mutex trajectory_mutex_;
  std::vector<std::pair<double, gtsam::Pose3>> trajectory_;
  
  // Visualization
  bool show_window_;
  cv::Mat trajectory_image_;
  static constexpr int WINDOW_WIDTH = 800;
  static constexpr int WINDOW_HEIGHT = 800;
  mutable double scale_;  // pixels per meter (mutable for dynamic scaling)
  gtsam::Point3 trajectory_center_;
  bool center_initialized_;
};

}  // namespace dyno
