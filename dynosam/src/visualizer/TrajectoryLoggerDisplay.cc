#include "dynosam/visualizer/TrajectoryLoggerDisplay.hpp"
#include <iomanip>
#include <algorithm>

namespace dyno {

TrajectoryLoggerDisplay::TrajectoryLoggerDisplay(const std::string& output_file, bool show_window)
    : output_file_(output_file), show_window_(show_window), scale_(50.0), center_initialized_(false) {
  trajectory_file_.open(output_file_);
  if (!trajectory_file_.is_open()) {
    LOG(WARNING) << "Failed to open trajectory output file: " << output_file_;
  } else {
    LOG(INFO) << "Trajectory logging to: " << output_file_;
  }
  
  if (show_window_) {
    trajectory_image_ = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
    LOG(INFO) << "Trajectory visualization window opened";
  }
}

TrajectoryLoggerDisplay::~TrajectoryLoggerDisplay() {
  if (trajectory_file_.is_open()) {
    trajectory_file_.close();
    LOG(INFO) << "Trajectory saved to: " << output_file_ << " (total poses: " << trajectory_.size() << ")";
  }
  
  if (show_window_) {
    // Show final trajectory for a moment before closing
    if (!trajectory_image_.empty()) {
      cv::imshow("Trajectory", trajectory_image_);
      cv::waitKey(1000);
    }
    cv::destroyWindow("Trajectory");
  }
}

cv::Point2i TrajectoryLoggerDisplay::projectTo2D(const gtsam::Point3& pt_3d) const {
  // Project 3D point to 2D image coordinates
  // Use XZ plane (top-down view) - typical for SLAM visualization
  double x = pt_3d.x() - trajectory_center_.x();
  double z = pt_3d.z() - trajectory_center_.z();
  
  int img_x = static_cast<int>(x * scale_) + WINDOW_WIDTH / 2;
  int img_y = static_cast<int>(z * scale_) + WINDOW_HEIGHT / 2;
  
  return cv::Point2i(img_x, img_y);
}

void TrajectoryLoggerDisplay::updateVisualization() {
  if (!show_window_) {
    return;
  }
  
  if (trajectory_.empty()) {
    // Show empty window with info
    trajectory_image_ = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    cv::putText(trajectory_image_, "Waiting for trajectory...", cv::Point(10, WINDOW_HEIGHT / 2), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::imshow("Trajectory", trajectory_image_);
    cv::waitKey(1);
    return;
  }
  
  // Clear image
  trajectory_image_ = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
  
  // Initialize center with first pose
  if (!center_initialized_ && !trajectory_.empty()) {
    trajectory_center_ = trajectory_[0].second.translation();
    center_initialized_ = true;
  }
  
  // Calculate bounding box of trajectory to auto-scale
  if (!trajectory_.empty()) {
    double min_x = trajectory_[0].second.translation().x();
    double max_x = min_x;
    double min_z = trajectory_[0].second.translation().z();
    double max_z = min_z;
    
    for (const auto& [ts, pose] : trajectory_) {
      const auto& t = pose.translation();
      min_x = std::min(min_x, t.x());
      max_x = std::max(max_x, t.x());
      min_z = std::min(min_z, t.z());
      max_z = std::max(max_z, t.z());
    }
    
    // Update center to middle of trajectory
    trajectory_center_ = gtsam::Point3((min_x + max_x) / 2.0, 
                                        trajectory_.back().second.translation().y(),
                                        (min_z + max_z) / 2.0);
    
    // Auto-scale to fit window
    double range_x = max_x - min_x;
    double range_z = max_z - min_z;
    double max_range = std::max(range_x, range_z);
    if (max_range > 0.1) {  // Only scale if there's meaningful movement
      // Scale to fit 80% of window
      double scale_factor = (WINDOW_WIDTH * 0.8) / max_range;
      // Use dynamic scale but cap it
      scale_ = std::min(scale_factor, 200.0);
    } else {
      scale_ = 50.0;  // Default scale
    }
  }
  
  // Draw trajectory
  if (trajectory_.size() > 1) {
    for (size_t i = 1; i < trajectory_.size(); ++i) {
      cv::Point2i pt1 = projectTo2D(trajectory_[i-1].second.translation());
      cv::Point2i pt2 = projectTo2D(trajectory_[i].second.translation());
      
      // Draw line even if slightly out of bounds (clamp to bounds)
      pt1.x = std::max(0, std::min(WINDOW_WIDTH - 1, pt1.x));
      pt1.y = std::max(0, std::min(WINDOW_HEIGHT - 1, pt1.y));
      pt2.x = std::max(0, std::min(WINDOW_WIDTH - 1, pt2.x));
      pt2.y = std::max(0, std::min(WINDOW_HEIGHT - 1, pt2.y));
      
      // Draw line (green for trajectory)
      cv::line(trajectory_image_, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }
  }
  
  // Draw current position (red circle)
  if (!trajectory_.empty()) {
    cv::Point2i current_pt = projectTo2D(trajectory_.back().second.translation());
    current_pt.x = std::max(0, std::min(WINDOW_WIDTH - 1, current_pt.x));
    current_pt.y = std::max(0, std::min(WINDOW_HEIGHT - 1, current_pt.y));
    cv::circle(trajectory_image_, current_pt, 8, cv::Scalar(0, 0, 255), -1);
    
    // Draw start position (blue circle)
    if (trajectory_.size() > 1) {
      cv::Point2i start_pt = projectTo2D(trajectory_[0].second.translation());
      start_pt.x = std::max(0, std::min(WINDOW_WIDTH - 1, start_pt.x));
      start_pt.y = std::max(0, std::min(WINDOW_HEIGHT - 1, start_pt.y));
      cv::circle(trajectory_image_, start_pt, 6, cv::Scalar(255, 0, 0), -1);
    }
  }
  
  // Draw coordinate axes at center
  cv::Point2i origin = cv::Point2i(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
  cv::line(trajectory_image_, origin, origin + cv::Point2i(30, 0), cv::Scalar(255, 0, 0), 2);  // X axis (red)
  cv::line(trajectory_image_, origin, origin + cv::Point2i(0, 30), cv::Scalar(0, 0, 255), 2);  // Z axis (blue)
  
  // Add text info
  std::string info = "Frames: " + std::to_string(trajectory_.size());
  cv::putText(trajectory_image_, info, cv::Point(10, 30), 
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
  
  if (!trajectory_.empty()) {
    const auto& last_pos = trajectory_.back().second.translation();
    std::string pos_info = "Pos: (" + std::to_string(last_pos.x()).substr(0, 5) + ", " +
                           std::to_string(last_pos.z()).substr(0, 5) + ")";
    cv::putText(trajectory_image_, pos_info, cv::Point(10, 60), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
  }
  
  // Show image
  cv::imshow("Trajectory", trajectory_image_);
  cv::waitKey(1);
}

void TrajectoryLoggerDisplay::spin(const BackendOutputPacket::ConstPtr& output) {
  if (!output) {
    LOG(WARNING) << "TrajectoryLoggerDisplay: received null output";
    return;
  }
  
  if (!trajectory_file_.is_open()) {
    LOG(WARNING) << "TrajectoryLoggerDisplay: trajectory file not open";
    return;
  }

  // Get current optimized pose
  const gtsam::Pose3& pose = output->T_world_camera;
  const double timestamp = output->timestamp;
  
  // Check if pose is valid (not identity/zero)
  const gtsam::Point3& t = pose.translation();
  const double translation_norm = t.norm();
  
  if (translation_norm < 1e-6) {
    LOG(WARNING) << "TrajectoryLoggerDisplay: received zero/identity pose at frame=" 
                 << output->frame_id << ", timestamp=" << timestamp;
  }
  
  LOG(INFO) << "TrajectoryLoggerDisplay: frame=" << output->frame_id 
            << ", timestamp=" << timestamp
            << ", pose=(" << t.x() << ", " << t.y() << ", " << t.z() << ")";

  // Save to file immediately (TUM format: timestamp tx ty tz qx qy qz qw)
  const gtsam::Rot3& R = pose.rotation();
  const gtsam::Quaternion q = R.toQuaternion();

  {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    trajectory_file_ << std::fixed << std::setprecision(6) 
                     << timestamp << " "
                     << t.x() << " " << t.y() << " " << t.z() << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    trajectory_file_.flush();  // Flush immediately for real-time monitoring
    
    // Also store in memory
    trajectory_.emplace_back(timestamp, pose);
    
    // Update visualization
    updateVisualization();
    
    // Log every 10 frames for monitoring
    if (trajectory_.size() % 10 == 0) {
      LOG(INFO) << "Trajectory logged: frame=" << output->frame_id 
                << ", timestamp=" << timestamp 
                << ", total_poses=" << trajectory_.size();
    }
  }
}

std::vector<std::pair<double, gtsam::Pose3>> TrajectoryLoggerDisplay::getTrajectory() const {
  std::lock_guard<std::mutex> lock(trajectory_mutex_);
  return trajectory_;
}

}  // namespace dyno
