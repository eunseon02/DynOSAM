/*
 * VoViewer Test Code for TUM RGBD Dataset
 * 
 * Compilation:
 *   cd /home/user/dev_ws
 *   source install/setup.bash
 *   g++ -std=c++17 test_voviewer.cc \
 *       -I/home/user/dev_ws/install/dynosam/include \
 *       -I/usr/include/eigen3 \
 *       -L/home/user/dev_ws/install/dynosam/lib \
 *       -ldynosam -lpangolin \
 *       -o test_voviewer
 * 
 * Or add to CMakeLists.txt:
 *   add_executable(test_voviewer test_voviewer.cc)
 *   target_link_libraries(test_voviewer ${PROJECT_NAME} pango_core pango_opengl pango_display pango_windowing pango_vars)
 */

#include "dynosam/visualizer/VoViewer.hpp"
#include "dynosam/visualizer/TumFile.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <Eigen/Dense>
#include <filesystem>

// Function to read TUM RGBD associated file
void LoadTUMAssociationFile(const std::string& association_file,
                            std::vector<std::string>& rgb_files,
                            std::vector<std::string>& depth_files,
                            std::vector<double>& timestamps) {
  std::ifstream fAssociation(association_file);
  if (!fAssociation.is_open()) {
    std::cerr << "Cannot open association file: " << association_file << std::endl;
    return;
  }

  rgb_files.clear();
  depth_files.clear();
  timestamps.clear();

  std::string line;
  while (std::getline(fAssociation, line)) {
    if (line.empty() || line[0] == '#') {
      continue;  // Skip empty lines and comments
    }

    std::stringstream ss(line);
    double t_rgb, t_depth;
    std::string rgb_file, depth_file;

    ss >> t_rgb >> rgb_file >> t_depth >> depth_file;

    timestamps.push_back(t_rgb);
    rgb_files.push_back(rgb_file);
    depth_files.push_back(depth_file);
  }

  fAssociation.close();
  std::cout << "Loaded " << timestamps.size() << " frames from association file" << std::endl;
}

// Function to generate dummy trajectory (simple forward motion)
std::vector<Eigen::Matrix4d> GenerateDummyTrajectory(size_t num_frames) {
  std::vector<Eigen::Matrix4d> trajectory;
  trajectory.reserve(num_frames);

  for (size_t i = 0; i < num_frames; ++i) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    
    // Simple forward motion along x-axis
    double x = i * 0.01;  // 1cm per frame
    double y = 0.0;
    double z = 0.0;
    
    // Small rotation around z-axis
    double angle = i * 0.001;  // Small rotation per frame
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ());
    
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
    
    trajectory.push_back(pose);
  }

  return trajectory;
}

int main(int argc, char** argv) {
  std::string association_file = "/root/data/tum-rgbd/fr2_desk/rgbd_dataset_freiburg2_large_with_loop_associated.txt";
  
  // Read association file
  std::vector<std::string> rgb_files;
  std::vector<std::string> depth_files;
  std::vector<double> timestamps;
  
  LoadTUMAssociationFile(association_file, rgb_files, depth_files, timestamps);
  
  if (rgb_files.empty()) {
    std::cerr << "No data loaded from association file!" << std::endl;
    return -1;
  }

  std::cout << "Creating VoViewer..." << std::endl;
  
  // Create VoViewer
  voViewer viewer("TUM RGBD Dataset Test Viewer");
  
  // Try to load ground truth if available
  std::filesystem::path assoc_path(association_file);
  std::filesystem::path dataset_dir = assoc_path.parent_path();
  std::string gt_file = (dataset_dir / "groundtruth.txt").string();
  
  std::vector<Eigen::Matrix4d> trajectory;
  std::vector<Eigen::Matrix4d> trajectory_GT;
  std::vector<double> gt_timestamps;
  
  bool has_groundtruth = false;
  if (std::filesystem::exists(gt_file)) {
    std::cout << "Found ground truth file: " << gt_file << std::endl;
    tum_file::getGTsequence(gt_file, trajectory_GT, gt_timestamps);
    if (!trajectory_GT.empty()) {
      has_groundtruth = true;
      std::cout << "Loaded " << trajectory_GT.size() << " ground truth poses" << std::endl;
    }
  } else {
    std::cout << "Ground truth file not found: " << gt_file << std::endl;
    std::cout << "Using dummy trajectory instead" << std::endl;
  }
  
  // Generate or use ground truth trajectory
  if (has_groundtruth) {
    // Use ground truth trajectory
    trajectory = trajectory_GT;
    
    // Update viewer with ground truth trajectory
    for (size_t i = 0; i < trajectory_GT.size(); ++i) {
      viewer.update_TrajectoryGT(trajectory_GT[i]);
    }
  } else {
    // Generate dummy trajectory for visualization
    trajectory = GenerateDummyTrajectory(timestamps.size());
  }
  
  std::cout << "Updating viewer with trajectory data..." << std::endl;
  
  // Update viewer with trajectory
  for (size_t i = 0; i < trajectory.size() && i < timestamps.size(); ++i) {
    viewer.update_Trajectory(trajectory[i]);
    
    // Set current camera pose
    viewer.set_CameraPoses(trajectory[i]);
    
    // Set ground truth pose
    if (has_groundtruth && i < trajectory_GT.size()) {
      viewer.set_gtPoses(trajectory_GT[i]);
    } else {
      viewer.set_gtPoses(trajectory[i]);
    }
    
    // Update every 100 frames to avoid too frequent updates
    if (i % 100 == 0) {
      std::cout << "Updated frame " << i << " / " << trajectory.size() << std::endl;
    }
    
    // Small delay to allow rendering
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  
  std::cout << "Trajectory visualization complete. Viewer window should be open." << std::endl;
  std::cout << "Viewer will run for 60 seconds. Close the window to exit early." << std::endl;
  
  // Keep the viewer running
  std::this_thread::sleep_for(std::chrono::seconds(60));
  
  return 0;
}
