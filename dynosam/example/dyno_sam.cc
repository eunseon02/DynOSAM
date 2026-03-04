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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Dense>
#include <png++/png.hpp>
#include <set>

#include "dynosam/dataprovider/ClusterSlamDataProvider.hpp"
#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/TartanAirShibuya.hpp"
#include "dynosam/dataprovider/TUMDataProvider.hpp"
#include "dynosam/dataprovider/ViodeDataProvider.hpp"
#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/pipeline/PipelineManager.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam/visualizer/OpenCVFrontendDisplay.hpp"
#include "dynosam/visualizer/VoViewer.hpp"
#include "dynosam/visualizer/EdgeVizUtils.hpp"
#include "dynosam/backend/BackendFactory.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/edge_map/localMap.hpp"
#include "dynosam/backend/edge_map/KeyFrame.hpp"
#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_common/Edge.hpp"
#include "dynosam_common/utils/Statistics.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/ImageContainer.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"

#include <pangolin/pangolin.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

// ROS2 includes for DynoPipelineManagerRos (optional)
#ifdef HAVE_DYNOSAM_ROS
#include "dynosam_ros/PipelineRos.hpp"
#include "dynosam_ros/Utils.hpp"
#include "rclcpp/executor.hpp"
#include "rclcpp/rclcpp.hpp"
#endif

DEFINE_string(path_to_kitti, "/root/data/kitti", "Path to KITTI dataset");
DEFINE_string(path_to_tum, "/root/data/TUM", "Path to TUM RGBD dataset");
DEFINE_string(tum_association, "", "Path to TUM association file (e.g., rgb.txt or depth.txt association)");
DECLARE_string(tum_detections_json);  // Defined in DataProviderFactory.cc
DEFINE_bool(use_tum, false, "Use TUM RGBD dataset instead of KITTI");
DEFINE_string(output_trajectory, "", "Output file path for TUM format trajectory (e.g., trajectory.txt)");
DEFINE_bool(use_pipeline, false, "Use full PipelineManager with backend (instead of tracker only)");
// TODO: (jesse) many better ways to do this with ros - just for now
DEFINE_string(
    params_folder_path, "",
    "Path to the folder containing the yaml files with the VIO parameters. "
    "If empty, tries to find params in install/share/dynosam/params or src/core/dynosam/params");

// Declare FLAGS_use_edge_feature (defined in RGBDInstanceFrontendModule.cc)
DECLARE_bool(use_edge_feature);

// Include header that defines g_frontend_module
#include "dynosam/frontend/FrontendModuleAccessor.hpp"

#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/OMDDataProvider.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>

// Load TUM RGBD dataset from association file
void LoadTUMImages(const std::string& strAssociationFilename,
                   std::vector<std::string>& vstrImageFilenamesRGB,
                   std::vector<std::string>& vstrImageFilenamesD,
                   std::vector<double>& vTimestamps) {
  std::ifstream fAssociation;
  fAssociation.open(strAssociationFilename.c_str());
  if (!fAssociation.is_open()) {
    LOG(FATAL) << "Cannot open association file: " << strAssociationFilename;
  }
  
  while (!fAssociation.eof()) {
    std::string s;
    getline(fAssociation, s);
    if (!s.empty() && s[0] != '#') {  // Skip empty lines and comments
      std::stringstream ss;
      ss << s;
      double t;
      std::string sRGB, sD;
      ss >> t;
      vTimestamps.push_back(t);
      ss >> sRGB;
      vstrImageFilenamesRGB.push_back(sRGB);
      ss >> t;  // Skip second timestamp
      ss >> sD;
      vstrImageFilenamesD.push_back(sD);
    }
  }
  fAssociation.close();
}

// Helper functions for edge-based visualization moved to library utility:
// `dynosam/visualizer/EdgeVizUtils.{hpp,cc}` (namespace `edge_viz`)

int main(int argc, char* argv[]) {
  using namespace dyno;
  
  // Build argument list with flag files first, then command line args
  // This ensures flag files are loaded before parsing command line arguments
  std::vector<std::string> argv_vec;
  argv_vec.push_back(argv[0]);  // program name
  
  // Try to find params folder to load flag files
  std::string params_path;
  
  // First, check if params_folder_path is provided in command line
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--params_folder_path=") == 0) {
      params_path = arg.substr(20);  // Extract path after "="
      break;
    }
  }
  
  // If not found, try to find params folder automatically
  if (params_path.empty()) {
    std::vector<std::string> candidate_paths = {
      "/home/user/dev_ws/install/dynosam/share/dynosam/params",
      "/home/user/dev_ws/src/core/dynosam/params"
    };
    for (const auto& candidate : candidate_paths) {
      std::ifstream test_file(candidate + "/PipelineParams.yaml");
      if (test_file.good()) {
        params_path = candidate;
        test_file.close();
        break;
      }
    }
  }
  
  // Ensure params_path ends with '/'
  if (!params_path.empty() && params_path.back() != '/') {
    params_path += "/";
  }
  
  // Load flag files if params folder found
  if (!params_path.empty() && std::filesystem::exists(params_path)) {
    LOG(INFO) << "Loading flag files from: " << params_path;
    for (const auto& entry : std::filesystem::directory_iterator(params_path)) {
      if (entry.is_regular_file() && entry.path().extension() == ".flags") {
        std::string flagfile_arg = "--flagfile=" + entry.path().string();
        argv_vec.push_back(flagfile_arg);
        LOG(INFO) << "  Added flag file: " << entry.path().filename();
      }
    }
  }
  
  // Add original command line arguments (they will override flag file settings)
  for (int i = 1; i < argc; ++i) {
    argv_vec.push_back(argv[i]);
  }
  
  // Convert to char** for ParseCommandLineFlags
  std::vector<char*> new_argv;
  for (auto& str : argv_vec) {
    new_argv.push_back(const_cast<char*>(str.c_str()));
  }
  new_argv.push_back(nullptr);
  
  int new_argc = new_argv.size() - 1;
  char** new_argv_ptr = new_argv.data();
  google::ParseCommandLineFlags(&new_argc, &new_argv_ptr, true);
  
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;
  // FLAGS_v is set by command line argument --v, don't override it
  // FLAGS_v = 30;
  
  // Log loaded flags for debugging (these are declared in RGBDInstanceFrontendModule.cc)
  // Note: We can't access them here directly, but they will be loaded from flag files

  FrontendParams fp;
  fp.tracker_params.feature_detector_type =
      TrackerParams::FeatureDetectorType::GFFT_CUDA;
  fp.tracker_params.max_dynamic_features_per_frame = 300;
  fp.tracker_params.prefer_provided_optical_flow = false;
  fp.tracker_params.prefer_provided_object_detection = false;
  // fp.tracker_params.feature_detector_type =
  // TrackerParams::FeatureDetectorType::ORB_SLAM_ORB;

  std::shared_ptr<Camera> camera;
  std::shared_ptr<FeatureTracker> tracker;

  // TUM RGBD dataset loading
  if (FLAGS_use_tum) {
    if (FLAGS_tum_association.empty()) {
      LOG(FATAL) << "TUM association file not specified! Use --tum_association=path/to/association.txt";
    }

    // Load DynoParams first to get camera params from CameraParams.yaml
    // Ensure params_path ends with '/'
    if (!params_path.empty() && params_path.back() != '/') {
      params_path += "/";
    }
    if (params_path.empty()) {
      LOG(FATAL) << "Could not find params folder. Please specify --params_folder_path";
    }
    LOG(INFO) << "Loading DynoParams from: " << params_path;
    DynoParams params(params_path);
    
    // Use camera params from DynoParams (loaded from CameraParams.yaml)
    CameraParams camera_params = params.camera_params_;
    // LOG(INFO) << "Using CameraParams from CameraParams.yaml: fx=" << camera_params.fx()
    //           << ", fy=" << camera_params.fy()
    //           << ", cx=" << camera_params.cu()
    //           << ", cy=" << camera_params.cv();

    // Use full pipeline with backend if requested
    if (FLAGS_use_pipeline) {
#ifdef HAVE_DYNOSAM_ROS
      LOG(INFO) << "Using full PipelineManagerRos with backend";
      
      // Initialize ROS2
      auto non_ros_args = dyno::initRosAndLogging(argc, argv);
      
      rclcpp::NodeOptions options;
      options.arguments(non_ros_args);
      options.use_intra_process_comms(true);
      
      // Create custom DynoNode that uses TUMDataProvider
      class TUMDynoPipelineManagerRos : public dyno::DynoPipelineManagerRos {
       public:
        TUMDynoPipelineManagerRos(const rclcpp::NodeOptions& options,
                                   const std::string& tum_path,
                                   const std::string& tum_association,
                                   const CameraParams& camera_params,
                                   const std::string& tum_detections_json)
            : dyno::DynoPipelineManagerRos(options),
              tum_path_(tum_path),
              tum_association_(tum_association),
              tum_detections_json_(tum_detections_json),
              camera_params_(camera_params) {}
        
       protected:
        dyno::DataProvider::Ptr createDataProvider() override {
          // Override to use TUMDataProvider instead of default
          return std::make_shared<TUMDataProvider>(
              tum_path_, tum_association_, camera_params_, tum_detections_json_);
        }
        
       private:
        std::string tum_path_;
        std::string tum_association_;
        std::string tum_detections_json_;
        CameraParams camera_params_;
      };
      
      // Create ROS pipeline with TUM dataset
      rclcpp::executors::MultiThreadedExecutor exec;
      auto ros_pipeline = std::make_shared<TUMDynoPipelineManagerRos>(
          options, FLAGS_path_to_tum, FLAGS_tum_association, camera_params,
          FLAGS_tum_detections_json);
      
      // Set params_folder_path parameter if not empty
      if (!FLAGS_params_folder_path.empty()) {
        ros_pipeline->declare_parameter("params_folder_path", FLAGS_params_folder_path);
      }
      
      // Initialize pipeline
      ros_pipeline->initalisePipeline();
      
      // Run pipeline
      exec.add_node(ros_pipeline);
      LOG(INFO) << "Starting ROS pipeline...";
      static int ros_frame_count = 0;
      while (rclcpp::ok()) {
        if (!ros_pipeline->spinOnce()) {
          break;
        }
        exec.spin_some();
        
        // Print statistics every 50 frames
        if (++ros_frame_count % 50 == 0) {
          LOG(INFO) << "\n=== Timing Statistics (frame " << ros_frame_count << ") ===\n"
                    << utils::Statistics::Print();
        }
      }
      
      ros_pipeline.reset();
      rclcpp::shutdown();
      
      LOG(INFO) << "\n=== Final Timing Statistics ===\n" << utils::Statistics::Print();
      return 0;
#else
      // Use non-ROS DynoPipelineManager
      LOG(INFO) << "Using non-ROS DynoPipelineManager with backend";
      
      // params and camera_params are already loaded above (before FLAGS_use_pipeline check)
      
      // Create TUM data provider
      auto data_provider = std::make_shared<TUMDataProvider>(
          FLAGS_path_to_tum, FLAGS_tum_association, camera_params,
          FLAGS_tum_detections_json);
      
      // Create displays
      std::string output_file = FLAGS_output_trajectory.empty() ? "/tmp/trajectory.txt" : FLAGS_output_trajectory;
      LOG(INFO) << "Creating displays with output file prefix: " << output_file;
      
      // Create a FrontendDisplay that also logs frontend poses and visualizes them
      // and includes edge-based visualization using VoViewer
      class FrontendPoseLoggerDisplay : public OpenCVFrontendDisplay {
       public:
        FrontendPoseLoggerDisplay(const std::string& output_file, bool show_window = true,
                                  const CameraParams& camera_params = CameraParams())
            : output_file_(output_file), show_window_(show_window), 
              edge_viewer_running_(false),
              camera_params_(camera_params) {
          frontend_trajectory_file_.open(output_file_ + ".frontend");
          if (!frontend_trajectory_file_.is_open()) {
            LOG(WARNING) << "Failed to open frontend trajectory output file: " << output_file_ + ".frontend";
          } else {
            LOG(INFO) << "Frontend pose logging to: " << output_file_ + ".frontend";
          }
          
          // Set camera params for edge visualization
          edge_viz::setCameraParams(camera_params.fx(), camera_params.fy(), 
                                     camera_params.cu(), camera_params.cv());
          
          if (show_window_) {
            // Start edge viewer thread if edge features are enabled (uses VoViewer)
            LOG(INFO) << "FrontendPoseLoggerDisplay: show_window_=true, FLAGS_use_edge_feature=" 
                      << FLAGS_use_edge_feature;
            if (FLAGS_use_edge_feature) {
              edge_viewer_running_ = true;
              edge_viewer_thread_ = std::thread(&FrontendPoseLoggerDisplay::runEdgeViewer, this);
              LOG(INFO) << "Edge-based VoViewer thread started";
            } else {
              LOG(WARNING) << "VoViewer not started: FLAGS_use_edge_feature is false";
            }
          } else {
            LOG(WARNING) << "VoViewer not started: show_window_ is false";
          }
        }
        
        ~FrontendPoseLoggerDisplay() {
          if (frontend_trajectory_file_.is_open()) {
            frontend_trajectory_file_.close();
            LOG(INFO) << "Frontend trajectory saved to: " << output_file_ + ".frontend" 
                      << " (total poses: " << frontend_pose_count_ << ")";
          }
          
          if (edge_viewer_running_) {
            edge_viewer_running_ = false;
            if (edge_viewer_thread_.joinable()) {
              edge_viewer_thread_.join();
            }
            // Clear global VoViewer pointer when viewer is destroyed
            dyno::g_vo_viewer = nullptr;
            LOG(INFO) << "Edge-based VoViewer thread stopped";
          }
        }
        
       protected:
        void spinOnceImpl(const VisionImuPacket::ConstPtr& input) override {
          // Call base class
          OpenCVFrontendDisplay::spinOnceImpl(input);
          
          // Log frontend pose if available
          if (input && frontend_trajectory_file_.is_open()) {
            const gtsam::Pose3& pose = input->cameraPose();
            const double timestamp = input->timestamp();
            const FrameId frame_id = input->frameId();
            
            const gtsam::Point3& t = pose.translation();
            const gtsam::Rot3& R = pose.rotation();
            const gtsam::Quaternion q = R.toQuaternion();
            
            // Save to file (TUM format: timestamp tx ty tz qx qy qz qw)
            frontend_trajectory_file_ << std::fixed << std::setprecision(6) 
                                       << timestamp << " "
                                       << t.x() << " " << t.y() << " " << t.z() << " "
                                       << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
            frontend_trajectory_file_.flush();
            
            frontend_pose_count_++;
            if (frontend_pose_count_ % 10 == 0) {
              VLOG(10) << "Frontend pose logged: frame=" << frame_id 
                        << ", timestamp=" << timestamp 
                        << ", total_poses=" << frontend_pose_count_;
            }
          }
        }
        
       private:
        void runEdgeViewer() {
          // Create VoViewer for edge-based visualization
          voViewer viewer("DynoSAM: Edge-based Local Map Viewer");
          
          // Set global VoViewer pointer for pause state checking
          dyno::g_vo_viewer = &viewer;
          
          std::vector<std::vector<cv::Point3d>> clusterClouds;
          std::vector<cv::Vec3b> clusterCloudColors;
          std::vector<std::vector<cv::Point3d>> localMapClouds;
          std::vector<Eigen::Matrix4d> slidingWindow;
          std::vector<std::vector<cv::Point3d>> environment_cloud;
          
          // Save edge keyframe trajectory
          std::string edge_kf_trajectory_file = output_file_ + ".edge_kf";
          std::ofstream edge_kf_trajectory_file_stream;
          bool trajectory_file_opened = false;
          
          LOG(INFO) << "Edge-based VoViewer loop started";
          
          // Set the map from frontend module (shared pointer)
          bool map_set = false;
          
          int loop_count = 0;
          while (edge_viewer_running_) {
            loop_count++;
            // Try to get frontend module
            auto frontend_module = g_frontend_module.lock();
            if (frontend_module) {
              // Set map once (shared pointer, so updates are automatically reflected)
              if (!map_set) {
                auto map = frontend_module->getMap();
                if (map) {
                  viewer.setMap(map);
                  map_set = true;
                  LOG(INFO) << "VoViewer: Map set successfully";
                }
              }
              
              // Get latest visualization snapshot (always available, updated every frame)
              auto latest_snap = frontend_module->getLatestVisualizationData();

              if (latest_snap) {
                // Always update pose/trajectory
                viewer.update_Trajectory(latest_snap->currentFramePose);
                viewer.set_CameraPoses(latest_snap->currentFramePose);
                viewer.set_gtPoses(latest_snap->currentFramePose);
                // Update heavy buffers only when the shared_ptr changes
                static std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> last_clusters;
                static std::shared_ptr<const std::vector<cv::Vec3b>> last_cluster_colors;
                static std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> last_local_map;
                static std::shared_ptr<const std::vector<Eigen::Matrix4d>> last_window;
                static std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> last_env;

                if (latest_snap->clusterClouds && latest_snap->clusterCloudColors &&
                    (latest_snap->clusterClouds != last_clusters ||
                     latest_snap->clusterCloudColors != last_cluster_colors)) {
                  viewer.update_covisibilityCloud(*latest_snap->clusterClouds,
                                                  *latest_snap->clusterCloudColors);
                  last_clusters = latest_snap->clusterClouds;
                  last_cluster_colors = latest_snap->clusterCloudColors;
                }

                if (latest_snap->localMapClouds && latest_snap->localMapClouds != last_local_map) {
                  viewer.update_localMap(*latest_snap->localMapClouds);
                  last_local_map = latest_snap->localMapClouds;
                }

                if (latest_snap->slidingWindow && latest_snap->slidingWindow != last_window) {
                  viewer.update_sliding_window(*latest_snap->slidingWindow);
                  last_window = latest_snap->slidingWindow;
                }

                if (latest_snap->environment_cloud && latest_snap->environment_cloud != last_env) {
                  viewer.update_Environment(*latest_snap->environment_cloud);
                  last_env = latest_snap->environment_cloud;
                }
              } else {
                static int empty_snap_count = 0;
                if (empty_snap_count % 100 == 0) {
                  LOG(WARNING) << "VoViewer: no visualization snapshot available yet";
                }
                empty_snap_count++;
              }

              // Optional: save keyframe trajectory (uses local_map, but not used for rendering)
              auto local_map = frontend_module->getLocalMap();
              if (local_map && !local_map->mvKeyFrames.empty()) {
                static size_t last_keyframe_count = 0;
                size_t current_keyframe_count = local_map->mvKeyFrames.size();
                if (current_keyframe_count != last_keyframe_count) {
                  edge_viz::saveEdgeKeyFrameTrajectory(edge_kf_trajectory_file, local_map);
                  last_keyframe_count = current_keyframe_count;
                }
              }
            } else {
              static int null_count = 0;
              if (null_count % 100 == 0) {
                LOG(WARNING) << "VoViewer: g_frontend_module is null or expired (loop_count=" 
                             << loop_count << ")";
              }
              null_count++;
            }
            
            // Log every 1000 iterations to confirm loop is running
            if (loop_count % 1000 == 0) {
              LOG(INFO) << "VoViewer loop running: iteration=" << loop_count 
                        << ", edge_viewer_running_=" << edge_viewer_running_;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          
          if (trajectory_file_opened) {
            edge_kf_trajectory_file_stream.close();
          }
          
          LOG(INFO) << "Edge-based VoViewer loop ended";
        }
        
        std::string output_file_;
        std::ofstream frontend_trajectory_file_;
        size_t frontend_pose_count_ = 0;
        
        // Edge-based visualization (uses VoViewer)
        bool show_window_;
        std::atomic<bool> edge_viewer_running_;
        std::thread edge_viewer_thread_;
        CameraParams camera_params_;
      };
      
      auto frontend_display = std::make_shared<FrontendPoseLoggerDisplay>(output_file, true, camera_params);

      // Backend display can be a no-op if trajectory logging is disabled
      class NullBackendDisplay : public BackendDisplay {
       protected:
        void spinOnceImpl(const BackendOutputPacket::ConstPtr&) override {}
      };

      auto backend_display = std::make_shared<NullBackendDisplay>();
      
      // Create backend factory
      // Use DefaultRegularBackendModuleFactory which is BackendFactory<NoVizPolicy, RegularBackendModuleTraits::MapType>
      BackendModuleFactory::Ptr backend_factory = DefaultRegularBackendModuleFactory::Create(params.backend_type);
      
      // Create pipeline manager
      // Note: make_shared uses perfect forwarding, so we need to ensure const reference
      // g_frontend_module will be set automatically in PipelineManager::loadPipelines
      auto pipeline = std::make_shared<DynoPipelineManager>(
          static_cast<const DynoParams&>(params), data_provider, frontend_display, backend_display, backend_factory);
      
      // Run pipeline
      LOG(INFO) << "Starting non-ROS pipeline...";
      static int pipeline_frame_count = 0;
      while (pipeline->spin()) {
        // Print statistics every 50 frames
        // if (++pipeline_frame_count % 100 == 0) {
        //   LOG(INFO) << "\n=== Timing Statistics (frame " << pipeline_frame_count << ") ===\n"
        //             << utils::Statistics::Print();
        // }
      }
      
      LOG(INFO) << "Pipeline finished";
      LOG(INFO) << "\n=== Final Timing Statistics ===\n" << utils::Statistics::Print();
      return 0;
#endif
    }

    // Original tracker-only code
    // Load TUM images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    LoadTUMImages(FLAGS_tum_association, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty()) {
      LOG(FATAL) << "No images found in TUM dataset!";
    } else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size()) {
      LOG(FATAL) << "Different number of images for rgb and depth in TUM dataset!";
    }
    
    camera = std::make_shared<Camera>(camera_params);
    tracker = std::make_shared<FeatureTracker>(fp, camera);
    
    // Create motion solver for pose estimation
    EgoMotionSolver::Params motion_solver_params;
    EgoMotionSolver motion_solver(motion_solver_params, camera_params);

    LOG(INFO) << "Starting TUM RGBD dataset processing with " << nImages << " images";

    // Storage for trajectory output
    std::vector<gtsam::Pose3> camera_poses;
    std::vector<double> pose_timestamps;
    std::ofstream trajectory_file;
    if (!FLAGS_output_trajectory.empty()) {
      trajectory_file.open(FLAGS_output_trajectory);
      if (!trajectory_file.is_open()) {
        LOG(WARNING) << "Failed to open trajectory output file: " << FLAGS_output_trajectory;
      } else {
        LOG(INFO) << "Saving trajectory to: " << FLAGS_output_trajectory;
      }
    }

    // Process TUM images
    for (int ni = 0; ni < nImages; ++ni) {
      dyno::FrameId frame_id = ni;
      dyno::Timestamp timestamp = vTimestamps[ni];

      // Load images
      std::string rgb_path = FLAGS_path_to_tum + "/" + vstrImageFilenamesRGB[ni];
      std::string depth_path = FLAGS_path_to_tum + "/" + vstrImageFilenamesD[ni];

      cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_UNCHANGED);
      cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

      if (rgb.empty()) {
        LOG(ERROR) << "Failed to load RGB image: " << rgb_path;
        continue;
      }
      if (depth.empty()) {
        LOG(ERROR) << "Failed to load depth image: " << depth_path;
        continue;
      }

      // Convert depth to float (TUM depth images are typically 16-bit)
      if (depth.type() == CV_16UC1) {
        double depth_scale = camera_params.hasDepthParams() 
                             ? camera_params.depthParams().depth_to_meters 
                             : (1.0 / 5000.0);  // Default TUM scale factor
        depth.convertTo(depth, CV_64F, depth_scale);
      }

      // Create empty optical flow and motion mask (TUM dataset doesn't provide these)
      cv::Mat optical_flow = cv::Mat::zeros(rgb.size(), CV_32FC2);
      cv::Mat motion = cv::Mat::zeros(rgb.size(), CV_32SC1);

      LOG(INFO) << frame_id << " " << timestamp;

      cv::Mat of_viz, motion_viz, depth_viz;
      of_viz = ImageType::OpticalFlow::toRGB(optical_flow);
      motion_viz = ImageType::MotionMask::toRGB(motion);
      depth_viz = ImageType::Depth::toRGB(depth);

      ImageContainer image_container(frame_id, timestamp);
      image_container.rgb(rgb)
          .depth(depth)
          .opticalFlow(optical_flow)
          .objectMotionMask(motion);

      auto frame = tracker->track(frame_id, timestamp, image_container);
      Frame::Ptr previous_frame = tracker->getPreviousFrame();
      
      // Print statistics every 50 frames
      static int tracker_frame_count = 0;
      if (++tracker_frame_count % 50 == 0) {
        LOG(INFO) << "\n=== Timing Statistics (frame " << tracker_frame_count << ") ===\n"
                  << utils::Statistics::Print();
      }

      // Estimate camera pose
      if (frame) {
        if (ni == 0) {
          // First frame: set to identity
          frame->T_world_camera_ = gtsam::Pose3::Identity();
        } else if (previous_frame) {
          // Estimate relative pose between frames
          Pose3SolverResult result = motion_solver.geometricOutlierRejection3d2d(
              previous_frame, frame, std::nullopt);
          
          if (result.status == TrackingStatus::VALID) {
            // Update pose: T_world_k = T_world_k_1 * T_k_1_k
            // result.best_result is T_k_1_k (relative pose from k-1 to k)
            frame->T_world_camera_ = previous_frame->T_world_camera_ * result.best_result;
          } else {
            // If pose estimation fails, use previous pose (or identity)
            LOG(WARNING) << "Failed to estimate pose at frame " << frame_id 
                         << ", using previous pose";
            frame->T_world_camera_ = previous_frame->T_world_camera_;
          }
        }
      }

      // Store camera pose for trajectory output
      if (frame && !FLAGS_output_trajectory.empty() && trajectory_file.is_open()) {
        const gtsam::Pose3& T_world_camera = frame->T_world_camera_;
        camera_poses.push_back(T_world_camera);
        pose_timestamps.push_back(timestamp);
      }

      cv::Mat tracking;
      if (previous_frame) {
        ImageTracksParams track_viz_params(true);
        track_viz_params.show_intermediate_tracking = true;
        tracking = tracker->computeImageTracks(*previous_frame, *frame, track_viz_params);
      }
      if (!tracking.empty()) cv::imshow("Tracking", tracking);

      LOG(INFO) << to_string(tracker->getTrackerInfo());
      cv::waitKey(1);
    }

    // Save trajectory in TUM format
    if (!FLAGS_output_trajectory.empty() && trajectory_file.is_open()) {
      LOG(INFO) << "Saving " << camera_poses.size() << " poses to TUM format trajectory file";
      for (size_t i = 0; i < camera_poses.size(); ++i) {
        const gtsam::Pose3& pose = camera_poses[i];
        const gtsam::Point3& t = pose.translation();
        const gtsam::Rot3& R = pose.rotation();
        const gtsam::Quaternion q = R.toQuaternion();
        
        // TUM format: timestamp tx ty tz qx qy qz qw
        trajectory_file << std::fixed << std::setprecision(6) 
                       << pose_timestamps[i] << " "
                       << t.x() << " " << t.y() << " " << t.z() << " "
                       << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
      }
      trajectory_file.close();
      LOG(INFO) << "Trajectory saved to: " << FLAGS_output_trajectory;
    }

    LOG(INFO) << "\n=== Final Timing Statistics ===\n" << utils::Statistics::Print();
    return 0;
  }

  // Original KITTI/other dataset loading
  KittiDataLoader::Params params;
  KittiDataLoader loader("/root/data/vdo_slam/kitti/kitti/0004/", params);
  // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-S2");
  // loader.setStartingFrame(600);
  // OMDDataLoader loader(
  //     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/");

  // TartanAirShibuyaLoader
  // loader("/root/data/TartanAir_shibuya/RoadCrossing07/");
  // ViodeLoader loader("/root/data/VIODE/city_day/mid");

  // auto detector = dyno::PyObjectDetectorWrapper::CreateYoloDetector();
  // CHECK_NOTNULL(detector);

  camera = std::make_shared<Camera>(*loader.getCameraParams());
  tracker = std::make_shared<FeatureTracker>(fp, camera);

  loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
                         cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
                         cv::Mat motion, gtsam::Pose3,
                         GroundTruthInputPacket) -> bool {
    // LOG(INFO) << utils::Statistics::Print();
    // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
    //                        cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
    //                        cv::Mat motion, GroundTruthInputPacket,
    //                        std::optional<ImuMeasurements> imu_measurements,
    //                        std::optional<cv::Mat>) -> bool {
    // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
    //                        cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth,
    //                        cv::Mat motion, GroundTruthInputPacket,
    //                        std::optional<cv::Mat>) -> bool {
    LOG(INFO) << frame_id << " " << timestamp;

    cv::Mat of_viz, motion_viz, depth_viz;
    of_viz = ImageType::OpticalFlow::toRGB(optical_flow);
    motion_viz = ImageType::MotionMask::toRGB(motion);
    depth_viz = ImageType::Depth::toRGB(depth);

    // ImageContainerDeprecate::Ptr container = ImageContainerDeprecate::Create(
    //     timestamp, frame_id, ImageWrapper<ImageType::RGBMono>(rgb),
    //     ImageWrapper<ImageType::Depth>(depth),
    //     ImageWrapper<ImageType::OpticalFlow>(optical_flow),
    //     ImageWrapper<ImageType::MotionMask>(motion));

    // cv::Mat boarder_mask;
    // vision_tools::computeObjectMaskBoundaryMask(
    //     motion,
    //     boarder_mask,
    //     8
    // );

    // cv::Scalar red = dyno::Color::red();

    // const ObjectIds instance_labels = vision_tools::getObjectLabels(motion);
    // for(const auto object_id : instance_labels) {
    //     std::vector<std::vector<cv::Point>> detected_contours;
    //     vision_tools::findObjectBoundingBox(motion,
    //     object_id,detected_contours);

    //     cv::drawContours(boarder_mask, detected_contours, -1, red, 8);
    // }

    // cv::imshow("Mask with boarder", boarder_mask);

    // cv::imshow("RGB", rgb);
    // cv::imshow("OF", of_viz);
    // cv::imshow("Motion", motion_viz);
    // // cv::waitKey(1);
    // cv::imshow("Depth", depth_viz);

    // auto object_detection_result = detector->process(rgb);
    // cv::imshow("Detection Result", object_detection_result.colouredMask());

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(motion);
    // image_container.rgb(rgb).depth(depth).opticalFlow(optical_flow);
    auto frame = tracker->track(frame_id, timestamp, image_container);
    Frame::Ptr previous_frame = tracker->getPreviousFrame();

    // if(frame_id == 605) {
    //   auto all_tracks = frame->static_features_.collectTracklets();
    //   frame->static_features_.markOutliers(all_tracks);
    // }

    // // motion_viz =
    // ImageType::MotionMask::toRGB(frame->image_container_.get<ImageType::MotionMask>());
    // // // cv::imshow("Motion", motion_viz);

    cv::Mat tracking;
    if (previous_frame) {
      ImageTracksParams track_viz_params(true);
      track_viz_params.show_intermediate_tracking = true;
      tracking = tracker->computeImageTracks(*previous_frame, *frame,
                                             track_viz_params);

      // if (imu_measurements) {
      //   const auto previous_timestamp = previous_frame->getTimestamp();

      //   CHECK_GE(imu_measurements->timestamps_[0], previous_timestamp);
      //   CHECK_LT(imu_measurements
      //                ->timestamps_[imu_measurements->timestamps_.cols() - 1],
      //            timestamp);

      //   LOG(INFO) << "Gotten imu messages!";

      //   CHECK(imu_measurements->synchronised_frame_id);
      //   CHECK_EQ(imu_measurements->synchronised_frame_id.value(),
      //            frame->getFrameId());
      // }
    }
    if (!tracking.empty()) cv::imshow("Tracking", tracking);

    LOG(INFO) << to_string(tracker->getTrackerInfo());
    
    // Print statistics every 50 frames
    static int loader_frame_count = 0;
    if (++loader_frame_count % 50 == 0) {
      LOG(INFO) << "\n=== Timing Statistics (frame " << loader_frame_count << ") ===\n"
                << utils::Statistics::Print();
    }
    
    const std::string path = "/root/results/misc/";
    // if (previous_frame && (char)cv::waitKey(0) == 's') {
    //   LOG(INFO) << "Saving...";
    //   // cv::imwrite(path + "omd_su4_rgb.png", rgb);
    //   // cv::imwrite(path + "omd_su4_of.png", of_viz);
    //   // cv::imwrite(path + "omd_su4_motion.png", motion_viz);
    //   // cv::imwrite(path + "omd_su4_depth.png", depth_viz);
    //   // cv::imwrite(
    //   //     path + "cluster_tracking_new" + std::to_string(frame_id) +
    //   ".png",
    //   //     tracking);
    // }
    cv::waitKey(1);

    return true;
  });

  while (loader.spin()) {
  }
  
  LOG(INFO) << "\n=== Final Timing Statistics ===\n" << utils::Statistics::Print();
}

// #include "dynosam/dataprovider/ProjectAriaDataProvider.hpp"
// #include "dynosam/frontend/vision/VisionTools.hpp"

// int main(int argc, char* argv[]) {

//     using namespace dyno;
//     google::ParseCommandLineFlags(&argc, &argv, true);
//     google::InitGoogleLogging(argv[0]);
//     FLAGS_logtostderr = 1;
//     FLAGS_colorlogtostderr = 1;
//     FLAGS_log_prefix = 1;

//     // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-S1");
//     ProjectARIADataLoader loader("/root/data/zed/acfr_3_moving_medium/");

//     loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp,
//     cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion) -> bool
//     {

//         LOG(INFO) << frame_id << " " << timestamp;

//         cv::imshow("RGB", rgb);
//         cv::imshow("OF", ImageType::OpticalFlow::toRGB(optical_flow));
//         cv::imshow("Motion", ImageType::MotionMask::toRGB(motion));
//         cv::imshow("Depth", ImageType::Depth::toRGB(depth));

//         cv::Mat shrunk_mask;
//         vision_tools::shrinkMask(motion, shrunk_mask, 20);
//         cv::imshow("Shrunk Motion",
//         ImageType::MotionMask::toRGB(shrunk_mask));

//         cv::waitKey(1);
//         return true;
//     });

//     while(loader.spin()) {}

// }
