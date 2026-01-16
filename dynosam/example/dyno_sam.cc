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

#include "dynosam/dataprovider/ClusterSlamDataProvider.hpp"
#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/TartanAirShibuya.hpp"
#include "dynosam/dataprovider/ViodeDataProvider.hpp"
#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/pipeline/PipelineManager.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam/visualizer/OpenCVFrontendDisplay.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/ImageContainer.hpp"
#include "dynosam_nn/PyObjectDetector.hpp"

DEFINE_string(path_to_kitti, "/root/data/kitti", "Path to KITTI dataset");
DEFINE_string(path_to_tum, "/root/data/TUM", "Path to TUM RGBD dataset");
DEFINE_string(tum_association, "", "Path to TUM association file (e.g., rgb.txt or depth.txt association)");
DEFINE_bool(use_tum, false, "Use TUM RGBD dataset instead of KITTI");
// TODO: (jesse) many better ways to do this with ros - just for now
DEFINE_string(
    params_folder_path, "dynosam/params",
    "Path to the folder containing the yaml files with the VIO parameters.");

#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/OMDDataProvider.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include <fstream>
#include <sstream>

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

int main(int argc, char* argv[]) {
  using namespace dyno;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;
  FLAGS_v = 30;

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

    // Load camera parameters (TUM uses standard camera params)
    // You may need to adjust this based on your camera configuration
    // Default TUM camera parameters - adjust these based on your dataset
    // Format: [fx, fy, cx, cy]
    CameraParams::IntrinsicsCoeffs intrinsics({525.0, 525.0, 319.5, 239.5});
    CameraParams::DistortionCoeffs distortion({0.0, 0.0, 0.0, 0.0});  // TUM typically has minimal distortion
    cv::Size image_size(640, 480);  // Adjust based on your TUM dataset
    auto distortion_model = CameraParams::stringToDistortion("radtan", "pinhole");
    
    CameraParams camera_params(intrinsics, distortion, image_size, distortion_model);
    camera = std::make_shared<Camera>(camera_params);
    tracker = std::make_shared<FeatureTracker>(fp, camera);

    LOG(INFO) << "Starting TUM RGBD dataset processing with " << nImages << " images";

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
        depth.convertTo(depth, CV_64F, 1.0 / 5000.0);  // TUM depth scale factor
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
