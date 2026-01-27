#include "dynosam/dataprovider/TUMDataProvider.hpp"
#include "dynosam_cv/ImageContainer.hpp"
#include <fstream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>

namespace dyno {

TUMDataProvider::TUMDataProvider(const std::string& tum_path,
                                 const std::string& association_file,
                                 const CameraParams& camera_params)
    : tum_path_(tum_path), camera_params_(camera_params) {
  loadAssociationFile(association_file);
  LOG(INFO) << "TUMDataProvider initialized with " << rgb_files_.size() << " frames";
}

void TUMDataProvider::loadAssociationFile(const std::string& association_file) {
  std::ifstream fAssociation(association_file);
  if (!fAssociation.is_open()) {
    LOG(FATAL) << "Cannot open association file: " << association_file;
  }

  while (!fAssociation.eof()) {
    std::string s;
    getline(fAssociation, s);
    if (!s.empty() && s[0] != '#') {
      std::stringstream ss;
      ss << s;
      double t;
      std::string sRGB, sD;
      ss >> t;
      timestamps_.push_back(t);
      ss >> sRGB;
      rgb_files_.push_back(sRGB);
      ss >> t;  // Skip second timestamp
      ss >> sD;
      depth_files_.push_back(sD);
    }
  }
  fAssociation.close();
}

bool TUMDataProvider::spin() {
  if (current_frame_ >= rgb_files_.size()) {
    return false;  // Finished
  }

  // Load images
  std::string rgb_path = tum_path_ + "/" + rgb_files_[current_frame_];
  std::string depth_path = tum_path_ + "/" + depth_files_[current_frame_];

  cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_UNCHANGED);
  cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);

  if (rgb.empty()) {
    LOG(ERROR) << "Failed to load RGB image: " << rgb_path;
    current_frame_++;
    return true;  // Continue despite error
  }
  if (depth.empty()) {
    LOG(ERROR) << "Failed to load depth image: " << depth_path;
    current_frame_++;
    return true;
  }

  // Convert depth to float (TUM depth images are typically 16-bit)
  if (depth.type() == CV_16UC1) {
    depth.convertTo(depth, CV_64F, 1.0 / 5000.0);  // TUM depth scale factor
  }

  // Create empty optical flow and motion mask (TUM dataset doesn't provide these)
  cv::Mat optical_flow = cv::Mat::zeros(rgb.size(), CV_32FC2);
  cv::Mat motion = cv::Mat::zeros(rgb.size(), CV_32SC1);

  // Create ImageContainer
  FrameId frame_id = current_frame_;
  Timestamp timestamp = timestamps_[current_frame_];
  ImageContainer::Ptr image_container = std::make_shared<ImageContainer>(frame_id, timestamp);
  image_container->rgb(rgb)
      .depth(depth)
      .opticalFlow(optical_flow)
      .objectMotionMask(motion);

  // Call callback
  if (image_container_callback_) {
    image_container_callback_(image_container);
  }

  current_frame_++;
  return true;
}

}  // namespace dyno
