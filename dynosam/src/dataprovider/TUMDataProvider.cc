#include "dynosam/dataprovider/TUMDataProvider.hpp"
#include "dynosam/dataprovider/DatasetLoader.hpp"
#include "dynosam_cv/ImageContainer.hpp"
#include "dynosam_common/utils/FileSystem.hpp"
#include <fstream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>

namespace dyno {

class TUMAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(TUMAllLoader)

  TUMAllLoader(const std::string& tum_path, const std::string& association_file,
               const std::string& detections_json_path = "")
      : tum_path_(tum_path), detections_json_path_(detections_json_path) {
    loadAssociationFile(association_file);
  }

  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, rgb_files_.size());
    std::string rgb_path = tum_path_ + "/" + rgb_files_[idx];
    cv::Mat rgb = cv::imread(rgb_path, cv::IMREAD_UNCHANGED);
    if (rgb.empty()) {
      LOG(ERROR) << "Failed to load RGB image: " << rgb_path;
    }
    return rgb;
  }

  cv::Mat getDepthImage(size_t idx) const {
    CHECK_LT(idx, depth_files_.size());
    std::string depth_path = tum_path_ + "/" + depth_files_[idx];
    cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (depth.empty()) {
      LOG(ERROR) << "Failed to load depth image: " << depth_path;
      return cv::Mat();
    }
    // Convert depth to float (TUM depth images are typically 16-bit)
    if (depth.type() == CV_16UC1) {
      depth.convertTo(depth, CV_64F, 1.0 / 5000.0);  // TUM depth scale factor
    }
    return depth;
  }

  cv::Mat getOpticalFlow(size_t idx) const {
    CHECK_LT(idx, rgb_files_.size());
    // Load RGB to get size for empty flow
    cv::Mat rgb = getRGB(idx);
    if (rgb.empty()) {
      return cv::Mat();
    }
    // Create empty optical flow (TUM dataset doesn't provide this)
    return cv::Mat::zeros(rgb.size(), CV_32FC2);
  }

  cv::Mat getInstanceMask(size_t idx) const {
    CHECK_LT(idx, rgb_files_.size());
    // Load RGB to get size for mask
    cv::Mat rgb = getRGB(idx);
    if (rgb.empty()) {
      return cv::Mat();
    }
    
    // Always return empty mask - bbox information will be loaded separately
    // in FeatureTracker via loadDetections() which populates ObjectBoundaryMaskResult
    // Create empty motion mask (TUM dataset doesn't provide this natively)
    return cv::Mat::zeros(rgb.size(), CV_32SC1);
  }
  
  std::string getDetectionsJsonPath() const {
    return detections_json_path_;
  }
  
  std::string getRGBFilename(size_t idx) const {
    if (idx < rgb_files_.size()) {
      return rgb_files_[idx];
    }
    return "";
  }

  double getTimestamp(size_t idx) const {
    CHECK_LT(idx, timestamps_.size());
    return timestamps_[idx];
  }

  size_t size() const { return rgb_files_.size(); }

 private:
  void loadAssociationFile(const std::string& association_file) {
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

  std::string tum_path_;
  std::string detections_json_path_;
  std::vector<std::string> rgb_files_;
  std::vector<std::string> depth_files_;
  std::vector<double> timestamps_;
};

struct TUMTimestampLoader : public TimestampBaseLoader {
  TUMAllLoader::Ptr loader_;

  TUMTimestampLoader(TUMAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

TUMDataProvider::TUMDataProvider(const std::string& tum_path,
                                 const std::string& association_file,
                                 const CameraParams& camera_params,
                                 const std::string& detections_json_path)
    : TUMProvider(std::filesystem::path(tum_path)), camera_params_(camera_params) {
  LOG(INFO) << "Starting TUMDataProvider with path: " << tum_path;
  if (!detections_json_path.empty()) {
    LOG(INFO) << "Using detection JSON file: " << detections_json_path;
  }

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<TUMAllLoader>(tum_path, association_file,
                                               detections_json_path);
  auto timestamp_loader = std::make_shared<TUMTimestampLoader>(loader);

  CHECK(getCameraParams());

  auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getRGB(idx); });

  auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getOpticalFlow(idx); });

  auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getDepthImage(idx); });

  auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getInstanceMask(idx); });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader);

  auto callback = [&, loader](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask) -> bool {
    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(instance_mask);
    
    // Load bbox detections from JSON file if available
    std::string detections_json_path = loader->getDetectionsJsonPath();
    std::string rgb_filename = loader->getRGBFilename(frame_id);
    
    if (!detections_json_path.empty() && !rgb_filename.empty()) {
      // Load detections from JSON file (returns ObjectDetectionResult with all detection info)
      static_objects::ObjectDetectionResult detection_result = 
          utils::loadDetections(detections_json_path, rgb_filename, rgb);
      
      if (detection_result.num() > 0) {
        // Store complete detection result (category_id, score, bbox, ellipse) in ImageContainer
        image_container.staticDetectionResult(detection_result);
        
        // Also create a mask with bbox regions filled with category_id as object ID
        // This mask will be used by FeatureTracker to populate boundary_mask_result
        cv::Mat bbox_mask = cv::Mat::zeros(rgb.size(), CV_32SC1);
        
        for (const auto& detection : detection_result.detections) {
          // Convert BBox2 [x_min, y_min, x_max, y_max] to cv::Rect (x, y, width, height)
          const BBox2& bbox = detection.bbox;
          int x = static_cast<int>(std::round(bbox[0]));
          int y = static_cast<int>(std::round(bbox[1]));
          int width = static_cast<int>(std::round(bbox[2] - bbox[0]));
          int height = static_cast<int>(std::round(bbox[3] - bbox[1]));
          
          // Clamp to image bounds
          x = std::max(0, std::min(x, rgb.cols - 1));
          y = std::max(0, std::min(y, rgb.rows - 1));
          width = std::max(1, std::min(width, rgb.cols - x));
          height = std::max(1, std::min(height, rgb.rows - y));
          
          cv::Rect bbox_rect(x, y, width, height);
          
          // Use category_id as object_id (ensure > 0, as 0 is background)
          int object_id = detection.category_id > 0 ? detection.category_id : 1;
          
          // Fill bbox region with object ID
          bbox_mask(bbox_rect).setTo(cv::Scalar(object_id));
        }
        
        // Replace the empty instance_mask with bbox-based mask
        image_container.replace<ImageType::MotionMask>(ImageContainer::kObjectMask, bbox_mask);
        VLOG(30) << "Loaded " << detection_result.num() 
                 << " detections from JSON for frame " << frame_id
                 << " (category_ids, scores, bboxes, ellipses) - stored in staticDetectionResult";
      }
    }

    CHECK(image_container_callback_);
    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);
  LOG(INFO) << "TUMDataProvider initialized with " << loader->size() << " frames";
}

}  // namespace dyno
