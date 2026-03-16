#include "dynosam/dataprovider/TUMDataProvider.hpp"
#include "dynosam/dataprovider/DatasetLoader.hpp"
#include "dynosam_cv/ImageContainer.hpp"
#include "dynosam_common/utils/FileSystem.hpp"
#include "dynosam_common/StaticObjects.hpp"
#include <fstream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace dyno {

class TUMAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(TUMAllLoader)

  TUMAllLoader(const std::string& tum_path, const std::string& association_file,
               const std::string& detections_json_path = "")
      : tum_path_(tum_path), detections_json_path_(detections_json_path) {
    loadAssociationFile(association_file);
    // Pre-load and cache all detections from JSON file if available
    if (!detections_json_path_.empty()) {
      loadDetectionsCache();
    }
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

    // Try to load a precomputed segmentation / instance mask image.
    // Convention (matched to generate_detection_files.py):
    //   - Python script saves per-frame id mask as:
    //       <tum_root>/masks/<rgb_basename_without_ext>_id.png
    //   - Each pixel value is a class / instance id (background = 0).
    try {
      if (!tum_path_.empty()) {
        namespace fs = std::filesystem;
        const std::string& rgb_rel = rgb_files_[idx];
        fs::path rgb_path(rgb_rel);
        std::string stem = rgb_path.stem().string();
        fs::path mask_path =
            fs::path(tum_path_) / "masks" / fs::path(stem + "_id.png");

        if (fs::exists(mask_path)) {
          cv::Mat mask_raw =
              cv::imread(mask_path.string(), cv::IMREAD_UNCHANGED);
          if (mask_raw.empty()) {
            LOG(WARNING) << "Mask file exists but failed to load: "
                         << mask_path.string();
          } else {
            // Convert to expected MotionMask type (CV_32SC1)
            cv::Mat mask_converted;
            mask_raw.convertTo(mask_converted, CV_32SC1);
            if (mask_converted.size() != rgb.size()) {
              LOG(WARNING) << "Mask size mismatch, expected " << rgb.cols << "x"
                           << rgb.rows << " but got " << mask_converted.cols
                           << "x" << mask_converted.rows << " for file "
                           << mask_path.string()
                           << ". Falling back to empty mask.";
            } else {
              return mask_converted;
            }
          }
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Exception while trying to load TUM instance mask: "
                   << e.what();
    }

    // Fallback: empty mask (no motion / instance labels)
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
  
  // Get cached detections for a specific frame (fast O(1) lookup)
  // Similar to DetectionsFromFile::detect() - just return from cache
  static_objects::ObjectDetectionResult getDetectionsForFrame(const std::string& rgb_filename) const {
    if (detections_cache_.empty()) {
      return static_objects::ObjectDetectionResult{};
    }
    
    // Extract base filename for matching (same as DetectionsFromFile)
    std::string basename = std::filesystem::path(rgb_filename).filename().string();
    
    auto it = detections_cache_.find(basename);
    if (it == detections_cache_.end()) {
      return static_objects::ObjectDetectionResult{};
    }
    
    return it->second;  // Return cached result directly
  }

 private:
  void loadDetectionsCache() {
    if (detections_json_path_.empty()) {
      return;
    }
    
    std::ifstream fin(detections_json_path_);
    if (!fin.is_open()) {
      LOG(WARNING) << "Failed to open JSON detection file: " << detections_json_path_;
      return;
    }
    
    nlohmann::json data;
    try {
      fin >> data;  // ⭐ JSON 파일 전체를 한 번에 읽어서 data에 저장
    } catch (const nlohmann::json::exception& e) {
      LOG(ERROR) << "Failed to parse JSON file: " << detections_json_path_ << ", error: " << e.what();
      return;
    }
    
    if (!data.is_array()) {
      LOG(ERROR) << "JSON file does not contain an array: " << detections_json_path_;
      return;
    }
    
    // ⭐ 모든 프레임을 순회하면서 미리 파싱 (DetectionsFromFile 패턴)
    size_t loaded_count = 0;
    for (auto& frame : data) {
      if (!frame.contains("file_name")) {
        continue;
      }
      
      std::string name = frame["file_name"].get<std::string>();
      name = std::filesystem::path(name).filename().string();  // basename만 저장
      
      static_objects::ObjectDetectionResult result;
      
      if (!frame.contains("detections") || !frame["detections"].is_array()) {
        detections_cache_[name] = result;  // Empty result for this frame
        continue;
      }
      
      // 각 detection 파싱해서 Detection 객체 생성
      for (auto& d : frame["detections"]) {
        if (!d.contains("category_id") || !d["category_id"].is_number()) {
          continue;
        }
        
        unsigned int cat = d["category_id"].get<unsigned int>();
        
        double score = 0.0;
        if (d.contains("detection_score") && d["detection_score"].is_number()) {
          score = d["detection_score"].get<double>();
        } else if (d.contains("score") && d["score"].is_number()) {
          score = d["score"].get<double>();
        }
        
        if (!d.contains("bbox") || !d["bbox"].is_array() || d["bbox"].size() != 4) {
          continue;
        }
        
        auto bb = d["bbox"];
        BBox2 bbox(bb[0].get<double>(), bb[1].get<double>(), 
                   bb[2].get<double>(), bb[3].get<double>());
        
        Ellipse ellipse;
        if (d.contains("ellipse") && d["ellipse"].is_array() && d["ellipse"].size() == 5) {
          auto ellipse_data = d["ellipse"];
          double cx = ellipse_data[0].get<double>();
          double cy = ellipse_data[1].get<double>();
          double width = ellipse_data[2].get<double>();
          double height = ellipse_data[3].get<double>();
          double angle = ellipse_data[4].get<double>();
          
          ellipse = Ellipse(Eigen::Vector2d(0.5 * width, 0.5 * height), 
                           angle, 
                           Eigen::Vector2d(cx, cy));
        } else {
          ellipse = Ellipse::FromBbox(bbox, 0.0);
        }
        
        result.detections.emplace_back(cat, score, bbox, ellipse);
      }
      
      detections_cache_[name] = result;  // ⭐ 메모리에 캐싱
      loaded_count++;
    }
    
    LOG(INFO) << "Loaded and cached " << loaded_count << " frames of detections from JSON file";
  }
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
  // Cache for pre-loaded detections (filename -> detection result)
  std::unordered_map<std::string, static_objects::ObjectDetectionResult> detections_cache_;
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
    
    // Get cached detections from pre-loaded JSON (fast O(1) lookup, like DetectionsFromFile::detect())
    std::string rgb_filename = loader->getRGBFilename(frame_id);
    
    if (!rgb_filename.empty()) {
      static_objects::ObjectDetectionResult detection_result = 
          loader->getDetectionsForFrame(rgb_filename);
      
      if (detection_result.num() > 0) {
        // Set input_image for the result (required by ObjectDetectionResult)
        detection_result.input_image = rgb;
        
        // Store complete static object detection result (category_id, score, bbox, ellipse) in ImageContainer
        // This is for static objects only - do NOT set objectMotionMask (that's for dynamic objects)
        image_container.staticDetectionResult(detection_result);
        VLOG(30) << "Loaded " << detection_result.num() 
                 << " static object detections from cache for frame " << frame_id
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
