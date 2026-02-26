#include "dynosam_common/utils/FileSystem.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <nlohmann/json.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/Ellipse.hpp"

namespace dyno {
namespace utils {

void throwExceptionIfPathInvalid(const std::string& image_path) {
  namespace fs = std::filesystem;
  if (!fs::exists(image_path)) {
    throw std::runtime_error("Path does not exist: " + image_path);
  }
}

void loadRGB(const std::string& image_path, cv::Mat& img) {
  throwExceptionIfPathInvalid(image_path);
  img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
}

void loadFlow(const std::string& image_path, cv::Mat& img) {
  throwExceptionIfPathInvalid(image_path);
  img = readOpticalFlow(image_path);
}

void loadDepth(const std::string& image_path, cv::Mat& img) {
  throwExceptionIfPathInvalid(image_path);
  img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
  img.convertTo(img, CV_64F);
}

void loadSemanticMask(const std::string& image_path, const cv::Size& size,
                      cv::Mat& mask) {
  throwExceptionIfPathInvalid(image_path);
  CHECK(!size.empty());

  mask = cv::Mat(size, CV_32SC1);

  std::ifstream file_mask;
  file_mask.open(image_path.c_str());

  int count = 0;
  while (!file_mask.eof()) {
    std::string s;
    getline(file_mask, s);
    if (!s.empty()) {
      std::stringstream ss;
      ss << s;
      int tmp;
      for (int i = 0; i < mask.cols; ++i) {
        ss >> tmp;
        if (tmp != 0) {
          mask.at<int>(count, i) = tmp;
        } else {
          mask.at<int>(count, i) = 0;
        }
      }
      count++;
    }
  }

  file_mask.close();
}

void loadMask(const std::string& image_path, cv::Mat& mask) {
  throwExceptionIfPathInvalid(image_path);
  mask = cv::imread(image_path, cv::IMREAD_UNCHANGED);
  mask.convertTo(mask, CV_32SC1);
}

bool loadDetections(const std::string& json_path,
                    const std::string& image_filename,
                    const cv::Size& image_size,
                    std::vector<int>& object_ids,
                    std::vector<cv::Rect>& bounding_boxes) {
  throwExceptionIfPathInvalid(json_path);
  CHECK(!image_size.empty());

  // Clear output vectors
  object_ids.clear();
  bounding_boxes.clear();

  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    LOG(WARNING) << "Failed to open JSON detection file: " << json_path;
    return false;
  }

  nlohmann::json data;
  try {
    json_file >> data;
  } catch (const nlohmann::json::exception& e) {
    LOG(ERROR) << "Failed to parse JSON file: " << json_path << ", error: " << e.what();
    return false;
  }

  if (!data.is_array()) {
    LOG(ERROR) << "JSON file does not contain an array: " << json_path;
    return false;
  }

  // Extract base filename for matching (without path)
  std::filesystem::path image_path_obj(image_filename);
  std::string image_basename = image_path_obj.filename().string();

  // Find matching frame in JSON data
  bool found_frame = false;
  for (const auto& frame : data) {
    if (!frame.contains("file_name")) {
      continue;
    }

    std::string frame_filename = frame["file_name"].get<std::string>();
    std::filesystem::path frame_path_obj(frame_filename);
    std::string frame_basename = frame_path_obj.filename().string();

    // Match by filename (with or without extension)
    if (frame_basename == image_basename ||
        frame_basename == image_path_obj.stem().string() ||
        image_basename == frame_path_obj.stem().string()) {
      found_frame = true;

      if (!frame.contains("detections") || !frame["detections"].is_array()) {
        LOG(WARNING) << "Frame found but no detections array in JSON: " << json_path;
        return false;
      }

      // Process each detection - extract bboxes only
      int object_id_counter = 1;  // Start from 1 (0 is background)
      for (const auto& detection : frame["detections"]) {
        if (!detection.contains("bbox") || !detection["bbox"].is_array() ||
            detection["bbox"].size() != 4) {
          LOG(WARNING) << "Invalid bbox in detection, skipping";
          continue;
        }

        // Get bbox: [x_min, y_min, x_max, y_max]
        double x_min = detection["bbox"][0].get<double>();
        double y_min = detection["bbox"][1].get<double>();
        double x_max = detection["bbox"][2].get<double>();
        double y_max = detection["bbox"][3].get<double>();

        // Convert to integer coordinates
        int x = static_cast<int>(std::round(x_min));
        int y = static_cast<int>(std::round(y_min));
        int width = static_cast<int>(std::round(x_max - x_min));
        int height = static_cast<int>(std::round(y_max - y_min));

        // Clamp to image bounds
        x = std::max(0, std::min(x, image_size.width - 1));
        y = std::max(0, std::min(y, image_size.height - 1));
        width = std::max(1, std::min(width, image_size.width - x));
        height = std::max(1, std::min(height, image_size.height - y));

        cv::Rect bbox(x, y, width, height);

        // Use category_id if available, otherwise use sequential ID
        int object_id = object_id_counter;
        if (detection.contains("category_id") && detection["category_id"].is_number()) {
          int category_id = detection["category_id"].get<int>();
          if (category_id > 0) {  // 0 is background
            object_id = category_id;
          }
        }

        // Store bbox and object ID
        object_ids.push_back(object_id);
        bounding_boxes.push_back(bbox);

        object_id_counter++;
      }

      break;  // Found matching frame, exit loop
    }
  }

  if (!found_frame) {
    LOG(WARNING) << "No matching frame found for image: " << image_filename
                 << " in JSON file: " << json_path;
    return false;
  }

  return true;
}

static_objects::ObjectDetectionResult loadDetections(
    const std::string& json_path,
    const std::string& image_filename,
    const cv::Mat& input_image) {
  static_objects::ObjectDetectionResult result;
  
  throwExceptionIfPathInvalid(json_path);
  CHECK(!input_image.empty());
  
  // Store input image
  input_image.copyTo(result.input_image);
  
  std::ifstream json_file(json_path);
  if (!json_file.is_open()) {
    LOG(WARNING) << "Failed to open JSON detection file: " << json_path;
    return result;
  }
  
  nlohmann::json data;
  try {
    json_file >> data;
  } catch (const nlohmann::json::exception& e) {
    LOG(ERROR) << "Failed to parse JSON file: " << json_path << ", error: " << e.what();
    return result;
  }
  
  if (!data.is_array()) {
    LOG(ERROR) << "JSON file does not contain an array: " << json_path;
    return result;
  }
  
  // Extract base filename for matching (without path)
  std::filesystem::path image_path_obj(image_filename);
  std::string image_basename = image_path_obj.filename().string();
  
  // Find matching frame in JSON data
  bool found_frame = false;
  for (const auto& frame : data) {
    if (!frame.contains("file_name")) {
      continue;
    }
    
    std::string frame_filename = frame["file_name"].get<std::string>();
    std::filesystem::path frame_path_obj(frame_filename);
    std::string frame_basename = frame_path_obj.filename().string();
    
    // Match by filename (with or without extension)
    if (frame_basename == image_basename ||
        frame_basename == image_path_obj.stem().string() ||
        image_basename == frame_path_obj.stem().string()) {
      found_frame = true;
      
      if (!frame.contains("detections") || !frame["detections"].is_array()) {
        LOG(WARNING) << "Frame found but no detections array in JSON: " << json_path;
        return result;
      }
      
      // Process each detection - extract all information
      for (const auto& detection : frame["detections"]) {
        // Get category_id
        if (!detection.contains("category_id") || !detection["category_id"].is_number()) {
          LOG(WARNING) << "Invalid category_id in detection, skipping";
          continue;
        }
        unsigned int category_id = detection["category_id"].get<unsigned int>();
        
        // Get detection_score
        double score = 0.0;
        if (detection.contains("detection_score") && detection["detection_score"].is_number()) {
          score = detection["detection_score"].get<double>();
        } else if (detection.contains("score") && detection["score"].is_number()) {
          score = detection["score"].get<double>();
        }
        
        // Get bbox: [x_min, y_min, x_max, y_max]
        if (!detection.contains("bbox") || !detection["bbox"].is_array() ||
            detection["bbox"].size() != 4) {
          LOG(WARNING) << "Invalid bbox in detection, skipping";
          continue;
        }
        
        double x_min = detection["bbox"][0].get<double>();
        double y_min = detection["bbox"][1].get<double>();
        double x_max = detection["bbox"][2].get<double>();
        double y_max = detection["bbox"][3].get<double>();
        BBox2 bbox(x_min, y_min, x_max, y_max);
        
        // Get ellipse (optional)
        Ellipse ellipse;
        if (detection.contains("ellipse") && detection["ellipse"].is_array() &&
            detection["ellipse"].size() == 5) {
          // ellipse format: [cx, cy, width, height, angle]
          double cx = detection["ellipse"][0].get<double>();
          double cy = detection["ellipse"][1].get<double>();
          double width = detection["ellipse"][2].get<double>();
          double height = detection["ellipse"][3].get<double>();
          double angle = detection["ellipse"][4].get<double>();
          
          ellipse = Ellipse(Eigen::Vector2d(0.5 * width, 0.5 * height), 
                           angle, 
                           Eigen::Vector2d(cx, cy));
        } else {
          // Create default ellipse from bbox if not provided
          ellipse = Ellipse::FromBbox(bbox, 0.0);
        }
        
        // Create Detection object
        result.detections.emplace_back(category_id, score, bbox, ellipse);
      }
      
      break;  // Found matching frame, exit loop
    }
  }
  
  if (!found_frame) {
    LOG(WARNING) << "No matching frame found for image: " << image_filename
                 << " in JSON file: " << json_path;
    return result;
  }
  
  // Create labelled_mask from detections (optional, can be empty if not needed)
  // For now, we leave it empty as it's not always needed for static objects
  
  return result;
}

std::vector<std::filesystem::path> getAllFilesInDir(
    const std::string& folder_path) {
  std::vector<std::filesystem::path> files_in_directory;
  std::copy(std::filesystem::directory_iterator(folder_path),
            std::filesystem::directory_iterator(),
            std::back_inserter(files_in_directory));
  std::sort(files_in_directory.begin(), files_in_directory.end());
  return files_in_directory;
}

void loadPathsInDirectory(
    std::vector<std::string>& file_paths, const std::string& folder_path,
    const std::function<bool(const std::string&)>& condition) {
  std::function<bool(const std::string&)> impl_condition;
  if (condition) {
    impl_condition = condition;
  } else {
    // if no condition is provided, set condition to always return true; adding
    // all the files found
    impl_condition = [](const std::string&) -> bool { return true; };
  }

  auto files_in_directory = getAllFilesInDir(folder_path);
  for (const std::string file_path : files_in_directory) {
    throwExceptionIfPathInvalid(file_path);

    // if condition is true, add
    if (impl_condition(file_path)) file_paths.push_back(file_path);
  }
}

std::vector<std::string> trimAndSplit(const std::string& input,
                                      const std::string& delimiter) {
  std::string trim_input = boost::algorithm::trim_right_copy(input);
  std::vector<std::string> split_line;
  boost::algorithm::split(split_line, trim_input, boost::is_any_of(delimiter));
  return split_line;
}

bool getLine(std::ifstream& fstream, std::vector<std::string>& split_lines) {
  std::string line;
  getline(fstream, line);

  split_lines.clear();

  if (line.empty()) return false;

  split_lines = trimAndSplit(line);
  return true;
}

}  // namespace utils
}  // namespace dyno
