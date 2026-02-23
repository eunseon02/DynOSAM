#pragma once

#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>
#include "dynosam_common/StaticObjects.hpp"

namespace dyno {
namespace utils {

void throwExceptionIfPathInvalid(const std::string& image_path);

void loadRGB(const std::string& image_path, cv::Mat& img);

// CV_32F (float)
void loadFlow(const std::string& image_path, cv::Mat& img);

// CV_64F (double)
void loadDepth(const std::string& image_path, cv::Mat& img);

// CV_32SC1
// this is old kitti style and loads from a .txt file (which is why we need the
// size)
void loadSemanticMask(const std::string& image_path, const cv::Size& size,
                      cv::Mat& mask);

// CV_32SC1
void loadMask(const std::string& image_path, cv::Mat& mask);

/**
 * @brief Load instance mask from JSON detection file.
 * 
 * Reads detection information from a JSON file and creates an instance mask
 * where each detection's bounding box is filled with a unique object ID.
 * 
 * JSON format expected:
 * [
 *   {
 *     "file_name": "image_001.png",
 *     "detections": [
 *       {
 *         "bbox": [x_min, y_min, x_max, y_max],
 *         "category_id": 1,
 *         "detection_score": 0.95,
 *         ...
 *       },
 *       ...
 *     ]
 *   },
 *   ...
 * ]
 * 
 * @param json_path Path to JSON detection file
 * @param image_filename Filename to match (e.g., "image_001.png" or just "image_001")
 * @param image_size Size of the output mask
 * @param mask Output mask (CV_32SC1) where each detection is filled with its category_id
 * @return true if detection found and mask created, false otherwise
 */

static_objects::ObjectDetectionResult loadDetections(
    const std::string& json_path,
    const std::string& image_filename,
    const cv::Mat& input_image);

/**
 * @brief Returns a ORDERED vector of all files in the given directory.
 * (jesse) is this the file name or the absolute file path?
 *
 * @param folder_path
 * @return std::vector<std::filesystem::path>
 */
std::vector<std::filesystem::path> getAllFilesInDir(
    const std::string& folder_path);

void loadPathsInDirectory(
    std::vector<std::string>& file_paths, const std::string& folder_path,
    const std::function<bool(const std::string&)>& condition =
        std::function<bool(const std::string&)>());

/**
 * @brief Gets the next line from the input ifstream and returns it as split
 * string (using white space as the delimieter). Any newline/carriage
 * return/trailing white space values are trimmed.
 *
 * @param fstream std::ifstream&
 * @param split_lines std::vector<std::string>&
 * @return true
 * @return false
 */
bool getLine(std::ifstream& fstream, std::vector<std::string>& split_lines);

/**
 * @brief Takes an input string and splits it using white space (" ") as the
 * delimiter. Trims any newline/carriage return/trailing white space values.
 *
 * @param input const std::string&
 * @return std::vector<std::string>
 */
std::vector<std::string> trimAndSplit(const std::string& input,
                                      const std::string& delimiter = " ");

}  // namespace utils
}  // namespace dyno
