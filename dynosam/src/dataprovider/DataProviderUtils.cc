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

#include "dynosam/dataprovider/DataProviderUtils.hpp"

#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/common/ImageTypes.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"  //for getObjectLabels
#include "dynosam/utils/OpenCVUtils.hpp"

namespace dyno {

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
  img = utils::readOpticalFlow(image_path);
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

void removeStaticObjectFromMask(const cv::Mat& instance_mask,
                                cv::Mat& motion_mask,
                                const GroundTruthInputPacket& gt_packet) {
  try {
    ImageType::SemanticMask::validate(instance_mask);
  } catch (const InvalidImageTypeException& e) {
    throw DynosamException(
        "Input mask to removeStaticObjectFromMask as it"
        "did not meet the requirements of ImageType::SemanticMask. Error: " +
        std::string(e.what()));
  }

  ObjectIds object_ids = vision_tools::getObjectLabels(instance_mask);

  instance_mask.copyTo(motion_mask);

  const FrameId frame_id = gt_packet.frame_id_;

  // collect only moving labels
  std::vector<int> moving_labels;
  for (const ObjectPoseGT& object_pose_gt : gt_packet.object_poses_) {
    // LOG_IF(WARNING, !object_pose_gt.motion_info_)
    //     << "Object Pose GT (object " << object_pose_gt.object_id_ << ", frame
    //     " << object_pose_gt.frame_id_ << " does not have motion info set!
    //     Cannot determine of object is moving or not!";

    if (!object_pose_gt.motion_info_) {
      LOG(WARNING) << "Object Pose GT (object " << object_pose_gt.object_id_
                   << ", frame " << object_pose_gt.frame_id_
                   << " does not have motion info set! Cannot determine of "
                      "object is moving or not!";
      continue;
    }

    const auto& motion_info = *object_pose_gt.motion_info_;
    const ObjectId& object_id = object_pose_gt.object_id_;
    if (motion_info.is_moving_) {
      moving_labels.push_back(object_id);
    }

    // this is just a sanity (debug) check to ensure all the labels in the image
    // match up with the ones we have already collected in the ground truth
    // packet
    const auto& it = std::find(object_ids.begin(), object_ids.end(), object_id);
    CHECK(it != object_ids.end())
        << "Object id " << object_id
        << " appears in gt packet but"
           "is not in mask when loading motion mask for Virtual Kitti at frame "
        << frame_id;
  }

  // iterate over each object and if not moving remove
  for (int i = 0; i < motion_mask.rows; i++) {
    for (int j = 0; j < motion_mask.cols; j++) {
      int label = motion_mask.at<int>(i, j);
      if (label == 0) {
        continue;
      }

      // check if label is in moving labels. if not, make zero!
      auto it = std::find(moving_labels.begin(), moving_labels.end(), label);
      if (it == moving_labels.end()) {
        motion_mask.at<int>(i, j) = 0;
      }
    }
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

}  // namespace dyno
