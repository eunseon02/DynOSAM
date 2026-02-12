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

#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <optional>
#include <memory>
#include <Eigen/Dense>

#include "dynosam/pipeline/ThreadSafeQueue.hpp"

namespace dyno {

struct ImageToDisplay {
  ImageToDisplay() = default;
  ImageToDisplay(const std::string& name, const cv::Mat& image)
      // clone necessary?
      : name_(name), image_(image.clone()) {}

  std::string name_;
  cv::Mat image_;
};

using ImageDisplayQueue = ThreadsafeQueue<ImageToDisplay>;

/**
 * @brief Snapshot for edge-based local map visualization (Option A)
 *
 * This message is published (typically every frame) by the frontend and consumed by the
 * visualization thread. Heavy data (point clouds, sliding window, environment) is passed
 * by shared_ptr so we can publish frequently without copying large buffers.
 *
 * The visualization thread can render directly from this snapshot:
 * - currentFramePose: always valid
 * - the shared_ptr fields may be null until the corresponding data becomes available
 */
struct EdgeVisualizationData {
  // Current camera pose in world (used for trajectory + camera pose)
  Eigen::Matrix4d currentFramePose{Eigen::Matrix4d::Identity()};

  // Individual edge clusters (raw covisibility point cloud)
  std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> clusterClouds;
  std::shared_ptr<const std::vector<cv::Vec3b>> clusterCloudColors;

  // Merged edge clusters (optimized local map)
  std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> localMapClouds;

  // Keyframe poses in the sliding window
  std::shared_ptr<const std::vector<Eigen::Matrix4d>> slidingWindow;

  // Accumulated environment point clouds over time
  std::shared_ptr<const std::vector<std::vector<cv::Point3d>>> environment_cloud;
};

// Pointer type for EdgeVisualizationData
using EdgeVisualizationDataPtr = std::shared_ptr<EdgeVisualizationData>;

class OpenCVImageDisplayQueue {
 public:
  OpenCVImageDisplayQueue(ImageDisplayQueue* display_queue, bool parallel_run);

  void process();

 private:
  ImageDisplayQueue* display_queue_;
  bool parallel_run_;
};

}  // namespace dyno
