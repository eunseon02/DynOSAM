/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <dynosam/dataprovider/DataProvider.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace dyno {

class DataProviderRos : public DataProvider {
 public:
  DataProviderRos(rclcpp::Node::SharedPtr node);
  virtual ~DataProviderRos() = default;

  const cv::Mat readRgbRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;
  const cv::Mat readDepthRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;
  const cv::Mat readFlowRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;
  const cv::Mat readMaskRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;

 protected:
  const cv_bridge::CvImageConstPtr readRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;

 protected:
  rclcpp::Node::SharedPtr node_;
};

}  // namespace dyno
