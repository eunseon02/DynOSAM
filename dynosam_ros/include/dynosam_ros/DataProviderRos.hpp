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

#include <dynosam/common/ImageTypes.hpp>
#include <dynosam/dataprovider/DataProvider.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "rclcpp/wait_set.hpp"
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

  template <class Rep = int64_t, class Period = std::milli>
  const CameraParams& waitAndSetCameraParams(
      const std::string& topic = "image/camera_info",
      const std::chrono::duration<Rep, Period>& time_to_wait =
          std::chrono::duration<Rep, Period>(-1)) {
    // rclcpp::wait_for_message<
  }

  CameraParams::Optional getCameraParams() const override;

 protected:
  const cv_bridge::CvImageConstPtr readRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const;

  template <typename IMAGETYPE>
  const cv::Mat convertRosImage(
      const sensor_msgs::msg::Image::ConstSharedPtr& img_msg) const {
    const cv_bridge::CvImageConstPtr cvb_image = readRosImage(img_msg);

    try {
      const cv::Mat img = cvb_image->image;
      image_traits<IMAGETYPE>::validate(img);
      return img;

    } catch (const InvalidImageTypeException& exception) {
      RCLCPP_FATAL_STREAM(node_->get_logger(),
                          image_traits<IMAGETYPE>::name()
                              << " Image msg was of the wrong type (validate "
                                 "failed with exception "
                              << exception.what() << "). "
                              << "ROS encoding type used was "
                              << cvb_image->encoding);
      rclcpp::shutdown();
    }
  }

 protected:
  rclcpp::Node::SharedPtr node_;
  CameraParams::Optional camera_params_;
};

}  // namespace dyno
