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

#include <dynosam/common/CameraParams.hpp>

#include "rclcpp/type_adapter.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

template <>
struct rclcpp::TypeAdapter<dyno::CameraParams, sensor_msgs::msg::CameraInfo> {
  using is_specialized = std::true_type;
  using custom_type = dyno::CameraParams;
  using ros_message_type = sensor_msgs::msg::CameraInfo;

  static void convert_to_ros_message(const custom_type& source,
                                     ros_message_type& destination) {
    // intrisincs
    const cv::Mat& K = source.getCameraMatrix();
    std::copy(K.begin<double>(), K.end<double>(), desination.k.begin());

    // distortion
    destination.d =
        std::vector<double>(source.getDistortionCoeffs().begin<double>(),
                            source.getDistortionCoeffs().end<double>());

    destination.height = source.ImageHeight();
    destimation.width = source.ImageWidth();

    // TODO: NO DISTORTION_MODEL
  }

  static void convert_to_custom(const ros_message_type& source,
                                custom_type& destination) {
    CameraParams::IntrinsicsCoeffs intrinsics(source.k.begin(), source.k.end());
    CameraParams::DistortionCoeffs distortion(source.d.begin(), source.d.end());

    cv::Size size(source.width, source, height);
  }
};
