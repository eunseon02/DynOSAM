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

#include "dynosam_ros/DataProviderRos.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/exact_time.h"
#include "message_filters/synchronizer.h"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace dyno {

class OnlineDataProviderRos : public DataProviderRos {
 public:
  OnlineDataProviderRos(rclcpp::Node::SharedPtr node);

 private:
  using SyncPolicy = message_filters::sync_policies::ExactTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image,
      sensor_msgs::msg::Image>;

  void imageSyncCallback(
      const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &flow_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &mask_msg);

 private:
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> flow_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> mask_image_sub_;

  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

}  // namespace dyno
