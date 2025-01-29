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

#include "dynosam_ros/OnlineDataProviderRos.hpp"

namespace dyno {

OnlineDataProviderRos::OnlineDataProviderRos(rclcpp::Node::SharedPtr node)
    : DataProviderRos(node) {
  rclcpp::Node *node_ptr = node_.get();
  rgb_image_sub_.subscribe(node_ptr, "image/rgb");
  depth_image_sub_.subscribe(node_ptr, "image/depth");
  flow_image_sub_.subscribe(node_ptr, "image/flow");
  mask_image_sub_.subscribe(node_ptr, "image/mask");

  static constexpr size_t kQueueSize = 20u;
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(kQueueSize), rgb_image_sub_, depth_image_sub_, flow_image_sub_,
      mask_image_sub_);

  sync_->registerCallback(std::bind(
      &OnlineDataProviderRos::imageSyncCallback, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

  RCLCPP_INFO_STREAM(
      node_->get_logger(),
      "OnlineDataProviderRos has been started. Subscribed to image topics: "
          << rgb_image_sub_.getSubscriber()->get_topic_name() << " "
          << depth_image_sub_.getSubscriber()->get_topic_name() << " "
          << flow_image_sub_.getSubscriber()->get_topic_name() << " "
          << mask_image_sub_.getSubscriber()->get_topic_name() << ".");
}

void OnlineDataProviderRos::imageSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &flow_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &mask_msg) {}

}  // namespace dyno
