/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "dynosam_ros/FrontendDisplayRos.hpp"

#include <glog/logging.h>
#include "rclcpp/qos.hpp"

namespace dyno {

FrontendDisplayRos::FrontendDisplayRos(rclcpp::Node::SharedPtr node) : node_(CHECK_NOTNULL(node)) {

    // const rclcpp::QoS& sensor_data_qos = rclcpp::SensorDataQoS();
    tracking_image_pub_ = image_transport::create_publisher(node.get(), "~/tracking_image");
    tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("~/tracked_points", rclcpp::SensorDataQoS());
}

void FrontendDisplayRos::spinOnce(const FrontendOutputPacketBase& frontend_output) {

}

}
