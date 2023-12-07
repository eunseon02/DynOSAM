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

#include "dynosam_ros/BackendDisplayRos.hpp"

#include <glog/logging.h>
#include "rclcpp/qos.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace dyno {

BackendDisplayRos::BackendDisplayRos(rclcpp::Node::SharedPtr node) {
    const rclcpp::QoS& sensor_data_qos = rclcpp::SensorDataQoS();
    static_tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("~/backend/static", 1);
    dynamic_tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("~/backend/dynamic", 1);

}


void BackendDisplayRos::spinOnce(const BackendOutputPacket::ConstPtr& backend_output) {

    {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        for(const auto& [tracklet_id, lmk] : backend_output->static_lmks_) {
            // publish static lmk's as white
            pcl::PointXYZRGB pt(lmk(0), lmk(1), lmk(2), 0, 0, 0);
            cloud.points.push_back(pt);
        }


        sensor_msgs::msg::PointCloud2 pc2_msg;
        pcl::toROSMsg(cloud, pc2_msg);
        pc2_msg.header.frame_id = "world";
        static_tracked_points_pub_->publish(pc2_msg);
    }

    {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        const size_t num_measurements = backend_output->dynamic_lmks_.size();

        for(size_t i = 0; i < num_measurements; i++) {
            const StatusLandmarkEstimate& sle = backend_output->dynamic_lmks_.at(i);
            const LandmarkStatus& status = sle.first;
            const LandmarkEstimate& le = sle.second;

            const Landmark lmk = le.second;

            const cv::Scalar colour = ColourMap::getObjectColour(status.label_);
            pcl::PointXYZRGB pt(lmk(0), lmk(1), lmk(2), colour(0), colour(1), colour(2));
            cloud.points.push_back(pt);
        }

        sensor_msgs::msg::PointCloud2 pc2_msg;
        pcl::toROSMsg(cloud, pc2_msg);
        pc2_msg.header.frame_id = "world";
        dynamic_tracked_points_pub_->publish(pc2_msg);
    }

}

} //dyno
