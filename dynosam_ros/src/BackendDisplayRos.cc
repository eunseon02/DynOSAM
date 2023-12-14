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

#include "dynosam_ros/RosUtils.hpp"

#include <glog/logging.h>
#include "rclcpp/qos.hpp"

#include <dynosam/visualizer/ColourMap.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace dyno {

BackendDisplayRos::BackendDisplayRos(rclcpp::Node::SharedPtr node) {
    const rclcpp::QoS& sensor_data_qos = rclcpp::SensorDataQoS();
    static_tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("~/backend/static", 1);
    dynamic_tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("~/backend/dynamic", 1);
    odometry_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("~/backend/odom", 1);
    object_pose_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("~/backend/composed_object_poses", 1);

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

    {
        visualization_msgs::msg::MarkerArray object_pose_marker_array;
        // static visualization_msgs::msg::Marker delete_marker;
        // delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

        // object_pose_marker_array.markers.push_back(delete_marker);

        for(const auto&[object_id, poses] : backend_output->object_poses_composed_) {
            //object centroid per frame
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world";
            marker.ns = "backend_composed_object_positions";
            marker.id = object_id;
            marker.type = visualization_msgs::msg::Marker::POINTS;
            marker.action = visualization_msgs::msg::Marker::ADD;
            // marker.header.stamp = node_->now();

            const cv::Scalar colour = ColourMap::getObjectColour(object_id);
            std_msgs::msg::ColorRGBA marker_colour;
            marker_colour.r = colour(0)/255.0;
            marker_colour.g = colour(1)/255.0;
            marker_colour.b = colour(2)/255.0;
            marker_colour.a = 1;


            marker.scale.x = 0.3;
            marker.scale.y = 0.3;
            marker.scale.z = 0.3;

            marker.pose.orientation.x = 0;
            marker.pose.orientation.y = 0;
            marker.pose.orientation.z = 0;
            marker.pose.orientation.w = 1;

            for(const gtsam::Pose3& object_pose : poses) {

                LOG(INFO) << object_pose;
                geometry_msgs::msg::Point p;
                p.x = object_pose.x();
                p.y = object_pose.y();
                p.z = object_pose.z();

                marker.points.push_back(p);
                marker.colors.push_back(marker_colour);
            }
            // marker.header.stamp = node_->now();
            // marker.pose.position.x = pose.x();
            // marker.pose.position.y = pose.y();
            // marker.pose.position.z = pose.z();
            // marker.pose.orientation.x = pose.rotation().toQuaternion().x();
            // marker.pose.orientation.y = pose.rotation().toQuaternion().y();
            // marker.pose.orientation.z = pose.rotation().toQuaternion().z();
            // marker.pose.orientation.w = pose.rotation().toQuaternion().w();
            // marker.scale.x = 1;
            // marker.scale.y = 1;
            // marker.scale.z = 1;
            // marker.color.a = 1.0; // Don't forget to set the alpha!

            // const cv::Scalar colour = ColourMap::getObjectColour(object_id);
            // marker.color.r = colour(0)/255.0;
            // marker.color.g = colour(1)/255.0;
            // marker.color.b = colour(2)/255.0;

            object_pose_marker_array.markers.push_back(marker);

        }

        object_pose_pub_->publish(object_pose_marker_array);
    }

    {
        nav_msgs::msg::Odometry odom_msg;
        utils::convertWithHeader(backend_output->T_world_camera_, odom_msg, backend_output->timestamp_, "world", "camera");
        odometry_pub_->publish(odom_msg);
    }
}

} //dyno
