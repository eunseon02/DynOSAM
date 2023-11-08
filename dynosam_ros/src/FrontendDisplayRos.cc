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

#include <dynosam/visualizer/ColourMap.hpp>
#include <dynosam/frontend/RGBDInstance-Definitions.hpp>
#include <dynosam/utils/SafeCast.hpp>

#include "dynosam_ros/FrontendDisplayRos.hpp"
#include "dynosam_ros/RosUtils.hpp"

#include <glog/logging.h>
#include "rclcpp/qos.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>

namespace dyno {

FrontendDisplayRos::FrontendDisplayRos(rclcpp::Node::SharedPtr node) : node_(CHECK_NOTNULL(node)) {

    // const rclcpp::QoS& sensor_data_qos = rclcpp::SensorDataQoS();
    tracking_image_pub_ = image_transport::create_publisher(node.get(), "tracking_image");
    static_tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("static", 2);
    dynamic_tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic", 2);
    odometry_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("odom", 2);
    object_pose_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("object_pose", 2);
}

void FrontendDisplayRos::spinOnce(const FrontendOutputPacketBase::ConstPtr& frontend_output) {
    auto rgbd_output = safeCast<FrontendOutputPacketBase, RGBDInstanceOutputPacket>(frontend_output);
    if(rgbd_output) {
        processRGBDOutputpacket(rgbd_output);
    }

    publishOdometry(frontend_output->T_world_camera_);
    publishDebugImage(frontend_output->debug_image_);
}



void FrontendDisplayRos::processRGBDOutputpacket(const RGBDInstanceOutputPacket::ConstPtr& rgbd_frontend_output) {
    CHECK(rgbd_frontend_output);
    publishStaticCloud(rgbd_frontend_output->static_landmarks_);
    publishObjectCloud(rgbd_frontend_output->dynamic_keypoint_measurements_, rgbd_frontend_output->dynamic_landmarks_);
    publishObjectPositions(rgbd_frontend_output->propogated_object_poses_);
}

void FrontendDisplayRos::publishStaticCloud(const Landmarks& static_landmarks) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;

    for(const Landmark& lmk : static_landmarks) {
        pcl::PointXYZRGB pt(lmk(0), lmk(1), lmk(2), 0, 0, 0);
        cloud.points.push_back(pt);
    }


    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(cloud, pc2_msg);
    pc2_msg.header.frame_id = "world";
    static_tracked_points_pub_->publish(pc2_msg);
}

void FrontendDisplayRos::publishObjectCloud(const StatusKeypointMeasurements& dynamic_measurements, const Landmarks& dynamic_landmarks) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    CHECK_EQ(dynamic_measurements.size(), dynamic_landmarks.size());

    const size_t num_measurements = dynamic_measurements.size();

    for(size_t i = 0; i < num_measurements; i++) {
        const StatusKeypointMeasurement& kpm = dynamic_measurements.at(i);
        const KeypointStatus& status = kpm.first;
        const Landmark& lmk = dynamic_landmarks.at(i);

        const cv::Scalar colour = ColourMap::getObjectColour(status.label_);
        pcl::PointXYZRGB pt(lmk(0), lmk(1), lmk(2), colour(0), colour(1), colour(1));
        cloud.points.push_back(pt);
    }

    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(cloud, pc2_msg);
    pc2_msg.header.frame_id = "world";
    dynamic_tracked_points_pub_->publish(pc2_msg);
}


void FrontendDisplayRos::publishObjectPositions(const std::map<ObjectId, gtsam::Pose3>& propogated_object_poses) {
    visualization_msgs::msg::MarkerArray marker_array;

    static visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    for(const auto&[object_id, pose] : propogated_object_poses) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.ns = "frontend_object_positions";
        marker.id = object_id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.header.stamp = node_->now();
        marker.pose.position.x = pose.x();
        marker.pose.position.y = pose.y();
        marker.pose.position.z = pose.z();
        marker.pose.orientation.x = pose.rotation().toQuaternion().x();
        marker.pose.orientation.y = pose.rotation().toQuaternion().y();
        marker.pose.orientation.z = pose.rotation().toQuaternion().z();
        marker.pose.orientation.w = pose.rotation().toQuaternion().w();
        marker.scale.x = 1;
        marker.scale.y = 1;
        marker.scale.z = 1;
        marker.color.a = 1.0; // Don't forget to set the alpha!

        const cv::Scalar colour = ColourMap::getObjectColour(object_id);
        marker.color.r = colour(0)/255.0;
        marker.color.g = colour(1)/255.0;
        marker.color.b = colour(2)/255.0;

        marker_array.markers.push_back(marker);
    }

    object_pose_pub_->publish(marker_array);
}



void FrontendDisplayRos::publishOdometry(const gtsam::Pose3& T_world_camera) {
    nav_msgs::msg::Odometry odom_msg;
    convert(T_world_camera, odom_msg);
    odom_msg.header.frame_id = "world";
    odom_msg.child_frame_id = "camera";
    odometry_pub_->publish(odom_msg);

}


void FrontendDisplayRos::publishDebugImage(const cv::Mat& debug_image) {
    if(debug_image.empty()) return;

    cv::Mat resized_image;
    cv::resize(debug_image, resized_image, cv::Size(640, 480));

    std_msgs::msg::Header hdr;
    sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(hdr, "bgr8", resized_image).toImageMsg();
    tracking_image_pub_.publish(msg);


}

}
