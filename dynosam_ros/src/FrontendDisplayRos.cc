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
    object_pose_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("composed_object_poses", 2);
    object_pose_path_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("composed_object_paths", 2);
    odometry_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("odom_path", 2);

    gt_odometry_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("~/ground_truth/odom", 2);
    gt_object_pose_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("~/ground_truth/object_poses", 2);
    gt_odom_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("~/ground_truth/odom_path", 2);
    gt_bounding_box_pub_= image_transport::create_publisher(node.get(), "~/ground_truth/bounding_boxes");


}

void FrontendDisplayRos::spinOnce(const FrontendOutputPacketBase::ConstPtr& frontend_output) {
    auto rgbd_output = safeCast<FrontendOutputPacketBase, RGBDInstanceOutputPacket>(frontend_output);
    if(rgbd_output) {
        processRGBDOutputpacket(rgbd_output);
    }

    RCLCPP_ERROR_STREAM(node_->get_logger(), "AHGJKDHSJKDFHDS");
    // publishOdometry(frontend_output->T_world_camera_, frontend_output->getTimestamp());
    // publishOdometryPath(frontend_output->T_world_camera_, frontend_output->getTimestamp());
    // publishDebugImage(frontend_output->debug_image_);

    // if(frontend_output->gt_packet_) {
    //     const auto& rgb_image = frontend_output->frame_.tracking_images_.get<ImageType::RGBMono>();
    //     publishGroundTruthInfo(frontend_output->getTimestamp(), frontend_output->gt_packet_.value(), rgb_image);
    // }
}



void FrontendDisplayRos::processRGBDOutputpacket(const RGBDInstanceOutputPacket::ConstPtr& rgbd_frontend_output) {
    CHECK(rgbd_frontend_output);
    publishStaticCloud(rgbd_frontend_output->static_landmarks_);
    publishObjectCloud(rgbd_frontend_output->dynamic_keypoint_measurements_, rgbd_frontend_output->dynamic_landmarks_);
    publishObjectPositions(rgbd_frontend_output->propogated_object_poses_, rgbd_frontend_output->getFrameId());
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
        pcl::PointXYZRGB pt(lmk(0), lmk(1), lmk(2), colour(0), colour(1), colour(2));
        cloud.points.push_back(pt);
    }

    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(cloud, pc2_msg);
    pc2_msg.header.frame_id = "world";
    dynamic_tracked_points_pub_->publish(pc2_msg);
}


void FrontendDisplayRos::publishObjectPositions(const std::map<ObjectId, gtsam::Pose3>& propogated_object_poses, FrameId frame_id) {
    visualization_msgs::msg::MarkerArray object_pose_marker_array;
    visualization_msgs::msg::MarkerArray object_path_marker_array;

    static visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    object_pose_marker_array.markers.push_back(delete_marker);
    object_path_marker_array.markers.push_back(delete_marker);

    for(const auto&[object_id, pose] : propogated_object_poses) {

        {
            //object centroid per frame
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world";
            marker.ns = "frontend_composed_object_positions";
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

            object_pose_marker_array.markers.push_back(marker);
        }

        {
            auto it = object_trajectories_.find(object_id);
            if(it == object_trajectories_.end()) {
                object_trajectories_[object_id] = gtsam::Pose3Vector();
                object_trajectories_update_[object_id] = frame_id;
            }

            object_trajectories_.at(object_id).push_back(pose);
            object_trajectories_update_[object_id] = frame_id; //update last seen frame
        }
    }

    //iterate over object trajectories and display the ones with enough poses and the ones weve seen recently
    for(const auto& [object_id, poses] : object_trajectories_) {
        const FrameId last_seen_frame = object_trajectories_update_.at(last_seen_frame);

        //if weve seen the object in the last 5 frames and the length is at least 2
        if(frame_id - last_seen_frame > 5u || poses.size() < 2u) {
            continue;
        }

        //draw a line list for viz
        //have to duplicate the first in each drawn pair so that we construct a complete line
        for(size_t i = 1; i < poses.size(); i++) {
            const gtsam::Pose3& prev_pose = poses.at(i-1);
            const gtsam::Pose3& curr_pose = poses.at(i);

            {
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "world";
                marker.ns = "frontend_composed_object_path";
                marker.id = object_id;
                marker.type = visualization_msgs::msg::Marker::LINE_LIST;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.header.stamp = node_->now();
                marker.pose.position.x = prev_pose.x();
                marker.pose.position.y = prev_pose.y();
                marker.pose.position.z = prev_pose.z();
                marker.pose.orientation.x = prev_pose.rotation().toQuaternion().x();
                marker.pose.orientation.y = prev_pose.rotation().toQuaternion().y();
                marker.pose.orientation.z = prev_pose.rotation().toQuaternion().z();
                marker.pose.orientation.w = prev_pose.rotation().toQuaternion().w();
                marker.scale.x = 0.1;
                // marker.scale.y = 1;
                // marker.scale.z = 1;
                marker.color.a = 1.0; // Don't forget to set the alpha!

                const cv::Scalar colour = ColourMap::getObjectColour(object_id);
                marker.color.r = colour(0)/255.0;
                marker.color.g = colour(1)/255.0;
                marker.color.b = colour(2)/255.0;

                object_path_marker_array.markers.push_back(marker);

            }

            {
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = "world";
                marker.ns = "frontend_composed_object_path";
                marker.id = object_id;
                marker.type = visualization_msgs::msg::Marker::LINE_LIST;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.header.stamp = node_->now();
                marker.pose.position.x = curr_pose.x();
                marker.pose.position.y = curr_pose.y();
                marker.pose.position.z = curr_pose.z();
                marker.pose.orientation.x = curr_pose.rotation().toQuaternion().x();
                marker.pose.orientation.y = curr_pose.rotation().toQuaternion().y();
                marker.pose.orientation.z = curr_pose.rotation().toQuaternion().z();
                marker.pose.orientation.w = curr_pose.rotation().toQuaternion().w();
                marker.scale.x = 0.1;
                // marker.scale.y = 1;
                // marker.scale.z = 1;
                marker.color.a = 1.0; // Don't forget to set the alpha!

                const cv::Scalar colour = ColourMap::getObjectColour(object_id);
                marker.color.r = colour(0)/255.0;
                marker.color.g = colour(1)/255.0;
                marker.color.b = colour(2)/255.0;

                object_path_marker_array.markers.push_back(marker);

            }
        }
    }

    object_pose_pub_->publish(object_pose_marker_array);
    object_pose_path_pub_->publish(object_path_marker_array);

}



void FrontendDisplayRos::publishOdometry(const gtsam::Pose3& T_world_camera, Timestamp timestamp) {
    LOG(ERROR) << timestamp;
    nav_msgs::msg::Odometry odom_msg;
    utils::convertWithHeader(T_world_camera, odom_msg, timestamp, "world", "camera");
    odometry_pub_->publish(odom_msg);
}

void FrontendDisplayRos::publishOdometryPath(const gtsam::Pose3& T_world_camera, Timestamp timestamp) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    utils::convertWithHeader(T_world_camera, pose_stamped, timestamp, "world");

    static std_msgs::msg::Header header;
    // header.stamp = utils::toRosTime(timestamp);
    header.frame_id = "world";
    odom_path_msg_.header = header;

    odom_path_msg_.poses.push_back(pose_stamped);
    odometry_path_pub_->publish(odom_path_msg_);

}


void FrontendDisplayRos::publishDebugImage(const cv::Mat& debug_image) {
    if(debug_image.empty()) return;

    cv::Mat resized_image;
    cv::resize(debug_image, resized_image, cv::Size(640, 480));

    std_msgs::msg::Header hdr;
    sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(hdr, "bgr8", resized_image).toImageMsg();
    tracking_image_pub_.publish(msg);
}

void FrontendDisplayRos::publishGroundTruthInfo(Timestamp timestamp, const GroundTruthInputPacket& gt_packet, const cv::Mat& rgb) {
    //odometry gt
    const gtsam::Pose3& T_world_camera = gt_packet.X_world_;
    nav_msgs::msg::Odometry odom_msg;
    utils::convertWithHeader(T_world_camera, odom_msg, timestamp, "world", "camera");
    gt_odometry_pub_->publish(odom_msg);

    //odom path gt
    geometry_msgs::msg::PoseStamped pose_stamped;
    utils::convertWithHeader(T_world_camera, pose_stamped, timestamp, "world");
    static std_msgs::msg::Header header;
    // header.stamp = utils::toRosTime(timestamp);
    header.frame_id = "world";
    gt_odom_path_msg_.header = header;
    gt_odom_path_msg_.poses.push_back(pose_stamped);

    gt_odom_path_pub_->publish(gt_odom_path_msg_);

    //prepare display image
    cv::Mat disp_image;
    rgb.copyTo(disp_image);

    //prepare gt object pose markers
    visualization_msgs::msg::MarkerArray object_pose_marker_array;
    static visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    object_pose_marker_array.markers.push_back(delete_marker);

    for(const auto& object_pose_gt : gt_packet.object_poses_) {
        const gtsam::Pose3 L_world = T_world_camera * object_pose_gt.L_camera_;
        const ObjectId object_id = object_pose_gt.object_id_;

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.ns = "ground_truth_object_poses";
        marker.id = object_id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.header.stamp = node_->now();
        marker.pose.position.x = L_world.x();
        marker.pose.position.y = L_world.y();
        marker.pose.position.z = L_world.z();
        marker.pose.orientation.x = L_world.rotation().toQuaternion().x();
        marker.pose.orientation.y = L_world.rotation().toQuaternion().y();
        marker.pose.orientation.z = L_world.rotation().toQuaternion().z();
        marker.pose.orientation.w = L_world.rotation().toQuaternion().w();
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.a = 1.0; // Don't forget to set the alpha!

        const cv::Scalar colour = ColourMap::getObjectColour(object_id);
        marker.color.r = colour(0)/255.0;
        marker.color.g = colour(1)/255.0;
        marker.color.b = colour(2)/255.0;

        visualization_msgs::msg::Marker text_marker = marker;
        text_marker.ns = "ground_truth_object_labels";
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.text = std::to_string(object_id);
        text_marker.pose.position.z += 1.0; //make it higher than the pose marker
        text_marker.scale.z = 0.7;

        object_pose_marker_array.markers.push_back(marker);
        object_pose_marker_array.markers.push_back(text_marker);

        //draw on bbox
        object_pose_gt.drawBoundingBox(disp_image);
    }

    gt_object_pose_pub_->publish(object_pose_marker_array);

    cv::Mat resized_image;
    cv::resize(disp_image, resized_image, cv::Size(640, 480));

    std_msgs::msg::Header hdr;
    sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(hdr, "bgr8", resized_image).toImageMsg();
    gt_bounding_box_pub_.publish(msg);
}

}
