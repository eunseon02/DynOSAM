#pragma once

#include <dynosam/common/Types.hpp>
#include <dynosam/common/PointCloudProcess.hpp> //for CloudPerObject
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "dynosam_ros/Display-Definitions.hpp"
#include "image_transport/image_transport.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <pcl_conversions/pcl_conversions.h>



namespace dyno {

using PointCloud2 = sensor_msgs::msg::PointCloud2;
using MarkerArray =
    visualization_msgs::msg::MarkerArray;  //! Typedef for MarkerArray msg

using PointCloud2Pub = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>;
using OdometryPub = rclcpp::Publisher<nav_msgs::msg::Odometry>;
using PathPub = rclcpp::Publisher<nav_msgs::msg::Path>;
using MarkerArrayPub = rclcpp::Publisher<MarkerArray>;


/**
 * @brief Common statelss (free) functions for all ROS displays
 * 
 */
struct DisplayCommon {
static CloudPerObject publishPointCloud(PointCloud2Pub::SharedPtr pub, const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera, const std::string& frame_id);
static void publishOdometry(OdometryPub::SharedPtr pub, const gtsam::Pose3& T_world_camera, Timestamp timestamp, const std::string& frame_id, const std::string& child_frame_id);
static void publishOdometryPath(PathPub::SharedPtr pub, const gtsam::Pose3Vector& poses, Timestamp latest_timestamp, const std::string& frame_id);
};




} //dyno