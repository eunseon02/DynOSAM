/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gtsam/geometry/Pose3.h>

#include <dynosam/common/Types.hpp>
#include <dynosam/visualizer/Colour.hpp>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "std_msgs/msg/color_rgba.hpp"

template <>
bool dyno::convert(const dyno::Timestamp& time_seconds, rclcpp::Time& time) {
  uint64_t nanoseconds = static_cast<uint64_t>(time_seconds * 1e9);
  time = rclcpp::Time(nanoseconds);
  return true;
}

template <>
bool dyno::convert(const rclcpp::Time& time, dyno::Timestamp& time_seconds) {
  uint64_t nanoseconds = time.nanoseconds();
  time_seconds = static_cast<double>(nanoseconds) / 1e9;
  return true;
}

template <>
bool dyno::convert(const dyno::Timestamp& time_seconds,
                   builtin_interfaces::msg::Time& time) {
  rclcpp::Time ros_time;
  convert(time_seconds, ros_time);
  time = ros_time;
  return true;
}

template <>
bool dyno::convert(const RGBA<float>& colour, std_msgs::msg::ColorRGBA& msg) {
  msg.r = colour.r;
  msg.g = colour.g;
  msg.b = colour.b;
  msg.a = colour.a;
}

template <>
bool dyno::convert(const Color& colour, std_msgs::msg::ColorRGBA& msg) {
  convert(RGBA<float>(colour), msg);
}

template <>
bool dyno::convert(const gtsam::Pose3& pose, geometry_msgs::msg::Pose& msg) {
  const gtsam::Rot3& rotation = pose.rotation();
  const gtsam::Quaternion& quaternion = rotation.toQuaternion();

  // Position
  msg.position.x = pose.x();
  msg.position.y = pose.y();
  msg.position.z = pose.z();

  // Orientation
  msg.orientation.w = quaternion.w();
  msg.orientation.x = quaternion.x();
  msg.orientation.y = quaternion.y();
  msg.orientation.z = quaternion.z();
  return true;
}

template <>
bool dyno::convert(const geometry_msgs::msg::Pose& msg, gtsam::Pose3& pose) {
  gtsam::Point3 translation(msg.position.x, msg.position.y, msg.position.z);

  gtsam::Rot3 rotation(msg.orientation.w, msg.orientation.x, msg.orientation.y,
                       msg.orientation.z);

  pose = gtsam::Pose3(rotation, translation);
  return true;
}

template <>
bool dyno::convert(const gtsam::Pose3& pose,
                   geometry_msgs::msg::PoseStamped& msg) {
  return convert<gtsam::Pose3, geometry_msgs::msg::Pose>(pose, msg.pose);
}

// will not do time or tf links or covariance....
template <>
bool dyno::convert(const gtsam::Pose3& pose, nav_msgs::msg::Odometry& odom) {
  return convert<gtsam::Pose3, geometry_msgs::msg::Pose>(pose, odom.pose.pose);
}

template <>
bool dyno::convert(const geometry_msgs::msg::Pose& pose,
                   geometry_msgs::msg::Transform& transform) {
  transform.translation.x = pose.position.x;
  transform.translation.y = pose.position.y;
  transform.translation.z = pose.position.z;

  transform.rotation.x = pose.orientation.x;
  transform.rotation.y = pose.orientation.y;
  transform.rotation.z = pose.orientation.z;
  transform.rotation.w = pose.orientation.w;
}

template <>
bool dyno::convert(const gtsam::Pose3& pose,
                   geometry_msgs::msg::Transform& transform) {
  transform.translation.x = pose.x();
  transform.translation.y = pose.y();
  transform.translation.z = pose.z();

  const gtsam::Rot3& rotation = pose.rotation();
  const gtsam::Quaternion& quaternion = rotation.toQuaternion();
  transform.rotation.x = quaternion.x();
  transform.rotation.y = quaternion.y();
  transform.rotation.z = quaternion.z();
  transform.rotation.w = quaternion.w();
  return true;
}

template <>
bool dyno::convert(const gtsam::Pose3& pose,
                   geometry_msgs::msg::TransformStamped& transform) {
  return convert<gtsam::Pose3, geometry_msgs::msg::Transform>(
      pose, transform.transform);
}

namespace dyno {
namespace utils {

Timestamp fromRosTime(const rclcpp::Time& time) {
  Timestamp timestamp;
  convert(time, timestamp);
  return timestamp;
}

rclcpp::Time toRosTime(Timestamp timestamp) {
  rclcpp::Time time;
  convert(timestamp, time);
  return time;
}

}  // namespace utils
}  // namespace dyno
