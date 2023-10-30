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

#include <dynosam/common/Types.hpp>
#include <gtsam/geometry/Pose3.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"

#include "nav_msgs/msg/odometry.hpp"




template <>
bool dyno::convert(const dyno::Timestamp& time_seconds, rclcpp::Time& time)
{
  uint64_t nanoseconds = static_cast<uint64_t>(time_seconds * 1e9);
  time = rclcpp::Time(nanoseconds);
  return true;
}

template <>
bool dyno::convert(const rclcpp::Time& time, dyno::Timestamp& time_seconds)
{
  uint64_t nanoseconds = time.nanoseconds();
  time_seconds = static_cast<double>(nanoseconds) / 1e9;
  return true;
}

template <>
bool dyno::convert(const dyno::Timestamp& time_seconds, builtin_interfaces::msg::Time& time)
{
  rclcpp::Time ros_time;
  convert(time_seconds, ros_time);
  time = ros_time;
  return true;
}


//will not do time or tf links
template <>
bool dyno::convert(const gtsam::Pose3& pose, nav_msgs::msg::Odometry& odom)
{
  const gtsam::Rot3& rotation = pose.rotation();
  const gtsam::Quaternion& quaternion = rotation.toQuaternion();

  // Position
  odom.pose.pose.position.x = pose.x();
  odom.pose.pose.position.y = pose.y();
  odom.pose.pose.position.z = pose.z();

  // Orientation
  odom.pose.pose.orientation.w = quaternion.w();
  odom.pose.pose.orientation.x = quaternion.x();
  odom.pose.pose.orientation.y = quaternion.y();
  odom.pose.pose.orientation.z = quaternion.z();
  return true;
}


namespace dyno
{

Timestamp fromRosTime(const rclcpp::Time& time)
{
  Timestamp timestamp;
  convert(time, timestamp);
  return timestamp;
}

rclcpp::Time toRosTime(Timestamp timestamp)
{
  rclcpp::Time time;
  convert(timestamp, time);
  return time;
}

}  // dyno
