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

#include <dynosam/common/Types.hpp>

#include "dynamic_slam_interfaces/msg/object_odometry.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace dyno {

using ObjectOdometry = dynamic_slam_interfaces::msg::ObjectOdometry;
using ObjectOdometryPub = rclcpp::Publisher<ObjectOdometry>;

//! Map of object id link (child frame id) to ObjectOdometry (for a single
//! frame, no frame ids)
using ObjectOdometryMap = gtsam::FastMap<std::string, ObjectOdometry>;

/**
 * @brief Class for managing the publishing and conversion of ObjectOdometry
 * messages.
 * DSD is shorthand for Dynamic SLAM Display.
 *
 */
class DSDTransport {
 public:
  DSDTransport(rclcpp::Node::SharedPtr node);

  // child_frame_id for objects
  static std::string constructObjectFrameLink(ObjectId object_id);

  // this is technically wrong as we should have a motion at k and a pose at k-1
  // to get velocity...
  static ObjectOdometry constructObjectOdometry(
      const gtsam::Pose3& motion_k, const gtsam::Pose3& pose_k,
      ObjectId object_id, Timestamp timestamp_k,
      const std::string& frame_id_link, const std::string& child_frame_id_link);

  static ObjectOdometryMap constructObjectOdometries(
      const MotionEstimateMap& motions_k, const ObjectPoseMap& poses,
      FrameId frame_id_k, Timestamp timestamp_k,
      const std::string& frame_id_link);

  class Publisher {
   public:
    void publishObjectOdometry();
    void publishObjectTransforms();

    inline FrameId getFrameId() const { return frame_id_; }
    inline Timestamp getTimestamp() const { return timestamp_; }

   private:
    rclcpp::Node::SharedPtr node_;
    ObjectOdometryPub::SharedPtr object_odom_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::string frame_id_link_;
    FrameId frame_id_;
    Timestamp timestamp_;

    ObjectOdometryMap object_odometries_;

    friend class DSDTransport;
    Publisher(rclcpp::Node::SharedPtr node,
              ObjectOdometryPub::SharedPtr object_odom_publisher,
              std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster,
              const MotionEstimateMap& motions, const ObjectPoseMap& poses,
              const std::string& frame_id_link, FrameId frame_id,
              Timestamp timestamp);
  };

  Publisher addObjectInfo(const MotionEstimateMap& motions_k,
                          const ObjectPoseMap& poses,
                          const std::string& frame_id_link, FrameId frame_id,
                          Timestamp timestamp);

 private:
  rclcpp::Node::SharedPtr node_;
  ObjectOdometryPub::SharedPtr object_odom_publisher_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

/**
 * @brief Shorthand for Dynamic Slam Display ROS
 *
 */
class DSDRos {
 public:
  DSDRos(const DisplayParams& params, rclcpp::Node::SharedPtr node);

  void publishVisualOdometry(const gtsam::Pose3& T_world_camera,
                             Timestamp timestamp, const bool publish_tf);
  void publishVisualOdometryPath(const gtsam::Pose3Vector& poses,
                                 Timestamp latest_timestamp);

  CloudPerObject publishStaticPointCloud(const StatusLandmarkVector& landmarks,
                                         const gtsam::Pose3& T_world_camera);

  // struct PubDynamicCloudOptions {
  //   //TODO: unused
  //   bool publish_object_bounding_box{true};

  //   // PubDynamicCloudOptions() = default;
  //   ~PubDynamicCloudOptions() = default;
  // };

  CloudPerObject publishDynamicPointCloud(const StatusLandmarkVector& landmarks,
                                          const gtsam::Pose3& T_world_camera);

 private:
 protected:
  const DisplayParams params_;
  rclcpp::Node::SharedPtr node_;
  //! Dynamic SLAM display transport for estimated object odometry
  DSDTransport dsd_transport_;
  //! TF broadcaster for the odometry.
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  OdometryPub::SharedPtr vo_publisher_;
  PathPub::SharedPtr vo_path_publisher_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      static_points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      dynamic_points_pub_;
};

}  // namespace dyno
