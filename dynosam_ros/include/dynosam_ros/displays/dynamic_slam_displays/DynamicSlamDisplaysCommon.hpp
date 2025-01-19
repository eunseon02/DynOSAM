#pragma once

#include <dynosam/common/Types.hpp>

#include "dynamic_slam_interfaces/msg/object_odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace dyno {

using ObjectOdometry = dynamic_slam_interfaces::msg::ObjectOdometry;
using ObjectOdometryPub = rclcpp::Publisher<ObjectOdometry>;

//! Map of object ids to ObjectOdometry (for a single frame, no frame ids)
using ObjectOdometryMap = gtsam::FastMap<ObjectId, ObjectOdometry>;

/**
 * @brief Class for managing the publishing and conversion of ObjectOdometry
 * messages.
 *
 */
class DynamicSlamDisplayCommon {
 public:
  // child_frame_id for objects
  static std::string constructObjectFrameLink(ObjectId object_id);

  // this is technically wrong as we should have a motion at k and a pose at k-1
  // to get velocity...
  static ObjectOdometry constructObjectOdometry(const gtsam::Pose3& motion_k,
                                                const gtsam::Pose3& pose_k,
                                                ObjectId object_id,
                                                Timestamp timestamp_k,
                                                const std::string& frame_id);

  static ObjectOdometryMap constructObjectOdometries(
      const MotionEstimateMap& motions_k, const ObjectPoseMap& poses,
      FrameId frame_id_k, Timestamp timestamp_k, const std::string& frame_id);

  class Publisher {
   public:
    void publishObjectOdometry();
    void publishObjectTransforms();

    inline FrameId getFrameId() const { return frame_id_; }

   private:
    rclcpp::Node::SharedPtr node_;
    ObjectOdometryPub::SharedPtr object_odom_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    gtsam::FastMap<ObjectId, ObjectOdometry> object_odometries_;

    FrameId frame_id_;
    Timestamp timestamp_;

    Publisher(rclcpp::Node::SharedPtr node,
              ObjectOdometryPub::SharedPtr object_odom_publisher,
              std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster,
              const MotionEstimateMap& motions, const ObjectPoseMap& poses,
              const std::string& frame_id, FrameId frame_id_k,
              Timestamp timestamp);
  };

  Publisher addObjectInfo(const MotionEstimateMap& motions,
                          const ObjectPoseMap& poses,
                          const std::string& frame_id, FrameId frame_id_k,
                          Timestamp timestamp);

 private:
  rclcpp::Node::SharedPtr node_;
  ObjectOdometryPub::SharedPtr object_odom_publisher_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace dyno

// class DynamicSlamInterfacesInbuiltDisplayCommon : public
// _InbuiltDisplayCommon {

// public:
//     DynamicSlamInterfacesInbuiltDisplayCommon(const DisplayParams& params,
//     rclcpp::Node::SharedPtr node); virtual
//     ~DynamicSlamInterfacesInbuiltDisplayCommon() = default;

//     using ObjectOdometry = dynamic_slam_interfaces::msg::ObjectOdometry;
//     using ObjectOdometryPub = rclcpp::Publisher<ObjectOdometry>;

//     void publishObjects(const MotionEstimateMap& motions, const
//     ObjectPoseMap& poses, FrameId frame_id, Timestamp timestamp) override;

// private:
//     ObjectOdometryPub::SharedPtr object_odometry_pub_;
// };

// #endif
