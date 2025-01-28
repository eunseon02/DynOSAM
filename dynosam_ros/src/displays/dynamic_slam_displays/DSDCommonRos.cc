#include "dynosam_ros/displays/dynamic_slam_displays/DSDCommonRos.hpp"

#include <glog/logging.h>

#include <dynosam/common/DynamicObjects.hpp>
#include <dynosam/visualizer/Colour.hpp>

#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"

namespace dyno {

DSDTransport::DSDTransport(rclcpp::Node::SharedPtr node) : node_(node) {
  object_odom_publisher_ =
      node->create_publisher<ObjectOdometry>("object_odometry", 1);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);

  VLOG(20) << "Constructed DSDTransport with effective namespace "
           << node_->get_effective_namespace();
}

std::string DSDTransport::constructObjectFrameLink(ObjectId object_id) {
  return "object_" + std::to_string(object_id) + "_link";
}

ObjectOdometry DSDTransport::constructObjectOdometry(
    const gtsam::Pose3& motion_k, const gtsam::Pose3& pose_k,
    ObjectId object_id, Timestamp timestamp_k, const std::string& frame_id_link,
    const std::string& child_frame_id_link) {
  ObjectOdometry object_odom;

  // technically this shoudl be k-1
  gtsam::Point3 body_velocity = calculateBodyMotion(motion_k, pose_k);

  nav_msgs::msg::Odometry odom_msg;
  utils::convertWithHeader(pose_k, odom_msg, timestamp_k, frame_id_link,
                           child_frame_id_link);

  std_msgs::msg::ColorRGBA colour_msg;
  convert(Color::uniqueId(object_id), colour_msg);

  object_odom.odom = odom_msg;
  // NO velocity!!
  object_odom.object_id = object_id;
  object_odom.colour = colour_msg;

  return object_odom;
}

ObjectOdometryMap DSDTransport::constructObjectOdometries(
    const MotionEstimateMap& motions_k, const ObjectPoseMap& poses,
    FrameId frame_id_k, Timestamp timestamp_k,
    const std::string& frame_id_link) {
  // need to get poses for k-1
  // TODO: no way to ensure that the motions are for frame k
  // this is a weird data-structure to use and motions are per frame and
  // ObjectPoseMap is over all k to K
  //  const FrameId frame_id_k_1 = frame_id_k - 1u;
  ObjectOdometryMap object_odom_map;
  for (const auto& [object_id, object_motion] : motions_k) {
    const gtsam::Pose3& motion_k = object_motion;

    if (!poses.exists(object_id, frame_id_k)) {
      LOG(WARNING) << "Cannot construct ObjectOdometry for object " << object_id
                   << ", at frame " << frame_id_k
                   << " Missing entry in ObjectPoseMap";
      continue;
    }

    const gtsam::Pose3& pose_k = poses.at(object_id, frame_id_k);

    const std::string child_frame_id_link = constructObjectFrameLink(object_id);

    object_odom_map.insert2(
        child_frame_id_link,
        constructObjectOdometry(motion_k, pose_k, object_id, timestamp_k,
                                frame_id_link, child_frame_id_link));
  }

  return object_odom_map;
}

void DSDTransport::Publisher::publishObjectOdometry() {
  for (const auto& [_, object_odom] : object_odometries_)
    object_odom_publisher_->publish(object_odom);
}

void DSDTransport::Publisher::publishObjectTransforms() {
  for (const auto& [object_child_frame, object_odom] : object_odometries_) {
    geometry_msgs::msg::TransformStamped t;
    dyno::convert(object_odom.odom.pose.pose, t.transform);

    t.header.stamp = node_->now();
    t.header.frame_id = frame_id_link_;
    t.child_frame_id = object_child_frame;

    tf_broadcaster_->sendTransform(t);
  }
}

DSDTransport::Publisher::Publisher(
    rclcpp::Node::SharedPtr node,
    ObjectOdometryPub::SharedPtr object_odom_publisher,
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster,
    const MotionEstimateMap& motions, const ObjectPoseMap& poses,
    const std::string& frame_id_link, FrameId frame_id, Timestamp timestamp)
    : node_(node),
      object_odom_publisher_(object_odom_publisher),
      tf_broadcaster_(tf_broadcaster),
      frame_id_link_(frame_id_link),
      frame_id_(frame_id),
      timestamp_(timestamp),
      object_odometries_(DSDTransport::constructObjectOdometries(
          motions, poses, frame_id, timestamp, frame_id_link)) {}

DSDTransport::Publisher DSDTransport::addObjectInfo(
    const MotionEstimateMap& motions_k, const ObjectPoseMap& poses,
    const std::string& frame_id_link, FrameId frame_id, Timestamp timestamp) {
  return Publisher(node_, object_odom_publisher_, tf_broadcaster_, motions_k,
                   poses, frame_id_link, frame_id, timestamp);
}

DSDRos::DSDRos(const DisplayParams& params, rclcpp::Node::SharedPtr node)
    : params_(params), node_(node), dsd_transport_(node) {
  vo_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>("odometry", 1);
  vo_path_publisher_ =
      node_->create_publisher<nav_msgs::msg::Path>("odometry_path", 1);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);

  static_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("static_cloud", 1);
  dynamic_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_cloud", 1);
}

void DSDRos::publishVisualOdometry(const gtsam::Pose3& T_world_camera,
                                   Timestamp timestamp, const bool publish_tf) {
  DisplayCommon::publishOdometry(vo_publisher_, T_world_camera, timestamp,
                                 params_.world_frame_id,
                                 params_.camera_frame_id);

  if (publish_tf) {
    geometry_msgs::msg::TransformStamped t;
    dyno::convert<gtsam::Pose3, geometry_msgs::msg::TransformStamped>(
        T_world_camera, t);

    t.header.stamp = node_->now();
    t.header.frame_id = params_.world_frame_id;
    t.child_frame_id = params_.camera_frame_id;
    tf_broadcaster_->sendTransform(t);
  }
}
void DSDRos::publishVisualOdometryPath(const gtsam::Pose3Vector& poses,
                                       Timestamp latest_timestamp) {
  DisplayCommon::publishOdometryPath(vo_path_publisher_, poses,
                                     latest_timestamp, params_.world_frame_id);
}

CloudPerObject DSDRos::publishStaticPointCloud(
    const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera) {
  return DisplayCommon::publishPointCloud(
      static_points_pub_, landmarks, T_world_camera, params_.world_frame_id);
}
CloudPerObject DSDRos::publishDynamicPointCloud(
    const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera) {
  return DisplayCommon::publishPointCloud(
      dynamic_points_pub_, landmarks, T_world_camera, params_.world_frame_id);
}

}  // namespace dyno
