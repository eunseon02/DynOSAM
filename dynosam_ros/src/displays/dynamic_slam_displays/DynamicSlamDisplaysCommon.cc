#include "dynosam_ros/displays/dynamic_slam_displays/DynamicSlamDisplaysCommon.hpp"

#include <glog/logging.h>

#include <dynosam/common/DynamicObjects.hpp>
#include <dynosam/visualizer/Colour.hpp>

#include "dynosam_ros/RosUtils.hpp"

namespace dyno {

std::string DynamicSlamDisplayCommon::constructObjectFrameLink(
    ObjectId object_id) {
  return "object_" + std::to_string(object_id) + "_link";
}

ObjectOdometry DynamicSlamDisplayCommon::constructObjectOdometry(
    const gtsam::Pose3& motion_k, const gtsam::Pose3& pose_k,
    ObjectId object_id, Timestamp timestamp_k, const std::string& frame_id) {
  ObjectOdometry object_odom;

  // technically this shoudl be k-1
  gtsam::Point3 body_velocity = calculateBodyMotion(motion_k, pose_k);

  const std::string child_frame_id = constructObjectFrameLink(object_id);

  nav_msgs::msg::Odometry odom_msg;
  utils::convertWithHeader(pose_k, odom_msg, timestamp_k, frame_id,
                           child_frame_id);

  std_msgs::msg::ColorRGBA colour_msg;
  convert(Color::uniqueId(object_id), colour_msg);

  object_odom.odom = odom_msg;
  // NO velocity!!
  object_odom.object_id = object_id;
  object_odom.colour = colour_msg;

  return object_odom;
}

ObjectOdometryMap DynamicSlamDisplayCommon::constructObjectOdometries(
    const MotionEstimateMap& motions_k, const ObjectPoseMap& poses,
    FrameId frame_id_k, Timestamp timestamp_k, const std::string& frame_id) {
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

    object_odom_map.insert2(object_id,
                            constructObjectOdometry(motion_k, pose_k, object_id,
                                                    timestamp_k, frame_id));
  }

  return object_odom_map;
}

void DynamicSlamDisplayCommon::Publisher::publishObjectOdometry() {}

void DynamicSlamDisplayCommon::Publisher::publishObjectTransforms() {}

}  // namespace dyno
