#include "dynosam_ros/displays/DisplaysCommon.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/memory.h>

#include <dynosam/common/PointCloudProcess.hpp>
#include <dynosam/visualizer/ColourMap.hpp>

#include "dynosam_ros/RosUtils.hpp"

namespace dyno {

CloudPerObject DisplayCommon::publishPointCloud(
    PointCloud2Pub::SharedPtr pub, const StatusLandmarkVector& landmarks,
    const gtsam::Pose3& T_world_camera, const std::string& frame_id) {
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  CloudPerObject clouds_per_obj =
      groupObjectCloud(landmarks, T_world_camera,
                       [&cloud](const pcl::PointXYZRGB& point, ObjectId) {
                         cloud.points.push_back(point);
                       });

  sensor_msgs::msg::PointCloud2 pc2_msg;
  pcl::toROSMsg(cloud, pc2_msg);
  pc2_msg.header.frame_id = frame_id;
  pub->publish(pc2_msg);

  return clouds_per_obj;
}

void DisplayCommon::publishOdometry(OdometryPub::SharedPtr pub,
                                    const gtsam::Pose3& T_world_camera,
                                    Timestamp timestamp,
                                    const std::string& frame_id,
                                    const std::string& child_frame_id) {
  nav_msgs::msg::Odometry odom_msg;
  utils::convertWithHeader(T_world_camera, odom_msg, timestamp, frame_id,
                           child_frame_id);
  pub->publish(odom_msg);
}

void DisplayCommon::publishOdometryPath(PathPub::SharedPtr pub,
                                        const gtsam::Pose3Vector& poses,
                                        Timestamp latest_timestamp,
                                        const std::string& frame_id) {
  nav_msgs::msg::Path path;
  for (const gtsam::Pose3& odom : poses) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    utils::convertWithHeader(odom, pose_stamped, latest_timestamp, frame_id);
    path.poses.push_back(pose_stamped);
  }

  path.header.stamp = utils::toRosTime(latest_timestamp);
  path.header.frame_id = frame_id;
  pub->publish(path);
}

}  // namespace dyno
