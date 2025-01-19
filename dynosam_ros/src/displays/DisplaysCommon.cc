#include "dynosam_ros/displays/DisplaysCommon.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/memory.h>

#include <dynosam/common/PointCloudProcess.hpp>
#include <dynosam/visualizer/ColourMap.hpp>

#include "dynosam_ros/RosUtils.hpp"


namespace dyno {

CloudPerObject DisplayCommon::publishPointCloud(PointCloud2Pub::SharedPtr pub, const StatusLandmarkVector& landmarks, const gtsam::Pose3& T_world_camera, const std::string& frame_id) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;

  CloudPerObject clouds_per_obj;

  for (const auto& status_estimate : landmarks) {
    Landmark lmk_world = status_estimate.value();
    const ObjectId object_id = status_estimate.objectId();
    if (status_estimate.referenceFrame() == ReferenceFrame::LOCAL) {
      lmk_world = T_world_camera * status_estimate.value();
    } else if (status_estimate.referenceFrame() == ReferenceFrame::OBJECT) {
      throw DynosamException(
          "Cannot display object point in the object reference frame");
    }

    pcl::PointXYZRGB pt;
    if (status_estimate.isStatic()) {
      // publish static lmk's as white
      pt = pcl::PointXYZRGB(lmk_world(0), lmk_world(1), lmk_world(2), 0, 0, 0);
    } else {
      const cv::Scalar colour = Color::uniqueId(object_id);
      pt = pcl::PointXYZRGB(lmk_world(0), lmk_world(1), lmk_world(2), colour(0),
                            colour(1), colour(2));
    }
    cloud.points.push_back(pt);
    clouds_per_obj[object_id].push_back(pt);
  }

  sensor_msgs::msg::PointCloud2 pc2_msg;
  pcl::toROSMsg(cloud, pc2_msg);
  pc2_msg.header.frame_id = frame_id;
  pub->publish(pc2_msg);

  return clouds_per_obj;
}

void DisplayCommon::publishOdometry(OdometryPub::SharedPtr pub, const gtsam::Pose3& T_world_camera, Timestamp timestamp, const std::string& frame_id, const std::string& child_frame_id) {
    nav_msgs::msg::Odometry odom_msg;
    utils::convertWithHeader(T_world_camera, odom_msg, timestamp,
                            frame_id, child_frame_id);
    pub->publish(odom_msg);
}


void DisplayCommon::publishOdometryPath(PathPub::SharedPtr pub, const gtsam::Pose3Vector& poses, Timestamp latest_timestamp, const std::string& frame_id) {

nav_msgs::msg::Path path;
  for (const gtsam::Pose3& odom : poses) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    utils::convertWithHeader(odom, pose_stamped, latest_timestamp,frame_id);
    path.poses.push_back(pose_stamped);
  }

  path.header.stamp = utils::toRosTime(latest_timestamp);
  path.header.frame_id = frame_id;
  pub->publish(path);
}

} //dyno