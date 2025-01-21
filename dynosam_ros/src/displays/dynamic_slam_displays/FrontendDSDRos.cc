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

#include "dynosam_ros/displays/dynamic_slam_displays/FrontendDSDRos.hpp"

#include <cv_bridge/cv_bridge.h>

#include <dynosam/utils/SafeCast.hpp>

namespace dyno {

FrontendDSDRos::FrontendDSDRos(const DisplayParams params,
                               rclcpp::Node::SharedPtr node)
    : FrontendDisplay(),
      DSDRos(params, node),
      dsd_ground_truth_transport_(node->create_sub_node("ground_truth")) {
  tracking_image_pub_ =
      image_transport::create_publisher(node.get(), "tracking_image");
}

void FrontendDSDRos::spinOnce(
    const FrontendOutputPacketBase::ConstPtr& frontend_output) {
  // publish debug imagery
  tryPublishDebugImagery(frontend_output);

  // publish ground truth
  tryPublishGroundTruth(frontend_output);

  // publish odometry
  tryPublishVisualOdometry(frontend_output);

  // attempt cast
  RGBDInstanceOutputPacket::ConstPtr rgbd_output =
      safeCast<FrontendOutputPacketBase, RGBDInstanceOutputPacket>(
          frontend_output);
  // publish object info
  if (rgbd_output) processRGBDOutputpacket(rgbd_output);
}

void FrontendDSDRos::tryPublishDebugImagery(
    const FrontendOutputPacketBase::ConstPtr& frontend_output) {
  if (!frontend_output->debug_imagery_) return;

  const DebugImagery& debug_imagery = *frontend_output->debug_imagery_;
  if (debug_imagery.tracking_image.empty()) return;

  std_msgs::msg::Header hdr;
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(hdr, "bgr8", debug_imagery.tracking_image)
          .toImageMsg();
  tracking_image_pub_.publish(msg);
}

void FrontendDSDRos::tryPublishGroundTruth(
    const FrontendOutputPacketBase::ConstPtr& frontend_output) {
  if (!frontend_output->gt_packet_ || !frontend_output->debug_imagery_) return;

  const DebugImagery& debug_imagery = *frontend_output->debug_imagery_;
  const cv::Mat& rgb_image = debug_imagery.rgb_viz;

  if (rgb_image.empty()) return;

  // collect gt poses and motions
  ObjectPoseMap poses;
  MotionEstimateMap motions;

  const GroundTruthInputPacket& gt_packet = frontend_output->gt_packet_.value();

  for (const auto& object_pose_gt : gt_packet.object_poses_) {
    // check we have a gt motion here
    // in the case that we dont, this might be the first time the object
    // appears...
    if (!object_pose_gt.prev_H_current_world_) {
      continue;
    }

    poses.insert22(object_pose_gt.object_id_, gt_packet.frame_id_,
                   object_pose_gt.L_world_);

    ReferenceFrameValue<gtsam::Pose3> gt_motion(
        *object_pose_gt.prev_H_current_world_, ReferenceFrame::GLOBAL);
    motions.insert({object_pose_gt.object_id_, gt_motion});
  }

  // will this result in confusing tf's since the gt object and estimated
  // objects use the same link?
  DSDTransport::Publisher publisher = dsd_ground_truth_transport_.addObjectInfo(
      motions, poses, params_.world_frame_id, frontend_output->getFrameId(),
      frontend_output->getTimestamp());
  publisher.publishObjectOdometry();
}
void FrontendDSDRos::tryPublishVisualOdometry(
    const FrontendOutputPacketBase::ConstPtr& frontend_output) {
  // publish vo
  constexpr static bool kPublishOdomAsTf = true;
  this->publishVisualOdometry(frontend_output->T_world_camera_,
                              frontend_output->getTimestamp(),
                              kPublishOdomAsTf);
}

void FrontendDSDRos::processRGBDOutputpacket(
    const RGBDInstanceOutputPacket::ConstPtr& rgbd_packet) {
  // publish path
  // why the camera poses are only in the RGBDInstanceOutputPacket and not in
  // the base... I have no idea :)
  this->publishVisualOdometryPath(rgbd_packet->camera_poses_,
                                  rgbd_packet->getTimestamp());

  // publish static cloud
  CHECK(rgbd_packet);
  this->publishStaticPointCloud(rgbd_packet->static_landmarks_,
                                rgbd_packet->T_world_camera_);

  // publish and collect dynamic cloud
  CloudPerObject clouds_per_obj = this->publishDynamicPointCloud(
      rgbd_packet->dynamic_landmarks_, rgbd_packet->T_world_camera_);

  const auto& object_motions = rgbd_packet->estimated_motions_;
  const auto& object_poses = rgbd_packet->propogated_object_poses_;

  DSDTransport::Publisher object_poses_publisher = dsd_transport_.addObjectInfo(
      object_motions, object_poses, params_.world_frame_id,
      rgbd_packet->getFrameId(), rgbd_packet->getTimestamp());
  object_poses_publisher.publishObjectOdometry();
  object_poses_publisher.publishObjectTransforms();
}

}  // namespace dyno
