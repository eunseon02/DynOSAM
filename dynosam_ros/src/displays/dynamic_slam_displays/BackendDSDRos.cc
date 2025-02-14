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

#include "dynosam_ros/displays/dynamic_slam_displays/BackendDSDRos.hpp"

namespace dyno {

BackendDSDRos::BackendDSDRos(const DisplayParams params,
                             rclcpp::Node::SharedPtr node)
    : BackendDisplay(), DSDRos(params, node) {}

void BackendDSDRos::spinOnce(
    const BackendOutputPacket::ConstPtr& backend_output) {
  // publish vo and path
  constexpr static bool kPublishOdomAsTf = false;
  this->publishVisualOdometry(backend_output->pose(),
                              backend_output->getTimestamp(), kPublishOdomAsTf);
  this->publishVisualOdometryPath(backend_output->optimized_camera_poses,
                                  backend_output->getTimestamp());

  // publish static cloud
  this->publishStaticPointCloud(backend_output->static_landmarks,
                                backend_output->pose());

  // publish dynamic cloud
  this->publishDynamicPointCloud(backend_output->dynamic_landmarks,
                                 backend_output->pose());

  const auto& object_motions = backend_output->optimized_object_motions;
  const auto& object_poses = backend_output->optimized_object_poses;

  // publish objects
  DSDTransport::Publisher object_poses_publisher = dsd_transport_.addObjectInfo(
      object_motions, object_poses, params_.world_frame_id,
      backend_output->getFrameId(), backend_output->getTimestamp());
  object_poses_publisher.publishObjectOdometry();
  object_poses_publisher.publishObjectPaths();
}

}  // namespace dyno
