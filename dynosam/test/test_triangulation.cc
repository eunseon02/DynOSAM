/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Yiduo Wang (yiduo.wang@sydney.edu.au)
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
#include "dynosam/backend/MonoBackendTools.hpp"
#include "dynosam/common/Types.hpp"

#include <gtest/gtest.h>

using namespace dyno;


TEST(MonoBackendTools, triangulatePoint3Vector)
{
  gtsam::Pose3 X_world_camera_prev;
  // Use Quaternion and translation to construct the current pose
  // Around y axis, 15 degree - equivalent to 15 degree to the right in yaw
  gtsam::Pose3 X_world_camera_curr(gtsam::Rot3(0.9914449, 0.0, 0.1305262, 0.0), gtsam::Point3(0.1, 0.1, 2.0));

  // std::cout << X_world_camera_prev << "\n";
  // std::cout << X_world_camera_curr << "\n";

  gtsam::Matrix3 obj_rot = Eigen::Matrix3d::Identity();
  // Euler angle x = 5 degree, y = 10 degree, z = 15 degree, order ZYX
  obj_rot <<  0.9512513, -0.2432154,  0.1896506, 
              0.2548870,  0.9661673, -0.0394135, 
             -0.1736482,  0.0858316,  0.9810603;

  // std::cout << obj_rot << "\n";

  gtsam::Matrix3 intrinsic = Eigen::Matrix3d::Identity();
  intrinsic(0, 0) = 320.0;
  intrinsic(1, 1) = 320.0;
  intrinsic(0, 2) = 320.0;
  intrinsic(1, 2) = 240.0;

  // std::cout << intrinsic << "\n";

  gtsam::Point3Vector points_prev, points_curr;
  points_prev.reserve(8);
  points_curr.reserve(8);

  // a 2*2*2 cude
  points_prev[0] = Eigen::Vector3d(5.0, 1.0, 5.0);
  points_prev[1] = Eigen::Vector3d(7.0, 1.0, 5.0);
  points_prev[2] = Eigen::Vector3d(5.0, 1.0, 7.0);
  points_prev[3] = Eigen::Vector3d(7.0, 1.0, 7.0);
  points_prev[4] = Eigen::Vector3d(5.0, -1.0, 5.0);
  points_prev[5] = Eigen::Vector3d(7.0, -1.0, 5.0);
  points_prev[6] = Eigen::Vector3d(5.0, -1.0, 7.0);
  points_prev[7] = Eigen::Vector3d(7.0, -1.0, 7.0);

  // same rotation as obj_rot
  gtsam::Pose3 H_prev_curr_world(gtsam::Rot3(0.9872283, 0.0317164, 0.0919997, 0.1261366), gtsam::Point3(2.0, 0.0, 0.0));
  for (int i = 0; i < 8; i++){
    points_curr[i] = H_prev_curr_world.transformFrom(points_prev[i]);
  }

  gtsam::Point2Vector observation_prev, observation_curr;
  observation_prev.reserve(8);
  observation_curr.reserve(8);

  for (int i = 0; i < 8; i++){
    gtsam::Point3 local_point_prev = intrinsic*X_world_camera_prev.transformTo(points_prev[i]);
    gtsam::Point3 local_point_curr = intrinsic*X_world_camera_curr.transformTo(points_curr[i]);

    observation_prev[i] = Eigen::Vector2d(local_point_prev.x()/local_point_prev.z(), local_point_prev.y()/local_point_prev.z());
    observation_curr[i] = Eigen::Vector2d(local_point_curr.x()/local_point_curr.z(), local_point_curr.y()/local_point_curr.z());
  }

  gtsam::Point3Vector points_world = dyno::mono_backend_tools::triangulatePoint3Vector(X_world_camera_prev, X_world_camera_curr, intrinsic, 
                                                                                       observation_prev, observation_curr, obj_rot);
  // EXPECT_EQ(expected_outliers, outliers);
}

