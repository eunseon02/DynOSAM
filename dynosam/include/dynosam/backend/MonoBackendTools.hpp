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

#pragma once

#include "dynosam/common/Types.hpp"

#include <gtsam/geometry/triangulation.h>


namespace dyno {

namespace mono_backend_tools {


/**
 * @brief Triangulation of a point cluster with rotation compensated
 * Solves for the 3D positions (approximation) of a point cluster at the previous timestep given the 2D observations in two camera frames
 *
 */

gtsam::Point3Vector triangulatePoint3Vector(
  const gtsam::Pose3& X_world_camera_prev,
  const gtsam::Pose3& X_world_camera_curr,
  const gtsam::Matrix3& intrinsic,
  const gtsam::Point2Vector& observation_prev,
  const gtsam::Point2Vector& observation_curr,
  const gtsam::Matrix3& obj_rotation);

gtsam::Point3Vector triangulatePoint3VectorNonExpanded(
  const gtsam::Pose3& X_world_camera_prev,
  const gtsam::Pose3& X_world_camera_curr,
  const gtsam::Matrix3& intrinsic,
  const gtsam::Point2Vector& observation_prev,
  const gtsam::Point2Vector& observation_curr,
  const gtsam::Matrix3& obj_rotation);


} //mono_backend_tools
} //dyno
