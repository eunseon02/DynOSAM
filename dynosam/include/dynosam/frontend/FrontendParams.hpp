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

#include <string>
#include <cmath>

namespace dyno {

struct FrontendParams {

  // tracking points params
  int max_tracking_points_bg = 800;
  int max_tracking_points_obj = 800;

  /// @brief  Maximum one feature per bucked with cell_size width and height (used for 2D occupancy grid for static features)
  int cell_size_static = 15;
  int cell_size_dynamic = 15;

  // scene flow thresholds
  double scene_flow_magnitude = 0.12;
  double scene_flow_percentage = 0.5;

  // depth thresholds
  double depth_background_thresh = 40.0;
  double depth_obj_thresh = 25.0;

  // ORB detector params
  int n_features = 1200;
  double scale_factor = 1.2;
  int n_levels = 8;
  int init_threshold_fast = 20;
  int min_threshold_fast = 7;

  int shrink_row = 0;
  int shrink_col = 0;


  //if the mono pipeline is selcted as the frontend then only mono related ransac variables will be used
  //if the pipeline is RGBD then the user can select either the pnp (3d2d ransac) or stereo (3d3d) solvers
  //for both object and ego motion

  //! Mono (2d2d) related params
  // if mono pipeline is used AND an additional inertial sensor is provided (e.g IMU)
  // then 2d point ransac will be used to estimate the camera pose
  bool ransac_use_2point_mono = false;
  bool ransac_randomize = true;
  //used for 2d2d
  double ransac_threshold_mono = 2.0*(1.0 - cos(atan(sqrt(2.0)*0.5/800.0)));
  bool optimize_2d2d_pose_from_inliers = false;

  //!When using RGBD pipeline, ego-motion will be sovled using pnp (3d2d correspondences). Else, stereo
  bool use_ego_motion_pnp = true;

  //! When using RGBD pipeline, object motion will be sovled using pnp (3d2d correspondences). Else, stereo
  bool use_object_motion_pnp = true;

  //!PnP (3d2d) related params
  //https://github.com/laurentkneip/opengv/issues/121
  // double ransac_threshold = 2.0*(1.0 - cos(atan(sqrt(2.0)*0.5/800.0)));
  //! equivalent to reprojection error in pixels
  double ransac_threshold_pnp = 1.0;
  //! Use 3D-2D tracking to remove outliers
  bool optimize_3d2d_pose_from_inliers = false;


  //! Stereo (3d3d) related params
  //TODO: not sure what this error realtes to!!!
  double ransac_threshold_stereo = 0.001;
  //! Use 3D-3D tracking to remove outliers
  bool optimize_3d3d_pose_from_inliers = false;

  //! Generic rasac params
  double ransac_iterations = 500;
  double ransac_probability = 0.995;

  static FrontendParams fromYaml(const std::string& file_path);

};

}
