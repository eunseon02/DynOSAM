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

#include "dynosam/backend/MonoBackendTools.hpp"
#include "dynosam/utils/Numerical.hpp"

namespace dyno {

namespace mono_backend_tools {


/**
 * @brief Triangulation of a point cluster with rotation compensated
 * Solves for the 3D positions (approximation) of a point cluster at the previous timestep given the 2D observations in two camera frames
 *
 */

gtsam::Point3Vector triangulatePoint3Vector(const gtsam::Pose3& X_world_camera_prev,
  const gtsam::Pose3& X_world_camera_curr,
  const gtsam::Matrix3& intrinsic,
  const gtsam::Point2Vector& observation_prev,
  const gtsam::Point2Vector& observation_curr,
  const gtsam::Matrix3& obj_rotation){

  const gtsam::Matrix3 X_world_camera_prev_rot = X_world_camera_prev.rotation().matrix();
  const gtsam::Point3 X_world_camera_prev_trans = X_world_camera_prev.translation();
  const gtsam::Matrix3 X_world_camera_curr_rot = X_world_camera_curr.rotation().matrix();
  const gtsam::Point3 X_world_camera_curr_trans = X_world_camera_curr.translation();

  gtsam::Matrix3 K_inv = intrinsic.inverse();

  gtsam::Matrix3 project_inv_prev = X_world_camera_prev_rot*K_inv;
  gtsam::Matrix3 project_inv_curr = X_world_camera_curr_rot*K_inv;

  // std::cout << observation_prev.size() << std::endl;
  // for (int i = 0; i < observation_prev.size(); i++){
  //   std::cout << "Obv prev " << i << "\n" << observation_prev[i] << std::endl;
  // }

  gtsam::Point2 centroid_2d_prev = computeCentroid(observation_prev);
  gtsam::Point3 centroid_2d_homo(centroid_2d_prev.x(), centroid_2d_prev.y(), 1.0);
  gtsam::Point3 centroid_3d_prev = K_inv*centroid_2d_homo;
  gtsam::Point3 coeffs_scale = (obj_rotation - Eigen::Matrix3d::Identity())*centroid_3d_prev;

  gtsam::Point3 results_single = obj_rotation*X_world_camera_prev_trans - X_world_camera_curr_trans;

  // std::cout << coeffs_scale << std::endl;
  // std::cout << results_single << std::endl;

  int n_points = observation_prev.size();
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Zero(3*n_points, 1+2*n_points);
  Eigen::VectorXd results(3*n_points);
  // std::cout << "Coeffs initial\n" << coeffs << std::endl;
  for (int i_points = 0; i_points < n_points; i_points++){
    gtsam::Point3 this_obv_prev(observation_prev[i_points].x(), observation_prev[i_points].y(), 1.0);
    gtsam::Point3 this_obv_curr(observation_curr[i_points].x(), observation_curr[i_points].y(), 1.0);

    gtsam::Point3 coeffs_prev = obj_rotation*(project_inv_prev*this_obv_prev);
    gtsam::Point3 coeffs_curr = project_inv_curr*this_obv_curr;

    // coeffs << coeffs_curr.x(), -coeffs_prev.x(),
    //           coeffs_curr.y(), -coeffs_prev.y(),
    //           coeffs_curr.z(), -coeffs_prev.z();
    coeffs.block<3, 1>(3*i_points, 0) = coeffs_scale;
    coeffs.block<3, 1>(3*i_points, 1+2*i_points) = coeffs_curr;
    coeffs.block<3, 1>(3*i_points, 2+2*i_points) = -coeffs_prev;

    // Eigen::Vector3d results(results_gtsam.x(), results_gtsam.y(), results_gtsam.z());
    results.block<3, 1>(3*i_points, 0) = results_single;
  }

  // std::cout << "Coeffs constructed\n" << coeffs << std::endl;

  gtsam::Point3Vector points_world;
  Eigen::VectorXd depths_lstsq = coeffs.colPivHouseholderQr().solve(results);

  // std::cout << depths_lstsq << std::endl;

  for (int i_points = 0; i_points < n_points; i_points++){
    gtsam::Point3 this_obv_prev(observation_prev[i_points].x(), observation_prev[i_points].y(), 1.0);
    gtsam::Point3 this_obv_curr(observation_curr[i_points].x(), observation_curr[i_points].y(), 1.0);
    Eigen::Vector2d depths(depths_lstsq(1+2*i_points), depths_lstsq(2+2*i_points));

    gtsam::Point3 this_point_curr = K_inv*(depths(0)*this_obv_curr);
    gtsam::Point3 this_point_prev = K_inv*(depths(1)*this_obv_prev);

    gtsam::Point3 this_point_world = X_world_camera_prev_rot*this_point_prev + X_world_camera_prev_trans;
    points_world.push_back(this_point_world);
  }



  return points_world;

}


/**
 * @brief Triangulation of a point cluster with rotation compensated, and not expanding H_world into H_L
 * Solves for the 3D positions (approximation) of a point cluster at the previous timestep given the 2D observations in two camera frames
 *
 */

gtsam::Point3Vector triangulatePoint3VectorNonExpanded(const gtsam::Pose3& X_world_camera_prev,
  const gtsam::Pose3& X_world_camera_curr,
  const gtsam::Matrix3& intrinsic,
  const gtsam::Point2Vector& observation_prev,
  const gtsam::Point2Vector& observation_curr,
  const gtsam::Matrix3& obj_rotation){

  const gtsam::Matrix3 X_world_camera_prev_rot = X_world_camera_prev.rotation().matrix();
  const gtsam::Point3 X_world_camera_prev_trans = X_world_camera_prev.translation();
  const gtsam::Matrix3 X_world_camera_curr_rot = X_world_camera_curr.rotation().matrix();
  const gtsam::Point3 X_world_camera_curr_trans = X_world_camera_curr.translation();

  gtsam::Matrix3 K_inv = intrinsic.inverse();

  gtsam::Matrix3 project_inv_prev = X_world_camera_prev_rot*K_inv;
  gtsam::Matrix3 project_inv_curr = X_world_camera_curr_rot*K_inv;

  // std::cout << observation_prev.size() << std::endl;
  // for (int i = 0; i < observation_prev.size(); i++){
  //   std::cout << "Obv prev " << i << "\n" << observation_prev[i] << std::endl;
  // }

  gtsam::Point3 results = obj_rotation*X_world_camera_prev_trans - X_world_camera_curr_trans;

  // std::cout << coeffs_scale << std::endl;
  // std::cout << results_single << std::endl;

  int n_points = observation_prev.size();
  gtsam::Point3Vector points_world;
  for (int i_points = 0; i_points < n_points; i_points++){
    gtsam::Point3 this_obv_prev(observation_prev[i_points].x(), observation_prev[i_points].y(), 1.0);
    gtsam::Point3 this_obv_curr(observation_curr[i_points].x(), observation_curr[i_points].y(), 1.0);

    gtsam::Point3 coeffs_prev = obj_rotation*(project_inv_prev*this_obv_prev);
    gtsam::Point3 coeffs_curr = project_inv_curr*this_obv_curr;

    coeffs << coeffs_curr.x(), -coeffs_prev.x(),
              coeffs_curr.y(), -coeffs_prev.y(),
              coeffs_curr.z(), -coeffs_prev.z();

    Eigen::VectorXd depths_lstsq = coeffs.colPivHouseholderQr().solve(results);

    gtsam::Point3 this_point_curr = K_inv*(depths(0)*this_obv_curr);
    gtsam::Point3 this_point_prev = K_inv*(depths(1)*this_obv_prev);

    gtsam::Point3 this_point_world = X_world_camera_prev_rot*this_point_prev + X_world_camera_prev_trans;
    points_world.push_back(this_point_world);
  }

  // std::cout << "Coeffs constructed\n" << coeffs << std::endl;


  // std::cout << depths_lstsq << std::endl;

  return points_world;

}


} //mono_backend_tools
} //dyno
