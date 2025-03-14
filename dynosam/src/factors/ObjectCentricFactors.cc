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
#include "dynosam/factors/ObjectCentricFactors.hpp"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace dyno {

gtsam::Point3 projectToObject(const gtsam::Pose3& X_k,
                              const gtsam::Pose3& s0_H_k_world,
                              const gtsam::Pose3& L_s0,
                              const gtsam::Point3 Z_k) {
  gtsam::Pose3 k_H_s0_k = (L_s0.inverse() * s0_H_k_world * L_s0).inverse();
  gtsam::Pose3 L_k = s0_H_k_world * L_s0;
  gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
  gtsam::Point3 projected_m_object = L_s0.inverse() * k_H_s0_W * X_k * Z_k;
  return projected_m_object;
}

gtsam::Vector3 ObjectCentricMotion::residual(const gtsam::Pose3& X_k,
                                             const gtsam::Pose3& s0_H_k_world,
                                             const gtsam::Point3& m_L,
                                             const gtsam::Point3& Z_k,
                                             const gtsam::Pose3& L_0) {
  // apply transform to put map point into world via its motion
  gtsam::Point3 m_world_k = s0_H_k_world * (L_0 * m_L);
  // put map_point_world into local camera coordinate
  gtsam::Point3 m_camera_k = X_k.inverse() * m_world_k;
  return m_camera_k - Z_k;
}

gtsam::Vector ObjectCentricMotionFactor::evaluateError(
    const gtsam::Pose3& camera_pose, const gtsam::Pose3& motion,
    const gtsam::Point3& point_object, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  if (J1) {
    // error w.r.t to camera pose
    Eigen::Matrix<double, 3, 6> df_dX =
        gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Point3>(
            std::bind(&ObjectCentricMotionFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, measurement_, L_0_),
            camera_pose, motion, point_object);
    *J1 = df_dX;
  }

  if (J2) {
    // error w.r.t to motion
    Eigen::Matrix<double, 3, 6> df_dH =
        gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Point3>(
            std::bind(&ObjectCentricMotionFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, measurement_, L_0_),
            camera_pose, motion, point_object);
    *J2 = df_dH;
  }

  if (J3) {
    // error w.r.t to point in local
    Eigen::Matrix<double, 3, 3> df_dm =
        gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Point3>(
            std::bind(&ObjectCentricMotionFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, measurement_, L_0_),
            camera_pose, motion, point_object);
    *J3 = df_dm;
  }

  return residual(camera_pose, motion, point_object, measurement_, L_0_);
}

gtsam::Vector DecoupledObjectCentricMotionFactor::evaluateError(
    const gtsam::Pose3& s0_H_k_world, const gtsam::Point3& m_L,
    boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2) const {
  auto reordered_resiudal = [&](const gtsam::Pose3& s0_H_k_world,
                                const gtsam::Point3& m_L) {
    return residual(X_k_, s0_H_k_world, m_L, Z_k_, L_0_);
  };

  if (J1) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3,
                                     gtsam::Point3>(reordered_resiudal,
                                                    s0_H_k_world, m_L);
    *J1 = J;
  }

  if (J2) {
    Eigen::Matrix<double, 3, 3> J =
        gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3,
                                     gtsam::Point3>(reordered_resiudal,
                                                    s0_H_k_world, m_L);
    *J2 = J;
  }

  return reordered_resiudal(s0_H_k_world, m_L);
}

gtsam::Vector StructurelessObjectCentricMotion2::residual(
    const gtsam::Pose3& X_k_1, const gtsam::Pose3& H_k_1,
    const gtsam::Pose3& X_k, const gtsam::Pose3& H_k,
    const gtsam::Point3& Z_k_1, const gtsam::Point3& Z_k,
    const gtsam::Pose3& L_0) {
  return projectToObject(X_k_1, H_k_1, L_0, Z_k_1) -
         projectToObject(X_k, H_k, L_0, Z_k);
}

gtsam::Vector StructurelessDecoupledObjectCentricMotion::evaluateError(
    const gtsam::Pose3& H_k_1, const gtsam::Pose3& H_k,
    boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2) const {
  // use lambda to create residual with arguments and variables
  auto reordered_resiudal = [&](const gtsam::Pose3& H_k_1,
                                const gtsam::Pose3& H_k) -> gtsam::Vector3 {
    return residual(X_k_1_, H_k_1, X_k_, H_k, Z_k_1_, Z_k_, L_0_);
  };

  if (J1) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3,
                                     gtsam::Pose3>(reordered_resiudal, H_k_1,
                                                   H_k);
    *J1 = J;
  }

  if (J2) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3,
                                     gtsam::Pose3>(reordered_resiudal, H_k_1,
                                                   H_k);
    *J2 = J;
  }

  return reordered_resiudal(H_k_1, H_k);
}

gtsam::Vector StructurelessObjectCentricMotionFactor2::evaluateError(
    const gtsam::Pose3& X_k_1, const gtsam::Pose3& H_k_1,
    const gtsam::Pose3& X_k, const gtsam::Pose3& H_k,
    boost::optional<gtsam::Matrix&> J1, boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3,
    boost::optional<gtsam::Matrix&> J4) const {
  if (J1) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_0_),
            X_k_1, H_k_1, X_k, H_k);
    *J1 = J;
  }

  if (J1) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_0_),
            X_k_1, H_k_1, X_k, H_k);
    *J1 = J;
  }

  if (J2) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative42<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_0_),
            X_k_1, H_k_1, X_k, H_k);
    *J2 = J;
  }

  if (J3) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative43<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_0_),
            X_k_1, H_k_1, X_k, H_k);
    *J3 = J;
  }

  if (J4) {
    Eigen::Matrix<double, 3, 6> J =
        gtsam::numericalDerivative44<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Pose3, gtsam::Pose3>(
            std::bind(&StructurelessObjectCentricMotionFactor2::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, std::placeholders::_4, Z_k_1_,
                      Z_k_, L_0_),
            X_k_1, H_k_1, X_k, H_k);
    *J4 = J;
  }

  return residual(X_k_1, H_k_1, X_k, H_k, Z_k_1_, Z_k_, L_0_);
}

gtsam::Vector ObjectCentricSmoothing::evaluateError(
    const gtsam::Pose3& motion_k_2, const gtsam::Pose3& motion_k_1,
    const gtsam::Pose3& motion_k, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  if (J1) {
    *J1 = gtsam::numericalDerivative31<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectCentricSmoothing::residual, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, L_0_),
        motion_k_2, motion_k_1, motion_k);
  }

  if (J2) {
    *J2 = gtsam::numericalDerivative32<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectCentricSmoothing::residual, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, L_0_),
        motion_k_2, motion_k_1, motion_k);
  }

  if (J3) {
    *J3 = gtsam::numericalDerivative33<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectCentricSmoothing::residual, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, L_0_),
        motion_k_2, motion_k_1, motion_k);
  }

  return residual(motion_k_2, motion_k_1, motion_k, L_0_);
}
gtsam::Vector ObjectCentricSmoothing::residual(const gtsam::Pose3& motion_k_2,
                                               const gtsam::Pose3& motion_k_1,
                                               const gtsam::Pose3& motion_k,
                                               const gtsam::Pose3& L_0) {
  const gtsam::Pose3 L_k_2 = motion_k_2 * L_0;
  const gtsam::Pose3 L_k_1 = motion_k_1 * L_0;
  const gtsam::Pose3 L_k = motion_k * L_0;

  gtsam::Pose3 k_2_H_k_1 = L_k_2.inverse() * L_k_1;
  gtsam::Pose3 k_1_H_k = L_k_1.inverse() * L_k;

  gtsam::Pose3 relative_motion = k_2_H_k_1.inverse() * k_1_H_k;

  return gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::Identity(),
                                            relative_motion);

  // const gtsam::Pose3 L_W_km2 = motion_k_2 * L_0;
  // const gtsam::Pose3 L_W_km1 = motion_k_1 * L_0;
  // const gtsam::Pose3 L_W_k = motion_k * L_0;

  // //in local
  // gtsam::Pose3 H_Lkm2_km1 = L_W_km2.inverse() * L_W_km1;
  // gtsam::Pose3 H_Lkm1_k = L_W_km1.inverse() * L_W_k;

  // gtsam::Pose3 H_W_km2_km1 = L_W_km1 * L_W_km2.inverse();
  // gtsam::Pose3 H_W_km1_k = L_W_k * L_W_km1.inverse();

  // //change in motion in the world frame (rotation), which is invariant
  // gtsam::Rot3 delta_W = (H_W_km2_km1.inverse() * H_W_km1_k).rotation();
  // //change in motion in the local frame (translation), which is invariant
  // gtsam::Point3 delta_local = (H_Lkm2_km1.inverse() *
  // H_Lkm1_k).translation();

  // return gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::Identity(),
  //                                           gtsam::Pose3(delta_W,
  //                                           delta_local));
}

}  // namespace dyno
