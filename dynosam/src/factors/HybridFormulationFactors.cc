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
#include "dynosam/factors/HybridFormulationFactors.hpp"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace dyno {

gtsam::Point3 HybridObjectMotion::projectToObject3(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Pose3& L_s0, const gtsam::Point3& Z_k) {
  gtsam::Pose3 k_H_s0_k = (L_s0.inverse() * e_H_k_world * L_s0).inverse();
  gtsam::Pose3 L_k = e_H_k_world * L_s0;
  gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
  gtsam::Point3 projected_m_object = L_s0.inverse() * k_H_s0_W * X_k * Z_k;
  return projected_m_object;
}

gtsam::Point3 HybridObjectMotion::projectToCamera3(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Pose3& L_e, const gtsam::Point3& m_L) {
  // apply transform to put map point into world via its motion
  const auto A = projectToCamera3Transform(X_k, e_H_k_world, L_e);
  gtsam::Point3 m_camera_k = A * m_L;
  return m_camera_k;
}

gtsam::Pose3 HybridObjectMotion::projectToCamera3Transform(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Pose3& L_e) {
  return X_k.inverse() * e_H_k_world * L_e;
}

// gtsam::Point3 HybridObjectMotion::projectToCamera3(
//     const gtsam::Pose3& X_k,
//     const gtsam::Pose3& e_H_k_world,
//     const gtsam::Pose3& L_e,
//     const gtsam::Point3& m_L,
//     gtsam::OptionalJacobian<3, 6> H_Xk,
//     gtsam::OptionalJacobian<3, 6> H_eHk,
//     gtsam::OptionalJacobian<3, 6> H_Le,
//     gtsam::OptionalJacobian<3, 6> H_mL) {

//   gtsam::Matrix36 H_A_Xk, H_A_eHk, H_A_Le;
//   gtsam::Pose3 A = projectToCamera3Transform(X_k, e_H_k_world, L_e,
//                                       H_Xk ? &H_A_Xk : {},
//                                       H_eHk ? &H_A_eHk : boost::none,
//                                       H_Le ? &H_A_Le : boost::none);

//   gtsam::Matrix33 H_mL_local;
//   gtsam::Point3 m_camera_k = A.transformFrom(m_L, H_mL ? &H_mL_local :
//   boost::none);

//   // Chain Jacobians
//   if (H_Xk)   *H_Xk   = H_mL_local * H_A_Xk;
//   if (H_eHk)  *H_eHk  = H_mL_local * H_A_eHk;
//   if (H_Le)   *H_Le   = H_mL_local * H_A_Le;
//   if (H_mL)   *H_mL   = H_mL_local;

//   return m_camera_k;
// }

// gtsam::Pose3 HybridObjectMotion::projectToCamera3Transform(
//     const gtsam::Pose3& X_k,
//     const gtsam::Pose3& e_H_k_world,
//     const gtsam::Pose3& L_e,
//     gtsam::OptionalJacobian<6, 6> H_Xk,
//     gtsam::OptionalJacobian<6, 6> H_eHk,
//     gtsam::OptionalJacobian<6, 6> H_Le) {

//   gtsam::Matrix66 H_inv_Xk, H_comp1, H_comp2;
//   gtsam::Pose3 X_k_inv = X_k.inverse(H_Xk ? &H_inv_Xk : boost::none);
//   gtsam::Pose3 comp1 = X_k_inv.compose(e_H_k_world, H_Xk ? &H_comp1 :
//   boost::none, H_eHk ? &H_comp2 : nullptr); gtsam::Pose3 result =
//   comp1.compose(L_e, H_Xk ? H_Xk : boost::none, H_Le ? &H_comp2 :
//   boost::none);

//   if (H_Xk) {
//     *H_Xk = H_comp2 * H_comp1 * H_inv_Xk;  // chain rule: dR/dX = dR/dC1 *
//     dC1/dXinv * dXinv/dX
//   }
//   return result;
// }

gtsam::Vector3 HybridObjectMotion::residual(const gtsam::Pose3& X_k,
                                            const gtsam::Pose3& e_H_k_world,
                                            const gtsam::Point3& m_L,
                                            const gtsam::Point3& Z_k,
                                            const gtsam::Pose3& L_e) {
  return projectToCamera3(X_k, e_H_k_world, L_e, m_L) - Z_k;
}

gtsam::Vector HybridMotionFactor::evaluateError(
    const gtsam::Pose3& X_k, const gtsam::Pose3& e_H_k_world,
    const gtsam::Point3& m_L, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  if (J1) {
    // error w.r.t to camera pose
    Eigen::Matrix<double, 3, 6> df_dX =
        gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Point3>(
            std::bind(&HybridMotionFactor::residual, std::placeholders::_1,
                      std::placeholders::_2, std::placeholders::_3, z_k_, L_e_),
            X_k, e_H_k_world, m_L);
    *J1 = df_dX;
  }

  if (J2) {
    // error w.r.t to motion
    Eigen::Matrix<double, 3, 6> df_dH =
        gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Point3>(
            std::bind(&HybridMotionFactor::residual, std::placeholders::_1,
                      std::placeholders::_2, std::placeholders::_3, z_k_, L_e_),
            X_k, e_H_k_world, m_L);
    *J2 = df_dH;
  }

  if (J3) {
    // error w.r.t to point in local
    Eigen::Matrix<double, 3, 3> df_dm =
        gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3,
                                     gtsam::Point3>(
            std::bind(&HybridMotionFactor::residual, std::placeholders::_1,
                      std::placeholders::_2, std::placeholders::_3, z_k_, L_e_),
            X_k, e_H_k_world, m_L);
    *J3 = df_dm;
  }

  return residual(X_k, e_H_k_world, m_L, z_k_, L_e_);
}

gtsam::Vector HybridSmoothingFactor::evaluateError(
    const gtsam::Pose3& e_H_km2_world, const gtsam::Pose3& e_H_km1_world,
    const gtsam::Pose3& e_H_k_world, boost::optional<gtsam::Matrix&> J1,
    boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3) const {
  if (J1) {
    *J1 = gtsam::numericalDerivative31<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&HybridSmoothingFactor::residual, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, L_e_),
        e_H_km2_world, e_H_km1_world, e_H_k_world);
  }

  if (J2) {
    *J2 = gtsam::numericalDerivative32<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&HybridSmoothingFactor::residual, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, L_e_),
        e_H_km2_world, e_H_km1_world, e_H_k_world);
  }

  if (J3) {
    *J3 = gtsam::numericalDerivative33<gtsam::Vector6, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3>(
        std::bind(&HybridSmoothingFactor::residual, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, L_e_),
        e_H_km2_world, e_H_km1_world, e_H_k_world);
  }

  return residual(e_H_km2_world, e_H_km1_world, e_H_k_world, L_e_);
}
gtsam::Vector HybridSmoothingFactor::residual(const gtsam::Pose3& e_H_km2_world,
                                              const gtsam::Pose3& e_H_km1_world,
                                              const gtsam::Pose3& e_H_k_world,
                                              const gtsam::Pose3& L_e) {
  const gtsam::Pose3 L_k_2 = e_H_km2_world * L_e;
  const gtsam::Pose3 L_k_1 = e_H_km1_world * L_e;
  const gtsam::Pose3 L_k = e_H_k_world * L_e;

  gtsam::Pose3 k_2_H_k_1 = L_k_2.inverse() * L_k_1;
  gtsam::Pose3 k_1_H_k = L_k_1.inverse() * L_k;

  gtsam::Pose3 relative_motion = k_2_H_k_1.inverse() * k_1_H_k;

  return gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::Identity(),
                                            relative_motion);
}

}  // namespace dyno
