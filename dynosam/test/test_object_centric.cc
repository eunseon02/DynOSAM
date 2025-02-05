/*
 *   Copyright (c) 2025
 *   All rights reserved.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/debug.h>

#include "dynosam/factors/ObjectCentricFactors.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "internal/helpers.hpp"
#include "internal/simulator.hpp"

using namespace dyno;

TEST(ObjectCentricEstimator, testObjectCentricProjection) {
  gtsam::Pose3 s0_H_k_world =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 21);
  gtsam::Pose3 cam_pose =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 19);
  gtsam::Pose3 L0 =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 12);

  gtsam::Point3 m_object =
      dyno::utils::perturbWithNoise(gtsam::Point3(0, 0, 0), 1.5, 42);

  // put point in Xk
  gtsam::Point3 m_camera = cam_pose.inverse() * s0_H_k_world * L0 * m_object;

  // test that the inverse H equation works
  //  gtsam::Pose3 k_H_s0_k = (L0.inverse() * s0_H_k_world * L0).inverse();
  //  gtsam::Pose3 L_k = s0_H_k_world * L0;
  //  gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
  //  gtsam::Point3 projected_m_object = L0.inverse() * k_H_s0_W * cam_pose *
  //  m_camera;
  gtsam::Point3 projected_m_object =
      dyno::projectToObject(cam_pose, s0_H_k_world, L0, m_camera);

  EXPECT_TRUE(gtsam::assert_equal(projected_m_object, m_object));
}

TEST(StructurelessObjectCentricMotionFactor2, testZeroError) {
  // construct point in L and then move it

  gtsam::Pose3 L_s = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);
  gtsam::Pose3 s_H_k(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                     gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 s_H_k_1 = utils::perturbWithNoise<gtsam::Pose3>(s_H_k, 0.3);

  // observing poses
  gtsam::Pose3 X_k = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);
  gtsam::Pose3 X_k_1 = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);

  gtsam::Point3 m_object(0.4, 1.0, 0.8);

  // measurements in camera at k-1 and k
  gtsam::Point3 Z_k = X_k.inverse() * s_H_k * L_s * m_object;
  gtsam::Point3 Z_k_1 = X_k_1.inverse() * s_H_k_1 * L_s * m_object;

  auto noise = gtsam::noiseModel::Isotropic::Sigma(3u, 0.1);

  StructurelessObjectCentricMotionFactor2 factor(0, 1, 2, 3, Z_k_1, Z_k, L_s,
                                                 noise);
  gtsam::Vector error = factor.evaluateError(X_k_1, s_H_k_1, X_k, s_H_k);
  EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(0, 0, 0), error, 1e-4));
}
