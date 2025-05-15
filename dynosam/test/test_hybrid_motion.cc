#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/debug.h>

#include "dynosam/backend/rgbd/impl/test_HybridFormulations.hpp"
#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "internal/helpers.hpp"
#include "internal/simulator.hpp"

using namespace dyno;

TEST(HybridObjectMotion, testProjections) {
  gtsam::Pose3 e_H_k_world =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 21);
  gtsam::Pose3 cam_pose =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 19);
  gtsam::Pose3 L0 =
      dyno::utils::perturbWithNoise(gtsam::Pose3::Identity(), 0.5, 12);

  gtsam::Point3 m_object =
      dyno::utils::perturbWithNoise(gtsam::Point3(0, 0, 0), 1.5, 42);

  // put point in Xk
  gtsam::Point3 m_camera = cam_pose.inverse() * e_H_k_world * L0 * m_object;

  gtsam::Point3 projected_m_object = dyno::HybridObjectMotion::projectToObject3(
      cam_pose, e_H_k_world, L0, m_camera);

  // basically checking the inverse operation
  gtsam::Point3 projected_m_camera = dyno::HybridObjectMotion::projectToCamera3(
      cam_pose, e_H_k_world, L0, projected_m_object);

  EXPECT_TRUE(gtsam::assert_equal(projected_m_object, m_object));
  EXPECT_TRUE(gtsam::assert_equal(projected_m_camera, m_camera));
}

TEST(StructurelessObjectCentricMotionFactor2, testZeroError) {
  // construct point in L and then move it
  using namespace dyno::test_hybrid;

  gtsam::Pose3 L_e = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);
  gtsam::Pose3 e_H_k_world(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                           gtsam::Point3(0.05, -0.10, 0.20));
  gtsam::Pose3 s_H_k_1 =
      utils::perturbWithNoise<gtsam::Pose3>(e_H_k_world, 0.3);

  // observing poses
  gtsam::Pose3 X_k = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);
  gtsam::Pose3 X_k_1 = utils::createRandomAroundIdentity<gtsam::Pose3>(0.4);

  gtsam::Point3 m_object(0.4, 1.0, 0.8);

  // measurements in camera at k-1 and k
  gtsam::Point3 Z_k = X_k.inverse() * e_H_k_world * L_e * m_object;
  gtsam::Point3 Z_k_1 = X_k_1.inverse() * s_H_k_1 * L_e * m_object;

  auto noise = gtsam::noiseModel::Isotropic::Sigma(3u, 0.1);

  StructurelessObjectCentricMotionFactor2 factor(0, 1, 2, 3, Z_k_1, Z_k, L_e,
                                                 noise);
  gtsam::Vector error = factor.evaluateError(X_k_1, s_H_k_1, X_k, e_H_k_world);
  EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(0, 0, 0), error, 1e-4));
}
