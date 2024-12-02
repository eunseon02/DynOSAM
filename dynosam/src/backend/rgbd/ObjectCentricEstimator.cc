/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/rgbd/ObjectCentricEstimator.hpp"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "dynosam/backend/BackendDefinitions.hpp"

namespace dyno {

class ObjectCentricMotionFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Point3> {
 public:
  typedef boost::shared_ptr<ObjectCentricMotionFactor> shared_ptr;
  typedef ObjectCentricMotionFactor This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>
      Base;

  gtsam::Point3 measurement_;
  gtsam::Pose3 L_0_;

  ObjectCentricMotionFactor(gtsam::Key camera_pose, gtsam::Key motion,
                            gtsam::Key point_object,
                            const gtsam::Point3& measurement,
                            const gtsam::Pose3& L_0,
                            gtsam::SharedNoiseModel model)
      : Base(model, camera_pose, motion, point_object),
        measurement_(measurement),
        L_0_(L_0) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& camera_pose, const gtsam::Pose3& motion,
      const gtsam::Point3& point_object,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override {
    if (J1) {
      // error w.r.t to camera pose
      Eigen::Matrix<double, 3, 6> df_dX =
          gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Point3>(
              std::bind(&ObjectCentricMotionFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, measurement_, L_0_),
              camera_pose, motion, point_object);
      *J1 = df_dX;
    }

    if (J2) {
      // error w.r.t to motion
      Eigen::Matrix<double, 3, 6> df_dH =
          gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Point3>(
              std::bind(&ObjectCentricMotionFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, measurement_, L_0_),
              camera_pose, motion, point_object);
      *J2 = df_dH;
    }

    if (J3) {
      // error w.r.t to point in local
      Eigen::Matrix<double, 3, 3> df_dm =
          gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Point3>(
              std::bind(&ObjectCentricMotionFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, measurement_, L_0_),
              camera_pose, motion, point_object);
      *J3 = df_dm;
    }

    return residual(camera_pose, motion, point_object, measurement_, L_0_);
  }

  static gtsam::Vector residual(const gtsam::Pose3& camera_pose,
                                const gtsam::Pose3& motion,
                                const gtsam::Point3& point_object,
                                const gtsam::Point3& measurement,
                                const gtsam::Pose3& L_0) {
    // apply transform to put map point into world via its motion
    gtsam::Point3 map_point_world = motion * (L_0 * point_object);
    // put map_point_world into local camera coordinate
    gtsam::Point3 map_point_camera = camera_pose.inverse() * map_point_world;
    return map_point_camera - measurement;
  }
};

class ObjectCentricSmoothing
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<ObjectCentricSmoothing> shared_ptr;
  typedef ObjectCentricSmoothing This;
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> Base;

  gtsam::Pose3 L_0_;

  ObjectCentricSmoothing(gtsam::Key motion_k_1, gtsam::Key motion_k,
                         const gtsam::Pose3& L_0, gtsam::SharedNoiseModel model)
      : Base(model, motion_k_1, motion_k), L_0_(L_0) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& motion_k_1, const gtsam::Pose3& motion_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override {
    gtsam::Matrix H1, H2;
    const gtsam::Pose3 L_k_1 = motion_k_1.compose(L_0_, H1);
    const gtsam::Pose3 L_k = motion_k.compose(L_0_, H2);

    gtsam::Pose3 hx =
        gtsam::traits<gtsam::Pose3>::Between(L_k_1, L_k, J1, J2);  // h(x)

    // if(J1) *J1 = H1 * (*J1);
    // if(J2) *J2 = H2 * (*J2);
    return gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::Identity(), hx);
  }
};

// here motion is P = X.inverse() * H
class AuxillaryPFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
 public:
  typedef boost::shared_ptr<AuxillaryPFactor> shared_ptr;
  typedef AuxillaryPFactor This;
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> Base;

  gtsam::Point3 measurement_;
  gtsam::Pose3 L_0_;

  AuxillaryPFactor(gtsam::Key motion, gtsam::Key point_object,
                   const gtsam::Point3& measurement, const gtsam::Pose3& L_0,
                   gtsam::SharedNoiseModel model)
      : Base(model, motion, point_object),
        measurement_(measurement),
        L_0_(L_0) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& P, const gtsam::Point3& point_object,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override {
    if (J1) {
      // error w.r.t to P
      Eigen::Matrix<double, 3, 6> df_dP =
          gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Point3>(
              std::bind(&AuxillaryPFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, measurement_, L_0_),
              P, point_object);
      *J1 = df_dP;
    }

    if (J2) {
      // error w.r.t to point in local
      Eigen::Matrix<double, 3, 3> df_dm =
          gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Point3>(
              std::bind(&AuxillaryPFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, measurement_, L_0_),
              P, point_object);
      *J2 = df_dm;
    }

    return residual(P, point_object, measurement_, L_0_);
  }

  static gtsam::Vector residual(const gtsam::Pose3& P,
                                const gtsam::Point3& point_object,
                                const gtsam::Point3& measurement,
                                const gtsam::Pose3& L_0) {
    return measurement - P * L_0 * point_object;
  }
};

class AuxillaryTransformFactor
    : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<AuxillaryTransformFactor> shared_ptr;
  typedef AuxillaryTransformFactor This;
  typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                   gtsam::Pose3>
      Base;

  const gtsam::Point3 Z_previous_;
  const gtsam::Point3 Z_current_;

  AuxillaryTransformFactor(gtsam::Key X_previous, gtsam::Key P_previous,
                           gtsam::Key X_current, gtsam::Key P_current,
                           const gtsam::Point3& Z_previous,
                           const gtsam::Point3& Z_current,
                           gtsam::SharedNoiseModel model)
      : Base(model, X_previous, P_previous, X_current, P_current),
        Z_previous_(Z_previous),
        Z_current_(Z_current) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_previous, const gtsam::Pose3& P_previous,
      const gtsam::Pose3& X_current, const gtsam::Pose3& P_current,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none,
      boost::optional<gtsam::Matrix&> J4 = boost::none) const override {
    if (J1) {
      // error w.r.t to X_prev
      Eigen::Matrix<double, 3, 6> df_dX_prev =
          gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&AuxillaryTransformFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        Z_previous_, Z_current_),
              X_previous, P_previous, X_current, P_current);
      *J1 = df_dX_prev;
    }

    if (J2) {
      // error w.r.t to P_prev
      Eigen::Matrix<double, 3, 6> df_dP_prev =
          gtsam::numericalDerivative42<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&AuxillaryTransformFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        Z_previous_, Z_current_),
              X_previous, P_previous, X_current, P_current);
      *J2 = df_dP_prev;
    }

    if (J3) {
      // error w.r.t to X_curr
      Eigen::Matrix<double, 3, 6> df_dX_curr =
          gtsam::numericalDerivative43<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&AuxillaryTransformFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        Z_previous_, Z_current_),
              X_previous, P_previous, X_current, P_current);
      *J3 = df_dX_curr;
    }

    if (J4) {
      // error w.r.t to P_curr
      Eigen::Matrix<double, 3, 6> df_dP_curr =
          gtsam::numericalDerivative44<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&AuxillaryTransformFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        Z_previous_, Z_current_),
              X_previous, P_previous, X_current, P_current);
      *J4 = df_dP_curr;
    }

    return residual(X_previous, P_previous, X_current, P_current, Z_previous_,
                    Z_current_);
  }

  static gtsam::Vector residual(const gtsam::Pose3& X_previous,
                                const gtsam::Pose3& P_previous,
                                const gtsam::Pose3& X_current,
                                const gtsam::Pose3& P_current,
                                const gtsam::Point3& Z_previous,
                                const gtsam::Point3& Z_current) {
    // gtsam::Pose3 H_current = X_current * P_current;
    // gtsam::Pose3 H_previous = X_previous * P_previous;
    gtsam::Pose3 H_current = X_current * P_current;
    gtsam::Pose3 H_previous = X_previous * P_previous;
    gtsam::Pose3 prev_H_current = H_current * H_previous.inverse();
    gtsam::Point3 m_previous_world = X_previous * Z_previous;
    gtsam::Point3 m_current_world = X_current * Z_current;
    return m_current_world - prev_H_current * m_previous_world;
  }
};

class ObjectCentricMotionOnlyFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
 public:
  typedef boost::shared_ptr<ObjectCentricMotionOnlyFactor> shared_ptr;
  typedef ObjectCentricMotionOnlyFactor This;
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> Base;

  gtsam::Point3 measurement_;
  gtsam::Pose3 X_k_;
  gtsam::Pose3 L_0_;

  ObjectCentricMotionOnlyFactor(gtsam::Key motion, gtsam::Key point_object,
                                const gtsam::Point3& measurement,
                                const gtsam::Pose3& X_k,
                                const gtsam::Pose3& L_0,
                                gtsam::SharedNoiseModel model)
      : Base(model, motion, point_object),
        measurement_(measurement),
        X_k_(X_k),
        L_0_(L_0) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& motion, const gtsam::Point3& point_object,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override {
    if (J1) {
      // error w.r.t to camera pose
      Eigen::Matrix<double, 3, 6> df_dH =
          gtsam::numericalDerivative21<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Point3>(
              std::bind(&ObjectCentricMotionOnlyFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        measurement_, X_k_, L_0_),
              motion, point_object);
      *J1 = df_dH;
    }

    if (J2) {
      // error w.r.t to camera pose
      Eigen::Matrix<double, 3, 3> df_dm =
          gtsam::numericalDerivative22<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Point3>(
              std::bind(&ObjectCentricMotionOnlyFactor::residual,
                        std::placeholders::_1, std::placeholders::_2,
                        measurement_, X_k_, L_0_),
              motion, point_object);
      *J2 = df_dm;
    }

    return residual(motion, point_object, measurement_, X_k_, L_0_);
  }

  static gtsam::Vector residual(const gtsam::Pose3& motion,
                                const gtsam::Point3& point_object,
                                const gtsam::Point3& measurement,
                                const gtsam::Pose3& X_k,
                                const gtsam::Pose3& L_0) {
    // apply transform to put map point into world via its motion
    gtsam::Point3 map_point_world = motion * (L_0 * point_object);
    // put map_point_world into local camera coordinate
    gtsam::Point3 map_point_camera = X_k.inverse() * map_point_world;
    return map_point_camera - measurement;
  }
};

class SmartHFactor
    : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<SmartHFactor> shared_ptr;
  typedef SmartHFactor This;
  typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                   gtsam::Pose3>
      Base;

  const gtsam::Point3 Z_previous_;
  const gtsam::Point3 Z_current_;

  SmartHFactor(gtsam::Key X_previous, gtsam::Key H_previous,
               gtsam::Key X_current, gtsam::Key H_current,
               const gtsam::Point3& Z_previous, const gtsam::Point3& Z_current,
               gtsam::SharedNoiseModel model)
      : Base(model, X_previous, H_previous, X_current, H_current),
        Z_previous_(Z_previous),
        Z_current_(Z_current) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_previous, const gtsam::Pose3& H_previous,
      const gtsam::Pose3& X_current, const gtsam::Pose3& H_current,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none,
      boost::optional<gtsam::Matrix&> J4 = boost::none) const override {
    if (J1) {
      // error w.r.t to X_prev
      Eigen::Matrix<double, 3, 6> df_dX_prev =
          gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&SmartHFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, Z_previous_, Z_current_),
              X_previous, H_previous, X_current, H_current);
      *J1 = df_dX_prev;
    }

    if (J2) {
      // error w.r.t to P_prev
      Eigen::Matrix<double, 3, 6> df_dP_prev =
          gtsam::numericalDerivative42<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&SmartHFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, Z_previous_, Z_current_),
              X_previous, H_previous, X_current, H_current);
      *J2 = df_dP_prev;
    }

    if (J3) {
      // error w.r.t to X_curr
      Eigen::Matrix<double, 3, 6> df_dX_curr =
          gtsam::numericalDerivative43<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&SmartHFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, Z_previous_, Z_current_),
              X_previous, H_previous, X_current, H_current);
      *J3 = df_dX_curr;
    }

    if (J4) {
      // error w.r.t to P_curr
      Eigen::Matrix<double, 3, 6> df_dP_curr =
          gtsam::numericalDerivative44<gtsam::Vector3, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&SmartHFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, Z_previous_, Z_current_),
              X_previous, H_previous, X_current, H_current);
      *J4 = df_dP_curr;
    }

    return residual(X_previous, H_previous, X_current, H_current, Z_previous_,
                    Z_current_);
  }

  static gtsam::Vector residual(const gtsam::Pose3& X_previous,
                                const gtsam::Pose3& H_previous,
                                const gtsam::Pose3& X_current,
                                const gtsam::Pose3& H_current,
                                const gtsam::Point3& Z_previous,
                                const gtsam::Point3& Z_current) {
    gtsam::Pose3 prev_H_current = H_current * H_previous.inverse();
    gtsam::Point3 m_previous_world = X_previous * Z_previous;
    gtsam::Point3 m_current_world = X_current * Z_current;
    return m_current_world - prev_H_current * m_previous_world;
  }
};

// class SmartMotionFactor : public gtsam::NonlinearFactor {

// public:
//     SmartMotionFactor() {}
//     ~SmartMotionFactor() {}

//     size_t dim() const override {}

//     std::shared_ptr<gtsam::GaussianFactor> linearize(const Values& c) const
//     override {
//       std::vector<gtsam::Matrix> A(size());

//       Vector b = -unwhitenedError(x, A);
//       // check(noiseModel_, b.size());

//       // Whiten the corresponding system now
//       if (noiseModel_)
//         noiseModel_->WhitenSystem(A, b);

//       // Fill in terms, needed to create JacobianFactor below
//       std::vector<std::pair<Key, Matrix> > terms(size());
//       for (size_t j = 0; j < size(); ++j) {
//         terms[j].first = keys()[j];
//         terms[j].second.swap(A[j]);
//       }

//       // TODO pass unwhitened + noise model to Gaussian factor
//       using noiseModel::Constrained;
//       if (noiseModel_ && noiseModel_->isConstrained())
//         return GaussianFactor::shared_ptr(
//             new JacobianFactor(terms, b,
//                 std::static_pointer_cast<Constrained>(noiseModel_)->unit()));
//       else {
//         return GaussianFactor::shared_ptr(new JacobianFactor(terms, b));
//       }
//     }

// };

// TODO: hack for now!
gtsam::FastMap<ObjectId, std::pair<FrameId, gtsam::Pose3>>
    ObjectCentricFormulation::L0_;

StateQuery<gtsam::Pose3> ObjectCentricAccessor::getSensorPose(
    FrameId frame_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  CHECK_NOTNULL(frame_node);
  return this->query<gtsam::Pose3>(frame_node->makePoseKey());
}

StateQuery<gtsam::Pose3> ObjectCentricAccessor::getObjectMotion(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node_k = map()->getFrame(frame_id);
  const auto frame_node_k_1 = map()->getFrame(frame_id - 1u);
  if (frame_node_k && frame_node_k_1) {
    const auto motion_key = frame_node_k->makeObjectMotionKey(object_id);
    StateQuery<gtsam::Pose3> motion_s0_k =
        this->query<gtsam::Pose3>(motion_key);

    StateQuery<gtsam::Pose3> motion_s0_k_1 = this->query<gtsam::Pose3>(
        frame_node_k_1->makeObjectMotionKey(object_id));

    // TODO: if motion_s0_k_1 but motion_s0_k then check that this frame ==s0,
    // then the motion will just be motion_s0_k (and should be identity)

    if (motion_s0_k && motion_s0_k_1) {
      // want a motion from k-1 to k, but we estimate s0 to k
      //^w_{k-1}H_k = ^w_{s0}H_k \: ^w_{s0}H_{k-1}^{-1}
      gtsam::Pose3 motion = motion_s0_k.get() * motion_s0_k_1->inverse();
      LOG(INFO) << "Obj motion " << motion;
      return StateQuery<gtsam::Pose3>(motion_key, motion);
    } else {
      return StateQuery<gtsam::Pose3>::NotInMap(
          frame_node_k->makeObjectMotionKey(object_id));
    }
  }
  LOG(WARNING) << "Could not construct object motion frame id=" << frame_id
               << " object id=" << object_id;
  return StateQuery<gtsam::Pose3>::InvalidMap();
}

StateQuery<gtsam::Pose3> ObjectCentricAccessor::getObjectPose(
    FrameId frame_id, ObjectId object_id) const {
  // we estimate a motion ^w_{s0}H_k, so we can compute a pose ^wL_k =
  // ^w_{s0}H_k * ^wL_{s0}
  const auto frame_node_k = map()->getFrame(frame_id);
  gtsam::Key motion_key = frame_node_k->makeObjectMotionKey(object_id);
  gtsam::Key pose_key = frame_node_k->makeObjectPoseKey(object_id);
  CHECK(frame_node_k);
  /// hmmm... if we do a query after we do an update but before an optimise then
  /// the motion will
  // be whatever we initalised it with
  // in the case of identity, the pose at k will just be L_s0 which we dont
  // want?
  StateQuery<gtsam::Pose3> motion_s0_k = this->query<gtsam::Pose3>(motion_key);
  // CHECK(false);

  if (motion_s0_k) {
    CHECK(L0_values_->exists(object_id));
    const gtsam::Pose3& L0 = L0_values_->at(object_id).second;
    // LOG(INFO) << "Object pose s0 " << L0;
    // LOG(INFO) << "Object motion " << motion_s0_k.get();
    const gtsam::Pose3 L_k = motion_s0_k.get() * L0;
    // LOG(INFO) << "Object pose " << L_k;

    return StateQuery<gtsam::Pose3>(pose_key, L_k);
  } else {
    return StateQuery<gtsam::Pose3>::NotInMap(pose_key);
  }
}
StateQuery<gtsam::Point3> ObjectCentricAccessor::getDynamicLandmark(
    FrameId frame_id, TrackletId tracklet_id) const {
  // we estimate a motion ^w_{s0}H_k, so we can compute a point ^wm_k =
  // ^w_{s0}H_k * ^wL_{s0} * ^{L_{s0}}m
  const auto frame_node_k = map()->getFrame(frame_id);
  const auto lmk_node = map()->getLandmark(tracklet_id);
  CHECK(frame_node_k);
  CHECK_NOTNULL(lmk_node);
  const auto object_id = lmk_node->object_id;
  // point in L_{s0}
  // NOTE: we use STATIC point key here
  gtsam::Key point_key = this->makeDynamicKey(tracklet_id);
  StateQuery<gtsam::Point3> point_local = this->query<gtsam::Point3>(point_key);

  // get motion from S0 to k
  gtsam::Key motion_key = frame_node_k->makeObjectMotionKey(object_id);
  StateQuery<gtsam::Pose3> motion_s0_k = this->query<gtsam::Pose3>(motion_key);

  // TODO: I guess can happen if we miss a motion becuae an object is not seen
  // for one frame?!?
  //  if (point_local)
  //    CHECK(motion_s0_k) << "We have a point " <<
  //    DynoLikeKeyFormatter(point_key)
  //                       << " but no motion at frame " << frame_id << " with
  //                       key: " << DynoLikeKeyFormatter(motion_key);
  if (point_local && motion_s0_k) {
    CHECK(L0_values_->exists(object_id));
    const gtsam::Pose3& L0 = L0_values_->at(object_id).second;
    // point in world at k
    const gtsam::Point3 m_k = motion_s0_k.get() * L0 * point_local.get();
    return StateQuery<gtsam::Point3>(point_key, m_k);
  } else {
    return StateQuery<gtsam::Point3>::NotInMap(point_key);
  }
}

StatusLandmarkEstimates ObjectCentricAccessor::getDynamicLandmarkEstimates(
    FrameId frame_id, ObjectId object_id) const {
  const auto frame_node = map()->getFrame(frame_id);
  const auto object_node = map()->getObject(object_id);
  CHECK_NOTNULL(frame_node);
  CHECK_NOTNULL(object_node);

  if (!frame_node->objectObserved(object_id)) {
    return StatusLandmarkEstimates{};
  }

  StatusLandmarkEstimates estimates;
  // unlike in the base version, iterate over all points on the object (i.e all
  // tracklets) as we can propogate all of them!!!!
  const auto& dynamic_landmarks = object_node->dynamic_landmarks;
  for (auto lmk_node : dynamic_landmarks) {
    const auto tracklet_id = lmk_node->tracklet_id;

    CHECK_EQ(object_id, lmk_node->object_id);

    // user defined function should put point in the world frame
    StateQuery<gtsam::Point3> lmk_query =
        this->getDynamicLandmark(frame_id, tracklet_id);
    if (lmk_query) {
      estimates.push_back(LandmarkStatus::DynamicInGLobal(
          lmk_query.get(),  // estimate
          frame_id, tracklet_id, object_id,
          LandmarkStatus::Method::OPTIMIZED  // this may not be correct!!
          )                                  // status
      );
    }
  }
  return estimates;
}

void ObjectCentricFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  // //acrew PointUpdateContextType for each object and trigger the update
  // const auto frame_node_k_1 = context.frame_node_k_1;
  // const auto frame_node_k = context.frame_node_k;
  // const auto object_id = context.getObjectId();

  // //TODO: for now lets just use k (which means we are dropping a
  // measurement!)
  // //just for initial testing!!
  // result.updateAffectedObject(frame_node_k_1->frame_id, object_id);
  // result.updateAffectedObject(frame_node_k->frame_id, object_id);

  // if(!point_contexts_.exists(object_id)) {
  //     point_contexts_.insert2(object_id,
  //     std::vector<PointUpdateContextType>());
  // }
  // point_contexts_.at(object_id).push_back(context);
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  auto theta_accessor = this->accessorFromTheta();

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(context.getObjectId());

  gtsam::Pose3 L_0;
  FrameId s0;
  std::tie(s0, L_0) = getL0(context.getObjectId(), frame_node_k_1->getId());
  auto landmark_motion_noise = noise_models_.landmark_motion_noise;
  // check that the first frame id is at least the initial frame for s0

  // TODO:this will not be the case with sliding/window as we reconstruct the
  // graph from a different starting point!!
  //  CHECK_GE(frame_node_k_1->getId(), s0);

  if (!isDynamicTrackletInMap(lmk_node)) {
    // TODO: this will not hold in the batch case as the first dynamic point we
    // get will not be the first point on the object (we will get the first
    // point seen within the window) so, where should be initalise the object
    // pose!?
    //  //this is a totally new tracklet so should be the first time we've seen
    //  it! CHECK_EQ(lmk_node->getFirstSeenFrame(), frame_node_k_1->getId());

    // mark as now in map
    is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);
    CHECK(isDynamicTrackletInMap(lmk_node));

    // use first point as initalisation?
    // in this case k is k-1 as we use frame_node_k_1
    gtsam::Pose3 s0_H_k = computeInitialHFromFrontend(context.getObjectId(),
                                                      frame_node_k_1->getId());
    LOG(INFO) << "s0_H_k " << s0_H_k;
    // measured point in camera frame
    const gtsam::Point3 m_camera =
        lmk_node->getMeasurement(frame_node_k_1).landmark;
    Landmark lmk_L0_init =
        L_0.inverse() * s0_H_k.inverse() * context.X_k_1_measured * m_camera;

    // initalise value //cannot initalise again the same -> it depends where L_0
    // is created, no?
    //  Landmark lmk_L0;
    //  getSafeQuery(
    //      lmk_L0,
    //      theta_accessor->query<Landmark>(point_key),
    //      lmk_L0_init
    //  );
    new_values.insert(point_key, lmk_L0_init);
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
  }

  auto dynamic_point_noise = noise_models_.dynamic_point_noise;
  if (context.is_starting_motion_frame) {
    // add factor at k-1
    new_factors.emplace_shared<ObjectCentricMotionFactor>(
        frame_node_k_1->makePoseKey(),  // pose key at previous frames,
        object_motion_key_k_1, point_key,
        lmk_node->getMeasurement(frame_node_k_1).landmark, L_0,
        landmark_motion_noise);
    // const gtsam::Pose3 X_world =
    //     getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
    // new_factors.emplace_shared<ObjectCentricMotionOnlyFactor>(
    //     object_motion_key_k_1, point_key,
    //     lmk_node->getMeasurement(frame_node_k_1).landmark, X_world, L_0,
    //     landmark_motion_noise);

    // new_factors.emplace_shared<AuxillaryPFactor>(
    //      object_motion_key_k_1, point_key,
    //      lmk_node->getMeasurement(frame_node_k_1).landmark, L_0,
    //     landmark_motion_noise
    // );
    // new_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,
    // Landmark>>(
    //           frame_node_k_1->makePoseKey(),  // pose key for this frame
    //           point_key, lmk_node->getMeasurement(frame_node_k_1).landmark,
    //           dynamic_point_noise);

    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
  }

  // add factor at k
  new_factors.emplace_shared<ObjectCentricMotionFactor>(
      frame_node_k->makePoseKey(),  // pose key at previous frames,
      object_motion_key_k, point_key,
      lmk_node->getMeasurement(frame_node_k).landmark, L_0,
      landmark_motion_noise);

  // const gtsam::Pose3 X_world =
  //     getInitialOrLinearizedSensorPose(frame_node_k->frame_id);
  // new_factors.emplace_shared<ObjectCentricMotionOnlyFactor>(
  //     object_motion_key_k, point_key,
  //     lmk_node->getMeasurement(frame_node_k).landmark, X_world, L_0,
  //     landmark_motion_noise);

  // new_factors.emplace_shared<AuxillaryPFactor>(
  //        object_motion_key_k, point_key,
  //        lmk_node->getMeasurement(frame_node_k).landmark, L_0,
  //       landmark_motion_noise
  //   );

  //  new_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,
  //  Landmark>>(
  //             frame_node_k->makePoseKey(),  // pose key for this frame
  //             point_key, lmk_node->getMeasurement(frame_node_k).landmark,
  //             dynamic_point_noise);

  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());

  // assume we've added both!!
  //  TODO: NEW STUFF AND TESTG!!!
  //  new_factors.emplace_shared<AuxillaryTransformFactor>(
  //    frame_node_k_1->makePoseKey(),
  //    object_motion_key_k_1,
  //    frame_node_k->makePoseKey(),
  //    object_motion_key_k,
  //    lmk_node->getMeasurement(frame_node_k_1).landmark,
  //    lmk_node->getMeasurement(frame_node_k).landmark,
  //    landmark_motion_noise
  //  );

  // new_factors.emplace_shared<SmartHFactor>(
  //   frame_node_k_1->makePoseKey(),
  //   object_motion_key_k_1,
  //   frame_node_k->makePoseKey(),
  //   object_motion_key_k,
  //   lmk_node->getMeasurement(frame_node_k_1).landmark,
  //   lmk_node->getMeasurement(frame_node_k).landmark,
  //   landmark_motion_noise
  // );
}

void ObjectCentricFormulation::objectUpdateContext(
    const ObjectUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  auto frame_node_k = context.frame_node_k;
  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());

  auto theta_accessor = this->accessorFromTheta();
  const auto frame_id = context.getFrameId();
  const auto object_id = context.getObjectId();

  if (!is_other_values_in_map.exists(object_motion_key_k)) {
    // gtsam::Pose3 motion;
    const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(frame_id);
    gtsam::Pose3 motion = computeInitialHFromFrontend(object_id, frame_id);
    LOG(INFO) << "Added motion at  "
              << DynoLikeKeyFormatter(object_motion_key_k);
    // gtsam::Pose3 motion;
    new_values.insert(object_motion_key_k, motion);
    is_other_values_in_map.insert2(object_motion_key_k, true);

    FrameId s0 = getL0(object_id, frame_id).first;
    if (s0 == frame_id) {
      // add prior
      //  new_factors.addPrior<gtsam::Pose3>(object_motion_key_k,
      //  gtsam::Pose3::Identity());
    }
  }

  if (frame_id < 2) return;

  auto frame_node_k_1 = map()->getFrame(frame_id - 1u);
  if (!frame_node_k_1) {
    return;
  }

  if (FLAGS_use_smoothing_factor && frame_node_k_1->objectObserved(object_id)) {
    // motion key at previous frame
    const gtsam::Symbol object_motion_key_k_1 =
        frame_node_k_1->makeObjectMotionKey(object_id);

    auto object_smoothing_noise = noise_models_.object_smoothing_noise;
    CHECK(object_smoothing_noise);
    CHECK_EQ(object_smoothing_noise->dim(), 6u);

    {
      ObjectId object_label_k_1, object_label_k;
      FrameId frame_id_k_1, frame_id_k;
      CHECK(reconstructMotionInfo(object_motion_key_k_1, object_label_k_1,
                                  frame_id_k_1));
      CHECK(reconstructMotionInfo(object_motion_key_k, object_label_k,
                                  frame_id_k));
      CHECK_EQ(object_label_k_1, object_label_k);
      CHECK_EQ(frame_id_k_1 + 1, frame_id_k);  // assumes
      // consequative frames
    }

    // if the motion key at k (motion from k-1 to k), and key at k-1 (motion
    //  from k-2 to k-1)
    // exists in the map or is about to exist via new values, add the
    //  smoothing factor
    if (is_other_values_in_map.exists(object_motion_key_k_1) &&
        is_other_values_in_map.exists(object_motion_key_k)) {
      new_factors.emplace_shared<ObjectCentricSmoothing>(
          object_motion_key_k_1, object_motion_key_k,
          getL0(object_id, frame_id).second, object_smoothing_noise);
      // new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      //     object_motion_key_k_1, object_motion_key_k,
      //     gtsam::Pose3::Identity(), object_smoothing_noise);
      // if(result.debug_info)
      // result.debug_info->getObjectInfo(context.getObjectId()).smoothing_factor_added
      // = true;
    }

    //   // // if(smoothing_added) {
    //   // //     //TODO: add back in
    //   // //     // object_debug_info.smoothing_factor_added = true;
    //   // // }
  }

  // //TODO: for now ignore the k-1'th frame on the first frame
  // if(!context.has_motion_pair) {
  //     return;
  // }
  // // return;

  // auto frame_node_k = context.frame_node_k;
  // const auto frame_id_k = context.getFrameId();
  // const auto object_id = context.getObjectId();

  // const gtsam::Key object_motion_key_k =
  // frame_node_k->makeObjectMotionKey(context.getObjectId()); auto
  // theta_accessor = this->accessorFromTheta();

  // FrameId s0; //first frame of objects trajectory
  // gtsam::Pose3 L0; //the fixed object frame in the world
  // std::tie(s0, L0) = getL0(object_id, frame_id_k);

  // if(context.has_motion_pair) {
  //     //attempt to init at k-1
  //     if(!is_other_values_in_map.exists(object_motion_key_k)) {
  //         // gtsam::Pose3 motion;
  //         // gtsam::Pose3 motion;
  //         gtsam::Pose3 motion =
  //         computeInitialHFromFrontend(context.getObjectId(),
  //         context.getFrameId()); new_values.insert(object_motion_key_k,
  //         motion); is_other_values_in_map.insert2(object_motion_key_k, true);
  //     }
  // }

  // //add value motion from s to k (if s==k, should be identity!)
  //  if(!is_other_values_in_map.exists(object_motion_key_k)) {
  //     // gtsam::Pose3 motion;
  //     // gtsam::Pose3 motion;
  //     gtsam::Pose3 motion =
  //     computeInitialHFromFrontend(context.getObjectId(),
  //     context.getFrameId()); new_values.insert(object_motion_key_k, motion);
  //     is_other_values_in_map.insert2(object_motion_key_k, true);
  // }

  // //sanity check
  // CHECK(point_contexts_.exists(object_id));
  // for (const PointUpdateContextType& point_context :
  // point_contexts_.at(object_id)) {
  //     const auto context_frame_node_k = point_context.frame_node_k;
  //     //colleced point contexts are at the right frame id
  //     CHECK_EQ(context_frame_node_k->getId(), frame_id_k);
  //     //collected points are on the right object
  //     CHECK_EQ(point_context.getObjectId(), object_id);

  //     const auto lmk_node = point_context.lmk_node;
  //     gtsam::Key point_key =
  //     this->makeDynamicKey(point_context.getTrackletId());

  //     //add points

  //     //ie new point
  //     if(!isDynamicTrackletInMap(lmk_node)) {
  //         is_dynamic_tracklet_in_map_.insert2(point_context.getTrackletId(),
  //         true); CHECK(isDynamicTrackletInMap(lmk_node));

  //         gtsam::Pose3 s0_H_k =
  //         computeInitialHFromFrontend(context.getObjectId(),
  //         context_frame_node_k->getId());
  //         //measured point in camera frame
  //         const gtsam::Point3 m_camera =
  //         lmk_node->getMeasurement(context_frame_node_k).landmark; Landmark
  //         lmk_L0_init = L0.inverse() * s0_H_k.inverse() *
  //         point_context.X_k_measured * m_camera;

  //         new_values.insert(point_key, lmk_L0_init);
  //         if(result.debug_info)
  //         result.debug_info->getObjectInfo(point_context.getObjectId()).num_new_dynamic_points++;

  //         //TODO: update result?
  //     }

  //     //add factor
  //     auto landmark_motion_noise = noise_models_.landmark_motion_noise;

  //     new_factors.emplace_shared<ObjectCentricMotionFactor>(
  //         context_frame_node_k->makePoseKey(), //pose key at previous frames,
  //         object_motion_key_k,
  //         point_key,
  //         lmk_node->getMeasurement(context_frame_node_k).landmark,
  //         L0,
  //         landmark_motion_noise
  //     );
  //     result.updateAffectedObject(context_frame_node_k->frame_id,
  //     point_context.getObjectId()); if(result.debug_info)
  //     result.debug_info->getObjectInfo(point_context.getObjectId()).num_motion_factors++;

  // }

  // clear point context for this object id
  point_contexts_.erase(object_id);
}

std::pair<FrameId, gtsam::Pose3> ObjectCentricFormulation::getL0(
    ObjectId object_id, FrameId frame_id) {
  if (L0_.exists(object_id)) {
    return L0_.at(object_id);
  }

  // else initalise from centroid?
  auto object_node = map()->getObject(object_id);
  CHECK(object_node);

  auto frame_node = map()->getFrame(frame_id);
  CHECK(frame_node);
  CHECK(frame_node->objectObserved(object_id));

  StatusLandmarkEstimates dynamic_landmarks;

  // measured/linearized camera pose at the first frame this object has been
  // seen
  const gtsam::Pose3 X_world = getInitialOrLinearizedSensorPose(frame_id);
  auto measurement_pairs = frame_node->getDynamicMeasurements(object_id);

  for (const auto& [lmk_node, measurement] : measurement_pairs) {
    CHECK(lmk_node->seenAtFrame(frame_id));
    CHECK_EQ(lmk_node->object_id, object_id);

    const gtsam::Point3 landmark_measurement_local = measurement.landmark;
    // const gtsam::Point3 landmark_measurement_world = X_world *
    // landmark_measurement_local;

    dynamic_landmarks.push_back(LandmarkStatus::DynamicInGLobal(
        landmark_measurement_local, frame_id, lmk_node->tracklet_id, object_id,
        LandmarkStatus::Method::MEASURED));
  }

  CloudPerObject object_clouds = groupObjectCloud(dynamic_landmarks, X_world);
  CHECK_EQ(object_clouds.size(), 1u);

  CHECK(object_clouds.exists(object_id));

  const auto dynamic_point_cloud = object_clouds.at(object_id);
  pcl::PointXYZ centroid;
  pcl::computeCentroid(dynamic_point_cloud, centroid);
  // TODO: outlier reject?
  gtsam::Point3 translation = pclPointToGtsam(centroid);
  gtsam::Pose3 center(gtsam::Rot3::Identity(), X_world * translation);

  // frame id should coincide with the first seen frame of this object...?
  L0_.insert2(object_id, std::make_pair(frame_id, center));
  return L0_.at(object_id);
}

gtsam::Pose3 ObjectCentricFormulation::computeInitialHFromFrontend(
    ObjectId object_id, FrameId frame_id) {
  gtsam::Pose3 L_0;
  FrameId s0;
  std::tie(s0, L_0) = getL0(object_id, frame_id);

  CHECK_LE(s0, frame_id);
  if (frame_id == s0) {
    // same frame so motion between them should be identity!
    // except for rotation?
    return gtsam::Pose3::Identity();
  }
  if (frame_id - 1 == s0) {
    // a motion that takes us from k-1 to k where k-1 == s0
    Motion3 motion;
    CHECK(map()->hasInitialObjectMotion(frame_id, object_id, &motion));
    return motion;
  } else {
    Motion3 composed_motion;

    // LOG(INFO) << "Computing initial motion from " << s0 << " to " <<
    // frame_id;

    // query from so+1 to k since we index backwards
    for (auto frame = s0 + 1; frame <= frame_id; frame++) {
      Motion3 motion;  // if fail just use identity?
      if (!map()->hasInitialObjectMotion(frame, object_id, &motion)) {
        LOG(INFO) << "No frontend motion at frame " << frame << " object id "
                  << object_id;
      }

      composed_motion = motion * composed_motion;
    }
    // after loop motion should be ^w_{s0}H_k
    return composed_motion;
  }
}

}  // namespace dyno
