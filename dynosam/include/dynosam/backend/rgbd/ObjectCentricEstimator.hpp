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
#pragma once

#include <gtsam/base/numericalDerivative.h>

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/common/StructuredContainers.hpp"  //for FrameRange
#include "dynosam/common/Types.hpp"                 //only needed for factors

namespace dyno {

template <size_t DIM, typename MOTION, typename POSE>
struct MotionFactorTraits {
  static constexpr size_t ZDim = DIM;
  using Motion = MOTION;
  using Pose = POSE;

  using Motions = std::vector<Motion, Eigen::aligned_allocator<Motion>>;
  using Poses = std::vector<Pose, Eigen::aligned_allocator<Pose>>;

  //! Measurement dimension
  using Z = Eigen::Matrix<double, ZDim, 1>;
  using ZVector = std::vector<Z, Eigen::aligned_allocator<Z>>;

  //! Motion dimension
  static constexpr size_t HDim =
      static_cast<size_t>(gtsam::traits<Motion>::dimension);
  //! Pose dimension
  static constexpr size_t XDim =
      static_cast<size_t>(gtsam::traits<Pose>::dimension);
  static constexpr size_t HessianDim = HDim + XDim;
  //! G blocks (derivatives wrt motion, H)
  using MatrixGD = Eigen::Matrix<double, ZDim, HDim>;
  //! F blocks (derivatives wrt pose, X)
  using MatrixFD = Eigen::Matrix<double, ZDim, XDim>;
  //! E blocks (derivatives wrt point, m)
  using MatrixED = Eigen::Matrix<double, ZDim, ZDim>;
  //! Combined GF blocks for Schur compliment
  using MatrixGFD = Eigen::Matrix<double, ZDim, HessianDim>;

  using GBlocks = std::vector<MatrixGD, Eigen::aligned_allocator<MatrixGD>>;
  using FBlocks = std::vector<MatrixFD, Eigen::aligned_allocator<MatrixFD>>;
  using EBlocks = std::vector<MatrixED, Eigen::aligned_allocator<MatrixED>>;
  using GFBlocks = std::vector<MatrixGFD, Eigen::aligned_allocator<MatrixGFD>>;
};

struct SmartMotionFactorParams {
  //! Threshold to decide whether to re-triangulate
  double retriangulation_threshold = 1e-5;
};

// should be object centric smart factor... :)
template <size_t DIM = 3u, typename MOTION = gtsam::Pose3,
          typename POSE = gtsam::Pose3>
class SmartMotionFactor : public gtsam::NonlinearFactor,
                          MotionFactorTraits<DIM, MOTION, POSE> {
 public:
  using Base = gtsam::NonlinearFactor;
  using This = SmartMotionFactor<DIM, MOTION, POSE>;
  using MotionTraits = MotionFactorTraits<DIM, MOTION, POSE>;
  using shared_ptr = boost::shared_ptr<This>;

  static constexpr size_t ZDim = MotionTraits::ZDim;
  static constexpr size_t HDim = MotionTraits::HDim;
  static constexpr size_t XDim = MotionTraits::XDim;

  using typename MotionTraits::Z;
  using typename MotionTraits::ZVector;

  using typename MotionTraits::Motion;
  using typename MotionTraits::Pose;

  using typename MotionTraits::Motions;
  using typename MotionTraits::Poses;

  using typename MotionTraits::MatrixED;
  using typename MotionTraits::MatrixFD;
  using typename MotionTraits::MatrixGD;
  using typename MotionTraits::MatrixGFD;

  using typename MotionTraits::EBlocks;
  using typename MotionTraits::FBlocks;
  using typename MotionTraits::GBlocks;
  using typename MotionTraits::GFBlocks;

  static constexpr int HessianDim = MotionTraits::HessianDim;

  SmartMotionFactor(const gtsam::Pose3& L_s0,
                    const gtsam::SharedNoiseModel& noise_model,
                    std::optional<gtsam::Point3> initial_point_l = {})
      : Base(),
        L_s0_(L_s0),
        noise_model_(noise_model),
        result_(initial_point_l) {}
  ~SmartMotionFactor() {}

 public:
  /// Return the dimension (number of rows!) of the factor.
  size_t dim() const override { return ZDim * this->measured_.size(); }

  void add(const Z& measured, const gtsam::Key& motion_key,
           const gtsam::Key& pose_key) {
    if (std::find(keys_.begin(), keys_.end(), motion_key) != keys_.end()) {
      throw std::invalid_argument(
          "SmartMotionFactor::add: adding duplicate measurement for motion "
          "key.");
    }
    if (std::find(keys_.begin(), keys_.end(), pose_key) != keys_.end()) {
      throw std::invalid_argument(
          "SmartMotionFactor::add: adding duplicate measurement for pose key.");
    }

    this->measured_.push_back(measured);

    this->keys_.push_back(motion_key);
    this->keys_.push_back(pose_key);
    this->motion_keys_.push_back(motion_key);
    this->pose_keys_.push_back(pose_key);
  }

  /// Return the 2D measurements (ZDim, in general).
  const ZVector& measured() const { return measured_; }

  size_t numMeasurements() const { return this->measured_.size(); }

  // these are no longer in Base::key order!! does this matter?!
  Motions motions(const gtsam::Values& values) const {
    Motions motions;
    for (const auto& k : motion_keys_) {
      motions.push_back(values.at<Motion>(k));
    }
    return motions;
  }

  Poses poses(const gtsam::Values& values) const {
    Poses poses;
    for (const auto& k : pose_keys_) {
      poses.push_back(values.at<Pose>(k));
    }
    return poses;
  }

  double totalReprojectionError(
      const Motions& motions, const Poses& poses,
      std::optional<gtsam::Point3> externalPoint = {}) const {
    // if (externalPoint)
    //   result_ = TriangulationResult(*externalPoint);
    // else
    //   result_ = triangulateSafe(cameras);

    // if (result_)
    //   // All good, just use version in base class
    //   return Base::totalReprojectionError(cameras, *result_);
    // else if (params_.degeneracyMode == HANDLE_INFINITY) {
    //   // Otherwise, manage the exceptions with rotation-only factors
    //   Unit3 backprojected = cameras.front().backprojectPointAtInfinity(
    //       this->measured_.at(0));
    //   return Base::totalReprojectionError(cameras, backprojected);
    // } else
    //   // if we don't want to manage the exceptions we discard the factor
    //   return 0.0;
  }
  /**
   * Calculate the error of the factor.
   * This is the log-likelihood, e.g. \f$ 0.5(h(x)-z)^2/\sigma^2 \f$ in case of
   * Gaussian. In this class, we take the raw prediction error \f$ h(x)-z \f$,
   * ask the noise model to transform it to \f$ (h(x)-z)^2/\sigma^2 \f$, and
   * then multiply by 0.5. Will be used in "error(Values)" function required by
   * NonlinearFactor base class
   */
  double totalReprojectionError(const Motions& motions, const Poses& poses,
                                const gtsam::Point3& point_l) const {
    gtsam::Vector error = whitenedError(motions, poses, point_l);
    return 0.5 * error.dot(error);
  }

  /**
   * Calculate vector of re-projection errors [h(x)-z] = [cameras.project(p) -
   * z], with the noise model applied.
   */
  gtsam::Vector whitenedError(const Motions& motions, const Poses& poses,
                              const gtsam::Point3& point_l) const {
    gtsam::Vector error = unwhitenedError(motions, poses, point_l);
    if (noise_model_) noise_model_->whitenInPlace(error);
    return error;
  }

  gtsam::Vector unwhitenedError(const Motions& motions, const Poses& poses,
                                const gtsam::Point3& point_l,
                                GBlocks* Gs = nullptr, FBlocks* Fs = nullptr,
                                EBlocks* Es = nullptr) const {
    return This::reprojectionError(motions, poses, point_l, Gs, Fs, Es);
  }

  // TODO: unhitened error?
  // template <class... OptArgs,
  //           typename = std::enable_if_t<sizeof...(OptArgs) != 0>>
  // gtsam::Vector unwhitenedError(const Motions& motions, const Poses& poses,
  //                               const gtsam::Point3& point_l,
  //                               OptArgs&&... optArgs) const {
  //   return unwhitenedError(motions, poses, point_l, (&optArgs)...);
  // }

  // TODO:computeJacobians

  // template<>
  // gtsam::Vector reprojectionError<3>(const Motions& motions, const Poses&
  // poses, const gtsam::Point3& point_w) const {
  //   CHECK_EQ(motions.size(), poses.size());
  // }

  // TODO: clean up and provide better call structure
  // see SmartProjectionFactor where external point can be used to recalculate
  // the internal result?
  gtsam::Vector reprojectionError(const gtsam::Values& values) const {
    return reprojectionError(motions(values), poses(values));
  }

  template <class... OptArgs,
            typename = std::enable_if_t<sizeof...(OptArgs) != 0>>
  gtsam::Vector reprojectionError(const gtsam::Values& values,
                                  OptArgs&&... optArgs) const {
    return reprojectionError(motions(values), poses(values), (&optArgs)...);
  }

  template <class... OptArgs,
            typename = std::enable_if_t<sizeof...(OptArgs) != 0>>
  gtsam::Vector reprojectionError(const Motions& motions, const Poses& poses,
                                  OptArgs&&... optArgs) const {
    return reprojectionError(motions, poses, (&optArgs)...);
  }

  gtsam::Vector reprojectionError(const Motions& motions, const Poses& poses,
                                  GBlocks* Gs = nullptr, FBlocks* Fs = nullptr,
                                  EBlocks* Es = nullptr) const {
    if (result_) {
      return this->reprojectionError(motions, poses, *result_, Gs, Fs, Es);
    } else {
      throw std::runtime_error("Result not computed!");
    }
  }

  double error(const gtsam::Values& c) const override {
    if (this->active(c)) {
      return totalReprojectionError(motions(c), poses(c));
    } else {  // else of active flag
      return 0.0;
    }
    return 0;
  }

  //   std::shared_ptr<gtsam::GaussianFactor> linearize(
  //       const Values& c) const override {
  //     std::vector<gtsam::Matrix> A(size());

  //     Vector b = -unwhitenedError(x, A);
  //     // check(noiseModel_, b.size());

  //     // Whiten the corresponding system now
  //     if (noiseModel_) noiseModel_->WhitenSystem(A, b);

  //     // Fill in terms, needed to create JacobianFactor below
  //     std::vector<std::pair<Key, Matrix>> terms(size());
  //     for (size_t j = 0; j < size(); ++j) {
  //       terms[j].first = keys()[j];
  //       terms[j].second.swap(A[j]);
  //     }

  //     // TODO pass unwhitened + noise model to Gaussian factor
  //     using noiseModel::Constrained;
  //     if (noiseModel_ && noiseModel_->isConstrained())
  //       return GaussianFactor::shared_ptr(new JacobianFactor(
  //           terms, b,
  //           std::static_pointer_cast<Constrained>(noiseModel_)->unit()));
  //     else {
  //       return GaussianFactor::shared_ptr(new JacobianFactor(terms, b));
  //     }
  //   }

  boost::shared_ptr<gtsam::GaussianFactor> linearize(
      const gtsam::Values& c) const override {
    // TODo: when to retriangulate point?
    // linearizeDamped(values)
    return createHessianFactor(motions(c), poses(c));
  }

  gtsam::SymmetricBlockMatrix createReducedMatrix(
      const gtsam::Values& c) const {
    return createReducedMatrix(motions(c), poses(c));
  }

  gtsam::SymmetricBlockMatrix createReducedMatrix(
      const Motions& motions, const Poses& poses, const double lambda = 0.0,
      bool diagonalDamping = false) const {
    // trinagulate safe -> get point which is used when getting the Jacobians
    // TODO: can we use the LOST values etc by making the cam pose: G = X.inv()
    // * H like we do for reprojection RANSAC? no degeneracy as we are 3D?

    GBlocks Gs;
    FBlocks Fs;
    EBlocks Es;
    gtsam::Vector b;

    // compute G, F, E and b blocks
    computeJacobiansWithTriangulatedPoint(motions, poses, Gs, Fs, Es, b);
    // TODO: add back in will fail tests!!!
    //  whitenJacobians(Gs, Fs, Es, b);

    const size_t m = numMeasurements();
    CHECK_EQ(m, Es.size());

    gtsam::Matrix E;
    EVectorToMatrix(Es, E);

    gtsam::print(b, "b term Es ");
    gtsam::print(E, "stacked Es ");

    // whiten Jacobians (how does this differ now we have 3?)
    // check difference between WhitenSystem and Whiten

    // do shuur compliment to get augmented Jacobian
    // no damping
    gtsam::Matrix EtE = E.transpose() * E;
    const Eigen::Matrix<double, N, N> P = (EtE).inverse();

    GFBlocks GFs;
    GFVectorsToGFBlocks(Gs, Fs, GFs);

    gtsam::SymmetricBlockMatrix augmentedHessian =
        SchurComplement(GFs, E, P, b);
    // how on earth to test

    // LOG(INFO) << augmentedHessian.full();
    // gtsam::Matrix view = augmentedHessian.selfadjointView();
    // LOG(INFO) << view;
    // LOG(INFO) << "Done";

    return augmentedHessian;
  }

  boost::shared_ptr<gtsam::RegularHessianFactor<HDim>> createHessianFactor(
      const Motions& motions, const Poses& poses, const double lambda = 0.0,
      bool diagonalDamping = false) const {
    const size_t m = this->numMeasurements();
    constexpr static auto HessianDim = MotionTraits::HessianDim;

    // if (params_.degeneracyMode == ZERO_ON_DEGENERACY && !result_) {
    if (!result_) {
      LOG(FATAL) << "Shoudl not get here";
      // gtsam::Matrix b = gtsam::Matrix::Zero(ZDim * num_measurements, 1);
      // gtsam::Matrix g = gtsam::Matrix::Zero(HessianDim * num_measurements,
      // 1); gtsam::Matrix G = gtsam::Matrix::Zero(HessianDim *
      // num_measurements, HessianDim * num_measurements);

      // gtsam::Matrix augmented_information(HessianDim * num_measurements + 1,
      // HessianDim * num_measurements + 1);
      gtsam::Matrix augmented_information =
          gtsam::Matrix::Zero(HessianDim * m + 1, HessianDim * m + 1);
      // augmented_information << G, g, g.transpose(), b.squaredNorm();

      return boost::make_shared<gtsam::RegularHessianFactor<HDim>>(
          this->keys_, constructSymmetricBlockMatrix(m, augmented_information));

    }

    else {
      gtsam::SymmetricBlockMatrix augmented_hessian =
          createReducedMatrix(motions, poses, lambda, diagonalDamping);
      return boost::make_shared<gtsam::RegularHessianFactor<HDim>>(
          this->keys_, augmented_hessian);
    }
  }

  //  private:
  void computeJacobiansWithTriangulatedPoint(const Motions& motions,
                                             const Poses& poses, GBlocks& Gs,
                                             FBlocks& Fs, EBlocks& Es,
                                             gtsam::Vector& b) const {
    if (result_) {
      computeJacobians(motions, poses, *result_, Gs, Fs, Es, b);
    } else {
      throw std::runtime_error("Result not computed!");
    }
  }

  void computeJacobians(const Motions& motions, const Poses& poses,
                        const gtsam::Point3& point_l, GBlocks& Gs, FBlocks& Fs,
                        EBlocks& Es, gtsam::Vector& b) const {
    b = -unwhitenedError(motions, poses, point_l, &Gs, &Fs, &Es);

    CHECK_EQ(motions.size(), Gs.size());
    CHECK_EQ(poses.size(), Fs.size());
  }

  void whitenJacobians(GBlocks& Gs, FBlocks& Fs, EBlocks& Es,
                       gtsam::Vector& b) const {
    CHECK(noise_model_);

    for (size_t i = 0; i < Gs.size(); i++) Gs[i] = noise_model_->Whiten(Gs[i]);
    for (size_t i = 0; i < Fs.size(); i++) Fs[i] = noise_model_->Whiten(Fs[i]);
    for (size_t i = 0; i < Es.size(); i++) Es[i] = noise_model_->Whiten(Es[i]);

    b = noise_model_->whiten(b);
  }

  std::optional<gtsam::Point3> triangulateSafe(const Motions& motions,
                                               const Poses& poses) const {
    if (numMeasurements() < 2) {
      return {};
    }

    // what if result is {}?
    bool retriangulate = decideIfTriangulate(motions, poses);
    if (retriangulate) {
    }
  }

  // template <size_t N = ZDim, typename = std::enable_if_t<N==3>>
  // gtsam::Vector reprojectionError(const Motions& motions, const Poses& poses,
  // const gtsam::Point3& point_l, GBlocks* Gs, FBlocks* Fs, EBlocks* Es) const
  // {

  // }

  // template <size_t N = ZDim, typename = std::enable_if_t<N==2>>
  // gtsam::Vector reprojectionError(const Motions& motions, const Poses& poses,
  // const gtsam::Point3& point_l, GBlocks* Gs, FBlocks* Fs, EBlocks* Es) const
  // { throw std::runtime_error("Not implemented N==2!!"); }

  gtsam::Vector reprojectionError(const Motions& motions, const Poses& poses,
                                  const gtsam::Point3& point_l,
                                  GBlocks* Gs = nullptr, FBlocks* Fs = nullptr,
                                  EBlocks* Es = nullptr) const {
    CHECK_EQ(motions.size(), poses.size());
    const auto measurements = measured();
    CHECK_EQ(motions.size(), measurements.size());

    const size_t m = measurements.size();
    // expected z, ie h(x)
    ZVector z;
    z.reserve(m);

    // Allocate derivatives
    if (Es) Es->resize(m);
    if (Fs) Fs->resize(m);
    if (Gs) Gs->resize(m);

    // Project and fill derivatives
    for (size_t i = 0; i < m; i++) {
      MatrixGD Gi;
      MatrixFD Fi;
      MatrixED Ei;

      // given the point in the object frame and associated camera pose and
      // motion, project the point into the camera frame
      auto project = [&](const Motion& motioni, const Pose& posei,
                         const gtsam::Point3& point_l) -> Z {
        const gtsam::Point3 map_point_world = motioni * (L_s0_ * point_l);
        return posei.inverse() * map_point_world;
      };

      Gi = gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Pose3,
                                        gtsam::Pose3, gtsam::Point3>(
          project, motions.at(i), poses.at(i), point_l);

      Fi = gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Pose3,
                                        gtsam::Pose3, gtsam::Point3>(
          project, motions.at(i), poses.at(i), point_l);

      Ei = gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Pose3,
                                        gtsam::Pose3, gtsam::Point3>(
          project, motions.at(i), poses.at(i), point_l);

      z.emplace_back(project(motions.at(i), poses.at(i), point_l));
      if (Gs) (*Gs)[i] = Gi;
      if (Fs) (*Fs)[i] = Fi;
      if (Es) (*Es)[i] = Ei;
      // if (E) E->block<ZDim, N>(ZDim * i, 0) = Ei;
    }

    // Fill Error factor
    gtsam::Vector b(ZDim * m);
    CHECK_EQ(m, z.size());
    for (size_t i = 0, row = 0; i < m; i++, row += ZDim) {
      gtsam::Vector bi = gtsam::traits<Z>::Local(measurements[i], z[i]);
      // if (ZDim == 3 && std::isnan(bi(1))) {  // if it is a stereo point and
      // the
      //                                       // right pixel is missing (nan)
      //   bi(1) = 0;
      // }
      b.segment<ZDim>(row) = bi;
    }
    return b;
  }

  bool decideIfTriangulate(const Motions& motions, const Poses& poses) const {
    // Several calls to linearize will be done from the same linearization
    // point, hence it is not needed to re-triangulate. Note that this is not
    // yet "selecting linearization", that will come later, and we only check if
    // the current linearization is the "same" (up to tolerance) w.r.t. the last
    // time we triangulated the point.
    CHECK_EQ(motions.size(), poses.size());
    const auto measurements = measured();
    CHECK_EQ(motions.size(), measurements.size());

    const size_t m = measurements.size();

    // sanity check: this should never fail!
    CHECK_EQ(camera_poses_triangulation_.size(),
             object_motions_triangulation_.size());

    bool retriangulate = false;

    // Definitely true if we do not have a previous linearization point or the
    // new linearization point includes more poses.
    // Only check against poses since we know motions and poses have the same
    // size and the cached poses/motions should have the same size
    if (camera_poses_triangulation_.empty() ||
        poses.size() != camera_poses_triangulation_.size())
      retriangulate = true;

    // Otherwise, check poses and motions against cache.
    if (!retriangulate) {
      for (size_t i = 0; i < poses.size(); i++) {
        if (!poses[i].equals(camera_poses_triangulation_[i],
                             params_.retriangulation_threshold) ||
            !motions[i].equals(object_motions_triangulation_[i],
                               params_.retriangulation_threshold)) {
          retriangulate =
              true;  // at least two poses are different, hence we retriangulate
          break;
        }
      }
    }

    // Store the current poses used for triangulation if we will re-triangulate.
    if (retriangulate) {
      camera_poses_triangulation_.clear();
      camera_poses_triangulation_.reserve(m);

      object_motions_triangulation_.clear();
      object_motions_triangulation_.reserve(m);
      for (size_t i = 0; i < m; i++) {
        camera_poses_triangulation_.push_back(poses[i]);
        object_motions_triangulation_.push_back(motions[i]);
      }
    }

    return retriangulate;
  }

  // TODO: should actually check that Es is the size of the measurements
  // Pull into base class to have static and non static versions
  static void EVectorToMatrix(const EBlocks& Es, gtsam::Matrix& E) {
    size_t m = Es.size();

    E.resize(ZDim * m, ZDim);
    for (size_t i = 0; i < m; i++) {
      const MatrixED& Ei = Es.at(i);
      E.block<ZDim, ZDim>(ZDim * i, 0) = Ei;
    }
  }

  static void GFVectorsToGFBlocks(const GBlocks& Gs, const FBlocks& Fs,
                                  GFBlocks& GFs) {
    CHECK_EQ(Gs.size(), Fs.size());

    GFs.resize(Fs.size());

    for (size_t i = 0; i < Fs.size(); i++) {
      const MatrixGD& Gi = Gs.at(i);
      const MatrixFD& Fi = Fs.at(i);
      MatrixGFD GFi;

      // LOG(INFO) << "Gi:\n " << Gi;
      // LOG(INFO) << "Fi:\n " << Fi;
      // gtsam::print(Fi, "Fi ");

      // concat blocks horizontally
      GFi.block(0, 0, ZDim, HDim) = Gi;
      GFi.block(0, HDim, ZDim, XDim) = Fi;
      GFs[i] = GFi;

      // LOG(INFO) << "GFs[i]:\n " << GFs[i];

      // gtsam::print(GFs[i], "GFs[i] ");
    }
  }

  static constexpr int N = 3;  // point size //only used for schur compliment?

  static gtsam::SymmetricBlockMatrix constructSymmetricBlockMatrix(
      size_t m, std::optional<gtsam::Matrix> M = {}) {
    // Create a SymmetricBlockMatrix (augmented hessian, with extra row/column
    // with info vector)
    size_t M1 = HessianDim * m + 1;
    std::vector<Eigen::DenseIndex> dims(m +
                                        1);  // this also includes the b term
    // vertical dimensions of ATA
    std::fill(dims.begin(), dims.end() - 1, HessianDim);
    dims.back() = 1;

    if (M) {
      // check sizes
      if (M->rows() != M1 || M->cols() != M1) {
        std::stringstream ss;
        ss << "Symmetric Block Matrix construction failed: input"
              " augmented hessian has incorrect size ("
           << M->rows() << " x " << M->cols()
           << ")"
              " when it should be a sqaure ("
           << M1 << " x " << M1 << ")";
        throw std::runtime_error(ss.str());
      } else {
        return gtsam::SymmetricBlockMatrix(dims, *M);
      }
    }

    return gtsam::SymmetricBlockMatrix(dims, gtsam::Matrix::Zero(M1, M1));
  }

  // static gtsam::SymmetricBlockMatrix SchurComplement(
  //     const GFBlocks& GFs, const gtsam::Matrix& E,
  //     const Eigen::Matrix<double, N, N>& P, const gtsam::Vector& b) {
  //   // a single point is observed in m cameras
  //   size_t m = GFs.size();
  //   gtsam::SymmetricBlockMatrix augmentedHessian =
  //       constructSymmetricBlockMatrix(m);

  //   // Blockwise Schur complement
  //   for (size_t i = 0; i < m; i++) {  // for each camera

  //     const MatrixGFD& GFi = GFs[i];
  //     const auto GFiT = GFi.transpose();
  //     const MatrixED Ei_P =  //
  //         E.block(ZDim * i, 0, ZDim, N) * P;

  //     // D = (Dx2) * ZDim
  //     augmentedHessian.setOffDiagonalBlock(
  //         i, m,
  //         GFiT * b.segment<ZDim>(ZDim * i)  // F' * b
  //             -
  //             GFiT *
  //                 (Ei_P *
  //                  (E.transpose() *
  //                   b)));  // D = (DxZDim) * (ZDimx3) * (N*ZDimm) * (ZDimm x
  //                   1)

  //     // (DxD) = (DxZDim) * ( (ZDimxD) - (ZDimx3) * (3xZDim) * (ZDimxD) )
  //     augmentedHessian.setDiagonalBlock(
  //         i, GFiT * (GFi -
  //                    Ei_P * E.block(ZDim * i, 0, ZDim, N).transpose() *
  //                    GFi));

  //     // upper triangular part of the hessian
  //     for (size_t j = i + 1; j < m; j++) {  // for each camera
  //       const MatrixGFD& GFj = GFs[j];

  //       // (DxD) = (Dx2) * ( (2x2) * (2xD) )
  //       augmentedHessian.setOffDiagonalBlock(
  //           i, j,
  //           -GFiT * (Ei_P * E.block(ZDim * j, 0, ZDim, N).transpose() *
  //           GFj));
  //     }
  //   }  // end of for over cameras

  //   augmentedHessian.diagonalBlock(m)(0, 0) += b.squaredNorm();
  //   LOG(INFO) << "augmentedHessian blocks " << augmentedHessian.nBlocks() <<
  //   " with m=" << m;

  //   // structurally augmented Hessian is now wrong as we built it out of GF
  //   and E blocks (to simulate the F and E blocks of the standard form)
  //   // Need to break the blocks back into G and F blocks since we have keys
  //   assocaited with both return augmentedHessian;
  // }

  static gtsam::SymmetricBlockMatrix SchurComplement(
      const GFBlocks& GFs, const gtsam::Matrix& E,
      const Eigen::Matrix<double, N, N>& P, const gtsam::Vector& b) {
    // a single point is observed in m cameras
    size_t m = GFs.size();
    gtsam::Matrix Et = E.transpose();

    gtsam::Matrix F_block_matrix(m * 3, m * HessianDim);
    F_block_matrix.setZero();
    // LOG(INFO) << "F=" << F_block_matrix;
    // gtsam::SymmetricBlockMatrix F_block_matrix(f_block_dims);
    // size_t block_idx = 0;
    for (size_t i = 0; i < m; i++) {
      const Eigen::Matrix<double, 3, HessianDim>& GFblock = GFs.at(i);
      // LOG(INFO) << GFblock;

      // Eigen::Matrix<double, 3, HDim> gblock = GFblock.leftCols(HDim);
      // Eigen::Matrix<double, 3, HDim> fblock = GFblock.rightCols(XDim);

      // Eigen::Matrix<double, 3, HessianDim> GF
      // set along diagonals i, j, p,q
      // LOG(INFO) <<  3*i << " " << HessianDim*i;
      // F_block_matrix.block( 3*i, HessianDim*i, 3, HessianDim) = GFblock;
      F_block_matrix.block<3, HessianDim>(3 * i, HessianDim * i) = GFblock;

      // F_block_matrix.setDiagonalBlock(2*i, gblock);
      // F_block_matrix.setDiagonalBlock(2*i+1, fblock);
    }

    gtsam::Matrix F = F_block_matrix;
    gtsam::Matrix Ft = F.transpose();
    // LOG(INFO) << "F=" << F;

    gtsam::Matrix g = Ft * (b - E * P * Et * b);
    gtsam::Matrix G = Ft * F - Ft * E * P * Et * F;

    // size of schur = num measurements * Hessian size + 1
    size_t aug_hessian_size = m * HessianDim + 1;
    gtsam::Matrix schur(aug_hessian_size, aug_hessian_size);

    schur << G, g, g.transpose(), b.squaredNorm();

    std::vector<Eigen::DenseIndex> dims(2 * m + 1);  // includes b term
    std::fill(dims.begin(), dims.end() - 1,
              HDim);  // assuming HDim and Xdim are the same size
    dims.back() = 1;

    gtsam::SymmetricBlockMatrix augmented_hessian(dims, schur);
    return augmented_hessian;
  }

 protected:
  const gtsam::Pose3 L_s0_;
  SmartMotionFactorParams params_;

  // TODO: not used
  //  Cache for Fblocks, to avoid a malloc ever time we re-linearize
  //  mutable FBlocks Fs;
  //  // Cache for Fblocks, to avoid a malloc ever time we re-linearize
  //  mutable GBlocks Gs;

  ZVector measured_;

  gtsam::SharedNoiseModel noise_model_;
  gtsam::KeyVector motion_keys_;
  gtsam::KeyVector pose_keys_;

  // making this mutable seems to break things... :)
  std::optional<gtsam::Point3> result_;
  mutable Poses camera_poses_triangulation_;  //! current triangulation poses
  mutable Motions
      object_motions_triangulation_;  //! current triangulation object motions
};

using ObjectCentricSmartFactor =
    SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

struct ObjectCentricProperties {
  inline gtsam::Symbol makeDynamicKey(TrackletId tracklet_id) const {
    return (gtsam::Symbol)DynamicLandmarkSymbol(0u, tracklet_id);
  }
};

using KeyFrameData = MultiFrameRangeData<ObjectId, gtsam::Pose3>;
using KeyFrameRange = KeyFrameData::FrameRangeT;

class ObjectCentricAccessor : public Accessor<Map3d2d>,
                              public ObjectCentricProperties {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectCentricAccessor)

  ObjectCentricAccessor(
      const SharedFormulationData& shared_data, Map3d2d::Ptr map,
      const KeyFrameData* key_frame_data,
      const gtsam::FastMap<TrackletId, FrameId>* tracklet_id_to_keyframe)
      : Accessor<Map3d2d>(shared_data, map),
        key_frame_data_(key_frame_data),
        tracklet_id_to_keyframe_(tracklet_id_to_keyframe) {}
  virtual ~ObjectCentricAccessor() {}

  StateQuery<gtsam::Pose3> getSensorPose(FrameId frame_id) const override;
  StateQuery<gtsam::Pose3> getObjectMotion(FrameId frame_id,
                                           ObjectId object_id) const override;
  StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id,
                                         ObjectId object_id) const override;
  StateQuery<gtsam::Point3> getDynamicLandmark(
      FrameId frame_id, TrackletId tracklet_id) const override;
  // in thie case we can actually propogate all object points ;)
  // if frame id does not exist or the object does not exist at this frame,
  // return an empty vector!!
  // use with, does object exist and does frame exist perhaps...?
  StatusLandmarkVector getDynamicLandmarkEstimates(
      FrameId frame_id, ObjectId object_id) const override;

  /**
   * @brief Get all dynamic object estimates in the local object frame
   *
   * @param object_id
   * @return StatusLandmarkVector
   */
  StatusLandmarkVector getLocalDynamicLandmarkEstimates(
      ObjectId object_id) const;

 private:
  struct DynamicLandmarkQuery {
    StateQuery<gtsam::Point3>* query_m_W = nullptr;
    StateQuery<gtsam::Point3>* query_m_L = nullptr;
    StateQuery<gtsam::Pose3>* query_H_W_e_k = nullptr;
    KeyFrameRange::ConstPtr* frame_range_ptr = nullptr;
  };

  // should treturn true if and only if all valid queries (ie, non-null queries)
  // were set with valid dataa!!
  bool getDynamicLandmarkImpl(FrameId frame_id, TrackletId tracklet_id,
                              DynamicLandmarkQuery& query) const;

  bool getDynamicLandmarkImpl(FrameId frame_id, TrackletId tracklet_id,
                              StateQuery<gtsam::Point3>* query_m_W,
                              StateQuery<gtsam::Point3>* query_m_L,
                              StateQuery<gtsam::Pose3>* query_H_W_e_k,
                              KeyFrameRange::ConstPtr* frame_range_ptr) const;

  const KeyFrameData* key_frame_data_;
  //! Tracklet Id to the Keyframe the point is represented in (ie. which frame)
  const gtsam::FastMap<TrackletId, FrameId>* tracklet_id_to_keyframe_;
};

// TODO: should all be in keyframe_object_centric namespace!!
class ObjectCentricFormulation : public Formulation<Map3d2d>,
                                 public ObjectCentricProperties {
 public:
  using Base = Formulation<Map3d2d>;
  using Base::AccessorTypePointer;
  using Base::ObjectUpdateContextType;
  using Base::PointUpdateContextType;

  DYNO_POINTER_TYPEDEFS(ObjectCentricFormulation)

  ObjectCentricFormulation(const FormulationParams& params,
                           typename Map::Ptr map,
                           const NoiseModels& noise_models,
                           const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}
  virtual ~ObjectCentricFormulation() {}

  virtual void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;
  virtual void objectUpdateContext(
      const ObjectUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  inline bool isDynamicTrackletInMap(
      const LandmarkNode3d2d::Ptr& lmk_node) const override {
    const TrackletId tracklet_id = lmk_node->tracklet_id;
    return is_dynamic_tracklet_in_map_.exists(tracklet_id);
  }

  // TODO: functions should be shared with accessor
  bool hasObjectKeyFrame(ObjectId object_id, FrameId frame_id) const;
  std::pair<FrameId, gtsam::Pose3> getObjectKeyFrame(ObjectId object_id,
                                                     FrameId frame_id) const;
  // get the estimated motion in the representation used directly by the
  // estimation (ie. not frame-2-frame)
  // TODO: should be in accessor!!!!
  StateQuery<Motion3ReferenceFrame> getEstimatedMotion(ObjectId object_id,
                                                       FrameId frame_id) const;

  std::pair<FrameId, gtsam::Pose3> forceNewKeyFrame(FrameId frame_id,
                                                    ObjectId object_id);

 protected:
  AccessorTypePointer createAccessor(
      const SharedFormulationData& shared_data) const override {
    return std::make_shared<ObjectCentricAccessor>(
        shared_data, this->map(), &key_frame_data_, &all_dynamic_landmarks_);
  }

  virtual std::string loggerPrefix() const override { return "object_centric"; }

 protected:
  std::pair<FrameId, gtsam::Pose3> getOrConstructL0(ObjectId object_id,
                                                    FrameId frame_id);

  // hacky update solution for now!!
  gtsam::Pose3 computeInitialH(ObjectId object_id, FrameId frame_id,
                               bool* keyframe_updated = nullptr);

  gtsam::Pose3 calculateObjectCentroid(ObjectId object_id,
                                       FrameId frame_id) const;

  // TODO: in the sliding window case the formulation gets reallcoated every
  // time so that L0 map is different, but the values will share the same H
  // (which is now from a different L0)!! make static (hack) for now
  // TODO: bad!! should not be static as this will also get held between
  // estimators!!!?
  // gtsam::FastMap<ObjectId, std::pair<FrameId, gtsam::Pose3>> L0_;
  // gtsam::FastMap<ObjectId, std::vector<KeyFrameRange>> L0_;

  // we need a separate way of tracking if a dynamic tracklet is in the map,
  // since each point is modelled uniquely simply used as an O(1) lookup, the
  // value is not actually used. If the key exists, we assume that the tracklet
  // is in the map
  // tracklet Id associated with reference frame (to track which KeyFrame the
  // point is in!!!)
  gtsam::FastMap<TrackletId, FrameId>
      is_dynamic_tracklet_in_map_;  //! the set of dynamic points that have been
                                    //! added by this updater. We use a separate
                                    //! map containing the tracklets as the keys
                                    //! are non-unique. This indicates values
                                    //! that are currently in the estimator so
                                    //! if values are removed from the
                                    //! esstimator they need to be updated here
                                    //! too!
  //! All tracks on the object and their keyframe and is shared with the
  //! accessor. This does not get cleared when the a new keyframe is added so
  //! that accessor still has access to the meta-data for each tracked point.
  gtsam::FastMap<TrackletId, FrameId> all_dynamic_landmarks_;

 private:
  KeyFrameData key_frame_data_;
};

}  // namespace dyno
