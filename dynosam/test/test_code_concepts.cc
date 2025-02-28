/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/treeTraversal-inst.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/utilities.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <atomic>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <exception>
#include <fstream>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <type_traits>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/visualizer/ColourMap.hpp"

std::vector<gtsam::Point3> createPoints() {
  // Create the set of ground-truth landmarks
  std::vector<gtsam::Point3> points;
  points.push_back(gtsam::Point3(10.0, 10.0, 10.0));
  points.push_back(gtsam::Point3(-10.0, 10.0, 10.0));
  points.push_back(gtsam::Point3(-10.0, -10.0, 10.0));
  points.push_back(gtsam::Point3(10.0, -10.0, 10.0));
  points.push_back(gtsam::Point3(10.0, 10.0, -10.0));
  points.push_back(gtsam::Point3(-10.0, 10.0, -10.0));
  points.push_back(gtsam::Point3(-10.0, -10.0, -10.0));
  points.push_back(gtsam::Point3(10.0, -10.0, -10.0));

  return points;
}

/* ************************************************************************* */
std::vector<gtsam::Pose3> createPoses(
    const gtsam::Pose3& init = gtsam::Pose3(gtsam::Rot3::Ypr(M_PI / 2, 0,
                                                             -M_PI / 2),
                                            gtsam::Point3(30, 0, 0)),
    const gtsam::Pose3& delta = gtsam::Pose3(
        gtsam::Rot3::Ypr(0, -M_PI / 4, 0),
        gtsam::Point3(sin(M_PI / 4) * 30, 0, 30 * (1 - sin(M_PI / 4)))),
    int steps = 8) {
  // Create the set of ground-truth poses
  // Default values give a circular trajectory, radius 30 at pi/4 intervals,
  // always facing the circle center
  std::vector<gtsam::Pose3> poses;
  int i = 1;
  poses.push_back(init);
  for (; i < steps; ++i) {
    poses.push_back(poses[i - 1].compose(delta));
  }

  return poses;
}

cv::Mat GetSquareImage(const cv::Mat& img, int target_width = 500) {
  int width = img.cols, height = img.rows;

  cv::Mat square = cv::Mat::zeros(target_width, target_width, img.type());

  int max_dim = (width >= height) ? width : height;
  float scale = ((float)target_width) / max_dim;
  cv::Rect roi;
  if (width >= height) {
    roi.width = target_width;
    roi.x = 0;
    roi.height = height * scale;
    roi.y = (target_width - roi.height) / 2;
  } else {
    roi.y = 0;
    roi.height = target_width;
    roi.width = width * scale;
    roi.x = (target_width - roi.width) / 2;
  }

  cv::resize(img, square(roi), roi.size());

  return square;
}

// we need compose, inverse (and thats it?)

template <typename T>
struct indexed_transform_traits;

// actually just want this for Pose operations, i guess? Do I both generalising?
//  this should happily cover all Eigen stuff as well :)
template <typename T, class = typename std::enable_if<is_gtsam_value_v<T>>>
struct GtsamIndexedTransformTraits {
  static T Inverse(const T& t) { return gtsam::traits<T>::Inverse(t); }

  static T Compose(const T& g, const T& h) {
    return gtsam::traits<T>::Compose(g, h);
  }
};
template <typename T>
struct indexed_transform_traits : GtsamIndexedTransformTraits<T> {};

/**
 * @brief Pose class that add the 3-indexed notation from the "Pose changes
 * from a different point of view" paper
 *
 */
template <typename T>
class IndexedTransform {
 public:
  using This = IndexedTransform<T>;

  // Top left index representing observing frame
  std::string tl;
  // Bottom left index, representing from frame
  std::string bl{};
  // Bottom right index, representing to frame
  std::string br{};
  // Data
  T transform;

  IndexedTransform(const std::string& top_left, const std::string& bottom_left,
                   const std::string& bottom_right, const T& t)
      : tl(top_left), bl(bottom_left), br(bottom_right), transform(t) {}

  const std::string& Origin() const { return tl; }
  const std::string& From() const { return bl; }
  const std::string& To() const { return br; }

  /**
   * @brief Returns true if the captured data represents a pose or a motion.
   * As per Equation 10 and 11 we can, if the top left and bottom left notation
   * is the same, it can be re-written in the ususal two index form
   *
   * @return true
   * @return false
   */
  inline bool isCoordinateTransform() const { return tl == bl; }

  operator T() { return transform; }

  This compose(const This& t) const {
    // T new_transform = transform * t.transform;
    T new_transform =
        indexed_transform_traits<T>::Compose(transform, t.transform);

    // if this is relative (left hand side), both must be relative
    if (isCoordinateTransform()) {
      if (!t.isCoordinateTransform()) {
        throw std::runtime_error("");
      } else {
        // check that the coordinates match
        if (br != t.tl) {
          // throw exception
          throw std::runtime_error("");
        }

        // transform is good, update the bottom right frame (to frame)
        std::string new_br = t.br;
        return IndexedTransform(tl, bl, new_br, new_transform);
      }
    } else {
      // we are global motion
      // if this is global (left hand side), the RHS (t) can be either relative
      // or global as long as the transforms match up
      // t's origin must match this (LHS) origin
      // t's to frame must match this (LHS) from frame
      if ((t.Origin() != Origin()) || (t.To() != From())) {
        throw std::runtime_error("");
      }

      if (t.isCoordinateTransform()) {
        std::string new_bl = t.From();
        // this should match the origin
        assert(new_bl == Origin());
        std::string new_br = t.To();
        return IndexedTransform(Origin(), new_bl, new_br, new_transform);
      } else {
        std::string new_bl = t.From();
        std::string new_br = From();
        return IndexedTransform(Origin(), new_bl, new_br, new_transform);
      }
    }
  }

  This operator*(const This& t) const { return compose(t); }

  This decompose(const This& t) const { return inverse().compose(t); }

  // update this one?
  // maybe an invert and inverse (invert operates on this one)
  This inverse() const {
    This tmp = *this;
    tmp.transform = indexed_transform_traits<T>::Inverse(transform);

    // always swap bottom index's
    std::swap(tmp.bl, tmp.br);

    // coordinate transform, this we also update the top left with the new
    // bottom left
    if (isCoordinateTransform()) {
      tmp.tl = tmp.bl;
    }
    return tmp;
  }
};

TEST(CodeConcepts, threeIndexNotationRelativeInverse) {
  gtsam::Pose3 A_B(gtsam::Rot3::Rodrigues(0.1, -0.2, 0.01),
                   gtsam::Point3(0.5, 3, 1));
  IndexedTransform<gtsam::Pose3> a_b("a", "a", "b", A_B);
  EXPECT_TRUE(a_b.isCoordinateTransform());

  auto b_a = a_b.inverse();
  EXPECT_EQ(b_a.Origin(), "b");
  EXPECT_EQ(b_a.From(), "b");
  EXPECT_EQ(b_a.To(), "a");

  EXPECT_EQ(a_b.Origin(), "a");
  EXPECT_EQ(a_b.From(), "a");
  EXPECT_EQ(a_b.To(), "b");

  EXPECT_TRUE(b_a.isCoordinateTransform());
  EXPECT_TRUE(gtsam::assert_equal(A_B.inverse(), b_a.transform));
}

// TODO: this is not correct!!!
TEST(CodeConcepts, threeIndexNotationAbsoluteInverse) {
  gtsam::Pose3 W_AB(gtsam::Rot3::Rodrigues(0.1, -0.2, 0.01),
                    gtsam::Point3(0.5, 3, 1));
  IndexedTransform<gtsam::Pose3> w_ab("w", "a", "b", W_AB);
  EXPECT_FALSE(w_ab.isCoordinateTransform());

  auto w_ba = w_ab.inverse();
  EXPECT_EQ(w_ba.Origin(), "w");
  EXPECT_EQ(w_ba.From(), "b");
  EXPECT_EQ(w_ba.To(), "a");

  EXPECT_EQ(w_ab.Origin(), "w");
  EXPECT_EQ(w_ab.From(), "a");
  EXPECT_EQ(w_ab.To(), "b");

  EXPECT_FALSE(w_ba.isCoordinateTransform());
  EXPECT_TRUE(gtsam::assert_equal(W_AB.inverse(), w_ba.transform));
}

TEST(CodeConcepts, threeIndexNotationRelativeComposition) {
  gtsam::Pose3 A(gtsam::Rot3::Rodrigues(0.3, 2, 3), gtsam::Point3(1, 2, 2));
  gtsam::Pose3 A_B(gtsam::Rot3::Rodrigues(0.1, -0.2, 0.01),
                   gtsam::Point3(0.5, 3, 1));
  IndexedTransform<gtsam::Pose3> a("w", "w", "a", A);
  IndexedTransform<gtsam::Pose3> a_b("a", "a", "b", A_B);

  auto result = a * a_b;
  EXPECT_EQ(result.Origin(), "w");
  EXPECT_EQ(result.From(), "w");
  EXPECT_EQ(result.To(), "b");
  EXPECT_TRUE(gtsam::assert_equal(A * A_B, result.transform));
}

// previsitor
struct Node {
  void operator()(const boost::shared_ptr<gtsam::ISAM2Clique>& clique) {
    auto conditional = clique->conditional();
    // it is FACTOR::const_iterator
    for (auto it = conditional->beginFrontals();
         it != conditional->endFrontals(); it++) {
      LOG(INFO) << dyno::DynoLikeKeyFormatter(*it);
    }
  }
};

// TEST(CodeConcepts, getISAM2Ordering) {
//     using namespace gtsam;
//     Cal3_S2::shared_ptr K(new Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0));

//   // Define the camera observation noise model, 1 pixel stddev
//   auto measurementNoise = noiseModel::Isotropic::Sigma(2, 1.0);

//   // Create the set of ground-truth landmarks
//   std::vector<Point3> points = createPoints();

//   // Create the set of ground-truth poses
//   std::vector<Pose3> poses = createPoses();

//   // Create an iSAM2 object. Unlike iSAM1, which performs periodic batch
//   steps
//   // to maintain proper linearization and efficient variable ordering, iSAM2
//   // performs partial relinearization/reordering at each step. A parameter
//   // structure is available that allows the user to set various properties,
//   such
//   // as the relinearization threshold and type of linear solver. For this
//   // example, we we set the relinearization threshold small so the iSAM2
//   result
//   // will approach the batch result.
//   ISAM2Params parameters;
//   parameters.relinearizeThreshold = 0.01;
//   parameters.relinearizeSkip = 1;
//   ISAM2 isam(parameters);

//   // Create a Factor Graph and Values to hold the new data
//   NonlinearFactorGraph graph;
//   Values initialEstimate;

//   // Loop over the poses, adding the observations to iSAM incrementally
//   for (size_t i = 0; i < poses.size(); ++i) {
//     // Add factors for each landmark observation
//     for (size_t j = 0; j < points.size(); ++j) {
//       PinholeCamera<Cal3_S2> camera(poses[i], *K);
//       Point2 measurement = camera.project(points[j]);
//       graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
//           measurement, measurementNoise, Symbol('x', i), Symbol('l', j), K);
//     }

//     // Add an initial guess for the current pose
//     // Intentionally initialize the variables off from the ground truth
//     static Pose3 kDeltaPose(Rot3::Rodrigues(-0.1, 0.2, 0.25),
//                             Point3(0.05, -0.10, 0.20));
//     initialEstimate.insert(Symbol('x', i), poses[i] * kDeltaPose);

//     // If this is the first iteration, add a prior on the first pose to set
//     the
//     // coordinate frame and a prior on the first landmark to set the scale
//     Also,
//     // as iSAM solves incrementally, we must wait until each is observed at
//     // least twice before adding it to iSAM.
//     if (i == 0) {
//       // Add a prior on pose x0, 30cm std on x,y,z and 0.1 rad on
//       roll,pitch,yaw static auto kPosePrior = noiseModel::Diagonal::Sigmas(
//           (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3))
//               .finished());
//       graph.addPrior(Symbol('x', 0), poses[0], kPosePrior);

//       // Add a prior on landmark l0
//       static auto kPointPrior = noiseModel::Isotropic::Sigma(3, 0.1);
//       graph.addPrior(Symbol('l', 0), points[0], kPointPrior);

//       // Add initial guesses to all observed landmarks
//       // Intentionally initialize the variables off from the ground truth
//       static Point3 kDeltaPoint(-0.25, 0.20, 0.15);
//       for (size_t j = 0; j < points.size(); ++j)
//         initialEstimate.insert<Point3>(Symbol('l', j), points[j] +
//         kDeltaPoint);

//     } else {
//       // Update iSAM with the new factors
//       isam.update(graph, initialEstimate);
//       // Each call to iSAM2 update(*) performs one iteration of the iterative
//       // nonlinear solver. If accuracy is desired at the expense of time,
//       // update(*) can be called additional times to perform multiple
//       optimizer
//       // iterations every step.
//       isam.update();
//       Values currentEstimate = isam.calculateEstimate();

//       // Clear the factor graph and values for the next iteration
//       graph.resize(0);
//       initialEstimate.clear();
//     }
//   }

//     // isam.saveGraph(dyno::getOutputFilePath("test_bayes_tree.dot"));

//     std::function<void(const boost::shared_ptr<gtsam::ISAM2Clique>&)>
//     node_func = Node();
//     //
//     dyno::factor_graph_tools::travsersal::depthFirstTraversalEliminiationOrder(isam,
//     node_func); LOG(INFO) <<
//     dyno::container_to_string(dyno::factor_graph_tools::travsersal::getEliminatonOrder(isam));
//     // Data rootdata;
//     // Node preVisitor;
//     // no_op postVisitor;
//     // gtsam::treeTraversal::DepthFirstForest(isam, rootdata, preVisitor,
//     postVisitor);

// }

// //https://github.com/zhixy/SolveAXXB/blob/master/axxb/conventionalaxxbsvdsolver.cc
// TEST(CodeConcepts, sylvesterEquation) {
//     // Eigen::MatrixXd A;
//     // A.resize(4, 4);
//     // A << 1, 0, 2, 3,
//     //      4, 1, 0, 2,
//     //      0, 5, 5, 6,
//     //      1, 7, 9, 0;

//     gtsam::Pose3 L_k_1 =
//     dyno::utils::createRandomAroundIdentity<gtsam::Pose3>(0.1); gtsam::Pose3
//     L_k_1_H_k = dyno::utils::createRandomAroundIdentity<gtsam::Pose3>(0.03);

//     //L_k = L_K_1 * ^{L_k}_kH_{k-1}
//     gtsam::Pose3 L_k = L_k_1 * L_k_1_H_k ;
//     gtsam::Pose3 W_k_1_H_k = L_k_1 * L_k_1_H_k * L_k_1.inverse();

//     Eigen::MatrixXd A = W_k_1_H_k.matrix();
//     Eigen::MatrixXd B = L_k_1_H_k.matrix();

//     //1 as we only have one A here
//     Eigen::MatrixXd m = Eigen::MatrixXd::Zero(12*1,12);
//     Eigen::VectorXd b = Eigen::VectorXd::Zero(12*1);

//     Eigen::Matrix3d Ra = A.topLeftCorner(3,3);
//     Eigen::Vector3d Ta = A.topRightCorner(3,1);
//     Eigen::Matrix3d Rb = B.topLeftCorner(3,3);
//     Eigen::Vector3d Tb = B.topRightCorner(3,1);

//     m.block<9,9>(12*0,0) = Eigen::MatrixXd::Identity(9,9) -
//     Eigen::kroneckerProduct(Ra,Rb); m.block<3,9>(12*0+9,0) =
//     Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(3,3),Tb.transpose());
//     m.block<3,3>(12*0+9,9) = Eigen::MatrixXd::Identity(3,3) - Ra;
//     b.block<3,1>(12*0+9,0) = Ta;

//     // //different to skew in gtsam...
//     // auto skew =[](const Eigen::Vector3d& u) {
//     //     Eigen::Matrix3d u_hat = Eigen::MatrixXd::Zero(3,3);
//     //     u_hat(0,1) = u(2);
//     //     u_hat(1,0) = -u(2);
//     //     u_hat(0,2) = -u(1);
//     //     u_hat(2,0) = u(1);
//     //     u_hat(1,2) = u(0);
//     //     u_hat(2,1) = -u(0);
//     //     return u_hat;
//     // };
//     // Eigen::Matrix3d Ta_skew = skew(Ta);
//     // // m.block<3,9>(12*0+9,0) =
//     Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(3,3),Tb.transpose());
//     // // m.block<3,3>(12*0+9,9) = Eigen::MatrixXd::Identity(3,3) - Ra;
//     // // b.block<3,1>(12*0+9,0) = Ta;
//     // m.block<3,9>(12*0+9,0) =
//     Eigen::kroneckerProduct(Ta_skew,Tb.transpose());
//     // m.block<3,3>(12*0+9,9) = Ta_skew - Ta_skew*Ra;

//     // Eigen::JacobiSVD<Eigen::MatrixXd> svd( m, Eigen::ComputeFullV |
//     Eigen::ComputeFullU );
//     // CHECK(svd.computeV())<<"fail to compute V";

//     Eigen::Matrix<double, 12, 1> x = m.bdcSvd(Eigen::ComputeThinU |
//     Eigen::ComputeThinV).solve(b); Eigen::Matrix3d R = Eigen::Map<
//     Eigen::Matrix<double, 3, 3, Eigen::RowMajor> >(x.data()); //row major

//     LOG(INFO) << x;

//     Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeThinU |
//     Eigen::ComputeThinV); gtsam::Matrix44 handeyetransformation =
//     gtsam::Matrix44::Identity(); handeyetransformation.topLeftCorner(3,3) =
//     svd.matrixU() * svd.matrixV().transpose();
//     handeyetransformation.topRightCorner(3,1) = x.block<3,1>(9,0);

//     handeyetransformation.topRightCorner(3,1) = x.block<3,1>(9,0);

//     // Eigen::Matrix3d R_alpha;
//     // R_alpha.row(0) = svd.matrixV().block<3,1>(0,11).transpose();
//     // R_alpha.row(1) = svd.matrixV().block<3,1>(3,11).transpose();
//     // R_alpha.row(2) = svd.matrixV().block<3,1>(6,11).transpose();
//     // //double a = std::fabs(R_alpha.determinant());
//     // //double alpha =
//     R_alpha.determinant()/(pow(std::fabs(R_alpha.determinant()),4./3.));
//     // double det = R_alpha.determinant();
//     // double alpha = std::pow(std::abs(det),4./3.)/det;
//     // Eigen::HouseholderQR<Eigen::Matrix3d> qr(R_alpha/alpha);

//     // gtsam::Matrix44 handeyetransformation = gtsam::Matrix44::Identity();
//     // Eigen::Matrix3d Q = qr.householderQ();
//     // Eigen::Matrix3d Rwithscale = alpha*Q.transpose()*R_alpha;
//     // Eigen::Vector3d R_diagonal = Rwithscale.diagonal();
//     // for(int i=0;i<3;i++)
//     // {
//     //     handeyetransformation.block<3,1>(0,i) =
//     int(R_diagonal(i)>=0?1:-1)*Q.col(i);
//     // }

//     // handeyetransformation.topRightCorner(3,1) =
//     svd.matrixV().block<3,1>(9,11)/alpha;

//     // LOG(INFO) << A;
//     // // LOG(INFO) << B;
//     // LOG(INFO) << handeyetransformation;

//     // // gtsam::Matrix44 calc_C = A*handeyetransformation -
//     handeyetransformation * B; LOG(INFO) << L_k_1;

//      LOG(INFO) << A*handeyetransformation;
//     LOG(INFO) << handeyetransformation * B;

//     LOG(INFO) << handeyetransformation;

//     // Eigen::RealSchur<Eigen::MatrixXd> SchurA(A);
//     // Eigen::MatrixXd R = SchurA.matrixT();
//     // Eigen::MatrixXd U = SchurA.matrixU();

//     // // Eigen::MatrixXd B = -A.transpose();
//     // Eigen::MatrixXd B;
//     // B.resize(2, 2);
//     // B << 0, -1,
//     //      1, 0;

//     // Eigen::RealSchur<Eigen::MatrixXd> SchurB(B);
//     // Eigen::MatrixXd S = SchurB.matrixT();
//     // Eigen::MatrixXd V = SchurB.matrixU();

//     // //C
//     // // Eigen::MatrixXd I_33 =  gtsam::Matrix33::Zero();
//     // Eigen::MatrixXd C;
//     // C.resize(4, 2);
//     // C << 1, 0,
//     //      2, 0,
//     //      0, 3,
//     //      1, 1;

//     // Eigen::MatrixXd F = (U.adjoint() * C) * V;

//     // Eigen::MatrixXd Y =
//     Eigen::internal::matrix_function_solve_triangular_sylvester(R, S, F);

//     // Eigen::MatrixXd X = (U * Y) * V.adjoint();
//     // LOG(INFO) << "X= " << X;

//     // Eigen::MatrixXd C_calc = A * X + X * B;
//     // LOG(INFO) << "C calc= " << C_calc;

// }

TEST(CodeConcepts, drawInformationMatrix) {
  gtsam::Values initial_estimate;
  gtsam::NonlinearFactorGraph graph;
  const auto model = gtsam::noiseModel::Isotropic::Sigma(3, 1);

  using namespace gtsam::symbol_shorthand;

  // //smaller example
  // gtsam::Pose3 first_pose;
  // graph.emplace_shared<gtsam::NonlinearEquality<gtsam::Pose3> >(X(1),
  // gtsam::Pose3());

  // // create factor noise model with 3 sigmas of value 1
  // // const auto model = gtsam::noiseModel::Isotropic::Sigma(3, 1);
  // // create stereo camera calibration object with .2m between cameras
  // const gtsam::Cal3_S2Stereo::shared_ptr K(
  //     new gtsam::Cal3_S2Stereo(1000, 1000, 0, 320, 240, 0.2));

  // //create and add stereo factors between first pose (key value 1) and the
  // three landmarks
  // //
  // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3>
  // >(gtsam::StereoPoint2(520, 480, 440), model, 1, 3, K);
  // //
  // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3>
  // >(gtsam::StereoPoint2(120, 80, 440), model, 1, 4, K);
  // //
  // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3>
  // >(gtsam::StereoPoint2(320, 280, 140), model, 1, 5, K);

  // // //create and add stereo factors between second pose and the three
  // landmarks
  // //
  // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3>
  // >(gtsam::StereoPoint2(570, 520, 490), model, 2, 3, K);
  // //
  // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3>
  // >(gtsam::StereoPoint2(70, 20, 490), model, 2, 4, K);
  //  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3>
  //  >(X(1), L(1), gtsam::Point3(1, 1, 5), model);
  // graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3>
  // >(X(1), L(2),  gtsam::Point3(1, 1, 5), model);
  // graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3>
  // >(X(1), L(3), gtsam::Point3::Identity(), model);

  // //create and add stereo factors between second pose and the three landmarks
  // graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3>
  // >(X(2),L(1), gtsam::Point3::Identity(), model);
  // graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3>
  // >(X(2), L(2), gtsam::Point3::Identity(), model);
  // //
  // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3>
  // >(gtsam::StereoPoint2(320, 270, 115), model, 2, 5, K);

  // // create Values object to contain initial estimates of camera poses and
  // // landmark locations

  // // create and add iniital estimates
  // initial_estimate.insert(X(1), first_pose);
  // initial_estimate.insert(X(2), gtsam::Pose3(gtsam::Rot3(),
  // gtsam::Point3(0.1, -0.1, 1.1))); initial_estimate.insert(L(1),
  // gtsam::Point3(1, 1, 5)); initial_estimate.insert(L(2), gtsam::Point3(-1, 1,
  // 5)); initial_estimate.insert(L(3), gtsam::Point3(1, -0.5, 5));

  // std::string calibration_loc =
  // gtsam::findExampleDataFile("VO_calibration.txt"); std::string pose_loc =
  // gtsam::findExampleDataFile("VO_camera_poses_large.txt"); std::string
  // factor_loc = gtsam::findExampleDataFile("VO_stereo_factors_large.txt");

  // // read camera calibration info from file
  // // focal lengths fx, fy, skew s, principal point u0, v0, baseline b
  // double fx, fy, s, u0, v0, b;
  // std::ifstream calibration_file(calibration_loc.c_str());
  // std::cout << "Reading calibration info" << std::endl;
  // calibration_file >> fx >> fy >> s >> u0 >> v0 >> b;

  // // create stereo camera calibration object
  // const gtsam::Cal3_S2Stereo::shared_ptr K(new gtsam::Cal3_S2Stereo(fx, fy,
  // s, u0, v0, b));

  // std::ifstream pose_file(pose_loc.c_str());
  // std::cout << "Reading camera poses" << std::endl;
  // int pose_id;
  // gtsam::MatrixRowMajor m(4, 4);
  // // read camera pose parameters and use to make initial estimates of camera
  // // poses
  // while (pose_file >> pose_id) {
  //     for (int i = 0; i < 16; i++) {
  //         pose_file >> m.data()[i];
  //     }
  //     initial_estimate.insert(gtsam::Symbol('x', pose_id), gtsam::Pose3(m));
  // }

  // // camera and landmark keys
  // size_t x, l;

  // // pixel coordinates uL, uR, v (same for left/right images due to
  // // rectification) landmark coordinates X, Y, Z in camera frame, resulting
  // from
  // // triangulation
  // double uL, uR, v, X, Y, Z;
  // std::ifstream factor_file(factor_loc.c_str());
  // std::cout << "Reading stereo factors" << std::endl;
  // // read stereo measurement details from file and use to create and add
  // // GenericStereoFactor objects to the graph representation
  // while (factor_file >> x >> l >> uL >> uR >> v >> X >> Y >> Z) {
  //     graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,
  //     gtsam::Point3> >(
  //         gtsam::StereoPoint2(uL, uR, v), model, gtsam::Symbol('x', x),
  //         gtsam::Symbol('l', l), K);
  //     // if the landmark variable included in this factor has not yet been
  //     added
  //     // to the initial variable value estimate, add it
  //     if (!initial_estimate.exists(gtsam::Symbol('l', l))) {
  // gtsam::Pose3 camPose = initial_estimate.at<gtsam::Pose3>(gtsam::Symbol('x',
  // x));
  //         // transformFrom() transforms the input Point3 from the camera pose
  //         space,
  //         // camPose, to the global space
  //         gtsam::Point3 worldPoint = camPose.transformFrom(gtsam::Point3(X,
  //         Y, Z)); initial_estimate.insert(gtsam::Symbol('l', l), worldPoint);
  //     }
  // }

  // auto gfg = graph.linearize(initial_estimate);
  // // gtsam::Matrix jacobian = gfg->jacobian().first;

  // gtsam::Ordering natural_ordering = gtsam::Ordering::Natural(*gfg);
  // gtsam::JacobianFactor jf(*gfg, natural_ordering);
  // gtsam::Matrix J = jf.jacobian().first;

  // {
  //     gtsam::Values pose_only_values;
  //     pose_only_values.insert(0, gtsam::Pose3::Identity());
  //     pose_only_values.insert(1, gtsam::Pose3::Identity());
  //     gtsam::NonlinearFactorGraph pose_only_graph;
  //     pose_only_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0,
  //     gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Sigma(6, 1));
  //     pose_only_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(1,
  //     gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Sigma(6, 1));

  //     gtsam::Ordering natural_ordering_pose =
  //     gtsam::Ordering::Natural(pose_only_graph); auto gfg_pose =
  //     pose_only_graph.linearize(pose_only_values); gtsam::JacobianFactor
  //     jf_pose(*gfg_pose, natural_ordering_pose);

  //         // auto size = natural_ordering.size();
  //         // auto size1 = jf_pose.cols() * 6;
  //     //cols should be num variables * 6 as each pose has dimenstions
  //     variables EXPECT_EQ(natural_ordering_pose.size() *
  //     6,(jf_pose.getA().cols()));

  //     //this just a view HORIZONTALLY, vertically it is the entire matrix
  //     const gtsam::VerticalBlockMatrix::constBlock Ablock =
  //     jf_pose.getA(jf_pose.find(0));
  //     //in this example there should be 2? blocks
  //     // LOG(INFO) << "Num blocks " << Ablock.nBlocks();
  //     LOG(INFO) << Ablock;

  //     for(gtsam::Key key : natural_ordering_pose) {
  //     //this will have rows = J.rows(), cols = dimension of the variable
  //     const gtsam::VerticalBlockMatrix::constBlock Ablock =
  //     jf_pose.getA(jf_pose.find(key)); const size_t var_dimensions =
  //     Ablock.cols();cv::Vec3b(255, 0, 0); EXPECT_EQ(Ablock.rows(),
  //     jf_pose.rows());

  //     LOG(INFO) << "Start Col " << Ablock.startCol() << " Start row " <<
  //     Ablock.startRow() << " dims " << var_dimensions;
  //     //eachs start col shoudl be dim(key) apart
  //     // for (int i = 0; i < J.rows(); ++i) {
  //     //     for (int j = 0; j < J.cols(); ++j) {
  //     //         if (std::fabs(J(i, j)) > 1e-15) {
  //     //             // make non zero elements blue
  //     //             J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
  //     //         }
  //     //     }
  //     // }

  //     }

  // }

  // graph.saveGraph(dyno::getOutputFilePath("test_graph.dot"));

  // LOG(INFO) << natural_ordering.size();
  // LOG(INFO) << J.cols();

  // cv::Mat J_img(cv::Size(J.cols(), J.rows()), CV_8UC3, cv::Scalar(255, 255,
  // 255));

  // std::vector<std::pair<gtsam::Key, cv::Mat>> column_blocks;
  // LOG(INFO) << J_img.cols;
  // for(gtsam::Key key : natural_ordering) {
  //     //this will have rows = J.rows(), cols = dimension of the variable
  //     // LOG(INFO) << key;
  //     const gtsam::VerticalBlockMatrix::constBlock Ablock =
  //     jf.getA(jf.find(key)); const size_t var_dimensions = Ablock.cols();
  //     EXPECT_EQ(Ablock.rows(), J.rows());

  //     // LOG(INFO) << "Start Col " << Ablock.startCol() << " Start row " <<
  //     Ablock.startRow() << " dims " << var_dimensions;

  //     for (int i = 0; i < J.rows(); ++i) {
  //         for (int j = Ablock.startCol(); j < (Ablock.startCol() +
  //         var_dimensions); ++j) {
  //             ASSERT_LT(j, J_img.cols);
  //             ASSERT_LT(i, J_img.rows);
  //             if (std::fabs(J(i, j)) > 1e-15) {
  //                 // make non zero elements blue
  //                 const auto colour =
  //                 dyno::ColourMap::getObjectColour((int)var_dimensions);
  //                 J_img.at<cv::Vec3b>(i, j) =  cv::Vec3b(colour[0],
  //                 colour[1], colour[2]);
  //                 // J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
  //                 // J_img.at<cv::Vec3b>(i, j)[2] = colour[2];
  //             }
  //         }
  //     }

  //     cv::Mat b_img(cv::Size(var_dimensions, Ablock.rows()), CV_8UC3,
  //     cv::Scalar(255, 255, 255)); for (int i = 0; i < Ablock.rows(); ++i) {
  //         for (int j = 0; j < var_dimensions; ++j) {
  //             // ASSERT_LT(j, J_img.cols);
  //             // ASSERT_LT(i, J_img.rows);
  //             if (std::fabs(Ablock(i, j)) > 1e-15) {
  //                 // make non zero elements blue
  //                 const auto colour =
  //                 dyno::ColourMap::getObjectColour((int)var_dimensions);
  //                 b_img.at<cv::Vec3b>(i, j) =  cv::Vec3b(colour[0],
  //                 colour[1], colour[2]);
  //                 // J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
  //                 // J_img.at<cv::Vec3b>(i, j)[2] = colour[2];
  //             }
  //         }
  //     }

  //     column_blocks.push_back(std::make_pair(key, b_img));

  // }

  // LOG(INFO) << column_blocks.size();

  // cv::Size desired_size(480, 480);

  // const int current_cols = J.cols();
  // const int desired_cols = desired_size.width;
  // const int desired_rows = desired_size.height;

  // double ratio = (double)desired_cols/(double)current_cols;
  // LOG(INFO) << ratio;
  // cv::Mat concat_column_blocks;
  // // cv::Mat concat_column_blocks = column_blocks.at(0);
  // cv::Mat labels;

  // for(size_t i = 1; i < column_blocks.size(); i++) {
  //     gtsam::Key key = column_blocks.at(i).first;
  //     cv::Mat current_block = column_blocks.at(i).second;
  //     int scaled_cols = ratio * (int)current_block.cols;
  //     cv::resize(current_block, current_block, cv::Size(scaled_cols,
  //     desired_rows), 0, 0, cv::INTER_NEAREST);

  //     //draw text info
  //     int baseline=0;
  //     // cv::Size textSize =
  //     cv::getTextSize(gtsam::DefaultKeyFormatter(key),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

  //     const int text_box_height = 30;
  //     cv::Mat text_box(cv::Size(current_block.cols, text_box_height),
  //     CV_8UC3, cv::Scalar(255, 255, 255));

  //     constexpr static double kFontScale = 0.5;
  //     constexpr static int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  //     constexpr static int kThickness = 2;
  //     //draw text mid way in box
  //     cv::putText(text_box, gtsam::DefaultKeyFormatter(key),
  //     cv::Point(text_box.cols/2, text_box.rows/2), kFontFace, kFontScale,
  //     cv::Scalar(0, 0, 0), kThickness);

  //     // cv::imshow("text", text_box);
  //     // cv::waitKey(0);

  //     cv::vconcat(text_box, current_block, current_block);

  //     if(i == 1) {
  //         cv::Mat first_block = column_blocks.at(0).second;
  //         gtsam::Key key_0 = column_blocks.at(0).first;

  //         int first_scaled_cols = ratio * first_block.cols;
  //         cv::resize(first_block, first_block, cv::Size(first_scaled_cols,
  //         desired_rows), 0, 0, cv::INTER_NEAREST);

  //         //add text
  //         cv::Mat text_box_first(cv::Size(first_block.cols, text_box_height),
  //         CV_8UC3, cv::Scalar(255, 255, 255)); cv::putText(text_box,
  //         gtsam::DefaultKeyFormatter(key_0), cv::Point(text_box_first.cols/2,
  //         text_box_first.rows/2), kFontFace, kFontScale, cv::Scalar(0, 0, 0),
  //         kThickness);

  //         //concat text box on top of jacobian block
  //         cv::vconcat(text_box, first_block, first_block);
  //         //now concat with next jacobian block
  //         cv::hconcat(first_block, current_block, concat_column_blocks);

  //     }
  //     else {
  //         //concat current jacobian block with other blocks
  //         cv::hconcat(concat_column_blocks, current_block,
  //         concat_column_blocks);
  //     }

  //     //if not last add vertical line
  //     if(i < column_blocks.size() - 1) {
  //         cv::Mat vert_line(cv::Size(2, concat_column_blocks.rows), CV_8UC3,
  //         cv::Scalar(0, 0, 0));
  //         //concat blocks with vertial line
  //         cv::hconcat(concat_column_blocks, vert_line, concat_column_blocks);
  //         // concat_column_blocks =
  //         dyno::utils::concatenateImagesHorizontally(concat_column_blocks,
  //         vert_line);
  //     }

  //     // //all blocks should have the same number of rows
  //     // double scaled_cols =

  //     // //add vertical line between each variable block
  //     // cv::Mat vert_line(cv::Size(2, concat_column_blocks.rows), CV_8UC3,
  //     cv::Scalar(0, 0, 0));
  //     // concat_column_blocks =
  //     dyno::utils::concatenateImagesHorizontally(concat_column_blocks,
  //     vert_line);
  // }

  // dyno::NonlinearFactorGraphManager nlfg(graph, initial_estimate);

  // cv::Mat J_1 = nlfg.drawBlockJacobian(gtsam::Ordering::NATURAL,
  // dyno::factor_graph_tools::DrawBlockJacobiansOptions::makeDynoSamOptions());

  // for (int i = 0; i < J.rows(); ++i) {
  //     for (int j = 0; j < J.cols(); ++j) {
  //         if (std::fabs(J(i, j)) > 1e-15) {
  //             // make non zero elements blue
  //             J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
  //         }
  //     }
  // }

  // gtsam::Matrix I = jf.information();
  // cv::Mat I_img(cv::Size(I.rows(), I.cols()), CV_8UC3, cv::Scalar(255, 255,
  // 255)); for (int i = 0; i < I.rows(); ++i) {
  //     for (int j = 0; j < I.cols(); ++j) {
  //         if (std::fabs(I(i, j)) > 1e-15) {
  //             // make non zero elements blue
  //             I_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
  //         }
  //     }
  // }
  // LOG(INFO) << dyno::to_string(concat_column_blocks.size());
  // //resize for viz
  // cv::resize(concat_column_blocks, concat_column_blocks, cv::Size(480, 480),
  // 0, 0, cv::INTER_NEAREST); cv::resize(J_img, J_img, cv::Size(480, 480), 0,
  // 0, cv::INTER_NEAREST);
  // // // cv::resize(I_img, I_img, cv::Size(480, 480), 0, 0, cv::INTER_CUBIC);

  // cv::imshow("Jacobian", concat_column_blocks);
  // cv::imshow("J orig", J_img);
  // cv::imshow("J_1", J_1);
  cv::waitKey(0);
}

TEST(CodeConcepts, testModifyOptionalString) {
  auto modify = [](std::optional<std::reference_wrapper<std::string>> string) {
    if (string) {
      string->get() = "udpated";
    }
  };

  std::string input = "before";
  modify(input);

  EXPECT_EQ(input, "udpated");
}

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/frontend/anms/NonMaximumSuppression.h"
#include "dynosam/frontend/anms/anms/anms.h"
#include "internal/helpers.hpp"

TEST(CodeConcepts, dynamicObjectSampling) {
  using namespace dyno;
  const std::string mask_path = getTestDataPath() + "/kitti_04_mask_000014.txt";

  cv::Size size(1242, 375);
  cv::Mat mask;
  dyno::loadSemanticMask(mask_path, size, mask);

  gtsam::FastMap<ObjectId, std::vector<cv::KeyPoint>> sampled_keypoints;
  for (int i = 0; i < mask.rows; i++) {
    for (int j = 0; j < mask.cols; j++) {
      const ObjectId object_id = mask.at<ObjectId>(i, j);

      if (object_id == background_label) {
        continue;
      }

      if (!sampled_keypoints.exists(object_id)) {
        sampled_keypoints.insert2(object_id, KeypointsCV{});
      }
      KeypointsCV& keypoints = sampled_keypoints.at(object_id);
      keypoints.push_back(utils::gtsamPointToKeyPoint(Keypoint(j, i)));
    }
  }
  cv::Mat mask_viz = dyno::ImageType::MotionMask::toRGB(mask);

  const int max_features_to_track = 50;
  static constexpr float tolerance = 0.001;
  Eigen::MatrixXd binning_mask;

  AdaptiveNonMaximumSuppression non_maximum_supression(
      AnmsAlgorithmType::RangeTree);
  for (auto& [object_id, opencv_keypoints] : sampled_keypoints) {
    // const int& number_tracked = object_tracking_info.num_track;
    const int number_tracked = opencv_keypoints.size();
    // int nr_corners_needed = std::max(
    //     max_features_to_track - number_tracked, 0);

    // std
    int nr_corners_needed = 100;

    std::vector<cv::KeyPoint>& max_keypoints = opencv_keypoints;
    // const size_t sampled_size = max_keypoints.size();
    max_keypoints = non_maximum_supression.suppressNonMax(
        opencv_keypoints, nr_corners_needed, tolerance, size.width, size.height,
        5, 5, binning_mask);

    LOG(INFO) << "Sampled " << number_tracked
              << " after anmns = " << max_keypoints.size();
    anms::VisualizeAll(mask_viz, max_keypoints,
                       "ANMS dynamic " + std::to_string(object_id));
  }
  // cv::waitKey(0);

  // cv::imshow("Mask", mask_viz);
  // cv::waitKey(0);
}
