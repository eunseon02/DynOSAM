// /*
//  *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
//  *   All rights reserved.

//  *   Permission is hereby granted, free of charge, to any person obtaining a copy
//  *   of this software and associated documentation files (the "Software"), to deal
//  *   in the Software without restriction, including without limitation the rights
//  *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  *   copies of the Software, and to permit persons to whom the Software is
//  *   furnished to do so, subject to the following conditions:

//  *   The above copyright notice and this permission notice shall be included in all
//  *   copies or substantial portions of the Software.

//  *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  *   SOFTWARE.
//  */

// #include "dynosam/frontend/vision/MotionSolver.hpp"
// #include "dynosam/utils/TimingStats.hpp"
// #include "dynosam/utils/GtsamUtils.hpp"
// #include "dynosam/utils/Numerical.hpp"
// #include "dynosam/utils/Accumulator.hpp"
// #include "dynosam/frontend/MonoInstance-Definitions.hpp"

// #include "dynosam/backend/FactorGraphTools.hpp"

// #include <eigen3/Eigen/Dense>
// #include <opencv4/opencv2/core/eigen.hpp>
// #include <opencv4/opencv2/core.hpp>


// #include "dynosam/frontend/vision/VisionTools.hpp"
// #include "dynosam/common/Types.hpp"
// #include "dynosam/utils/GtsamUtils.hpp"

// #include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
// #include "dynosam/factors/Pose3FlowProjectionFactor.h"
// #include "dynosam/backend/BackendDefinitions.hpp"

// #include <opengv/types.hpp>

// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h> //for now?
// #include <gtsam/nonlinear/Values.h>


// #include <glog/logging.h>

// namespace dyno {

// // RelativeObjectMotionSolver::MotionResult RelativeObjectMotionSolver::solve(const GenericCorrespondences<Keypoint, Keypoint>& correspondences) const {

// //     const size_t& n_matches = correspondences.size();

// //     if(n_matches < 5u) {
// //         return MotionResult::NotEnoughCorrespondences();
// //     }


// //     std::vector<cv::Point2d> ref_kps, curr_kps;
// //     TrackletIds tracklets;
// //     for(size_t i = 0u; i < n_matches; i ++) {
// //         const auto& corres = correspondences.at(i);
// //         const Keypoint& ref_kp = corres.ref_;
// //         const Keypoint& cur_kp = corres.cur_;

// //         ref_kps.push_back(utils::gtsamPointToCv(ref_kp));
// //         curr_kps.push_back(utils::gtsamPointToCv(cur_kp));

// //         tracklets.push_back(corres.tracklet_id_);
// //     }

// //     constexpr int method = cv::RANSAC;
// //     constexpr double  prob = 0.999;
// // 	constexpr double threshold = 1.0;
// //     constexpr int max_iterations = 500;
// //     const cv::Mat K = camera_params_.getCameraMatrix();

// //     cv::Mat ransac_inliers;  // a [1 x N] vector

// //     const cv::Mat E = cv::findEssentialMat(
// //         ref_kps,
// //         curr_kps,
// //         K,
// //         method,
// //         prob,
// //         threshold,
// //         max_iterations,
// //         ransac_inliers); //ransac params?

// //     CHECK(ransac_inliers.rows == tracklets.size());

// //     cv::Mat R1, R2, t;
// //     try {
// //         cv::decomposeEssentialMat(E, R1, R2, t);
// //     }
// //     catch(const cv::Exception& e) {
// //         LOG(WARNING) << "decomposeEssentialMat failed with error " << e.what();
// //         return MotionResult::Unsolvable();
// //     }

// //     TrackletIds inliers, outliers;
// //     for (int i = 0; i < ransac_inliers.rows; i++)
// //     {
// //         if(ransac_inliers.at<int>(i)) {
// //             const auto& corres = correspondences.at(i);
// //             inliers.push_back(corres.tracklet_id_);
// //         }

// //     }

// //     determineOutlierIds(inliers, tracklets, outliers);
// //     CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());

// //     EssentialDecompositionResult result(
// //         utils::cvMatToGtsamRot3(R1),
// //         utils::cvMatToGtsamRot3(R2),
// //         utils::cvMatToGtsamPoint3(t)
// //     );

// //     // LOG(INFO) << "Solving RelativeObjectMotionSolver with inliers " << inliers.size() << " outliers " <<  outliers.size();
// //     return MotionResult(result, tracklets, inliers, outliers);
// // }

// EgoMotionSolver::EgoMotionSolver(const FrontendParams& params, const CameraParams& camera_params)
//     : params_(params), camera_params_(camera_params) {}

// Pose3SolverResult EgoMotionSolver::geometricOutlierRejection2d2d(
//                             Frame::Ptr frame_k_1,
//                             Frame::Ptr frame_k,
//                             std::optional<gtsam::Rot3> R_curr_ref)
// {
//     //get correspondences
//     RelativePoseCorrespondences correspondences;
//     //this does not create proper bearing vectors (at leas tnot for 3d-2d pnp solve)
//     //bearing vectors are also not undistorted atm!!
//     {
//         utils::TimingStatsCollector track_dynamic_timer("mono_frame_correspondences");
//         frame_k->getCorrespondences(correspondences, *frame_k_1, KeyPointType::STATIC, frame_k->imageKeypointCorrespondance());
//     }

//     Pose3SolverResult result;

//     const size_t& n_matches = correspondences.size();

//     if(n_matches < 5u) {
//         result.status = TrackingStatus::FEW_MATCHES;
//         return result;
//     }

//     gtsam::Matrix K = camera_params_.getCameraMatrixEigen();
//     K = K.inverse();

//     TrackletIds tracklets;
//     //NOTE: currently without distortion! the correspondences should be made into bearing vector elsewhere!
//     BearingVectors ref_bearing_vectors, cur_bearing_vectors;
//     for(size_t i = 0u; i < n_matches; i ++) {
//         const auto& corres = correspondences.at(i);
//         const Keypoint& ref_kp = corres.ref_;
//         const Keypoint& cur_kp = corres.cur_;

//         gtsam::Vector3 ref_versor = (K * gtsam::Vector3(ref_kp(0), ref_kp(1), 1.0));
//         gtsam::Vector3 cur_versor = (K * gtsam::Vector3(cur_kp(0), cur_kp(1), 1.0));

//         ref_versor = ref_versor.normalized();
//         cur_versor = cur_versor.normalized();

//         ref_bearing_vectors.push_back(ref_versor);
//         cur_bearing_vectors.push_back(cur_versor);

//         tracklets.push_back(corres.tracklet_id_);
//     }

//     RelativePoseAdaptor adapter(ref_bearing_vectors, cur_bearing_vectors);

//     const bool use_2point_mono = params_.ransac_use_2point_mono && R_curr_ref;
//     if(use_2point_mono) {
//         adapter.setR12((*R_curr_ref).matrix());
//     }



//     gtsam::Pose3 best_pose;
//     std::vector<int> ransac_inliers;
//     bool success = false;
//     if(use_2point_mono) {
//         success = runRansac<RelativePoseProblemGivenRot>(
//                 std::make_shared<RelativePoseProblemGivenRot>(
//                     adapter, params_.ransac_randomize
//                 ),
//                 params_.ransac_threshold_mono,
//                 params_.ransac_iterations,
//                 params_.ransac_probability,
//                 params_.optimize_2d2d_pose_from_inliers,
//                 best_pose,
//                 ransac_inliers
//             );
//     }
//     else {
//         success = runRansac<RelativePoseProblem>(
//                 std::make_shared<RelativePoseProblem>(
//                     adapter,
//                     RelativePoseProblem::NISTER,
//                     params_.ransac_randomize
//                 ),
//                 params_,
//                 best_pose,
//                 ransac_inliers
//             );
//     }

//     if(!success) {
//         result.status = TrackingStatus::INVALID;
//     }
//     else {
//         constructTrackletInliers(
//             result.inliers, result.outliers,
//             correspondences,
//             ransac_inliers,
//             tracklets
//         );
//         // NOTE: 2-point always returns the identity rotation, hence we have to
//         // substitute it:
//         if (use_2point_mono) {
//             CHECK(R_curr_ref->equals(best_pose.rotation()));
//         }
//         result.status = TrackingStatus::VALID;
//         result.best_pose = best_pose;

//     }

//     return result;


// }



// Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
//                             Frame::Ptr frame_k_1,
//                             Frame::Ptr frame_k,
//                             std::optional<gtsam::Rot3> R_curr_ref)
// {
//     AbsolutePoseCorrespondences correspondences;
//     //this does not create proper bearing vectors (at leas tnot for 3d-2d pnp solve)
//     //bearing vectors are also not undistorted atm!!
//     //TODO: change to use landmarkWorldProjectedBearingCorrespondance and then change motion solver to take already projected bearing vectors
//     {
//         frame_k->getCorrespondences(correspondences, *frame_k_1, KeyPointType::STATIC, frame_k->landmarkWorldKeypointCorrespondance());
//     }

//     return geometricOutlierRejection3d2d(correspondences, R_curr_ref);
// }

// Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
//                             const AbsolutePoseCorrespondences& correspondences,
//                             std::optional<gtsam::Rot3> R_curr_ref)
// {

//     utils::TimingStatsCollector timer("motion_sovler.solve_3d2d");
//     Pose3SolverResult result;
//     const size_t& n_matches = correspondences.size();

//     if(n_matches < 5u) {
//         result.status = TrackingStatus::FEW_MATCHES;
//         VLOG(5) << "3D2D tracking failed as there are to few matches" << n_matches;
//         return result;
//     }


//     gtsam::Matrix K = camera_params_.getCameraMatrixEigen();
//     K = K.inverse();

//     TrackletIds tracklets, inliers, outliers;
//     //NOTE: currently without distortion! the correspondences should be made into bearing vector elsewhere!
//     BearingVectors bearing_vectors;
//     Landmarks points;
//     for(size_t i = 0u; i < n_matches; i ++) {
//         const AbsolutePoseCorrespondence& corres = correspondences.at(i);
//         const Keypoint& kp = corres.cur_;
//         //make Bearing vector
//         gtsam::Vector3 versor = (K * gtsam::Vector3(kp(0), kp(1), 1.0));
//         versor = versor.normalized();
//         bearing_vectors.push_back(versor);

//         points.push_back(corres.ref_);
//         tracklets.push_back(corres.tracklet_id_);
//     }

//     const double reprojection_error = params_.ransac_threshold_pnp;
//     const double avg_focal_length =
//         0.5 * static_cast<double>(camera_params_.fx() +
//                                   camera_params_.fy());
//     const double threshold =
//         1.0 - std::cos(std::atan(std::sqrt(2.0) * reprojection_error /
//                                  avg_focal_length));


//     AbsolutePoseAdaptor adapter(bearing_vectors, points);

//     gtsam::Pose3 best_pose;
//     std::vector<int> ransac_inliers;

//     bool success = runRansac<AbsolutePoseProblem>(
//         std::make_shared<AbsolutePoseProblem>(
//             adapter, AbsolutePoseProblem::KNEIP
//         ),
//         threshold,
//         params_.ransac_iterations,
//         params_.ransac_probability,
//         params_.optimize_3d2d_pose_from_inliers,
//         best_pose,
//         ransac_inliers
//     );

//     if(success) {
//         constructTrackletInliers(
//             result.inliers, result.outliers,
//             correspondences,
//             ransac_inliers,
//             tracklets
//         );

//         if(result.inliers.size() < 5u) {
//             result.status = TrackingStatus::FEW_MATCHES;
//         }
//         else {
//             result.status = TrackingStatus::VALID;
//             result.best_pose = best_pose;
//         }

//     }
//     else {
//         result.status = TrackingStatus::INVALID;
//     }

//     return result;

// }


// Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d3d(
//                             Frame::Ptr frame_k_1,
//                             Frame::Ptr frame_k,
//                             std::optional<gtsam::Rot3> R_curr_ref)
// {
//     PointCloudCorrespondences correspondences;
//     {
//         utils::TimingStatsCollector("pc_correspondences");
//         frame_k->getCorrespondences(correspondences, *frame_k_1, KeyPointType::STATIC, frame_k->landmarkWorldPointCloudCorrespondance());
//     }

//     return geometricOutlierRejection3d3d(correspondences, R_curr_ref);
// }


// Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d3d(
//                             const PointCloudCorrespondences& correspondences,
//                             std::optional<gtsam::Rot3> R_curr_ref)
// {
//     const size_t& n_matches = correspondences.size();

//     Pose3SolverResult result;
//     if(n_matches < 5) {
//         result.status = TrackingStatus::FEW_MATCHES;
//         return result;
//     }

//     TrackletIds tracklets;
//     BearingVectors ref_bearing_vectors, cur_bearing_vectors;

//     for(size_t i = 0u; i < n_matches; i ++) {
//         const auto& corres = correspondences.at(i);
//         const Landmark& ref_lmk = corres.ref_;
//         const Landmark& cur_lmk = corres.cur_;
//         ref_bearing_vectors.push_back(ref_lmk);
//         cur_bearing_vectors.push_back(cur_lmk);

//         tracklets.push_back(corres.tracklet_id_);
//     }

//     //! Setup adapter.
//     Adapter3d3d adapter(ref_bearing_vectors, cur_bearing_vectors);

//     if(R_curr_ref) {
//         adapter.setR12((*R_curr_ref).matrix());
//     }

//     gtsam::Pose3 best_pose;
//     std::vector<int> ransac_inliers;

//     bool success = runRansac<Problem3d3d>(
//         std::make_shared<Problem3d3d>(
//             adapter, params_.ransac_randomize
//         ),
//         params_.ransac_threshold_stereo,
//         params_.ransac_iterations,
//         params_.ransac_probability,
//         params_.optimize_3d3d_pose_from_inliers,
//         best_pose,
//         ransac_inliers
//     );

//     if(success) {
//         constructTrackletInliers(
//             result.inliers, result.outliers,
//             correspondences,
//             ransac_inliers,
//             tracklets
//         );

//         result.status = TrackingStatus::VALID;
//         result.best_pose = best_pose;
//     }
//     else {
//         result.status = TrackingStatus::INVALID;
//     }

//     return result;
// }


// void EgoMotionSolver::refineJointPoseOpticalFlow(
//         //TODO: solver result could be const!?
//         Pose3SolverResult& solver_result,
//         const Frame::Ptr frame_k_1,
//         const Frame::Ptr frame_k,
//         gtsam::Pose3& refined_pose,
//         gtsam::Point2Vector& refined_flows,
//         TrackletIds& inliers)
// {
//     CHECK(solver_result.status == TrackingStatus::VALID);
//     gtsam::NonlinearFactorGraph graph;
//     gtsam::Values values;

//     gtsam::SharedNoiseModel flow_noise =
//         gtsam::noiseModel::Isotropic::Sigma(2u, 0.1);

//     const static double k_huber_value = 0.0001;
//     //robust noise model!
//     flow_noise = gtsam::noiseModel::Robust::Create(
//             gtsam::noiseModel::mEstimator::Huber::Create(k_huber_value), flow_noise);

//     gtsam::SharedNoiseModel flow_prior_noise =
//         gtsam::noiseModel::Isotropic::Sigma(2u, 0.3);

//     //pose (this might not actually be the camera pose, as we use this to sovle motion too)
//     gtsam::Pose3 pose = solver_result.best_pose;
//     const gtsam::Pose3 T_world_k_1 = frame_k_1->getPose();

//     auto flowSymbol = [](TrackletId tracklet) -> gtsam::Symbol {
//         return gtsam::symbol_shorthand::F(static_cast<uint64_t>(tracklet));
//     };

//     //pose at frame k
//     const gtsam::Symbol pose_key('X', 0);
//     values.insert(pose_key, pose);

//     using Calibration = Camera::CalibrationType;
//     using Pose3FlowProjectionFactorCalib = Pose3FlowProjectionFactor<Calibration>;

//     auto gtsam_calibration = boost::make_shared<Calibration>(
//         frame_k_1->getFrameCamera().calibration());

//     for(TrackletId tracklet_id : solver_result.inliers) {
//         Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
//         Feature::Ptr feature_k = frame_k->at(tracklet_id);

//         CHECK_NOTNULL(feature_k_1);
//         CHECK_NOTNULL(feature_k);

//         CHECK(feature_k_1->hasDepth());
//         CHECK(feature_k->hasDepth());

//         const Keypoint kp_k_1 = feature_k_1->keypoint_;
//         const Depth depth_k_1 = feature_k_1->depth_;

//         const gtsam::Point2 flow = feature_k_1->measured_flow_;
//         CHECK(gtsam::equal(kp_k_1 + flow, feature_k->keypoint_)) << gtsam::Point2(kp_k_1 + flow) << " " << feature_k->keypoint_ << " object id " << feature_k->instance_label_ << " flow " << flow;

//         gtsam::Symbol flow_symbol(flowSymbol(tracklet_id));
//         auto flow_factor = boost::make_shared<Pose3FlowProjectionFactorCalib>(
//             flow_symbol,
//             pose_key,
//             kp_k_1,
//             depth_k_1,
//             T_world_k_1,
//             *gtsam_calibration,
//             flow_noise
//         );
//         graph.add(flow_factor);

//         //add prior factor on each flow
//         graph.addPrior<gtsam::Point2>(flow_symbol, flow, flow_prior_noise);

//         values.insert(flow_symbol, flow);

//     }

//     double error_before = graph.error(values);
//     std::vector<double> post_errors;
//     std::set<gtsam::Symbol> outlier_flows;
//     // std::vector<Pose3FlowProjectionFactorCalib::shared_ptr> outlier_flow_factors;
//     //graph we will mutate by removing outlier factors
//     gtsam::NonlinearFactorGraph mutable_graph = graph;
//     gtsam::Values optimised_values = values;


//     gtsam::LevenbergMarquardtParams opt_params;
//     if(VLOG_IS_ON(200))
//         opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

//     optimised_values = gtsam::LevenbergMarquardtOptimizer(mutable_graph, optimised_values, opt_params).optimize();
//     double error_after = mutable_graph.error(optimised_values);
//     post_errors.push_back(error_after);

//     gtsam::FactorIndices outlier_factors = factor_graph_tools::determineFactorOutliers<Pose3FlowProjectionFactorCalib>(
//         mutable_graph,
//         optimised_values
//     );

//     //if we have outliers, enter iteration loop
//     if(outlier_factors.size() > 0u) {
//         for(size_t itr = 0; itr < 4; itr++) {


//             //currently removing factors from graph makes them nullptr
//             gtsam::NonlinearFactorGraph mutable_graph_with_null = mutable_graph;
//             for(auto outlier_idx : outlier_factors) {
//                 auto factor = mutable_graph_with_null.at(outlier_idx);
//                 gtsam::Symbol flow_symbol = factor->keys()[0];
//                 CHECK_EQ(flow_symbol.chr(), 'f');

//                 outlier_flows.insert(flow_symbol);
//                 mutable_graph_with_null.remove(outlier_idx);
//             }
//             //now iterate over graph and add factors that are not null to ensure all factors are ok
//             mutable_graph.resize(0);
//             for (size_t i = 0; i < mutable_graph_with_null.size(); i++) {

//                 auto factor = mutable_graph_with_null.at(i);
//                 if(factor) {
//                     mutable_graph.add(factor);
//                 }
//             }
//             LOG(INFO) << "Removed " << outlier_factors.size() << " factors on iteration: " << itr;

//             optimised_values.update(pose_key, pose);
//             //do we use values or optimised values here?
//             optimised_values = gtsam::LevenbergMarquardtOptimizer(mutable_graph, optimised_values, opt_params).optimize();
//             error_after = mutable_graph.error(optimised_values);
//             post_errors.push_back(error_after);

//             outlier_factors = factor_graph_tools::determineFactorOutliers<Pose3FlowProjectionFactorCalib>(
//                 mutable_graph,
//                 optimised_values
//             );

//             if(outlier_factors.size() == 0) {
//                 break;
//             }
//         }
//     }


//     size_t initial_size = graph.size();
//     size_t inlier_size = mutable_graph.size();
//     error_after = mutable_graph.error(optimised_values);
//     LOG(INFO) << "Joint optical-flow/pose refinement - error before: "
//         << error_before << " error after: " << error_after
//         << " with initial size " << initial_size << " inlier size " << inlier_size;

//     //recover values!
//     refined_pose = optimised_values.at<gtsam::Pose3>(pose_key);

//     //for each outlier edge, update the set of inliers

//     for(TrackletId tracklet_id : solver_result.inliers) {
//         gtsam::Symbol flow_symbol(flowSymbol(tracklet_id));

//         if(outlier_flows.find(flow_symbol) != outlier_flows.end()) {
//             ///HACK - mark as outlier!!
//             Feature::Ptr feature_k = frame_k->at(tracklet_id);
//             feature_k->inlier_ = false;
//         }
//         else {
//             gtsam::Point2 refined_flow = optimised_values.at<gtsam::Point2>(flow_symbol);
//             refined_flows.push_back(refined_flow);
//             //for now all inliers
//             inliers.push_back(tracklet_id);
//         }

//     }
// }


// ObjectMotionSovler::ObjectMotionSovler(const FrontendParams& params, const CameraParams& camera_params) : EgoMotionSolver(params, camera_params) {}

// Pose3SolverResult ObjectMotionSovler::geometricOutlierRejection3d2d(
//                             Frame::Ptr frame_k_1,
//                             Frame::Ptr frame_k,
//                             const gtsam::Pose3& T_world_k,
//                             ObjectId object_id)
// {
//     utils::TimingStatsCollector timer("motion_solver.object_solve3d2d");
//     AbsolutePoseCorrespondences dynamic_correspondences;
//     // get the corresponding feature pairs
//     bool corr_result = frame_k->getDynamicCorrespondences(
//         dynamic_correspondences,
//         *frame_k_1,
//         object_id,
//         frame_k->landmarkWorldKeypointCorrespondance());
//     //TODO:::
//     // CHECK(corr_result);

//     const size_t& n_matches = dynamic_correspondences.size();

//     Pose3SolverResult geometric_result = EgoMotionSolver::geometricOutlierRejection3d2d(dynamic_correspondences);
//     Pose3SolverResult motion_model_result = motionModelOutlierRejection3d2d(
//         dynamic_correspondences,
//         frame_k_1,
//         frame_k,
//         T_world_k,
//         object_id
//     );

//     Pose3SolverResult result;
//     if(geometric_result.status == TrackingStatus::VALID && motion_model_result.status == TrackingStatus::VALID) {
//         if(geometric_result.inliers.size() >= motion_model_result.inliers.size()) {
//             VLOG(10) << "Geometric model used (inliers " << geometric_result.inliers.size() << " vs " << motion_model_result.inliers.size();
//             result = geometric_result;
//         }
//         else {
//             VLOG(10) << "Motion motion model used (inliers " << motion_model_result.inliers.size() << " vs " << geometric_result.inliers.size();
//             result = motion_model_result;
//         }
//     }
//     //if no motion model, fall back on gemoetric result
//     else {
//         result = geometric_result;
//     }

//     if(result.status == TrackingStatus::VALID) {

//         const gtsam::Pose3 G_w = result.best_pose.inverse();
//         //Use the original result as the input to the refine joint optical flow function
//         //the result.best_pose variable is actually equivalent to ^wG^{-1}
//         //and we want to solve something in the form
//         //e(T, flow) = [u,v]_{k-1} + {k-1}_flow_k - pi(T^{-1}^wm_{k-1})
//         //so T must take the point from k-1 in the world frame to the local frame at k-1
//         //^wG^{-1} = ^wX_k \: {k-1}^wH_k (which takes does this) but the error term uses the inverse of T
//         //hence we must parse in the inverse of G
//         gtsam::Pose3 refined_pose;
//         gtsam::Point2Vector refined_flows;
//         TrackletIds refined_inliers;
//         // this->refineJointPoseOpticalFlow(
//         //     result,
//         //     frame_k_1,
//         //     frame_k,
//         //     refined_pose,
//         //     refined_flows,
//         //     refined_inliers
//         // );

//         // //original flow image that goes from k to k+1 (gross, im sorry!)
//         // const cv::Mat flow_image = frame_k->image_container_.get<ImageType::OpticalFlow>();
//         // const cv::Mat& motion_mask = frame_k->image_container_.get<ImageType::MotionMask>();

//         // //HACK: internally just mark as outlier if it is so!!
//         // //update flow and depth
//         // for(size_t i = 0; i < refined_inliers.size(); i++) {
//         //     TrackletId tracklet_id = refined_inliers.at(i);
//         //     gtsam::Point2 refined_flow = refined_flows.at(i);

//         //     const Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
//         //     Feature::Ptr feature_k = frame_k->at(tracklet_id);

//         //     CHECK_EQ(feature_k->instance_label_, object_id);

//         //     const Keypoint kp_k_1 = feature_k_1->keypoint_;
//         //     Keypoint refined_keypoint = kp_k_1 + refined_flow;

//         //     //check boundaries?

//         //     //update keypoint!!
//         //     feature_k->keypoint_ = refined_keypoint;
//         //     ObjectId predicted_label = functional_keypoint::at<ObjectId>(refined_keypoint, motion_mask);
//         //     if(predicted_label != object_id) {
//         //         feature_k->inlier_ = false;
//         //         //TODO: other fields of the feature does not get updated? Inconsistencies as measured flow, predicted kp
//         //         //etc are no longer correct!!?
//         //         continue;
//         //     }

//         //     //we now have to update the prediced keypoint using the original flow!!
//         //     //TODO: code copied from feature tracker
//         //     const int x = functional_keypoint::u(refined_keypoint);
//         //     const int y = functional_keypoint::v(refined_keypoint);
//         //     double flow_xe = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[0]);
//         //     double flow_ye = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[1]);
//         //     //the measured flow after the origin has been updated
//         //     OpticalFlow new_measured_flow(flow_xe, flow_ye);
//         //     feature_k->measured_flow_ = new_measured_flow;
//         //     // TODO: check predicted flow is within image
//         //     Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(refined_keypoint, new_measured_flow);
//         //     feature_k->predicted_keypoint_ = predicted_kp;


//         //     //woudl need to update the measured flow in frame_k_1 since, right now, flow is k-1 to k
//         //     //TODO:update depth
//         // }
//         // //TODO: this runs over all tracklets so massive waste of time doing this here
//         // //TODO: big refactor api
//         //still need to take the inverse as we get the inverse of G out
//         const gtsam::Pose3 G_w_refined = refined_pose.inverse();
//         const gtsam::Pose3 H_w = T_world_k * G_w;
//         result.best_pose = H_w;

//         //outliers are being marked in the solveObjectMotion
//         // //update inliers/outliers
//         // result.inliers = refined_inliers;
//         // determineOutlierIds(inliers, tracklets, outliers);

//         if(params_.refine_object_motion_esimate) {
//             refineLocalObjectMotionEstimate(
//                 result,
//                 frame_k_1,
//                 frame_k,
//                 object_id
//             );
//         }
//     }

//     //if not valid, return motion result as is
//     return result;

// }


// Pose3SolverResult ObjectMotionSovler::geometricOutlierRejection3d3d(
//                             Frame::Ptr frame_k_1,
//                             Frame::Ptr frame_k,
//                             const gtsam::Pose3& T_world_k,
//                             ObjectId object_id)
// {
//     PointCloudCorrespondences dynamic_correspondences;
//     // get the corresponding feature pairs
//     bool corr_result = frame_k->getDynamicCorrespondences(
//         dynamic_correspondences,
//         *frame_k_1,
//         object_id,
//         frame_k->landmarkWorldPointCloudCorrespondance());

//     const size_t& n_matches = dynamic_correspondences.size();

//     Pose3SolverResult result;
//     result = EgoMotionSolver::geometricOutlierRejection3d3d(dynamic_correspondences);

//     if(result.status == TrackingStatus::VALID) {
//         const gtsam::Pose3 G_w = result.best_pose.inverse();
//         const gtsam::Pose3 H_w = T_world_k * G_w;
//         result.best_pose = H_w;
//     }

//     //if not valid, return motion result as is
//     return result;
// }

// //TODO: dont actually need all these variables
// Pose3SolverResult ObjectMotionSovler::motionModelOutlierRejection3d2d(
//                             const AbsolutePoseCorrespondences& dynamic_correspondences,
//                             Frame::Ptr frame_k_1,
//                             Frame::Ptr frame_k,
//                             const gtsam::Pose3& T_world_k,
//                             ObjectId object_id)
// {
//     Pose3SolverResult result;

//     //get object motions in previous frame (so k-2 to k-1)
//     const MotionEstimateMap& motion_estiamtes_k_1 = frame_k_1->motion_estimates_;
//     if(!motion_estiamtes_k_1.exists(object_id)) {
//         result.status = TrackingStatus::INVALID;
//         return result;
//     }

//     //previous motion model: k-2 to k-1 in w
//     const gtsam::Pose3 motion_model = motion_estiamtes_k_1.at(object_id);

//     using Calibration = Camera::CalibrationType;
//     const auto calibration = camera_params_.constructGtsamCalibration<Calibration>();

//     auto I = gtsam::traits<gtsam::Pose3>::Identity();
//     gtsam::PinholeCamera<Calibration> camera(I, calibration);

//     const double reprojection_error = params_.ransac_threshold_pnp;

//     TrackletIds tracklets;
//     TrackletIds inlier_tracklets;

//     utils::Accumulatord total_repr_error, inlier_repr_error;

//     const size_t& n_matches = dynamic_correspondences.size();
//     for(size_t i = 0u; i < n_matches; i ++) {
//         const AbsolutePoseCorrespondence& corres = dynamic_correspondences.at(i);
//         tracklets.push_back(corres.tracklet_id_);

//         const Keypoint& kp_k = corres.cur_;
//         //the landmark int the world frame at k-1
//         const Landmark& w_lmk_k_1 = corres.ref_;

//         //using the motion, put the lmk in the world frame at k
//         const Landmark w_lmk_k = motion_model * w_lmk_k_1;
//         //using camera pose, put the lmk in the camera frame at k
//         const Landmark c_lmk_c = T_world_k.inverse() * w_lmk_k;

//         try {
//             double repr_error = camera.reprojectionError(c_lmk_c, kp_k).squaredNorm();

//             total_repr_error.Add(repr_error);
//             if(repr_error < reprojection_error) {
//                 inlier_tracklets.push_back(corres.tracklet_id_);
//                 inlier_repr_error.Add(repr_error);
//             }
//         }
//         catch (const gtsam::CheiralityException&) {}

//     }

//     TrackletIds outliers;
//     determineOutlierIds(inlier_tracklets, tracklets, outliers);

//     VLOG(20) << "(Object) motion model inliers/total(error) "
//         << inlier_tracklets.size() << "(" << inlier_repr_error.Mean() << ") / "
//         << tracklets.size() << "(" << total_repr_error.Mean() << ")";

//     result.best_pose = motion_model;
//     result.inliers = inlier_tracklets;
//     result.outliers = outliers;
//     result.status = TrackingStatus::VALID;
//     return result;

// }


// void ObjectMotionSovler::refineLocalObjectMotionEstimate(
//     Pose3SolverResult& solver_result,
//     Frame::Ptr frame_k_1,
//     Frame::Ptr frame_k,
//     ObjectId object_id,
//     const RefinementSolver& solver) const
// {
//     CHECK(solver_result.status == TrackingStatus::VALID);

//     gtsam::NonlinearFactorGraph graph;
//     gtsam::Values values;

//     gtsam::SharedNoiseModel landmark_motion_noise =
//         gtsam::noiseModel::Isotropic::Sigma(3u, 0.001);

//     gtsam::SharedNoiseModel projection_noise =
//         gtsam::noiseModel::Isotropic::Sigma(2u, 2);

//     static constexpr auto k_huber_value = 0.01;

//     //make robust
//     landmark_motion_noise = gtsam::noiseModel::Robust::Create(
//             gtsam::noiseModel::mEstimator::Huber::Create(k_huber_value), landmark_motion_noise);
//     projection_noise = gtsam::noiseModel::Robust::Create(
//             gtsam::noiseModel::mEstimator::Huber::Create(k_huber_value), projection_noise);

//     const gtsam::Key pose_k_1_key = CameraPoseSymbol(frame_k_1->getFrameId());
//     const gtsam::Key pose_k_key = CameraPoseSymbol(frame_k->getFrameId());
//     const gtsam::Key object_motion_key = ObjectMotionSymbol(object_id, frame_k->getFrameId());

//     values.insert(pose_k_1_key, frame_k_1->getPose());
//     values.insert(pose_k_key, frame_k->getPose());
//     values.insert(object_motion_key, solver_result.best_pose);

//     auto gtsam_calibration = boost::make_shared<Camera::CalibrationType>(
//         frame_k_1->getFrameCamera().calibration());


//     auto pose_prior = gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001);
//     graph.addPrior(pose_k_1_key, frame_k_1->getPose(), pose_prior);
//     graph.addPrior(pose_k_key, frame_k->getPose(), pose_prior);

//     utils::TimingStatsCollector timer("motion_solver.object_nlo_refinement");
//     for(TrackletId tracklet_id : solver_result.inliers) {
//         Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
//         Feature::Ptr feature_k = frame_k->at(tracklet_id);

//         CHECK_NOTNULL(feature_k_1);
//         CHECK_NOTNULL(feature_k);

//         CHECK(feature_k_1->hasDepth());
//         CHECK(feature_k->hasDepth());

//         const Keypoint kp_k_1 = feature_k_1->keypoint_;
//         const Keypoint kp_k = feature_k->keypoint_;

//         const gtsam::Point3 lmk_k_1_world = frame_k_1->backProjectToWorld(tracklet_id);
//         const gtsam::Point3 lmk_k_world = frame_k->backProjectToWorld(tracklet_id);

//         const gtsam::Point3 lmk_k_1_local = frame_k_1->backProjectToCamera(tracklet_id);
//         const gtsam::Point3 lmk_k_local = frame_k->backProjectToCamera(tracklet_id);

//         const gtsam::Key lmk_k_1_key = DynamicLandmarkSymbol(frame_k_1->getFrameId(), tracklet_id);
//         const gtsam::Key lmk_k_key = DynamicLandmarkSymbol(frame_k->getFrameId(), tracklet_id);

//         //add initial for points
//         values.insert(lmk_k_1_key, lmk_k_1_world);
//         values.insert(lmk_k_key, lmk_k_world);

//         if(solver == RefinementSolver::PointError) {
//             graph.emplace_shared<PoseToPointFactor>(
//                 pose_k_1_key, //pose key at previous frames
//                 lmk_k_1_key,
//                 lmk_k_1_local,
//                 projection_noise
//             );

//             graph.emplace_shared<PoseToPointFactor>(
//                     pose_k_key, //pose key at current frames
//                     lmk_k_key,
//                     lmk_k_local,
//                     projection_noise
//             );
//         }
//         else if(solver == RefinementSolver::ProjectionError) {
//             graph.emplace_shared<GenericProjectionFactor>(
//                     kp_k_1,
//                     projection_noise,
//                     pose_k_1_key,
//                     lmk_k_1_key,
//                     gtsam_calibration,
//                     false, false
//             );

//             graph.emplace_shared<GenericProjectionFactor>(
//                     kp_k,
//                     projection_noise,
//                     pose_k_key,
//                     lmk_k_key,
//                     gtsam_calibration,
//                     false, false
//             );
//         }

//         graph.emplace_shared<LandmarkMotionTernaryFactor>(
//             lmk_k_1_key,
//             lmk_k_key,
//             object_motion_key,
//             landmark_motion_noise
//         );

//     }

//     double error_before = graph.error(values);
//     // std::vector<double> post_errors;
//     // std::set<TrackletId> outlier_tracks;


//     gtsam::NonlinearFactorGraph mutable_graph = graph;
//     gtsam::Values optimised_values = values;

//      gtsam::LevenbergMarquardtParams opt_params;
//     if(VLOG_IS_ON(200))
//         opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

//     optimised_values = gtsam::LevenbergMarquardtOptimizer(mutable_graph, optimised_values, opt_params).optimize();
//     double error_after = mutable_graph.error(optimised_values);
//     // post_errors.push_back(error_after);

//     // gtsam::FactorIndices outlier_factors = factor_graph_tools::determineFactorOutliers<LandmarkMotionTernaryFactor>(
//     //     mutable_graph,
//     //     optimised_values
//     // );

//     // //if we have outliers, enter iteration loop
//     // if(outlier_factors.size() > 0u) {
//     //     for(size_t itr = 0; itr < 4; itr++) {


//     //         //currently removing factors from graph makes them nullptr
//     //         gtsam::NonlinearFactorGraph mutable_graph_with_null = mutable_graph;
//     //         for(auto outlier_idx : outlier_factors) {
//     //             auto factor = mutable_graph_with_null.at(outlier_idx);
//     //             DynamicPointSymbol point_symbol = factor->keys()[0];
//     //             outlier_tracks.insert(point_symbol.trackletId());
//     //             mutable_graph_with_null.remove(outlier_idx);
//     //         }
//     //         //now iterate over graph and add factors that are not null to ensure all factors are ok
//     //         mutable_graph.resize(0);
//     //         for (size_t i = 0; i < mutable_graph_with_null.size(); i++) {

//     //             auto factor = mutable_graph_with_null.at(i);
//     //             if(factor) {
//     //                 mutable_graph.add(factor);
//     //             }
//     //         }
//     //         // LOG(INFO) << "Removed " << outlier_factors.size() << " factors on iteration: " << itr;

//     //         values.insert(object_motion_key, solver_result.best_pose);
//     //         //do we use values or optimised values here?
//     //         optimised_values = gtsam::LevenbergMarquardtOptimizer(mutable_graph, optimised_values, opt_params).optimize();
//     //         error_after = mutable_graph.error(optimised_values);
//     //         post_errors.push_back(error_after);

//     //         outlier_factors = factor_graph_tools::determineFactorOutliers<LandmarkMotionTernaryFactor>(
//     //             mutable_graph,
//     //             optimised_values
//     //         );

//     //         if(outlier_factors.size() == 0) {
//     //             break;
//     //         }
//     //     }
//     // }


//     // size_t initial_size = graph.size();
//     // size_t inlier_size = mutable_graph.size();
//     // error_after = mutable_graph.error(optimised_values);
//     // LOG(INFO) << "Object Motion refinement - error before: "
//     //     << error_before << " error after: " << error_after
//     //     << " with initial size " << initial_size << " inlier size " << inlier_size;

//     //recover values!
//     solver_result.best_pose = optimised_values.at<gtsam::Pose3>(object_motion_key);

//     //for each outlier edge, update the set of inliers
//     // for(const auto tracklet_id : outlier_tracks) {
//     //     frame_k->at(tracklet_id)->inlier_ = false;

//     // }

// }

// } //dyno

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

#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/utils/TimingStats.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/frontend/MonoInstance-Definitions.hpp"

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/core.hpp>


#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include <opengv/types.hpp>

#include <glog/logging.h>

namespace dyno {

// RelativeObjectMotionSolver::MotionResult RelativeObjectMotionSolver::solve(const GenericCorrespondences<Keypoint, Keypoint>& correspondences) const {

//     const size_t& n_matches = correspondences.size();

//     if(n_matches < 5u) {
//         return MotionResult::NotEnoughCorrespondences();
//     }


//     std::vector<cv::Point2d> ref_kps, curr_kps;
//     TrackletIds tracklets;
//     for(size_t i = 0u; i < n_matches; i ++) {
//         const auto& corres = correspondences.at(i);
//         const Keypoint& ref_kp = corres.ref_;
//         const Keypoint& cur_kp = corres.cur_;

//         ref_kps.push_back(utils::gtsamPointToCv(ref_kp));
//         curr_kps.push_back(utils::gtsamPointToCv(cur_kp));

//         tracklets.push_back(corres.tracklet_id_);
//     }

//     constexpr int method = cv::RANSAC;
//     constexpr double  prob = 0.999;
// 	constexpr double threshold = 1.0;
//     constexpr int max_iterations = 500;
//     const cv::Mat K = camera_params_.getCameraMatrix();

//     cv::Mat ransac_inliers;  // a [1 x N] vector

//     const cv::Mat E = cv::findEssentialMat(
//         ref_kps,
//         curr_kps,
//         K,
//         method,
//         prob,
//         threshold,
//         max_iterations,
//         ransac_inliers); //ransac params?

//     CHECK(ransac_inliers.rows == tracklets.size());

//     cv::Mat R1, R2, t;
//     try {
//         cv::decomposeEssentialMat(E, R1, R2, t);
//     }
//     catch(const cv::Exception& e) {
//         LOG(WARNING) << "decomposeEssentialMat failed with error " << e.what();
//         return MotionResult::Unsolvable();
//     }

//     TrackletIds inliers, outliers;
//     for (int i = 0; i < ransac_inliers.rows; i++)
//     {
//         if(ransac_inliers.at<int>(i)) {
//             const auto& corres = correspondences.at(i);
//             inliers.push_back(corres.tracklet_id_);
//         }

//     }

//     determineOutlierIds(inliers, tracklets, outliers);
//     CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());

//     EssentialDecompositionResult result(
//         utils::cvMatToGtsamRot3(R1),
//         utils::cvMatToGtsamRot3(R2),
//         utils::cvMatToGtsamPoint3(t)
//     );

//     // LOG(INFO) << "Solving RelativeObjectMotionSolver with inliers " << inliers.size() << " outliers " <<  outliers.size();
//     return MotionResult(result, tracklets, inliers, outliers);
// }

EgoMotionSolver::EgoMotionSolver(const FrontendParams& params, const CameraParams& camera_params)
    : params_(params), camera_params_(camera_params) {}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection2d2d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            std::optional<gtsam::Rot3> R_curr_ref)
{
    //get correspondences
    RelativePoseCorrespondences correspondences;
    //this does not create proper bearing vectors (at leas tnot for 3d-2d pnp solve)
    //bearing vectors are also not undistorted atm!!
    {
        utils::TimingStatsCollector track_dynamic_timer("mono_frame_correspondences");
        frame_k->getCorrespondences(correspondences, *frame_k_1, KeyPointType::STATIC, frame_k->imageKeypointCorrespondance());
    }

    Pose3SolverResult result;

    const size_t& n_matches = correspondences.size();

    if(n_matches < 5u) {
        result.status = TrackingStatus::FEW_MATCHES;
        return result;
    }

    gtsam::Matrix K = camera_params_.getCameraMatrixEigen();
    K = K.inverse();

    TrackletIds tracklets;
    //NOTE: currently without distortion! the correspondences should be made into bearing vector elsewhere!
    BearingVectors ref_bearing_vectors, cur_bearing_vectors;
    for(size_t i = 0u; i < n_matches; i ++) {
        const auto& corres = correspondences.at(i);
        const Keypoint& ref_kp = corres.ref_;
        const Keypoint& cur_kp = corres.cur_;

        gtsam::Vector3 ref_versor = (K * gtsam::Vector3(ref_kp(0), ref_kp(1), 1.0));
        gtsam::Vector3 cur_versor = (K * gtsam::Vector3(cur_kp(0), cur_kp(1), 1.0));

        ref_versor = ref_versor.normalized();
        cur_versor = cur_versor.normalized();

        ref_bearing_vectors.push_back(ref_versor);
        cur_bearing_vectors.push_back(cur_versor);

        tracklets.push_back(corres.tracklet_id_);
    }

    RelativePoseAdaptor adapter(ref_bearing_vectors, cur_bearing_vectors);

    const bool use_2point_mono = params_.ransac_use_2point_mono && R_curr_ref;
    if(use_2point_mono) {
        adapter.setR12((*R_curr_ref).matrix());
    }



    gtsam::Pose3 best_pose;
    std::vector<int> ransac_inliers;
    bool success = false;
    if(use_2point_mono) {
        success = runRansac<RelativePoseProblemGivenRot>(
                std::make_shared<RelativePoseProblemGivenRot>(
                    adapter, params_.ransac_randomize
                ),
                params_.ransac_threshold_mono,
                params_.ransac_iterations,
                params_.ransac_probability,
                params_.optimize_2d2d_pose_from_inliers,
                best_pose,
                ransac_inliers
            );
    }
    else {
        success = runRansac<RelativePoseProblem>(
                std::make_shared<RelativePoseProblem>(
                    adapter,
                    RelativePoseProblem::NISTER,
                    params_.ransac_randomize
                ),
                params_,
                best_pose,
                ransac_inliers
            );
    }

    if(!success) {
        result.status = TrackingStatus::INVALID;
    }
    else {
        constructTrackletInliers(
            result.inliers, result.outliers,
            correspondences,
            ransac_inliers,
            tracklets
        );
        // NOTE: 2-point always returns the identity rotation, hence we have to
        // substitute it:
        if (use_2point_mono) {
            CHECK(R_curr_ref->equals(best_pose.rotation()));
        }
        result.status = TrackingStatus::VALID;
        result.best_pose = best_pose;

    }

    return result;


}



Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            std::optional<gtsam::Rot3> R_curr_ref)
{
    AbsolutePoseCorrespondences correspondences;
    //this does not create proper bearing vectors (at leas tnot for 3d-2d pnp solve)
    //bearing vectors are also not undistorted atm!!
    //TODO: change to use landmarkWorldProjectedBearingCorrespondance and then change motion solver to take already projected bearing vectors
    {
        utils::TimingStatsCollector("frame_correspondences");
        frame_k->getCorrespondences(correspondences, *frame_k_1, KeyPointType::STATIC, frame_k->landmarkWorldKeypointCorrespondance());
    }

    return geometricOutlierRejection3d2d(correspondences, R_curr_ref);
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
                            const AbsolutePoseCorrespondences& correspondences,
                            std::optional<gtsam::Rot3> R_curr_ref)
{
    Pose3SolverResult result;
    const size_t& n_matches = correspondences.size();

    if(n_matches < 5u) {
        result.status = TrackingStatus::FEW_MATCHES;
        VLOG(5) << "3D2D tracking failed as there are to few matches" << n_matches;
        return result;
    }


    gtsam::Matrix K = camera_params_.getCameraMatrixEigen();
    K = K.inverse();

    TrackletIds tracklets, inliers, outliers;
    //NOTE: currently without distortion! the correspondences should be made into bearing vector elsewhere!
    BearingVectors bearing_vectors;
    Landmarks points;
    for(size_t i = 0u; i < n_matches; i ++) {
        const AbsolutePoseCorrespondence& corres = correspondences.at(i);
        const Keypoint& kp = corres.cur_;
        //make Bearing vector
        gtsam::Vector3 versor = (K * gtsam::Vector3(kp(0), kp(1), 1.0));
        versor = versor.normalized();
        bearing_vectors.push_back(versor);

        points.push_back(corres.ref_);
        tracklets.push_back(corres.tracklet_id_);
    }

    const double reprojection_error = params_.ransac_threshold_pnp;
    const double avg_focal_length =
        0.5 * static_cast<double>(camera_params_.fx() +
                                  camera_params_.fy());
    const double threshold =
        1.0 - std::cos(std::atan(std::sqrt(2.0) * reprojection_error /
                                 avg_focal_length));


    AbsolutePoseAdaptor adapter(bearing_vectors, points);

    gtsam::Pose3 best_pose;
    std::vector<int> ransac_inliers;

    bool success = runRansac<AbsolutePoseProblem>(
        std::make_shared<AbsolutePoseProblem>(
            adapter, AbsolutePoseProblem::KNEIP
        ),
        threshold,
        params_.ransac_iterations,
        params_.ransac_probability,
        params_.optimize_3d2d_pose_from_inliers,
        best_pose,
        ransac_inliers
    );

    if(success) {
        constructTrackletInliers(
            result.inliers, result.outliers,
            correspondences,
            ransac_inliers,
            tracklets
        );

        result.status = TrackingStatus::VALID;
        result.best_pose = best_pose;
    }
    else {
        result.status = TrackingStatus::INVALID;
    }

    return result;

}


Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d3d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            std::optional<gtsam::Rot3> R_curr_ref)
{
    PointCloudCorrespondences correspondences;
    {
        utils::TimingStatsCollector("pc_correspondences");
        frame_k->getCorrespondences(correspondences, *frame_k_1, KeyPointType::STATIC, frame_k->landmarkWorldPointCloudCorrespondance());
    }

    return geometricOutlierRejection3d3d(correspondences, R_curr_ref);
}


Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d3d(
                            const PointCloudCorrespondences& correspondences,
                            std::optional<gtsam::Rot3> R_curr_ref)
{
    const size_t& n_matches = correspondences.size();

    Pose3SolverResult result;
    if(n_matches < 5) {
        result.status = TrackingStatus::FEW_MATCHES;
        return result;
    }

    TrackletIds tracklets;
    BearingVectors ref_bearing_vectors, cur_bearing_vectors;

    for(size_t i = 0u; i < n_matches; i ++) {
        const auto& corres = correspondences.at(i);
        const Landmark& ref_lmk = corres.ref_;
        const Landmark& cur_lmk = corres.cur_;
        ref_bearing_vectors.push_back(ref_lmk);
        cur_bearing_vectors.push_back(cur_lmk);

        tracklets.push_back(corres.tracklet_id_);
    }

    //! Setup adapter.
    Adapter3d3d adapter(ref_bearing_vectors, cur_bearing_vectors);

    if(R_curr_ref) {
        adapter.setR12((*R_curr_ref).matrix());
    }

    gtsam::Pose3 best_pose;
    std::vector<int> ransac_inliers;

    bool success = runRansac<Problem3d3d>(
        std::make_shared<Problem3d3d>(
            adapter, params_.ransac_randomize
        ),
        params_.ransac_threshold_stereo,
        params_.ransac_iterations,
        params_.ransac_probability,
        params_.optimize_3d3d_pose_from_inliers,
        best_pose,
        ransac_inliers
    );

    if(success) {
        constructTrackletInliers(
            result.inliers, result.outliers,
            correspondences,
            ransac_inliers,
            tracklets
        );

        result.status = TrackingStatus::VALID;
        result.best_pose = best_pose;
    }
    else {
        result.status = TrackingStatus::INVALID;
    }

    return result;
}

ObjectMotionSovler::ObjectMotionSovler(const FrontendParams& params, const CameraParams& camera_params) : EgoMotionSolver(params, camera_params) {}

Pose3SolverResult ObjectMotionSovler::geometricOutlierRejection3d2d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            const gtsam::Pose3& T_world_k,
                            ObjectId object_id)
{
    AbsolutePoseCorrespondences dynamic_correspondences;
    // get the corresponding feature pairs
    bool corr_result = frame_k->getDynamicCorrespondences(
        dynamic_correspondences,
        *frame_k_1,
        object_id,
        frame_k->landmarkWorldKeypointCorrespondance());

    const size_t& n_matches = dynamic_correspondences.size();

    Pose3SolverResult result;
    result = EgoMotionSolver::geometricOutlierRejection3d2d(dynamic_correspondences);

    if(result.status == TrackingStatus::VALID) {
        const gtsam::Pose3 G_w = result.best_pose.inverse();
        const gtsam::Pose3 H_w = T_world_k * G_w;
        result.best_pose = H_w;
    }

    //if not valid, return motion result as is
    return result;

}


Pose3SolverResult ObjectMotionSovler::geometricOutlierRejection3d3d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            const gtsam::Pose3& T_world_k,
                            ObjectId object_id)
{
    PointCloudCorrespondences dynamic_correspondences;
    // get the corresponding feature pairs
    bool corr_result = frame_k->getDynamicCorrespondences(
        dynamic_correspondences,
        *frame_k_1,
        object_id,
        frame_k->landmarkWorldPointCloudCorrespondance());

    const size_t& n_matches = dynamic_correspondences.size();

    Pose3SolverResult result;
    result = EgoMotionSolver::geometricOutlierRejection3d3d(dynamic_correspondences);

    if(result.status == TrackingStatus::VALID) {
        const gtsam::Pose3 G_w = result.best_pose.inverse();
        const gtsam::Pose3 H_w = T_world_k * G_w;
        result.best_pose = H_w;
    }

    //if not valid, return motion result as is
    return result;
}

} //dyno
