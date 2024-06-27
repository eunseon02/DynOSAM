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
#include "dynosam/utils/Numerical.hpp"
#include "dynosam/frontend/MonoInstance-Definitions.hpp"

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/core.hpp>


#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"

#include <opengv/types.hpp>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h> //for now?
#include <gtsam/nonlinear/Values.h>


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

        if(result.inliers.size() < 5u) {
            result.status = TrackingStatus::FEW_MATCHES;
        }
        else {
            result.status = TrackingStatus::VALID;
            result.best_pose = best_pose;
        }

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
    //TODO:::
    // CHECK(corr_result);

    const size_t& n_matches = dynamic_correspondences.size();

    Pose3SolverResult result;
    result = EgoMotionSolver::geometricOutlierRejection3d2d(dynamic_correspondences);

    if(result.status == TrackingStatus::VALID) {
        const gtsam::Pose3 G_w = result.best_pose.inverse();
        const gtsam::Pose3 H_w = T_world_k * G_w;
        result.best_pose = H_w;

        if(params_.refine_object_motion_esimate) {
            refineLocalObjectMotionEstimate(
                result,
                frame_k_1,
                frame_k,
                object_id
            );
        }
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

void ObjectMotionSovler::refineLocalObjectMotionEstimate(
    Pose3SolverResult& solver_result,
    Frame::Ptr frame_k_1,
    Frame::Ptr frame_k,
    ObjectId object_id) const
{
    CHECK(solver_result.status == TrackingStatus::VALID);

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    auto gtsam_calibration = boost::make_shared<Camera::CalibrationType>(
        frame_k_1->getFrameCamera().calibration());

    gtsam::SharedNoiseModel landmark_motion_noise =
        gtsam::noiseModel::Isotropic::Sigma(3u, 0.001);

    gtsam::SharedNoiseModel projection_noise =
        gtsam::noiseModel::Isotropic::Sigma(2u, 2);
    // gtsam::SharedNoiseModel projection_noise =
    //     gtsam::noiseModel::Isotropic::Sigma(3u, 0.01);

    static constexpr auto k_huber_value = 0.0001;

    //make robust
    landmark_motion_noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(k_huber_value), landmark_motion_noise);
    projection_noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(k_huber_value), projection_noise);

    const gtsam::Key pose_k_1_key = CameraPoseSymbol(frame_k_1->getFrameId());
    const gtsam::Key pose_k_key = CameraPoseSymbol(frame_k->getFrameId());
    const gtsam::Key object_motion_key = ObjectMotionSymbol(object_id, frame_k->getFrameId());

    values.insert(pose_k_1_key, frame_k_1->getPose());
    values.insert(pose_k_key, frame_k->getPose());
    values.insert(object_motion_key, solver_result.best_pose);


    auto pose_prior = gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001);
    graph.addPrior(pose_k_1_key, frame_k_1->getPose(), pose_prior);
    graph.addPrior(pose_k_key, frame_k->getPose(), pose_prior);


    for(TrackletId tracklet_id : solver_result.inliers) {
        Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
        Feature::Ptr feature_k = frame_k->at(tracklet_id);

        CHECK_NOTNULL(feature_k_1);
        CHECK_NOTNULL(feature_k);

        CHECK(feature_k_1->hasDepth());
        CHECK(feature_k->hasDepth());

        const Keypoint kp_k_1 = feature_k_1->keypoint_;
        const Keypoint kp_k = feature_k->keypoint_;

        const gtsam::Point3 lmk_k_1_world = frame_k_1->backProjectToWorld(tracklet_id);
        const gtsam::Point3 lmk_k_world = frame_k->backProjectToWorld(tracklet_id);

        const gtsam::Point3 lmk_k_1_local = frame_k_1->backProjectToCamera(tracklet_id);
        const gtsam::Point3 lmk_k_local = frame_k->backProjectToCamera(tracklet_id);

        const gtsam::Key lmk_k_1_key = DynamicLandmarkSymbol(frame_k_1->getFrameId(), tracklet_id);
        const gtsam::Key lmk_k_key = DynamicLandmarkSymbol(frame_k->getFrameId(), tracklet_id);

        //add initial for points
        values.insert(lmk_k_1_key, lmk_k_1_world);
        values.insert(lmk_k_key, lmk_k_world);

        // graph.emplace_shared<PoseToPointFactor>(
        //         pose_k_1_key, //pose key at previous frames
        //         lmk_k_1_key,
        //         lmk_k_1_local,
        //         projection_noise
        //     );

        // graph.emplace_shared<PoseToPointFactor>(
        //         pose_k_key, //pose key at current frames
        //         lmk_k_key,
        //         lmk_k_local,
        //         projection_noise
        //     );

        graph.emplace_shared<GenericProjectionFactor>(
                kp_k_1,
                projection_noise,
                pose_k_1_key,
                lmk_k_1_key,
                gtsam_calibration,
                false, false
        );

        graph.emplace_shared<GenericProjectionFactor>(
                kp_k,
                projection_noise,
                pose_k_key,
                lmk_k_key,
                gtsam_calibration,
                false, false
        );

        graph.emplace_shared<LandmarkMotionTernaryFactor>(
            lmk_k_1_key,
            lmk_k_key,
            object_motion_key,
            landmark_motion_noise
        );

    }

    LOG(INFO) << "Constructing local object motion refinement optimisation for object " << object_id;
    gtsam::LevenbergMarquardtParams opt_params;
    if(VLOG_IS_ON(200))
        opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

    double error_before = graph.error(values);
    const gtsam::Values optimised_values = gtsam::LevenbergMarquardtOptimizer(graph, values, opt_params).optimize();
    double error_after = graph.error(optimised_values);

    static constexpr auto kConfidence = 0.99;
    gtsam::FactorIndices outlier_indicies;
    for (size_t k = 0; k < graph.size(); k++) {
      if (graph[k]) {
        double weighted_threshold = 0.5 * chi_squared_quantile(graph[k]->dim(), kConfidence); // 0.5 derives from the error definition in gtsam

        //mark as outlier
        if(weighted_threshold < graph[k]->error(optimised_values)) {
            outlier_indicies.push_back(k);
        }
      }
    }

    LOG(INFO)
        << "Marked " << outlier_indicies.size() << " factors are outliers out of " << graph.size()
        << " error before=" << error_before << ", error after=" << error_after;

    //TODO: right now ignoring outliers
    solver_result.best_pose = optimised_values.at<gtsam::Pose3>(object_motion_key);

    //recover values
    // for(TrackletId tracklet_id : solver_result.inliers) {
    //     const gtsam::Key lmk_k_1_key = DynamicLandmarkSymbol(frame_k_1->getFrameId(), tracklet_id);
    //     const gtsam::Key lmk_k_key = DynamicLandmarkSymbol(frame_k->getFrameId(), tracklet_id);

    //     // const gtsam::Point3 lmk_k_1_world_recovered = frame_k_1->backProjectToWorld(tracklet_id);
    //     // const gtsam::Point3 lmk_k_world = frame_k->backProjectToWorld(tracklet_id);
    //     //
    // }

}

} //dyno
