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

#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/frontend/MonoInstance-Definitions.hpp"

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/core.hpp>

#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>

#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/point_cloud/methods.hpp>
#include <opengv/types.hpp>

#include <glog/logging.h>

namespace dyno {

namespace motion_solver_tools {

/**
 * @brief Specalisation for 3D -> 2D absolute pose solving method
 *
 * @tparam
 * @param correspondences
 * @param params
 * @param camera_params
 * @return MotionResult
 */
template<>
MotionSolverResult<gtsam::Pose3> solveMotion(const GenericCorrespondences<Landmark, Keypoint>& correspondences, const FrontendParams& params, const CameraParams& camera_params) {
    using MotionResult = MotionSolverResult<gtsam::Pose3>;
    const size_t& n_matches = correspondences.size();

    if(n_matches < 5u) {
        return MotionResult::NotEnoughCorrespondences();
    }

    gtsam::Matrix K = gtsam::Matrix::Identity(3, 3);
    cv::cv2eigen(camera_params.getCameraMatrix(), K);

    K = K.inverse();

    TrackletIds tracklets, inliers, outliers;
    //NOTE: currently without distortion! the correspondances should be made into bearing vector elsewhere!
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

    opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors, points );

    // create a Ransac object
    AbsolutePoseSacProblem ransac;
    // create an AbsolutePoseSacProblem
    // (algorithm is selectable: KNEIP, GAO, or EPNP)
    std::shared_ptr<AbsolutePoseProblem>
        absposeproblem_ptr(
        new AbsolutePoseProblem(
        adapter, AbsolutePoseProblem::KNEIP ) );

    // LOG(INFO) << "Solving ransac";
    // run ransac
    ransac.sac_model_ = absposeproblem_ptr;
    //https://github.com/laurentkneip/opengv/issues/121
    ransac.threshold_ = 2.0*(1.0 - cos(atan(sqrt(2.0)*0.5/800.0)));
    // ransac.threshold_ = .5*(1.0 - cos(atan(sqrt(2.0)*0.5/800.0)));
    // LOG(INFO) << "Solving ransac";
    ransac.max_iterations_ = 500;
    if(!ransac.computeModel(0)) {
        LOG(WARNING) << "Could not compute ransac mode";
        return MotionResult::Unsolvable();
    }

    // LOG(INFO) << "here";
    // // get the result
    opengv::transformation_t best_transformation =
        ransac.model_coefficients_;

    gtsam::Pose3 opengv_transform = utils::openGvTfToGtsamPose3(best_transformation);


    CHECK(ransac.inliers_.size() <= correspondences.size());
    for(int inlier_idx : ransac.inliers_) {
        const auto& corres = correspondences.at(inlier_idx);
        inliers.push_back(corres.tracklet_id_);
    }

    determineOutlierIds(inliers, tracklets, outliers);
    CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());
    CHECK_EQ(inliers.size(), ransac.inliers_.size());


    return MotionSolverResult<gtsam::Pose3>(opengv_transform, tracklets, inliers, outliers);
}



/**
 * @brief Specalisation for 2D -> 2D relative pose solving method
 *
 * @tparam
 * @param correspondences
 * @param params
 * @param camera_params
 * @return MotionResult
 */
template<>
MotionSolverResult<gtsam::Pose3> solveMotion(const GenericCorrespondences<Keypoint, Keypoint>& correspondences, const FrontendParams& params, const CameraParams& camera_params) {
    using MotionResult = MotionSolverResult<gtsam::Pose3>;
    const size_t& n_matches = correspondences.size();

    if(n_matches < 5u) {
        return MotionResult::NotEnoughCorrespondences();
    }

    gtsam::Matrix K = gtsam::Matrix::Identity(3, 3);
    cv::cv2eigen(camera_params.getCameraMatrix(), K);

    K = K.inverse();

    TrackletIds tracklets, inliers, outliers;
    //NOTE: currently without distortion! the correspondances should be made into bearing vector elsewhere!
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

    opengv::relative_pose::CentralRelativeAdapter adapter(ref_bearing_vectors, cur_bearing_vectors );

    // create a Ransac object
    RelativePoseSacProblem ransac;
    std::shared_ptr<RelativePoseProblem>
        relposeproblem_ptr(
        new RelativePoseProblem(
            adapter,
            RelativePoseProblem::NISTER
            )
        );
    // LOG(INFO) << "Solving ransac";
    // run ransac
    ransac.sac_model_ = relposeproblem_ptr;
    //https://github.com/laurentkneip/opengv/issues/121
    // ransac.threshold_ = 1.0 - cos(atan(sqrt(2.0)*0.5/800.0));
    ransac.threshold_ = 2.0*(1.0 - cos(atan(sqrt(2.0)*0.5/800.0)));
    // LOG(INFO) << "Solving ransac";
    ransac.max_iterations_ = 500;
    if(!ransac.computeModel(0)) {
        LOG(WARNING) << "Could not compute ransac mode";
        return MotionResult::Unsolvable();
    }

    // LOG(INFO) << "here";
    // // get the result
    opengv::transformation_t best_transformation =
        ransac.model_coefficients_;

    gtsam::Pose3 opengv_transform = utils::openGvTfToGtsamPose3(best_transformation);


    CHECK(ransac.inliers_.size() <= correspondences.size());
    for(int inlier_idx : ransac.inliers_) {
        const auto& corres = correspondences.at(inlier_idx);
        inliers.push_back(corres.tracklet_id_);
    }

    determineOutlierIds(inliers, tracklets, outliers);
    CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());
    CHECK_EQ(inliers.size(), ransac.inliers_.size());


    return MotionResult(opengv_transform, tracklets, inliers, outliers);
}

// /**
//  * @brief Specalisation for 3D -> 3D using the point cloud alignment method
//  *
//  * @tparam
//  * @param correspondences Should be points expressed in the ref and current frames
//  * @param params
//  * @param camera_params
//  * @return MotionResult
//  */
// template<>
// MotionResult solveMotion(const GenericCorrespondences<Landmark, Landmark>& correspondences, const FrontendParams& params, const CameraParams& camera_params) {
//     const size_t& n_matches = correspondences.size();

//     if(n_matches < 3u) {
//         return MotionResult::NotEnoughCorrespondences();
//     }

//     TrackletIds tracklets, inliers, outliers;
//     //NOTE: currently without distortion! the correspondances should be made into bearing vector elsewhere!
//     Landmarks points_ref, point_curr;
//     for(size_t i = 0u; i < n_matches; i ++) {
//         const auto& corres = correspondences.at(i);

//         points_ref.push_back(corres.ref_);
//         point_curr.push_back(corres.cur_);
//         tracklets.push_back(corres.tracklet_id_);
//     }

//     // create a 3D-3D adapter
//     opengv::point_cloud::PointCloudAdapter adapter(
//         points_ref, point_curr );
//     // create a RANSAC object
//     opengv::sac::Ransac<opengv::sac_problems::point_cloud::PointCloudSacProblem> ransac;
//     // create the sample consensus problem
//     std::shared_ptr<opengv::sac_problems::point_cloud::PointCloudSacProblem>
//         relposeproblem_ptr(
//         new opengv::sac_problems::point_cloud::PointCloudSacProblem(adapter) );
//     // run ransac
//     ransac.sac_model_ = relposeproblem_ptr;
//     // ransac.threshold_ = threshold;
//     // ransac.max_iterations_ = maxIterations;
//     ransac.computeModel(0);

//     transformation_t best_transformation =
//         ransac.model_coefficients_;

//     gtsam::Pose3 opengv_transform = utils::openGvTfToGtsamPose3(best_transformation);

//     CHECK(ransac.inliers_.size() <= correspondences.size());
//     for(int inlier_idx : ransac.inliers_) {
//         const auto& corres = correspondences.at(inlier_idx);
//         inliers.push_back(corres.tracklet_id_);
//     }

//     determineOutlierIds(inliers, tracklets, outliers);
//     CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());
//     CHECK_EQ(inliers.size(), ransac.inliers_.size());

//     MotionResult result(opengv_transform);
//     result.inliers = inliers;
//     result.outliers = outliers;
//     result.ids_used = tracklets;
//     return result;
// }

} // motion_solver_tools


AbsoluteCameraMotionSolver::MotionResult AbsoluteCameraMotionSolver::solve(const GenericCorrespondences<Landmark, Keypoint>& correspondences) const {
    return motion_solver_tools::solveMotion<ResultType, Landmark, Keypoint>(correspondences, params_, camera_params_);
}

template<>
MotionSolverResult<gtsam::Pose3> AbsoluteObjectMotionSolver<Landmark, Keypoint>::solve(const GenericCorrespondences<Landmark, Keypoint>& correspondences) const {
    const MotionResult result = motion_solver_tools::solveMotion<ResultType, Landmark, Keypoint>(correspondences, params_, camera_params_);
        if(result.valid()) {
            const gtsam::Pose3 G_w = result.get().inverse();
            const gtsam::Pose3 H_w = T_world_camera_ * G_w;
            return MotionResult(H_w, result.ids_used_, result.inliers_, result.outliers_);
        }

        //if not valid, return motion result as is
        return result;
}

RelativeCameraMotionSolver::MotionResult RelativeCameraMotionSolver::solve(const GenericCorrespondences<Keypoint, Keypoint>& correspondences) const {
    return motion_solver_tools::solveMotion<ResultType, Keypoint, Keypoint>(correspondences, params_, camera_params_);
}

RelativeObjectMotionSolver::MotionResult RelativeObjectMotionSolver::solve(const GenericCorrespondences<Keypoint, Keypoint>& correspondences) const {

    const size_t& n_matches = correspondences.size();

    if(n_matches < 5u) {
        return MotionResult::NotEnoughCorrespondences();
    }


    std::vector<cv::Point2d> ref_kps, curr_kps;
    TrackletIds tracklets;
    for(size_t i = 0u; i < n_matches; i ++) {
        const auto& corres = correspondences.at(i);
        const Keypoint& ref_kp = corres.ref_;
        const Keypoint& cur_kp = corres.cur_;

        ref_kps.push_back(utils::gtsamPointToCv(ref_kp));
        curr_kps.push_back(utils::gtsamPointToCv(cur_kp));

        tracklets.push_back(corres.tracklet_id_);
    }

    constexpr int method = cv::RANSAC;
    constexpr double  prob = 0.999;
	constexpr double threshold = 1.0;
    constexpr int max_iterations = 500;
    const cv::Mat K = camera_params_.getCameraMatrix();

    cv::Mat ransac_inliers;  // a [1 x N] vector

    const cv::Mat E = cv::findEssentialMat(
        ref_kps,
        curr_kps,
        K,
        method,
        prob,
        threshold,
        max_iterations,
        ransac_inliers); //ransac params?

    CHECK(ransac_inliers.rows == tracklets.size());

    cv::Mat R1, R2, t;
    try {
        cv::decomposeEssentialMat(E, R1, R2, t);
    }
    catch(const cv::Exception& e) {
        LOG(WARNING) << "decomposeEssentialMat failed with error " << e.what();
        return MotionResult::Unsolvable();
    }

    TrackletIds inliers, outliers;
    for (int i = 0; i < ransac_inliers.rows; i++)
    {
        if(ransac_inliers.at<int>(i)) {
            const auto& corres = correspondences.at(i);
            inliers.push_back(corres.tracklet_id_);
        }

    }

    determineOutlierIds(inliers, tracklets, outliers);
    CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());

    EssentialDecompositionResult result(
        utils::cvMatToGtsamRot3(R1),
        utils::cvMatToGtsamRot3(R2),
        utils::cvMatToGtsamPoint3(t)
    );

    // LOG(INFO) << "Solving RelativeObjectMotionSolver with inliers " << inliers.size() << " outliers " <<  outliers.size();
    return MotionResult(result, tracklets, inliers, outliers);
}

} //dyno
