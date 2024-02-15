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
#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"


#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>


#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>

#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/point_cloud/methods.hpp>

#include <opengv/sac/Ransac.hpp>

#include <optional>
#include <glog/logging.h>

// PnP (3d2d)
using AbsolutePoseProblem = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
using AbsolutePoseAdaptor = opengv::absolute_pose::CentralAbsoluteAdapter;

// Mono (2d2d) using 5-point ransac
using RelativePoseProblem = opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
// Mono (2d2d, with given rotation) MonoTranslationOnly:
// TranslationOnlySacProblem 2-point ransac
using RelativePoseProblemGivenRot =
    opengv::sac_problems::relative_pose::TranslationOnlySacProblem;
using RelativePoseAdaptor = opengv::relative_pose::CentralRelativeAdapter;

// Stereo (3d3d)
// Arun's problem (3-point ransac)
using Problem3d3d = opengv::sac_problems::point_cloud::PointCloudSacProblem;
using Adapter3d3d = opengv::point_cloud::PointCloudAdapter;

namespace dyno {

template <class SampleConsensusProblem>
bool runRansac(
    std::shared_ptr<SampleConsensusProblem> sample_consensus_problem_ptr,
    const double& threshold,
    const int& max_iterations,
    const double& probability,
    const bool& do_nonlinear_optimization,
    gtsam::Pose3& best_pose,
    std::vector<int>& inliers)
{
    CHECK(sample_consensus_problem_ptr);
    inliers.clear();

     //! Create ransac
    opengv::sac::Ransac<SampleConsensusProblem> ransac(
        max_iterations, threshold, probability);

    //! Setup ransac
    ransac.sac_model_ = sample_consensus_problem_ptr;

    //! Run ransac
    bool success = ransac.computeModel(0);

    if (success) {
      if (ransac.iterations_ >= max_iterations && ransac.inliers_.empty()) {
        success = false;
        best_pose = gtsam::Pose3();
        inliers = {};
      } else {
        best_pose =
            utils::openGvTfToGtsamPose3(ransac.model_coefficients_);
        inliers = ransac.inliers_;


        if (do_nonlinear_optimization) {
          opengv::transformation_t optimized_pose;
          sample_consensus_problem_ptr->optimizeModelCoefficients(
              inliers, ransac.model_coefficients_, optimized_pose);
          best_pose = Eigen::MatrixXd(optimized_pose);
        }
      }
    } else {
      CHECK(ransac.inliers_.empty());
      best_pose = gtsam::Pose3();
      inliers.clear();
    }

    return success;
}


template <class SampleConsensusProblem>
bool runRansac(
    std::shared_ptr<SampleConsensusProblem> sample_consensus_problem_ptr,
    const FrontendParams& params,
    gtsam::Pose3& best_pose,
    std::vector<int>& inliers)
{
    return runRansac<SampleConsensusProblem>(
        sample_consensus_problem_ptr,
        params.ransac_threshold_pnp,
        params.ransac_iterations,
        params.ransac_probability,
        params.optimize_3d2d_pose_from_inliers,
        best_pose,
        inliers
    );
}


class EssentialDecompositionResult; //forward declare

template<typename T>
struct SolverResult {
    T best_pose;
    TrackletIds inliers;
    TrackletIds outliers;
    TrackingStatus status;
    double iterations; //current iterations
    double probability; // current probability
};

using Pose3SolverResult = SolverResult<gtsam::Pose3>;


//TODO: eventually when we have a map, should we look up these values from there (the optimized versions, not the tracked ones?)
class EgoMotionSolver {
public:
    EgoMotionSolver(const FrontendParams& params, const CameraParams& camera_params);
    virtual ~EgoMotionSolver() = default;

    /**
     * @brief Runs 2d-2d PnP with optional Rotation (ie. from IMU)
     *
     * @param frame_k_1
     * @param frame_k
     * @param R_curr_ref Should rotate from ref -> curr
     * @return Pose3SolverResult
     */
    Pose3SolverResult geometricOutlierRejection2d2d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            std::optional<gtsam::Rot3> R_curr_ref = {});

    /**
     * @brief Runs 3d-2d PnP with optional Rotation (i.e from IMU)
     *
     * @param frame_k_1
     * @param frame_k
     * @param R_curr_ref
     * @return Pose3SolverResult
     */
    Pose3SolverResult geometricOutlierRejection3d2d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            std::optional<gtsam::Rot3> R_curr_ref = {});

    Pose3SolverResult geometricOutlierRejection3d2d(
                            const AbsolutePoseCorrespondences& correspondences,
                            std::optional<gtsam::Rot3> R_curr_ref = {});


    Pose3SolverResult geometricOutlierRejection3d3d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            std::optional<gtsam::Rot3> R_curr_ref = {});

    Pose3SolverResult geometricOutlierRejection3d3d(
                            const PointCloudCorrespondences& correspondences,
                            std::optional<gtsam::Rot3> R_curr_ref = {});


protected:
    template<typename Ref, typename Curr>
    void constructTrackletInliers(TrackletIds& inliers, TrackletIds& outliers,
        const GenericCorrespondences<Ref, Curr>& correspondences,
        const std::vector<int>& ransac_inliers,
        const TrackletIds tracklets)
    {

        CHECK_EQ(correspondences.size(), tracklets.size());
        CHECK(ransac_inliers.size() <= correspondences.size());
        for(int inlier_idx : ransac_inliers) {
            const auto& corres = correspondences.at(inlier_idx);
            inliers.push_back(corres.tracklet_id_);
        }

        determineOutlierIds(inliers, tracklets, outliers);
        CHECK_EQ((inliers.size() + outliers.size()), tracklets.size());
        CHECK_EQ(inliers.size(), ransac_inliers.size());
    }

protected:
    const FrontendParams params_;
    const CameraParams camera_params_;

};

class ObjectMotionSovler : protected EgoMotionSolver {

public:
    ObjectMotionSovler(const FrontendParams& params, const CameraParams& camera_params);

    Pose3SolverResult geometricOutlierRejection3d2d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            const gtsam::Pose3& T_world_k,
                            ObjectId object_id);

    Pose3SolverResult geometricOutlierRejection3d3d(
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            const gtsam::Pose3& T_world_k,
                            ObjectId object_id);



};



} //dyno
