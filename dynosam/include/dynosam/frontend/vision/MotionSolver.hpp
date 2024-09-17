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

#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/Pose3FlowProjectionFactor.h"


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
    T best_result; //TODO: rename as not always pose!!
    TrackletIds inliers;
    TrackletIds outliers;
    TrackingStatus status;
};

using Pose3SolverResult = SolverResult<gtsam::Pose3>;


struct OpticalFlowAndPoseOptimizerParams {
    double flow_sigma{10.0};
    double flow_prior_sigma{3.33};
    double k_huber{0.001};
    bool outlier_reject{true};
    //When true, this indicates that the optical flow images go from k to k+1 (rather than k-1 to k, when false)
    //this left over from some original implementations.
    //This param is used when updated the frames after optimization
    bool flow_is_future{true};

    //thresholds to use when updating the depth
    //TODO: we should not actually need to update the depth like this,
    //better for the frame to cache itself and we just clear the depth cache
    double max_background_depth_threshold{40};
    double max_object_depth_threshold{25};
};


/**
 * @brief Joinly refines optical flow with with the given pose
 * using the error term:
 * e = [u,v]_{k_1} + f_{k-1, k} - \pi(X^{-1} \: m_k)
 * where f is flow, [u,v]_{k-1} is the observed keypoint at k-1, X is the pose
 * and m_k is the back-projected keypoint at k.
 *
 * The parsed tracklets are the set of correspondances with which to build the optimisation problem
 * and the refined inliers will be a subset of these tracklets. THe number of refined flows
 * should be the number of refined inliers and be a 1-to-1 match
 *
 */
class OpticalFlowAndPoseOptimizer {

public:
    struct ResultType {
        gtsam::Pose3 refined_pose;
        gtsam::Point2Vector refined_flows;
        ObjectId object_id;
    };
    using Result =  SolverResult<ResultType>;

    OpticalFlowAndPoseOptimizer(const OpticalFlowAndPoseOptimizerParams& params) : params_(params) {}

    /**
     * @brief Builds the factor-graph problem using the set of specificed correspondences (tracklets)
     * in frame k-1 and k and the initial pose.
     *
     * The optimisation joinly refines optical flow with with the given pose
     * using the error term:
     * e = [u,v]_{k_1} + f_{k-1, k} - \pi(X^{-1} \: m_k)
     * where f is flow, [u,v]_{k-1} is the observed keypoint at k-1, X is the pose
     * and m_k is the back-projected keypoint at k.
     *
     * The parsed tracklets are the set of correspondances with which to build the optimisation problem
     * and the refined inliers will be a subset of these tracklets. THe number of refined flows
     * should be the number of refined inliers and be a 1-to-1 match
     *
     * This is agnostic to if the problem is solving for a motion or a pose so the user must make sure the initial pose
     * is in the right form.
     *
     * @tparam CALIBRATION
     * @param frame_k_1
     * @param frame_k
     * @param tracklets
     * @param initial_pose
     * @return Result
     */
    template<typename CALIBRATION>
    Result optimize(
        const Frame::Ptr frame_k_1,
        const Frame::Ptr frame_k,
        const TrackletIds& tracklets,
        const gtsam::Pose3& initial_pose) const;

    /**
     * @brief Builds the factor-graph problem using the set of specificed correspondences (tracklets)
     * in frame k-1 and k and the initial pose.
     * Unlike the optimize only version this also update the features within the frames as outliers
     * after optimisation. It will also update the feature data (depth keypoint etc...) with the refined flows.
     *
     * It will NOT update the frame with the result pose as this could be any pose.
     *
     * @tparam CALIBRATION
     * @param frame_k_1
     * @param frame_k
     * @param tracklets
     * @param initial_pose
     * @return Result
     */
    template<typename CALIBRATION>
    Result optimizeAndUpdate(
        Frame::Ptr frame_k_1,
        Frame::Ptr frame_k,
        const TrackletIds& tracklets,
        const gtsam::Pose3& initial_pose) const;

private:
    void updateFrameOutliersWithResult(const Result& result, Frame::Ptr frame_k_1, Frame::Ptr frame_k) const;

private:
    OpticalFlowAndPoseOptimizerParams params_;

};

struct MotionOnlyRefinementOptimizerParams {
    double landmark_motion_sigma{0.001};
    double projection_sigma{2.0};
    double k_huber{0.0001};
    bool outlier_reject{true};
};

class MotionOnlyRefinementOptimizer {

public:
    MotionOnlyRefinementOptimizer(const MotionOnlyRefinementOptimizerParams& params) : params_(params) {}
    enum RefinementSolver { ProjectionError, PointError };

    template<typename CALIBRATION>
    Pose3SolverResult optimize(
        const Frame::Ptr frame_k_1,
        const Frame::Ptr frame_k,
        const TrackletIds& tracklets,
        const ObjectId object_id,
        const gtsam::Pose3& initial_motion,
        const RefinementSolver& solver = RefinementSolver::ProjectionError) const;

    template<typename CALIBRATION>
    Pose3SolverResult optimizeAndUpdate(
        Frame::Ptr frame_k_1,
        Frame::Ptr frame_k,
        const TrackletIds& tracklets,
        const ObjectId object_id,
        const gtsam::Pose3& initial_motion,
        const RefinementSolver& solver = RefinementSolver::ProjectionError) const;

private:
    MotionOnlyRefinementOptimizerParams params_;

};



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


    // //TODO: refactor API using the result struct
    // void refineJointPoseOpticalFlow(
    //     Pose3SolverResult& solver_result,
    //     const Frame::Ptr frame_k_1,
    //     const Frame::Ptr frame_k,
    //     gtsam::Pose3& refined_pose,
    //     gtsam::Point2Vector& refined_flows,
    //     TrackletIds& inliers //inlier set from the inliers in result
    // );

    // /**
    //  * @brief Joinly refines optical flow with with the given pose
    //  * using the error term:
    //  * e = [u,v]_{k_1} + f_{k-1, k} - \pi(X^{-1} \: m_k)
    //  * where f is flow, [u,v]_{k-1} is the observed keypoint at k-1, X is the pose
    //  * and m_k is the back-projected keypoint at k.
    //  *
    //  * The parsed tracklets are the set of correspondances with which to build the optimisation problem
    //  * and the refined inliers will be a subset of these tracklets. THe number of refined flows
    //  * should be the number of refined inliers and be a 1-to-1 match
    //  *
    //  *
    //  * @param frame_k_1
    //  * @param frame_k
    //  * @param tracklets
    //  * @param initial_pose
    //  * @param refined_pose
    //  * @param refined_flows
    //  * @param refined_inliers
    //  * @param refined_outliers
    //  */
    // static void jointRefinePoseOpticalFlow(
    //     const Frame::Ptr frame_k_1,
    //     const Frame::Ptr frame_k,
    //     const TrackletIds& tracklets,
    //     const gtsam::Pose3& initial_pose,
    //     gtsam::Pose3& refined_pose,
    //     gtsam::Point2Vector& refined_flows,
    //     TrackletIds& refined_inliers,
    //     TrackletIds* refined_outliers = nullptr;
    // );



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

    Pose3SolverResult motionModelOutlierRejection3d2d(
                            const AbsolutePoseCorrespondences& dynamic_correspondences,
                            Frame::Ptr frame_k_1,
                            Frame::Ptr frame_k,
                            const gtsam::Pose3& T_world_k,
                            ObjectId object_id);

protected:




    // //assumes we have updated frames with latest pose (camera) and the result is from the geometricOutlierReject function
    // //only works with stereo
    // void refineLocalObjectMotionEstimate(
    //                         Pose3SolverResult& solver_result,
    //                         Frame::Ptr frame_k_1,
    //                         Frame::Ptr frame_k,
    //                         ObjectId object_id,
    //                         const RefinementSolver& solver = RefinementSolver::ProjectionError) const;






};



} //dyno


#include "dynosam/frontend/vision/MotionSolver-inl.hpp"
