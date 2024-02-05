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
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"


#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac/Ransac.hpp>

#include <optional>
#include <glog/logging.h>

using AbsolutePoseProblem = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
using AbsolutePoseSacProblem = opengv::sac::Ransac<AbsolutePoseProblem>;

using RelativePoseProblem = opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
using RelativePoseSacProblem = opengv::sac::Ransac<RelativePoseProblem>;

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
        params.ransac_threshold,
        params.ransac_iterations,
        params.ransac_probability,
        params.optimize_3d3d_pose_from_inliers,
        best_pose,
        inliers
    );
}



template<typename Result>
class MotionSolverResult : public std::optional<Result> {
public:
    using Base = std::optional<Result>;
    using This = MotionSolverResult<Result>;
    enum Status { VALID, NOT_ENOUGH_CORRESPONDENCES, NOT_ENOUGH_INLIERS, UNSOLVABLE };
    Status status;

    TrackletIds ids_used_;
    TrackletIds inliers_;
    TrackletIds outliers_;

private:
    MotionSolverResult(Status s) : status(s) {};

public:
    MotionSolverResult() {}

    MotionSolverResult(
        const Result& result,
        const TrackletIds& ids_used,
        const TrackletIds& inliers,
        const TrackletIds& outliers)
    : status(VALID),
      ids_used_(ids_used),
      inliers_(inliers),
      outliers_(outliers)
    {
        Base::emplace(result);

        {
            //debug check
            //inliers and outliers together should be the set of ids_used
            CHECK_EQ(ids_used_.size(), inliers_.size() + outliers_.size());
        }
    }

    operator const Result&() const { return get(); }

    static This NotEnoughCorrespondences() { return This(NOT_ENOUGH_CORRESPONDENCES); }
    static This NotEnoughInliers() { return This(NOT_ENOUGH_INLIERS); }
    static This Unsolvable() { return This(UNSOLVABLE); }

    inline bool valid() const { return status == VALID; }
    inline bool notEnoughCorrespondences() const { return status == NOT_ENOUGH_CORRESPONDENCES; }
    inline bool notEnoughInliers() const { return status == NOT_ENOUGH_INLIERS; }
    inline bool unsolvable() const { return status == UNSOLVABLE; }

    const Result& get() const {
        if (!Base::has_value()) throw std::runtime_error("MotionSolverResult has no value");
        return Base::value();
    }

};


namespace motion_solver_tools {

/**
 * @brief Generic Motion solver free function called from the MotionSolver class
 *
 * Expects the result to be in left hand side multiply form so if solving for camera pose, the pose should be T_world_camera
 * where P_world = T_world_camera * P_camera
 *
 * @tparam RefType
 * @tparam CurrType
 * @param correspondences
 * @param params
 * @param camera_params
 * @return MotionResult
 */
template<typename Result, typename RefType, typename CurrType>
MotionSolverResult<Result> solveMotion(const GenericCorrespondences<RefType, CurrType>& correspondences, const FrontendParams& params, const CameraParams& camera_params);

} //motion_solver_tools




template<typename Result, typename RefType, typename CurrType>
class MotionSolverBase {

public:
    using ResultType = Result;
    using ReferenceType = RefType;
    using CurrentType = CurrType;
    using MotionResult = MotionSolverResult<ResultType>;

    MotionSolverBase(const FrontendParams& params, const CameraParams& camera_params) : params_(params), camera_params_(camera_params) {}
    virtual ~MotionSolverBase() = default;

    virtual MotionResult solve(const GenericCorrespondences<RefType, CurrType>& correspondences) const = 0;

protected:
    const FrontendParams params_;
    const CameraParams camera_params_;

};


template<typename Result, typename RefType, typename CurrType>
class ObjectMotionSolverBase : public MotionSolverBase<Result, RefType, CurrType> {

public:
    using Base = MotionSolverBase<Result, RefType, CurrType>;
    using MotionResult = typename Base::MotionResult;

    ObjectMotionSolverBase(const FrontendParams& params, const CameraParams& camera_params,  const gtsam::Pose3& T_world_camera) : Base(params, camera_params), T_world_camera_(T_world_camera) {}

protected:
    const gtsam::Pose3 T_world_camera_;

};


class AbsoluteCameraMotionSolver : public MotionSolverBase<gtsam::Pose3, Landmark, Keypoint> {

public:
    using Base = MotionSolverBase<gtsam::Pose3, Landmark, Keypoint>;
    using MotionResult = Base::MotionResult;

    AbsoluteCameraMotionSolver(const FrontendParams& params, const CameraParams& camera_params) : Base(params, camera_params) {}

    MotionResult solve(const GenericCorrespondences<Landmark, Keypoint>& correspondences) const override;

};


template<typename RefType, typename CurrType>
class AbsoluteObjectMotionSolver : public ObjectMotionSolverBase<gtsam::Pose3, RefType, CurrType> {

public:
    using This = AbsoluteObjectMotionSolver<RefType, CurrType>;
    using Base = ObjectMotionSolverBase<gtsam::Pose3, RefType, CurrType>;
    using MotionResult = typename Base::MotionResult;

    AbsoluteObjectMotionSolver(const FrontendParams& params, const CameraParams& camera_params, const gtsam::Pose3& T_world_camera) : Base(params, camera_params, T_world_camera) {}

    MotionResult solve(const GenericCorrespondences<RefType, CurrType>& correspondences) const override;

};

class RelativeCameraMotionSolver : public MotionSolverBase<gtsam::Pose3, Keypoint, Keypoint> {

public:
    using Base = MotionSolverBase<gtsam::Pose3, Keypoint, Keypoint>;
    using MotionResult = Base::MotionResult;

    RelativeCameraMotionSolver(const FrontendParams& params, const CameraParams& camera_params) : Base(params, camera_params) {}

    MotionResult solve(const GenericCorrespondences<Keypoint, Keypoint>& correspondences) const override;

};

class EssentialDecompositionResult; //forward declare

class RelativeObjectMotionSolver : public MotionSolverBase<EssentialDecompositionResult, Keypoint, Keypoint> {

public:
    using Base = MotionSolverBase<EssentialDecompositionResult, Keypoint, Keypoint>;
    using MotionResult = Base::MotionResult;

    RelativeObjectMotionSolver(const FrontendParams& params, const CameraParams& camera_params) : Base(params, camera_params) {}

    MotionResult solve(const GenericCorrespondences<Keypoint, Keypoint>& correspondences) const override;

};






// class MotionSolver {

// public:
//     MotionSolver(const FrontendParams& params, const CameraParams& camera_params);

//     template<typename RefType, typename CurrType>
//     MotionResult solveCameraPose(const GenericCorrespondences<RefType, CurrType>& correspondences) const {
//         return motion_solver_tools::solveMotion<RefType, CurrType>(correspondences, params_, camera_params_);
//     }

//     template<typename RefType, typename CurrType>
//     MotionResult solveObjectMotion(const GenericCorrespondences<RefType, CurrType>& correspondences, const gtsam::Pose3& T_world_camera) const {
//         const MotionResult result = motion_solver_tools::solveMotion<RefType, CurrType>(correspondences, params_, camera_params_);
//         if(result.valid()) {
//             const gtsam::Pose3 G_w = result.get().inverse();
//             const gtsam::Pose3 H_w = T_world_camera * G_w;
//             return MotionResult(H_w, result.ids_used_, result.inliers_, result.outliers_);
//         }

//         //if not valid, return motion result as is
//         return result;
//     }




// protected:
//     const FrontendParams params_;
//     const CameraParams camera_params_;

// };

} //dyno
