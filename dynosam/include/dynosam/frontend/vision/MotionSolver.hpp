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

using AbsolutePoseProblem = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
using AbsolutePoseSacProblem = opengv::sac::Ransac<AbsolutePoseProblem>;

using RelativePoseProblem = opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
using RelativePoseSacProblem = opengv::sac::Ransac<RelativePoseProblem>;

namespace dyno {

class MotionResult : public std::optional<gtsam::Pose3> {
public:
    enum Status { VALID, NOT_ENOUGH_CORRESPONDENCES, NOT_ENOUGH_INLIERS, UNSOLVABLE };
    Status status;

    TrackletIds ids_used_;
    TrackletIds inliers_;
    TrackletIds outliers_;

private:
    MotionResult(Status s) : status(s) {};

public:
    MotionResult() {}

    MotionResult(
        const gtsam::Pose3& pose,
        const TrackletIds& ids_used,
        const TrackletIds& inliers,
        const TrackletIds& outliers)
    : status(VALID),
      ids_used_(ids_used),
      inliers_(inliers),
      outliers_(outliers)
    {
        emplace(pose);

        {
            //debug check
            //inliers and outliers together should be the set of ids_used
            CHECK_EQ(ids_used_.size(), inliers_.size() + outliers_.size());
        }
    }

    operator const gtsam::Pose3&() const { return get(); }

    static MotionResult NotEnoughCorrespondences() { return MotionResult(NOT_ENOUGH_CORRESPONDENCES); }
    static MotionResult NotEnoughInliers() { return MotionResult(NOT_ENOUGH_INLIERS); }
    static MotionResult Unsolvable() { return MotionResult(UNSOLVABLE); }

    inline bool valid() const { return status == VALID; }
    inline bool notEnoughCorrespondences() const { return status == NOT_ENOUGH_CORRESPONDENCES; }
    inline bool notEnoughInliers() const { return status == NOT_ENOUGH_INLIERS; }
    inline bool unsolvable() const { return status == UNSOLVABLE; }

    const gtsam::Pose3& get() const {
        if (!has_value()) throw std::runtime_error("MotionResult has no value");
        return value();
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
template<typename RefType, typename CurrType>
MotionResult solveMotion(const GenericCorrespondences<RefType, CurrType>& correspondences, const FrontendParams& params, const CameraParams& camera_params);

} //motion_solver_tools









class MotionSolver {

public:
    MotionSolver(const FrontendParams& params, const CameraParams& camera_params);

    template<typename RefType, typename CurrType>
    MotionResult solveCameraPose(const GenericCorrespondences<RefType, CurrType>& correspondences) const {
        return motion_solver_tools::solveMotion<RefType, CurrType>(correspondences, params_, camera_params_);
    }

    template<typename RefType, typename CurrType>
    MotionResult solveObjectMotion(const GenericCorrespondences<RefType, CurrType>& correspondences, const gtsam::Pose3& T_world_camera) const {
        const MotionResult result = motion_solver_tools::solveMotion<RefType, CurrType>(correspondences, params_, camera_params_);
        if(result.valid()) {
            const gtsam::Pose3 G_w = result.get().inverse();
            const gtsam::Pose3 H_w = T_world_camera * G_w;
            return MotionResult(H_w, result.ids_used_, result.inliers_, result.outliers_);
        }

        //if not valid, return motion result as is
        return result;
    }




protected:
    const FrontendParams params_;
    const CameraParams camera_params_;

};

} //dyno
