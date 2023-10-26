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
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/types.hpp>

#include <glog/logging.h>

namespace dyno {

MotionSolver::MotionSolver(const FrontendParams& params) : params_(params) {}


gtsam::Pose3 MotionSolver::solveCameraPose(const AbsolutePoseCorrespondences& correspondences,  TrackletIds& inliers, TrackletIds& outliers) {
    const size_t& n_matches = correspondences.size();


    TrackletIds tracklets_;
    BearingVectors bearing_vectors;
    Landmarks points;
    for(size_t i = 0u; i < n_matches; i ++) {
        const AbsolutePoseCorrespondence& corres = correspondences.at(i);
        const Keypoint& kp = corres.cur_;
        //make Bearing vector
        gtsam::Vector3 versor(kp(0), kp(1), 1.0);
        versor = versor.normalized();
        bearing_vectors.push_back(versor);

        points.push_back(corres.ref_);
        tracklets_.push_back(corres.tracklet_id_);
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

    LOG(INFO) << "Solving ransac";
    // run ransac
    ransac.sac_model_ = absposeproblem_ptr;
    //https://github.com/laurentkneip/opengv/issues/121
    ransac.threshold_ = 2*(1.0 - cos(atan(sqrt(2.0)*0.5/800.0)));
    // ransac.threshold_ = threshold;
    // ransac.max_iterations_ = maxIterations;
    if(!ransac.computeModel(0)) {
        LOG(WARNING) << "Could not compute ransac mode";
        return gtsam::Pose3::Identity();
    }

    LOG(INFO) << "Solving ransac";
    // // get the result
    opengv::transformation_t best_transformation =
        ransac.model_coefficients_;


    gtsam::Matrix poseMat = gtsam::Matrix::Identity(4, 4);
    poseMat.block<3, 4>(0, 0) = best_transformation;

    gtsam::Pose3 opengv_transform(poseMat); //opengv has rotation as the inverse
    gtsam::Pose3 T_world(opengv_transform.rotation().inverse(), opengv_transform.translation());

    inliers.clear();
    outliers.clear();
    CHECK(ransac.inliers_.size() <= correspondences.size());
    for(int inlier_idx : ransac.inliers_) {
        const auto& corres = correspondences.at(inlier_idx);
        inliers.push_back(corres.tracklet_id_);
    }

    determineOutlierIds(inliers, tracklets_, outliers);
    CHECK_EQ((inliers.size() + outliers.size()), tracklets_.size());
    CHECK_EQ(inliers.size(), ransac.inliers_.size());

    return T_world;
    // return gtsam::Pose3::Identity();
}

} //dyno
