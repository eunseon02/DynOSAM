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

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>

#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/types.hpp>

#include <glog/logging.h>

namespace dyno {

MotionSolver::MotionSolver(const FrontendParams& params, const CameraParams& camera_params) : params_(params), camera_params_(camera_params) {}


gtsam::Pose3 MotionSolver::solveCameraPose(const AbsolutePoseCorrespondences& correspondences,  TrackletIds& inliers, TrackletIds& outliers) {
    const size_t& n_matches = correspondences.size();

    gtsam::Matrix K = gtsam::Matrix::Identity(3, 3);
    cv::cv2eigen(camera_params_.getCameraMatrix(), K);

    K = K.inverse();

    TrackletIds tracklets_;
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
    ransac.threshold_ = 1.0 - cos(atan(sqrt(2.0)*0.5/800.0));
    ransac.max_iterations_ = 100;
    if(!ransac.computeModel(0)) {
        LOG(WARNING) << "Could not compute ransac mode";
        return gtsam::Pose3::Identity();
    }

    // // get the result
    opengv::transformation_t best_transformation =
        ransac.model_coefficients_;


    gtsam::Matrix poseMat = gtsam::Matrix::Identity(4, 4);
    poseMat.block<3, 4>(0, 0) = best_transformation;

    gtsam::Pose3 opengv_transform(poseMat); //opengv has rotation as the inverse
    gtsam::Pose3 T_world(opengv_transform.rotation(), opengv_transform.translation());

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

    if(VLOG_IS_ON(10)) {
        LOG(INFO) << "PnP RANSAC success with\n"
            << " - inliers/outliers: " << inliers.size() << "/" << outliers.size();
    }

    return T_world;
}

} //dyno
