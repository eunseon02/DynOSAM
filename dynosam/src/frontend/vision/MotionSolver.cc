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

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/types.hpp>

#include <glog/logging.h>

namespace dyno {

MotionSolver::MotionSolver(const FrontendParams& params) : params_(params) {}


gtsam::Pose3 MotionSolver::solveCameraPose(const Keypoints& current_keypoints, const Landmarks& previous_points) {
    CHECK_EQ(current_keypoints.size(), previous_points.size());
    const size_t& n_matches = previous_points.size();

    BearingVectors bearing_vectors;
    for(size_t i = 0u; i < n_matches; i ++) {
        //make Bearing vector
        gtsam::Vector3 versor(current_keypoints.at(i)(0), current_keypoints.at(i)(1), 1.0);
        versor = versor.normalized();
        bearing_vectors.push_back(versor);
    }


    opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors, previous_points );

    // create a Ransac object
    AbsolutePoseSacProblem ransac;
    // create an AbsolutePoseSacProblem
    // (algorithm is selectable: KNEIP, GAO, or EPNP)
    std::shared_ptr<AbsolutePoseProblem>
        absposeproblem_ptr(
        new AbsolutePoseProblem(
        adapter, AbsolutePoseProblem::KNEIP ) );

    // run ransac
    ransac.sac_model_ = absposeproblem_ptr;
    // ransac.threshold_ = threshold;
    // ransac.max_iterations_ = maxIterations;
    ransac.computeModel();
    // get the result
    opengv::transformation_t best_transformation =
        ransac.model_coefficients_;


}

} //dyno
