/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/factors/LandmarkMotionPoseFactor.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <exception>

using namespace dyno;


TEST(LandmarkMotionPoseFactor, visualiseJacobiansWithNonZeros) {

    gtsam::Pose3 L1(gtsam::Rot3::Rodrigues(-0.1, 0.2, 0.25),
                            gtsam::Point3(0.05, -0.10, 0.20));

    gtsam::Pose3 L2(gtsam::Rot3::Rodrigues(0.3, 0.2, -0.5),
                            gtsam::Point3(0.5, -0.15, 0.1));

    gtsam::Point3 p1(0.1, 2, 4);
    gtsam::Point3 p2(0.2, 3, 2);

    auto object_pose_k_1_key = ObjectPoseSymbol(0, 0);
    auto object_pose_k_key = ObjectPoseSymbol(0, 1);

    auto object_point_key_k_1 = DynamicLandmarkSymbol(0, 1);
    auto object_point_key_k = DynamicLandmarkSymbol(1, 1);

    LOG(INFO) << (std::string)object_point_key_k_1;

    gtsam::Values values;
    values.insert(object_pose_k_1_key, L1);
    values.insert(object_pose_k_key, L2);
    values.insert(object_point_key_k_1, p1);
    values.insert(object_point_key_k, p2);

    auto landmark_motion_noise = gtsam::noiseModel::Isotropic::Sigma(3u, 0.1);

    gtsam::NonlinearFactorGraph graph;
    graph.emplace_shared<LandmarkMotionPoseFactor>(
                        object_point_key_k_1,
                        object_point_key_k,
                        object_pose_k_1_key,
                        object_pose_k_key,
                        landmark_motion_noise
                    );

    NonlinearFactorGraphManager nlfgm(graph, values);

    cv::Mat block_jacobians = nlfgm.drawBlockJacobian(
        gtsam::Ordering::OrderingType::COLAMD,
        factor_graph_tools::DrawBlockJacobiansOptions::makeDynoSamOptions());

    cv::imshow("LandmarkMotionPoseFactor block jacobians", block_jacobians);
    cv::waitKey(0);

}
