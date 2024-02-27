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

#include "internal/simulator.hpp"
#include "internal/helpers.hpp"
#include "dynosam/backend/RGBDBackendModule.hpp"


#include <glog/logging.h>
#include <gtest/gtest.h>
#include <exception>
#include <vector>
#include <iterator>


TEST(RGBDBackendModule, constructSimpleGraph) {

    //make camera with a constant motion model starting at zero
    dyno_testing::ScenarioBody::Ptr camera = std::make_shared<dyno_testing::ScenarioBody>(
        std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
            gtsam::Pose3::Identity(),
            //motion only in x
            gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.1, 0, 0))
        )
    );
    dyno_testing::RGBDScenario scenario(camera);

    //add one obect
    const size_t num_points = 5;
    dyno_testing::ObjectBody::Ptr object1 = std::make_shared<dyno_testing::ObjectBody>(
        std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
            gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(10, 0, 0)),
            //motion only in x
            gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.2, 0, 0))
        ),
        std::make_unique<dyno_testing::ConstantPointsVisitor>(num_points)
    );

    scenario.addObjectBody(1, object1);

    dyno::Map3d::Ptr map = dyno::Map3d::create();
    dyno::RGBDBackendModule backend(
        dyno::BackendParams{},
        dyno_testing::makeDefaultCameraPtr(),
        map);

    for(size_t i = 0; i < 6; i++) {
        auto output = scenario.getOutput(i);

        std::stringstream ss;
        ss << output->T_world_camera_ << "\n";
        ss << dyno::container_to_string(output->dynamic_landmarks_);

        LOG(INFO) << ss.str();
        backend.spinOnce(output);

        backend.saveGraph("rgbd_graph_" + std::to_string(i) + ".dot");
        backend.saveTree("rgbd_bayes_tree_" + std::to_string(i) + ".dot");
    }


}
