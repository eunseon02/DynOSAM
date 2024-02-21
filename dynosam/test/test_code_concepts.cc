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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <exception>
#include <opencv4/opencv2/opencv.hpp>

#include <optional>
#include <atomic>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/utilities.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/dataset.h>

#include <string>
#include <fstream>
#include <iostream>

cv::Mat GetSquareImage( const cv::Mat& img, int target_width = 500 )
{
    int width = img.cols,
       height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}

TEST(CodeConcepts, drawInformationMatrix) {
    // std::string g2oFile = gtsam::findExampleDataFile("pose3example.txt");
    // gtsam::NonlinearFactorGraph::shared_ptr graph;
    // gtsam::Values::shared_ptr initial;
    // bool is3D = true;
    // std::tie(graph, initial) = gtsam::readG2o(g2oFile, is3D);

    // auto gfg = graph->linearize(*initial);

    // gtsam::Matrix jacobian = gfg->jacobian().first;

    gtsam::Values initial_estimate;
    gtsam::NonlinearFactorGraph graph;
    const auto model = gtsam::noiseModel::Isotropic::Sigma(3, 1);

    std::string calibration_loc = gtsam::findExampleDataFile("VO_calibration.txt");
    std::string pose_loc = gtsam::findExampleDataFile("VO_camera_poses_large.txt");
    std::string factor_loc = gtsam::findExampleDataFile("VO_stereo_factors_large.txt");

    // read camera calibration info from file
    // focal lengths fx, fy, skew s, principal point u0, v0, baseline b
    double fx, fy, s, u0, v0, b;
    std::ifstream calibration_file(calibration_loc.c_str());
    std::cout << "Reading calibration info" << std::endl;
    calibration_file >> fx >> fy >> s >> u0 >> v0 >> b;

    // create stereo camera calibration object
    const gtsam::Cal3_S2Stereo::shared_ptr K(new gtsam::Cal3_S2Stereo(fx, fy, s, u0, v0, b));

    std::ifstream pose_file(pose_loc.c_str());
    std::cout << "Reading camera poses" << std::endl;
    int pose_id;
    gtsam::MatrixRowMajor m(4, 4);
    // read camera pose parameters and use to make initial estimates of camera
    // poses
    while (pose_file >> pose_id) {
        for (int i = 0; i < 16; i++) {
            pose_file >> m.data()[i];
        }
        initial_estimate.insert(gtsam::Symbol('x', pose_id), gtsam::Pose3(m));
    }

    // camera and landmark keys
    size_t x, l;

    // pixel coordinates uL, uR, v (same for left/right images due to
    // rectification) landmark coordinates X, Y, Z in camera frame, resulting from
    // triangulation
    double uL, uR, v, X, Y, Z;
    std::ifstream factor_file(factor_loc.c_str());
    std::cout << "Reading stereo factors" << std::endl;
    // read stereo measurement details from file and use to create and add
    // GenericStereoFactor objects to the graph representation
    while (factor_file >> x >> l >> uL >> uR >> v >> X >> Y >> Z) {
        graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3> >(
            gtsam::StereoPoint2(uL, uR, v), model, gtsam::Symbol('x', x), gtsam::Symbol('l', l), K);
        // if the landmark variable included in this factor has not yet been added
        // to the initial variable value estimate, add it
        if (!initial_estimate.exists(gtsam::Symbol('l', l))) {
            gtsam::Pose3 camPose = initial_estimate.at<gtsam::Pose3>(gtsam::Symbol('x', x));
            // transformFrom() transforms the input Point3 from the camera pose space,
            // camPose, to the global space
            gtsam::Point3 worldPoint = camPose.transformFrom(gtsam::Point3(X, Y, Z));
            initial_estimate.insert(gtsam::Symbol('l', l), worldPoint);
        }
    }

    auto gfg = graph.linearize(initial_estimate);
    // gtsam::Matrix jacobian = gfg->jacobian().first;

    gtsam::Ordering natural_ordering = gtsam::Ordering::Natural(*gfg);
    gtsam::JacobianFactor jf(*gfg, natural_ordering);
    gtsam::Matrix J = jf.jacobian().first;

    cv::Mat J_img(cv::Size(J.rows(), J.cols()), CV_8UC3, cv::Scalar(255, 255, 255));
    for (int i = 0; i < J.rows(); ++i) {
        for (int j = 0; j < J.cols(); ++j) {
            if (std::fabs(J(i, j)) > 1e-15) {
                // make non zero elements blue
                J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
            }
        }
    }

    // gtsam::Matrix I = jf.information();
    // cv::Mat I_img(cv::Size(I.rows(), I.cols()), CV_8UC3, cv::Scalar(255, 255, 255));
    // for (int i = 0; i < I.rows(); ++i) {
    //     for (int j = 0; j < I.cols(); ++j) {
    //         if (std::fabs(I(i, j)) > 1e-15) {
    //             // make non zero elements blue
    //             I_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
    //         }
    //     }
    // }

    //resize for viz
    cv::resize(J_img, J_img, cv::Size(480, 480), 0, 0, cv::INTER_CUBIC);
    // cv::resize(I_img, I_img, cv::Size(480, 480), 0, 0, cv::INTER_CUBIC);

    cv::imshow("Jacobian", J_img);
    // cv::imshow("Information", I_img);
    cv::waitKey(0);

}


TEST(CodeConcepts, testModifyOptionalString) {

    auto modify = [](std::optional<std::reference_wrapper<std::string>> string) {
        if(string) {
           string->get() = "udpated";
        }
    };

    std::string input = "before";
    modify(input);

    EXPECT_EQ(input, "udpated");
}
