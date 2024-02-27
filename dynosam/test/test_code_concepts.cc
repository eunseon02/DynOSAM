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

#include "dynosam/visualizer/ColourMap.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"

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

#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <gtsam/base/Matrix.h>

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
    gtsam::Values initial_estimate;
    gtsam::NonlinearFactorGraph graph;
    const auto model = gtsam::noiseModel::Isotropic::Sigma(3, 1);

    using namespace gtsam::symbol_shorthand;


    // //smaller example
    // gtsam::Pose3 first_pose;
    // graph.emplace_shared<gtsam::NonlinearEquality<gtsam::Pose3> >(X(1), gtsam::Pose3());

    // // create factor noise model with 3 sigmas of value 1
    // // const auto model = gtsam::noiseModel::Isotropic::Sigma(3, 1);
    // // create stereo camera calibration object with .2m between cameras
    // const gtsam::Cal3_S2Stereo::shared_ptr K(
    //     new gtsam::Cal3_S2Stereo(1000, 1000, 0, 320, 240, 0.2));

    // //create and add stereo factors between first pose (key value 1) and the three landmarks
    // // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3> >(gtsam::StereoPoint2(520, 480, 440), model, 1, 3, K);
    // // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3> >(gtsam::StereoPoint2(120, 80, 440), model, 1, 4, K);
    // // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3> >(gtsam::StereoPoint2(320, 280, 140), model, 1, 5, K);

    // // //create and add stereo factors between second pose and the three landmarks
    // // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3> >(gtsam::StereoPoint2(570, 520, 490), model, 2, 3, K);
    // // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3> >(gtsam::StereoPoint2(70, 20, 490), model, 2, 4, K);
    //  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3> >(X(1), L(1), gtsam::Point3(1, 1, 5), model);
    // graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3> >(X(1), L(2),  gtsam::Point3(1, 1, 5), model);
    // graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3> >(X(1), L(3), gtsam::Point3::Identity(), model);

    // //create and add stereo factors between second pose and the three landmarks
    // graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3> >(X(2),L(1), gtsam::Point3::Identity(), model);
    // graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3,gtsam::Point3> >(X(2), L(2), gtsam::Point3::Identity(), model);
    // // graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3> >(gtsam::StereoPoint2(320, 270, 115), model, 2, 5, K);

    // // create Values object to contain initial estimates of camera poses and
    // // landmark locations

    // // create and add iniital estimates
    // initial_estimate.insert(X(1), first_pose);
    // initial_estimate.insert(X(2), gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0.1, -0.1, 1.1)));
    // initial_estimate.insert(L(1), gtsam::Point3(1, 1, 5));
    // initial_estimate.insert(L(2), gtsam::Point3(-1, 1, 5));
    // initial_estimate.insert(L(3), gtsam::Point3(1, -0.5, 5));

    // std::string calibration_loc = gtsam::findExampleDataFile("VO_calibration.txt");
    // std::string pose_loc = gtsam::findExampleDataFile("VO_camera_poses_large.txt");
    // std::string factor_loc = gtsam::findExampleDataFile("VO_stereo_factors_large.txt");

    // // read camera calibration info from file
    // // focal lengths fx, fy, skew s, principal point u0, v0, baseline b
    // double fx, fy, s, u0, v0, b;
    // std::ifstream calibration_file(calibration_loc.c_str());
    // std::cout << "Reading calibration info" << std::endl;
    // calibration_file >> fx >> fy >> s >> u0 >> v0 >> b;

    // // create stereo camera calibration object
    // const gtsam::Cal3_S2Stereo::shared_ptr K(new gtsam::Cal3_S2Stereo(fx, fy, s, u0, v0, b));

    // std::ifstream pose_file(pose_loc.c_str());
    // std::cout << "Reading camera poses" << std::endl;
    // int pose_id;
    // gtsam::MatrixRowMajor m(4, 4);
    // // read camera pose parameters and use to make initial estimates of camera
    // // poses
    // while (pose_file >> pose_id) {
    //     for (int i = 0; i < 16; i++) {
    //         pose_file >> m.data()[i];
    //     }
    //     initial_estimate.insert(gtsam::Symbol('x', pose_id), gtsam::Pose3(m));
    // }

    // // camera and landmark keys
    // size_t x, l;

    // // pixel coordinates uL, uR, v (same for left/right images due to
    // // rectification) landmark coordinates X, Y, Z in camera frame, resulting from
    // // triangulation
    // double uL, uR, v, X, Y, Z;
    // std::ifstream factor_file(factor_loc.c_str());
    // std::cout << "Reading stereo factors" << std::endl;
    // // read stereo measurement details from file and use to create and add
    // // GenericStereoFactor objects to the graph representation
    // while (factor_file >> x >> l >> uL >> uR >> v >> X >> Y >> Z) {
    //     graph.emplace_shared<gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3> >(
    //         gtsam::StereoPoint2(uL, uR, v), model, gtsam::Symbol('x', x), gtsam::Symbol('l', l), K);
    //     // if the landmark variable included in this factor has not yet been added
    //     // to the initial variable value estimate, add it
    //     if (!initial_estimate.exists(gtsam::Symbol('l', l))) {
    //         gtsam::Pose3 camPose = initial_estimate.at<gtsam::Pose3>(gtsam::Symbol('x', x));
    //         // transformFrom() transforms the input Point3 from the camera pose space,
    //         // camPose, to the global space
    //         gtsam::Point3 worldPoint = camPose.transformFrom(gtsam::Point3(X, Y, Z));
    //         initial_estimate.insert(gtsam::Symbol('l', l), worldPoint);
    //     }
    // }

    // auto gfg = graph.linearize(initial_estimate);
    // // gtsam::Matrix jacobian = gfg->jacobian().first;

    // gtsam::Ordering natural_ordering = gtsam::Ordering::Natural(*gfg);
    // gtsam::JacobianFactor jf(*gfg, natural_ordering);
    // gtsam::Matrix J = jf.jacobian().first;

    // {
    //     gtsam::Values pose_only_values;
    //     pose_only_values.insert(0, gtsam::Pose3::Identity());
    //     pose_only_values.insert(1, gtsam::Pose3::Identity());
    //     gtsam::NonlinearFactorGraph pose_only_graph;
    //     pose_only_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Sigma(6, 1));
    //     pose_only_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(1, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Sigma(6, 1));


    //     gtsam::Ordering natural_ordering_pose = gtsam::Ordering::Natural(pose_only_graph);
    //     auto gfg_pose = pose_only_graph.linearize(pose_only_values);
    //     gtsam::JacobianFactor jf_pose(*gfg_pose, natural_ordering_pose);

    //         // auto size = natural_ordering.size();
    //         // auto size1 = jf_pose.cols() * 6;
    //     //cols should be num variables * 6 as each pose has dimenstions variables
    //     EXPECT_EQ(natural_ordering_pose.size() * 6,(jf_pose.getA().cols()));

    //     //this just a view HORIZONTALLY, vertically it is the entire matrix
    //     const gtsam::VerticalBlockMatrix::constBlock Ablock = jf_pose.getA(jf_pose.find(0));
    //     //in this example there should be 2? blocks
    //     // LOG(INFO) << "Num blocks " << Ablock.nBlocks();
    //     LOG(INFO) << Ablock;

    //     for(gtsam::Key key : natural_ordering_pose) {
    //     //this will have rows = J.rows(), cols = dimension of the variable
    //     const gtsam::VerticalBlockMatrix::constBlock Ablock = jf_pose.getA(jf_pose.find(key));
    //     const size_t var_dimensions = Ablock.cols();cv::Vec3b(255, 0, 0);
    //     EXPECT_EQ(Ablock.rows(), jf_pose.rows());

    //     LOG(INFO) << "Start Col " << Ablock.startCol() << " Start row " << Ablock.startRow() << " dims " << var_dimensions;
    //     //eachs start col shoudl be dim(key) apart
    //     // for (int i = 0; i < J.rows(); ++i) {
    //     //     for (int j = 0; j < J.cols(); ++j) {
    //     //         if (std::fabs(J(i, j)) > 1e-15) {
    //     //             // make non zero elements blue
    //     //             J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
    //     //         }
    //     //     }
    //     // }

    //     }

    // }

    // graph.saveGraph(dyno::getOutputFilePath("test_graph.dot"));

    // LOG(INFO) << natural_ordering.size();
    // LOG(INFO) << J.cols();

    // cv::Mat J_img(cv::Size(J.cols(), J.rows()), CV_8UC3, cv::Scalar(255, 255, 255));

    // std::vector<std::pair<gtsam::Key, cv::Mat>> column_blocks;
    // LOG(INFO) << J_img.cols;
    // for(gtsam::Key key : natural_ordering) {
    //     //this will have rows = J.rows(), cols = dimension of the variable
    //     // LOG(INFO) << key;
    //     const gtsam::VerticalBlockMatrix::constBlock Ablock = jf.getA(jf.find(key));
    //     const size_t var_dimensions = Ablock.cols();
    //     EXPECT_EQ(Ablock.rows(), J.rows());

    //     // LOG(INFO) << "Start Col " << Ablock.startCol() << " Start row " << Ablock.startRow() << " dims " << var_dimensions;



    //     for (int i = 0; i < J.rows(); ++i) {
    //         for (int j = Ablock.startCol(); j < (Ablock.startCol() + var_dimensions); ++j) {
    //             ASSERT_LT(j, J_img.cols);
    //             ASSERT_LT(i, J_img.rows);
    //             if (std::fabs(J(i, j)) > 1e-15) {
    //                 // make non zero elements blue
    //                 const auto colour = dyno::ColourMap::getObjectColour((int)var_dimensions);
    //                 J_img.at<cv::Vec3b>(i, j) =  cv::Vec3b(colour[0], colour[1], colour[2]);
    //                 // J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
    //                 // J_img.at<cv::Vec3b>(i, j)[2] = colour[2];
    //             }
    //         }
    //     }

    //     cv::Mat b_img(cv::Size(var_dimensions, Ablock.rows()), CV_8UC3, cv::Scalar(255, 255, 255));
    //     for (int i = 0; i < Ablock.rows(); ++i) {
    //         for (int j = 0; j < var_dimensions; ++j) {
    //             // ASSERT_LT(j, J_img.cols);
    //             // ASSERT_LT(i, J_img.rows);
    //             if (std::fabs(Ablock(i, j)) > 1e-15) {
    //                 // make non zero elements blue
    //                 const auto colour = dyno::ColourMap::getObjectColour((int)var_dimensions);
    //                 b_img.at<cv::Vec3b>(i, j) =  cv::Vec3b(colour[0], colour[1], colour[2]);
    //                 // J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
    //                 // J_img.at<cv::Vec3b>(i, j)[2] = colour[2];
    //             }
    //         }
    //     }

    //     column_blocks.push_back(std::make_pair(key, b_img));

    // }

    // LOG(INFO) << column_blocks.size();

    // cv::Size desired_size(480, 480);

    // const int current_cols = J.cols();
    // const int desired_cols = desired_size.width;
    // const int desired_rows = desired_size.height;

    // double ratio = (double)desired_cols/(double)current_cols;
    // LOG(INFO) << ratio;
    // cv::Mat concat_column_blocks;
    // // cv::Mat concat_column_blocks = column_blocks.at(0);
    // cv::Mat labels;

    // for(size_t i = 1; i < column_blocks.size(); i++) {
    //     gtsam::Key key = column_blocks.at(i).first;
    //     cv::Mat current_block = column_blocks.at(i).second;
    //     int scaled_cols = ratio * (int)current_block.cols;
    //     cv::resize(current_block, current_block, cv::Size(scaled_cols, desired_rows), 0, 0, cv::INTER_NEAREST);

    //     //draw text info
    //     int baseline=0;
    //     // cv::Size textSize = cv::getTextSize(gtsam::DefaultKeyFormatter(key),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    //     const int text_box_height = 30;
    //     cv::Mat text_box(cv::Size(current_block.cols, text_box_height), CV_8UC3, cv::Scalar(255, 255, 255));

    //     constexpr static double kFontScale = 0.5;
    //     constexpr static int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
    //     constexpr static int kThickness = 2;
    //     //draw text mid way in box
    //     cv::putText(text_box, gtsam::DefaultKeyFormatter(key), cv::Point(text_box.cols/2, text_box.rows/2), kFontFace, kFontScale, cv::Scalar(0, 0, 0), kThickness);

    //     // cv::imshow("text", text_box);
    //     // cv::waitKey(0);

    //     cv::vconcat(text_box, current_block, current_block);

    //     if(i == 1) {
    //         cv::Mat first_block = column_blocks.at(0).second;
    //         gtsam::Key key_0 = column_blocks.at(0).first;

    //         int first_scaled_cols = ratio * first_block.cols;
    //         cv::resize(first_block, first_block, cv::Size(first_scaled_cols, desired_rows), 0, 0, cv::INTER_NEAREST);

    //         //add text
    //         cv::Mat text_box_first(cv::Size(first_block.cols, text_box_height), CV_8UC3, cv::Scalar(255, 255, 255));
    //         cv::putText(text_box, gtsam::DefaultKeyFormatter(key_0), cv::Point(text_box_first.cols/2, text_box_first.rows/2), kFontFace, kFontScale, cv::Scalar(0, 0, 0), kThickness);

    //         //concat text box on top of jacobian block
    //         cv::vconcat(text_box, first_block, first_block);
    //         //now concat with next jacobian block
    //         cv::hconcat(first_block, current_block, concat_column_blocks);

    //     }
    //     else {
    //         //concat current jacobian block with other blocks
    //         cv::hconcat(concat_column_blocks, current_block, concat_column_blocks);
    //     }

    //     //if not last add vertical line
    //     if(i < column_blocks.size() - 1) {
    //         cv::Mat vert_line(cv::Size(2, concat_column_blocks.rows), CV_8UC3, cv::Scalar(0, 0, 0));
    //         //concat blocks with vertial line
    //         cv::hconcat(concat_column_blocks, vert_line, concat_column_blocks);
    //         // concat_column_blocks = dyno::utils::concatenateImagesHorizontally(concat_column_blocks, vert_line);
    //     }


    //     // //all blocks should have the same number of rows
    //     // double scaled_cols =

    //     // //add vertical line between each variable block
    //     // cv::Mat vert_line(cv::Size(2, concat_column_blocks.rows), CV_8UC3, cv::Scalar(0, 0, 0));
    //     // concat_column_blocks = dyno::utils::concatenateImagesHorizontally(concat_column_blocks, vert_line);
    // }

    // dyno::NonlinearFactorGraphManager nlfg(graph, initial_estimate);

    // cv::Mat J_1 = nlfg.drawBlockJacobian(gtsam::Ordering::NATURAL, dyno::factor_graph_tools::DrawBlockJacobiansOptions::makeDynoSamOptions());

    // for (int i = 0; i < J.rows(); ++i) {
    //     for (int j = 0; j < J.cols(); ++j) {
    //         if (std::fabs(J(i, j)) > 1e-15) {
    //             // make non zero elements blue
    //             J_img.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
    //         }
    //     }
    // }

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
    // LOG(INFO) << dyno::to_string(concat_column_blocks.size());
    // //resize for viz
    // cv::resize(concat_column_blocks, concat_column_blocks, cv::Size(480, 480), 0, 0, cv::INTER_NEAREST);
    // cv::resize(J_img, J_img, cv::Size(480, 480), 0, 0, cv::INTER_NEAREST);
    // // // cv::resize(I_img, I_img, cv::Size(480, 480), 0, 0, cv::INTER_CUBIC);

    // cv::imshow("Jacobian", concat_column_blocks);
    // cv::imshow("J orig", J_img);
    // cv::imshow("J_1", J_1);
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
