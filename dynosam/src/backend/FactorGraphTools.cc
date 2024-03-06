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

#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"

#include "gtsam/linear/GaussianConditional.h"

#include <opencv4/opencv2/viz/types.hpp>
#include <gtsam/slam/BetweenFactor.h>


namespace dyno {

namespace factor_graph_tools {


SmartProjectionFactor::shared_ptr constructSmartProjectionFactor(
    gtsam::SharedNoiseModel smart_noise,
    boost::shared_ptr<CalibrationType> K,
    SmartProjectionFactorParams projection_params)
{
    CHECK(smart_noise);
    CHECK(K);

    return boost::make_shared<SmartProjectionFactor>(
                smart_noise,
                K,
                projection_params);
}

SmartProjectionFactor::shared_ptr constructSmartProjectionFactor(
    gtsam::SharedNoiseModel smart_noise,
    boost::shared_ptr<CalibrationType> K,
    SmartProjectionFactorParams projection_params,
    Keypoint measurement,
    FrameId frame_id)
{
    SmartProjectionFactor::shared_ptr smart_factor = constructSmartProjectionFactor(
        smart_noise,
        K,
        projection_params);
    CHECK(smart_factor);

    addSmartProjectionMeasurement(smart_factor, measurement, frame_id);
    return smart_factor;

}

void addBetweenFactor(FrameId from_frame, FrameId to_frame, const gtsam::Pose3 from_pose_to, gtsam::SharedNoiseModel noise_model, gtsam::NonlinearFactorGraph& graph) {
    CHECK(noise_model);
    CHECK_EQ(noise_model->dim(), 6u);

    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        CameraPoseSymbol(from_frame),
        CameraPoseSymbol(to_frame),
        from_pose_to,
        noise_model
    );
}


void addSmartProjectionMeasurement(SmartProjectionFactor::shared_ptr smart_factor, Keypoint measurement, FrameId frame_id) {
    smart_factor->add(measurement, CameraPoseSymbol(frame_id));
}

SparsityStats computeHessianSparsityStats(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg, const std::optional<gtsam::Ordering>& ordering) {
    if(ordering) {
        return SparsityStats(gaussian_fg->hessian(*ordering).first);
    }

    return SparsityStats(gaussian_fg->hessian().first);
}

SparsityStats computeJacobianSparsityStats(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg, const std::optional<gtsam::Ordering>& ordering) {
    //NOTE: GaussianFactorGraph::jacobian(Ordering) SHOULD be the same as calling gtsam::JacobianFactor(GaussianFactorGraph, ordering)
    //and it is in the code when this was tested but I guess this could change, leading to consistencies with how the Jacobain factor is constructed
    //in differnet fg_tool functions
    if(ordering) {
        return SparsityStats(gaussian_fg->jacobian(*ordering).first);
    }

    return SparsityStats(gaussian_fg->jacobian().first);
}

std::pair<SparsityStats, cv::Mat> computeRFactor(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg, const gtsam::Ordering ordering) {
    //TODO: why do we need to specify the ordering twice?
    //I think we should probaly construct the graph with native ordeing but then elimate with provided ordering?
    gtsam::JacobianFactor jacobian_factor(*gaussian_fg);

    gtsam::GaussianConditional::shared_ptr conditional;
    gtsam::JacobianFactor::shared_ptr joint_factor;

    /// Not entirely sure the difference between the joint factor we get out and the inital jacobian factor we get in
    /// I believe the difference is in the ordering which we do NOT apply when constructing the jacobian_factor. I'm not sure this is correct.
    std::tie(conditional, joint_factor) = jacobian_factor.eliminate(ordering);
    CHECK(conditional);

    const gtsam::VerticalBlockMatrix::constBlock R = conditional->R();

    //drawn image
    cv::Mat R_img(cv::Size(R.cols(), R.rows()), CV_8UC3, cv::viz::Color::white());
    for (int i = 0; i < R.rows(); ++i) {
        for (int j = 0; j < R.cols(); ++j) {
            //only draw if non zero
            if (std::fabs(R(i, j)) > 1e-15) {
                R_img.at<cv::Vec3b>(i, j) =  (cv::Vec3b)cv::viz::Color::black();
            }
        }
    }

    cv::imshow("R", R_img);
    cv::waitKey(0);


}

cv::Mat drawBlockJacobians(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg, const gtsam::Ordering& ordering, const DrawBlockJacobiansOptions& options) {
    const gtsam::JacobianFactor jacobian_factor(*gaussian_fg, ordering);
    const gtsam::Matrix J = jacobian_factor.jacobian().first;

    //each cv::mat will be the drawing of a column block (associated with one variable) in the jacobian
    //Each block will have dimensions determiend by dim of the variable (determiend by the key)
    //and number of rows of the full jacobian matrix, J
    std::vector<std::pair<gtsam::Key, cv::Mat>> column_blocks;


    for(gtsam::Key key : ordering) {
        //this will have rows = J.rows(), cols = dimension of the variable
        const gtsam::VerticalBlockMatrix::constBlock Ablock = jacobian_factor.getA(jacobian_factor.find(key));
        const int var_dimensions = Ablock.cols();
        CHECK_EQ(Ablock.rows(), J.rows());

        //drawn image of each vertical block
        cv::Mat b_img(cv::Size(var_dimensions, Ablock.rows()), CV_8UC3, cv::viz::Color::white());
        for (int i = 0; i < Ablock.rows(); ++i) {
            for (int j = 0; j < var_dimensions; ++j) {
                //only draw if non zero
                if (std::fabs(Ablock(i, j)) > 1e-15) {
                    const auto colour = options.colour_selector(key);
                    b_img.at<cv::Vec3b>(i, j) =  cv::Vec3b(colour[0], colour[1], colour[2]);
                }
            }
        }

        column_blocks.push_back(std::make_pair(key, b_img));
    }

    CHECK(!options.desired_size.empty()); //cannot be empty

    const cv::Size& desired_size = options.desired_size;

    const int current_cols = J.cols();
    const int desired_cols = desired_size.width;
    const int desired_rows = desired_size.height;

    //define a ratio which which to scale each block (viz) individually
    //since we're going to add things like spacing we want to scale
    //up each block so its visible but dont want to scale up the line spacing etc.
    //the ratio ensures we get close to the final desired size of the TOTAL image
    double ratio;
    //want positive ratio
    if(current_cols < desired_cols) {
        ratio = (double)desired_cols/(double)current_cols;
    }
    else {
        ratio = (double)current_cols/(double)desired_cols;
    }

    auto scale_and_draw_label = [&options, &ratio, &desired_rows](cv::Mat& current_block, gtsam::Key key) {
        //we all each jacobiab horizintally so only need to scale the columns
        const int scaled_cols = ratio * (int)current_block.cols;

        //scale current block to the desired size with INTER_NEAREST - this keeps the pixels looking clear and visible
        cv::resize(current_block, current_block, cv::Size(scaled_cols, desired_rows), 0, 0, cv::INTER_NEAREST);

        if(options.draw_label) {
            //draw text info
            cv::Mat text_box(cv::Size(current_block.cols, options.text_box_height), CV_8UC3, cv::viz::Color::white());

            constexpr static double kFontScale = 0.5;
            constexpr static int kFontFace = cv::FONT_HERSHEY_DUPLEX;
            constexpr static int kThickness = 1;
            //draw text mid way in box
            cv::putText(text_box, options.label_formatter(key), cv::Point(text_box.cols/2, text_box.rows/2), kFontFace, kFontScale, cv::viz::Color::black(), kThickness);
            //add text box ontop of jacobian block
            cv::vconcat(text_box, current_block, current_block);
        }
    }; //end scale_and_draw_label


    if(column_blocks.size() == 1) {
        gtsam::Key key = column_blocks.at(0).first;
        cv::Mat current_block = column_blocks.at(0).second;
        scale_and_draw_label(current_block, key);
        return current_block;
    }

    //the final drawn image
    cv::Mat concat_column_blocks;

    for(size_t i = 1; i < column_blocks.size(); i++) {
        gtsam::Key key = column_blocks.at(i).first;
        cv::Mat current_block = column_blocks.at(i).second;
        scale_and_draw_label(current_block, key);


        if(i == 1) {
            cv::Mat first_block = column_blocks.at(0).second;
            gtsam::Key key_0 = column_blocks.at(0).first;
            scale_and_draw_label(first_block, key_0);
            //concat current jacobian block with other blocks
            cv::hconcat(first_block, current_block, concat_column_blocks);

        }
        else {
            //concat current jacobian block with other blocks
            cv::hconcat(concat_column_blocks, current_block, concat_column_blocks);
        }

        //if not last, add vertical line
        if(i < column_blocks.size() - 1) {
            cv::Mat vert_line(cv::Size(2, concat_column_blocks.rows), CV_8UC3, cv::viz::Color::black());
            //concat blocks with vertial line
            cv::hconcat(concat_column_blocks, vert_line, concat_column_blocks);
        }
    }
    return concat_column_blocks;
}

} //factor_graph_tools
} //dyno
