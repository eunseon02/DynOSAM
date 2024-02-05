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


size_t getAssociatedFactors(std::vector<gtsam::NonlinearFactorGraph::iterator>& associated_factors, gtsam::NonlinearFactorGraph& graph, gtsam::Key query_key) {
    associated_factors.clear();

    for(auto it = graph.begin(); it != graph.end(); it++) {
        auto factor = *it;
        if(factor->find(query_key) != factor->end()) {
            associated_factors.push_back(it);
        }
    }

    return associated_factors.size();
}

size_t getAssociatedFactors(gtsam::FactorIndices& associated_factors, gtsam::NonlinearFactorGraph& graph, gtsam::Key query_key) {
    associated_factors.clear();
    std::vector<gtsam::NonlinearFactorGraph::iterator> associated_factor_iters;
    size_t result = getAssociatedFactors(associated_factor_iters, graph, query_key);

    for(const gtsam::NonlinearFactorGraph::iterator& iter : associated_factor_iters) {
        //index in graph
        associated_factors.push_back(std::distance(graph.begin(), iter));
    }
    return result;
}

size_t getAssociatedFactors(std::vector<gtsam::NonlinearFactor::shared_ptr>& associated_factors, gtsam::NonlinearFactorGraph& graph, gtsam::Key query_key) {
    associated_factors.clear();
    std::vector<gtsam::NonlinearFactorGraph::iterator> associated_factor_iters;
    size_t result = getAssociatedFactors(associated_factor_iters, graph, query_key);

    for(const gtsam::NonlinearFactorGraph::iterator& iter : associated_factor_iters) {
        associated_factors.push_back(*iter);
    }
    return result;
}



} //factor_graph_tools
} //dyno
