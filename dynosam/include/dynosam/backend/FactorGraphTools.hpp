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
#include "dynosam/logger/Logger.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/linear/GaussianFactorGraph.h>

namespace dyno {

namespace factor_graph_tools {

SmartProjectionFactor::shared_ptr constructSmartProjectionFactor(
    gtsam::SharedNoiseModel smart_noise,
    boost::shared_ptr<CalibrationType> K,
    SmartProjectionFactorParams projection_params);


SmartProjectionFactor::shared_ptr constructSmartProjectionFactor(
    gtsam::SharedNoiseModel smart_noise,
    boost::shared_ptr<CalibrationType> K,
    SmartProjectionFactorParams projection_params,
    Keypoint measurement,
    FrameId frame_id);


void addBetweenFactor(
    FrameId from_frame, FrameId to_frame, const gtsam::Pose3 from_pose_to, gtsam::SharedNoiseModel noise_model, gtsam::NonlinearFactorGraph& graph);


void addSmartProjectionMeasurement(SmartProjectionFactor::shared_ptr smart_factor, Keypoint measurement, FrameId frame_id);

//TODO: should be templated on DERIVDFACTOR
//needs to be parse by ref so that we can return iterator (not const_iterator) which allows us to remove factors if necessary
//could do const and non const versions
//should not try and remove factors by iterator as graph.erase(itr) will reaarrange the graph but graph.remove(index) will not
size_t getAssociatedFactors(std::vector<gtsam::NonlinearFactorGraph::iterator>& associated_factors, gtsam::NonlinearFactorGraph& graph, gtsam::Key query_key);
size_t getAssociatedFactors(gtsam::FactorIndices& associated_factors, gtsam::NonlinearFactorGraph& graph, gtsam::Key query_key);
size_t getAssociatedFactors(std::vector<gtsam::NonlinearFactor::shared_ptr>& associated_factors, gtsam::NonlinearFactorGraph& graph, gtsam::Key query_key);


} //factor_graph_tools

struct SparsityStats {
    gtsam::Matrix hessian;
    size_t nr_zero_elements{0};
    size_t nr_elements{0};

    SparsityStats() {};
    SparsityStats(const gtsam::Matrix& H) : hessian(H), nr_zero_elements(0), nr_elements(0) {
        nr_elements = H.rows() * H.cols();
        for (int i = 0; i < H.rows(); ++i) {
            for (int j = 0; j < H.cols(); ++j) {
                if (std::fabs(H(i, j)) < 1e-15) {
                    nr_zero_elements += 1;
                }
            }
        }
    }

    void save(const std::string& file_name) {
        const std::string out_path = getOutputFilePath(file_name);
        VLOG(20) << "Writing hessian matrix to file " << out_path;
        gtsam::save(hessian, "hessian", out_path);
    }
};

template<class GRAPH>
class FactorGraphManager {
public:
    using FactorGraphType = GRAPH;
    using This = FactorGraphManager<FactorGraphType>;

    using FactorType = typename GRAPH::FactorType;
    using SharedFactor = std::shared_ptr<FactorType>;

    FactorGraphManager(const FactorGraphType graph) : graph_(graph) {}

    FactorGraphType getAssociatedFactors(gtsam::Key key) {
        std::vector<SharedFactor> factors;

        const size_t num_connected = factor_graph_tools::getAssociatedFactors(factors, graph_, key);

        if(num_connected < 1) {
            return FactorGraphType{};
        }

        FactorGraphType connected_factors;
        for(const auto& f : factors) {
            connected_factors += f;
        }
        return connected_factors;
    }


    SparsityStats computeSparsityStats(const gtsam::Values& linearization_point,
        bool save_hessian = false, const std::string& file = "hessian.txt") const;




private:
    FactorGraphType graph_;
};


//specalisation

template<>
inline SparsityStats FactorGraphManager<gtsam::NonlinearFactorGraph>::computeSparsityStats(
    const gtsam::Values& linearization_point, bool save_hessian, const std::string& file) const {
    auto gfg = graph_.linearize(linearization_point);

    SparsityStats stats(gfg->hessian().first);

    if(save_hessian) stats.save(file);

    return stats;
}



} //dyno
