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
#include "dynosam/visualizer/ColourMap.hpp"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/linear/GaussianFactorGraph.h>

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

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

template<typename GRAPH>
size_t getAssociatedFactors(std::vector<typename GRAPH::const_iterator>& associated_factors, const GRAPH& graph, gtsam::Key query_key) {
    associated_factors.clear();

    for(auto it = graph.begin(); it != graph.end(); it++) {
        auto factor = *it;
        if(factor->find(query_key) != factor->end()) {
            associated_factors.push_back(it);
        }
    }
    return associated_factors.size();
}

template<typename GRAPH>
size_t getAssociatedFactors(gtsam::FactorIndices& associated_factors, const GRAPH& graph, gtsam::Key query_key) {
    associated_factors.clear();
    std::vector<typename GRAPH::const_iterator> associated_factor_iters;
    size_t result = getAssociatedFactors(associated_factor_iters, graph, query_key);

    for(const typename GRAPH::const_iterator& iter : associated_factor_iters) {
        //index in graph
        associated_factors.push_back(std::distance(graph.begin(), iter));
    }
    return result;
}

template<typename GRAPH>
size_t getAssociatedFactors(std::vector<typename GRAPH::sharedFactor>& associated_factors,const GRAPH& graph, gtsam::Key query_key) {
    associated_factors.clear();
    std::vector<typename GRAPH::const_iterator> associated_factor_iters;
    size_t result = getAssociatedFactors(associated_factor_iters, graph, query_key);

    for(const typename GRAPH::const_iterator& iter : associated_factor_iters) {
        associated_factors.push_back(*iter);
    }
    return result;
}




struct SparsityStats {
    gtsam::Matrix matrix;
    size_t nr_zero_elements{0};
    size_t nr_elements{0};

    SparsityStats() {};
    SparsityStats(const gtsam::Matrix& M) : matrix(M), nr_zero_elements(0), nr_elements(0) {
        nr_elements = M.rows() * M.cols();
        for (int i = 0; i < M.rows(); ++i) {
            for (int j = 0; j < M.cols(); ++j) {
                if (std::fabs(M(i, j)) < 1e-15) {
                    nr_zero_elements += 1;
                }
            }
        }
    }

    void saveOnOutputPath(const std::string& file_name) {
        save(getOutputFilePath(file_name));
    }

    void save(const std::string& file_path) {
        VLOG(20) << "Writing matrix matrix to file_path " << file_path;
        gtsam::save(matrix, "sparse_matrix", file_path);
    }
};

SparsityStats computeHessianSparsityStats(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg, const std::optional<gtsam::Ordering>& ordering = {});
SparsityStats computeJacobianSparsityStats(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg, const std::optional<gtsam::Ordering>& ordering = {});

struct DrawBlockJacobiansOptions {
    cv::Size desired_size = cv::Size(480, 480);

    bool draw_label = true; //if true, the labels for each vertical block (ie, the variable names) will be draw atop each block
    int text_box_height = 30; //text box for label
    gtsam::KeyFormatter label_formatter = gtsam::DefaultKeyFormatter; //function used to generate the label, if draw_label== true

    //! Alias to a function that takes a key and returns a colour. Used to colour the non-zero elements of the Jacobian
    //! depending on the variable (block)
    using ColourSelector = std::function<cv::Scalar(gtsam::Key)>;

    inline static cv::Scalar DefaultColourSelector(gtsam::Key) {
        static const cv::Scalar Black(0, 0, 0);
        return Black;
    }

    ColourSelector colour_selector = DefaultColourSelector;

    /**
     * @brief Constructs a set of options which has the label formatter and colour selector set to the Dyno sam versions.
     *
     * Specifcially the label formatter uses DynoLikeKeyFormatter and the colour selector uses knowledge of the keys in our
     * graph structructure to pick colours for our variables
     *
     * @param options
     * @return DrawBlockJacobiansOptions
     */
    static DrawBlockJacobiansOptions makeDynoSamOptions(std::optional<DrawBlockJacobiansOptions> options = {}) {
        DrawBlockJacobiansOptions dyno_sam_options;
        if(options) dyno_sam_options = *options;

        //override label and colour formatter fucntion for dynosam
        dyno_sam_options.label_formatter = DynoLikeKeyFormatter;

        auto dyno_sam_colour_selector = [](gtsam::Key key) {
            SymbolChar chr = DynoChrExtractor(key);
            //but gross to use an obejct related function but this just gets us a nice colour
            return ColourMap::getObjectColour((int)chr);
        };

        dyno_sam_options.colour_selector = dyno_sam_colour_selector;
        return dyno_sam_options;
    }

};

cv::Mat drawBlockJacobians(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg, const gtsam::Ordering& ordering, const DrawBlockJacobiansOptions& options);


} //factor_graph_tools


template<class GRAPH>
class FactorGraphManager {
public:
    using FactorGraphType = GRAPH;
    using This = FactorGraphManager<FactorGraphType>;

    using FactorType = typename GRAPH::FactorType;
    using sharedFactor = typename GRAPH::sharedFactor;
    using value_type = sharedFactor;

    using iterator = typename GRAPH::iterator;
    using const_iterator = typename GRAPH::const_iterator;

    FactorGraphManager(const FactorGraphType graph) : graph_(graph) {}
    virtual ~FactorGraphManager() = default;


    FactorGraphType getAssociatedFactors(gtsam::Key key) const {
        std::vector<sharedFactor> factors;

        //while the manager is NOT a grpah, it does complete MOST of the concept requirements of a
        //gtsam::FactorGraph but providing iterator, shared factor etc... type alias and std container definitions
        //(begin and end) so that we can use this with templated GRAPH functions
        const size_t num_connected = factor_graph_tools::getAssociatedFactors(factors, *this, key);

        if(num_connected < 1) {
            return FactorGraphType{};
        }

        FactorGraphType connected_factors;
        for(const auto& f : factors) {
            connected_factors += f;
        }
        return connected_factors;
    }

    virtual gtsam::GaussianFactorGraph::shared_ptr linearize() const = 0;

    const FactorGraphType& getGraph() const { return graph_; }

    gtsam::JacobianFactor constructJacobian(const gtsam::Ordering::OrderingType& ordering_type) {
        return constructJacobian(gtsam::Ordering::Create(ordering_type, graph_));
    }

    gtsam::JacobianFactor constructJacobian(const gtsam::Ordering& ordering) {
        auto gfg = this->linearize();
        return gtsam::JacobianFactor(*gfg, ordering);
    }


    cv::Mat drawBlockJacobian(const gtsam::Ordering& ordering, const factor_graph_tools::DrawBlockJacobiansOptions& options) const {
        return factor_graph_tools::drawBlockJacobians(
            this->linearize(),
            ordering,
            options
        );
    }

    cv::Mat drawBlockJacobian(const gtsam::Ordering::OrderingType& ordering_type, const factor_graph_tools::DrawBlockJacobiansOptions& options) const {
        return drawBlockJacobian(gtsam::Ordering::Create(ordering_type, graph_), options);
    }



    factor_graph_tools::SparsityStats computeJacobianSparsityStats(const gtsam::Ordering& ordering, bool save = false, const std::string& file = "jacobian.txt") const {
        factor_graph_tools::SparsityStats stats = factor_graph_tools::computeJacobianSparsityStats(this->linearize(), ordering);

        if(save) stats.saveOnOutputPath(file);

        return stats;
    }

    factor_graph_tools::SparsityStats computeHessianSparsityStats(const gtsam::Ordering& ordering, bool save = false, const std::string& file = "hessian.txt") const {
        factor_graph_tools::SparsityStats stats = factor_graph_tools::computeHessianSparsityStats(this->linearize(), ordering);

        if(save) stats.saveOnOutputPath(file);

        return stats;
    }


    //alows stl access
    const_iterator begin() const { return graph_.begin(); }
    const_iterator end() const { return graph_.end(); }

protected:
    const FactorGraphType graph_;
};


class NonlinearFactorGraphManager : public FactorGraphManager<gtsam::NonlinearFactorGraph> {

public:
    using Base = FactorGraphManager<gtsam::NonlinearFactorGraph>;

    NonlinearFactorGraphManager(const gtsam::NonlinearFactorGraph& graph, gtsam::Values& values) : Base(graph), values_(values) {}

    gtsam::GaussianFactorGraph::shared_ptr linearize() const override {
        return graph_.linearize(values_);
    }

    // /**
    //  * @brief Constructs a factor_graph_tools::DrawBlockJacobiansOptions where the colour_selector function uses the type of value
    //  * (since the NonlinearFactorGraphManager knows the values) to colour the non-zero elements.
    //  *
    //  * @param options
    //  * @return factor_graph_tools::DrawBlockJacobiansOptions
    //  */
    // factor_graph_tools::DrawBlockJacobiansOptions optionsByValue(std::optional<factor_graph_tools::DrawBlockJacobiansOptions> options);

protected:
    const gtsam::Values values_;
};


} //dyno
