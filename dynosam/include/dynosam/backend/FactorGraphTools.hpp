/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <glog/logging.h>
#include <gtsam/base/treeTraversal-inst.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/viz/types.hpp>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/utils/Numerical.hpp"
#include "dynosam/visualizer/ColourMap.hpp"

namespace dyno {

namespace factor_graph_tools {

SmartProjectionFactor::shared_ptr constructSmartProjectionFactor(
    gtsam::SharedNoiseModel smart_noise, boost::shared_ptr<CalibrationType> K,
    SmartProjectionFactorParams projection_params);

SmartProjectionFactor::shared_ptr constructSmartProjectionFactor(
    gtsam::SharedNoiseModel smart_noise, boost::shared_ptr<CalibrationType> K,
    SmartProjectionFactorParams projection_params, Keypoint measurement,
    FrameId frame_id);

void addBetweenFactor(FrameId from_frame, FrameId to_frame,
                      const gtsam::Pose3 from_pose_to,
                      gtsam::SharedNoiseModel noise_model,
                      gtsam::NonlinearFactorGraph& graph);

void addSmartProjectionMeasurement(
    SmartProjectionFactor::shared_ptr smart_factor, Keypoint measurement,
    FrameId frame_id);

// expects DERIVEDFACTOR to be a NoiseModelFactor so that we can use
// whitenedError
template <typename DERIVEDFACTOR>
gtsam::FactorIndices determineFactorOutliers(
    const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values,
    double confidence = 0.99) {
  gtsam::FactorIndices outlier_indicies;
  for (size_t k = 0; k < graph.size(); k++) {
    if (!graph[k]) continue;

    auto derived_factor = boost::dynamic_pointer_cast<DERIVEDFACTOR>(graph[k]);
    if (derived_factor) {
      double weighted_threshold =
          0.5 *
          chi_squared_quantile(
              derived_factor->dim(),
              confidence);  // 0.5 derives from the error definition in gtsam
      // NOTE: double casting as we expect DERIVEDFACTOR to be a
      // NoiseModelFactor
      gtsam::NoiseModelFactor::shared_ptr nm_factor =
          boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(derived_factor);
      CHECK(nm_factor);
      // all noise models must be gaussian otherwise whittening will be
      // reweighted
      auto robust = boost::dynamic_pointer_cast<gtsam::noiseModel::Robust>(
          nm_factor->noiseModel());
      auto gaussian_nm_factor =
          robust ? nm_factor->cloneWithNewNoiseModel(robust->noise())
                 : nm_factor;
      CHECK(gaussian_nm_factor);

      double error = gaussian_nm_factor->error(values);

      if (error > weighted_threshold) {
        outlier_indicies.push_back(k);
      }
    }
  }
  return outlier_indicies;
}

template <typename GRAPH>
size_t getAssociatedFactors(
    std::vector<typename GRAPH::const_iterator>& associated_factors,
    const GRAPH& graph, gtsam::Key query_key) {
  associated_factors.clear();

  for (auto it = graph.begin(); it != graph.end(); it++) {
    auto factor = *it;
    if (factor->find(query_key) != factor->end()) {
      associated_factors.push_back(it);
    }
  }
  return associated_factors.size();
}

template <typename GRAPH>
size_t getAssociatedFactors(gtsam::FactorIndices& associated_factors,
                            const GRAPH& graph, gtsam::Key query_key) {
  associated_factors.clear();
  std::vector<typename GRAPH::const_iterator> associated_factor_iters;
  size_t result =
      getAssociatedFactors(associated_factor_iters, graph, query_key);

  for (const typename GRAPH::const_iterator& iter : associated_factor_iters) {
    // index in graph
    associated_factors.push_back(std::distance(graph.begin(), iter));
  }
  return result;
}

template <typename GRAPH>
size_t getAssociatedFactors(
    std::vector<typename GRAPH::sharedFactor>& associated_factors,
    const GRAPH& graph, gtsam::Key query_key) {
  associated_factors.clear();
  std::vector<typename GRAPH::const_iterator> associated_factor_iters;
  size_t result =
      getAssociatedFactors(associated_factor_iters, graph, query_key);

  for (const typename GRAPH::const_iterator& iter : associated_factor_iters) {
    associated_factors.push_back(*iter);
  }
  return result;
}

// wrapper on the gtsam tree traversal functions to make life easier
struct travsersal {
 public:
  template <typename CLQUE>
  using CliqueVisitor = std::function<void(const boost::shared_ptr<CLQUE>&)>;

  // Do nothing
  template <typename CLQUE>
  inline static void NoopCliqueVisitor(const boost::shared_ptr<CLQUE>&) {}

 private:
  // helper data structures
  struct Data {};

  /**
   * @brief Definition of VISITOR_PRE struct as per gtsam::treeTraversal
   *
   * Simply contains a user defined function which takes a shared_ptr to the
   * CLIQUE as an argument. Simplies the interface through which a tree can be
   * traversed
   *
   * @tparam CLQUE
   */
  template <typename CLQUE>
  struct NodeVisitor {
    using Visitor = CliqueVisitor<CLQUE>;

    NodeVisitor(const Visitor& visitor) : visitor_(visitor) {}

    // This operator can be used for both pre and post order traversal since the
    // templated search does not care about the (existance of) return type in
    // the pre-vis case
    Data operator()(const boost::shared_ptr<CLQUE>& clique, Data&) {
      visitor_(clique);
      return Data{};
    }

    const Visitor visitor_;
  };

 public:
  /**
   * @brief Traverses a tree and calls the user defined visitor function at
   * every pre-visit.
   *
   *
   * @tparam CLIQUE
   * @param bayes_tree
   * @param pre_op_visitor
   */
  template <typename CLIQUE>
  static void depthFirstTraversal(const gtsam::BayesTree<CLIQUE>& bayes_tree,
                                  const CliqueVisitor<CLIQUE>& post_op_visitor,
                                  const CliqueVisitor<CLIQUE>& pre_op_visitor) {
    Data rootdata;
    NodeVisitor<CLIQUE> post_visitor(post_op_visitor);
    NodeVisitor<CLIQUE> pre_visitor(pre_op_visitor);
    // TODO: see gtsam/inference/inference-inst.h where the elimate function is
    // actually called they have both post and pre order traversal and i'm not
    // sure which one is used... the bayes tree paper talkes about solving where
    // a parse up from the leaves is computed and then a pass down where the
    // optimal assignment is retrieved I believe that the pass up (which must be
    // post-order) is also equivalent to the elimination ordering

    // NOTE: not using threaded version
    // TbbOpenMPMixedScope threadLimiter; // Limits OpenMP threads since we're
    // mixing TBB and OpenMP treeTraversal::DepthFirstForestParallel(bayesTree,
    // rootData, preVisitor, postVisitor);
    gtsam::treeTraversal::DepthFirstForest(bayes_tree, rootdata, pre_visitor,
                                           post_visitor);
  }

  template <typename CLIQUE>
  static void depthFirstTraversalEliminiationOrder(
      const gtsam::BayesTree<CLIQUE>& bayes_tree,
      const CliqueVisitor<CLIQUE>& visitor) {
    depthFirstTraversal(bayes_tree, visitor, NoopCliqueVisitor<CLIQUE>);
  }

  /// Gets keys in elimination order by traversing the tree in post-order
  template <typename CLIQUE>
  static gtsam::KeySet getEliminatonOrder(
      const gtsam::BayesTree<CLIQUE>& bayes_tree) {
    gtsam::KeySet keys;
    CliqueVisitor<CLIQUE> visitor =
        [&keys](const boost::shared_ptr<CLIQUE>& clique) -> void {
      auto conditional = clique->conditional();
      // it is FACTOR::const_iterator
      for (auto it = conditional->beginFrontals();
           it != conditional->endFrontals(); it++) {
        keys.insert(*it);
      }
    };

    depthFirstTraversalEliminiationOrder(bayes_tree, visitor);

    return keys;
  }

  template <typename CLIQUE>
  static gtsam::KeySet getLeafKeys(const gtsam::BayesTree<CLIQUE>& bayes_tree) {
    gtsam::KeySet keys;
    CliqueVisitor<CLIQUE> visitor =
        [&keys](const boost::shared_ptr<CLIQUE>& clique) -> void {
      if (clique->children.size() == 0u) {
        auto conditional = clique->conditional();
        // it is FACTOR::const_iterator
        for (auto it = conditional->beginFrontals();
             it != conditional->endFrontals(); it++) {
          keys.insert(*it);
        }
      }
    };

    depthFirstTraversalEliminiationOrder(bayes_tree, visitor);

    return keys;
  }
};

struct SparsityStats {
  gtsam::Matrix matrix;
  size_t nr_zero_elements{0};
  size_t nr_elements{0};

  SparsityStats(){};
  SparsityStats(const gtsam::Matrix& M)
      : matrix(M), nr_zero_elements(0), nr_elements(0) {
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

SparsityStats computeHessianSparsityStats(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const std::optional<gtsam::Ordering>& ordering = {});
SparsityStats computeJacobianSparsityStats(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const std::optional<gtsam::Ordering>& ordering = {});

// TODO: we can make all these printing functions really nice in the future
// for now, just make for printing affected keys or regular printing
template <class CLIQUE>
void dotBayesTree(
    std::ostream& s, boost::shared_ptr<CLIQUE> clique,
    const gtsam::KeyFormatter& keyFormatter,
    const gtsam::FastMap<gtsam::Key, std::string>& colour_map = {},
    int parentnum = 0) {
  static int num = 0;
  bool first = true;
  std::stringstream out;
  out << num;
  std::string parent = out.str();
  parent += "[label=\"";

  // frontal key since this is what is used to add to nodes [j] (ie.e j is a
  // frontal key)
  bool keyInColourMap = false;
  std::string colour;

  for (auto key : clique->conditional_->frontals()) {
    if (!first) parent += ", ";
    first = false;
    parent += keyFormatter(key);

    keyInColourMap = colour_map.exists(key);
    // only for the last key? Assumes all keys have the same colour?
    if (keyInColourMap) colour = colour_map.at(key);
  }

  if (clique->parent()) {
    parent += " : ";
    s << parentnum << "->" << num << "\n";
  }

  first = true;
  for (auto parentKey : clique->conditional_->parents()) {
    if (!first) parent += ", ";
    first = false;
    parent += keyFormatter(parentKey);
  }
  //  parent += "\"];\n";

  if (keyInColourMap) {
    parent += "\",color=" + colour;
    parent += "];\n";
  } else {
    parent += "\"];\n";
  }

  s << parent;
  parentnum = num;

  for (auto c : clique->children) {
    num++;
    dotBayesTree(s, c, keyFormatter, colour_map, parentnum);
  }
}

template <class CLIQUE>
void saveBayesTree(
    const gtsam::BayesTree<CLIQUE>& tree, const std::string& filename,
    const gtsam::KeyFormatter& keyFormatter,
    const gtsam::FastMap<gtsam::Key, std::string> colour_map = {}) {
  std::ofstream of(filename.c_str());

  if (tree.roots().empty())
    throw std::invalid_argument(
        "the root of Bayes tree has not been initialized!");
  of << "digraph G{\n";
  for (const auto& root : tree.roots())
    dotBayesTree(of, root, keyFormatter, colour_map);
  of << "}";
  std::flush(of);
}

std::pair<SparsityStats, cv::Mat> computeRFactor(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const gtsam::Ordering ordering);

struct DrawBlockJacobiansOptions {
  cv::Size desired_size = cv::Size(480, 480);

  bool draw_label = true;    // if true, the labels for each vertical block (ie,
                             // the variable names) will be draw atop each block
  int text_box_height = 30;  // text box for label
  gtsam::KeyFormatter label_formatter =
      gtsam::DefaultKeyFormatter;  // function used to generate the label, if
                                   // draw_label== true

  //! Alias to a function that takes a key and returns a colour. Used to colour
  //! the non-zero elements of the Jacobian depending on the variable (block)
  using ColourSelector = std::function<cv::Scalar(gtsam::Key)>;

  inline static cv::Scalar DefaultColourSelector(gtsam::Key) {
    return cv::viz::Color::black();
  }

  ColourSelector colour_selector = DefaultColourSelector;

  /**
   * @brief Constructs a set of options which has the label formatter and colour
   * selector set to the Dyno sam versions.
   *
   * Specifcially the label formatter uses DynoLikeKeyFormatter and the colour
   * selector uses knowledge of the keys in our graph structructure to pick
   * colours for our variables
   *
   * @param options
   * @return DrawBlockJacobiansOptions
   */
  static DrawBlockJacobiansOptions makeDynoSamOptions(
      std::optional<DrawBlockJacobiansOptions> options = {}) {
    DrawBlockJacobiansOptions dyno_sam_options;
    if (options) dyno_sam_options = *options;

    // override label and colour formatter fucntion for dynosam
    dyno_sam_options.label_formatter = DynoLikeKeyFormatter;

    auto dyno_sam_colour_selector = [](gtsam::Key key) {
      return Color::uniqueId((size_t)key);
    };

    dyno_sam_options.colour_selector = dyno_sam_colour_selector;
    return dyno_sam_options;
  }
};

cv::Mat drawBlockJacobians(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
                           const gtsam::Ordering& ordering,
                           const DrawBlockJacobiansOptions& options);

template <typename T>
void toGraphFileFormat(std::ostream& os, const T& t);

template <typename T>
std::ostream& seralizeFactorToGraphFileFormat(std::ostream& os, const T& t,
                                              const std::string& tag) {
  static_assert(is_gtsam_factor_v<T>);

  // TAG <keys> VALUE
  // where the value is provided by toGraphFileFormat and should be in the form
  // <measurement> <covaraince>

  os << tag << " ";
  const gtsam::Factor& factor = static_cast<const gtsam::Factor&>(t);
  for (const gtsam::Key key : factor) {
    os << key << " ";
  }
  toGraphFileFormat<T>(os, t);
  os << '\n';
  return os;
}

template <typename T>
std::ostream& seralizeValueToGraphFileFormat(std::ostream& os, const T& t,
                                             gtsam::Key key,
                                             const std::string& tag) {
  os << tag << " " << key << " ";
  // TAG <key> VALUE
  // where the value is provided by toGraphFileFormat and should be the value
  // itself
  toGraphFileFormat<T>(os, t);
  os << '\n';
  return os;
}

}  // namespace factor_graph_tools

template <class GRAPH>
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

    // while the manager is NOT a grpah, it does complete MOST of the concept
    // requirements of a gtsam::FactorGraph but providing iterator, shared
    // factor etc... type alias and std container definitions (begin and end) so
    // that we can use this with templated GRAPH functions
    const size_t num_connected =
        factor_graph_tools::getAssociatedFactors(factors, *this, key);

    if (num_connected < 1) {
      return FactorGraphType{};
    }

    FactorGraphType connected_factors;
    for (const auto& f : factors) {
      connected_factors += f;
    }
    return connected_factors;
  }

  virtual gtsam::GaussianFactorGraph::shared_ptr linearize() const = 0;

  const FactorGraphType& getGraph() const { return graph_; }

  gtsam::JacobianFactor constructJacobian(
      const gtsam::Ordering::OrderingType& ordering_type) {
    return constructJacobian(gtsam::Ordering::Create(ordering_type, graph_));
  }

  gtsam::JacobianFactor constructJacobian(const gtsam::Ordering& ordering) {
    auto gfg = this->linearize();
    return gtsam::JacobianFactor(*gfg, ordering);
  }

  cv::Mat drawBlockJacobian(
      const gtsam::Ordering& ordering,
      const factor_graph_tools::DrawBlockJacobiansOptions& options) const {
    return factor_graph_tools::drawBlockJacobians(this->linearize(), ordering,
                                                  options);
  }

  cv::Mat drawBlockJacobian(
      const gtsam::Ordering::OrderingType& ordering_type,
      const factor_graph_tools::DrawBlockJacobiansOptions& options) const {
    return drawBlockJacobian(gtsam::Ordering::Create(ordering_type, graph_),
                             options);
  }

  factor_graph_tools::SparsityStats computeJacobianSparsityStats(
      const gtsam::Ordering& ordering, bool save = false,
      const std::string& file = "jacobian.txt") const {
    factor_graph_tools::SparsityStats stats =
        factor_graph_tools::computeJacobianSparsityStats(this->linearize(),
                                                         ordering);

    if (save) stats.saveOnOutputPath(file);

    return stats;
  }

  factor_graph_tools::SparsityStats computeHessianSparsityStats(
      const gtsam::Ordering& ordering, bool save = false,
      const std::string& file = "hessian.txt") const {
    factor_graph_tools::SparsityStats stats =
        factor_graph_tools::computeHessianSparsityStats(this->linearize(),
                                                        ordering);

    if (save) stats.saveOnOutputPath(file);

    return stats;
  }

  // alows stl access
  const_iterator begin() const { return graph_.begin(); }
  const_iterator end() const { return graph_.end(); }

 protected:
  const FactorGraphType graph_;
};

class NonlinearFactorGraphManager
    : public FactorGraphManager<gtsam::NonlinearFactorGraph> {
 public:
  using Base = FactorGraphManager<gtsam::NonlinearFactorGraph>;

  NonlinearFactorGraphManager(const gtsam::NonlinearFactorGraph& graph,
                              const gtsam::Values& values)
      : Base(graph), values_(values) {}

  gtsam::GaussianFactorGraph::shared_ptr linearize() const override {
    return graph_.linearize(values_);
  }

  // strong dependancy on a factor graph constructed from THIS system
  // e.g. using the type of symbols defined in dynosam and the type of factors
  // we use the type of symbol and the characters here to determine the type of
  // factor and the USE of this factor (e.g. a between factor could be a
  // smoothing factor or an odom factor)
  void writeDynosamGraphFile(const std::string& filename) const;

  // /**
  //  * @brief Constructs a factor_graph_tools::DrawBlockJacobiansOptions where
  //  the colour_selector function uses the type of value
  //  * (since the NonlinearFactorGraphManager knows the values) to colour the
  //  non-zero elements.
  //  *
  //  * @param options
  //  * @return factor_graph_tools::DrawBlockJacobiansOptions
  //  */
  // factor_graph_tools::DrawBlockJacobiansOptions
  // optionsByValue(std::optional<factor_graph_tools::DrawBlockJacobiansOptions>
  // options);

 protected:
  const gtsam::Values values_;
};

}  // namespace dyno
