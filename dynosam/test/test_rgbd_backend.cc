/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/debug.h>
#include <gtsam/linear/GaussianEliminationTree.h>
#include <gtsam/nonlinear/ISAM2-impl.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearISAM.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <exception>
#include <iterator>
#include <vector>

#include "dynosam/backend/BackendPipeline.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/RGBDBackendModule.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/frontend/FrontendPipeline.hpp"
#include "internal/helpers.hpp"
#include "internal/simulator.hpp"

void recursiveMarkAffectedKeys(const gtsam::Key& key,
                               const gtsam::ISAM2Clique::shared_ptr& clique,
                               std::set<gtsam::Key>& additionalKeys) {
  // Check if the separator keys of the current clique contain the specified key
  if (std::find(clique->conditional()->beginParents(),
                clique->conditional()->endParents(),
                key) != clique->conditional()->endParents()) {
    // Mark the frontal keys of the current clique
    for (gtsam::Key i : clique->conditional()->frontals()) {
      additionalKeys.insert(i);
    }

    // Recursively mark all of the children
    for (const gtsam::ISAM2Clique::shared_ptr& child : clique->children) {
      recursiveMarkAffectedKeys(key, child, additionalKeys);
    }
  }
  // If the key was not found in the separator/parents, then none of its
  // children can have it either
}

TEST(RGBDBackendModule, smallKITTIDataset) {
  using namespace dyno;
  using OfflineFrontend =
      FrontendOfflinePipeline<RGBDBackendModule::ModuleTraits>;
  const auto file_path = getTestDataPath() + "/small_frontend.bson";

  OfflineFrontend::UniquePtr offline_frontend =
      std::make_unique<OfflineFrontend>("offline-rgbdfrontend", file_path);

  dyno::BackendParams backend_params;
  backend_params.useLogger(false);
  backend_params.min_dynamic_obs_ = 1u;

  dyno::RGBDBackendModule backend(
      backend_params, dyno_testing::makeDefaultCameraPtr(),
      dyno::RGBDBackendModule::UpdaterType::ObjectCentric);

  gtsam::ISAM2Params isam2_params;
  isam2_params.factorization = gtsam::ISAM2Params::Factorization::QR;
  // isam2_params.relinearizeSkip = 1;
  gtsam::ISAM2 isam2(isam2_params);
  //   gtsam::NonlinearISAM isam(1, gtsam::EliminateQR);
  // gtsam::IncrementalFixedLagSmoother smoother(2.0, isam2_params);

  backend.callback =
      [&](const dyno::Formulation<dyno::Map3d2d>::UniquePtr& formulation,
          dyno::FrameId frame_id, const gtsam::Values& new_values,
          const gtsam::NonlinearFactorGraph& new_factors) -> void {
    LOG(INFO) << "In backend callback " << frame_id;

    auto result = isam2.update(new_factors, new_values);

    isam2.getFactorsUnsafe().saveGraph(
        dyno::getOutputFilePath("small_isam_graph_" + std::to_string(frame_id) +
                                ".dot"),
        dyno::DynoLikeKeyFormatter);

    gtsam::FastMap<gtsam::Key, std::string> coloured_affected_keys;
    for (const auto& key : result.markedKeys) {
      coloured_affected_keys.insert2(key, "red");
    }

    dyno::factor_graph_tools::saveBayesTree(
        isam2,
        dyno::getOutputFilePath("small_oc_bayes_tree_" +
                                std::to_string(frame_id) + ".dot"),
        dyno::DynoLikeKeyFormatter, coloured_affected_keys);
  };

  auto output = offline_frontend->process(offline_frontend->getInputPacket());
  while (output != nullptr) {
    backend.spinOnce(output);
    output = offline_frontend->process(offline_frontend->getInputPacket());
  }
}

TEST(RGBDBackendModule, constructSimpleGraph) {
  // make camera with a constant motion model starting at zero
  dyno_testing::ScenarioBody::Ptr camera =
      std::make_shared<dyno_testing::ScenarioBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3::Identity(),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, 0.1, 0.0),
                           gtsam::Point3(0.1, 0.05, 0))));
  // needs to be at least 3 overlap so we can meet requirements in graph
  // TODO: how can we do 1 point but with lots of overlap (even infinity
  // overlap?)

  const double H_R_sigma = 0.1;
  const double H_t_sigma = 0.2;
  const double dynamic_point_sigma = 0.1;

  dyno_testing::RGBDScenario::NoiseParams noise_params;
  noise_params.H_R_sigma = H_R_sigma;
  noise_params.H_t_sigma = H_t_sigma;
  noise_params.dynamic_point_sigma = dynamic_point_sigma;

  dyno_testing::RGBDScenario scenario(
      camera, std::make_shared<dyno_testing::SimpleStaticPointsGenerator>(7, 5),
      noise_params);

  // add one obect
  const size_t num_points = 10;
  const size_t obj1_overlap = 4;
  const size_t obj2_overlap = 5;
  dyno_testing::ObjectBody::Ptr object1 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(2, 0, 0)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.2, 0.1, 0.0),
                           gtsam::Point3(0.2, 0, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj1_overlap));

  dyno_testing::ObjectBody::Ptr object2 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1, 0.4, 0.1)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.2, 0, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj2_overlap));

  scenario.addObjectBody(1, object1);
  scenario.addObjectBody(2, object2);

  //   SETDEBUG("IncrementalFixedLagSmoother update", true);
  //   CHECK(gtsam::isDebugVersion());
  //   CHECK(ISDEBUG("IncrementalFixedLagSmoother update"));
  dyno::BackendParams backend_params;
  backend_params.useLogger(false);
  backend_params.min_dynamic_obs_ = 3u;
  backend_params.dynamic_point_noise_sigma_ = dynamic_point_sigma;

  //   dyno::FormulationHooks hooks;
  //   hooks.ground_truth_packets_request = [&scenario] () ->
  //   std::optional<GroundTruthPacketMap> {
  //     return scenario.getGroundTruths();
  //   };

  //   hooks.backend_params_request = [&] () -> const BackendParams& {
  //     return backend_params;
  //   };

  dyno::RGBDBackendModule backend(
      backend_params, dyno_testing::makeDefaultCameraPtr(),
      dyno::RGBDBackendModule::UpdaterType::MotionInWorld);

  gtsam::ISAM2Params isam2_params;
  isam2_params.evaluateNonlinearError = true;
  isam2_params.factorization = gtsam::ISAM2Params::Factorization::QR;
  // isam2_params.relinearizeSkip = 1;
  gtsam::ISAM2 isam2(isam2_params);
  //   gtsam::NonlinearISAM isam(1, gtsam::EliminateQR);
  // gtsam::IncrementalFixedLagSmoother smoother(2.0, isam2_params);

  gtsam::Values opt_values;

  backend.callback =
      [&](const dyno::Formulation<dyno::Map3d2d>::UniquePtr& formulation,
          dyno::FrameId frame_id, const gtsam::Values& new_values,
          const gtsam::NonlinearFactorGraph& new_factors) -> void {
    LOG(INFO) << "In backend callback " << frame_id;
    // return;

    // gtsam::IncrementalFixedLagSmoother::KeyTimestampMap timestamp_map;
    // for (const auto& key_value : new_values) {

    //     gtsam::Symbol sym(key_value.key);
    //     if(sym.chr() == dyno::kPoseSymbolChar) {
    //         timestamp_map[key_value.key] =  2;
    //     }

    //     dyno::ObjectId object_id;
    //     dyno::FrameId k;
    //     if(dyno::reconstructMotionInfo(key_value.key, object_id, k)) {
    //         if(k <= frame_id) {
    //             //get smoother to remove key
    //             timestamp_map[key_value.key] =  0;
    //             LOG(INFO) << "Remove Key " <<
    //             dyno::DynoLikeKeyFormatter(key_value.key);
    //         }
    //         else {
    //             timestamp_map[key_value.key] =  1;
    //         }
    //     }
    //     else {
    //         timestamp_map[key_value.key] =  1;
    //     }

    // }

    gtsam::KeyVector marginalizableKeys;
    gtsam::FastMap<gtsam::Key, int> constrained_keys;
    gtsam::FastList<gtsam::Key> noRelinKeys;
    // for (const auto& factors : new_factors) {
    //   gtsam::KeyVector keys = factors->keys();
    //   for(const auto key : keys) {
    //     dyno::ObjectId object_id;
    //     dyno::FrameId k;
    //     if(dyno::reconstructMotionInfo(key, object_id, k)) {
    //         if(k < frame_id)  marginalizableKeys.push_back(key);
    //     }

    //   }
    // }

    // new keys
    const auto new_keys = new_values.keys();
    const auto old_keys = isam2.getLinearizationPoint().keys();

    for (const auto key : old_keys) {
      dyno::ObjectId object_id;
      dyno::FrameId k;
      if (dyno::reconstructMotionInfo(key, object_id, k)) {
        if (k < frame_id) {
          noRelinKeys.push_back(key);
        }
      }
    }
    // for (const auto key : new_keys) {
    //   dyno::ObjectId object_id;
    //   dyno::FrameId k;
    // //   constrained_keys[key] = 0;
    //   if (dyno::reconstructMotionInfo(key, object_id, k)) {
    //     // if(k < frame_id) {
    //     //     marginalizableKeys.push_back(key);
    //     //     LOG(INFO) << "Adding marginal var at" << object_id << " f = "
    //     <<
    //     //     k; constrained_keys[key] = 0;
    //     // }
    //     // else {
    //     //     constrained_keys[key] = 1;
    //     // }
    //     // if (k == frame_id) {
    //     //   constrained_keys[key] = 1;
    //     // }
    //   }
    //   // else {
    //   //     constrained_keys[key] = 1;
    //   // }
    // }
    // for(const auto key : new_keys) {
    //     constrained_keys[key] = 1;
    // }

    for (const auto& key_value : new_values) {
      gtsam::Symbol sym(key_value.key);
      // put object motion keys at start
      if (sym.chr() == dyno::kObjectMotionSymbolChar) {
        constrained_keys[(gtsam::Key)sym] = 1;
        // marginalizableKeys.push_back(key_value.key);
      }
      // else {
      //     constrained_keys[(gtsam::Key)sym] = 1;
      // }

      if (sym.chr() == dyno::kStaticLandmarkSymbolChar) {
        constrained_keys[(gtsam::Key)sym] = 0;
      } else if (sym.chr() == dyno::kPoseSymbolChar) {
        constrained_keys[(gtsam::Key)sym] = 0;
      } else if (sym.chr() == dyno::kDynamicLandmarkSymbolChar) {
        constrained_keys[(gtsam::Key)sym] = 1;
      }
      //   else{
      //     constrained_keys[(gtsam::Key)sym] = 3;
      //   }
      //   const gtsam::Value& value = key_value.value;

      //   // Print key using the custom KeyFormatter
      //   std::cout << gtsam::DefaultKeyFormatter(key) << std::endl;
    }

    std::set<gtsam::Key> additionalKeys;
    for (auto key : marginalizableKeys) {
      LOG(INFO) << "Marginal keys " << gtsam::DefaultKeyFormatter(key);
      gtsam::ISAM2Clique::shared_ptr clique = isam2[key];
      for (const gtsam::ISAM2Clique::shared_ptr& child : clique->children) {
        recursiveMarkAffectedKeys(key, child, additionalKeys);
      }
    }

    gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(),
                                        additionalKeys.end());

    gtsam::ISAM2UpdateParams update_params;
    update_params.noRelinKeys = noRelinKeys;
    // update_params.constrainedKeys = constrained_keys;
    // update_params.extraReelimKeys = additionalMarkedKeys;

    // for (const auto& factors : new_factors) {
    //   factors->printKeys("Factor: ");
    // }

    gtsam::ISAM2Result result;
    try {
      // smoother.update(new_factors, new_values, timestamp_map);
      // result = smoother.getISAM2Result();
      result = isam2.update(new_factors, new_values, update_params);

      isam2.getFactorsUnsafe().saveGraph(
          dyno::getOutputFilePath("isam_graph_" + std::to_string(frame_id) +
                                  ".dot"),
          dyno::DynoLikeKeyFormatter);

      // if (marginalizableKeys.size() > 0) {
      //     gtsam::FastList<gtsam::Key> leafKeys(marginalizableKeys.begin(),
      //         marginalizableKeys.end());
      //     isam2.marginalizeLeaves(leafKeys);
      // }

    } catch (gtsam::IndeterminantLinearSystemException& e) {
      LOG(INFO) << "Caught exception " << e.what();
      const auto& graph = formulation->getGraph();
      const auto& initial_estimate = formulation->getTheta();

      graph.saveGraph(
          dyno::getOutputFilePath("construct_simple_graph_fail_test_" +
                                  std::to_string(frame_id) + ".dot"));

      // LOG(INFO) << "Full graph";
      //  for (const auto& key_value : initial_estimate) {
      // gtsam::Key key = key_value.key;
      // // const gtsam::Value& value = key_value.value;

      // // Print key using the custom KeyFormatter
      // std::cout << gtsam::DefaultKeyFormatter(key) << std::endl;
      // }

      // dyno::NonlinearFactorGraphManager nlfg(graph, initial_estimate);
      // cv::Mat J = nlfg.drawBlockJacobian(gtsam::Ordering::COLAMD,
      // dyno::factor_graph_tools::DrawBlockJacobiansOptions());
      // cv::imshow("Jacobians", J);
      // cv::waitKey(0);
    }

    if (isam2.empty()) return;

    // smoother.

    // isam2.calculateBestEstimate();
    gtsam::FastMap<gtsam::Key, std::string> coloured_affected_keys;
    for (const auto& key : result.markedKeys) {
      coloured_affected_keys.insert2(key, "red");
    }

    // dyno::factor_graph_tools::saveBayesTree(
    //     smoother.getISAM2(),
    //     dyno::getOutputFilePath("oc_bayes_tree_" + std::to_string(frame_id) +
    //                             ".dot"),
    //     dyno::DynoLikeKeyFormatter,
    //     coloured_affected_keys);

    dyno::factor_graph_tools::saveBayesTree(
        isam2,
        dyno::getOutputFilePath("oc_bayes_tree_" + std::to_string(frame_id) +
                                ".dot"),
        dyno::DynoLikeKeyFormatter, coloured_affected_keys);
    //  dyno::factor_graph_tools::saveBayesTree(
    //     isam.bayesTree(),
    //     dyno::getOutputFilePath("oc_bayes_tree_" + std::to_string(frame_id) +
    //                             ".dot"),
    //     dyno::DynoLikeKeyFormatter,
    //     coloured_affected_keys);

    LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
              << " error after " << result.getErrorAfter();
    opt_values = isam2.calculateEstimate();
    // const auto& graph = formulation->getGraph();
    // const auto& theta = formulation->getTheta();
    // LOG(INFO) << "Formulation error: " << graph.error(theta);

    // gtsam::LevenbergMarquardtOptimizer problem(graph, theta);
    // // save the result of the optimisation and log after all runs
    // opt_values = problem.optimize();
    // LOG(INFO) << "Formulation error after: " << graph.error(opt_values);
  };

  for (size_t i = 0; i < 10; i++) {
    dyno::RGBDInstanceOutputPacket::Ptr output_gt, output_noisy;
    std::tie(output_gt, output_noisy) = scenario.getOutput(i);

    // std::stringstream ss;
    // ss << output_gt->T_world_camera_ << "\n";
    // ss << dyno::container_to_string(output->dynamic_landmarks_);

    // LOG(INFO) << ss.str();
    backend.spinOnce(output_noisy);

    LOG(INFO) << "Spun backend";
  }

  gtsam::NonlinearFactorGraph full_graph = backend.new_updater_->getGraph();
  full_graph.saveGraph(
      dyno::getOutputFilePath("construct_simple_graph_test.dot"),
      dyno::DynoLikeKeyFormatter);

  // log results of LM optimisation with different suffix
  dyno::BackendMetaData backend_info;
  backend.new_updater_->accessorFromTheta()->postUpdateCallback(backend_info);
  backend.new_updater_->logBackendFromMap(backend_info);

  backend_info.suffix = "LM_opt";
  backend.new_updater_->updateTheta(opt_values);
  backend.new_updater_->accessorFromTheta()->postUpdateCallback(backend_info);
  backend.new_updater_->logBackendFromMap(backend_info);
}

TEST(RGBDBackendModule, testCliques) {
  using namespace dyno;
  // the simplest dynamic graph
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;

  gtsam::Pose3 H_0_1 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.3, 0.2, 0.1),
                                    gtsam::Point3(0, 1, 1));
  gtsam::Pose3 H_1_2 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.5, 0.1, 0.1),
                                    gtsam::Point3(1, 1.5, 1));
  gtsam::Pose3 H_0_2 = H_0_1 * H_1_2;

  gtsam::Point3 dyn_point_1_world = gtsam::Point3(2, 1, 3);
  gtsam::Point3 dyn_point_2_world = gtsam::Point3(1, 1, 3);
  gtsam::Point3 dyn_point_3_world = gtsam::Point3(3, 0.5, 2);

  static gtsam::Point3 p0 = gtsam::Point3(1, 3, 4);
  static gtsam::Rot3 R0 =
      gtsam::Rot3::Expmap((gtsam::Vector(3) << 0.2, 0.1, 0.12).finished());
  static gtsam::Point3 p1 = gtsam::Point3(1, 2, 1);
  static gtsam::Rot3 R1 =
      gtsam::Rot3::Expmap((gtsam::Vector(3) << 0.1, 0.2, 1.570796).finished());
  static gtsam::Point3 p2 = gtsam::Point3(2, 2, 1);
  static gtsam::Rot3 R2 =
      gtsam::Rot3::Expmap((gtsam::Vector(3) << 0.2, 0.3, 3.141593).finished());
  static gtsam::Point3 p3 = gtsam::Point3(-1, 1, 0);
  static gtsam::Rot3 R3 =
      gtsam::Rot3::Expmap((gtsam::Vector(3) << 0.1, 0.4, 4.712389).finished());

  static gtsam::Pose3 pose0 = gtsam::Pose3(R0, p0);
  static gtsam::Pose3 pose1 = gtsam::Pose3(R1, p1);
  static gtsam::Pose3 pose2 = gtsam::Pose3(R2, p2);
  static gtsam::Pose3 pose3 = gtsam::Pose3(R3, p3);

  auto landmark_noise = gtsam::noiseModel::Isotropic::Sigma(3u, 10);

  // static point seen at frames 0, 1 and 2
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(0), StaticLandmarkSymbol(0), gtsam::Point3(1, 2, 3),
      landmark_noise);
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(1), StaticLandmarkSymbol(0), gtsam::Point3(2, 2, 3),
      landmark_noise);

  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(2), StaticLandmarkSymbol(0), gtsam::Point3(3, 2, 3),
      landmark_noise);

  // motion between frames 0 and 1
  // add motion factor for the 3 tracklet with tracklet id = 1, 2, 3
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(0, 1), DynamicLandmarkSymbol(1, 1),
      ObjectMotionSymbol(1, 1), landmark_noise);

  // motion between frames 1 and 2
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(1, 1), DynamicLandmarkSymbol(2, 1),
      ObjectMotionSymbol(1, 2), landmark_noise);

  // tracklet 2
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(0, 2), DynamicLandmarkSymbol(1, 2),
      ObjectMotionSymbol(1, 1), landmark_noise);
  // motion between frames 1 and 2
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(1, 2), DynamicLandmarkSymbol(2, 2),
      ObjectMotionSymbol(1, 2), landmark_noise);

  // tracklet 3
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(0, 3), DynamicLandmarkSymbol(1, 3),
      ObjectMotionSymbol(1, 1), landmark_noise);
  // motion between frames 1 and 2
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(1, 3), DynamicLandmarkSymbol(2, 3),
      ObjectMotionSymbol(1, 2), landmark_noise);

  // tracklet 1
  // add dynamic point obs frame 0
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(0), DynamicLandmarkSymbol(0, 1),
      pose0.inverse() * dyn_point_1_world, landmark_noise);

  // add dynamic point obs frame 1
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(1), DynamicLandmarkSymbol(1, 1),
      pose1.inverse() * H_0_1 * dyn_point_1_world, landmark_noise);

  // add dynamic point obs frame 2
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(2), DynamicLandmarkSymbol(2, 1),
      pose2.inverse() * H_0_2 * dyn_point_1_world, landmark_noise);

  // tracklet 2
  // add dynamic point obs frame 0
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(0), DynamicLandmarkSymbol(0, 2),
      pose0.inverse() * dyn_point_2_world, landmark_noise);

  // add dynamic point obs frame 1
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(1), DynamicLandmarkSymbol(1, 2),
      pose1.inverse() * H_0_1 * dyn_point_2_world, landmark_noise);

  // add dynamic point obs frame 2
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(2), DynamicLandmarkSymbol(2, 2),
      pose2.inverse() * H_0_2 * dyn_point_2_world, landmark_noise);

  // tracklet 3
  // add dynamic point obs frame 0
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(0), DynamicLandmarkSymbol(0, 3),
      pose0.inverse() * dyn_point_3_world, landmark_noise);

  // add dynamic point obs frame 1
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(1), DynamicLandmarkSymbol(1, 3),
      pose1.inverse() * H_0_1 * dyn_point_3_world, landmark_noise);

  // add dynamic point obs frame 2
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(2), DynamicLandmarkSymbol(2, 3),
      pose2.inverse() * H_0_2 * dyn_point_3_world, landmark_noise);

  static gtsam::SharedNoiseModel pose_model(
      gtsam::noiseModel::Isotropic::Sigma(6, 0.1));

  // add two poses
  graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
      CameraPoseSymbol(0), CameraPoseSymbol(1), pose0.between(pose1),
      pose_model));
  graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
      CameraPoseSymbol(1), CameraPoseSymbol(2), pose1.between(pose2),
      pose_model));

  // add prior on first pose
  graph.addPrior(CameraPoseSymbol(0), pose0, pose_model);

  // now add inital values

  // add static point
  initial.insert(StaticLandmarkSymbol(0), gtsam::Point3(0, 1, 1));

  // add dynamic points

  // frame 0
  initial.insert(DynamicLandmarkSymbol(0, 1), dyn_point_1_world);
  initial.insert(DynamicLandmarkSymbol(0, 2), dyn_point_2_world);
  initial.insert(DynamicLandmarkSymbol(0, 3), dyn_point_3_world);

  // frame 1
  initial.insert(DynamicLandmarkSymbol(1, 1), H_0_1 * dyn_point_1_world);
  initial.insert(DynamicLandmarkSymbol(1, 2), H_0_1 * dyn_point_2_world);
  initial.insert(DynamicLandmarkSymbol(1, 3), H_0_1 * dyn_point_3_world);

  // frame 2
  initial.insert(DynamicLandmarkSymbol(2, 1), H_0_2 * dyn_point_1_world);
  initial.insert(DynamicLandmarkSymbol(2, 2), H_0_2 * dyn_point_2_world);
  initial.insert(DynamicLandmarkSymbol(2, 3), H_0_2 * dyn_point_3_world);

  // add two motions
  initial.insert(ObjectMotionSymbol(1, 1), gtsam::Pose3::Identity());
  initial.insert(ObjectMotionSymbol(1, 2), gtsam::Pose3::Identity());

  // add three poses
  initial.insert(CameraPoseSymbol(0), pose0);
  initial.insert(CameraPoseSymbol(1), pose1);
  initial.insert(CameraPoseSymbol(2), pose2);

  dyno::NonlinearFactorGraphManager nlfgm(graph, initial);
  nlfgm.writeDynosamGraphFile(dyno::getOutputFilePath("test_graph.g2o"));

  // graph.saveGraph(dyno::getOutputFilePath("small_graph.dot"),
  // dyno::DynoLikeKeyFormatter);
  gtsam::ISAM2BayesTree::shared_ptr bayesTree = nullptr;
  {  //
    // gtsam::ISAM2 isam2;
    gtsam::VariableIndex affectedFactorsVarIndex(graph);
    gtsam::Ordering order = gtsam::Ordering::Colamd(affectedFactorsVarIndex);
    auto linearized = graph.linearize(initial);

    bayesTree = gtsam::ISAM2JunctionTree(
                    gtsam::GaussianEliminationTree(
                        *linearized, affectedFactorsVarIndex, order))
                    .eliminate(gtsam::EliminatePreferCholesky)
                    .first;

    bayesTree->saveGraph(dyno::getOutputFilePath("elimated_tree.dot"),
                         dyno::DynoLikeKeyFormatter);
  }

  // {

  //     gtsam::ISAM2UpdateParams isam_update_params;

  //     gtsam::FastMap<gtsam::Key, int> constrainedKeys;
  //     //this includes the motions, where do we want these?
  //     //maybe BETWEEN the dynnamic keys
  //     //we want motions always on the right but impossible since parent?
  //     for(const auto& [keys, value] : initial) {
  //         constrainedKeys.insert2(keys, 1);
  //     }
  //     //put previous dynamic keys lower in the graph
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(0, 1), 0);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(1, 1), 0);

  //     constrainedKeys.insert2(DynamicLandmarkSymbol(0, 2), 0);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(1, 2), 0);

  //     constrainedKeys.insert2(DynamicLandmarkSymbol(0, 2), 0);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(1, 2), 0);

  //     ///put current keys later in the graph
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 1), 2);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 2), 2);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 3), 2);

  //     isam_update_params.constrainedKeys = constrainedKeys;
  //     isam2.update(graph, initial, isam_update_params);
  // }

  // isam2.saveGraph(dyno::getOutputFilePath("small_tree_original.dot"),
  // dyno::DynoLikeKeyFormatter);

  // add new factors to  motion
  // the simplest dynamic graph
  gtsam::NonlinearFactorGraph new_graph;
  gtsam::Values new_values;

  gtsam::Pose3 H_2_3 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.4, 0.15, 0.1),
                                    gtsam::Point3(0, 3, 1));
  gtsam::Pose3 H_0_3 = H_0_2 * H_2_3;

  // add dynamic obs
  new_graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(2, 1), DynamicLandmarkSymbol(3, 1),
      ObjectMotionSymbol(1, 3), landmark_noise);

  new_graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(2, 2), DynamicLandmarkSymbol(3, 2),
      ObjectMotionSymbol(1, 3), landmark_noise);

  new_graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(2, 3), DynamicLandmarkSymbol(3, 3),
      ObjectMotionSymbol(1, 3), landmark_noise);

  // add pose-point constraints
  new_graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(3), DynamicLandmarkSymbol(3, 1),
      pose3.inverse() * H_0_3 * dyn_point_1_world, landmark_noise);
  new_graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(3), DynamicLandmarkSymbol(3, 2),
      pose3.inverse() * H_0_3 * dyn_point_2_world, landmark_noise);

  new_graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(3), DynamicLandmarkSymbol(3, 3),
      pose3.inverse() * H_0_3 * dyn_point_3_world, landmark_noise);

  // add static obs
  new_graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(3), StaticLandmarkSymbol(0), gtsam::Point3(4, 2, 3),
      landmark_noise);

  // add odom
  new_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
      CameraPoseSymbol(2), CameraPoseSymbol(3), pose2.between(pose3),
      pose_model));

  // add initials
  new_values.insert(DynamicLandmarkSymbol(3, 1), H_0_3 * dyn_point_1_world);
  new_values.insert(DynamicLandmarkSymbol(3, 2), H_0_3 * dyn_point_2_world);
  new_values.insert(DynamicLandmarkSymbol(3, 3), H_0_3 * dyn_point_3_world);
  new_values.insert(ObjectMotionSymbol(1, 3), gtsam::Pose3::Identity());

  new_values.insert(CameraPoseSymbol(3), pose3);

  {
    gtsam::NonlinearFactorGraph full_graph(graph);
    full_graph.add_factors(new_graph);

    gtsam::Values all_values(initial);
    all_values.insert(new_values);

    gtsam::VariableIndex affectedFactorsVarIndex(full_graph);
    gtsam::Ordering order = gtsam::Ordering::ColamdConstrainedLast(
        affectedFactorsVarIndex, full_graph.keyVector(), true);
    auto linearized = full_graph.linearize(all_values);

    bayesTree = gtsam::ISAM2JunctionTree(
                    gtsam::GaussianEliminationTree(
                        *linearized, affectedFactorsVarIndex, order))
                    .eliminate(gtsam::EliminatePreferCholesky)
                    .first;

    bayesTree->saveGraph(dyno::getOutputFilePath("elimated_tree_1.dot"),
                         dyno::DynoLikeKeyFormatter);
  }

  // gtsam::NonlinearFactorGraph all = graph;
  // all.add_factors(new_graph);
  // all.saveGraph(dyno::getOutputFilePath("small_graph_updated.dot"),
  // dyno::DynoLikeKeyFormatter);

  //  {

  //     gtsam::ISAM2UpdateParams isam_update_params;

  //     gtsam::FastMap<gtsam::Key, int> constrainedKeys;
  //     //this includes the motions, where do we want these?
  //     //maybe BETWEEN the dynnamic keys
  //     //we want motions always on the right but impossible since parent?
  //     for(const auto& [keys, value] : new_values) {
  //         constrainedKeys.insert2(keys, 1);
  //     }
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 1), 0);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 2), 0);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 3), 0);

  //     ///put current keys later in the graph
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(3, 1), 2);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(3, 2), 2);
  //     constrainedKeys.insert2(DynamicLandmarkSymbol(3, 3), 2);

  //     isam_update_params.constrainedKeys = constrainedKeys;
  //     isam2.update(new_graph, new_values, isam_update_params);
  // }

  // isam2.saveGraph(dyno::getOutputFilePath("small_tree_updated.dot"),
  // dyno::DynoLikeKeyFormatter);
}

TEST(RGBDBackendModule, writeOutSimpleGraph) {
  using namespace dyno;
  // the simplest dynamic graph
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;

  gtsam::Pose3 H_0_1 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.3, 0.2, 0.1),
                                    gtsam::Point3(0, 1, 1));
  gtsam::Pose3 H_1_2 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.5, 0.1, 0.1),
                                    gtsam::Point3(1, 1.5, 1));
  gtsam::Pose3 H_0_2 = H_0_1 * H_1_2;

  gtsam::Point3 dyn_point_1_world = gtsam::Point3(2, 1, 3);
  gtsam::Point3 dyn_point_2_world = gtsam::Point3(1, 1, 3);
  gtsam::Point3 dyn_point_3_world = gtsam::Point3(3, 0.5, 2);

  static gtsam::Point3 p0 = gtsam::Point3(1, 3, 4);
  static gtsam::Rot3 R0 =
      gtsam::Rot3::Expmap((gtsam::Vector(3) << 0.2, 0.1, 0.12).finished());
  static gtsam::Point3 p1 = gtsam::Point3(1, 2, 1);
  static gtsam::Rot3 R1 =
      gtsam::Rot3::Expmap((gtsam::Vector(3) << 0.1, 0.2, 1.570796).finished());
  static gtsam::Point3 p2 = gtsam::Point3(2, 2, 1);
  static gtsam::Rot3 R2 =
      gtsam::Rot3::Expmap((gtsam::Vector(3) << 0.2, 0.3, 3.141593).finished());
  static gtsam::Point3 p3 = gtsam::Point3(-1, 1, 0);
  static gtsam::Rot3 R3 =
      gtsam::Rot3::Expmap((gtsam::Vector(3) << 0.1, 0.4, 4.712389).finished());

  static gtsam::Pose3 pose0 = gtsam::Pose3(R0, p0);
  static gtsam::Pose3 pose1 = gtsam::Pose3(R1, p1);
  static gtsam::Pose3 pose2 = gtsam::Pose3(R2, p2);
  static gtsam::Pose3 pose3 = gtsam::Pose3(R3, p3);

  auto landmark_noise = gtsam::noiseModel::Isotropic::Sigma(3u, 10);

  // static point seen at frames 0, 1 and 2
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(0), StaticLandmarkSymbol(0), gtsam::Point3(1, 2, 3),
      landmark_noise);
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(1), StaticLandmarkSymbol(0), gtsam::Point3(2, 2, 3),
      landmark_noise);

  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(2), StaticLandmarkSymbol(0), gtsam::Point3(3, 2, 3),
      landmark_noise);

  // motion between frames 0 and 1
  // add motion factor for the 3 tracklet with tracklet id = 1, 2, 3
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(0, 1), DynamicLandmarkSymbol(1, 1),
      ObjectMotionSymbol(1, 1), landmark_noise);

  // motion between frames 1 and 2
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(1, 1), DynamicLandmarkSymbol(2, 1),
      ObjectMotionSymbol(1, 2), landmark_noise);

  // tracklet 2
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(0, 2), DynamicLandmarkSymbol(1, 2),
      ObjectMotionSymbol(1, 1), landmark_noise);
  // motion between frames 1 and 2
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(1, 2), DynamicLandmarkSymbol(2, 2),
      ObjectMotionSymbol(1, 2), landmark_noise);

  // tracklet 3
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(0, 3), DynamicLandmarkSymbol(1, 3),
      ObjectMotionSymbol(1, 1), landmark_noise);
  // motion between frames 1 and 2
  graph.emplace_shared<LandmarkMotionTernaryFactor>(
      DynamicLandmarkSymbol(1, 3), DynamicLandmarkSymbol(2, 3),
      ObjectMotionSymbol(1, 2), landmark_noise);

  // tracklet 1
  // add dynamic point obs frame 0
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(0), DynamicLandmarkSymbol(0, 1),
      pose0.inverse() * dyn_point_1_world, landmark_noise);

  // add dynamic point obs frame 1
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(1), DynamicLandmarkSymbol(1, 1),
      pose1.inverse() * H_0_1 * dyn_point_1_world, landmark_noise);

  // add dynamic point obs frame 2
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(2), DynamicLandmarkSymbol(2, 1),
      pose2.inverse() * H_0_2 * dyn_point_1_world, landmark_noise);

  // tracklet 2
  // add dynamic point obs frame 0
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(0), DynamicLandmarkSymbol(0, 2),
      pose0.inverse() * dyn_point_2_world, landmark_noise);

  // add dynamic point obs frame 1
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(1), DynamicLandmarkSymbol(1, 2),
      pose1.inverse() * H_0_1 * dyn_point_2_world, landmark_noise);

  // add dynamic point obs frame 2
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(2), DynamicLandmarkSymbol(2, 2),
      pose2.inverse() * H_0_2 * dyn_point_2_world, landmark_noise);

  // tracklet 3
  // add dynamic point obs frame 0
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(0), DynamicLandmarkSymbol(0, 3),
      pose0.inverse() * dyn_point_3_world, landmark_noise);

  // add dynamic point obs frame 1
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(1), DynamicLandmarkSymbol(1, 3),
      pose1.inverse() * H_0_1 * dyn_point_3_world, landmark_noise);

  // add dynamic point obs frame 2
  graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
      CameraPoseSymbol(2), DynamicLandmarkSymbol(2, 3),
      pose2.inverse() * H_0_2 * dyn_point_3_world, landmark_noise);

  static gtsam::SharedNoiseModel pose_model(
      gtsam::noiseModel::Isotropic::Sigma(6, 0.1));

  // add two poses
  graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
      CameraPoseSymbol(0), CameraPoseSymbol(1), pose0.between(pose1),
      pose_model));
  graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
      CameraPoseSymbol(1), CameraPoseSymbol(2), pose1.between(pose2),
      pose_model));

  // add prior on first pose
  graph.addPrior(CameraPoseSymbol(0), pose0, pose_model);

  // now add inital values

  // add static point
  initial.insert(StaticLandmarkSymbol(0), gtsam::Point3(0, 1, 1));

  // add dynamic points

  // frame 0
  initial.insert(DynamicLandmarkSymbol(0, 1), dyn_point_1_world);
  initial.insert(DynamicLandmarkSymbol(0, 2), dyn_point_2_world);
  initial.insert(DynamicLandmarkSymbol(0, 3), dyn_point_3_world);

  // frame 1
  initial.insert(DynamicLandmarkSymbol(1, 1), H_0_1 * dyn_point_1_world);
  initial.insert(DynamicLandmarkSymbol(1, 2), H_0_1 * dyn_point_2_world);
  initial.insert(DynamicLandmarkSymbol(1, 3), H_0_1 * dyn_point_3_world);

  // frame 2
  initial.insert(DynamicLandmarkSymbol(2, 1), H_0_2 * dyn_point_1_world);
  initial.insert(DynamicLandmarkSymbol(2, 2), H_0_2 * dyn_point_2_world);
  initial.insert(DynamicLandmarkSymbol(2, 3), H_0_2 * dyn_point_3_world);

  // add two motions
  initial.insert(ObjectMotionSymbol(1, 1), gtsam::Pose3::Identity());
  initial.insert(ObjectMotionSymbol(1, 2), gtsam::Pose3::Identity());

  // add three poses
  initial.insert(CameraPoseSymbol(0), pose0);
  initial.insert(CameraPoseSymbol(1), pose1);
  initial.insert(CameraPoseSymbol(2), pose2);

  dyno::NonlinearFactorGraphManager nlfgm(graph, initial);
  // nlfgm.writeDynosamGraphFile(dyno::getOutputFilePath("test_graph.g2o"));

  graph.saveGraph(dyno::getOutputFilePath("simple_dynamic_graph.dot"),
                  dyno::DynoLikeKeyFormatter);
  gtsam::GaussianFactorGraph::shared_ptr linearised_graph = nlfgm.linearize();
  gtsam::JacobianFactor jacobian_factor(*linearised_graph);
  gtsam::Matrix information = jacobian_factor.information();

  LOG(INFO) << information.rows() << " " << information.cols();

  writeMatrixWithPythonFormat(
      information,
      dyno::getOutputFilePath("simple_dynamic_graph_information.txt"));
}
