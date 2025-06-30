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

#include "dynosam/backend/RGBDBackendModule.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/optimizers/SlidingWindowOptimization.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam/backend/rgbd/impl/test_HybridFormulations.hpp"
#include "dynosam/common/Flags.hpp"
#include "dynosam/factors/LandmarkMotionPoseFactor.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/LandmarkPoseSmoothingFactor.hpp"
#include "dynosam/factors/ObjectKinematicFactor.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"

DEFINE_int32(opt_window_size, 10, "Sliding window size for optimisation");
DEFINE_int32(opt_window_overlap, 4, "Overlap for window size optimisation");

// DEFINE_bool(use_full_batch_opt, true,
//             "Use full batch optimisation if true, else sliding window");

DEFINE_int32(optimization_mode, 0,
             "0: Full-batch, 1: sliding-window, 2: incremental");

DEFINE_bool(
    use_vo_factor, true,
    "If true, use visual odometry measurement as factor from the frontend");

DEFINE_bool(
    use_identity_rot_L_for_init, false,
    "For experiments: set the initalisation point of L with identity rotation");
DEFINE_bool(corrupt_L_for_init, false,
            "For experiments: corrupt the initalisation point for L with "
            "gaussian noise");
DEFINE_double(corrupt_L_for_init_sigma, 0.2,
              "For experiments: sigma value to correupt initalisation point "
              "for L. When corrupt_L_for_init is true");

DEFINE_bool(init_LL_with_identity, false, "For experiments");
DEFINE_bool(init_H_with_identity, true, "For experiments");

// declared in BackendModule.hpp so it can be used accross multiple backends
DEFINE_string(updater_suffix, "",
              "Suffix for updater to denote specific experiments");

namespace dyno {

RGBDBackendModule::RGBDBackendModule(const BackendParams& backend_params,
                                     Camera::Ptr camera,
                                     const UpdaterType& updater_type,
                                     ImageDisplayQueue* display_queue)
    : Base(backend_params, display_queue),
      camera_(CHECK_NOTNULL(camera)),
      updater_type_(updater_type) {
  CHECK_NOTNULL(map_);

  // TODO: functioanlise and streamline with BackendModule
  noise_models_.static_point_noise = gtsam::noiseModel::Isotropic::Sigma(
      3u, backend_params.static_point_noise_sigma_);
  noise_models_.dynamic_point_noise = gtsam::noiseModel::Isotropic::Sigma(
      3u, backend_params.dynamic_point_noise_sigma_);
  // set in base!
  noise_models_.landmark_motion_noise = gtsam::noiseModel::Isotropic::Sigma(
      3u, backend_params.motion_ternary_factor_noise_sigma_);

  if (backend_params.use_robust_kernals_) {
    noise_models_.static_point_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(
            backend_params.k_huber_3d_points_),
        noise_models_.static_point_noise);

    noise_models_.dynamic_point_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(
            backend_params.k_huber_3d_points_),
        noise_models_.dynamic_point_noise);

    // TODO: not k_huber_3d_points_ not just used for 3d points
    noise_models_.landmark_motion_noise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(
            backend_params.k_huber_3d_points_),
        noise_models_.landmark_motion_noise);
  }

  CHECK_NOTNULL(noise_models_.static_point_noise);
  CHECK_NOTNULL(noise_models_.dynamic_point_noise);
  CHECK_NOTNULL(noise_models_.landmark_motion_noise);

  noise_models_.dynamic_point_noise->print("Dynamic Point Noise");
  noise_models_.landmark_motion_noise->print("Landmark motion noise");
  // CHECK(false);

  gtsam::ISAM2Params isam2_params;
  // isam2_params.factorization = gtsam::ISAM2Params::Factorization::QR;
  // isam2_params.relinearizeSkip = 2;
  isam2_params.relinearizeThreshold = 0.01;
  isam2_params.relinearizeSkip = 1;
  isam2_params.keyFormatter = DynoLikeKeyFormatter;
  // isam2_params.enablePartialRelinearizationCheck = true;
  isam2_params.evaluateNonlinearError = true;
  smoother_ = std::make_unique<gtsam::ISAM2>(isam2_params);
  fixed_lag_smoother_ =
      std::make_unique<gtsam::IncrementalFixedLagSmoother>(5, isam2_params);
  dynamic_fixed_lag_smoother_ =
      std::make_unique<gtsam::IncrementalFixedLagSmoother>(5, isam2_params);

  new_updater_ = std::move(makeUpdater());
  sliding_window_condition_ = std::make_unique<SlidingWindow>(
      FLAGS_opt_window_size, FLAGS_opt_window_overlap);

  SlidingWindowOptimization::Params sw_params;
  sw_params.window_size = FLAGS_opt_window_size;
  sw_params.overlap = FLAGS_opt_window_overlap;
  sliding_window_opt_ = std::make_unique<SlidingWindowOptimization>(sw_params);
}

RGBDBackendModule::~RGBDBackendModule() {
  LOG(INFO) << "Destructing RGBDBackendModule";

  if (base_params_.use_logger_) {
    auto backend_info = createBackendMetadata();
    new_updater_->accessorFromTheta()->postUpdateCallback(backend_info);
    new_updater_->logBackendFromMap(backend_info);
  }
}

RGBDBackendModule::SpinReturn RGBDBackendModule::boostrapSpinImpl(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  first_frame_id_ = frame_k;
  CHECK_EQ(spin_state_.frame_id, frame_k);
  LOG(INFO) << "Running backend " << frame_k;
  CHECK(!(bool)sliding_window_condition_->check(
      frame_k));  // trigger the check to update the first frame call. Bit
                  // gross!

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  addInitialStates(input, new_updater_.get(), new_values, new_factors);

  if (callback) {
    callback(new_updater_, frame_k, new_values, new_factors);
  }

  smoother_->update(new_factors, new_values);
  sliding_window_opt_->update(new_factors, new_values, frame_k);
  // smoother_->update(new_factors, new_values);
  // updateNavStateFromFormulation(frame_k, new_updater_.get());

  // TODO: sanity checks that vision states are inline with the other frame idss
  // etc

  return {State::Nominal, nullptr};
}

RGBDBackendModule::SpinReturn RGBDBackendModule::nominalSpinImpl(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  const Timestamp timestamp = input->getTimestamp();
  LOG(INFO) << "Running backend " << frame_k;
  CHECK_EQ(spin_state_.frame_id, frame_k);

  // Pose estimate from the front-end
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  addStates(input, new_updater_.get(), new_values, new_factors);
  // gtsam::Pose3 T_world_cam_k_frontend;
  // updateMap(T_world_cam_k_frontend, input);

  // gtsam::Values new_values;
  // gtsam::NonlinearFactorGraph new_factors;

  // new_updater_->addOdometry(frame_k, T_world_cam_k_frontend, new_values,
  //                           new_factors);

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  update_params.do_backtrack =
      false;  // apparently this is v important for making the results == ICRA

  {
    LOG(INFO) << "Starting updateStaticObservations";
    utils::TimingStatsCollector timer("backend.update_static_obs");
    new_updater_->updateStaticObservations(frame_k, new_values, new_factors,
                                           update_params);
  }

  gtsam::Values new_dynamic_values;
  gtsam::NonlinearFactorGraph new_dynamic_factors;

  {
    LOG(INFO) << "Starting updateDynamicObservations";
    utils::TimingStatsCollector timer("backend.update_dynamic_obs");
    new_updater_->updateDynamicObservations(frame_k, new_values, new_factors,
                                            update_params);
  }

  if (callback) {
    callback(new_updater_, frame_k, new_values, new_factors);
  }

  LOG(INFO) << "Starting any updates";

  bool incremental = false;

  // 0: Full-batch, 1: sliding-window, 2: incremental
  const int optimization_mode = FLAGS_optimization_mode;

  if (optimization_mode == 2) {
    LOG(INFO) << "Updating incremental";
    auto tic = utils::Timer::tic();

    // / This is not doing a full deep copy: it is keeping same shared_ptrs for
    // factors but copying the isam result.
    ISAM2 smoother_backup(*smoother_);

    gtsam::ISAM2Result result;
    try {
      result = smoother_->update(new_factors, new_values);
      smoother_->update();
    } catch (gtsam::IndeterminantLinearSystemException& e) {
      const gtsam::Key& var = e.nearbyVariable();
      LOG(ERROR) << "gtsam::IndeterminantLinearSystemException with variable "
                 << DynoLikeKeyFormatter(var);

      // // Add priors on all variables to fix indeterminant linear system
      const gtsam::Values values = smoother_->calculateEstimate();
      gtsam::NonlinearFactorGraph nfg;

      ApplyFunctionalSymbol afs;
      afs.cameraPose([&nfg, &values](FrameId, const gtsam::Symbol& sym) {
           const gtsam::Key& key = sym;
           gtsam::Pose3 pose = values.at<gtsam::Pose3>(key);
           gtsam::Vector6 sigmas;
           sigmas.head<3>().setConstant(0.001);  // rotation
           sigmas.tail<3>().setConstant(0.01);   // translation
           gtsam::SharedNoiseModel noise =
               gtsam::noiseModel::Diagonal::Sigmas(sigmas);
           nfg.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(key, pose,
                                                                noise);
         })
          .objectMotion([&nfg, &values](FrameId, ObjectId,
                                        const gtsam::LabeledSymbol& sym) {
            const gtsam::Key& key = sym;
            gtsam::Pose3 pose = values.at<gtsam::Pose3>(key);
            gtsam::Vector6 sigmas;
            sigmas.head<3>().setConstant(0.001);  // rotation
            sigmas.tail<3>().setConstant(0.01);   // translation
            gtsam::SharedNoiseModel noise =
                gtsam::noiseModel::Diagonal::Sigmas(sigmas);
            nfg.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(key, pose,
                                                                 noise);
          })
          .
          operator()(var);

      // afs callback did not occur
      if (nfg.size() == 0) {
        LOG(FATAL) << DynoLikeKeyFormatter(var)
                   << " not recognised in indeterminant exception handling";
      }

      gtsam::NonlinearFactorGraph new_factors_mutable;
      new_factors_mutable.push_back(new_factors.begin(), new_factors.end());
      new_factors_mutable.push_back(nfg.begin(), nfg.end());

      // Update with graph and GN optimized values
      try {
        // Update smoother
        LOG(ERROR) << "Attempting to update smoother with added prior factors";
        *smoother_ = smoother_backup;  // reset isam to backup
        result = smoother_->update(new_factors_mutable, new_values);
      } catch (...) {
        // Catch the rest of exceptions.
        // TODO: for experiments I guess keep going...?
        LOG(FATAL) << "Smoother recovery failed. Most likely, the additional "
                      "prior factors were insufficient to keep the system from "
                      "becoming indeterminant.";
        // return false;
      }

    } catch (gtsam::ValuesKeyDoesNotExist& e) {
      LOG(FATAL) << "gtsam::ValuesKeyDoesNotExist with variable "
                 << DynoLikeKeyFormatter(e.key());
    }

    auto toc = utils::Timer::toc<std::chrono::nanoseconds>(tic);
    int64_t milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc).count();

    // smoother_->update();
    LOG(INFO) << "ISAM2 result. Error before " << result.getErrorBefore()
              << " error after " << result.getErrorAfter();
    gtsam::Values optimised_values = smoother_->calculateEstimate();

    // TODO: currently for ECMR we write at every frame becuase eventually will
    // run out of memory and if we dont log
    //  at each frame we wont get any results!!!
    std::string file_name =
        new_updater_->getFullyQualifiedName() + "_isam2_timing";
    const std::string suffix = FLAGS_updater_suffix;
    if (!suffix.empty()) {
      file_name += ("_" + suffix);
    }
    file_name += ".csv";
    const std::string file_path = getOutputFilePath(file_name);

    static bool is_first = true;

    if (is_first) {
      // clear the file first
      std::ofstream clear_file(file_path, std::ios::out | std::ios::trunc);
      if (!clear_file.is_open()) {
        LOG(FATAL) << "Error clearing file: " << file_path;
      }
      clear_file.close();  // Close the stream to ensure truncation is complete
      is_first = false;
    }

    std::fstream file(file_path, std::ios::in | std::ios::out | std::ios::app);
    file.precision(15);
    file << milliseconds << "," << frame_k << "," << optimised_values.size()
         << "," << smoother_->getFactorsUnsafe().size() << "\n";
    file.close();

    new_updater_->updateTheta(optimised_values);
    // new_updater_->updateTheta(dynamic_fixed_lag_smoother_->calculateEstimate());
  } else if (optimization_mode == 0) {
    LOG(INFO) << " full batch frame " << base_params_.full_batch_frame;
    if (base_params_.full_batch_frame - 1 == (int)frame_k) {
      LOG(INFO) << " Doing full batch at frame " << frame_k;

      // graph.error(values);
      gtsam::LevenbergMarquardtParams opt_params;
      opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

      const auto theta = new_updater_->getTheta();
      const auto graph = new_updater_->getGraph();
      utils::StatsCollector(new_updater_->getFullyQualifiedName() +
                            ".full_batch_opt_num_vars_all")
          .AddSample(theta.size());

      double error_before = graph.error(theta);
      utils::TimingStatsCollector timer(new_updater_->getFullyQualifiedName() +
                                        ".full_batch_opt");

      gtsam::LevenbergMarquardtOptimizer problem(graph, theta, opt_params);
      gtsam::Values optimised_values = problem.optimize();
      double error_after = graph.error(optimised_values);

      utils::StatsCollector(new_updater_->getFullyQualifiedName() +
                            ".inner_iterations")
          .AddSample(problem.getInnerIterations());
      utils::StatsCollector(new_updater_->getFullyQualifiedName() +
                            ".iterations")
          .AddSample(problem.iterations());

      new_updater_->updateTheta(optimised_values);
      LOG(INFO) << " Error before sliding window: " << error_before
                << " error after: " << error_after;
    }
  } else if (optimization_mode == 1) {
    const auto sw_result =
        sliding_window_opt_->update(new_factors, new_values, frame_k);
    LOG(INFO) << "Sliding window result - " << sw_result.optimized;

    if (sw_result.optimized) {
      new_updater_->updateTheta(sw_result.result);
    }

  } else {
    LOG(FATAL) << "Unknown optimisation mode" << optimization_mode;
  }
  LOG(INFO) << "Done any udpates";

  auto accessor = new_updater_->accessorFromTheta();
  // update internal nav state based on the initial/optimised estimated in the
  // formulation this is also necessary to update the internal timestamp/frameid
  // variables within the VisionImuBackendModule
  updateNavStateFromFormulation(frame_k, new_updater_.get());

  // TODO: sanity checks that vision states are inline with the other frame idss
  // etc

  utils::TimingStatsCollector timer(new_updater_->getFullyQualifiedName() +
                                    ".post_update");
  BackendMetaData backend_info = createBackendMetadata();
  new_updater_->accessorFromTheta()->postUpdateCallback(
      backend_info);  // force update every time (slow! and just for testing)

  BackendOutputPacket::Ptr backend_output =
      constructOutputPacket(frame_k, timestamp);
  backend_output->involved_timestamp = input->involved_timestamps_;

  debug_info_ = DebugInfo();

  return {State::Nominal, backend_output};
}

void RGBDBackendModule::addInitialStates(
    const RGBDInstanceOutputPacket::ConstPtr& input,
    FormulationType* formulation, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors) {
  CHECK(formulation);

  const FrameId frame_k = input->getFrameId();
  const Timestamp timestamp = input->getTimestamp();
  const auto& X_k_initial = input->T_world_camera_;

  // update map
  updateMapWithMeasurements(frame_k, input, X_k_initial);

  // update formulation with initial states
  if (input->pim_) {
    LOG(INFO) << "Initialising backend with IMU states!";
    this->addInitialVisualInertialState(
        frame_k, formulation, new_values, new_factors, noise_models_,
        gtsam::NavState(X_k_initial, gtsam::Vector3(0, 0, 0)),
        gtsam::imuBias::ConstantBias{});

  } else {
    LOG(INFO) << "Initialising backend with VO only states!";
    this->addInitialVisualState(frame_k, formulation, new_values, new_factors,
                                noise_models_, X_k_initial);
  }
}
void RGBDBackendModule::addStates(
    const RGBDInstanceOutputPacket::ConstPtr& input,
    FormulationType* formulation, gtsam::Values& new_values,
    gtsam::NonlinearFactorGraph& new_factors) {
  CHECK(formulation);

  const FrameId frame_k = input->getFrameId();

  const gtsam::NavState predicted_nav_state = this->addVisualInertialStates(
      frame_k, formulation, new_values, new_factors, noise_models_,
      input->T_k_1_k_, input->pim_);

  updateMapWithMeasurements(frame_k, input, predicted_nav_state.pose());
}

void RGBDBackendModule::updateMapWithMeasurements(
    FrameId frame_id_k, const RGBDInstanceOutputPacket::ConstPtr& input,
    const gtsam::Pose3& X_k_w) {
  CHECK_EQ(frame_id_k, input->getFrameId());

  map_->updateObservations(input->collectStaticLandmarkKeypointMeasurements());
  map_->updateObservations(input->collectDynamicLandmarkKeypointMeasurements());
  map_->updateSensorPoseMeasurement(frame_id_k, Pose3Measurement(X_k_w));

  // collected motion estimates for this current frame (ie. new motions!)
  // not handling the case where the update is incremental and other motions
  // have changed but right now the backend is not designed to handle this and
  // we currently dont run the backend with smoothing (tracking) in the
  // frontend.
  const auto estimated_motions =
      input->object_motions_.toEstimateMap(frame_id_k);
  map_->updateObjectMotionMeasurements(frame_id_k, estimated_motions);
}

std::tuple<gtsam::Values, gtsam::NonlinearFactorGraph>
RGBDBackendModule::constructGraph(FrameId from_frame, FrameId to_frame,
                                  bool set_initial_camera_pose_prior,
                                  std::optional<gtsam::Values> initial_theta) {
  CHECK_LT(from_frame, to_frame);
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  auto updater = std::move(makeUpdater());

  if (initial_theta) {
    // update initial linearisation points (could be from a previous
    // optimisation)
    // TODO: currently cannot set theta becuase this will be ALL the previous
    // values and not just the ones in
    // TODO: for now dont do this as we have to handle covariance/santiy checks
    // differently as some modules expect values to be new and will check that
    // a value does not exist yet (becuase it shouldn't in that iteration, but
    // overall it may!)
    //  updater->updateTheta(*initial_theta);
  }

  UpdateObservationParams update_params;
  update_params.do_backtrack = false;
  update_params.enable_debug_info = true;

  UpdateObservationResult results(update_params);

  CHECK_GE(from_frame, map_->firstFrameId());
  CHECK_LE(to_frame, map_->lastFrameId());

  // TODO: pick how new values are going to be used as right now they get
  // appened internallty to the thera and I think this is very slow so we should
  // not do this...

  for (auto frame_id = from_frame; frame_id <= to_frame; frame_id++) {
    LOG(INFO) << "Constructing dynamic graph at frame " << frame_id
              << " in loop (" << from_frame << " -> " << to_frame << ")";

    // pose estimate from frontend
    gtsam::Pose3 T_world_camera_k;
    CHECK(map_->hasInitialSensorPose(frame_id, &T_world_camera_k));

    // if first frame
    if (frame_id == from_frame) {
      // add first pose
      updater->setInitialPose(T_world_camera_k, frame_id, new_values);

      if (set_initial_camera_pose_prior)
        updater->setInitialPosePrior(T_world_camera_k, frame_id, new_factors);
    } else {
      updater->addOdometry(frame_id, T_world_camera_k, new_values, new_factors);
      // no backtrack
      results += updater->updateDynamicObservations(frame_id, new_values,
                                                    new_factors, update_params);
    }
    results += updater->updateStaticObservations(frame_id, new_values,
                                                 new_factors, update_params);
  }

  return {new_values, new_factors};
}

bool RGBDBackendModule::buildSlidingWindowOptimisation(
    FrameId frame_k, gtsam::Values& optimised_values, double& error_before,
    double& error_after) {
  auto condition_result = sliding_window_condition_->check(frame_k);
  if (condition_result) {
    const auto start_frame = condition_result.starting_frame;
    const auto end_frame = condition_result.ending_frame;
    LOG(INFO) << "Running dynamic slam window on between frames " << start_frame
              << " - " << end_frame;

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    {
      utils::TimingStatsCollector timer(new_updater_->getFullyQualifiedName() +
                                        ".sliding_window_construction");
      std::tie(values, graph) = constructGraph(start_frame, end_frame, true,
                                               new_updater_->getTheta());
      LOG(INFO) << " Finished graph construction";
    }

    error_before = graph.error(values);
    gtsam::LevenbergMarquardtParams opt_params;
    if (VLOG_IS_ON(20))
      opt_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;

    utils::TimingStatsCollector timer(new_updater_->getFullyQualifiedName() +
                                      ".sliding_window_optimise");
    utils::StatsCollector(new_updater_->getFullyQualifiedName() +
                          ".sliding_window_optimise_num_vars_all")
        .AddSample(values.size());
    try {
      optimised_values =
          gtsam::LevenbergMarquardtOptimizer(graph, values, opt_params)
              .optimize();
      LOG(INFO) << "Finished op!";
    } catch (const gtsam::ValuesKeyDoesNotExist& e) {
      LOG(FATAL)
          << "gtsam::ValuesKeyDoesNotExist: key does not exist in the values"
          << DynoLikeKeyFormatter(e.key());
    }
    error_after = graph.error(optimised_values);

    return true;
  }
  return false;
}

Formulation<RGBDBackendModule::RGBDMap>::UniquePtr
RGBDBackendModule::makeUpdater() {
  FormulationParams formulation_params;
  // TODO: why are we copying params over???
  formulation_params.min_static_observations = base_params_.min_static_obs_;
  // formulation_params.min_dynamic_observations =
  // base_params_.min_dynamic_obs_;
  formulation_params.min_dynamic_observations = 2u;
  formulation_params.use_smoothing_factor = base_params_.use_smoothing_factor;

  FormulationHooks hooks = createFormulationHooks();

  if (updater_type_ == RGBDFormulationType::WCME) {
    LOG(INFO) << "Using WCME";
    return std::make_unique<WorldMotionFormulation>(
        formulation_params, getMap(), noise_models_, hooks);

  } else if (updater_type_ == RGBDFormulationType::WCPE) {
    LOG(INFO) << "Using WCPE";
    return std::make_unique<WorldPoseFormulation>(formulation_params, getMap(),
                                                  noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::HYBRID) {
    LOG(INFO) << "Using HYBRID";
    return std::make_unique<HybridFormulation>(formulation_params, getMap(),
                                               noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::TESTING_HYBRID_SD) {
    LOG(INFO) << "Using Hybrid Structureless Decoupled. Warning this is a "
                 "testing only formulation!";
    return std::make_unique<test_hybrid::StructurelessDecoupledFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::TESTING_HYBRID_D) {
    LOG(INFO) << "Using Hybrid Decoupled. Warning this is a testing only "
                 "formulation!";
    return std::make_unique<test_hybrid::DecoupledFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::TESTING_HYBRID_S) {
    LOG(INFO) << "Using Hybrid Structurless. Warning this is a testing only "
                 "formulation!";
    return std::make_unique<test_hybrid::StructurlessFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else if (updater_type_ == RGBDFormulationType::TESTING_HYBRID_SMF) {
    LOG(INFO) << "Using Hybrid Smart Motion Factor. Warning this is a testing "
                 "only formulation!";
    return std::make_unique<test_hybrid::SmartStructurlessFormulation>(
        formulation_params, getMap(), noise_models_, hooks);
  } else {
    CHECK(false) << "Not implemented";
  }
}

BackendMetaData RGBDBackendModule::createBackendMetadata() const {
  // TODO: cache?
  BackendMetaData backend_info;
  backend_info.logging_suffix = FLAGS_updater_suffix;
  backend_info.backend_params = &base_params_;
  return backend_info;
}

FormulationHooks RGBDBackendModule::createFormulationHooks() const {
  // TODO: cache?
  FormulationHooks hooks;

  hooks.ground_truth_packets_request =
      [&]() -> std::optional<GroundTruthPacketMap> {
    return shared_module_info.getGroundTruthPackets();
  };

  return hooks;
}

BackendOutputPacket::Ptr RGBDBackendModule::constructOutputPacket(
    FrameId frame_k, Timestamp timestamp) const {
  CHECK_NOTNULL(new_updater_);
  return RGBDBackendModule::constructOutputPacket(new_updater_, frame_k,
                                                  timestamp);
}

BackendOutputPacket::Ptr RGBDBackendModule::constructOutputPacket(
    const Formulation<RGBDMap>::UniquePtr& formulation, FrameId frame_k,
    Timestamp timestamp) {
  auto accessor = formulation->accessorFromTheta();

  auto backend_output = std::make_shared<BackendOutputPacket>();
  backend_output->timestamp = timestamp;
  backend_output->frame_id = frame_k;
  backend_output->T_world_camera = accessor->getSensorPose(frame_k).get();
  backend_output->static_landmarks = accessor->getFullStaticMap();
  // backend_output->optimized_object_motions =
  //     accessor->getObjectMotions(frame_k);
  backend_output->dynamic_landmarks =
      accessor->getDynamicLandmarkEstimates(frame_k);
  auto map = formulation->map();
  for (FrameId frame_id : map->getFrameIds()) {
    backend_output->optimized_camera_poses.push_back(
        accessor->getSensorPose(frame_id).get());
  }

  // fill temporal map information
  for (ObjectId object_id : map->getObjectIds()) {
    const auto& object_node = map->getObject(object_id);
    CHECK_NOTNULL(object_node);

    // TODO: based on measurements not on estimation so check that we have
    // landmarks for this object first?
    TemporalObjectMetaData temporal_object_info;
    temporal_object_info.object_id = object_id;
    temporal_object_info.first_seen = object_node->getFirstSeenFrame();
    temporal_object_info.last_seen = object_node->getLastSeenFrame();
    backend_output->temporal_object_data.push_back(temporal_object_info);
  }

  backend_output->optimized_object_motions = accessor->getObjectMotions();
  backend_output->optimized_object_poses = accessor->getObjectPoses();
  return backend_output;
}

}  // namespace dyno
