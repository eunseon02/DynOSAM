/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/LooselyDistributedRGBDBackendModule.hpp"

#include <glog/logging.h>
#include <gtsam/nonlinear/Marginals.h>
#include <tbb/tbb.h>

namespace dyno {

LooselyDistributedRGBDBackendModule::LooselyDistributedRGBDBackendModule(
    const BackendParams& backend_params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : Base(backend_params, display_queue), camera_(CHECK_NOTNULL(camera)) {
  LOG(INFO) << "Creating LooselyDistributedRGBDBackendModule";

  // TODO: set isam params
  dynamic_isam2_params_.keyFormatter = DynoLikeKeyFormatter;
  dynamic_isam2_params_.evaluateNonlinearError = true;

  static_isam2_params_.keyFormatter = DynoLikeKeyFormatter;
  static_isam2_params_.evaluateNonlinearError = true;

  static_estimator_ = gtsam::ISAM2(static_isam2_params_);

  FormulationHooks hooks;
  hooks.ground_truth_packets_request =
      [&]() -> std::optional<GroundTruthPacketMap> {
    return this->getGroundTruthPackets();
  };

  FormulationParams formulation_params;
  formulation_params.min_static_observations = base_params_.min_static_obs_;
  formulation_params.min_dynamic_observations = base_params_.min_dynamic_obs_;
  formulation_params.use_smoothing_factor = base_params_.use_smoothing_factor;
  formulation_params.suffix = "static";

  static_formulation_ = std::make_unique<ObjectCentricFormulation>(
      formulation_params, RGBDMap::create(), noise_models_, hooks);
}

LooselyDistributedRGBDBackendModule::~LooselyDistributedRGBDBackendModule() {
  LOG(INFO) << "Desctructing LooselyDistributedRGBDBackendModule";

  if (base_params_.use_logger_) {
    logBackendFromEstimators();
  }
}

// implementation taken from ObjectMotionSolverSAM
// TODO: update api
LooselyDistributedRGBDBackendModule::SpinReturn
LooselyDistributedRGBDBackendModule::boostrapSpinImpl(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  const Timestamp timestamp = input->getTimestamp();
  // TODO: sovle smoother
  //  non-sequentially?
  Pose3Measurement optimized_camera_pose =
      bootstrapUpdateStaticEstimator(input);

  // collect measurements into dynamic and static
  std::vector<PerObjectUpdate> dynamic_updates =
      collectMeasurements(input, optimized_camera_pose);

  tbb::parallel_for_each(
      dynamic_updates.begin(), dynamic_updates.end(),
      [&](const PerObjectUpdate& update) { this->implSolvePerObject(update); });

  return {State::Nominal, nullptr};
}

LooselyDistributedRGBDBackendModule::SpinReturn
LooselyDistributedRGBDBackendModule::nominalSpinImpl(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  const Timestamp timestamp = input->getTimestamp();
  // TODO: sovle smoother
  //  non-sequentially?
  Pose3Measurement optimized_camera_pose = nominalUpdateStaticEstimator(input);
  // collect measurements into dynamic and static
  std::vector<PerObjectUpdate> dynamic_updates =
      collectMeasurements(input, optimized_camera_pose);
  tbb::parallel_for_each(
      dynamic_updates.begin(), dynamic_updates.end(),
      [&](const PerObjectUpdate& update) { this->implSolvePerObject(update); });
  // get estimator
  // should add previous measurements
  // updaet map
  // update estimator
  auto backend_output = constructOutputPacket(frame_k, timestamp);
  backend_output->involved_timestamp = input->involved_timestamps_;

  return {State::Nominal, backend_output};
}

Pose3Measurement
LooselyDistributedRGBDBackendModule::bootstrapUpdateStaticEstimator(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  auto map = static_formulation_->map();

  map->updateObservations(input->collectStaticLandmarkKeypointMeasurements());

  Pose3Measurement T_W_k_frontend(input->T_world_camera_);
  map->updateSensorPoseMeasurement(frame_k, T_W_k_frontend);

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  static_formulation_->addSensorPoseValue(T_W_k_frontend, frame_k, new_values);

  auto initial_pose_prior =
      static_formulation_->noiseModels().initial_pose_prior;
  static_formulation_->addSensorPosePriorFactor(
      T_W_k_frontend, initial_pose_prior, frame_k, new_factors);

  static_estimator_.update(new_factors, new_values);

  gtsam::SharedGaussian gaussian_pose_prior =
      boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
          initial_pose_prior);
  CHECK(gaussian_pose_prior)
      << "initial pose prior must be a Gaussian noise model!";

  return Pose3Measurement(T_W_k_frontend, gaussian_pose_prior);
}

Pose3Measurement
LooselyDistributedRGBDBackendModule::nominalUpdateStaticEstimator(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  auto map = static_formulation_->map();

  map->updateObservations(input->collectStaticLandmarkKeypointMeasurements());

  Pose3Measurement T_W_k_frontend(input->T_world_camera_);
  // we dont have an uncertainty from the frontend
  map->updateSensorPoseMeasurement(frame_k, T_W_k_frontend);

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  UpdateObservationParams update_params;
  update_params.enable_debug_info = true;
  update_params.do_backtrack = true;

  static_formulation_->addOdometry(frame_k, T_W_k_frontend, new_values,
                                   new_factors);
  static_formulation_->updateStaticObservations(frame_k, new_values,
                                                new_factors, update_params);

  gtsam::ISAM2Result result = static_estimator_.update(new_factors, new_values);

  VLOG(5) << "Finished LC Static update k" << frame_k
          << "  error before: " << result.errorBefore.value_or(NaN)
          << " error after: " << result.errorAfter.value_or(NaN);

  gtsam::Values optimised_values = static_estimator_.calculateBestEstimate();
  static_formulation_->updateTheta(optimised_values);

  auto accessor = static_formulation_->accessorFromTheta();
  StateQuery<gtsam::Pose3> T_w_k_opt_query = accessor->getSensorPose(frame_k);

  CHECK(T_w_k_opt_query);

  gtsam::Matrix66 T_w_k_cov;
  {
    utils::TimingStatsCollector timer("lc_rgbd_backend.camera_pose_cov_calc");
    gtsam::Marginals marginals(static_estimator_.getFactorsUnsafe(),
                               optimised_values,
                               gtsam::Marginals::Factorization::CHOLESKY);
    T_w_k_cov = marginals.marginalCovariance(T_w_k_opt_query.key());
  }

  return Pose3Measurement(T_w_k_opt_query.get(), T_w_k_cov);
}

std::vector<LooselyDistributedRGBDBackendModule::PerObjectUpdate>
LooselyDistributedRGBDBackendModule::collectMeasurements(
    RGBDInstanceOutputPacket::ConstPtr input,
    const Pose3Measurement& X_k_measurement) const {
  GenericTrackedStatusVector<LandmarkKeypointStatus> all_dynamic_measurements =
      input->collectDynamicLandmarkKeypointMeasurements();

  gtsam::FastMap<ObjectId, GenericTrackedStatusVector<LandmarkKeypointStatus>>
      measurements_by_object;
  for (const auto& measurement : all_dynamic_measurements) {
    const auto& object_id = measurement.objectId();
    if (!measurements_by_object.exists(object_id)) {
      measurements_by_object.insert2(
          object_id, GenericTrackedStatusVector<LandmarkKeypointStatus>{});
    }
    measurements_by_object.at(object_id).push_back(measurement);
  }

  const auto frame_id_k = input->getFrameId();
  const auto& estimated_motions =
      input->object_motions_.toEstimateMap(frame_id_k);

  std::vector<PerObjectUpdate> object_updates;
  for (const auto& [object_id, collected_measurements] :
       measurements_by_object) {
    PerObjectUpdate object_update;
    object_update.frame_id = frame_id_k;
    object_update.object_id = object_id;
    object_update.measurements = collected_measurements;
    object_update.X_k_measurement = X_k_measurement;

    CHECK(estimated_motions.exists(object_id));
    object_update.H_k_measurement = estimated_motions.at(object_id);

    object_updates.push_back(object_update);
  }

  return object_updates;
}

LooselyCoupledObjectSAM::Ptr LooselyDistributedRGBDBackendModule::getEstimator(
    ObjectId object_id, bool* is_object_new) {
  std::lock_guard<std::mutex> lock(mutex_);

  // make new estimator if needed
  if (!sam_estimators_.exists(object_id)) {
    LOG(INFO) << "Making new DecoupledObjectSAM for object " << object_id;

    FormulationHooks hooks;
    hooks.ground_truth_packets_request =
        [&]() -> std::optional<GroundTruthPacketMap> {
      return this->getGroundTruthPackets();
    };

    LooselyCoupledObjectSAM::Params params;
    params.isam = dynamic_isam2_params_;
    // // make this prior not SO small
    NoiseModels noise_models = NoiseModels::fromBackendParams(base_params_);
    sam_estimators_.insert2(object_id,
                            std::make_shared<LooselyCoupledObjectSAM>(
                                params, object_id, noise_models, hooks));

    if (is_object_new) *is_object_new = true;
  } else {
    if (is_object_new) *is_object_new = false;
  }

  return sam_estimators_.at(object_id);
}

bool LooselyDistributedRGBDBackendModule::implSolvePerObject(
    const PerObjectUpdate& object_update) {
  const auto object_id = object_update.object_id;
  const auto frame_id_k = object_update.frame_id;
  const auto& measurements = object_update.measurements;

  bool is_object_new;
  LooselyCoupledObjectSAM::Ptr estimator =
      getEstimator(object_id, &is_object_new);

  CHECK_NOTNULL(estimator);
  auto map = estimator->map();

  const Pose3Measurement& X_k_measurement = object_update.X_k_measurement;
  const Motion3ReferenceFrame& H_k = object_update.H_k_measurement;

  // hack for now - if the object is new only update its map
  // this will create nodes in the Map but not update the estimator
  // only update the estimator otherwise!!
  if (is_object_new) {
    map->updateObservations(measurements);
    map->updateSensorPoseMeasurement(frame_id_k, X_k_measurement);
  } else {
    estimator->update(frame_id_k, measurements, X_k_measurement, H_k);
  }
}

BackendOutputPacket::Ptr
LooselyDistributedRGBDBackendModule::constructOutputPacket(
    FrameId frame_k, Timestamp timestamp) const {
  auto backend_output = std::make_shared<BackendOutputPacket>();
  backend_output->timestamp = timestamp;
  backend_output->frame_id = frame_k;

  for (const auto& [object_id, estimator] : sam_estimators_) {
    const ObjectPoseMap per_object_poses = estimator->getObjectPoses();
    const ObjectMotionMap per_object_motions =
        estimator->getFrame2FrameMotions();

    backend_output->optimized_object_motions += per_object_motions;
    backend_output->optimized_object_poses += per_object_poses;
    backend_output->dynamic_landmarks +=
        estimator->getDynamicLandmarks(frame_k);
  }

  auto accessor = static_formulation_->accessorFromTheta();
  auto map = static_formulation_->map();

  backend_output->static_landmarks = accessor->getFullStaticMap();
  backend_output->T_world_camera = accessor->getSensorPose(frame_k).get();
  for (FrameId frame_id : map->getFrameIds()) {
    backend_output->optimized_camera_poses.push_back(
        accessor->getSensorPose(frame_id).get());
  }

  return backend_output;
}

void LooselyDistributedRGBDBackendModule::logBackendFromEstimators() {
  // TODO: name + suffix
  const std::string name = "parallel_object_centric";

  BackendLogger::UniquePtr logger = std::make_unique<BackendLogger>(name);

  Timestamp timestamp_k = this->spin_state_.timestamp;
  FrameId frame_id_k = this->spin_state_.frame_id;

  LOG(INFO) << "Logging Parallel RGBD backend at frame " << frame_id_k;

  BackendOutputPacket::Ptr output =
      constructOutputPacket(frame_id_k, timestamp_k);

  const auto& gt_packets = this->getGroundTruthPackets();

  logger->logObjectMotion(output->optimized_object_motions, gt_packets);
  logger->logObjectPose(output->optimized_object_poses, gt_packets);

  // duplicated code from constructOutputPacket but we need the frame ids!!!
  auto accessor = static_formulation_->accessorFromTheta();
  auto map = static_formulation_->map();
  for (FrameId frame_id : map->getFrameIds()) {
    StateQuery<gtsam::Pose3> X_k_query = accessor->getSensorPose(frame_id);
    logger->logCameraPose(frame_id, X_k_query.get(), gt_packets);
  }

  // TODO: not logging points!!!

  logger.reset();
}

}  // namespace dyno
