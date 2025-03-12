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

#include "dynosam/backend/ParallelRGBDBackendModule.hpp"

#include <glog/logging.h>
#include <gtsam/nonlinear/Marginals.h>
#include <tbb/tbb.h>

namespace dyno {

ParallelRGBDBackendModule::ParallelRGBDBackendModule(
    const BackendParams& backend_params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : Base(backend_params, display_queue), camera_(CHECK_NOTNULL(camera)) {
  LOG(INFO) << "Creating ParallelRGBDBackendModule";

  // TODO: set isam params
  dynamic_isam2_params_.keyFormatter = DynoLikeKeyFormatter;
  dynamic_isam2_params_.evaluateNonlinearError = true;
  dynamic_isam2_params_.enableDetailedResults = true;
  dynamic_isam2_params_.relinearizeThreshold = 0.01;
  dynamic_isam2_params_.relinearizeSkip = 1;

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

ParallelRGBDBackendModule::~ParallelRGBDBackendModule() {
  LOG(INFO) << "Desctructing ParallelRGBDBackendModule";

  if (base_params_.use_logger_) {
    logBackendFromEstimators();

    const std::string file_path =
        getOutputFilePath("parallel_isam2_results.bson");
    JsonConverter::WriteOutJson(result_map_, file_path,
                                JsonConverter::Format::BSON);
  }
}

// implementation taken from ObjectMotionSolverSAM
// TODO: update api
ParallelRGBDBackendModule::SpinReturn
ParallelRGBDBackendModule::boostrapSpinImpl(
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

  // lazy update (not parallel)
  for (const PerObjectUpdate& update : dynamic_updates) {
    const auto object_id = update.object_id;
    ParallelObjectISAM::Ptr estimator = getEstimator(object_id);
    const auto result = estimator->getResult();

    if (!result.was_smoother_ok) {
      LOG(WARNING) << "Could not record results for object smoother j="
                   << object_id << " as smoother was not ok";
      continue;
    }

    CHECK_EQ(result.frame_id, frame_k);
    result_map_.insert22(object_id, result.frame_id, result);
  }

  new_objects_estimators_.clear();
  return {State::Nominal, nullptr};
}

ParallelRGBDBackendModule::SpinReturn
ParallelRGBDBackendModule::nominalSpinImpl(
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

  for (const PerObjectUpdate& update : dynamic_updates) {
    const auto object_id = update.object_id;
    ParallelObjectISAM::Ptr estimator = getEstimator(object_id);
    const auto result = estimator->getResult();

    if (!result.was_smoother_ok) {
      LOG(WARNING) << "Could not record results for object smoother j="
                   << object_id << " as smoother was not ok";
      continue;
    }

    CHECK_EQ(result.frame_id, frame_k);
    result_map_.insert22(object_id, result.frame_id, result);
  }

  new_objects_estimators_.clear();
  return {State::Nominal, backend_output};
}

Pose3Measurement ParallelRGBDBackendModule::bootstrapUpdateStaticEstimator(
    RGBDInstanceOutputPacket::ConstPtr input) {
  utils::TimingStatsCollector timer("parallel_object_sam.static_estimator");

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

  {
    utils::TimingStatsCollector timer(
        "parallel_object_sam.static_estimator.update");
    static_estimator_.update(new_factors, new_values);
  }

  gtsam::SharedGaussian gaussian_pose_prior =
      boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
          initial_pose_prior);
  CHECK(gaussian_pose_prior)
      << "initial pose prior must be a Gaussian noise model!";

  return Pose3Measurement(T_W_k_frontend, gaussian_pose_prior);
}

Pose3Measurement ParallelRGBDBackendModule::nominalUpdateStaticEstimator(
    RGBDInstanceOutputPacket::ConstPtr input) {
  utils::TimingStatsCollector timer("parallel_object_sam.static_estimator");

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

  gtsam::ISAM2Result result;
  {
    utils::TimingStatsCollector timer(
        "parallel_object_sam.static_estimator.update");
    result = static_estimator_.update(new_factors, new_values);
  }

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
    utils::TimingStatsCollector timer(
        "parallel_object_sam.camera_pose_cov_calc");
    gtsam::Marginals marginals(static_estimator_.getFactorsUnsafe(),
                               optimised_values,
                               gtsam::Marginals::Factorization::CHOLESKY);
    T_w_k_cov = marginals.marginalCovariance(T_w_k_opt_query.key());
  }

  return Pose3Measurement(T_w_k_opt_query.get(), T_w_k_cov);
}

std::vector<ParallelRGBDBackendModule::PerObjectUpdate>
ParallelRGBDBackendModule::collectMeasurements(
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
  std::stringstream ss;
  ss << "Objects with updates: ";
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
    ss << object_id << " ";
  }

  // log inf
  LOG(INFO) << ss.str();

  return object_updates;
}

ParallelObjectISAM::Ptr ParallelRGBDBackendModule::getEstimator(
    ObjectId object_id, bool* is_object_new) {
  std::lock_guard<std::mutex> lock(mutex_);

  bool is_new = false;
  // make new estimator if needed
  if (!sam_estimators_.exists(object_id)) {
    LOG(INFO) << "Making new ParallelObjectISAM for object " << object_id;

    FormulationHooks hooks;
    hooks.ground_truth_packets_request =
        [&]() -> std::optional<GroundTruthPacketMap> {
      return this->getGroundTruthPackets();
    };

    ParallelObjectISAM::Params params;
    params.num_optimzie = 4u;
    params.isam = dynamic_isam2_params_;
    // // make this prior not SO small
    NoiseModels noise_models = NoiseModels::fromBackendParams(base_params_);
    sam_estimators_.insert2(
        object_id, std::make_shared<ParallelObjectISAM>(params, object_id,
                                                        noise_models, hooks));

    is_new = true;
  }
  if (is_object_new) *is_object_new = is_new;

  if (is_new) new_objects_estimators_.push_back(object_id);

  return sam_estimators_.at(object_id);
}

bool ParallelRGBDBackendModule::implSolvePerObject(
    const PerObjectUpdate& object_update) {
  const auto object_id = object_update.object_id;
  const auto frame_id_k = object_update.frame_id;
  const auto& measurements = object_update.measurements;

  bool is_object_new;
  ParallelObjectISAM::Ptr estimator = getEstimator(object_id, &is_object_new);

  // if object is new, dont update the smoother
  bool should_update_smoother = !is_object_new;

  CHECK_NOTNULL(estimator);
  auto map = estimator->map();

  const Pose3Measurement& X_k_measurement = object_update.X_k_measurement;
  const Motion3ReferenceFrame& H_k = object_update.H_k_measurement;

  // hack for now - if the object is new only update its map
  // this will create nodes in the Map but not update the estimator
  // only update the estimator otherwise!!

  // if new or last object update was more than 1 frame ago
  // this may be wrong if the smoother was not updated correctly...
  FrameId last_update_frame = estimator->getResult().frame_id;

  // Should this be last_update_frame == frame_id_k - 1u
  // if its more than that.... unsure
  if (!is_object_new && (frame_id_k > 0) &&
      (last_update_frame < (frame_id_k - 1u))) {
    VLOG(5) << "Only update k=" << frame_id_k << " j= " << object_id
            << " as object is not new but has reappeared. Previous update was "
            << last_update_frame;
    should_update_smoother = false;
  }

  // only update acts like the boostrap mode of the BackendModule
  // we dont want to update the smoother or add dynamic measurements yet
  // since we need at least two valid frames
  //  if (only_update) {
  //    map->updateObservations(measurements);
  //    map->updateSensorPoseMeasurement(frame_id_k, X_k_measurement);

  //   MotionEstimateMap motion_estimate;
  //   motion_estimate.insert({object_id, H_k});
  //   map->updateObjectMotionMeasurements(frame_id_k, motion_estimate);
  // } else {
  //   estimator->update(frame_id_k, measurements, X_k_measurement, H_k);
  // }

  estimator->update(frame_id_k, measurements, X_k_measurement, H_k,
                    should_update_smoother);

  // if (only_update) {

  // } else {
  //   estimator->update(frame_id_k, measurements, X_k_measurement, H_k);
  // }
}

BackendOutputPacket::Ptr ParallelRGBDBackendModule::constructOutputPacket(
    FrameId frame_k, Timestamp timestamp) const {
  auto backend_output = std::make_shared<BackendOutputPacket>();
  backend_output->timestamp = timestamp;
  backend_output->frame_id = frame_k;

  for (const auto& [object_id, estimator] : sam_estimators_) {
    // slow lookup for now!!
    // dont construct output if object is new
    // same logic as implSolvePerObject where we dont update the estimator on
    // the first pass, we just update the initial measurements etc...
    if (std::find(new_objects_estimators_.begin(),
                  new_objects_estimators_.end(),
                  object_id) != new_objects_estimators_.end()) {
      continue;
    }

    const ObjectPoseMap per_object_poses = estimator->getObjectPoses();
    const ObjectMotionMap per_object_motions =
        estimator->getFrame2FrameMotions();

    backend_output->optimized_object_motions += per_object_motions;
    backend_output->optimized_object_poses += per_object_poses;

    // since we only show the current object map, get the landmarks only at the
    // current frame this should return an empty vector if the object does not
    // exist at the current frame
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

void ParallelRGBDBackendModule::logBackendFromEstimators() {
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
