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
}

LooselyDistributedRGBDBackendModule::~LooselyDistributedRGBDBackendModule() {
  LOG(INFO) << "Desctructing LooselyDistributedRGBDBackendModule";
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
  SensorPoseMeasurement optimized_camera_pose(input->T_world_camera_);
  // collect measurements into dynamic and static
  std::vector<PerObjectUpdate> dynamic_updates =
      collectMeasurements(input, optimized_camera_pose);
  // //get estimator

  tbb::parallel_for_each(
      dynamic_updates.begin(), dynamic_updates.end(),
      [&](const PerObjectUpdate& update) { this->implSolvePerObject(update); });
  // updaet map
  // update estimator
  //  auto backend_output = constructOutputPacket(frame_k, timestamp);
  //  backend_output->involved_timestamp = input->involved_timestamps_;

  return {State::Nominal, nullptr};
}

LooselyDistributedRGBDBackendModule::SpinReturn
LooselyDistributedRGBDBackendModule::nominalSpinImpl(
    RGBDInstanceOutputPacket::ConstPtr input) {
  const FrameId frame_k = input->getFrameId();
  const Timestamp timestamp = input->getTimestamp();
  // TODO: sovle smoother
  //  non-sequentially?
  SensorPoseMeasurement optimized_camera_pose(input->T_world_camera_);
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

std::vector<LooselyDistributedRGBDBackendModule::PerObjectUpdate>
LooselyDistributedRGBDBackendModule::collectMeasurements(
    RGBDInstanceOutputPacket::ConstPtr input,
    const SensorPoseMeasurement& X_k_measurement) const {
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
    // hooks.ground_truth_packets_request =
    //     geometric_solver_->objectMotionParams().ground_truth_packets_request;

    LooselyCoupledObjectSAM::Params params;
    params.isam = dynamic_isam2_params_;
    // // make this prior not SO small
    NoiseModels noise_models = NoiseModels::fromBackendParams(base_params_);
    noise_models.initial_pose_prior =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.001);
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

  const gtsam::Pose3& X_W_k = object_update.X_k_measurement;
  const auto H_k = object_update.H_k_measurement;

  // hack for now - if the object is new only update its map
  // this will create nodes in the Map but not update the estimator
  // only update the estimator otherwise!!
  if (is_object_new) {
    map->updateObservations(measurements);
    map->updateSensorPoseMeasurement(frame_id_k, X_W_k);
  } else {
    estimator->update(frame_id_k, measurements, X_W_k, H_k);
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
    backend_output->static_landmarks += estimator->getDynamicLandmarks(frame_k);
  }

  return backend_output;
}

}  // namespace dyno
