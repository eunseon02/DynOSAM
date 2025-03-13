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

#include "dynosam/backend/rgbd/impl/ParallelObjectISAM.hpp"

#include "dynosam/utils/TimingStats.hpp"

namespace dyno {

ParallelObjectISAM::ParallelObjectISAM(
    const Params& params, ObjectId object_id, const NoiseModels& noise_models,
    const FormulationHooks& formulation_hooks)
    : params_(params),
      object_id_(object_id),
      map_(Map::create()),
      expected_style_(MotionRepresentationStyle::F2F) {
  smoother_ = std::make_shared<gtsam::ISAM2>(params_.isam);

  FormulationParams formulation_params;
  formulation_params.suffix = "object_" + std::to_string(object_id);
  // HACK for now so that we get object motions at every frame!!!?
  formulation_params.min_dynamic_observations = 3u;

  decoupled_formulation_ = std::make_shared<ObjectCentricFormulation>(
      formulation_params, map_, noise_models, formulation_hooks);
  accessor_ = decoupled_formulation_->accessorFromTheta();
}

StateQuery<Motion3ReferenceFrame> ParallelObjectISAM::getFrame2FrameMotion(
    FrameId frame_id) const {
  // this is in the form of our accessor
  StateQuery<Motion3ReferenceFrame> H_W_km1_k =
      accessor_->getObjectMotionReferenceFrame(frame_id, object_id_);
  if (!H_W_km1_k) return H_W_km1_k;
  // CHECK(H_W_km1_k) << "Failed to get object motion at j=" << object_id_
  //                  << " k=" << frame_id;
  CHECK(H_W_km1_k->style() == MotionRepresentationStyle::F2F);
  CHECK(H_W_km1_k->origin() == ReferenceFrame::GLOBAL);
  CHECK(H_W_km1_k->to() == frame_id);

  return H_W_km1_k;
}

Motion3ReferenceFrame ParallelObjectISAM::getKeyFramedMotion(
    FrameId frame_id) const {
  // assumes the motion actually exists??!!!
  Motion3ReferenceFrame H_W_s0_k =
      decoupled_formulation_->getEstimatedMotion(object_id_, frame_id);
  CHECK(H_W_s0_k.style() == MotionRepresentationStyle::KF);
  CHECK(H_W_s0_k.origin() == ReferenceFrame::GLOBAL);
  CHECK(H_W_s0_k.to() == frame_id);
  return H_W_s0_k;
}

ObjectMotionMap ParallelObjectISAM::getFrame2FrameMotions() const {
  ObjectMotionMap motions;
  for (const FrameId& frame_id : map()->getFrameIds()) {
    auto f2f_motion = this->getFrame2FrameMotion(frame_id);
    if (f2f_motion) motions.insert22(object_id_, frame_id, f2f_motion.get());
  }
  return motions;
}

ObjectMotionMap ParallelObjectISAM::getKeyFramedMotions() const {
  ObjectMotionMap motions;
  for (const FrameId& frame_id : map()->getFrameIds()) {
    motions.insert22(object_id_, frame_id, this->getKeyFramedMotion(frame_id));
  }
  return motions;
}

StatusLandmarkVector ParallelObjectISAM::getDynamicLandmarks(
    FrameId frame_id) const {
  return accessor_->getDynamicLandmarkEstimates(frame_id, object_id_);
}

void ParallelObjectISAM::updateFormulation(
    FrameId frame_k, const Pose3Measurement& X_W_k,
    gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values) {
  // no need to update and static things ;)
  // TODO: hack to ensure we add the pose to the internal values on the first
  // run for the previous frame!!
  if (frame_k > 0) {
    auto frame_km1 = frame_k - 1u;

    if (!accessor_->exists(CameraPoseSymbol(frame_km1))) {
      LOG(INFO) << "Previous camera pose does not exist!! k=" << frame_km1
                << "j=" << object_id_;
      Pose3Measurement T_W_cam_km1;
      CHECK(this->map()->hasInitialSensorPose(frame_km1, &T_W_cam_km1));

      const gtsam::Pose3& initial_X_W_km1 = T_W_cam_km1.measurement();
      CHECK(T_W_cam_km1.hasModel());
      const gtsam::SharedGaussian& uncertainty_X_W_km1 = T_W_cam_km1.model();

      decoupled_formulation_->addSensorPoseValue(initial_X_W_km1, frame_km1,
                                                 new_values);
      decoupled_formulation_->addSensorPosePriorFactor(
          initial_X_W_km1, uncertainty_X_W_km1, frame_km1, new_factors);
    }
  }

  const gtsam::Pose3& initial_X_W_k = X_W_k.measurement();
  CHECK(X_W_k.hasModel());
  const gtsam::SharedGaussian& uncertainty_X_W_k = X_W_k.model();

  decoupled_formulation_->addSensorPoseValue(initial_X_W_k, frame_k,
                                             new_values);
  decoupled_formulation_->addSensorPosePriorFactor(
      initial_X_W_k, uncertainty_X_W_k, frame_k, new_factors);

  UpdateObservationParams update_params;
  update_params.do_backtrack = false;
  update_params.enable_debug_info = true;
  VLOG(10) << "ParallelObjectISAM: Starting formulation update k=" << frame_k
           << " j= " << object_id_;
  decoupled_formulation_->updateDynamicObservations(frame_k, new_values,
                                                    new_factors, update_params);
  // TODO: use update result - if the object is not updated, should we remove
  // the frame node at k-1...?
}

bool ParallelObjectISAM::updateSmoother(FrameId frame_k,
                                        const Pose3Measurement& X_W_k) {
  // only clear result when we update the smoother...
  // result_ = Result();
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  updateFormulation(frame_k, X_W_k, new_factors, new_values);

  // do first optimisation
  dyno::utils::TimingStatsCollector timer(
      "parallel_object_sam.optimize." +
      decoupled_formulation_->getFullyQualifiedName());

  bool is_smoother_ok = optimize(&result_.isam_result, new_factors, new_values);

  if (is_smoother_ok) {
    // use dummy isam result when running optimize without new values/factors
    // as we want to use the result to determine which values were
    // changed/marked
    // TODO: maybe we actually need to append results together?
    static gtsam::ISAM2Result dummy_result;
    const auto& max_extra_iterations =
        static_cast<size_t>(params_.num_optimzie);
    VLOG(30) << "Doing extra iteration nr: " << max_extra_iterations;
    for (size_t n_iter = 0; n_iter < max_extra_iterations && is_smoother_ok;
         ++n_iter) {
      is_smoother_ok = optimize(&dummy_result);
    }
  }

  result_.was_smoother_ok = is_smoother_ok;

  VLOG(5) << "ParallelObjectISAM: update complete k=" << frame_k
          << " j= " << object_id_
          << " error before: " << result_.isam_result.errorBefore.value_or(NaN)
          << " error after: " << result_.isam_result.errorAfter.value_or(NaN);
  return is_smoother_ok;
}

bool ParallelObjectISAM::optimize(
    gtsam::ISAM2Result* result, const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::Values& new_values, const ISAM2UpdateParams& update_params) {
  CHECK_NOTNULL(result);
  CHECK(smoother_);

  try {
    *result = smoother_->update(new_factors, new_values, update_params);
    // decoupled_formulation_->updateTheta(new_values);

  } catch (gtsam::IndeterminantLinearSystemException& e) {
    LOG(FATAL) << "gtsam::IndeterminantLinearSystemException with variable "
               << DynoLikeKeyFormatter(e.nearbyVariable());
  }
  return true;
}

void ParallelObjectISAM::updateStates() {
  gtsam::Values previous_estimate = this->getEstimate();
  gtsam::Values estimate = smoother_->calculateEstimate();
  // gtsam::Values estimate = previous_estimate;

  // frame ids at which a motion had a large change
  // TODO: this does not include the new motions?
  FrameIds motions_changed;

  const double motion_delta = 0.1;  // TODO: ehhhhhh magic number for now!!

  const auto previous_motions = previous_estimate.extract<gtsam::Pose3>(
      gtsam::Symbol::ChrTest(kObjectMotionSymbolChar));
  for (const auto& [motion_key, previous_motion_estimate] : previous_motions) {
    if (estimate.exists(motion_key)) {
      gtsam::Pose3 new_motion_estimate = estimate.at<gtsam::Pose3>(motion_key);
      if (!new_motion_estimate.equals(previous_motion_estimate, motion_delta)) {
        ObjectId object_id;
        FrameId frame_id;
        CHECK(reconstructMotionInfo(motion_key, object_id, frame_id));
        CHECK_EQ(object_id, object_id_);

        motions_changed.push_back(frame_id);
      }
    }
  }

  result_.motions_with_large_change = motions_changed;
  result_.large_motion_change_delta = motion_delta;
  result_.motion_variable_status.clear();

  // update with detailed results
  auto detailed_results = result_.isam_result.details();
  // CHECK(detailed_results);
  if (result_.was_smoother_ok && detailed_results) {
    // get all motion symbols
    const auto motion_symbols = estimate.extract<gtsam::Pose3>(
        gtsam::Symbol::ChrTest(kObjectMotionSymbolChar));
    for (const auto& [motion_key, _] : motion_symbols) {
      // TODO: unclear if variables not involved with anything will be in the
      // variable status at all!!
      if (detailed_results->variableStatus.exists(motion_key)) {
        ObjectId object_id;
        FrameId frame_id;
        CHECK(reconstructMotionInfo(motion_key, object_id, frame_id));

        // LOG(INFO) << "Motion key" << DynoLikeKeyFormatter(motion_key) << " at
        // k=" << frame_id << " key= "<< motion_key;
        CHECK_EQ(object_id, object_id_);
        // add variable status associated with the frame id of this motion
        result_.motion_variable_status.insert2(
            frame_id, detailed_results->variableStatus.at(motion_key));
      }
    }
  } else {
    LOG(WARNING) << "Could not update detailed motion results for frame "
                 << result_.frame_id << " as smoother status is "
                 << std::boolalpha << " " << result_.was_smoother_ok
                 << " or detailed results not available "
                 << (bool)detailed_results;
  }

  LOG(INFO) << "Motion change at frames "
            << container_to_string(motions_changed) << " for j=" << object_id_;

  decoupled_formulation_->updateTheta(estimate);
}

void to_json(json& j, const ParallelObjectISAM::Result& result) {
  j["was_smoother_ok"] = result.was_smoother_ok;
  j["frame_id"] = result.frame_id;
  j["isam_result"] = result.isam_result;
  j["motions_with_large_change"] = result.motions_with_large_change;
  j["large_motion_change_delta"] = result.large_motion_change_delta;
  j["motion_variable_status"] = result.motion_variable_status;
}

}  // namespace dyno
