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

#include "dynosam/backend/rgbd/impl/DecoupledObjectSAM.hpp"

#include "dynosam/utils/TimingStats.hpp"

namespace dyno {

DecoupledObjectSAM::DecoupledObjectSAM(
    ObjectId object_id, const NoiseModels& noise_models,
    const FormulationHooks& formulation_hooks,
    const gtsam::ISAM2Params& isam_params)
    : object_id_(object_id),
      map_(Map::create()),
      expected_style_(MotionRepresentationStyle::F2F) {
  smoother_ = std::make_shared<gtsam::ISAM2>(isam_params);

  FormulationParams formulation_params;
  formulation_params.suffix = "object_" + std::to_string(object_id);
  // HACK for now so that we get object motions at every frame!!!?
  formulation_params.min_dynamic_observations = 1u;

  decoupled_formulation_ =
      std::make_shared<keyframe_object_centric::DecoupledFormulation>(
          formulation_params, map_, noise_models, formulation_hooks);
  accessor_ = decoupled_formulation_->accessorFromTheta();
}

Motion3ReferenceFrame DecoupledObjectSAM::getFrame2FrameMotion(
    FrameId frame_id) const {
  // this is in the form of our accessor
  StateQuery<gtsam::Pose3> H_W_km1_k =
      accessor_->getObjectMotion(frame_id, object_id_);
  CHECK(H_W_km1_k);

  return Motion3ReferenceFrame(H_W_km1_k.get(), MotionRepresentationStyle::F2F,
                               ReferenceFrame::GLOBAL, frame_id - 1u, frame_id);
}

// currently no way of checking (with the object) the type of motion we have!!
Motion3ReferenceFrame DecoupledObjectSAM::getKeyFramedMotion(
    FrameId frame_id) const {
  // not in form of accessor but in form of estimation
  const auto frame_node_k = map_->getFrame(frame_id);
  CHECK_NOTNULL(frame_node_k);

  auto motion_key = frame_node_k->makeObjectMotionKey(object_id_);
  // raw access the theta in the accessor!!

  StateQuery<gtsam::Pose3> H_W_s0_k =
      accessor_->query<gtsam::Pose3>(motion_key);
  CHECK(H_W_s0_k);

  CHECK(decoupled_formulation_->hasObjectKeyFrame(object_id_, frame_id));
  // s0
  auto [reference_frame, _] =
      decoupled_formulation_->getObjectKeyFrame(object_id_, frame_id);

  return Motion3ReferenceFrame(H_W_s0_k.get(), MotionRepresentationStyle::KF,
                               ReferenceFrame::GLOBAL, reference_frame,
                               frame_id);
}

void DecoupledObjectSAM::updateFormulation(
    FrameId frame_k, gtsam::NonlinearFactorGraph& new_factors,
    gtsam::Values& new_values) {
  // no need to update and static or odometry things ;)
  UpdateObservationParams update_params;
  update_params.do_backtrack = false;
  update_params.enable_debug_info = true;
  VLOG(10) << "DecoupledObjectSAM: Starting formulation update k=" << frame_k
           << " j= " << object_id_;
  decoupled_formulation_->updateDynamicObservations(frame_k, new_values,
                                                    new_factors, update_params);
}

bool DecoupledObjectSAM::updateSmoother(FrameId frame_k) {
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  updateFormulation(frame_k, new_factors, new_values);

  // do first optimisation
  dyno::utils::TimingStatsCollector timer(
      "decoupled_object_sam.optimize." +
      decoupled_formulation_->getFullyQualifiedName());
  bool is_smoother_ok = optimize(&result_, new_factors, new_values);

  VLOG(5) << "DecoupledObjectSAM: update complete k=" << frame_k
          << " j= " << object_id_
          << "  error before: " << result_.errorBefore.value_or(NaN)
          << " error after: " << result_.errorAfter.value_or(NaN);
  return is_smoother_ok;
}

bool DecoupledObjectSAM::optimize(
    gtsam::ISAM2Result* result, const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::Values& new_values) {
  CHECK_NOTNULL(result);
  CHECK(smoother_);

  try {
    *result = smoother_->update(new_factors, new_values);
  } catch (gtsam::IndeterminantLinearSystemException& e) {
    LOG(FATAL) << "gtsam::IndeterminantLinearSystemException with variable "
               << DynoLikeKeyFormatter(e.nearbyVariable());
  }
  estimate_ = smoother_->calculateEstimate();
  decoupled_formulation_->updateTheta(estimate_);

  return true;
}

}  // namespace dyno
