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

namespace dyno {

DecoupledObjectSAM::DecoupledObjectSAM(ObjectId object_id,
                                       const gtsam::ISAM2Params& isam_params)
    : object_id_(object_id),
      map_(Map::create()),
      expected_style_(MotionRepresentationStyle::F2F) {
  smoother_ = std::make_shared<gtsam::ISAM2>(isam_params);

  FormulationParams formulation_params;
  // HACK for now so that we get object motions at every frame!!!?
  formulation_params.min_dynamic_observations = 1u;

  NoiseModels noise_models;
  FormulationHooks hooks;

  decoupled_formulation_ =
      std::make_shared<keyframe_object_centric::DecoupledFormulation>(
          formulation_params, map_, noise_models, hooks);
  accessor_ = decoupled_formulation_->accessorFromTheta();
}

Motion3ReferenceFrame DecoupledObjectSAM::getFrame2FrameMotion(
    FrameId frame_id) const {
  // this is in the form of our accessor
  StateQuery<gtsam::Pose3> H_w =
      accessor_->getObjectMotion(frame_id, object_id_);
  CHECK(H_w);

  return Motion3ReferenceFrame(H_w.get(), MotionRepresentationStyle::F2F,
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
  StateQuery<gtsam::Pose3> motion_s0_k =
      accessor_->query<gtsam::Pose3>(motion_key);
  CHECK(motion_s0_k);

  CHECK(decoupled_formulation_->hasObjectKeyFrame(object_id_, frame_id));
  // s0
  auto [reference_frame, _] =
      decoupled_formulation_->getObjectKeyFrame(object_id_, frame_id);

  return Motion3ReferenceFrame(motion_s0_k.get(), MotionRepresentationStyle::KF,
                               ReferenceFrame::GLOBAL, reference_frame,
                               frame_id);
}

void DecoupledObjectSAM::updateSmoother(FrameId frame_k) {
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;
  // no need to update and static or odometry things ;)
  UpdateObservationParams update_params;
  update_params.do_backtrack = false;
  update_params.enable_debug_info = true;
  decoupled_formulation_->updateDynamicObservations(frame_k, new_values,
                                                    new_factors, update_params);

  result_ = smoother_->update(new_factors, new_values);
  estimate_ = smoother_->calculateEstimate();

  decoupled_formulation_->updateTheta(estimate_);
}

}  // namespace dyno
