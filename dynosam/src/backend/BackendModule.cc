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

#include "dynosam/backend/BackendModule.hpp"

#include <glog/logging.h>

#include "dynosam/backend/BackendDefinitions.hpp"

namespace dyno {

BackendModule::BackendModule(const BackendParams& params,
                             ImageDisplayQueue* display_queue)
    : Base("backend"),
      base_params_(params),
      display_queue_(display_queue)

{
  setFactorParams(params);

  // create callback to update gt_packet_map_ values so the derived classes dont
  // need to manage this
  // TODO: this logic is exactly the same as in FrontendModule - functionalise!!
  registerInputCallback([=](BackendInputPacket::ConstPtr input) {
    if (input->gt_packet_)
      gt_packet_map_.insert2(input->getFrameId(), *input->gt_packet_);

    const BackendSpinState previous_spin_state = spin_state_;

    // update spin state
    spin_state_ = BackendSpinState(input->getFrameId(), input->getTimestamp(),
                                   previous_spin_state.iteration + 1);
  });
}

void BackendModule::setFactorParams(const BackendParams& backend_params) {
  // set static noise
  //  noise_models_.static_pixel_noise = gtsam::noiseModel::Isotropic::Sigma(2u,
  //  backend_params.static_smart_projection_noise_sigma_);
  //  CHECK(noise_models_.static_pixel_noise);

  // set dynamic noise
  //  noise_models_.dynamic_pixel_noise =
  //  gtsam::noiseModel::Isotropic::Sigma(2u,
  //  backend_params.dynamic_smart_projection_noise_sigma_);
  //  CHECK(noise_models_.dynamic_pixel_noise);

  gtsam::Vector6 odom_sigmas;
  odom_sigmas.head<3>().setConstant(backend_params.odometry_rotation_sigma_);
  odom_sigmas.tail<3>().setConstant(backend_params.odometry_translation_sigma_);
  noise_models_.odometry_noise =
      gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);
  CHECK(noise_models_.odometry_noise);

  noise_models_.initial_pose_prior =
      gtsam::noiseModel::Isotropic::Sigma(6u, 0.000001);
  CHECK(noise_models_.initial_pose_prior);

  noise_models_.landmark_motion_noise = gtsam::noiseModel::Isotropic::Sigma(
      3u, backend_params.motion_ternary_factor_noise_sigma_);
  CHECK(noise_models_.landmark_motion_noise);

  gtsam::Vector6 object_constant_vel_sigmas;
  object_constant_vel_sigmas.head<3>().setConstant(
      backend_params.constant_object_motion_rotation_sigma_);
  object_constant_vel_sigmas.tail<3>().setConstant(
      backend_params.constant_object_motion_translation_sigma_);
  noise_models_.object_smoothing_noise =
      gtsam::noiseModel::Diagonal::Sigmas(object_constant_vel_sigmas);
  CHECK(noise_models_.object_smoothing_noise);
}

void BackendModule::validateInput(const BackendInputPacket::ConstPtr&) const {}

}  // namespace dyno
