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

#include "dynosam/backend/rgbd/impl/ObjectCentricFormulations.hpp"

#include "dynosam/backend/rgbd/ObjectCentricEstimator.hpp"  //only for now as this is where the factors are!!

namespace dyno {
namespace keyframe_object_centric {

void StructurelessDecoupledFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(context.getObjectId());
  auto landmark_motion_noise = noise_models_.landmark_motion_noise;

  gtsam::Pose3 X_K_1 =
      this->getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
  gtsam::Pose3 X_K =
      this->getInitialOrLinearizedSensorPose(frame_node_k->frame_id);

  gtsam::Pose3 L_0;
  FrameId s0;
  std::tie(s0, L_0) = getL0(context.getObjectId(), frame_node_k_1->getId());
  auto dynamic_point_noise = noise_models_.dynamic_point_noise;

  new_factors.emplace_shared<StructurelessDecoupledObjectCentricMotion>(
      object_motion_key_k_1, object_motion_key_k, X_K_1, X_K,
      lmk_node->getMeasurement(frame_node_k_1).landmark,
      lmk_node->getMeasurement(frame_node_k).landmark, L_0,
      dynamic_point_noise);

  result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
}

void DecoupledFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  auto theta_accessor = this->accessorFromTheta();

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(context.getObjectId());

  gtsam::Pose3 L_0;
  FrameId s0;
  std::tie(s0, L_0) = getL0(context.getObjectId(), frame_node_k_1->getId());
  auto landmark_motion_noise = noise_models_.landmark_motion_noise;

  // TODO:this will not be the case with sliding/window as we reconstruct the
  // graph from a different starting point!!
  //  CHECK_GE(frame_node_k_1->getId(), s0);

  if (!isDynamicTrackletInMap(lmk_node)) {
    // mark as now in map
    is_dynamic_tracklet_in_map_.insert2(context.getTrackletId(), true);
    CHECK(isDynamicTrackletInMap(lmk_node));

    // use first point as initalisation?
    // in this case k is k-1 as we use frame_node_k_1
    gtsam::Pose3 s0_H_k_world = computeInitialHFromFrontend(
        context.getObjectId(), frame_node_k_1->getId());
    gtsam::Pose3 L_k = s0_H_k_world * L_0;
    // H from k to s0 in frame k (^wL_k)
    //  gtsam::Pose3 k_H_s0_k = L_0 * s0_H_k_world.inverse() *  L_0.inverse();
    gtsam::Pose3 k_H_s0_k = (L_0.inverse() * s0_H_k_world * L_0).inverse();
    gtsam::Pose3 k_H_s0_W = L_k * k_H_s0_k * L_k.inverse();
    // LOG(INFO) << "s0_H_k " << s0_H_k;
    // measured point in camera frame
    const gtsam::Point3 m_camera =
        lmk_node->getMeasurement(frame_node_k_1).landmark;
    Landmark lmk_L0_init =
        L_0.inverse() * k_H_s0_W * context.X_k_1_measured * m_camera;

    // initalise value //cannot initalise again the same -> it depends where L_0
    // is created, no?
    Landmark lmk_L0;
    getSafeQuery(lmk_L0, theta_accessor->query<Landmark>(point_key),
                 lmk_L0_init);
    new_values.insert(point_key, lmk_L0);
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
  }

  auto dynamic_point_noise = noise_models_.dynamic_point_noise;
  if (context.is_starting_motion_frame) {
    gtsam::Pose3 X_k_1 =
        this->getInitialOrLinearizedSensorPose(frame_node_k_1->frame_id);
    new_factors.emplace_shared<DecoupledObjectCentricMotionFactor>(
        object_motion_key_k_1, point_key,
        lmk_node->getMeasurement(frame_node_k_1).landmark, L_0, X_k_1,
        dynamic_point_noise);
    result.updateAffectedObject(frame_node_k_1->frame_id,
                                context.getObjectId());
  }

  gtsam::Pose3 X_k =
      this->getInitialOrLinearizedSensorPose(frame_node_k->frame_id);

  new_factors.emplace_shared<DecoupledObjectCentricMotionFactor>(
      object_motion_key_k, point_key,
      lmk_node->getMeasurement(frame_node_k).landmark, L_0, X_k,
      dynamic_point_noise);
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
}

void StructurlessFormulation::dynamicPointUpdateCallback(
    const PointUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  const auto lmk_node = context.lmk_node;
  const auto frame_node_k_1 = context.frame_node_k_1;
  const auto frame_node_k = context.frame_node_k;

  gtsam::Key point_key = this->makeDynamicKey(context.getTrackletId());

  const gtsam::Key object_motion_key_k =
      frame_node_k->makeObjectMotionKey(context.getObjectId());
  const gtsam::Key object_motion_key_k_1 =
      frame_node_k_1->makeObjectMotionKey(context.getObjectId());
  auto landmark_motion_noise = noise_models_.landmark_motion_noise;

  gtsam::Pose3 L_0;
  FrameId s0;
  std::tie(s0, L_0) = getL0(context.getObjectId(), frame_node_k_1->getId());
  auto dynamic_point_noise = noise_models_.dynamic_point_noise;

  new_factors.emplace_shared<StructurelessObjectCentricMotionFactor2>(
      frame_node_k_1->makePoseKey(), object_motion_key_k_1,
      frame_node_k->makePoseKey(), object_motion_key_k,
      lmk_node->getMeasurement(frame_node_k_1).landmark,
      lmk_node->getMeasurement(frame_node_k).landmark, L_0,
      dynamic_point_noise);

  result.updateAffectedObject(frame_node_k_1->frame_id, context.getObjectId());
  result.updateAffectedObject(frame_node_k->frame_id, context.getObjectId());
}

}  // namespace keyframe_object_centric
}  // namespace dyno
