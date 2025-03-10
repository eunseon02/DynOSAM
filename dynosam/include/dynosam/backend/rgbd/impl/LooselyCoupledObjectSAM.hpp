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

#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/rgbd/ObjectCentricEstimator.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/common/Types.hpp"  //only needed for factors

namespace dyno {

// using namespace keyframe_object_centric;

// TODO: should rename to SAMAgent!! (and tracking not decoupled!!)
// Incremental tracking SAMAgent (with camera pose stuff!!)
// to make it truly incremental!!!
// TODO: right now copy of DecoupledObjectSAM which is used in the front-end
// (when first developed)
class LooselyCoupledObjectSAM {
 public:
  DYNO_POINTER_TYPEDEFS(LooselyCoupledObjectSAM)

  struct Params {
    //! Number additional iSAM updates to run
    int num_optimzie = 2;
    gtsam::ISAM2Params isam{};
  };

  using Map = ObjectCentricFormulation::Map;

  template <typename DERIVEDSTATUS>
  using MeasurementStatusVector = Map::MeasurementStatusVector<DERIVEDSTATUS>;

  LooselyCoupledObjectSAM(const Params& params, ObjectId object_id,
                          const NoiseModels& noise_models,
                          const FormulationHooks& formulation_hooks);

  // what motion representation should this be in? GLOBAL? Do ne need a new
  // repsentation for KF object centric?
  template <typename DERIVEDSTATUS>
  void update(FrameId frame_k,
              const MeasurementStatusVector<DERIVEDSTATUS>& measurements,
              const gtsam::Pose3& X_world_k,
              const Motion3ReferenceFrame& motion_frame) {
    VLOG(5) << "LooselyCoupledObjectSAM::update running for k= " << frame_k
            << ", j= " << object_id_;

    this->updateMap(frame_k, measurements, X_world_k, motion_frame);

    // updating the smoothing will update the formulation and run
    // update on the optimizer. the internal results_ object is updated
    const bool is_smoother_ok = this->updateSmoother(frame_k, X_world_k);

    if (is_smoother_ok) {
      updateStates();
    }
  }

  const gtsam::Values& getEstimate() const {
    return decoupled_formulation_->getTheta();
  }
  const gtsam::ISAM2Result& getISAM2Result() const { return result_; }

  inline Map::Ptr map() const { return map_; }

  Motion3ReferenceFrame getFrame2FrameMotion(FrameId frame_id) const;
  Motion3ReferenceFrame getKeyFramedMotion(FrameId frame_id) const;

  // all frames
  ObjectPoseMap getObjectPoses() const { return accessor_->getObjectPoses(); }
  // all frames
  ObjectMotionMap getFrame2FrameMotions() const;
  ObjectMotionMap getKeyFramedMotions() const;

  // due to the nature of this formulation, this will be the accumulated cloud!!
  StatusLandmarkVector getDynamicLandmarks(FrameId frame_id) const;

 private:
  template <typename DERIVEDSTATUS>
  void updateMap(FrameId frame_k,
                 const MeasurementStatusVector<DERIVEDSTATUS>& measurements,
                 const gtsam::Pose3& X_world_k,
                 const Motion3ReferenceFrame& motion_frame) {
    map_->updateObservations(measurements);
    map_->updateSensorPoseMeasurement(frame_k, X_world_k);

    const FrameId to = motion_frame.to();
    if (to != frame_k) {
      throw DynosamException(
          "LooselyCoupledObjectSAM::updateMap failed as the 'to' frame of the "
          "initial motion was not the same as expected frame id");
    }

    // check style of motion is self consistent
    if (!expected_style_) {
      expected_style_ = motion_frame.style();
    } else {
      CHECK_EQ(expected_style_.value(), motion_frame.style());
    }

    // TODO: now we have camera pose ;)

    // do we want global?
    MotionEstimateMap motion_estimate;
    motion_estimate.insert({object_id_, motion_frame});
    map_->updateObjectMotionMeasurements(frame_k, motion_estimate);
  }

  bool updateSmoother(FrameId frame_k, const gtsam::Pose3& X_world_k);

  void updateFormulation(FrameId frame_k, const gtsam::Pose3& X_world_k,
                         gtsam::NonlinearFactorGraph& new_factors,
                         gtsam::Values& new_values);

  bool optimize(
      gtsam::ISAM2Result* result,
      const gtsam::NonlinearFactorGraph& new_factors =
          gtsam::NonlinearFactorGraph(),
      const gtsam::Values& new_values = gtsam::Values(),
      const ISAM2UpdateParams& update_params = gtsam::ISAM2UpdateParams());

  void updateStates();

 private:
  const Params params_;
  const ObjectId object_id_;
  Map::Ptr map_;
  ObjectCentricFormulation::Ptr decoupled_formulation_;
  Accessor<Map>::Ptr accessor_;
  std::shared_ptr<gtsam::ISAM2> smoother_;
  gtsam::ISAM2Result result_;
  //! style of motion expected to be used as input. Set on the first run and all
  //! motions are expected to then follow the same style
  std::optional<MotionRepresentationStyle> expected_style_;
};

}  // namespace dyno
