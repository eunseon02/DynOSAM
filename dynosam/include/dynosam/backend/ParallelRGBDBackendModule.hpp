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

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/RGBDBackendDefinitions.hpp"
#include "dynosam/backend/rgbd/impl/ParallelObjectISAM.hpp"
#include "dynosam/common/Flags.hpp"
#include "dynosam/common/Map.hpp"

namespace dyno {

class ParallelRGBDBackendModule
    : public BackendModuleType<RGBDBackendModuleTraits> {
 public:
  DYNO_POINTER_TYPEDEFS(ParallelRGBDBackendModule)

  using Base = BackendModuleType<RGBDBackendModuleTraits>;
  using RGBDMap = Base::MapType;

  ParallelRGBDBackendModule(const BackendParams& backend_params,
                            Camera::Ptr camera,
                            ImageDisplayQueue* display_queue = nullptr);
  ~ParallelRGBDBackendModule();

 private:
  using SpinReturn = Base::SpinReturn;

  SpinReturn boostrapSpinImpl(
      RGBDInstanceOutputPacket::ConstPtr input) override;
  SpinReturn nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;

  Pose3Measurement bootstrapUpdateStaticEstimator(
      RGBDInstanceOutputPacket::ConstPtr input);
  Pose3Measurement nominalUpdateStaticEstimator(
      RGBDInstanceOutputPacket::ConstPtr input);

  struct PerObjectUpdate {
    FrameId frame_id;
    ObjectId object_id;
    GenericTrackedStatusVector<LandmarkKeypointStatus> measurements;
    Pose3Measurement X_k_measurement;
    Motion3ReferenceFrame H_k_measurement;
  };

  std::vector<PerObjectUpdate> collectMeasurements(
      RGBDInstanceOutputPacket::ConstPtr input,
      const Pose3Measurement& X_k_measurement) const;

  ParallelObjectISAM::Ptr getEstimator(ObjectId object_id,
                                       bool* is_object_new = nullptr);

  void parallelObjectSolve(const std::vector<PerObjectUpdate>& object_updates);
  bool implSolvePerObject(const PerObjectUpdate& object_update);

  BackendOutputPacket::Ptr constructOutputPacket(FrameId frame_k,
                                                 Timestamp timestamp) const;

  void logBackendFromEstimators();

 private:
  Camera::Ptr camera_;

  mutable std::mutex mutex_;

  gtsam::ISAM2Params static_isam2_params_;
  ObjectCentricFormulation::UniquePtr static_formulation_;
  gtsam::ISAM2 static_estimator_;

  gtsam::ISAM2Params dynamic_isam2_params_;
  gtsam::FastMap<ObjectId, ParallelObjectISAM::Ptr> sam_estimators_;

  //! Vector of object ids that are new for this frame. Cleared after each spin
  ObjectIds new_objects_estimators_;

  //! used to cache the result of each update which will we log to file
  GenericObjectCentricMap<ParallelObjectISAM::Result> result_map_;
};

}  // namespace dyno
