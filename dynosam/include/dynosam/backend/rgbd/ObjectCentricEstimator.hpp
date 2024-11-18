/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/common/Map.hpp"

namespace dyno {

struct ObjectCentricProperties {
  inline gtsam::Symbol makeDynamicKey(TrackletId tracklet_id) const {
    return gtsam::Symbol(kDynamicLandmarkSymbolChar, tracklet_id);
  }
};

class ObjectCentricAccessor : public Accessor<Map3d2d>,
                              public ObjectCentricProperties {
 public:
  ObjectCentricAccessor(
      const gtsam::Values* theta, Map3d2d::Ptr map,
      const gtsam::FastMap<ObjectId, std::pair<FrameId, gtsam::Pose3>>*
          L0_values)
      : Accessor<Map3d2d>(theta, map), L0_values_(L0_values) {}
  virtual ~ObjectCentricAccessor() {}

  StateQuery<gtsam::Pose3> getSensorPose(FrameId frame_id) const override;
  StateQuery<gtsam::Pose3> getObjectMotion(FrameId frame_id,
                                           ObjectId object_id) const override;
  StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id,
                                         ObjectId object_id) const override;
  StateQuery<gtsam::Point3> getDynamicLandmark(
      FrameId frame_id, TrackletId tracklet_id) const override;
  // in thie case we can actually propogate all object points ;)
  StatusLandmarkEstimates getDynamicLandmarkEstimates(
      FrameId frame_id, ObjectId object_id) const override;

 private:
  const gtsam::FastMap<ObjectId, std::pair<FrameId, gtsam::Pose3>>*
      L0_values_;  // for now!!
};

class ObjectCentricFormulation : public Formulation<Map3d2d>,
                                 public ObjectCentricProperties {
 public:
  using Base = Formulation<Map3d2d>;
  using Base::AccessorTypePointer;
  using Base::ObjectUpdateContextType;
  using Base::PointUpdateContextType;

  DYNO_POINTER_TYPEDEFS(ObjectCentricFormulation)

  ObjectCentricFormulation(const FormulationParams& params,
                           typename Map::Ptr map,
                           const NoiseModels& noise_models)
      : Base(params, map, noise_models) {}
  virtual ~ObjectCentricFormulation() {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;
  void objectUpdateContext(const ObjectUpdateContextType& context,
                           UpdateObservationResult& result,
                           gtsam::Values& new_values,
                           gtsam::NonlinearFactorGraph& new_factors) override;

  inline bool isDynamicTrackletInMap(
      const LandmarkNode3d2d::Ptr& lmk_node) const override {
    const TrackletId tracklet_id = lmk_node->tracklet_id;
    return is_dynamic_tracklet_in_map_.exists(tracklet_id);
  }

 protected:
  AccessorTypePointer createAccessor(
      const gtsam::Values* values) const override {
    return std::make_shared<ObjectCentricAccessor>(values, this->map(), &L0_);
  }

  std::string loggerPrefix() const override { return "object_centric"; }

 private:
  std::pair<FrameId, gtsam::Pose3> getL0(ObjectId object_id, FrameId frame_id);
  gtsam::Pose3 computeInitialHFromFrontend(ObjectId object_id,
                                           FrameId frame_id);

  gtsam::FastMap<ObjectId, std::vector<PointUpdateContextType>> point_contexts_;
  gtsam::FastMap<ObjectId, std::pair<FrameId, gtsam::Pose3>> L0_;

  // we need a separate way of tracking if a dynamic tracklet is in the map,
  // since each point is modelled uniquely simply used as an O(1) lookup, the
  // value is not actually used. If the key exists, we assume that the tracklet
  // is in the map
  gtsam::FastMap<TrackletId, bool>
      is_dynamic_tracklet_in_map_;  //! thr set of dynamic points that have been
                                    //! added by this updater. We use a separate
                                    //! map containing the tracklets as the keys
                                    //! are non-unique
};

}  // namespace dyno
