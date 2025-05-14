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

#include "dynosam/backend/rgbd/HybridEstimator.hpp"

namespace dyno {
namespace test_hybrid {

class StructurelessDecoupledFormulation : public HybridFormulation {
 public:
  DYNO_POINTER_TYPEDEFS(StructurelessDecoupledFormulation)

  using Base = HybridFormulation;
  StructurelessDecoupledFormulation(const FormulationParams& params,
                                    typename Map::Ptr map,
                                    const NoiseModels& noise_models,
                                    const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  std::string loggerPrefix() const override {
    return "object_centric_structureless_decoupled";
  }
};

class DecoupledFormulation : public HybridFormulation {
 public:
  DYNO_POINTER_TYPEDEFS(DecoupledFormulation)

  using Base = HybridFormulation;
  using Base::Map;

  DecoupledFormulation(const FormulationParams& params, typename Map::Ptr map,
                       const NoiseModels& noise_models,
                       const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  std::string loggerPrefix() const override {
    return "object_centric_decoupled";
  }
};

class StructurlessFormulation : public HybridFormulation {
 public:
  DYNO_POINTER_TYPEDEFS(StructurlessFormulation)

  using Base = HybridFormulation;
  StructurlessFormulation(const FormulationParams& params,
                          typename Map::Ptr map,
                          const NoiseModels& noise_models,
                          const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  std::string loggerPrefix() const override {
    return "object_centric_structureless";
  }
};

class SmartStructurlessFormulation : public HybridFormulation {
 public:
  DYNO_POINTER_TYPEDEFS(SmartStructurlessFormulation)

  using Base = HybridFormulation;
  SmartStructurlessFormulation(const FormulationParams& params,
                               typename Map::Ptr map,
                               const NoiseModels& noise_models,
                               const FormulationHooks& hooks)
      : Base(params, map, noise_models, hooks) {}

  void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  std::string loggerPrefix() const override {
    return "object_centric_smart_structureless";
  }

 private:
  gtsam::FastMap<TrackletId, HybridSmartFactor::shared_ptr>
      tracklet_id_to_smart_factor_;
  gtsam::FastMap<TrackletId, gtsam::FactorIndex>
      tracklet_id_to_smart_factor_index_;
};

}  // namespace test_hybrid
}  // namespace dyno
