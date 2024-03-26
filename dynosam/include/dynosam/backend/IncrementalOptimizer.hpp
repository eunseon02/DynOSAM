/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */
#pragma once

#include "dynosam/backend/Optimizer.hpp"
#include "dynosam/backend/DynoISAM2.hpp"

#include <gtsam/nonlinear/NonlinearISAM.h>

namespace dyno {


template<typename MEASUREMENT_TYPE>
class IncrementalOptimizer : public Optimizer<MEASUREMENT_TYPE> {

public:
    using Base = Optimizer<MEASUREMENT_TYPE>;
    using This = IncrementalOptimizer<MEASUREMENT_TYPE>;
    using MapType = typename Base::MapType;
    using MeasurementType = typename Base::MeasurementType;

    IncrementalOptimizer()
    {
        DynoISAM2Params isam_params;
    // isam_params.findUnusedFactorSlots = false; //this is very important rn as we naively keep track of slots
        isam_params.enableDetailedResults = true;
        // isam_params.factorizationTranslator("QR");
        isam_params.relinearizeThreshold = 0.01;
        isam_params.relinearizeSkip = 1;
        isam_params.enablePartialRelinearizationCheck = true;
        // isam_params.enableRelinearization = false;

        // smoother_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(2, isam_params);
        smoother_ = std::make_unique<DynoISAM2>(isam_params);
    }

    const DynoISAM2& getSmoother() const {
        return *smoother_;
    }

    const DynoISAM2Result& getResult() const {
        return smoother_result_;
    }


private:

    bool shouldOptimize(const BackendSpinState&) const override {
        return true;
    }

    void updateImpl(const BackendSpinState&, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors, const typename MapType::Ptr) override {
        new_values_.insert(new_values);
        new_factors_ += new_factors;
    }

    std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> optimize() override {
        // gtsam::KeyVector keyVector(new_factors_.begin(), new_factors_.end());

        // gtsam::PrintKeyVector(new_factors_.keyVector(), " new keys", DynoLikeKeyFormatter);

        DynoISAM2UpdateParams isam_update_params;
        isam_update_params.setOrderingFunctions([=](const DynoISAM2UpdateParams& updateParams,
                                        const DynoISAM2Result& result,
                                        const gtsam::KeySet& affectedKeysSet,
                                        const gtsam::VariableIndex& affectedFactorsVarIndex) -> gtsam::Ordering {
            return LatestVariablesLastOrdering(updateParams, result, affectedKeysSet, affectedFactorsVarIndex);
        });


        smoother_result_ = smoother_->update(new_factors_, new_values_, isam_update_params);

        new_factors_.resize(0);
        new_values_.clear();

        return { smoother_->calculateBestEstimate(), smoother_->getFactorsUnsafe() };
    }

    void logStats() {

    }

    gtsam::Ordering MotionLastOrdering(const DynoISAM2UpdateParams& updateParams,
                                        const DynoISAM2Result& result,
                                        const gtsam::KeySet& affectedKeysSet,
                                        const gtsam::VariableIndex& affectedFactorsVarIndex) const
    {
        const FrameId last_updated_frame = this->getLastUpdatedState().frame_id;

        gtsam::FastMap<gtsam::Key, int> constrainedKeys;
        for(const gtsam::Key key : affectedKeysSet) {
            auto chr = DynoChrExtractor(key);

            if(chr == kObjectMotionSymbolChar) {

                ObjectId object_label;
                FrameId frame_id;
                CHECK(reconstructMotionInfo(key, object_label, frame_id));

                if(frame_id == last_updated_frame) {
                    constrainedKeys.insert2(key, 1);
                    LOG(INFO) << "Constrained motion " << DynoLikeKeyFormatter(key);
                }

                //
            }
            // else {
            //     constrainedKeys.insert2(key, 0);
            // }
        }
        return Ordering::ColamdConstrained(affectedFactorsVarIndex, constrainedKeys);
    }


    gtsam::Ordering LatestVariablesLastOrdering(const DynoISAM2UpdateParams& updateParams,
                                        const DynoISAM2Result& result,
                                        const gtsam::KeySet& affectedKeysSet,
                                        const gtsam::VariableIndex& affectedFactorsVarIndex) const
    {
        const FrameId last_updated_frame = this->getLastUpdatedState().frame_id;

        gtsam::FastMap<gtsam::Key, int> constrainedKeys;
        for(const gtsam::Key key : affectedKeysSet) {
            auto chr = DynoChrExtractor(key);

            FrameId variable_frame;
            int group = 0;

            if(chr == kPoseSymbolChar) {
                variable_frame = gtsam::Symbol(key).index();
                group = 3;
            }
            // else if(chr == kObjectMotionSymbolChar) {
            //     ObjectId object_label;
            //     CHECK(reconstructMotionInfo(key, object_label, variable_frame));
            //     group = 2;
            // }
            else if(chr == kDynamicLandmarkSymbolChar) {
                variable_frame = DynamicPointSymbol(key).frameId();
                group = 1;
            }

            if(variable_frame == last_updated_frame) {
                constrainedKeys.insert2(key, group);
            }

            // LOG(INFO) << "Adding group: " << group << " for key " << DynoLikeKeyFormatterVerbose(key);
        }
        return Ordering::ColamdConstrained(affectedFactorsVarIndex, constrainedKeys);
    }


private:
    std::unique_ptr<DynoISAM2> smoother_;
    gtsam::Values new_values_;
    gtsam::NonlinearFactorGraph new_factors_;

    DynoISAM2Result smoother_result_;

};

template<typename MEASUREMENT_TYPE>
class ISAMOptimizer : public Optimizer<MEASUREMENT_TYPE> {

public:
    using Base = Optimizer<MEASUREMENT_TYPE>;
    using This = ISAMOptimizer<MEASUREMENT_TYPE>;
    using MapType = typename Base::MapType;
    using MeasurementType = typename Base::MeasurementType;

    ISAMOptimizer() : isam_(5)
    {
    //     DynoISAM2Params isam_params;
    // // isam_params.findUnusedFactorSlots = false; //this is very important rn as we naively keep track of slots
    //     isam_params.enableDetailedResults = true;
    //     // isam_params.factorizationTranslator("QR");
    //     isam_params.relinearizeThreshold = 0.01;
    //     isam_params.relinearizeSkip = 1;
    //     isam_params.enablePartialRelinearizationCheck = true;
    //     // isam_params.enableRelinearization = false;

    //     // smoother_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(2, isam_params);
    //     smoother_ = std::make_unique<DynoISAM2>(isam_params);
    }


    bool shouldOptimize(const BackendSpinState&) const override {
        return true;
    }

    void updateImpl(const BackendSpinState&, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors, const typename MapType::Ptr) {
        new_values_.insert(new_values);
        new_factors_ += new_factors;
    }

    std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> optimize() override {
        // DynoISAM2UpdateParams isam_update_params;
        // smoother_result_ = smoother_->update(new_factors_, new_values_, isam_update_params);
        isam_.update(new_factors_, new_values_);

        new_factors_.resize(0);
        new_values_.clear();

        return { isam_.estimate(), isam_.getFactorsUnsafe() };
    }

    void logStats() {

    }

    const gtsam::NonlinearISAM& getSmoother() const {
        return isam_;
    }

    // const DynoISAM2Result& getResult() const {
    //     return smoother_result_;
    // }


private:
    // std::unique_ptr<DynoISAM2> smoother_;
    gtsam::NonlinearISAM isam_;
    gtsam::Values new_values_;
    gtsam::NonlinearFactorGraph new_factors_;

    DynoISAM2Result smoother_result_;

};


} //dyno
