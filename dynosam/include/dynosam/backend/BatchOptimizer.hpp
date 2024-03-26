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
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <optional>

#include <glog/logging.h>

namespace dyno {

struct BatchOptimizerParams {

    //! if provided, the function should get the FrameId of the last frame which is used to determine if a full batch opt should be run!
    std::optional<std::function<FrameId()>> get_last_frame;
    gtsam::LevenbergMarquardtParams lm_params;

};


template<typename MEASUREMENT_TYPE>
class BatchOptimizer : public Optimizer<MEASUREMENT_TYPE> {

public:
    using Base = Optimizer<MEASUREMENT_TYPE>;
    using This = BatchOptimizer<MEASUREMENT_TYPE>;
    using MapType = typename Base::MapType;
    using MeasurementType = typename Base::MeasurementType;

    BatchOptimizer(const BatchOptimizerParams& params) : params_(params) {}


protected:

    bool shouldOptimize(const BackendSpinState& state_k) const override {
        if(params_.get_last_frame) {
            FrameId last_frame = (*params_.get_last_frame)();
            const auto frame_id = state_k.frame_id;
            LOG(WARNING) << frame_id << " " << last_frame;
            // -1u becuase reasons...?
            if(frame_id == last_frame - 1u) {
                VLOG(5) << "Should optimize is true as current frame == last frame in the data provider (" << last_frame << ").";
                return true;
            }
        }
        return false;
    }

    void updateImpl(const BackendSpinState&, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors, const typename MapType::Ptr) override {
        initial_.insert_or_assign(new_values);
        graph_ += new_factors;
    }

    std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> optimize() override {
        VLOG(5) << "Beginning batch optimization";

        gtsam::LevenbergMarquardtParams lm_params = params_.lm_params;
        lm_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;


        gtsam::Values opt = gtsam::LevenbergMarquardtOptimizer(graph_, initial_, lm_params).optimize();
        return std::make_pair(opt, graph_);
    }

    void logStats() override {

    }

    inline BatchOptimizerParams getParams() const { return params_; }

private:
    const BatchOptimizerParams params_;

    gtsam::Values initial_;
    gtsam::NonlinearFactorGraph graph_;

};

} //dyno
