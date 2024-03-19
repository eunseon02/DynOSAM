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

namespace dyno {


template<typename MEASUREMENT_TYPE>
class IncrementalOptimizer : public Optimizer<MEASUREMENT_TYPE> {

public:
    using Base = Optimizer<MEASUREMENT_TYPE>;
    using This = IncrementalOptimizer<MEASUREMENT_TYPE>;
    using MapType = typename Base::MapType;
    using MeasurementType = typename Base::MeasurementType;


    bool shouldOptimize(FrameId) const {
        return true;
    }

    void update(FrameId, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors, const typename MapType::Ptr) {
        new_values_ = new_values;
        new_factors_ = new_factors;
    }

    std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> optimize() {

    }

    void logStats() {

    }

    const DynoISAM2& getSmoother() const {
        return *smoother_;
    }


private:
    std::unique_ptr<DynoISAM2> smoother_;
    gtsam::Values new_values_;
    gtsam::NonlinearFactorGraph new_factors_;

};

} //dyno
