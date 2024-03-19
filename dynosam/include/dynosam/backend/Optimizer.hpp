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

#include "dynosam/common/Types.hpp"
#include "dynosam/common/Map.hpp"

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace dyno
{

template<typename MEASUREMENT_TYPE>
class Optimizer {

public:
    //Must be a gtsam value type and meet all the concept criteria for that
    using MeasurementType = MEASUREMENT_TYPE;
    using This = Optimizer<MEASUREMENT_TYPE>;

    using MapType = Map<MeasurementType>;

    DYNO_POINTER_TYPEDEFS(This)

    Optimizer() = default;
    virtual ~Optimizer() {}

    virtual bool shouldOptimize(FrameId frame_id) const = 0;

    // /**
    //  * @brief Get all the values that the optimizer will be optimzing over. This may not be the ENTIRE set of states (from the perspective
    //  * of the whole system, e.g. if marginalization is used etc).
    //  *
    //  * The full state should be accessed via the map
    //  *
    //  * @return gtsam::Values
    //  */
    // virtual gtsam::Values getValues() const = 0;


    // /**
    //  * @brief Get the NonlinearFactorGraph that the optimizer will use when optimize is next called. This may not be the ENTIRE set of factors
    //  * constructed by the BackendModule (e.g. if marginazliation is used etc)
    //  *
    //  *
    //  * @return gtsam::NonlinearFactorGraph
    //  */
    // virtual gtsam::NonlinearFactorGraph getFactors() const = 0;

    //TODO: save graph

    virtual void update(FrameId frame_id_k, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors, const typename MapType::Ptr map) = 0;
    virtual std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> optimize() = 0;
    virtual void logStats() = 0;


};

} // namespace dyno
