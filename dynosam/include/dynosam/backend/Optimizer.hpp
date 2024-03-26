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

#include "dynosam/backend/BackendDefinitions.hpp"

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


    /**
     * @brief Updates and potentially optimizes the set of values given the factor graph. Update is intended to be called
     * incrementally (ie. every frame) with the set of new_values and new_factors. This reflects the interface for isam/isam2.
     *
     * In the case of a batch solver, the updateImpl function should simply append the new values/factors to an existing set
     * until the system is ready to sovle.
     *
     * The logic is as follows:
     * calls overwritten updateImpl to update the internal set of states and factors.
     *
     * if shouldOptimize is true,
     * then optimize() is called and the values returned by getValues() and getFactors() are updated.
     * The value returned by getLastOptimizedState is also updated.
     *
     * NOTE: the values of getValues() and getFactors() are ONLY updated when shouldOptimize() returns true, however, the user can overwrite this
     * by setting force_optimize == true
     *
     *
     * True is returned if the function also optimzies the set of values.
     *
     * @param state_k
     * @param new_values
     * @param new_factors
     * @param map
     * @param bool force_optimize. Forces the update step to also run optimize. Default value: false.
     * @return true
     * @return false
     */
    bool update(const BackendSpinState& state_k, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors, const typename MapType::Ptr map, bool force_optimize = false) {
        last_updated_state_ = state_k;
        updateImpl(state_k, new_values, new_factors, map);

        if(shouldOptimize(state_k) || force_optimize) {
            gtsam::Values estimate;
            gtsam::NonlinearFactorGraph graph;

            last_optimzied_state_ = state_k;
            std::tie(estimate, graph) = this->optimize();

            // update internal estimate and graph values to reflect the updated optimized state
            values_ = estimate;
            graph_ = graph;
            return true;
        }
        return false;
    }


     /**
     * @brief Get all the optimized values as returned by optimize().first. This may not be the ENTIRE set of states (from the perspective
     * of the whole system, e.g. if marginalization is used etc).
     *
     * These values are updated when optimize() is called as indicated when attemptOptimize() returns true
     *
     * The full state should be accessed via the map
     *
     * @return gtsam::Values
     */
    inline const gtsam::Values& getValues() const { return values_; }


    /**
     * @brief Get the NonlinearFactorGraph that the optimizer will use when optimize is next called. This may not be the ENTIRE set of factors
     * constructed by the BackendModule (e.g. if marginazliation is used etc)
     *
     * These values are updated when optimize() is called as indicated when update() returns true
     *
     * @return gtsam::NonlinearFactorGraph
     */
    inline const gtsam::NonlinearFactorGraph& getFactors() const { return graph_; }

    /**
     * @brief Get the Last Updated State object.
     * This is updated when update() is called
     *
     * @return const BackendSpinState&
     */
    inline const BackendSpinState& getLastUpdatedState() const { return last_updated_state_; }

    /**
     * @brief Get the Last Optimized State object.
     * This is updated when the internal optimize function is called, which happens when update() returns true.
     *
     *
     * @return const BackendSpinState&
     */
    inline const BackendSpinState& getLastOptimizedState() const { return last_optimzied_state_; }

protected:

    /**
     * @brief Virtual function that indicates that the system should be optimized.
     *
     * Used by the update() function to decide if optimize() should be called
     *
     * @param state_k
     * @return true
     * @return false
     */
    virtual bool shouldOptimize(const BackendSpinState& state_k) const = 0;

    /**
     * @brief Internal update function that should update the classes internal state/graph representation with new values.
     *
     * This function is intended to be called an incremental fashion, with new values/factors being new per frame, to reflect the isam/isam2 structure.
     *
     * @param state_k
     * @param new_values
     * @param new_factors
     * @param map
     */
    virtual void updateImpl(const BackendSpinState& state_k, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors, const typename MapType::Ptr map) = 0;

    virtual std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> optimize() = 0;
    virtual void logStats() = 0;


private:
    BackendSpinState last_updated_state_;
    //NOTE: assumes that frames start from 0
    BackendSpinState last_optimzied_state_;

    gtsam::Values values_; //! Values updated from the optimizer when optimize() is called
    gtsam::NonlinearFactorGraph graph_; //! gtsam::NonlinearFactorGraph updated from the optimizer when optimize() is called


};



} // namespace dyno
