// /*
//  *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
//  (jesse.morris@sydney.edu.au)
//  *   All rights reserved.

//  *   Permission is hereby granted, free of charge, to any person obtaining a
//  copy
//  *   of this software and associated documentation files (the "Software"), to
//  deal
//  *   in the Software without restriction, including without limitation the
//  rights
//  *   to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell
//  *   copies of the Software, and to permit persons to whom the Software is
//  *   furnished to do so, subject to the following conditions:

//  *   The above copyright notice and this permission notice shall be included
//  in all
//  *   copies or substantial portions of the Software.

//  *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR
//  *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE
//  *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM,
//  *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE
//  *   SOFTWARE.
//  */

// #include "dynosam/backend/ISAM2Agent.hpp"

// namespace dyno {

// ISAM2Agent::ISAM2Agent(const Params& params)
// : isam2_agent_params_(params)
// {
//     smoother_ = std::make_shared<gtsam::ISAM2>(params.isam);
// }

// bool ISAM2Agent::optimize(
//     gtsam::ISAM2Result* result,
//     const UpdateParams& update_params)

// {
//     dyno::utils::TimingStatsCollector::UniquePtr timer;
//     if(update_params.name) timer =
//     std::make_unique<dyno::utils::TimingStatsCollector>(update_params.name.value());

//     bool is_smoother_ok = optimizeImpl(&result_, update_params);

//     if (is_smoother_ok) {
//         // use dummy isam result when running optimize without new
//         values/factors
//         // as we want to use the result to determine which values were
//         // changed/marked
//         // TODO: maybe we actually need to append results together?
//         static gtsam::ISAM2Result dummy_result;
//         const auto& max_extra_iterations =
//             static_cast<size_t>(isam2_agent_params_.num_optimzie);
//         VLOG(30) << "Doing extra iteration nr: " << max_extra_iterations;
//         for (size_t n_iter = 0; n_iter < max_extra_iterations &&
//         is_smoother_ok;
//             ++n_iter) {
//             is_smoother_ok = optimizeImpl(&dummy_result);
//         }
//     }
// }

// bool ISAM2Agent::optimizeImpl(
//     gtsam::ISAM2Result* result,
//     const UpdateParams& update_params)
// {
//     CHECK_NOTNULL(result);
//     CHECK(smoother_);

//     try {
//         *result = smoother_->update(
//             update_params.new_factors,
//             update_params.new_values,
//             update_params.isam2_update_params);
//     } catch (gtsam::IndeterminantLinearSystemException& e) {
//     LOG(FATAL) << "gtsam::IndeterminantLinearSystemException with variable "
//                 << formatKey(e.nearbyVariable());
//     }
//     return true;
// }

// } //dyno
