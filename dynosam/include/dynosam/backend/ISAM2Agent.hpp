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

// #pragma once

// #include "dynosam/backend/BackendDefinitions.hpp"

// #include <gtsam/nonlinear/ISAM2.h>
// #include <gtsam/nonlinear/ISAM2Params.h>
// #include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/Values.h>

// namespace dyno {

// class ISAM2Agent {

// public:
//     static gtsam::ISAM2Params makeDefaultISAM2Params() {
//         gtsam::ISAM2Params params;
//         params.keyFormatter = DynoLikeKeyFormatter;
//         params.evaluateNonlinearError = true;
//         return params;
//     }

//     struct Params {
//         //! Number additional iSAM updates to run
//         int num_optimzie = 2;
//         gtsam::ISAM2Params isam = ISAM2Agent::makeDefaultISAM2Params();
//     };

//     struct UpdateParams {
//         gtsam::NonlinearFactorGraph new_factors =
//         gtsam::NonlinearFactorGraph(); gtsam::Values new_values =
//         gtsam::Values(); ISAM2UpdateParams isam2_update_params =
//         gtsam::ISAM2UpdateParams();

//         //! If present, used for logging
//         std::optional<std::string> name;
//     };

//     explicit ISAM2Agent(const Params& params);
//     virtual ~ISAM2Agent() = default;

//     virtual bool optimize(
//         gtsam::ISAM2Result* result,
//         const UpdateParams& update_params);

//     /**
//      * @brief Return the keyformatter used by ISAM2.
//      *
//      * This is specified by the ISAM2Params which is found (nested) in
//      * the ISAM2Agent::Params struct.
//      *
//      * @return const gtsam::KeyFormatter&
//      */
//     inline const gtsam::KeyFormatter& keyFormatter() const {
//         return isam2_agent_params_.isam.keyFormatter;
//     }

//     inline std::string formatKey(const gtsam::Key& key) const {
//         return keyFormatter()(key);
//     }

// protected:
//     bool optimizeImpl(
//         gtsam::ISAM2Result* result,
//         const UpdateParams& update_params = UpdateParams{});

//     const Params isam2_agent_params_;
//     std::shared_ptr<gtsam::ISAM2> smoother_;

// };

// } //dyno
