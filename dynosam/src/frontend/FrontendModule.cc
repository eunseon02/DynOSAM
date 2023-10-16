/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/frontend/FrontendModule.hpp"

#include <glog/logging.h>

namespace dyno {

FrontendModule::FrontendModule(const FrontendModuleParams& params)
    :   base_params_(params), frontend_state_(State::Boostrap) {}

FrontendOutputPacketBase::ConstPtr FrontendModule::spinOnce(FrontendInputPacketBase::ConstPtr input) {
    CHECK(input);
    SpinReturn spin_return{State::Boostrap, nullptr};
    switch (frontend_state_)
        {
        case State::Boostrap: {
            spin_return = boostrapSpin(input);
        } break;
        case State::Nominal: {
            spin_return = nominalSpin(input);
        } break;
        default: {
            LOG(FATAL) << "Unrecognized Frontend state.";
            break;
        }
    }

    frontend_state_ = spin_return.first;
    return spin_return.second;
}

} //dyno
