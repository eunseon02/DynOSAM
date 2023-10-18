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

#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/FrontendOutputPacket.hpp"

#include <type_traits>

namespace dyno {

/**
 * @brief Base class to actually do processing. Data passed to this module from the frontend
 *
 */
class FrontendModule {

public:
    DYNO_POINTER_TYPEDEFS(FrontendModule)

    enum class State {
        Boostrap = 0u, //! Initalize Frontend
        Nominal = 1u //! Run Frontend
    };

    using SpinReturn = std::pair<State, FrontendOutputPacketBase::ConstPtr>;

    FrontendModule(const FrontendParams& params);
    ~FrontendModule() = default;

    FrontendOutputPacketBase::ConstPtr spinOnce(FrontendInputPacketBase::ConstPtr input);

protected:
    virtual bool validateImageContainer(const ImageContainer::Ptr& image_container) const = 0;

    virtual SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) = 0;
    virtual SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) = 0;

protected:
    const FrontendParams base_params_;

private:
    std::atomic<State> frontend_state_;

};


// class FrontendModuleTyped

// template<typename ExpectedFrontendInputPacket, typename ExpectedInputImagePacket>
// struct FrontendModuleTypeChekcer {

//     static_assert(std::is_base_of_v<FrontendInputPacketBase,ExpectedFrontendInputPacket> == true);
//     static_assert(std::is_base_of_v<InputImagePacketBase,ExpectedInputImagePacket> == true);

//     //! To match the const types from the pipeline which use constptr
//     using ExpectedFrontendInputPacketConstPtr = std::shared_ptr<const ExpectedFrontendInputPacket>;
//     using ExpectedInputImagePacketConstPtr = std::shared_ptr<const ExpectedInputImagePacket>;



// };






} //dyno
