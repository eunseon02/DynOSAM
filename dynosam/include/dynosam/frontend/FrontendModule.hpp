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
#include "dynosam/common/ModuleBase.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/FrontendOutputPacket.hpp"

#include "dynosam/visualizer/Visualizer-Definitions.hpp"

#include <type_traits>

namespace dyno {


struct InvalidImageContainerException : public DynosamException {
    InvalidImageContainerException(const ImageContainer& container, const std::string& what)
    : DynosamException("Image container with config: " + container.toString() + "\n was invalid - " + what) {}
};

/**
 * @brief Base class to actually do processing. Data passed to this module from the frontend
 *
 */
class FrontendModule : public ModuleBase<FrontendInputPacketBase, FrontendOutputPacketBase>  {

public:
    DYNO_POINTER_TYPEDEFS(FrontendModule)

    using Base = ModuleBase<FrontendInputPacketBase, FrontendOutputPacketBase>;
    using Base::SpinReturn;


    FrontendModule(const FrontendParams& params, ImageDisplayQueue* display_queue = nullptr);
    ~FrontendModule() = default;


protected:

    void validateInput(const FrontendInputPacketBase::ConstPtr& input) const override;
    virtual bool validateImageContainer(const ImageContainer::Ptr& image_container, std::string& reason) const = 0;

protected:
    const FrontendParams base_params_;
    ImageDisplayQueue* display_queue_;

};







} //dyno
