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
#include "dynosam/frontend/FrontendInputPacket.hpp"

struct FrontendOutputPacketBase {
    DYNO_POINTER_TYPEDEFS(FrontendOutputPacketBase)

};


// template<typename DerivedFrontendInputPacket, typename DerivedInputImageBase>

namespace dyno {

class FrontendPipeline : public SIMOPipelineModule<FrontendInputPacketBase, FrontendOutputPacketBase> {

public:
    DYNO_POINTER_TYPEDEFS(FrontendPipeline)

    using SIMO =
      SIMOPipelineModule<FrontendInputPacketBase, FrontendOutputPacketBase>;
    using OutputQueue = typename SIMO::OutputQueue;

    FrontendPipeline(const std::string& module_name);

    FrontendOutputPacketBase::ConstPtr process(const FrontendInputPacketBase::ConstPtr& input) override;

protected:
    virtual FrontendOutputPacketBase::ConstPtr boostrapSpin(const FrontendInputPacketBase::ConstPtr input) = 0;
    virtual FrontendOutputPacketBase::ConstPtr nominalSpin(const FrontendInputPacketBase::ConstPtr input) = 0;


};


} //dyno
