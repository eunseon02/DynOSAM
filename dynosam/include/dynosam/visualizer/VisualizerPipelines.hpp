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
#include "dynosam/visualizer/Display.hpp"
#include "dynosam/frontend/FrontendOutputPacket.hpp"
#include "dynosam/backend/BackendOutputPacket.hpp"
#include "dynosam/pipeline/ThreadSafeQueue.hpp"

#include <glog/logging.h>


namespace dyno {


/**
 * @brief Generic pipeline for display
 *
 * @tparam INPUT
 */
template<typename INPUT>
class DisplayPipeline : public SIMOPipelineModule<INPUT, NullPipelinePayload> {

public:
  using Input = INPUT;
  using This = DisplayPipeline<Input>;
  using Base =  SIMOPipelineModule<Input, NullPipelinePayload>;
  using Display = DisplayBase<Input>;
  DYNO_POINTER_TYPEDEFS(This)

  using InputQueue = typename Base::InputQueue;
  using InputConstPtr = typename Display::InputConstPtr;


  DisplayPipeline(const std::string& name, InputQueue* input_queue, typename Display::Ptr display)
    : Base(name, input_queue), display_(CHECK_NOTNULL(display)) {}

  NullPipelinePayload::ConstPtr process(const InputConstPtr& input) override {
    display_->spinOnce(CHECK_NOTNULL(input));
    static auto null_payload = std::make_shared<NullPipelinePayload>();
    return null_payload;
  }

private:
  typename Display::Ptr display_;

};

using FrontendVizPipeline = DisplayPipeline<FrontendOutputPacketBase>;
using BackendVizPipeline = DisplayPipeline<BackendOutputPacket>;

// struct VisualizerInput : public PipelinePayload {
//   DYNO_POINTER_TYPEDEFS(VisualizerInput)
//   DYNO_DELETE_COPY_CONSTRUCTORS(VisualizerInput)

//   const FrontendOutputPacketBase::Ptr frontend_output_;
//   const BackendOutputPacket::Ptr backend_output_;

//   explicit VisualizerInput(
//     const FrontendOutputPacketBase::Ptr& frontend_output,
//     const BackendOutputPacket::Ptr& backend_output)
//   : frontend_output_(frontend_output), backend_output_(backend_output) {}
// };


// class VisualizerPipeline : public MIMOPipelineModule<VisualizerInput, NullPipelinePayload> {

// public:
//     using Base =  MIMOPipelineModule<VisualizerInput, NullPipelinePayload>;
//     using InputQueue = typename Base::InputQueue;

//     VisualizerPipeline() = default;

//     VisualizerInput::ConstPtr getInputPacket() override;

//     void shutdownQueues() override;

//     //! Checks if the module has work to do (should check input queues are empty)
//     bool hasWork() const override;


// protected:
//   NullPipelinePayload::ConstPtr process(const VisualizerInput::ConstPtr& input) override;

// private:
//   //! Input Queues
//   ThreadsafeQueue<FrontendOutputPacketBase::Ptr> frontend_queue_;
//   ThreadsafeQueue<BackendOutputPacket::Ptr> backend_queue_;

// };








} //dyno
