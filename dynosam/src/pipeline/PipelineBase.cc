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

#include "dynosam/pipeline/PipelineBase.hpp"

#include "dynosam/utils/Timing.hpp"
#include "dynosam/utils/TimingStats.hpp"

#include <glog/logging.h>

#include <memory>
#include <thread>  // std::this_thread::sleep_for
#include <chrono>  // std::chrono::

namespace dyno {

bool AbstractPipelineModuleBase::spin()
  {
    LOG(INFO) << "Starting module " << module_name_;
    while (!isShutdown())
    {
      spinOnce();
    //   using namespace std::chrono_literals;
    //   std::this_thread::sleep_for(1ms);  // give CPU thread some sleep time... //TODO: only if threaded?
    }
    return true;
  }

void AbstractPipelineModuleBase::shutdown()
{
    LOG(INFO) << "Shutting down module " << module_name_;
    is_shutdown_ = true;
    shutdownQueues();
}

bool AbstractPipelineModuleBase::isWorking() const
{
    return is_thread_working_ || hasWork();
}


void AbstractPipelineModuleBase::notifyFailures(PipelineReturnStatus result)
{
    for (const auto& failure_callbacks : on_failure_callbacks_)
    {
        if (failure_callbacks)
        {
            failure_callbacks(result);
        }
        else
        {
            LOG(ERROR) << "Invalid OnFailureCallback for module: " << module_name_;
        }
    }
}

void AbstractPipelineModuleBase::registerOnFailureCallback(const OnPipelineFailureCallback& callback_)
{
    CHECK(callback_);
    on_failure_callbacks_.push_back(callback_);
}



} //dyno
