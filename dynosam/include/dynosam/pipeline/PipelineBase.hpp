/*
 *   Copyright (c) 2023 Jesse Morris
 *   All rights reserved.
 */
#pragma once

#include "dynosam/pipeline/PipelineBase-Definitions.hpp"
#include "dynosam/pipeline/ThreadSafeQueue.hpp"
#include "dynosam/utils/Timing.hpp"
#include "dynosam/utils/TimingStats.hpp"

#include <atomic>
#include <functional>  // for function
#include <exception>
#include <memory>
#include <string>
#include <utility>  // for move
#include <vector>
#include <thread>  // std::this_thread::sleep_for
#include <chrono>  // std::chrono::

#include <glog/logging.h>

#include <memory>
#include <thread>

namespace dyno
{

class AbstractPipelineModuleBase
{
public:
  AbstractPipelineModuleBase(const std::string& module_name) : module_name_(module_name)
  {
  }

  virtual ~AbstractPipelineModuleBase() = default;

  virtual bool spin() = 0;
  virtual void shutdownQueues() = 0;
  virtual bool hasWork() const = 0;

  virtual inline void shutdown()
  {
    LOG(INFO) << "Shutting down module " << module_name_;
    is_shutdown = true;
    shutdownQueues();
  }

  bool isShutdown() const
  {
    return is_shutdown;
  }

  inline bool isWorking() const
  {
    return is_thread_working || hasWork();
  }

  inline std::string getModuleName() const
  {
    return module_name_;
  }

  void notifyFailures(PipelineReturnCode result)
  {
    for (const auto& failure_callbacks : on_failure_callbacks)
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

  virtual void registerOnFailureCallback(const OnPipelineFailureCallback& callback_)
  {
    CHECK(callback_);
    on_failure_callbacks.push_back(callback_);
  }


protected:
  const std::string module_name_;
  std::atomic_bool is_thread_working{ false };

private:
  std::vector<OnPipelineFailureCallback> on_failure_callbacks;
  //! Thread related members.
  std::atomic_bool is_shutdown{ false };
};

template <typename INPUT, typename OUTPUT>
class AbstractPipelineModule : public AbstractPipelineModuleBase
{
public:
  using InputConstSharedPtr = std::shared_ptr<const INPUT>;
  //! The output is instead a shared ptr, since many users might need the output
  // using OutputConstUniquePtr = std::unique_ptr<const OUTPUT>;
  using OutputConstSharedPtr = std::shared_ptr<const OUTPUT>;

  using OnProcessCallback = std::function<void(const InputConstSharedPtr&, const OutputConstSharedPtr&)>;

  AbstractPipelineModule(const std::string& module_name) : AbstractPipelineModuleBase(module_name)
  {
  }

  virtual ~AbstractPipelineModule() = default;

  bool spin() override
  {
    LOG(INFO) << "Starting module " << module_name_;
    while (!isShutdown())
    {
      spinOnce();
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1ms);  // give CPU thread some sleep time...
    }
    return true;
  }

  PipelineReturnCode spinOnce()
  {
    if (isShutdown())
    {
      return PipelineReturnCode::IS_SHUTDOWN;
    }

    PipelineReturnCode return_code;
    utils::TimingStatsCollector timing_stats(module_name_);

    InputConstSharedPtr input = nullptr;
    is_thread_working = false;
    try
    {
      input = getInputPacket();
    }
    catch (const std::exception& e)
    {
      // context::shutdown("Exception raised in pipeline " + module_name_ +
      //                   " on getInputPacket(): " + std::string(e.what()));
    }
    is_thread_working = true;
    // this->logTiming("get_input_packet", utils::Timer::toc(tic_spin));

    if (input)
    {
      // Transfer the ownership of input to the actual pipeline module.
      // From this point on, you cannot use input, since process owns it.
      auto tic_process = utils::Timer::tic();
      OutputConstSharedPtr output = nullptr;
      try
      {
        output = process(input);
        // this->logTiming("process", utils::Timer::toc(tic_process));
      }
      catch (const std::exception& e)
      {
        // context::shutdown("Exception raised in pipeline " + module_name_ + " on process(): " + std::string(e.what()));
      }

      if (output)
      {
        // Received a valid output, send to output queue
        if (!pushOutputPacket(output))
        {
          LOG_EVERY_N(WARNING, 2) << "Module: " << module_name_ << " - Output push failed.";
          is_thread_working = false;
          return_code = PipelineReturnCode::OUTPUT_PUSH_FAILURE;
        }
        else
        {
          emitProcessCallbacks(input, output);
          is_thread_working = false;
          return_code = PipelineReturnCode::SUCCESS;
        }
      }
      else
      {
        return_code = PipelineReturnCode::PROCESSING_FAILURE;
        notifyFailures(return_code);
        is_thread_working = false;
      }
    }
    else
    {
      is_thread_working = false;
      return_code = PipelineReturnCode::GET_PACKET_FAILURE;
    }

    return return_code;
  }

  // these happen in the same thread as the module they are attached to so should happen quickly
  virtual void registerOnProcessCallback(const OnProcessCallback& callback_)
  {
    CHECK(callback_);
    on_process_callbacks.push_back(callback_);
  }

protected:
  virtual InputConstSharedPtr getInputPacket() = 0;
  virtual OutputConstSharedPtr process(const InputConstSharedPtr& input) = 0;
  virtual bool pushOutputPacket(const OutputConstSharedPtr& output_packet) const = 0;

private:
  void emitProcessCallbacks(const InputConstSharedPtr& input_packet, const OutputConstSharedPtr& output_packet)
  {
    for (OnProcessCallback callbacks : on_process_callbacks)
    {
      callbacks(input_packet, output_packet);
    }
  }

private:
  std::vector<OnProcessCallback> on_process_callbacks;
};

// MultiInputMultiOutput -> abstract class and user must implement the getInputPacket
template <typename INPUT, typename OUTPUT>
class MIMOPipelineModule : public AbstractPipelineModule<INPUT, OUTPUT>
{
public:
  using APM = AbstractPipelineModule<INPUT, OUTPUT>;
  using OutputQueue = ThreadsafeQueue<typename APM::OutputConstSharedPtr>;

  //! Callback used to send data to other pipeline modules, makes use of
  //! shared pointer since the data may be shared between several modules.
  using OutputCallback = std::function<void(const typename APM::OutputConstSharedPtr& output)>;


  MIMOPipelineModule(const std::string& module_name) : AbstractPipelineModule<INPUT, OUTPUT>(module_name)
  {
  }

  virtual void registerOutputQueue(OutputQueue* output_queue_)
  {
    CHECK(output_queue_);
    output_queues.push_back(output_queue_);
  }

protected:
  bool pushOutputPacket(const typename APM::OutputConstSharedPtr& output_packet) const override
  {
    auto tic_callbacks = utils::Timer::tic();
    //! We need to make our packet shared in order to send it to multiple
    //! other modules.
    //! Call all callbacks
    for (OutputQueue* queue : output_queues)
    {
      CHECK(queue);
      queue->push(output_packet);
    }
    static constexpr auto kTimeLimitCallbacks = std::chrono::milliseconds(10);
    auto callbacks_duration = utils::Timer::toc(tic_callbacks);
    LOG_IF(WARNING, callbacks_duration > kTimeLimitCallbacks)
        << "Pushing output packet to queues for module: " << this->module_name_
        << " are taking very long! Current latency: " << callbacks_duration.count() << " ms.";
    return true;
  }

private:
  //! Output callbacks that will be called on each spinOnce if
  //! an output is present.
  std::vector<OutputQueue*> output_queues;
};

/** @brief SIMOPipelineModule Single Input Multiple Output (SIMO) pipeline
 * module.
 * Receives Input packets via a threadsafe queue, and sends output packets
 * to a list of registered callbacks with a specific signature.
 * This is useful when there are multiple modules expecting results from this
 * module.
 */
template <typename INPUT, typename OUTPUT>
class SIMOPipelineModule : public MIMOPipelineModule<INPUT, OUTPUT>
{
public:
  using Base = MIMOPipelineModule<INPUT, OUTPUT>;
  using InputQueue = ThreadsafeQueue<typename Base::InputConstSharedPtr>;
  using OutputQueue = typename Base::OutputQueue;

  SIMOPipelineModule(const std::string& module_name, InputQueue* input_queue_, bool parallel_run = true)
    : MIMOPipelineModule<INPUT, OUTPUT>(module_name), input_queue(CHECK_NOTNULL(input_queue_)), parallel_run_(parallel_run)
  {
  }

protected:
  typename Base::InputConstSharedPtr getInputPacket() override
  {
    typename Base::InputConstSharedPtr input = nullptr;

    bool queue_state;
    if(parallel_run_) {
      queue_state = input_queue->popBlocking(input);
    }
    else {
      queue_state = input_queue->pop(input);
    }

    // bool queue_state = input_queue->pop(input);
    if (queue_state)
    {
      return input;
    }
    else
    {
      LOG_IF(WARNING, !Base::isShutdown() && !hasWork())
          << "Module: " << Base::module_name_ << " - didn't return an output.";
      return nullptr;
    }
  }

  void shutdownQueues() override
  {
    input_queue->shutdown();
  }

  //! Checks if the module has work to do (should check input queues are empty)
  bool hasWork() const override
  {
    return !input_queue->isShutdown() && !input_queue->empty();
  }

private:
  InputQueue* input_queue;
  bool parallel_run_;
};

}  // namespace dyno
