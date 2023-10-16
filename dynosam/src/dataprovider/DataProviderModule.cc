/*
 *   Copyright (c) 2023 Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/dataprovider/DataProviderModule.hpp"
#include "dynosam/utils/Numerical.hpp"

#include <glog/logging.h>


namespace dyno {

DataProviderModule::DataProviderModule(const std::string& module_name) : MIMO(module_name) {}


void DataProviderModule::shutdownQueues() {
    packet_queue_.shutdown();
    //call the virtual shutdown method for the derived dataprovider module
    this->onShutdown();
}


FrontendInputPacketBase::ConstPtr DataProviderModule::getInputPacket() {
  if(isShutdown()) {
    return nullptr;
  }

  InputImagePacketBase::Ptr packet = nullptr;
  //TODO: shoudl not pop blocking if threaded!!
  bool queue_state = packet_queue_.popBlocking(packet);

  //TODO: gt?

  if(!queue_state) {
     LOG(WARNING)
        << "Module: " << MIMO::module_name_ << " - queue is down";
        return nullptr;
  }

  CHECK(packet);
  return std::make_shared<FrontendInputPacketBase>(packet);
}

bool DataProviderModule::hasWork() const {
  return !packet_queue_.empty() && !packet_queue_.isShutdown();
}



bool DataProviderModuleImu::getTimeSyncedImuMeasurements(
    const Timestamp& timestamp,
    ImuMeasurements* imu_meas) {
  CHECK_NOTNULL(imu_meas);
  CHECK_LT(timestamp_last_sync_, timestamp)
      << "Timestamps out of order:\n"
      << " - Last Frame Timestamp = " << timestamp_last_sync_ << '\n'
      << " - Current Timestamp = " << timestamp;

  if (imu_buffer_.size() == 0) {
    VLOG(1) << "No IMU measurements available yet, dropping this frame.";
    return false;
  }

  // Extract imu measurements between consecutive frames.
  if (dyno::fpEqual(timestamp_last_sync_, 0.0)) {
    // TODO(Toni): wouldn't it be better to get all IMU measurements up to
    // this
    // timestamp? We should add a method to the IMU buffer for that.
    VLOG(1) << "Skipping first frame, because we do not have a concept of "
               "a previous frame timestamp otherwise.";
    timestamp_last_sync_ = timestamp;
    return false;
  }

  ThreadsafeImuBuffer::QueryResult query_result =
      ThreadsafeImuBuffer::QueryResult::kDataNeverAvailable;
  bool log_error_once = true;
  while (
      !MIMO::isShutdown() &&
      (query_result = imu_buffer_.getImuDataInterpolatedUpperBorder(
           timestamp_last_sync_,
           timestamp,
           &imu_meas->timestamps_,
           &imu_meas->acc_gyr_)) !=
          ThreadsafeImuBuffer::QueryResult::kDataAvailable) {
    VLOG(1) << "No IMU data available. Reason:\n";
    switch (query_result) {
      case ThreadsafeImuBuffer::QueryResult::kDataNotYetAvailable: {
        if (log_error_once) {
          LOG(WARNING) << "Waiting for IMU data...";
          log_error_once = false;
        }
        continue;
      }
      case ThreadsafeImuBuffer::QueryResult::kQueueShutdown: {
        LOG(WARNING)
            << "IMU buffer was shutdown. Shutting down DataProviderModule.";
        MIMO::shutdown();
        return false;
      }
      case ThreadsafeImuBuffer::QueryResult::kDataNeverAvailable: {
        LOG(WARNING)
            << "Asking for data before start of IMU stream, from timestamp: "
            << timestamp_last_sync_ << " to timestamp: " << timestamp;
        // Ignore frames that happened before the earliest imu data
        timestamp_last_sync_ = timestamp;
        return false;
      }
      case ThreadsafeImuBuffer::QueryResult::
          kTooFewMeasurementsAvailable: {
        LOG(WARNING) << "No IMU measurements here, and IMU data stream already "
                        "passed this time region"
                     << "from timestamp: " << timestamp_last_sync_
                     << " to timestamp: " << timestamp;
        return false;
      }
      case ThreadsafeImuBuffer::QueryResult::kDataAvailable: {
        LOG(FATAL) << "We should not be inside this while loop if IMU data is "
                      "available...";
        return false;
      }
    }
  }
  timestamp_last_sync_ = timestamp;

  VLOG(10) << "////////////////////////////////////////// Creating packet!\n"
           << "STAMPS IMU rows : \n"
           << imu_meas->timestamps_.rows() << '\n'
           << "STAMPS IMU cols : \n"
           << imu_meas->timestamps_.cols() << '\n'
           << "STAMPS IMU: \n"
           << imu_meas->timestamps_ << '\n'
           << "ACCGYR IMU rows : \n"
           << imu_meas->acc_gyr_.rows() << '\n'
           << "ACCGYR IMU cols : \n"
           << imu_meas->acc_gyr_.cols() << '\n'
           << "ACCGYR IMU: \n"
           << imu_meas->acc_gyr_;

  return true;
}

} //dyno
