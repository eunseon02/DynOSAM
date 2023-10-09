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

#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/dataprovider/DataProviderModule.hpp"

#include <functional>

#include <glog/logging.h>

namespace dyno {

DataProvider::DataProvider(DataProviderModule* module) {
    CHECK_NOTNULL(module);
    registerFrameCallback(std::bind(&DataProviderModule::fillFrameQueue, module, std::placeholders::_1));

    shutdown_data_provider_module_callback_ = std::bind(&DataProviderModule::shutdown, module);

    CHECK(frame_input_callback_);
    CHECK(shutdown_data_provider_module_callback_);

}

DataProvider::~DataProvider() {
    shutdown();
}

void DataProvider::shutdown() {
    LOG(INFO) << "Shutting down data provider and associated module";
    shutdown_ = true;
    shutdown_data_provider_module_callback_();
}


DataProvider::DataProvider(DataProviderModuleImu* module) : DataProvider(dynamic_cast<DataProviderModule*>(module)) {
    CHECK_NOTNULL(module);
    registerImuMultiCallback(std::bind(
        static_cast<void(DataProviderModuleImu::*)(const ImuMeasurements&)>(&DataProviderModuleImu::fillImuQueue),
        module,
        std::placeholders::_1));
    registerImuSingleCallback(std::bind(
        static_cast<void(DataProviderModuleImu::*)(const ImuMeasurement&)>(&DataProviderModuleImu::fillImuQueue),
        module,
        std::placeholders::_1));

    CHECK(imu_multi_input_callback_);
    CHECK(imu_single_input_callback_);
}

} //dyno
