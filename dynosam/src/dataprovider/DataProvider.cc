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
#include "dynosam/dataprovider/DataInterfacePipeline.hpp"

#include <functional>

#include <glog/logging.h>

namespace dyno {

DataProvider::DataProvider(DataInterfacePipeline* module) {
    CHECK_NOTNULL(module);
    registerImageContainerCallback(std::bind(&DataInterfacePipeline::fillImageContainerQueue, module, std::placeholders::_1));
    CHECK(image_container_callback_);

}

DataProvider::~DataProvider() {
    shutdown();
}


void DataProvider::setStartingFrame(int starting_frame) {
    if(starting_frame != -1)
        requested_starting_frame_id_ = starting_frame;
}

void DataProvider::setEndingFrame(int ending_frame) {
    if(ending_frame != -1)
        requested_ending_frame_id_ = ending_frame;
}

void DataProvider::reset() {
    is_first_spin_ = true;
    active_frame_id_ = 0;
}

bool DataProvider::spin() {
    if(!isValid()) {
        LOG(WARNING) << "Dataset cannot spin as the derived dataprovider is not ready to run!";
        return false;
    }

    //have to check isValid first otherwise this function may not be valid
    const size_t dataset_size = getDatasetSize();
    LOG(INFO) << dataset_size;

    //initalise and check
    if(is_first_spin_) {
        // by default start dataset at 0
        size_t starting_frame = 0u;
        // by default end dataset at the last frame
        size_t ending_frame = dataset_size;

        if(requested_starting_frame_id_ != -1) {
            starting_frame = requested_starting_frame_id_;
        }

        if(requested_ending_frame_id_ != -1) {
            ending_frame = requested_ending_frame_id_;
        }

        //check ending frame first
        if(ending_frame > dataset_size) {
            throw DataProviderException("Requested ending frame is greater than the size of the dataset (" + std::to_string(ending_frame) + " > " + std::to_string(dataset_size));
        }

        if(starting_frame > ending_frame) {
            throw DataProviderException("Requested starting frame is invalid - either less than 0 or greater than the ending frame (starting frame= " + std::to_string(starting_frame) + ", ending frame= " + std::to_string(ending_frame));
        }

        //update varibles actually used in the spin
        active_frame_id_ = starting_frame;
        ending_frame_id_ = ending_frame;
        is_first_spin_ = false;

        LOG(INFO) << active_frame_id_ << " " << ending_frame_id_;
    }


    if(active_frame_id_ >= ending_frame_id_) {
        LOG_FIRST_N(INFO, 1) << "Finished dataset";
        return false;
    }


    utils::TimingStatsCollector("dataset_spin");
    if(!spinOnce(active_frame_id_)) {
        LOG(ERROR) << "Processing single frame failed at frame id " << active_frame_id_;
        return false;
    }

    active_frame_id_++;
    return true;
}

void DataProvider::shutdown() {
    LOG(INFO) << "Shutting down data provider and associated module";
    shutdown_ = true;
}


DataProvider::DataProvider(DataInterfacePipelineImu* module) : DataProvider(dynamic_cast<DataInterfacePipeline*>(module)) {
    CHECK_NOTNULL(module);
    registerImuMultiCallback(std::bind(
        static_cast<void(DataInterfacePipelineImu::*)(const ImuMeasurements&)>(&DataInterfacePipelineImu::fillImuQueue),
        module,
        std::placeholders::_1));
    registerImuSingleCallback(std::bind(
        static_cast<void(DataInterfacePipelineImu::*)(const ImuMeasurement&)>(&DataInterfacePipelineImu::fillImuQueue),
        module,
        std::placeholders::_1));

    CHECK(imu_multi_input_callback_);
    CHECK(imu_single_input_callback_);
}

} //dyno
