/*
 *   Copyright (c) 2023 Jesse Morris
 *   All rights reserved.
 */
#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/vision/Frame.hpp"

#include "dynosam/dataprovider/DataInterfacePipeline.hpp"

#include <functional>

namespace dyno {


/**
 * @brief A data provider is a module that actually gets the inididual image and/or IMU data from some source and
 * packets it into a Frame and IMU form and sends it the the DataInterfacePipeline via callback functions where the data is
 * synchronized and sent to the frontend
 *
 *
 */
class DataProvider {

//TODO: add gt callback - synchronize to timestamp? or frame?

public:
    DYNO_POINTER_TYPEDEFS(DataProvider)
    DYNO_DELETE_COPY_CONSTRUCTORS(DataProvider)

    using ImageContainerCallback = std::function<void(ImageContainer::Ptr)>;
    using ImuSingleInputCallback = std::function<void(const ImuMeasurement&)>;
    using ImuMultiInputCallback = std::function<void(const ImuMeasurements&)>;

    using GroundTruthPacketCallback = std::function<void(const GroundTruthInputPacket&)>;

    //this one will not guarnatee a binding of bind the data prover module
    DataProvider() = default;
    DataProvider(DataInterfacePipeline* module);
    DataProvider(DataInterfacePipelineImu* module);

    virtual ~DataProvider();


    inline void registerImuSingleCallback(
      const ImuSingleInputCallback& callback) {
        imu_single_input_callback_ = callback;
    }

    inline void registerImuMultiCallback(const ImuMultiInputCallback& callback) {
        imu_multi_input_callback_ = callback;
    }

    inline void registerImageContainerCallback(const ImageContainerCallback& callback) {
        image_container_callback_ = callback;
    }

    inline void registerGroundTruthPacketCallback(const GroundTruthPacketCallback& callback) {
        ground_truth_packet_callback_ = callback;
    }

    /**
     * @brief Spins the dataset for one "step" of the dataset
     *
     * Returns true if the dataset still has data to process
     * @return true
     * @return false
     */
    virtual bool spin() = 0;

    virtual void shutdown();

protected:
    ImageContainerCallback image_container_callback_;
    ImuSingleInputCallback imu_single_input_callback_;
    ImuMultiInputCallback imu_multi_input_callback_;

    GroundTruthPacketCallback ground_truth_packet_callback_;

    // Shutdown switch to stop data provider.
    std::atomic_bool shutdown_ = {false};
};

} //dyno
