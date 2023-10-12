/*
 *   Copyright (c) 2023 Jesse Morris
 *   All rights reserved.
 */
#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/vision/Frame.hpp"

#include "dynosam/dataprovider/DataProviderModule.hpp"

#include <functional>

namespace dyno {


/**
 * @brief A data provider is a module that actually gets the inididual image and/or IMU data from some source and
 * packets it into a Frame and IMU form and sends it the the DataProviderModule via callback functions where the data is
 * synchronized and sent to the frontend
 *
 *
 */
class DataProvider {

//TODO: add gt callback - synchronize to timestamp? or frame?

public:
    DYNO_POINTER_TYPEDEFS(DataProvider)
    DYNO_DELETE_COPY_CONSTRUCTORS(DataProvider)

    using InputImagePacketInputCallback = std::function<void(InputImagePacketBase::Ptr)>;
    using ImuSingleInputCallback = std::function<void(const ImuMeasurement&)>;
    using ImuMultiInputCallback = std::function<void(const ImuMeasurements&)>;

    //this one will not guarnatee a binding of bind the data prover module
    DataProvider() = default;
    DataProvider(DataProviderModule* module);
    DataProvider(DataProviderModuleImu* module);

    virtual ~DataProvider();


    inline void registerImuSingleCallback(
      const ImuSingleInputCallback& callback) {
        imu_single_input_callback_ = callback;
    }

    inline void registerImuMultiCallback(const ImuMultiInputCallback& callback) {
        imu_multi_input_callback_ = callback;
    }

    inline void registerInputImagesCallback(const InputImagePacketInputCallback& callback) {
        image_input_callback_ = callback;
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
    InputImagePacketInputCallback image_input_callback_;
    ImuSingleInputCallback imu_single_input_callback_;
    ImuMultiInputCallback imu_multi_input_callback_;

    // Shutdown switch to stop data provider.
    std::atomic_bool shutdown_ = {false};
};

} //dyno
