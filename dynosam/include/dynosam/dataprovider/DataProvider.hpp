/*
 *   Copyright (c) 2023 Jesse Morris
 *   All rights reserved.
 */
#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/dataprovider/DataInterfacePipeline.hpp"

#include <functional>

namespace dyno {


class DataProviderException : public DynosamException {
public:
    DataProviderException(const std::string& what) : DynosamException(what) {}
};

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
     * @brief Virtual Image Container Preprocessor function that is registered to the data interface pipeline
     *
     * By default does nothing and just returns the argument.
     * This function is called (via callback) when a ImageContainer is sent to the pipeline via DataProvider::image_container_callback_
     * and enables each data-provider to implement their own data-preprocessing as necessary
     *
     * @param image_container
     * @return ImageContainer::Ptr
     */
    inline virtual ImageContainer::Ptr imageContainerPreprocessor(ImageContainer::Ptr image_container) {
        return image_container;
    }


    //TODO: the set starting frame stuff does not make sense to be here as we could not do this for a
    //datset that is online - instead put this intop the dataset provider class... This should just have a virtual spin
    //which is totally generic. Just changing for the RSS stuff....
    /**
     * @brief Spins the dataset for one "step" of the dataset
     *
     * Returns true if the dataset still has data to process
     * @return true
     * @return false
     */
    bool spin();

    //ignore if -1. How to reset to original starting frame? we only have -1 so that in main we can call setStartingFrame without doing any checks
    void setStartingFrame(int starting_frame);
    //ignore if -1
    void setEndingFrame(int ending_frame);

    /**
     * @brief Resets the dataset to the "start", such that the next call to spin will
     * begin the dataset again from the starting frame id. The starting/ending id will return to
     * whatever the last call to setStartingFrame/setEndingFrame set the values to, or default
     * if they have not been called
     *
     * Whatever changes to the dataset, ie. updating the starting and ending frames will take affect
     * after reset is called
     *
     */
    void reset();

    virtual void shutdown();

    //is ready to run
    virtual bool isValid() const = 0;
    virtual size_t getDatasetSize() const = 0;
    virtual bool spinOnce(size_t frame_id) = 0;



protected:
    ImageContainerCallback image_container_callback_;
    ImuSingleInputCallback imu_single_input_callback_;
    ImuMultiInputCallback imu_multi_input_callback_;

    GroundTruthPacketCallback ground_truth_packet_callback_;

    // Shutdown switch to stop data provider.
    std::atomic_bool shutdown_ = {false};

private:
    size_t active_frame_id_ = 0u;
    size_t ending_frame_id_ = 0u; //Start at 0, but this will be set upon the first call to spin()

    int requested_starting_frame_id_{-1};
    int requested_ending_frame_id_{-1};

    bool is_first_spin_{true};
};

} //dyno
