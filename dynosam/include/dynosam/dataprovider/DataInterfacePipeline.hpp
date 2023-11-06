/*
 *   Copyright (c) ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.
 */
#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam/pipeline/ThreadSafeQueue.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/imu/ThreadSafeImuBuffer.hpp"

namespace dyno {

//TODO: gt
/**
 * @brief Takes data, synchronizes it and sends it to the output queue which should be connected to the frontend
 * User needs to implement
 * InputConstSharedPtr getInputPacket() = 0;
 * that takes data from the internal queues and processes them
 */
class DataInterfacePipeline : public MIMOPipelineModule<FrontendInputPacketBase, FrontendInputPacketBase> {

public:
    DYNO_POINTER_TYPEDEFS(DataInterfacePipeline)

    using MIMO =
      MIMOPipelineModule<FrontendInputPacketBase, FrontendInputPacketBase>;
    using OutputQueue = typename MIMO::OutputQueue;

    DataInterfacePipeline(bool parallel_run = false);
    virtual ~DataInterfacePipeline() = default;

    //TODO: later should be vision only module
    virtual FrontendInputPacketBase::ConstPtr getInputPacket() override;

    //expects input packet
    virtual inline void fillImageContainerQueue(ImageContainer::Ptr image_container) {
        packet_queue_.push(image_container);
    }

    virtual inline void addGroundTruthPacket(const GroundTruthInputPacket& gt_packet) {
        ground_truth_packets_[gt_packet.frame_id_] = gt_packet;
    }

protected:
    virtual void onShutdown() {}


private:
    inline MIMO::OutputConstSharedPtr process(const MIMO::InputConstSharedPtr& input) override {
        return input;
    }

    virtual bool hasWork() const override;

    //! Called when general shutdown of PipelineModule is triggered.
    void shutdownQueues() override;

protected:
    ThreadsafeQueue<ImageContainer::Ptr> packet_queue_;
    std::atomic_bool parallel_run_;

    std::map<FrameId, GroundTruthInputPacket> ground_truth_packets_;

};


class DataInterfacePipelineImu : public DataInterfacePipeline {

public:
    DataInterfacePipelineImu(const std::string& module_name);

    virtual inline void fillImuQueue(const ImuMeasurements& imu_measurements) {
        imu_buffer_.addMeasurements(imu_measurements.timestamps_,
                                            imu_measurements.acc_gyr_);
    }

    virtual inline void fillImuQueue(const ImuMeasurement& imu_measurement) {
        imu_buffer_.addMeasurement(imu_measurement.timestamp_,
                                            imu_measurement.acc_gyr_);
    }

protected:
    /**
     * @brief getTimeSyncedImuMeasurements Time synchronizes the IMU buffer
     * with the given timestamp (this is typically the timestamp of a left img)
     *
     * False if synchronization failed, true otherwise.
     *
     * @param timestamp const Timestamp& for the IMU data to query
     * @param imu_meas ImuMeasurements* IMU measurements to be populated and returned
     * @return true
     * @return false
     */
    bool getTimeSyncedImuMeasurements(const Timestamp& timestamp,
                                    ImuMeasurements* imu_meas);

private:
    ThreadsafeImuBuffer imu_buffer_;

    Timestamp timestamp_last_sync_ {0}; //! Time of last IMU synchronisation (and represents the time of the last frame)

};


} //dyno
