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
#include "dynosam/common/ModuleBase.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/utils/SafeCast.hpp"

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendOutputPacket.hpp"
#include "dynosam/backend/Optimizer.hpp"


#include "dynosam/visualizer/Visualizer-Definitions.hpp" //for ImageDisplayQueueOptional


namespace dyno {

template<typename DERIVED_INPUT_PACKET, typename MEASUREMENT_TYPE, typename BASE_INPUT_PACKET = BackendInputPacket>
struct BackendModuleTraits {
    using DerivedPacketType = DERIVED_INPUT_PACKET;
    using DerivedPacketTypeConstPtr = std::shared_ptr<const DerivedPacketType>;

    using BasePacketType = BackendInputPacket;
    //BasePacketType is the type that gets passed to the module via the pipeline and must be a base class
    //since we pass data along the pipelines via poniters
    static_assert(std::is_base_of_v<BasePacketType, DerivedPacketType>);

    //Must be a gtsam value type and meet all the concept criteria for that
    using MeasurementType = MEASUREMENT_TYPE;

    static constexpr size_t Dim() {
        return gtsam::traits<MeasurementType>::dimension;
    }
};


/**
 * @brief Base class to actually do processing. Data passed to this module from the frontend
 *
 */
class BackendModule : public ModuleBase<BackendInputPacket, BackendOutputPacket>  {

public:
    DYNO_POINTER_TYPEDEFS(BackendModule)

    using Base = ModuleBase<BackendInputPacket, BackendOutputPacket>;
    using Base::SpinReturn;

    BackendModule(const BackendParams& params, BackendLogger::UniquePtr logger, ImageDisplayQueue* display_queue);
    virtual ~BackendModule() = default;


protected:

    //called in ModuleBase immediately before the spin function is called
    virtual void validateInput(const BackendInputPacket::ConstPtr& input) const override;
    void setFactorParams(const BackendParams& backend_params);

    // Redefine base input since these will be cast up by the BackendModuleType class
    // to a new type which we want to refer to as the input type
    // BaseInput is a ConstPtr to the type defined by BackendInputPacket
    using BaseInputConstPtr =  Base::InputConstPtr;
    using BaseInput = Base::Input;


protected:
    const BackendParams base_params_;
    //NOTE: this is copied directly from the frontend module.
    GroundTruthPacketMap gt_packet_map_; //! Updated in the backend module base via InputCallback (see BackendModule constructor).

    BackendLogger::UniquePtr logger_{nullptr};
    ImageDisplayQueue* display_queue_{nullptr}; //! Optional display queue

    BackendSpinState spin_state_; //! Spin state of the backend. Updated in the backend module base via InputCallback (see BackendModule constructor).

    // //! Camera related parameters
    // Camera::Ptr camera_;
    // boost::shared_ptr<Camera::CalibrationType> gtsam_calibration_; //Ugh, this version of gtsam still uses some boost


    //params for factors
    SmartProjectionFactorParams static_projection_params_; //! Projection factor params for static points
    gtsam::SharedNoiseModel static_pixel_noise_; //! 2d isotropic pixel noise on static points
    gtsam::SharedNoiseModel dynamic_pixel_noise_; //! 2d isotropic pixel noise on dynamic points
    gtsam::SharedNoiseModel static_projection_noise_; //! Projection factor noise for static points
    gtsam::SharedNoiseModel odometry_noise_; //! Between factor noise for between two consequative poses
    gtsam::SharedNoiseModel initial_pose_prior_;
    gtsam::SharedNoiseModel landmark_motion_noise_; //! Noise on the landmark tenrary factor
    gtsam::SharedNoiseModel object_smoothing_noise_; //! Contant velocity noise model between motions

private:

};

template<class MODULE_TRAITS>
class BackendModuleType : public BackendModule {

public:
    using ModuleTraits = MODULE_TRAITS;
    //A Dervied BackedInputPacket type (e.g. RGBDOutputPacketType)
    using DerivedPacketType = typename ModuleTraits::DerivedPacketType;
    using DerivedPacketTypeConstPtr = typename ModuleTraits::DerivedPacketTypeConstPtr;
    using MeasurementType = typename ModuleTraits::MeasurementType;
    using This = BackendModuleType<ModuleTraits>;
    using Base = BackendModule;

    using MapType = Map<MeasurementType>;
    using OptimizerType = Optimizer<MeasurementType>;

    DYNO_POINTER_TYPEDEFS(This)

    using Base::SpinReturn;
    //Define the input type to the derived input type, defined in the MODULE_TRAITS
    //this is the derived Input packet that is passed to the boostrap/nominal Spin Impl functions that must be implemented in the derived
    //class that does the provessing on this module
    using InputConstPtr = DerivedPacketTypeConstPtr;
    using OutputConstPtr = Base::OutputConstPtr;

    BackendModuleType(const BackendParams& params, typename MapType::Ptr map, typename OptimizerType::Ptr optimizer, BackendLogger::UniquePtr logger, ImageDisplayQueue* display_queue)
    : Base(params, std::move(logger), display_queue), map_(CHECK_NOTNULL(map)), optimizer_(CHECK_NOTNULL(optimizer)) {}

    virtual ~BackendModuleType() {
        // if(logger_) {
        //     //write out logging -> do this here as we dont want to this type of logging per frame
        //     //we can just wait till the end and write out everything still in the map
        //     //the get landmarks call will have more than just the values in the map
        //     //collect all tracklets and their ids
        //     gtsam::FastMap<TrackletId, ObjectId> tracklet_object_id_mapping;
        //     for(const auto& [tracklet_id, lmk_node] : map_->getLandmarks()) {
        //         tracklet_object_id_mapping.insert2(tracklet_id, lmk_node->getObjectId());
        //     }
        //     logger_->logTrackletIdToObjectId(tracklet_object_id_mapping);
        // }

    }

    inline const typename MapType::Ptr getMap() {  return map_; }
    inline const typename OptimizerType::Ptr getOptimzier() {  return optimizer_; }

    //factor graph helper functions
    bool safeAddObjectSmoothingFactor(
        gtsam::Key motion_key_k_1, gtsam::Key motion_key_k, const gtsam::Values& new_values, gtsam::NonlinearFactorGraph& factors) const
    {
        CHECK(object_smoothing_noise_);
        CHECK_EQ(object_smoothing_noise_->dim(), 6u);

        {
            ObjectId object_label_k_1, object_label_k;
            FrameId frame_id_k_1, frame_id_k;
            CHECK(reconstructMotionInfo(motion_key_k_1, object_label_k_1, frame_id_k_1));
            CHECK(reconstructMotionInfo(motion_key_k, object_label_k, frame_id_k));
            CHECK_EQ(object_label_k_1, object_label_k);
            CHECK_EQ(frame_id_k_1 + 1, frame_id_k); //assumes consequative frames
        }

        //if the motion key at k (motion from k-1 to k), and key at k-1 (motion from k-2 to k-1)
        //exists in the map or is about to exist via new values, add the smoothing factor
        if(map_->exists(motion_key_k_1, new_values) && map_->exists(motion_key_k, new_values)) {
            factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                motion_key_k_1,
                motion_key_k,
                gtsam::Pose3::Identity(),
                object_smoothing_noise_
            );
            return true;
        }

        //smoothing factor not added
        return false;
    }

    // //gets all object poses. This iterates over each object in the map
    // //and recomputes all the object poses given the values in the map
    // ObjectPoseMap getObjectPoses(bool init_translation_from_gt) const;


    // void logBackendFromMap(FrameId frame_k);

protected:
    virtual SpinReturn boostrapSpinImpl(InputConstPtr input) = 0;
    virtual SpinReturn nominalSpinImpl(InputConstPtr input) = 0;

    typename MapType::Ptr map_;
    typename OptimizerType::Ptr optimizer_;

private:
    SpinReturn boostrapSpin(Base::BaseInputConstPtr base_input) override {
        return boostrapSpinImpl(attemptCast(base_input));
    }

    SpinReturn nominalSpin(Base::BaseInputConstPtr base_input) override {
        return nominalSpinImpl(attemptCast(base_input));
    }

    DerivedPacketTypeConstPtr attemptCast(Base::BaseInputConstPtr base_input) {
        DerivedPacketTypeConstPtr deriverd_input = safeCast<Base::BaseInput, DerivedPacketType>(base_input);

        checkAndThrow((bool)deriverd_input, "Failed to cast " + type_name<Base::BaseInput>() + " to " + type_name<DerivedPacketType>() + " in BackendModuleType");
        return deriverd_input;
    }


};

// template<class MODULE_TRAITS>
// ObjectPoseMap BackendModuleType<MODULE_TRAITS>::getObjectPoses(bool init_translation_from_gt) const {
//     const ObjectIds object_ids = map_->getObjectIds();
//     ObjectPoseMap object_pose_map;

//     for(auto object_id : object_ids) {
//         const auto& object_node = map_->getObject(object_id);
//         object_pose_map.insert2(
//             object_id, object_node->computePoseMap(gt_packet_map_, init_translation_from_gt));
//     }
//     return object_pose_map;
// }

// template<class MODULE_TRAITS>
// void BackendModuleType<MODULE_TRAITS>::logBackendFromMap(FrameId frame_k) {
//     if(!logger_) {
//         return;
//     }

//     //TODO: only composed poses?
//     auto composed_poses = getObjectPoses(FLAGS_init_object_pose_from_gt);

//     //get MotionestimateMap
//     const MotionEstimateMap motions = map_->getMotionEstimates(frame_k);
//     logger_->logObjectMotion(gt_packet_map_, frame_k, motions);

//     StateQuery<gtsam::Pose3> X_k_query = map_->getPoseEstimate(frame_k);

//     if(X_k_query) {
//         std::optional<gtsam::Pose3> X_k_1_query;
//         if(frame_k > 0) {
//             X_k_1_query = map_->getPoseEstimate(frame_k - 1u);
//         }
//         logger_->logCameraPose(gt_packet_map_, frame_k, X_k_query.get(), X_k_1_query);
//     }
//     else {
//         LOG(WARNING) << "Could not log camera pose estimate at frame " << frame_k;
//     }


//     logger_->logObjectPose(gt_packet_map_, frame_k, composed_poses);
// }



} //dyno
