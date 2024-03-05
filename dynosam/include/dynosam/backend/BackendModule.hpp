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

#include "dynosam/visualizer/Visualizer-Definitions.hpp" //for ImageDisplayQueueOptional


namespace dyno {

template<typename DERIVED_INPUT_PACKET, typename MEASUREMENT_TYPE>
struct BackendModuleTraits {
    using DerivedPacketType = DERIVED_INPUT_PACKET;
    using DerivedPacketTypeConstPtr = std::shared_ptr<const DerivedPacketType>;

    using MeasurementType = MEASUREMENT_TYPE;

    static constexpr size_t Dim() {
        return gtsam::traits<MeasurementType>::dimension;
    }
};


//TODO: backend doesnt need camera
//TODO: should do like RGBD/MonoBackendModuleBase... which also knows about the map. Means we can move a lot of common functionality to the base class
/**
 * @brief Base class to actually do processing. Data passed to this module from the frontend
 *
 */
class BackendModule : public ModuleBase<BackendInputPacket, BackendOutputPacket>  {

public:
    DYNO_POINTER_TYPEDEFS(BackendModule)

    using Base = ModuleBase<BackendInputPacket, BackendOutputPacket>;
    using Base::SpinReturn;

    BackendModule(const BackendParams& params, ImageDisplayQueue* display_queue, bool use_logger);
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
    GroundTruthPacketMap gt_packet_map_; //! Updated in the frontend module base via InputCallback (see FrontendModule constructor)

    BackendLogger::UniquePtr logger_;

    // //! Camera related parameters
    // Camera::Ptr camera_;
    // boost::shared_ptr<Camera::CalibrationType> gtsam_calibration_; //Ugh, this version of gtsam still uses some boost

    ImageDisplayQueue* display_queue_{nullptr}; //! Optional display queue

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

    DYNO_POINTER_TYPEDEFS(This)

    using Base::SpinReturn;
    //Define the input type to the derived input type, defined in the MODULE_TRAITS
    //this is the derived Input packet that is passed to the boostrap/nominal Spin Impl functions that must be implemented in the derived
    //class that does the provessing on this module
    using InputConstPtr = DerivedPacketTypeConstPtr;
    using OutputConstPtr = Base::OutputConstPtr;

    BackendModuleType(const BackendParams& params, typename MapType::Ptr map, ImageDisplayQueue* display_queue, bool use_logger = true)
    : Base(params, display_queue, use_logger), map_(CHECK_NOTNULL(map)) {}

    virtual ~BackendModuleType() = default;

    inline const typename MapType::Ptr getMap() {  return map_; }

    //factor graph helper functions
    bool safeAddObjectSmoothingFactor(
        gtsam::Key motion_key_k_1, gtsam::Key motion_key_k, const gtsam::Values& new_values, gtsam::NonlinearFactorGraph& factors)
    {
        CHECK(object_smoothing_noise_);
        CHECK_EQ(object_smoothing_noise_->dim(), 6u);

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

protected:
    virtual SpinReturn boostrapSpinImpl(InputConstPtr input) = 0;
    virtual SpinReturn nominalSpinImpl(InputConstPtr input) = 0;

    typename MapType::Ptr map_;

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




} //dyno
