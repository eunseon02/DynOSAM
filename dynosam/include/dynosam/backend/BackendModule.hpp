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
#include "dynosam/frontend/vision/Frame.hpp"

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendOutputPacket.hpp"

#include "dynosam/visualizer/Visualizer-Definitions.hpp" //for ImageDisplayQueueOptional


namespace dyno {

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


    BackendModule(const BackendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue);
    ~BackendModule() = default;


protected:
    void validateInput(const BackendInputPacket::ConstPtr& input) const override;
    void setFactorParams(const BackendParams& backend_params);


protected:
    const BackendParams base_params_;

    //NOTE: this is copied directly from the frontend module.
    GroundTruthPacketMap gt_packet_map_; //! Updated in the frontend module base via InputCallback (see FrontendModule constructor)

    //! Camera related parameters
    Camera::Ptr camera_;
    boost::shared_ptr<Camera::CalibrationType> gtsam_calibration_; //Ugh, this version of gtsam still uses some boost

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



};


} //dyno
