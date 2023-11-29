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

#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/common/Camera.hpp"

#include "dynosam/frontend/MonoInstance-Definitions.hpp" // for MonocularInstanceOutputPacket

namespace dyno {

class MonoBackendModule : public BackendModule {

public:
    MonoBackendModule(const BackendParams& backend_params, Camera::Ptr camera);

    using SpinReturn = BackendModule::SpinReturn;


private:
    SpinReturn boostrapSpin(BackendInputPacket::ConstPtr input) override;
    SpinReturn nominalSpin(BackendInputPacket::ConstPtr input) override;

    SpinReturn monoBoostrapSpin(MonocularInstanceOutputPacket::ConstPtr input);
    SpinReturn monoNominalSpin(MonocularInstanceOutputPacket::ConstPtr input);

    //updates the data structures relevant to smart factors, label and type maps
    void updateStaticObservations(const StatusKeypointMeasurements& measurements, const FrameId frame_id);

    void setFactorParams(const BackendParams& backend_params);



private:
    Camera::Ptr camera_;
    boost::shared_ptr<Camera::CalibrationType> gtsam_calibration_; //Ugh, this version of gtsam still uses some boost

    // //data structures to hold measurements and landmarks
    // TrackletIdSmartFactorMap tracklet_smart_factor_map_;
    // TrackletIdLabelMap tracklet_to_label_map_;
    // TrackletIdToTypeMap tracklet_to_type_map_;
    // TrackletIdSlotMap tracklet_to_slot_; //also indicates if tracklet factor is in the map (either as smart or projection factor). -1 means not in the map
    ProjectionFactorStatusMap projection_factor_map_;


    //params for factors
    SmartProjectionFactorParams static_projection_params_; //! Projection factor params for static points
    gtsam::SharedNoiseModel static_smart_noise_; //! Projection factor noise for static points



};


} //dyno
