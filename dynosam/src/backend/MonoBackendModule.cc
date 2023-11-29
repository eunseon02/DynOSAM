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

#include "dynosam/backend/MonoBackendModule.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/frontend/MonoInstance-Definitions.hpp"

#include <glog/logging.h>

namespace dyno {

MonoBackendModule::MonoBackendModule(const BackendParams& backend_params, Camera::Ptr camera)
    :   BackendModule(backend_params),
        camera_(CHECK_NOTNULL(camera))
{

    const auto& camera_params = camera_->getParams();
    gtsam_calibration_ = boost::make_shared<Camera::CalibrationType>(camera_params.constructGtsamCalibration<Camera::CalibrationType>());

    CHECK(gtsam_calibration_);
    setFactorParams(backend_params);
}


MonoBackendModule::SpinReturn MonoBackendModule::boostrapSpin(BackendInputPacket::ConstPtr input) {
    MonocularInstanceOutputPacket::ConstPtr mono_output = safeCast<BackendInputPacket, MonocularInstanceOutputPacket>(input);
    checkAndThrow((bool)mono_output, "Failed to cast BackendInputPacket to MonocularInstanceOutputPacket in MonoBackendModule");

    return monoBoostrapSpin(mono_output);

}
MonoBackendModule::SpinReturn MonoBackendModule::nominalSpin(BackendInputPacket::ConstPtr input) {
    MonocularInstanceOutputPacket::ConstPtr mono_output = safeCast<BackendInputPacket, MonocularInstanceOutputPacket>(input);
    checkAndThrow((bool)mono_output, "Failed to cast BackendInputPacket to MonocularInstanceOutputPacket in MonoBackendModule");

    return monoNominalSpin(mono_output);
}


MonoBackendModule::SpinReturn MonoBackendModule::monoBoostrapSpin(MonocularInstanceOutputPacket::ConstPtr input) {

    //1. Triangulate initial map (static)
    //2. Triangulation function for object points (requires two frames min)
    CHECK(input);
    //check cameras the same?
    updateStaticObservations(input->static_keypoint_measurements_, input->getFrameId());
    return {State::Nominal, nullptr};
}

MonoBackendModule::SpinReturn MonoBackendModule::monoNominalSpin(MonocularInstanceOutputPacket::ConstPtr input) {
    CHECK(input);
    updateStaticObservations(input->static_keypoint_measurements_, input->getFrameId());
    return {State::Nominal, nullptr};
}


void MonoBackendModule::updateStaticObservations(const StatusKeypointMeasurements& measurements, const FrameId frame_id) {
    for(const StatusKeypointMeasurement& static_measurement : measurements) {
        const KeypointStatus& status = static_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = static_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::STATIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        if(!projection_factor_map_.exists(tracklet_id)) {
            //if the TrackletIdToTypeMap does not have this tracklet, then it should be the first time we have seen
            //it and therefore, should not be in any of the other data structures
            SmartProjectionFactor::shared_ptr smart_factor =
                factor_graph_tools::constructSmartProjectionFactor(
                    static_smart_noise_,
                    gtsam_calibration_,
                    static_projection_params_,
                    kp,
                    frame_id
                );

            ProjectionFactorStatus projection_status(tracklet_id, smart_factor, object_id);
            projection_factor_map_.add(smart_factor);
        }

        //sanity check that the object label is still the same for the tracked object
        CHECK_EQ(object_id, tracklet_to_label_map_.at(tracklet_id).object_id);

        const ProjectionFactorType factor_type = projection_factor_map_.getFactorTypeProperty(tracklet_id);

        // if(factor_type ==  ProjectionFactorType::SMART) {
        //     SmartProjectionFactor::shared_ptr smart_factor = tracklet_smart_factor_map_.at(tracklet_id);
        //     smart_factor->add(kp, gtsam::Symbol(kPoseSymbolChar, frame_id));
        // }
        // else {

        // }

    }
}


void MonoBackendModule::setFactorParams(const BackendParams& backend_params) {
    //set static projection smart noise
    static_smart_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, backend_params.smart_projection_noise_sigma_);
    CHECK(static_smart_noise_);
}


} //dyno
