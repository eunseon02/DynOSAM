/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/backend/RGBDBackendModule.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"

#include <glog/logging.h>

namespace dyno {

RGBDBackendModule::RGBDBackendModule(const BackendParams& backend_params, Camera::Ptr camera, Map3d::Ptr map, ImageDisplayQueue* display_queue)
    : BackendModule(backend_params, camera, display_queue), map_(CHECK_NOTNULL(map)) {}

RGBDBackendModule::~RGBDBackendModule() {}

RGBDBackendModule::SpinReturn
RGBDBackendModule::boostrapSpin(BackendInputPacket::ConstPtr input) {
    RGBDInstanceOutputPacket::ConstPtr rgbd_output = safeCast<BackendInputPacket, RGBDInstanceOutputPacket>(input);
    checkAndThrow((bool)rgbd_output, "Failed to cast BackendInputPacket to RGBDInstanceOutputPacket in RGBDBackendModule");

    return rgbdBoostrapSpin(rgbd_output);
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::nominalSpin(BackendInputPacket::ConstPtr input) {
    RGBDInstanceOutputPacket::ConstPtr rgbd_output = safeCast<BackendInputPacket, RGBDInstanceOutputPacket>(input);
    checkAndThrow((bool)rgbd_output, "Failed to cast BackendInputPacket to RGBDInstanceOutputPacket in RGBDBackendModule");

    return rgbdNominalSpin(rgbd_output);
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::rgbdBoostrapSpin(RGBDInstanceOutputPacket::ConstPtr input) {

    {
        utils::TimingStatsCollector("map.update_observations");
        map_->updateObservations(input->static_landmarks_);
        map_->updateObservations(input->dynamic_landmarks_);
    }

    return {State::Nominal, nullptr};
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::rgbdNominalSpin(RGBDInstanceOutputPacket::ConstPtr input) {

    {
        utils::TimingStatsCollector("map.update_observations");
        map_->updateObservations(input->static_landmarks_);
        map_->updateObservations(input->dynamic_landmarks_);
    }

    return {State::Nominal, nullptr};
}



} //dyno
