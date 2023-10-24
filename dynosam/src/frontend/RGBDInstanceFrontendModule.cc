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

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include "dynosam/frontend/RGBDInstancePacket.hpp"
#include "dynosam/utils/SafeCast.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {


RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera)
    : FrontendModule(frontend_params), camera_(camera)
{
    CHECK_NOTNULL(camera_);
    tracker_ = std::make_unique<FeatureTracker>(frontend_params, camera_);
}

bool RGBDInstanceFrontendModule::validateImageContainer(const ImageContainer::Ptr& image_container) const {
    return image_container->hasDepth() && image_container->hasSemanticMask();
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::boostrapSpin(FrontendInputPacketBase::ConstPtr input) {
    // FrontendInputPacketBase input_v = *input;
    // RGBDInstancePacket::Ptr image_packet = safeCast<InputImagePacketBase, RGBDInstancePacket>(input_v.image_packet_);
    // CHECK(image_packet);

    // size_t n_optical_flow, n_new_tracks;

    // InputImages tracking_images(image_packet->rgb_, image_packet->optical_flow_, image_packet->instance_mask_);
    // tracker_->track(image_packet->frame_id_, image_packet->timestamp_, tracking_images, n_optical_flow, n_new_tracks);

    LOG(INFO) << "In RGBD instance module frontend boostrap";
    return {State::Nominal, nullptr};
}


FrontendModule::SpinReturn RGBDInstanceFrontendModule::nominalSpin(FrontendInputPacketBase::ConstPtr input) {
    // FrontendInputPacketBase input_v = *input;
    // RGBDInstancePacket::Ptr image_packet = safeCast<InputImagePacketBase, RGBDInstancePacket>(input_v.image_packet_);
    // CHECK(image_packet);


    // LOG(INFO) << "In RGBD instance module frontend nominal";

    // size_t n_optical_flow, n_new_tracks;
    // InputImages tracking_images(image_packet->rgb_, image_packet->optical_flow_, image_packet->instance_mask_);
    // tracker_->track(image_packet->frame_id_, image_packet->timestamp_, tracking_images, n_optical_flow, n_new_tracks);


    // // cv::imshow("RGB", image_packet->rgb_);
    // // cv::waitKey(1);
    // auto output = std::make_shared<FrontendOutputPacketBase>() ;
    // output->input = input;

    return {State::Nominal, nullptr};
}

} //dyno
