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

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {



//inherit from to add more IMAGE data
struct InputImagePacketBase {
    DYNO_POINTER_TYPEDEFS(InputImagePacketBase)

    const FrameId frame_id_;
    const Timestamp timestamp_;
    const cv::Mat rgb_; //could be either rgb or greyscale?
    const cv::Mat optical_flow_;

    InputImagePacketBase(const FrameId frame_id, const Timestamp timestamp, const cv::Mat& rgb, const cv::Mat& optical_flow)
        :   frame_id_(frame_id), timestamp_(timestamp), rgb_(rgb), optical_flow_(optical_flow)
    {
        CHECK(rgb_.type() == CV_8UC1 || rgb_.type() == CV_8UC3) << "The provided rgb image is not grayscale or rgb";
        CHECK(!rgb.empty()) << "The provided rgb image is empty!";

        CHECK(optical_flow_.type() == CV_32F) << "The provided optical flow image is not of datatype CV_32F";
        CHECK(!optical_flow_.empty()) << "The provided optical flow image is empty!";
    }

    virtual ~InputImagePacketBase() = default;
};


//inherit to add more sensor data (eg imu)
struct FrontendInputPacketBase {
    DYNO_POINTER_TYPEDEFS(FrontendInputPacketBase)

    InputImagePacketBase::Ptr image_packet_;
    GroundTruthInputPacket::Optional optional_gt_;

    FrontendInputPacketBase() : image_packet_(nullptr), optional_gt_(std::nullopt) {}

    FrontendInputPacketBase(InputImagePacketBase::Ptr image_packet, GroundTruthInputPacket::Optional optional_gt = std::nullopt)
    : image_packet_(CHECK_NOTNULL(image_packet)), optional_gt_(optional_gt)
    {
        if(optional_gt) {
            CHECK_EQ(optional_gt_->timestamp, image_packet_->timestamp_);
        }
    }


    virtual ~FrontendInputPacketBase() = default;
};




} //dyno
