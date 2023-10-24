/*
 *   Copyright (c) 2023 Jesse Morris (jesse.morris@sydney.edu.au)
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
#include "dynosam/frontend/vision/Feature.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"


namespace dyno {

//should this be here?
using TrackingInputImages = ImageContainerSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>;

class Frame {

public:
    DYNO_POINTER_TYPEDEFS(Frame)
    DYNO_DELETE_COPY_CONSTRUCTORS(Frame)

    std::vector<Feature::Ptr> static_features_;
    std::vector<Feature::Ptr> dynamic_features_;

    Frame(FrameId frame_id, Timestamp timestamp, const TrackingInputImages& tracking_images)
        :   frame_id_(frame_id), timestamp_(timestamp), tracking_images_(tracking_images) {}

    const FrameId frame_id_;
    const Timestamp timestamp_;
    const TrackingInputImages tracking_images_;
};





} //dyno
