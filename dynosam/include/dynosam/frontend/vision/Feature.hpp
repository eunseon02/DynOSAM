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

#include <vector>

namespace dyno {

enum KeyPointType {
    STATIC,
    DYNAMIC
};

//! Expected label for the background in a semantic or motion mask
constexpr static ObjectId background_label = 0u;


struct Feature {

    DYNO_POINTER_TYPEDEFS(Feature)

    KeypointCV keypoint_;
    KeypointCV predicted_keypoint_; //from optical flow
    size_t age_;
    KeyPointType type_;
    TrackletId tracklet_id_;
    FrameId frame_id_;
    bool inlier_{false};
    ObjectId label_; //should be background_label if static
};

using FeaturePtrs = std::vector<Feature::Ptr>;

}
