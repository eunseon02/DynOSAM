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

#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"

namespace dyno {

decltype(TrackletIdManager::instance_) TrackletIdManager::instance_;

FeatureTrackerBase::FeatureTrackerBase(const FrontendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
  : params_(params),
    img_size_(camera->getParams().imageSize()),
    camera_(camera),
    display_queue_(display_queue) {}


bool FeatureTrackerBase::isWithinShrunkenImage(const Keypoint& kp) const {
    const auto shrunken_row = params_.shrink_row;
    const auto shrunken_col = params_.shrink_col;

    const int predicted_col = functional_keypoint::u(kp);
    const int predicted_row = functional_keypoint::v(kp);

    const auto image_rows = img_size_.height;
    const auto image_cols = img_size_.width;
    return (predicted_row > shrunken_row && predicted_row < (image_rows - shrunken_row) &&
        predicted_col > shrunken_col && predicted_col < (image_cols - shrunken_col));

}

}
