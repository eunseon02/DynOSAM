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

#include "dynosam/frontend/vision/FeatureDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>

#include <glog/logging.h>

namespace dyno {

template<>
FunctionalDetector::Ptr FunctionalDetector::Create<cv::GFTTDetector>(const TrackerParams& tracker_params) {
    LOG(INFO) << "Creating cv::GFTTDetector";

    auto feature_detector_ = cv::GFTTDetector::create(
          tracker_params.max_nr_keypoints_before_anms,
          tracker_params.gfft_params.quality_level,
          tracker_params.min_distance_btw_tracked_and_detected_features,
          tracker_params.gfft_params.block_size,
          tracker_params.gfft_params.use_harris_corner_detector,
          tracker_params.gfft_params.k);
    auto functional_detector = [=](const cv::Mat& img, KeypointsCV& keypoints, const cv::Mat& mask) -> void {
        CHECK_NOTNULL(feature_detector_)->detect(img, keypoints, mask);
    };

    return std::make_shared<FunctionalDetector>(tracker_params,functional_detector);
}

} //namespace dyno
