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

#include "dynosam/frontend/vision/VisionTools.hpp"

namespace dyno {


RGBDProcessor::RGBDProcessor(const FrontendParams& params, Camera::Ptr camera)
    : params_(params), camera_(camera) {}


void RGBDProcessor::updateDepth(Frame::Ptr frame, ImageWrapper<ImageType::Depth> disparity) {
    cv::Mat depth;
    disparityToDepth(disparity.image, depth);

    setDepths(frame->static_features_, depth, params_.depth_background_thresh);
    setDepths(frame->dynamic_features_, depth, params_.depth_obj_thresh);
}

void RGBDProcessor::disparityToDepth(const cv::Mat& disparity, cv::Mat& depth) {
    disparity.copyTo(depth);
    for (int i = 0; i < disparity.rows; i++)
     {
    for (int j = 0; j < disparity.cols; j++)
    {
      if (disparity.at<double>(i, j) < 0)
      {
        depth.at<double>(i, j) = 0;
      }
      else
      {
        depth.at<double>(i, j) = params_.base_line / (disparity.at<double>(i, j) / params_.depth_scale_factor);
      }
    }
  }
}

void RGBDProcessor::setDepths(FeaturePtrs features, const cv::Mat& depth, double max_threshold) {
    for(Feature::Ptr feature : features) {

        if(!feature) continue;

        const int x = functional_keypoint::u(feature->keypoint_);
        const int y = functional_keypoint::v(feature->keypoint_);
        const Depth d = depth.at<Depth>(y, x);


        if(d > max_threshold || d <= 0) {
            feature->markInvalid();
        }

        //if now invalid or happens to be invalid from a previous frame, make depth invalud too
        if(!feature->usable()) {
            feature->depth_ = Feature::invalid_depth;
        }
        else {
            feature->depth_ = d;
        }
    }
}


} //dyno
