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
#include "dynosam/common/Camera.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {



class FrameProcessor {

public:
    FrameProcessor(const FrontendParams& params, Camera::Ptr camera);

    const FrontendParams& getParams() const { return params_; }

    //assume everything rectified, depth etc at this stage, previous frame has pose
    void getCorrespondences(AbsolutePoseCorrespondences& correspondences, const Frame& previous_frame, const Frame& current_frame, KeyPointType kp_type) const;
    void getCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, const Frame& current_frame, KeyPointType kp_type) const;

    static ObjectIds getObjectLabels(const ImageWrapper<ImageType::MotionMask>& image);
    static ObjectIds getObjectLabels(const ImageWrapper<ImageType::SemanticMask>& image);


protected:
    void getStaticCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, const Frame& current_frame) const;
    void getDynamicCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, const Frame& current_frame) const;

    void getCorrespondencesFromContainer(FeaturePairs& correspondences, const FeatureContainer& previous_features, const FeatureContainer& current_features) const;

private:
    static ObjectIds getObjectLabels(const cv::Mat& image);

protected:
    const FrontendParams params_;
    Camera::Ptr camera_;

};


class RGBDProcessor : public FrameProcessor {

public:
    RGBDProcessor(const FrontendParams& params, Camera::Ptr camera);

    //this is really disparty not depth
    void updateDepth(Frame::Ptr frame, ImageWrapper<ImageType::Depth> disparity);
    void disparityToDepth(const cv::Mat& disparity, cv::Mat& depth);

    //expects frame pose to be hpdated for both frames
    ImageWrapper<ImageType::MotionMask> calculateMotionMask(const Frame& previous_frame, const Frame& current_frame);


private:

    //copy but everything is a ptr anyway??
    void setDepths(FeatureContainer features, const cv::Mat& depth, double max_threshold);


};


void determineOutlierIds(const TrackletIds& inliers, const TrackletIds& tracklets, TrackletIds& outliers);



} //dyno
