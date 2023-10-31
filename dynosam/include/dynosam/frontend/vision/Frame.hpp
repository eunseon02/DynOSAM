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
#include "dynosam/common/ImageContainer.hpp"
#include "dynosam/common/StructuredContainers.hpp"
#include "dynosam/common/Camera.hpp"
#include "dynosam/common/DynamicObjects.hpp"
#include "dynosam/frontend/vision/Feature.hpp"

#include <functional>



namespace dyno {

//should this be here?
using TrackingInputImages = ImageContainerSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>;

class Frame {

public:
    DYNO_POINTER_TYPEDEFS(Frame)
    DYNO_DELETE_COPY_CONSTRUCTORS(Frame)

    using FeatureFilterIterator = internal::filter_iterator<FeatureContainer>;


    const FrameId frame_id_;
    const Timestamp timestamp_;
    Camera::Ptr camera_;
    const TrackingInputImages tracking_images_;

    gtsam::Pose3 T_world_camera_ = gtsam::Pose3::Identity();

    FeatureContainer static_features_;
    FeatureContainer dynamic_features_;
    std::map<ObjectId, DynamicObjectObservation> object_observations_;

    void updateDepths(const ImageWrapper<ImageType::Depth>& depth, double max_static_depth, double max_dynamic_depth);

    FeatureFilterIterator staticUsableBegin();
    FeatureFilterIterator dynamicUsableBegin();

private:
    void updateDepthsFeatureContainer(FeatureContainer& container, const ImageWrapper<ImageType::Depth>& depth, double max_depth);


public:
//    ObjectIds initial_object_labels_; //!Initial object semantic labels as provided by the input semantic/motion mask (does not include background label)

    //also static points that are not used?

    Frame(
        FrameId frame_id,
        Timestamp timestamp,
        Camera::Ptr camera,
        const TrackingInputImages& tracking_images,
        const FeatureContainer& static_features,
        const FeatureContainer& dynamic_features);

private:

};






} //dyno
