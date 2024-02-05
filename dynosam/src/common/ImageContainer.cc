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

#include "dynosam/common/ImageContainer.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"


#include <opencv4/opencv2/opencv.hpp>
#include <exception>

namespace dyno {


ImageContainer::Ptr ImageContainer::Create(
        const Timestamp timestamp,
        const FrameId frame_id,
        const ImageWrapper<ImageType::RGBMono>& img,
        const ImageWrapper<ImageType::Depth>& depth,
        const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
        const ImageWrapper<ImageType::SemanticMask>& semantic_mask,
        const ImageWrapper<ImageType::MotionMask>& motion_mask,
        const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation)
{
    std::shared_ptr<ImageContainer> container(
        new ImageContainer(
            timestamp,
            frame_id,
            img,
            depth,
            optical_flow,
            semantic_mask,
            motion_mask,
            class_segmentation)
        );
    container->validateSetup();
    return container;
}


ImageContainer::ImageContainer(
        const Timestamp timestamp,
        const FrameId frame_id,
        const ImageWrapper<ImageType::RGBMono>& img,
        const ImageWrapper<ImageType::Depth>& depth,
        const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
        const ImageWrapper<ImageType::SemanticMask>& semantic_mask,
        const ImageWrapper<ImageType::MotionMask>& motion_mask,
        const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation)
    :  Base(img, depth, optical_flow, semantic_mask, motion_mask, class_segmentation),
       timestamp_(timestamp),
       frame_id_(frame_id)
    { }


std::string ImageContainer::toString() const {
    std::stringstream ss;
    ss << "Timestamp: " << timestamp_ << "\n";
    ss << "Frame Id: " << frame_id_ << "\n";

    const std::string image_config = isImageRGB() ? "RGB" : "Grey";
    ss << "Configuration: " << image_config
        << " Depth (" << hasDepth() << ") Semantic Mask (" << hasSemanticMask()
        << ") Motion Mask (" << hasMotionMask()
        << ") Class Segmentation (" << hasClassSegmentation() << ")" << "\n";

    return ss.str();
}


ImageContainer::Ptr ImageContainer::Create(
        const Timestamp timestamp,
        const FrameId frame_id,
        const ImageWrapper<ImageType::RGBMono>& img,
        const ImageWrapper<ImageType::Depth>& depth,
        const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
        const ImageWrapper<ImageType::SemanticMask>& semantic_mask) {

    return Create(timestamp, frame_id, img, depth, optical_flow, semantic_mask, ImageWrapper<ImageType::MotionMask>(), ImageWrapper<ImageType::ClassSegmentation>());

}

ImageContainer::Ptr ImageContainer::Create(
        const Timestamp timestamp,
        const FrameId frame_id,
        const ImageWrapper<ImageType::RGBMono>& img,
        const ImageWrapper<ImageType::Depth>& depth,
        const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
        const ImageWrapper<ImageType::MotionMask>& motion_mask,
        const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation) {

    return Create(timestamp, frame_id, img, depth, optical_flow, ImageWrapper<ImageType::SemanticMask>(), motion_mask, class_segmentation);
}



void ImageContainer::validateSetup() const {
    //check image sizes are the same
    Base::validateSetup();

    //TODO: shoudl eventually change to exception
    CHECK(!getImage().empty()) << "RGBMono image must not be empty!";
    CHECK(!getOpticalFlow().empty()) << "OPticalFlow image must not be empty!";

    //must have at least one of semantic mask or motion mask
    CHECK(!getSemanticMask().empty() || !getMotionMask().empty()) << "At least one of the masks (semantic or motion) must be valid. Both are empty";

    //should only provide one mask (for clarity - which one to use if both given?)
    if(hasSemanticMask()) {
        CHECK(!hasMotionMask()) << "Semantic mask has been provided, so motion mask should be empty!";
    }

    if(hasMotionMask()) {
        CHECK(!hasSemanticMask()) << "Motion mask has been provided, so semantic mask should be empty!";
    }
}

} //dyno
