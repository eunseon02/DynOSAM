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

#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"


#include <opencv4/opencv2/opencv.hpp>
#include <exception>

namespace dyno {

void validateMask(const cv::Mat& input, const std::string& name) {
    if(input.type() != CV_32SC1) {
        throw InvalidImageTypeException(
            name + " image was not CV_32SC1. Input image type was " + utils::cvTypeToString(input.type())
        );
    }
}

void ImageType::RGBMono::validate(const cv::Mat& input) {
    if(input.type() != CV_8UC1 && input.type() != CV_8UC3) {
        throw InvalidImageTypeException(
            "RGBMono image was not CV_8UC1 or CV_8UC3. Input image type was " + utils::cvTypeToString(input.type())
        );
    }
}


void ImageType::Depth::validate(const cv::Mat& input){
     if(input.type() != CV_64F) {
        throw InvalidImageTypeException(
            "Depth image was notCV_64F. Input image type was " + utils::cvTypeToString(input.type())
        );
    }

}


void ImageType::OpticalFlow::validate(const cv::Mat& input) {
    if(input.type() != CV_32FC2) {
        throw InvalidImageTypeException(
            "OpticalFlow image was not CV_32FC2. Input image type was " + utils::cvTypeToString(input.type())
        );
    }
}


void ImageType::SemanticMask::validate(const cv::Mat& input) {
    validateMask(input, name());
}


void ImageType::MotionMask::validate(const cv::Mat& input) {
    validateMask(input, name());
}




ImageContainer::ImageContainer(
        const Timestamp timestamp,
        const FrameId frame_id,
        const ImageWrapper<ImageType::RGBMono>& img,
        const ImageWrapper<ImageType::Depth>& depth,
        const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
        const ImageWrapper<ImageType::SemanticMask>& semantic_mask,
        const ImageWrapper<ImageType::MotionMask>& motion_mask)
    :  timestamp_(timestamp),
       frame_id_(frame_id),
       img_(img.image),
       depth_(depth.image),
       optical_flow_(optical_flow.image),
       semantic_mask_(semantic_mask.image),
       motion_mask_(motion_mask.image)
    { validateSetup(); }



ImageContainer::Ptr ImageContainer::Create(
        const Timestamp timestamp,
        const FrameId frame_id,
        const ImageWrapper<ImageType::RGBMono>& img,
        const ImageWrapper<ImageType::Depth>& depth,
        const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
        const ImageWrapper<ImageType::SemanticMask>& semantic_mask) {

    return std::shared_ptr<ImageContainer>(new ImageContainer(timestamp, frame_id, img, depth, optical_flow, semantic_mask, ImageWrapper<ImageType::MotionMask>()));

}


void ImageContainer::validateSetup() const {
    //TODO: shoudl eventually change to exception
    CHECK(!img_.empty()) << "RGBMono image must not be empty!";
    CHECK(!optical_flow_.empty()) << "OPticalFlow image must not be empty!";

    //must have at least one of semantic mask or motion mask
    CHECK(!semantic_mask_.empty() || !motion_mask_.empty()) << "At least one of the masks (semantic or motion) must be valid. Both are empty";

    //should only provide one mask (for clarity - which one to use if both given?)
    if(hasSemanticMask()) {
        CHECK(!hasMotionMask()) << "Semantic mask has been provided, so motion mask should be empty!";
    }

    if(hasMotionMask()) {
        CHECK(!hasSemanticMask()) << "Motion mask has been provided, so semantic mask should be empty!";
    }
}

} //dyno
