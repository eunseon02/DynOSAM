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

#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

#include <type_traits>

#include <exception>

namespace dyno {

//thrown when an image type validation step fails
class InvalidImageTypeException : public std::runtime_error {
public:

    InvalidImageTypeException(const std::string& reason) :
        std::runtime_error("Invalid image type exception: " + reason) {}

};


struct ImageType {

    struct RGBMono {
        constexpr static size_t index = 0u;
        static void validate(const cv::Mat& input);
        static std::string name() { return "RGBMono"; }
    };

    struct Depth {
        constexpr static size_t index = RGBMono::index + 1u;
        static void validate(const cv::Mat& input);
        static std::string name() { return "Depth"; }
    };
    struct OpticalFlow {
        constexpr static size_t index = Depth::index + 1u;
        static void validate(const cv::Mat& input);
        static std::string name() { return "OpticalFlow"; }
    };
    struct SemanticMask {
        constexpr static size_t index = OpticalFlow::index + 1u;
        static void validate(const cv::Mat& input);
        static std::string name() { return "SemanticMask"; }
    };
    struct MotionMask {
        constexpr static size_t index = SemanticMask::index + 1u;
        static void validate(const cv::Mat& input);
        static std::string name() { return "MotionMask"; }
    };
};



template<typename T>
struct ImageWrapper {
    using Type = T;

    const cv::Mat image;

    ImageWrapper(const cv::Mat& img) : image(img) {

        const std::string name = Type::name();

        if(img.empty()) {
            throw InvalidImageTypeException("Image was empty for type " + name);
        }
        //should throw excpetion if problematic
        Type::validate(img);
    }

    ImageWrapper() : image() {}

    inline bool exists() const {
        return !image.empty();
    }
};


class ImageContainer {

public:
    DYNO_POINTER_TYPEDEFS(ImageContainer)
    virtual ~ImageContainer() = default;

    /**
     * @brief Returns true if the set of input images does not contain depth and
     * therefore should be used as part of a Monocular VO pipeline
     *
     * @return true
     * @return false
     */
    inline bool isMonocular() const { return !hasDepth(); }
    inline bool hasDepth() const { return !depth_.empty(); }
    inline bool hasSemanticMask() const { return !semantic_mask_.empty(); }
    inline bool hasMotionMask() const { return !motion_mask_.empty(); }

    inline Timestamp getTimestamp() const { return timestamp_; }
    inline FrameId getFrameId() const { return frame_id_; }

    /**
     * @brief Returns image. Could be RGB or greyscale
     *
     * @return const cv::Mat&
     */
    const cv::Mat& getImage() const { return img_; }

    const cv::Mat& getDepth() const { return depth_; }

    const cv::Mat& getOpticalFlow() const { return optical_flow_; }

    const cv::Mat& getSemanticMask() const { return semantic_mask_; }

    const cv::Mat& getMotionMask() const { return motion_mask_; }

    /**
     * @brief Construct an image container equivalent to RGBD + Semantic Mask input
     *
     * @param img
     * @param depth
     * @param optical_flow
     * @param semantic_mask
     * @return ImageContainer::Ptr
     */
    static ImageContainer::Ptr Create(
        const Timestamp timestamp,
        const FrameId frame_id,
        const ImageWrapper<ImageType::RGBMono>& img,
        const ImageWrapper<ImageType::Depth>& depth,
        const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
        const ImageWrapper<ImageType::SemanticMask>& semantic_mask);

protected:
    explicit ImageContainer(
        const Timestamp timestamp,
        const FrameId frame_id,
        const ImageWrapper<ImageType::RGBMono>& img,
        const ImageWrapper<ImageType::Depth>& depth,
        const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
        const ImageWrapper<ImageType::SemanticMask>& semantic_mask,
        const ImageWrapper<ImageType::MotionMask>& motion_mask);

private:
    void validateSetup() const;

private:
    const Timestamp timestamp_;
    const FrameId frame_id_;
    const cv::Mat img_;
    const cv::Mat depth_;
    const cv::Mat optical_flow_;
    const cv::Mat semantic_mask_;
    const cv::Mat motion_mask_;
};


// // //inherit from to add more IMAGE data
// struct InputImagePacketBase {
//     DYNO_POINTER_TYPEDEFS(InputImagePacketBase)

//     const FrameId frame_id_;
//     const Timestamp timestamp_;
//     const cv::Mat rgb_; //could be either rgb or greyscale?
//     const cv::Mat optical_flow_;

//     InputImagePacketBase(const FrameId frame_id, const Timestamp timestamp, const cv::Mat& rgb, const cv::Mat& optical_flow)
//         :   frame_id_(frame_id), timestamp_(timestamp), rgb_(rgb), optical_flow_(optical_flow)
//     {
//         CHECK(rgb_.type() == CV_8UC1 || rgb_.type() == CV_8UC3) << "The provided rgb image is not grayscale or rgb";
//         CHECK(!rgb.empty()) << "The provided rgb image is empty!";

//         CHECK(optical_flow_.type() == CV_32FC2) << "The provided optical flow image is not of datatype CV_32FC2 ("
//             << "was " << utils::cvTypeToString(optical_flow_.type()) << ")";
//         CHECK(!optical_flow_.empty()) << "The provided optical flow image is empty!";
//     }

//     virtual ~InputImagePacketBase() = default;

//     //should be overwritten in derived class to make the visualzier work properly
//     virtual void draw(cv::Mat& img) const {

//     }
// };


//inherit to add more sensor data (eg imu)
//can be of any InputImagePacketBase with this as the default. We use this so we can
//create a FrontendInputPacketBase type with the InputPacketType as the actual derived type
//and not the base class so different modules do not have to do their own casting to get image data
struct FrontendInputPacketBase {
    DYNO_POINTER_TYPEDEFS(FrontendInputPacketBase)


    ImageContainer::Ptr image_container_;
    GroundTruthInputPacket::Optional optional_gt_;

    FrontendInputPacketBase() : image_container_(nullptr), optional_gt_(std::nullopt) {}

    FrontendInputPacketBase(ImageContainer::Ptr image_container, GroundTruthInputPacket::Optional optional_gt = std::nullopt)
    : image_container_(image_container), optional_gt_(optional_gt)
    {
        CHECK(image_container_);

        if(optional_gt) {
            CHECK_EQ(optional_gt_->frame_id_, image_container_->getFrameId());
        }
    }


    virtual ~FrontendInputPacketBase() = default;
};







} //dyno
