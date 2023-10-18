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
#include "dynosam/utils/Tuple.hpp"
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

class InvalidImageContainerException : public std::runtime_error {
public:
    InvalidImageContainerException(const std::string& what) : std::runtime_error(what) {}
};


struct ImageType {

    struct RGBMono {
        static void validate(const cv::Mat& input);
        static std::string name() { return "RGBMono"; }
    };

    struct Depth {
        static void validate(const cv::Mat& input);
        static std::string name() { return "Depth"; }
    };
    struct OpticalFlow {
        static void validate(const cv::Mat& input);
        static std::string name() { return "OpticalFlow"; }
    };
    struct SemanticMask {
        static void validate(const cv::Mat& input);
        static std::string name() { return "SemanticMask"; }
    };
    struct MotionMask {
        static void validate(const cv::Mat& input);
        static std::string name() { return "MotionMask"; }
    };
};


template<typename T>
struct ImageWrapper {
    using Type = T;

    //TODO: for now remove const (which is dangerous i guess, but need assignment operator)?
    //what about copying image data instead of just reference? Right now, everything is ref
    cv::Mat image;

    ImageWrapper(const cv::Mat& img) : image(img) {

        const std::string name = Type::name();

        if(img.empty()) {
            throw InvalidImageTypeException("Image was empty for type " + name);
        }
        //should throw excpetion if problematic
        Type::validate(img);
    }


    ImageWrapper() : image() {}

    /**
     * @brief Returns an ImageWrapper with the underlying image data cloned
     *
     * @return ImageWrapper<Type>
     */
    ImageWrapper<Type> clone() {
        return ImageWrapper<Type>(image.clone());
    }

    inline bool exists() const {
        return !image.empty();
    }
};



template<typename... ImageTypes>
class ImageContainerSubset {
public:
    using This = ImageContainerSubset<ImageTypes...>;
    using ImageTypesTuple = std::tuple<ImageTypes...>;
    using WrappedImageTypesTuple = std::tuple<ImageWrapper<ImageTypes>...>;

    /**
     * @brief Alias for an ImageType (eg. RGBMono, Depth etc) which is the Ith element type
     * of the parameter pack ImageTypes...
     *
     * @tparam I
     */
    template <size_t I>
    using ImageTypeStruct = std::tuple_element_t<I, ImageTypesTuple>;

    /**
     * @brief Corresponds to an alias for a type ImageWrapper<T> where T is the ImageTypeStruct,
     * eg. RGBMono, Depth... at index I (or simply ImageTypeStruct<I>)
     *
     * @tparam I
     */
    template <size_t I>
    using WrappedImageTypeStruct = std::tuple_element_t<I, WrappedImageTypesTuple>;

    /**
     * @brief Number of image types contained in this subset
     *
     */
    static constexpr size_t N = sizeof...(ImageTypes);


    explicit ImageContainerSubset(const ImageWrapper<ImageTypes>&... input_image_wrappers) : image_storage_(input_image_wrappers...)
    {
         //this is (not-pure) virtual function could be overloaded. This means that we're calling an overloaded function
         //in the base constructor which leads to undefined behaviour...?
        validateSetup();
    }
    virtual ~ImageContainerSubset() = default;


    /**
     * @brief Compile time index of the requested ImageType in the container.
     *
     * eg. If ImageTypes... unpacks to Container = ImageContainerSubset<RGBMono, Depth, OpticalFlow>
     *
     * If the requested ImageType is not in the class parameter pack, the code will fail to compile.
     *
     * Container::Index<RGBMono>() == 0u
     * Container::Index<Depth>() == 1u
     * Container::Index<OpticalFlow>() == 2u
     *
     * @tparam ImageType
     * @return constexpr size_t
     */
    template<typename ImageType>
    constexpr static size_t Index() {
        return internal::tuple_element_index_v<ImageType, ImageTypesTuple>;
    }

    /**
     * @brief Compile time index of the requested ImageType in the container instance.
     *
     * See Index()
     *
     * @tparam ImageType
     * @return constexpr size_t
     */
    template<typename ImageType>
    constexpr size_t index() const {
        return This::Index<ImageType>();
    }

    template<typename... SubsetImageTypes>
    ImageContainerSubset<SubsetImageTypes...> makeSubset(const bool clone = false) {
        //the tuple of image wrappers to construct for the subset class
        using Subset = ImageContainerSubset<SubsetImageTypes...>;
        using WrappedImageTypesTupleSubset = typename Subset::WrappedImageTypesTuple;
        static constexpr size_t SubN = Subset::N;

        //wrappers to populate from the current set? what about copying etc
        WrappedImageTypesTupleSubset subset_wrappers;
        for (size_t i = 0; i < SubN; i++) {
            //get the type at the new subset
            internal::select_apply<SubN>(i, [&](auto stream_index) {
                internal::select_apply<SubN>(stream_index, [&](auto I) {
                    using SubType = typename Subset::ImageTypeStruct<I>;
                    //find this image type in the current container as the idnex may be different
                    constexpr size_t current_index = This::Index<SubType>();
                    //with the index i of the subset wrapper (corresponding with type SubType)
                    //get the image wrapper in this container using the request (sub) type
                    //TODO:do we want cv::Mat data to be const (refer counter) or at some point clone? See ImageType where the cv::mat is stored?
                    auto current_wrapped_image = std::get<current_index>(image_storage_);

                    //if invalid but request, throw runtime error becuase whats the point of requesting an invalid image
                    if(!current_wrapped_image.exists()) {
                        throw std::runtime_error(
                            "Error constructing ImageContainerSubset - requested image type " + type_name<SubType>() + " is empty in parent container");
                    }

                    if(clone) {
                        std::get<I>(subset_wrappers) = current_wrapped_image.clone();
                    }
                    else {
                        std::get<I>(subset_wrappers) = current_wrapped_image;
                    }

                });
            });
        }

        //this is going to do a bunch of copies?
        auto construct_subset = [&](auto&&... args) { return Subset(args...); };
        return std::apply(construct_subset, subset_wrappers);
    }

    /**
     * @brief Returns true if the corresponding image corresponding to the requested ImageType exists (i.e is not empty)
     *
     * @tparam ImageType
     * @return true
     * @return false
     */
    template<typename ImageType>
    bool exists() const {
        return This::getImageWrapper<ImageType>().exists();
    }

    /**
     * @brief Gets the corresponding image (cv::Mat) corresponding to the requested ImageType.
     * ImageType must be in the class parameter pack (ImageTypes...) or the code will fail to compile.
     *
     * No safety checks are done and so the returned image may be empty
     *
     * @tparam ImageType
     * @return cv::Mat
     */
    template<typename ImageType>
    cv::Mat get() const {
        return std::get<Index<ImageType>()>(image_storage_).image;
    }

    /**
     * @brief Safely Gets the corresponding image (cv::Mat) corresponding to the requested ImageType.
     * If the requested image does not exist (empty), false will be returned.
     *
     * ImageType must be in the class parameter pack (ImageTypes...) or the code will fail to compile.
     *
     * @tparam ImageType
     * @param src
     * @return true
     * @return false
     */
    template<typename ImageType>
    bool safeGet(cv::Mat& src) const {
        if(!This::exists<ImageType>()) return false;

        src = This::get<ImageType>();
        return true;
    }

    /**
     * @brief Safely clones the corresponding image (cv::Mat) corresponding to the requested ImageType.
     * If the requested image does not exist (empty), false will be returned.
     *
     * ImageType must be in the class parameter pack (ImageTypes...) or the code will fail to compile.
     *
     * @tparam ImageType
     * @param dst
     * @return true
     * @return false
     */
    template<typename ImageType>
    bool cloneImage(cv::Mat& dst) const {
        cv::Mat tmp;
        if(!This::safeGet<ImageType>(tmp)) {
            return false;
        }

        tmp.copyTo(dst);
        return true;
    }

protected:

    /**
     * @brief Gets the corresponding ImageWrapper corresponding to the ImageType.
     * ImageType must be in the class parameter pack (ImageTypes...) or the code will fail to compile.
     *
     * @tparam ImageType
     * @return const ImageWrapper<ImageType>&
     */
    template<typename ImageType>
    const ImageWrapper<ImageType>& getImageWrapper() const {
        constexpr size_t idx =  This::Index<ImageType>();
        return std::get<idx>(image_storage_);
    }

    /**
     * @brief Validates that the input images (if not empty) are the same size.
     * Throws InvalidImageContainerException if invalid.
     *
     * Compares all non-empty images against the first image size
     *
     * Can be overwritten.
     *
     */
    virtual void validateSetup() const {
        cv::Size required_size;

        for (size_t i = 0; i < N; i++) {
            //get the type at the new subset
            internal::select_apply<N>(i, [&](auto stream_index) {
                using ImageTypeStruct = typename This::ImageTypeStruct<stream_index>;
                //only check if exists?
                if(This::exists<ImageTypeStruct>()) {
                    //if required size is empty, then first check
                    //set required size to be the image size and then compare against
                    //this size on future iterations
                    const bool is_first = required_size.empty();
                    if(is_first) {
                        required_size = This::get<ImageTypeStruct>().size();
                    }
                    else {
                        //compare against required size
                        const cv::Size incoming_size = This::get<ImageTypeStruct>().size();
                        if(incoming_size != required_size) {
                            throw InvalidImageContainerException(
                                "Non-empty images were not all the same size. First image (type: " +
                                type_name<This::ImageTypeStruct<0u>>() + ") was of size " + to_string(required_size) +
                                " and other image (type: " + type_name<ImageTypeStruct>() + ") was of size " + to_string(incoming_size)
                            );
                        }
                    }
                }
            });
        }
    }

protected:
    const WrappedImageTypesTuple image_storage_;

};


class ImageContainer : public ImageContainerSubset<
    ImageType::RGBMono,
    ImageType::Depth,
    ImageType::OpticalFlow,
    ImageType::SemanticMask,
    ImageType::MotionMask> {

public:
    DYNO_POINTER_TYPEDEFS(ImageContainer)

    using Base = ImageContainerSubset<
        ImageType::RGBMono,
        ImageType::Depth,
        ImageType::OpticalFlow,
        ImageType::SemanticMask,
        ImageType::MotionMask>;


    /**
     * @brief Returns image. Could be RGB or greyscale
     *
     * @return cv::Mat
     */
    cv::Mat getImage() const { return Base::get<ImageType::RGBMono>(); }

    cv::Mat getDepth() const { return Base::get<ImageType::Depth>(); }

    cv::Mat getOpticalFlow() const { return Base::get<ImageType::OpticalFlow>(); }

    cv::Mat getSemanticMask() const { return Base::get<ImageType::SemanticMask>(); }

    cv::Mat getMotionMask() const { return Base::get<ImageType::MotionMask>(); }

    /**
     * @brief Returns true if the set of input images does not contain depth and
     * therefore should be used as part of a Monocular VO pipeline
     *
     * @return true
     * @return false
     */
    inline bool isMonocular() const { return !hasDepth(); }
    inline bool hasDepth() const { return !getDepth().empty(); }
    inline bool hasSemanticMask() const { return !getSemanticMask().empty(); }
    inline bool hasMotionMask() const { return !getMotionMask().empty(); }

    /**
     * @brief True if the image has 3 channels and is therefore expected to be RGB.
     * The alternative is greyscale
     *
     * @return true
     * @return false
     */
    inline bool isImageRGB() const { return getImage().channels() == 3; }

    inline Timestamp getTimestamp() const { return timestamp_; }
    inline FrameId getFrameId() const { return frame_id_; }



    std::string toString() const;

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
    void validateSetup() const override;

private:
    const Timestamp timestamp_;
    const FrameId frame_id_;
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
