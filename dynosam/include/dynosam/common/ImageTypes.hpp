/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <glog/logging.h>

#include <exception>
#include <opencv4/opencv2/opencv.hpp>
#include <type_traits>

#include "dynosam/common/Exceptions.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/Tuple.hpp"

namespace dyno {

template <typename IMAGETYPE>
struct image_traits;

class InvalidImageTypeException : public DynosamException {
 public:
  InvalidImageTypeException(const std::string& reason)
      : DynosamException("Invalid image type exception: " + reason) {}
};

template <typename IMAGETYPE>
struct image_traits_impl {
  using type = IMAGETYPE;
  constexpr static int OpenCVType = IMAGETYPE::OpenCVType;
  static void validate(const cv::Mat& input) { IMAGETYPE::validate(input); }
  static std::string name() { return IMAGETYPE::name(); }
};

template <typename T>
struct ImageWrapper;

struct ImageBase {
  cv::Mat image;  //! Underlying image

  ImageBase() = default;
  ImageBase(const cv::Mat& img);
  virtual ~ImageBase() {}

  operator cv::Mat&() { return image; }
  operator const cv::Mat&() const { return image; }

  template <typename IMAGETYPE>
  const ImageWrapper<IMAGETYPE>& cast() const;

  bool exists() const;
};

/**
 * @brief ImageWrapper for a ImageType that defines a particular type of input
 * image. The ImageType template is used to create different types (actual c++
 * types) to differntial cv::Mat's from each other - eg. a type for a RGB image,
 * Optical Flow or Motion Mask.
 *
 * ImageType must define a static validate(const cv::Mat&) function which throws
 * an expcetion if the cv::Mat does not meed the requirements of the ImageType
 * and a static name() function which returns a string. Must also have a static
 * toRGB(const ImageWrapper<Type>& image) fucntion.
 *
 *
 * @tparam IMAGETYPE
 */
template <typename IMAGETYPE>
struct ImageWrapper : public ImageBase {
  using Type = IMAGETYPE;
  using This = ImageWrapper<Type>;

  ImageWrapper() = default;
  ImageWrapper(const cv::Mat& img);
  ImageWrapper(const ImageBase& image_base);

  /**
   * @brief Returns an ImageWrapper with the underlying image data cloned
   *
   * @return ImageWrapper<Type>
   */
  ImageWrapper<Type> clone() const;

  cv::Mat toRGB() const { return IMAGETYPE::toRGB(*this); }
};

// define ImageBase::cast here since now ImageWrapper has been declared
template <typename IMAGETYPE>
const ImageWrapper<IMAGETYPE>& ImageBase::cast() const {
  return dynamic_cast<const ImageWrapper<IMAGETYPE>&>(*this);
}

struct ImageType {
  /**
   * @brief Defines the input RGB/greyscale (mono) image.
   *
   *
   */
  struct RGBMono {
    static void validate(const cv::Mat& input);
    static std::string name() { return "RGBMono"; }
    static cv::Mat toRGB(const ImageWrapper<RGBMono>& image);
    static cv::Mat toMono(const ImageWrapper<RGBMono>& image);
  };

  // really should be disparity?
  // depends on dataset. the inputc can be disparity, but by the time we get to
  // the frontend it MUST be depth
  struct Depth {
    constexpr static int OpenCVType =
        CV_64F;  //! Expected opencv image type for depth (or disparity) image
    static void validate(const cv::Mat& input);
    static std::string name() { return "Depth"; }
    static cv::Mat toRGB(const ImageWrapper<Depth>& image);
  };
  struct OpticalFlow {
    constexpr static int OpenCVType =
        CV_32FC2;  //! Expected opencv image type OpticalFlow image
    static void validate(const cv::Mat& input);
    static std::string name() { return "OpticalFlow"; }
    static cv::Mat toRGB(const ImageWrapper<OpticalFlow>& image);
  };
  struct SemanticMask {
    constexpr static int OpenCVType =
        CV_32SC1;  //! Expected opencv image type for SemanticMask image type
    static void validate(const cv::Mat& input);
    static std::string name() { return "SemanticMask"; }
    static cv::Mat toRGB(const ImageWrapper<SemanticMask>& image);
  };
  struct MotionMask {
    constexpr static int OpenCVType =
        CV_32SC1;  //! Expected opencv image type for MotionMask image type
    static void validate(const cv::Mat& input);
    static std::string name() { return "MotionMask"; }
    static cv::Mat toRGB(const ImageWrapper<MotionMask>& image);
  };
  struct ClassSegmentation {
    constexpr static int OpenCVType =
        CV_32SC1;  //! Expected opencv image type for ClassSegmentation image
                   //! type
    static void validate(const cv::Mat& input);
    static std::string name() { return "ClassSegmentation"; }
    static cv::Mat toRGB(const ImageWrapper<ClassSegmentation>& image);

    enum Labels {
      Undefined = 0,
      Road,
      Rider,
    };
  };

  static_assert(SemanticMask::OpenCVType == MotionMask::OpenCVType);
  static_assert(ClassSegmentation::OpenCVType == MotionMask::OpenCVType);
};

template <>
struct image_traits<ImageType::RGBMono>
    : public image_traits_impl<ImageType::RGBMono> {};

template <>
struct image_traits<ImageType::Depth>
    : public image_traits_impl<ImageType::Depth> {};

template <>
struct image_traits<ImageType::OpticalFlow>
    : public image_traits_impl<ImageType::OpticalFlow> {};

template <>
struct image_traits<ImageType::SemanticMask>
    : public image_traits_impl<ImageType::SemanticMask> {};

template <>
struct image_traits<ImageType::MotionMask>
    : public image_traits_impl<ImageType::MotionMask> {};

}  // namespace dyno

#include "dynosam/common/ImageTypes-inl.hpp"
