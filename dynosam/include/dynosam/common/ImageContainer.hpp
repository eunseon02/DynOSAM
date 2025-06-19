/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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
#include "dynosam/common/ImageTypes.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/Tuple.hpp"

namespace dyno {

class ImageContainerConstructionException : public DynosamException {
 public:
  ImageContainerConstructionException(const std::string& what)
      : DynosamException(what) {}
};

template <typename... ImageTypes>
class ImageContainerSubset {
 public:
  using This = ImageContainerSubset<ImageTypes...>;
  using ImageTypesTuple = std::tuple<ImageTypes...>;
  using WrappedImageTypesTuple = std::tuple<ImageWrapper<ImageTypes>...>;

  /**
   * @brief Alias for an ImageType (eg. RGBMono, Depth etc) which is the Ith
   * element type of the parameter pack ImageTypes...
   *
   * @tparam I
   */
  template <size_t I>
  using ImageTypeStruct = std::tuple_element_t<I, ImageTypesTuple>;

  /**
   * @brief Corresponds to an alias for a type ImageWrapper<T> where T is the
   * ImageTypeStruct, eg. RGBMono, Depth... at index I (or simply
   * ImageTypeStruct<I>)
   *
   * @tparam I
   */
  template <size_t I>
  using WrappedImageTypeStruct =
      std::tuple_element_t<I, WrappedImageTypesTuple>;

  /**
   * @brief Number of image types contained in this subset
   *
   */
  static constexpr size_t N = sizeof...(ImageTypes);

  ImageContainerSubset() {}
  explicit ImageContainerSubset(
      const ImageWrapper<ImageTypes>&... input_image_wrappers)
      : image_storage_(input_image_wrappers...) {
    // this is (not-pure) virtual function could be overloaded. This means that
    // we're calling an overloaded function in the base constructor which leads
    // to undefined behaviour...?
    // validateSetup();
  }
  virtual ~ImageContainerSubset() = default;

  // TODO: need to do test for default assignment
  // eg ImageContainerSubset sub;
  // sub = ImageContainerSubset(...types...)

  /**
   * @brief Compile time index of the requested ImageType in the container.
   *
   * eg. If ImageTypes... unpacks to Container = ImageContainerSubset<RGBMono,
   * Depth, OpticalFlow>
   *
   * If the requested ImageType is not in the class parameter pack, the code
   * will fail to compile.
   *
   * Container::Index<RGBMono>() == 0u
   * Container::Index<Depth>() == 1u
   * Container::Index<OpticalFlow>() == 2u
   *
   * @tparam ImageType
   * @return constexpr size_t
   */
  template <typename ImageType>
  constexpr static size_t Index() {
    return internal::tuple_element_index_v<ImageType, ImageTypesTuple>;
  }

  /**
   * @brief Compile time index of the requested ImageType in the container
   * instance.
   *
   * See Index()
   *
   * @tparam ImageType
   * @return constexpr size_t
   */
  template <typename ImageType>
  constexpr size_t index() const {
    return This::Index<ImageType>();
  }

  // would also be nice to template on an ImageContainerSubset
  template <typename... SubsetImageTypes>
  ImageContainerSubset<SubsetImageTypes...> makeSubset(
      const bool clone = false) const;

  This clone() const { return this->makeSubset<ImageTypes...>(true); }

  /**
   * @brief Returns true if the corresponding image corresponding to the
   * requested ImageType exists (i.e is not empty)
   *
   * @tparam ImageType
   * @return true
   * @return false
   */
  template <typename ImageType>
  bool exists() const {
    return This::getImageWrapper<ImageType>().exists();
  }

  /**
   * @brief Gets the corresponding image (cv::Mat) corresponding to the
   * requested ImageType. ImageType must be in the class parameter pack
   * (ImageTypes...) or the code will fail to compile.
   *
   * No safety checks are done and so the returned image may be empty
   *
   * @tparam ImageType
   * @return cv::Mat
   */
  template <typename ImageType>
  cv::Mat get() const {
    return std::get<Index<ImageType>()>(image_storage_).image;
  }

  template <typename ImageType>
  cv::Mat& get() {
    return std::get<Index<ImageType>()>(image_storage_).image;
  }

  /**
   * @brief Gets the corresponding ImageWrapper corresponding to the ImageType.
   * ImageType must be in the class parameter pack (ImageTypes...) or the code
   * will fail to compile.
   *
   * @tparam ImageType
   * @return const ImageWrapper<ImageType>&
   */
  template <typename ImageType>
  const ImageWrapper<ImageType>& getImageWrapper() const {
    constexpr size_t idx = This::Index<ImageType>();
    return std::get<idx>(image_storage_);
  }

  /**
   * @brief Safely Gets the corresponding image (cv::Mat) corresponding to the
   * requested ImageType. If the requested image does not exist (empty), false
   * will be returned.
   *
   * ImageType must be in the class parameter pack (ImageTypes...) or the code
   * will fail to compile.
   *
   * @tparam ImageType
   * @param src
   * @return true
   * @return false
   */
  template <typename ImageType>
  bool safeGet(cv::Mat& src) const {
    if (!This::exists<ImageType>()) return false;

    src = This::get<ImageType>();
    return true;
  }

  /**
   * @brief Safely clones the corresponding image (cv::Mat) corresponding to the
   * requested ImageType. If the requested image does not exist (empty), false
   * will be returned.
   *
   * ImageType must be in the class parameter pack (ImageTypes...) or the code
   * will fail to compile.
   *
   * @tparam ImageType
   * @param dst
   * @return true
   * @return false
   */
  template <typename ImageType>
  bool cloneImage(cv::Mat& dst) const {
    cv::Mat tmp;
    if (!This::safeGet<ImageType>(tmp)) {
      return false;
    }

    tmp.copyTo(dst);
    return true;
  }

 protected:
  /**
   * @brief Validates that the input images (if not empty) are the same size.
   * Throws ImageContainerConstructionException if invalid.
   *
   * Compares all non-empty images against the first image size
   *
   * Can be overwritten.
   *
   */
  virtual void validateSetup() const;

 protected:
  WrappedImageTypesTuple image_storage_;  // cannot be const so we can do
                                          // default
};

class UndistorterRectifier;  // TODO:!!

// TODO: make some clone or better tested copy function so we can control when
// cv::mats are cloned difficult as we want the behaviour in the base class but
// the base class does not know the derived type this and also implicit casting
// to other subset types
class ImageContainer
    : public ImageContainerSubset<
          ImageType::RGBMono, ImageType::Depth, ImageType::OpticalFlow,
          ImageType::SemanticMask, ImageType::MotionMask,
          ImageType::ClassSegmentation> {
 public:
  DYNO_POINTER_TYPEDEFS(ImageContainer)

  using Base =
      ImageContainerSubset<ImageType::RGBMono, ImageType::Depth,
                           ImageType::OpticalFlow, ImageType::SemanticMask,
                           ImageType::MotionMask, ImageType::ClassSegmentation>;

  // TODO: gross, think of a better way of doing this (I think will need to have
  // the base know abot derived type, but
  //  unsure how to do this as we already have much templating!!)
  inline auto toBaseSubset(bool clone = false) const {
    // auto construct_subset = [&](auto&&... args) { return
    // Base::makeSubset(args..); }; return std::apply(construct_subset,
    // subset_wrappers);
    //     return Base::makeSubset<Base::ImageTypesTuple...>(clone);
    return Base::makeSubset<ImageType::RGBMono, ImageType::Depth,
                            ImageType::OpticalFlow, ImageType::SemanticMask,
                            ImageType::MotionMask,
                            ImageType::ClassSegmentation>(clone);
  }

  /**
   * @brief Returns image. Could be RGB or greyscale
   *
   * @return cv::Mat
   */
  cv::Mat getImage() const { return Base::get<ImageType::RGBMono>(); }

  cv::Mat getDepth() const { return Base::get<ImageType::Depth>(); }

  cv::Mat getOpticalFlow() const { return Base::get<ImageType::OpticalFlow>(); }

  cv::Mat getSemanticMask() const {
    return Base::get<ImageType::SemanticMask>();
  }

  cv::Mat getMotionMask() const { return Base::get<ImageType::MotionMask>(); }

  cv::Mat getClassSegmentation() const {
    return Base::get<ImageType::ClassSegmentation>();
  }

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
  inline bool hasClassSegmentation() const {
    return !getClassSegmentation().empty();
  }

  /**
   * @brief True if the image has 3 channels and is therefore expected to be
   * RGB. The alternative is greyscale
   *
   * @return true
   * @return false
   */
  inline bool isImageRGB() const { return getImage().channels() == 3; }

  inline Timestamp getTimestamp() const { return timestamp_; }
  inline FrameId getFrameId() const { return frame_id_; }

  std::string toString() const;

  // TODO:hack
  cv::Mat right_stereo_rgb;

  /**
   * @brief Construct an image container equivalent to RGBD + Semantic Mask
   * input
   *
   * @param img
   * @param depth
   * @param optical_flow
   * @param semantic_mask
   * @return ImageContainer::Ptr
   */
  static ImageContainer::Ptr Create(
      const Timestamp timestamp, const FrameId frame_id,
      const ImageWrapper<ImageType::RGBMono>& img,
      const ImageWrapper<ImageType::Depth>& depth,
      const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
      const ImageWrapper<ImageType::SemanticMask>& semantic_mask);

  /**
   * @brief Construct an image container equivalent to RGBD + Motion Mask input
   *
   * @param timestamp
   * @param frame_id
   * @param img
   * @param depth
   * @param optical_flow
   * @param motion_mask
   * @param class_segmentation
   * @return ImageContainer::Ptr
   */
  static ImageContainer::Ptr Create(
      const Timestamp timestamp, const FrameId frame_id,
      const ImageWrapper<ImageType::RGBMono>& img,
      const ImageWrapper<ImageType::Depth>& depth,
      const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
      const ImageWrapper<ImageType::MotionMask>& motion_mask,
      const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation);

  /**
   * @brief Construct an image container equivalent to RGBD + Motion Mask input
   *
   * @param timestamp
   * @param frame_id
   * @param img
   * @param depth
   * @param optical_flow
   * @param motion_mask
   * @return ImageContainer::Ptr
   */
  static ImageContainer::Ptr Create(
      const Timestamp timestamp, const FrameId frame_id,
      const ImageWrapper<ImageType::RGBMono>& img,
      const ImageWrapper<ImageType::Depth>& depth,
      const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
      const ImageWrapper<ImageType::MotionMask>& motion_mask);
  // TODO:would be nice to have this in the base class (maybe )
  // TODO: this should really be a free function which is templated but issues
  // making it nice and elegant (easily template deducted + ImageCotnainer has
  // const variables in it) ideally also templated on the base class!!
  static ImageContainer::Ptr RectifyImages(
      ImageContainer::Ptr images, const UndistorterRectifier& undistorter);

 protected:
  using Base::clone;  // make private so we can implement this one!

  /**
   * @brief Static construction of a full image container and calls
   * validateSetup on the resulting object.
   *
   * Used by each public Create function to construct the underlying
   * ImageContainer. validateSetup must called after construction as it is a
   * virtual function.
   *
   * @param timestamp
   * @param frame_id
   * @param img
   * @param depth
   * @param optical_flow
   * @param semantic_mask
   * @param motion_mask
   * @param class_segmentation
   * @return ImageContainer::Ptr
   */
  static ImageContainer::Ptr Create(
      const Timestamp timestamp, const FrameId frame_id,
      const ImageWrapper<ImageType::RGBMono>& img,
      const ImageWrapper<ImageType::Depth>& depth,
      const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
      const ImageWrapper<ImageType::SemanticMask>& semantic_mask,
      const ImageWrapper<ImageType::MotionMask>& motion_mask,
      const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation);

 private:
  explicit ImageContainer(
      const Timestamp timestamp, const FrameId frame_id,
      const ImageWrapper<ImageType::RGBMono>& img,
      const ImageWrapper<ImageType::Depth>& depth,
      const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
      const ImageWrapper<ImageType::SemanticMask>& semantic_mask,
      const ImageWrapper<ImageType::MotionMask>& motion_mask,
      const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation);

  void validateSetup() const override;

 private:
  const Timestamp timestamp_;
  const FrameId frame_id_;
};

class MismatchedImageWrapperTypes : public DynosamException {
 public:
  MismatchedImageWrapperTypes(const std::string& requested_type,
                              const std::string& stored_type)
      : DynosamException("Attempting to retrieve value with requested type \"" +
                         requested_type + "\", but stored type is " +
                         stored_type),
        requested_type_name(requested_type),
        stored_type_name(stored_type) {}

  const std::string requested_type_name;
  const std::string stored_type_name;
};

class ImageKeyAlreadyExists : public DynosamException {
 public:
  ImageKeyAlreadyExists(const std::string& image_key)
      : DynosamException("Image key \"" + image_key +
                         "\" already exists in the container"),
        key(image_key) {}

  const std::string key;
};

class ImageKeyDoesNotExist : public DynosamException {
 public:
  ImageKeyDoesNotExist(const std::string& image_key)
      : DynosamException("Image key \"" + image_key +
                         "\" does not exist in the container"),
        key(image_key) {}

  const std::string key;
};

class ImageContainerBase {
 private:
  /**
   * @brief Internal data-structure to manage a key-image mapping pair
   *
   * The image is stored as a type erased ImageWrapper which is defined upon
   * construction and recoverable by dynamic pointer casting.
   *
   * Note: Move constructors and assignments are made explicit.
   * According to chat-gpt (seems to check out with the documentation):
   * "you declare any constructor, especially a templated constructor,
   *  the compiler does not generate copy/move constructors or assignments
   * unless you explicitly say".
   *
   * We can make them default to let the compiler figure out how to define them
   * since all types are well defined.
   *
   */
  struct KeyImagePair {
    const std::string key;
    const std::string stored_type;
    std::unique_ptr<ImageBase> ptr;

    template <typename IMAGETYPE>
    KeyImagePair(const std::string& name,
                 std::unique_ptr<ImageWrapper<IMAGETYPE>> image_wrapper)
        : key(name),
          stored_type(type_name<IMAGETYPE>()),
          ptr(std::move(image_wrapper)) {}

    // Move constructor
    KeyImagePair(KeyImagePair&&) noexcept = default;
    // Move assignment
    KeyImagePair& operator=(KeyImagePair&&) noexcept = default;

    template <typename IMAGETYPE>
    static KeyImagePair Create(const std::string& key, const cv::Mat& image) {
      return KeyImagePair(key,
                          std::make_unique<ImageWrapper<IMAGETYPE>>(image));
    }

    template <typename IMAGETYPE>
    const ImageWrapper<IMAGETYPE>& cast() const {
      auto* casted = dynamic_cast<ImageWrapper<IMAGETYPE>*>(ptr.get());
      if (!casted) {
        const auto requested_type = type_name<ImageType>();
        throw MismatchedImageWrapperTypes(requested_type, stored_type);
      }
      return *casted;
    }
  };

  gtsam::FastMap<std::string, KeyImagePair> images_;

 public:
  ImageContainerBase() {}

  template <typename IMAGETYPE>
  void add(const std::string& key, const cv::Mat& image) {
    if (exists(key)) {
      throw ImageKeyAlreadyExists(key);
    }

    KeyImagePair key_image = KeyImagePair::Create<IMAGETYPE>(key, image);
    images_.emplace(key, std::move(key_image));
  }

  template <typename IMAGETYPE>
  const ImageWrapper<IMAGETYPE>& at(const std::string& key) const {
    if (!exists(key)) {
      throw ImageKeyDoesNotExist(key);
    }

    const KeyImagePair& key_image = images_.at(key);
    return key_image.cast<IMAGETYPE>();
  }

  inline bool exists(const std::string& key) const {
    return images_.exists(key);
  }
  inline size_t size() const { return images_.size(); }
};

}  // namespace dyno

#include "dynosam/common/ImageContainer-inl.hpp"
