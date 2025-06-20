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

#include "dynosam/common/ImageContainer.hpp"

#include <exception>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/vision/UndistortRectifier.hpp"  //annoything this has to be here - better make (undisort iamges) into a free fucntion later!!
#include "dynosam/utils/OpenCVUtils.hpp"

namespace dyno {

ImageContainer ImageContainer::clone() const {
  ImageContainer container(this->frameId(), this->timestamp());

  // explicit deep copies for all key image pairs
  for (const auto& [k, v] : images_) {
    container.images_.emplace(k, v.clone());
  }

  return container;
}

const ImageWrapper<ImageType::RGBMono>& ImageContainer::rgb() const {
  return this->at<ImageType::RGBMono>(kRGB);
}
const ImageWrapper<ImageType::Depth>& ImageContainer::depth() const {
  return this->at<ImageType::Depth>(kDepth);
}
const ImageWrapper<ImageType::OpticalFlow>& ImageContainer::opticalFlow()
    const {
  return this->at<ImageType::OpticalFlow>(kOPticalFlow);
}
const ImageWrapper<ImageType::MotionMask>& ImageContainer::objectMotionMask()
    const {
  return this->at<ImageType::MotionMask>(kObjectMask);
}
const ImageWrapper<ImageType::RGBMono>& ImageContainer::rightRgb() const {
  return this->at<ImageType::RGBMono>(kRightRgb);
}

ImageWrapper<ImageType::RGBMono>& ImageContainer::rgb() {
  return this->at<ImageType::RGBMono>(kRGB);
}
ImageWrapper<ImageType::Depth>& ImageContainer::depth() {
  return this->at<ImageType::Depth>(kDepth);
}
ImageWrapper<ImageType::OpticalFlow>& ImageContainer::opticalFlow() {
  return this->at<ImageType::OpticalFlow>(kOPticalFlow);
}
ImageWrapper<ImageType::MotionMask>& ImageContainer::objectMotionMask() {
  return this->at<ImageType::MotionMask>(kObjectMask);
}
ImageWrapper<ImageType::RGBMono>& ImageContainer::rightRgb() {
  return this->at<ImageType::RGBMono>(kRightRgb);
}

ImageContainer& ImageContainer::rgb(const cv::Mat& image) {
  return this->add<ImageType::RGBMono>(kRGB, image);
}

ImageContainer& ImageContainer::depth(const cv::Mat& image) {
  return this->add<ImageType::Depth>(kDepth, image);
}

ImageContainer& ImageContainer::opticalFlow(const cv::Mat& image) {
  return this->add<ImageType::OpticalFlow>(kOPticalFlow, image);
}

ImageContainer& ImageContainer::objectMotionMask(const cv::Mat& image) {
  return this->add<ImageType::MotionMask>(kOPticalFlow, image);
}

ImageContainer& ImageContainer::rightRgb(const cv::Mat& image) {
  return this->add<ImageType::RGBMono>(kRightRgb, image);
}

std::string ImageContainer::toString() const {
  std::stringstream ss;
  ss << "[ ";
  for (const auto& [key, key_image_pair] : images_) {
    CHECK_NOTNULL(key_image_pair.ptr);
    ss << "{" << key << ": " << key_image_pair.ptr->toString() << "} ";
  }
  ss << "]";
  return ss.str();
}

std::ostream& operator<<(std::ostream& os,
                         const ImageContainer& image_container) {
  os << image_container.toString();
  return os;
}

ImageContainerDeprecate::Ptr ImageContainerDeprecate::Create(
    const Timestamp timestamp, const FrameId frame_id,
    const ImageWrapper<ImageType::RGBMono>& img,
    const ImageWrapper<ImageType::Depth>& depth,
    const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
    const ImageWrapper<ImageType::SemanticMask>& semantic_mask,
    const ImageWrapper<ImageType::MotionMask>& motion_mask,
    const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation) {
  std::shared_ptr<ImageContainerDeprecate> container(
      new ImageContainerDeprecate(timestamp, frame_id, img, depth, optical_flow,
                                  semantic_mask, motion_mask,
                                  class_segmentation));
  container->validateSetup();
  return container;
}

ImageContainerDeprecate::Ptr ImageContainerDeprecate::RectifyImages(
    ImageContainerDeprecate::Ptr images,
    const UndistorterRectifier& undistorter) {
  // do deep copy (the underlying ImageWrapper class has a deep copy in its copy
  // constructor so the image data is cloned)
  ImageContainerDeprecate::Ptr undistorted_images =
      std::make_shared<ImageContainerDeprecate>(*images);

  static constexpr size_t N = ImageContainerDeprecate::N;
  for (size_t i = 0; i < N; i++) {
    internal::select_apply<N>(i, [&](auto I) {
      using ImageType = typename ImageContainerDeprecate::ImageTypeStruct<I>;

      // get reference to image and modify in place
      // TODO: bug!!?
      cv::Mat& distorted_image = undistorted_images->template get<ImageType>();
      undistorter.undistortRectifyImage(distorted_image, distorted_image);
    });
  }

  return undistorted_images;
}

ImageContainerDeprecate::ImageContainerDeprecate(
    const Timestamp timestamp, const FrameId frame_id,
    const ImageWrapper<ImageType::RGBMono>& img,
    const ImageWrapper<ImageType::Depth>& depth,
    const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
    const ImageWrapper<ImageType::SemanticMask>& semantic_mask,
    const ImageWrapper<ImageType::MotionMask>& motion_mask,
    const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation)
    : Base(img, depth, optical_flow, semantic_mask, motion_mask,
           class_segmentation),
      timestamp_(timestamp),
      frame_id_(frame_id) {}

std::string ImageContainerDeprecate::toString() const {
  std::stringstream ss;
  ss << "Timestamp: " << timestamp_ << "\n";
  ss << "Frame Id: " << frame_id_ << "\n";

  const std::string image_config = isImageRGB() ? "RGB" : "Grey";
  ss << "Configuration: " << image_config << " Depth (" << hasDepth()
     << ") Semantic Mask (" << hasSemanticMask() << ") Motion Mask ("
     << hasMotionMask() << ") Class Segmentation (" << hasClassSegmentation()
     << ")"
     << "\n";

  return ss.str();
}

ImageContainerDeprecate::Ptr ImageContainerDeprecate::Create(
    const Timestamp timestamp, const FrameId frame_id,
    const ImageWrapper<ImageType::RGBMono>& img,
    const ImageWrapper<ImageType::Depth>& depth,
    const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
    const ImageWrapper<ImageType::SemanticMask>& semantic_mask) {
  return Create(timestamp, frame_id, img, depth, optical_flow, semantic_mask,
                ImageWrapper<ImageType::MotionMask>(),
                ImageWrapper<ImageType::ClassSegmentation>());
}

ImageContainerDeprecate::Ptr ImageContainerDeprecate::Create(
    const Timestamp timestamp, const FrameId frame_id,
    const ImageWrapper<ImageType::RGBMono>& img,
    const ImageWrapper<ImageType::Depth>& depth,
    const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
    const ImageWrapper<ImageType::MotionMask>& motion_mask) {
  return Create(timestamp, frame_id, img, depth, optical_flow,
                ImageWrapper<ImageType::SemanticMask>(), motion_mask,
                ImageWrapper<ImageType::ClassSegmentation>());
}

ImageContainerDeprecate::Ptr ImageContainerDeprecate::Create(
    const Timestamp timestamp, const FrameId frame_id,
    const ImageWrapper<ImageType::RGBMono>& img,
    const ImageWrapper<ImageType::Depth>& depth,
    const ImageWrapper<ImageType::OpticalFlow>& optical_flow,
    const ImageWrapper<ImageType::MotionMask>& motion_mask,
    const ImageWrapper<ImageType::ClassSegmentation>& class_segmentation) {
  return Create(timestamp, frame_id, img, depth, optical_flow,
                ImageWrapper<ImageType::SemanticMask>(), motion_mask,
                class_segmentation);
}

void ImageContainerDeprecate::validateSetup() const {
  // check image sizes are the same
  Base::validateSetup();

  // TODO: shoudl eventually change to exception
  CHECK(!getImage().empty()) << "RGBMono image must not be empty!";
  CHECK(!getOpticalFlow().empty()) << "OPticalFlow image must not be empty!";

  // must have at least one of semantic mask or motion mask
  CHECK(!getSemanticMask().empty() || !getMotionMask().empty())
      << "At least one of the masks (semantic or motion) must be valid. Both "
         "are empty";

  // should only provide one mask (for clarity - which one to use if both
  // given?)
  if (hasSemanticMask()) {
    CHECK(!hasMotionMask())
        << "Semantic mask has been provided, so motion mask should be empty!";
  }

  if (hasMotionMask()) {
    CHECK(!hasSemanticMask())
        << "Motion mask has been provided, so semantic mask should be empty!";
  }
}

}  // namespace dyno
