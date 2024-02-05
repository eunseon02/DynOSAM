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


#include <opencv4/opencv2/opencv.hpp>

#define CHECK_MAT_TYPES(mat1, mat2)                                                                                    \
  using namespace dyno::utils;                                                                                     \
  CHECK_EQ(mat1.type(), mat2.type()) << "Matricies should be of the same type ( " << cvTypeToString(mat1.type())       \
                                     << " vs. " << cvTypeToString(mat2.type()) << ")."


namespace dyno {
namespace utils {

void drawCircleInPlace(cv::Mat& img, const cv::Point2d& point, const cv::Scalar& colour, const double msize = 0.4);

std::string cvTypeToString(int type);
std::string cvTypeToString(const cv::Mat& mat);

// Compares two opencv matrices up to a certain tolerance error.
bool compareCvMatsUpToTol(const cv::Mat& mat1, const cv::Mat& mat2, const double& tol = 1e-7);

cv::Mat concatenateImagesHorizontally(const cv::Mat& left_img, const cv::Mat& right_img);

cv::Mat concatenateImagesVertically(const cv::Mat& top_img, const cv::Mat& bottom_img);

void flowToRgb(const cv::Mat& flow, cv::Mat& rgb);


/**
 * @brief Given an image that operates like a mask (ie. is a single channel image where each pixel value correspondes to a label of some type)
 * draws the mask over an input image.
 *
 * The value at each pixel (labels) are treated like object labels which are used to generate a unique colour for the mask.
 * The background label is used to indicate which pixel value should be ignored - this could be the background or simply unknown pixel values
 *
 * @param mask
 * @param background_label
 * @param rgb
 * @return cv::Mat
 */
cv::Mat labelMaskToRGB(const cv::Mat& mask, int background_label, const cv::Mat& rgb);

/**
 * @brief Same as labelMaskToRGB(mask, background label) but instead draws the mask over a black image so only the mask values are shown
 *
 * @param mask
 * @param background_label
 * @return cv::Mat
 */
cv::Mat labelMaskToRGB(const cv::Mat& mask, int background_label);


void drawLabeledBoundingBox(const cv::Mat& image, const std::string& label, const cv::Scalar& colour, const cv::Rect& bounding_box);


/**
 * I have absolutely no idea why but OpenCV seemds to have removed support for the read/write optical flow functions in 4.x
 * I have taken this implementation from OpenCV 3.4 (https://github.com/opencv/opencv_contrib/blob/3.4/modules/optflow/src/optical_flow_io.cpp)
*/


/**
 * @brief The function readOpticalFlow loads a flow field from a file and returns it as a single matrix.
 * Resulting Mat has a type CV_32FC2 - floating-point, 2-channel. First channel corresponds to the
 * flow in the horizontal direction (u), second - vertical (v).
 *
 * @param path
 * @return cv::Mat
 */
cv::Mat readOpticalFlow( const std::string& path );

/**
 * @brief The function stores a flow field in a file, returns true on success, false otherwise.
 * The flow field must be a 2-channel, floating-point matrix (CV_32FC2). First channel corresponds
 * to the flow in the horizontal direction (u), second - vertical (v).
 *
 * @param path
 * @param flow
 * @return true
 * @return false
 */
bool writeOpticalFlow( const std::string& path, const cv::Mat& flow);


}
}
