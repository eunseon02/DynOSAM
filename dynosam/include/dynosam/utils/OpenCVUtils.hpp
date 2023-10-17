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

cv::Mat concatenateImagesHorizontally(const cv::Mat& left_img, const cv::Mat& right_img);

cv::Mat concatenateImagesVertically(const cv::Mat& top_img, const cv::Mat& bottom_img);

void flowToRgb(const cv::Mat& flow, cv::Mat& rgb);

void semanticMaskToRgb(const cv::Mat& rgb, const cv::Mat& mask, cv::Mat& mask_viz);

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
