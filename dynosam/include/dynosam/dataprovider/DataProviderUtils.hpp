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
#include <gtsam/geometry/Pose3.h>

namespace dyno {

void throwExceptionIfPathInvalid(const std::string image_path);

void loadRGB(const std::string& image_path, cv::Mat& img);

//CV_32F (float)
void loadFlow(const std::string& image_path, cv::Mat& img);

//CV_64F (double)
void loadDepth(const std::string& image_path, cv::Mat& img);

//CV_32SC1
void loadSemanticMask(const std::string& image_path, const cv::Size& size, cv::Mat& mask);

// /**
//  * @brief Converts RAW kitti (object) pose information into a full pose.
//  *
//  * The rotation information is expected to be in raw form (-pi to pi) which then be handled within the function
//  * No frame change is made.
//  *
//  * @param tx
//  * @param ty
//  * @param tz
//  * @param ry
//  * @param rx
//  * @param rz
//  * @return gtsam::Pose3
//  */
// gtsam::Pose3 constructkittiObjectPose(double tx, double ty, double tz, double ry, double rx, double rz);


} //dyno
