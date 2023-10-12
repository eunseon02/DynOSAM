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

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"

#include <filesystem>
#include <exception>
#include <fstream>

#include <glog/logging.h>

#include <opencv4/opencv2/opencv.hpp>

namespace dyno {

void throwExceptionIfPathInvalid(const std::string image_path) {
    namespace fs = std::filesystem;
    if(!fs::exists(image_path)) {
        throw std::runtime_error("Path does not exist: " + image_path);
    }
}

void loadRGB(const std::string& image_path, cv::Mat& img) {
    throwExceptionIfPathInvalid(image_path);
    img = cv::imread(image_path, cv::IMREAD_UNCHANGED);

}

void loadFlow(const std::string& image_path, cv::Mat& img) {
    throwExceptionIfPathInvalid(image_path);
    img = utils::readOpticalFlow(image_path);
}

void loadDepth(const std::string& image_path, cv::Mat& img) {
    throwExceptionIfPathInvalid(image_path);
    img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    img.convertTo(img, CV_64F);
}


void loadSemanticMask(const std::string& image_path, const cv::Size& size, cv::Mat& mask) {
    throwExceptionIfPathInvalid(image_path);
    CHECK(!size.empty());

    mask = cv::Mat(size, CV_32SC1);

    std::ifstream file_mask;
    file_mask.open(image_path.c_str());

    int count = 0;
    while (!file_mask.eof())
    {
        std::string s;
        getline(file_mask, s);
        if (!s.empty())
        {
            std::stringstream ss;
            ss << s;
            int tmp;
            for (int i = 0; i < mask.cols; ++i)
            {
                ss >> tmp;
                if (tmp != 0)
                {
                mask.at<int>(count, i) = tmp;
                }
                else
                {
                mask.at<int>(count, i) = 0;
                }
            }
            count++;
        }
    }

  file_mask.close();


}


} //dyno
