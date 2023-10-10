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

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/dataprovider/DataProviderUtils.hpp"

namespace dyno {

std::string RGBDataFolder::getFolderName() const {
    return "image_0";
}

cv::Mat RGBDataFolder::getItem(size_t idx) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << idx;
    const std::string file_path = (std::string)absolute_folder_path_  + "/" + ss.str() + ".png";

    cv::Mat rgb;
    loadRGB(file_path, rgb);

    CHECK(!rgb.empty());
    return rgb;
}


std::string OpticalFlowDataFolder::getFolderName() const {
    return "flow";
}

cv::Mat OpticalFlowDataFolder::getItem(size_t idx) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << idx;
    const std::string file_path = (std::string)absolute_folder_path_  + "/" + ss.str() + ".flo";

    cv::Mat flow;
    loadFlow(file_path, flow);

    CHECK(!flow.empty());
    return flow;
}



cv::Mat DepthDataFolder::getItem(size_t idx) {
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(6) << idx;
  const std::string file_path = (std::string)absolute_folder_path_  + "/" + ss.str() + ".png";

  cv::Mat depth;
  loadDepth(file_path, depth);

  CHECK(!depth.empty());
  return depth;
}

std::string TimestampFile::getFolderName() const {
    return "times.txt";
}

double TimestampFile::getItem(size_t idx) {
    return times.at(idx);
}

void TimestampFile::onPathInit() {
    std::ifstream times_stream((std::string)absolute_folder_path_, std::ios::in);

    while (!times_stream.eof())
  {
    std::string s;
    getline(times_stream, s);
    if (!s.empty())
    {
      std::stringstream ss;
      ss << s;
      double t;
      ss >> t;
      times.push_back(t);
    }
  }
  times_stream.close();


}


} //dyno
