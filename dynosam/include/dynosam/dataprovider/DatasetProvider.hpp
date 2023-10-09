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

#include "dynosam/dataprovider/DataProvider.hpp"

#include <gtsam/geometry/Pose3.h>
#include <opencv4/opencv2/opencv.hpp>
#include <functional>

namespace dyno {

//! Given a fully qualified image path, load an image
using LoadImageFunction = std::function<cv::Mat(const std::string&)>;



/**
 * @brief This loads data from a given source (path to sequence) using a set of given
 * PerFolderDataLoaders which know how to load image data from a subfolder in the sequence
 *
 *
 */
class DatasetProvider : public DataProvider {

public:
    struct ImageLoaderDetails {
        std::string folder_name_;
        std::string file_suffix_;
    };
    //! A pair containing a ImageLoaderDetails and a function that should load an image from within the function
    // The folder name (.first) only makes sense in the context of the DatasetProvider::path_to_dataset which should
    // be the parent folder. The fully qualified name for the loading function is constructed from the ImageLoaderDetails
    // and the dataset path
    using ImageLoader = std::pair<ImageLoaderDetails, LoadImageFunction>;

    DatasetProvider(const std::string& path_to_dataset, std::initializer_list<ImageLoader> image_loaders = {});

protected:
    virtual gtsam::Pose3 parsePose(const std::vector<double>& data) const = 0;

protected:
  const std::string path_to_sequence_;
  const std::vector<ImageLoader> image_loaders_;

  std::vector<gtsam::Pose3> camera_pose_gt_;
  std::vector<Timestamp> timestamps_;

private:
  // should correspond with the frame_id of the data
  size_t index_ = 0;



};

} //dyno
