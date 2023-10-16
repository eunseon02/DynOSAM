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

#include "dynosam/utils/Macros.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>

#include <opencv4/opencv2/opencv.hpp>
#include <vector>


namespace dyno
{

using Timestamp = double;
using Timestamps = Eigen::Matrix<Timestamp, 1, Eigen::Dynamic>;

using ObjectId = int;
using ObjectIds = std::vector<ObjectId>;

using FrameId = size_t;


//TODO: (jesse) should these be here?
struct ObjectPoseGT {
    DYNO_POINTER_TYPEDEFS(ObjectPoseGT)

    size_t frame_id;
    ObjectId object_id;
    gtsam::Pose3 L_camera; //object pose in camera frame
    cv::Rect bounding_box; //box of detection on image plane
};

struct GroundTruthInputPacket {
    DYNO_POINTER_TYPEDEFS(GroundTruthInputPacket)

    gtsam::Pose3 X_world; //camera pose in world frame
    std::vector<ObjectPoseGT> object_poses;
    Timestamp timestamp;
    size_t frame_id;
};


/**
 * @brief Get demangled class name
 *
 * @param name
 * @return std::string
 */
std::string demangle(const char* name);


/**
 * @brief Get demangled class name of type T from an instance
 * @tparam T
 * @param t
 * @return std::string
 */
template <class T>
std::string type_name(const T& t)
{
  return demangle(typeid(t).name());
}


/**
 * @brief Get a demangled class name of type T from a templated type
 *
 * @tparam T
 * @return std::string
 */
template <class T>
std::string type_name()
{
  return demangle(typeid(T).name());
}





} // namespace dyno
