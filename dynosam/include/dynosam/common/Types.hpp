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
#include <optional>


namespace dyno
{

static constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();

using Timestamp = double;
using Timestamps = Eigen::Matrix<Timestamp, 1, Eigen::Dynamic>;

using ObjectId = int;
using ObjectIds = std::vector<ObjectId>;

using FrameId = size_t;
using Depth = double;
using Depths = std::vector<double>;

using TrackletId = long int;  // -1 for invalid landmarks. // int would be too
                            // small if it is 16 bits!
using TrackletIds = std::vector<TrackletId>;

using BearingVectors =
    std::vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>>;

using Landmark = gtsam::Point3;
using Landmarks = gtsam::Point3Vector; //! Vector of Landmarks using gtsam's definition for allocation
using LandmarkMap = std::unordered_map<TrackletId, Landmark>;

using Keypoint = gtsam::Point2;
using Keypoints = gtsam::Point2Vector; //! Vector of 2D keypoints using gtsam's definition for allocation

using KeypointCV = cv::KeyPoint;
using KeypointsCV = std::vector<KeypointCV>;

//Optional string that can be modified directly (similar to old-stype boost::optional)
//to access the mutable reference the internal string must be accessed with get()
// e.g. optional->get() = "updated string value";
//This is to overcome the fact that the stdlib does not support std::optional<T&> directly
using OptionalString = std::optional<std::reference_wrapper<std::string>>;

/**
 * @brief Get demangled class name
 *
 * @param name
 * @return std::string
 */
std::string demangle(const char* name);

template<typename T>
std::string to_string(const T& t);

template<typename Input, typename Output>
bool convert(const Input&, Output&);


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
