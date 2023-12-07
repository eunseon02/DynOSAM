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

#include <glog/logging.h>
#include <type_traits>


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

using OpticalFlow = gtsam::Point2; //! Observed optical flow vector (ordered x, y)

using Keypoint = gtsam::Point2;
using Keypoints = gtsam::Point2Vector; //! Vector of 2D keypoints using gtsam's definition for allocation

using KeypointCV = cv::KeyPoint;
using KeypointsCV = std::vector<KeypointCV>;

using Motion3 = gtsam::Pose3;
using MotionMap = std::unordered_map<TrackletId, Motion3>; //! Map of tracklet ids to Motion3 (gtsam::Pose3)


//! Expected label for the background in a semantic or motion mask
constexpr static ObjectId background_label = 0u;

enum KeyPointType {
    STATIC,
    DYNAMIC
};


enum ReferenceFrame {
  WORLD,
  CAMERA,
  OBJECT
};

// /**
//  * @brief Reference wrapper for a type T with operating casting
//  *
//  * @tparam T
//  */
// template<typename T>
// struct Ref {
//   using Type = T;

//   Type& value;
//   operator Type&() { return value; }
//   operator const Type&() const { return value; }
// };

/**
 * @brief Estimate with a reference frame and operator casting
 *
 * If the estimate E should be const, template on const E
 * If estimate E should be a reference, can template on E&
 *
 * @tparam E
 */
template<typename E>
struct ReferenceFrameEstimate {
  using Estimate = E;

  using ConstEstimate = std::add_const_t<Estimate>; //!	Const qualification of M. Regardless whether M is aleady const qualified.

  Estimate estimate_;
  const ReferenceFrame frame_;

  ReferenceFrameEstimate() {}
  ReferenceFrameEstimate(ConstEstimate& estimate, ReferenceFrame frame) : estimate_(estimate), frame_(frame) {}

  operator Estimate&() { return estimate_; }
  operator const Estimate&() const { return estimate_; }
  operator const ReferenceFrame&() const { return frame_; }

};

/**
 * @brief Map of key to an estimate containting a reference frame
 *
 * @tparam Key
 * @tparam Estimate
 */
template<typename Key, typename Estimate>
using EstimateMap = std::unordered_map<Key, ReferenceFrameEstimate<Estimate>>;

/// @brief Map of object ids to ReferenceFrameEstimate's of motions
using MotionEstimateMap = EstimateMap<ObjectId, Motion3>;




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


template<class Container>
inline std::string container_to_string(const Container& container) {
  std::stringstream ss;
  for(const auto& c : container) {
    ss << c << " ";
  }
  return ss.str();
}




} // namespace dyno
