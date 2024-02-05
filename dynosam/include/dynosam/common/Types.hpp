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

#include <gtsam/base/FastMap.h>

#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <optional>

#include <glog/logging.h>
#include <type_traits>
#include <string_view>


namespace dyno
{

template<typename T>
struct traits;

static constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();

using Timestamp = double;
using Timestamps = Eigen::Matrix<Timestamp, 1, Eigen::Dynamic>;

using ObjectId = int;
using ObjectIds = std::vector<ObjectId>;

using FrameId = size_t;
using FrameIds = std::vector<FrameId>;

using Depth = double;
using Depths = std::vector<double>;

using TrackletId = long int;  // -1 for invalid landmarks. // int would be too
                            // small if it is 16 bits!
using TrackletIds = std::vector<TrackletId>;

using BearingVectors =
    std::vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>>;

using Landmark = gtsam::Point3;
using Landmarks = gtsam::Point3Vector; //! Vector of Landmarks using gtsam's definition for allocation
using LandmarkMap = gtsam::FastMap<TrackletId, Landmark>;

using OpticalFlow = gtsam::Point2; //! Observed optical flow vector (ordered x, y)

using Keypoint = gtsam::Point2;
using Keypoints = gtsam::Point2Vector; //! Vector of 2D keypoints using gtsam's definition for allocation

using KeypointCV = cv::KeyPoint;
using KeypointsCV = std::vector<KeypointCV>;

using Motion3 = gtsam::Pose3;
using MotionMap = gtsam::FastMap<TrackletId, Motion3>; //! Map of tracklet ids to Motion3 (gtsam::Pose3)


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




/**
 * @brief Metadata of a landmark. Includes type (static/dynamic) and label.
 *
 * Label may be background at which point the KeyPointType should be background_label
 * Also includes information on how the landamrk was estimated, age etc...
 *
 */
struct LandmarkStatus {

    enum Method { MEASURED, TRIANGULATED, OPTIMIZED };

    ObjectId label_; //! Will be 0 if background
    Method method_; //! How the landmark was constructed

    LandmarkStatus(ObjectId label, Method method) : label_(label), method_(method) {}

    inline static LandmarkStatus Static(Method method) {
      return LandmarkStatus(background_label, method);
    }


};

/**
 * @brief Metadata of a keypoint. Includes type (static/dynamic) and label.
 *
 * Label may be background at which point the KeyPointType should be background_label
 *
 */
struct KeypointStatus {
  const KeyPointType kp_type_;
  const ObjectId label_; //! Will be 0 if background

  KeypointStatus(KeyPointType kp_type, ObjectId label) : kp_type_(kp_type), label_(label) {}

  inline bool isStatic() const {
    const bool is_static = (kp_type_ == KeyPointType::STATIC);
    {
      //sanity check
      if(is_static) CHECK_EQ(label_, background_label) << "Keypoint Type is STATIC but label is not background label (" << background_label << ")";
    }
    return is_static;
  }

  inline static KeypointStatus Static() {
    return KeypointStatus(KeyPointType::STATIC, background_label);
  }

  inline static KeypointStatus Dynamic(ObjectId label) {
    CHECK(label != background_label);
    return KeypointStatus(KeyPointType::DYNAMIC, label);
  }
};

/// @brief A generic tracked estimate/measurement, associated with a trackletId
/// @tparam M An estimate or measurement
template<typename M>
using GenericMTrack = std::pair<TrackletId, M>;


/// @brief A generic tracked tracked estimate/measurement, associated with some
/// status type
/// @tparam MStatus An associated status obejct
/// @tparam M An estimate or measurement
template<typename MStatus, typename M>
using GenericStatusMEstimate = std::pair<MStatus, GenericMTrack<M>>;

/// @brief A vector of GenericStatusMEstimate, indicating tracked estimates/measurements
/// and associated status types
/// @tparam MStatus An associated status obejct
/// @tparam M An estimate or measurement
template<typename MStatus, typename M>
using GenericStatusMEstimates = std::vector<GenericStatusMEstimate<MStatus, M>>;

/// @brief A pair relating a tracklet ID with an estiamted landmark
using LandmarkEstimate = GenericMTrack<Landmark>;
/// @brief A pair relating a Landmark estimate (TrackletId + Landmark) with a status - inidicating type and the object label
using StatusLandmarkEstimate = GenericStatusMEstimate<LandmarkStatus, Landmark>;
/// @brief A vector of StatusLandmarkEstimate
using StatusLandmarkEstimates = GenericStatusMEstimates<LandmarkStatus, Landmark>;



/// @brief A pair relating a tracklet ID with an observed keypoint
using KeypointMeasurement = GenericMTrack<Keypoint>;
/// @brief A pair relating a Keypoint measurement (TrackletId + Keypoint) with a status - inidicating the keypoint type and the object label
using StatusKeypointMeasurement = GenericStatusMEstimate<KeypointStatus, Keypoint>;
/// @brief A vector of StatusKeypointMeasurements
using StatusKeypointMeasurements = GenericStatusMEstimates<KeypointStatus, Keypoint>;


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
using EstimateMap = gtsam::FastMap<Key, ReferenceFrameEstimate<Estimate>>;

/// @brief Map of object ids to ReferenceFrameEstimate's of motions
using MotionEstimateMap = EstimateMap<ObjectId, Motion3>;

/// @brief Map of object poses per frame per object
using ObjectPoseMap = gtsam::FastMap<ObjectId, gtsam::FastMap<FrameId, gtsam::Pose3>>;




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


template<class Container>
inline std::string container_to_string(const Container& container) {
  std::stringstream ss;
  for(const auto& c : container) {
    ss << c << " ";
  }
  return ss.str();
}



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
constexpr std::string type_name()
{
  return demangle(typeid(T).name());
}



// template<typename T>
// struct io_traits {

//   //T must be inside dyno namespace to work
//   static std::string ToString(const T& t, const std::string& s = "") {
//     using std::to_string;
//     // This calls std::to_string if std::to_string is defined;
//     // Otherwise it uses ADL to find the correct to_string function
//     const std::string repr = to_string(t);

//     if(s.empty()) return repr;
//     else return s + " " + repr;
//   }

//   static void Print(const T& t, const std::string& s = "") {
//     const std::string str = ToString(t, s);
//     std::cout << str << std::endl;
//   }

// };


// template<typename T>
// struct traits : public io_traits<T> {};



} // namespace dyno
