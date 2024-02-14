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


//TODO: should make variables private
template<typename VALUE>
struct TrackedValueStatus {
  using Value = VALUE;
  Value value_;
  FrameId frame_id_;
  TrackletId tracklet_id_;
  ObjectId label_; //! Will be 0 if background

  //Constexpr value used for the frame_id when it is NA (not applicable)
  //this may be the case when the TrackedValueStatus object represents a time-invariant
  //value (e.g a static point) and the value of the frame_id is therefore meaingless
  //NOTE:this is possibly dangerous if we are not careful with casting (e.g. int to FrameId) since
  //the overflow coulld land us at a meaningless frame
  static constexpr auto MeaninglessFrame = std::numeric_limits<FrameId>::max();

  //not we will have implicit casting of all other frame_ids to int here which could be dangerous
  TrackedValueStatus(
    const Value& value,
    FrameId frame_id,
    TrackletId tracklet_id,
    ObjectId label)
    : value_(value),
      frame_id_(frame_id),
      tracklet_id_(tracklet_id),
      label_(label) {}

  virtual ~TrackedValueStatus() = default;

  inline bool isStatic() const {
    return label_ == background_label;
  }

  /**
   * @brief Returns true if frame_id is equal to MeaninglessFrame (e.g. std::numeric_limits<FrameId>::quiet_NaN)
   * and should indicatate that the value represented by this TrackedValueStatus is time-invariant, e.g. for a static point
   * which does not change overtime.
   * @return true
   * @return false
   */
  inline bool isTimeInvariant() const {
    return frame_id_ == MeaninglessFrame;
  }
};

/// @brief Check if derived DERIVEDSTATUS us in factor derived from Status<Value>
/// @tparam DERIVEDSTATUS derived type
/// @tparam VALUE expected value to templated base Status on
template<typename DERIVEDSTATUS, typename VALUE>
inline constexpr bool IsDerivedTrackedValueStatus = std::is_base_of_v<TrackedValueStatus<VALUE>, DERIVEDSTATUS>;


/**
 * @brief Metadata of a landmark. Includes type (static/dynamic) and label.
 *
 * Label may be background at which point the KeyPointType should be background_label
 * Also includes information on how the landamrk was estimated, age etc...
 *
 */
struct LandmarkStatus : public TrackedValueStatus<Landmark> {
    using Base = TrackedValueStatus<Landmark>;
    using Base::Value;
    enum Method { MEASURED, TRIANGULATED, OPTIMIZED };
    Method method_;

    /**
     * @brief Construct a new Landmark Status object
     *
     * @param lmk
     * @param frame_id
     * @param tracklet_id
     * @param label
     * @param method
     */
    LandmarkStatus(const Landmark& lmk, FrameId frame_id, TrackletId tracklet_id,  ObjectId label, Method method)
    : Base(lmk, frame_id, tracklet_id, label), method_(method) {}

    inline static LandmarkStatus Static(const Landmark& lmk, FrameId frame_id, TrackletId tracklet_id, Method method) {
      return LandmarkStatus(lmk, frame_id, tracklet_id, background_label, method);
    }

    inline static LandmarkStatus Dynamic(const Landmark& lmk, FrameId frame_id, TrackletId tracklet_id, ObjectId label, Method method) {
      CHECK(label != background_label);
      return LandmarkStatus(lmk, frame_id, tracklet_id, label, method);
    }


};

/**
 * @brief Metadata of a keypoint. Includes type (static/dynamic) and label.
 *
 * Label may be background at which point the KeyPointType should be background_label
 *
 */
struct KeypointStatus : public TrackedValueStatus<Keypoint> {
  using Base = TrackedValueStatus<Keypoint>;
  using Base::Value;
  KeyPointType kp_type_;

  /**
   * @brief Construct a new Keypoint Status object
   *
   * @param kp
   * @param frame_id
   * @param tracklet_id
   * @param label
   * @param kp_type
   */
  KeypointStatus(const Keypoint& kp, FrameId frame_id, TrackletId tracklet_id,  ObjectId label,KeyPointType kp_type)
  : Base(kp, frame_id, tracklet_id, label), kp_type_(kp_type) {}

  inline static KeypointStatus Static(const Keypoint& kp, FrameId frame_id, TrackletId tracklet_id) {
    return KeypointStatus(kp, frame_id, background_label, tracklet_id, KeyPointType::STATIC);
  }

  inline static KeypointStatus Dynamic(const Keypoint& kp, FrameId frame_id, TrackletId tracklet_id,  ObjectId label) {
    CHECK(label != background_label);
    return KeypointStatus(kp, frame_id, tracklet_id, label, KeyPointType::DYNAMIC);
  }
};


template<class DERIVEDSTATUS, typename VALUE = typename DERIVEDSTATUS::Value>
struct IsStatus {
  static_assert(IsDerivedTrackedValueStatus<DERIVEDSTATUS, VALUE>, "DERIVEDSTATUS does not derive from Status<Value>");
  using type = DERIVEDSTATUS;
  using value = VALUE;
};

template<typename DERIVEDSTATUS, typename VALUE = typename DERIVEDSTATUS::Value>
class GenericTrackedStatusVector : public std::vector<DERIVEDSTATUS> {
public:

  //check if the DERIVEDSTATUS meets requirements
  //and alias to the value type of the status
  using Value = typename IsStatus<DERIVEDSTATUS, VALUE>::value;

  using Base = std::vector<DERIVEDSTATUS>;
  using Base::Base;

  /** Conversion to a standard STL container */
  operator std::vector<DERIVEDSTATUS>() const {
    return std::vector<DERIVEDSTATUS>(this->begin(), this->end());
  }

};

using StatusLandmarkEstimate = IsStatus<LandmarkStatus>::type;
/// @brief A vector of StatusLandmarkEstimate
using StatusLandmarkEstimates = GenericTrackedStatusVector<LandmarkStatus>;


using StatusKeypointMeasurement = IsStatus<KeypointStatus>::type;
/// @brief A vector of StatusKeypointMeasurements
using StatusKeypointMeasurements = GenericTrackedStatusVector<KeypointStatus>;



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
