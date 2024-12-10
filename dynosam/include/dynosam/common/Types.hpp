/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include <glog/logging.h>
#include <gtsam/base/FastMap.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>

#include <boost/optional.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

#include "dynosam/common/Flags.hpp"  //for common glags DECLARATIONS
#include "dynosam/utils/Macros.hpp"

/** \mainpage C++ API documentation
 *  DynoSAM is a C++ library for Dynamic Visual SLAM.
 *
 * TODO: add link back to project page when it is live....
 *
 */

namespace dyno {

template <typename T>
struct traits;

static constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();

using Timestamp = double;
using Timestamps = Eigen::Matrix<Timestamp, 1, Eigen::Dynamic>;

/// @brief Discrete object id, j.
using ObjectId = int;
using ObjectIds = std::vector<ObjectId>;

/// @brief Discrete time-step k.
using FrameId = size_t;
using FrameIds = std::vector<FrameId>;

using Depth = double;
using Depths = std::vector<double>;

/// @brief Unique tracklet/landmark id, i.
using TrackletId = long int;  // -1 for invalid landmarks. // int would be too
                              // small if it is 16 bits!
using TrackletIds = std::vector<TrackletId>;

using BearingVectors =
    std::vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>>;

using Landmark = gtsam::Point3;
using Landmarks = gtsam::Point3Vector;  //! Vector of Landmarks using gtsam's
                                        //! definition for allocation
using LandmarkMap = gtsam::FastMap<TrackletId, Landmark>;

using OpticalFlow =
    gtsam::Point2;  //! Observed optical flow vector (ordered x, y)

using Keypoint = gtsam::Point2;
using Keypoints = gtsam::Point2Vector;  //! Vector of 2D keypoints using gtsam's
                                        //! definition for allocation

using KeypointCV = cv::KeyPoint;
using KeypointsCV = std::vector<KeypointCV>;

using Motion3 = gtsam::Pose3;
using MotionMap =
    gtsam::FastMap<TrackletId,
                   Motion3>;  //! Map of tracklet ids to Motion3 (gtsam::Pose3)

struct LandmarkKeypoint {
  LandmarkKeypoint() = default;
  LandmarkKeypoint(const Landmark& l, const Keypoint& kp)
      : landmark(l), keypoint(kp) {}

  Landmark landmark;
  Keypoint keypoint;
};

/**
 * @brief Keypoint with depth
 *
 */
struct KeypointDepth {
 public:
  KeypointDepth() = default;
  KeypointDepth(const Keypoint& p, const Depth& d) : keypoint(p), depth(d) {}

 public:
  Keypoint keypoint;
  Depth depth;
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
std::string type_name(const T& t) {
  return demangle(typeid(t).name());
}

/**
 * @brief Get a demangled class name of type T from a templated type
 *
 * @tparam T
 * @return std::string
 */
template <class T>
constexpr std::string type_name() {
  return demangle(typeid(T).name());
}

//! Expected label for the background in a semantic or motion mask
constexpr static ObjectId background_label = 0u;

enum KeyPointType { STATIC, DYNAMIC };

enum ReferenceFrame { GLOBAL, LOCAL, OBJECT };

// TODO: depricate and replace with the IndexedPose thing!!!
/**
 * @brief Estimate with a reference frame and operator casting
 *
 * If the estimate E should be const, template on const E
 * If estimate E should be a reference, can template on E&
 *
 * @tparam E
 */
template <typename E>
struct ReferenceFrameValue {
  using Estimate = E;

  using ConstEstimate =
      std::add_const_t<Estimate>;  //!	Const qualification of M. Regardless
                                   //! whether M is aleady const qualified.

  Estimate estimate_;
  ReferenceFrame frame_;

  ReferenceFrameValue() {}
  ReferenceFrameValue(ConstEstimate& estimate, ReferenceFrame frame)
      : estimate_(estimate), frame_(frame) {}

  operator Estimate&() { return estimate_; }
  operator const Estimate&() const { return estimate_; }
  operator const ReferenceFrame&() const { return frame_; }
};

template <typename VALUE>
class TrackedValueStatus {
 public:
  using Value = VALUE;
  using This = TrackedValueStatus<Value>;

  // Constexpr value used for the frame_id when it is NA (not applicable)
  // this may be the case when the TrackedValueStatus object represents a
  // time-invariant value (e.g a static point) and the value of the frame_id is
  // therefore meaingless NOTE:this is possibly dangerous if we are not careful
  // with casting (e.g. int to FrameId) since the overflow coulld land us at a
  // meaningless frame
  static constexpr auto MeaninglessFrame = std::numeric_limits<FrameId>::max();

  // for IO
  TrackedValueStatus() {}

  // not we will have implicit casting of all other frame_ids to int here which
  // could be dangerous
  TrackedValueStatus(const Value& value, FrameId frame_id,
                     TrackletId tracklet_id, ObjectId label,
                     ReferenceFrame reference_frame)
      : value_(value, reference_frame),
        frame_id_(frame_id),
        tracklet_id_(tracklet_id),
        label_(label) {}

  virtual ~TrackedValueStatus() = default;

  const Value& value() const { return value_; }
  Value& value() { return value_; }

  TrackletId trackletId() const { return tracklet_id_; }
  ObjectId objectId() const { return label_; }
  FrameId frameId() const { return frame_id_; }

  const ReferenceFrame& referenceFrame() const { return value_; }
  ReferenceFrame& referenceFrame() { return value_; }

  ReferenceFrameValue<Value>& referenceFrameValue() { return value_; }
  const ReferenceFrameValue<Value>& referenceFrameValue() const {
    return value_;
  }

  bool operator==(const This& other) const {
    return gtsam::traits<Value>::Equals(this->value(), other.value()) &&
           frame_id_ == other.frame_id_ && tracklet_id_ == other.tracklet_id_ &&
           label_ == other.label_ &&
           this->referenceFrame() == other.referenceFrame();
  }

  friend std::ostream& operator<<(std::ostream& os, const This& t) {
    os << type_name<Value>() << ": " << t.value() << "\n";
    os << "frame id: " << t.frameId() << "\n";
    os << "tracklet id: " << t.trackletId() << "\n";
    os << "object id: " << t.objectId() << "\n";
    return os;
  }

  inline bool isStatic() const { return label_ == background_label; }

  /**
   * @brief Returns true if frame_id is equal to MeaninglessFrame (e.g.
   * std::numeric_limits<FrameId>::quiet_NaN) and should indicatate that the
   * value represented by this TrackedValueStatus is time-invariant, e.g. for a
   * static point which does not change overtime.
   * @return true
   * @return false
   */
  inline bool isTimeInvariant() const { return frame_id_ == MeaninglessFrame; }

 protected:
  ReferenceFrameValue<Value> value_;
  FrameId frame_id_;
  TrackletId tracklet_id_;
  ObjectId label_;  //! Will be 0 if background
};

/// @brief Check if derived DERIVEDSTATUS us in factor derived from
/// Status<Value>
/// @tparam DERIVEDSTATUS derived type
/// @tparam VALUE expected value to templated base Status on
template <typename DERIVEDSTATUS, typename VALUE>
inline constexpr bool IsDerivedTrackedValueStatus =
    std::is_base_of_v<TrackedValueStatus<VALUE>, DERIVEDSTATUS>;

/**
 * @brief Defines a single visual measurement with tracking label, object id,
 * frame id and measurement.
 *
 * @tparam T
 */
template <typename T>
struct VisualMeasurementStatus : public TrackedValueStatus<T> {
  using Base = TrackedValueStatus<T>;
  using Base::Value;
  //! dimension of the measurement. Must have gtsam::dimension traits
  constexpr static int dim = gtsam::traits<T>::dimension;

  VisualMeasurementStatus() = default;
  VisualMeasurementStatus(const Base& base)
      : VisualMeasurementStatus(base.value(), base.frameId(), base.trackletId(),
                                base.objectId(), base.referenceFrame()) {}

  VisualMeasurementStatus(const T& m, FrameId frame_id, TrackletId tracklet_id,
                          ObjectId label, ReferenceFrame reference_frame)
      : Base(m, frame_id, tracklet_id, label, reference_frame) {}

  inline static VisualMeasurementStatus Static(const T& m, FrameId frame_id,
                                               TrackletId tracklet_id,
                                               ReferenceFrame reference_frame) {
    return VisualMeasurementStatus(m, frame_id, tracklet_id, background_label,
                                   reference_frame);
  }

  inline static VisualMeasurementStatus Dynamic(
      const T& m, FrameId frame_id, TrackletId tracklet_id, ObjectId label,
      ReferenceFrame reference_frame) {
    CHECK(label != background_label);
    return VisualMeasurementStatus(m, frame_id, tracklet_id, label,
                                   reference_frame);
  }

  inline static VisualMeasurementStatus StaticInLocal(const T& m,
                                                      FrameId frame_id,
                                                      TrackletId tracklet_id) {
    return VisualMeasurementStatus(m, frame_id, tracklet_id, background_label,
                                   ReferenceFrame::LOCAL);
  }

  inline static VisualMeasurementStatus DynamicInLocal(const T& m,
                                                       FrameId frame_id,
                                                       TrackletId tracklet_id,
                                                       ObjectId label) {
    CHECK(label != background_label);
    return VisualMeasurementStatus(m, frame_id, tracklet_id, label,
                                   ReferenceFrame::LOCAL);
  }

  inline static VisualMeasurementStatus StaticInGlobal(const T& m,
                                                       FrameId frame_id,
                                                       TrackletId tracklet_id) {
    return VisualMeasurementStatus(m, frame_id, tracklet_id, background_label,
                                   ReferenceFrame::GLOBAL);
  }

  inline static VisualMeasurementStatus DynamicInGLobal(const T& m,
                                                        FrameId frame_id,
                                                        TrackletId tracklet_id,
                                                        ObjectId label) {
    CHECK(label != background_label);
    return VisualMeasurementStatus(m, frame_id, tracklet_id, label,
                                   ReferenceFrame::GLOBAL);
  }
};

/**
 * @brief Defines a measurement with associated covariance matrix.
 *
 * We define the dimensions of this type and provide an equals function to make
 * it compatible with gtsam::traits<T>::dimension and gtsam::traits<T>::Equals.
 *
 * //TODO: maybe change covariance to just gtsam::NoiseModel?
 *
 * @tparam T measurement type
 * @tparam D dimension of the measurement. Used to construct the covariance
 * matrix.
 */
template <typename T, int D = gtsam::traits<T>::dimension>
struct MeasurementWithCovariance {
  using This = MeasurementWithCovariance<T, D>;
  using Covariance = Eigen::Matrix<double, D, D>;
  using Sigmas = Eigen::Matrix<double, D, 1>;
  //! Need to have dimension to satisfy gtsam::dimemsions
  enum { dimension = D };
  T measurement;
  Covariance covariance;

  MeasurementWithCovariance() = default;
  MeasurementWithCovariance(const T& m) : measurement(m), covariance() {}
  MeasurementWithCovariance(const T& m, const Covariance& cov)
      : measurement(m), covariance(cov) {}
  MeasurementWithCovariance(const T& m, const Sigmas& sigmas)
      : measurement(m), covariance(sigmas.matrix().asDiagonal()) {}
  MeasurementWithCovariance(const std::pair<T, Covariance>& pair)
      : MeasurementWithCovariance(pair.first, pair.second) {}

  operator const T&() const { return measurement; }
  operator const Covariance&() const { return covariance; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const This& measurement_with_cov) {
    os << "m: " << measurement_with_cov.measurement
       << ", cov: " << measurement_with_cov.covariance;
    return os;
  }

  inline void print(const std::string& s = "") const {
    std::cout << s << *this << std::endl;
  }

  inline bool equals(const MeasurementWithCovariance& other,
                     double tol = 1e-8) const {
    return gtsam::traits<T>::Equals(measurement, other.measurement, tol) &&
           gtsam::traits<Covariance>::Equals(covariance, other.covariance, tol);
  }
};

/// @brief Alias to a visual measurement with a fixed-sized covariance matrix.
/// @tparam T Measurement type (e.g. 2D keypoint, 3D landmark)
template <typename T>
using VisualMeasurementWithCovStatus =
    VisualMeasurementStatus<MeasurementWithCovariance<T>>;

/// @brief Alias for a landmark measurement with 3x3 covariance.
typedef VisualMeasurementWithCovStatus<Landmark> LandmarkStatus;
/// @brief Alias for a keypoint measurement with 2x2 covariance.
typedef VisualMeasurementWithCovStatus<Keypoint> KeypointStatus;
/// @brief Alias for a depth measurement with scalar covariance.
typedef VisualMeasurementWithCovStatus<Depth> DepthStatus;

using LandmarkKeypointStatus = TrackedValueStatus<LandmarkKeypoint>;

template <class DERIVEDSTATUS, typename VALUE = typename DERIVEDSTATUS::Value>
struct IsStatus {
  static_assert(IsDerivedTrackedValueStatus<DERIVEDSTATUS, VALUE>,
                "DERIVEDSTATUS does not derive from Status<Value>");
  using type = DERIVEDSTATUS;
  using value = VALUE;
};

template <typename DERIVEDSTATUS,
          typename VALUE = typename DERIVEDSTATUS::Value>
class GenericTrackedStatusVector : public std::vector<DERIVEDSTATUS> {
 public:
  // check if the DERIVEDSTATUS meets requirements
  // and alias to the value type of the status
  using Value = typename IsStatus<DERIVEDSTATUS, VALUE>::value;
  using This = GenericTrackedStatusVector<DERIVEDSTATUS, VALUE>;

  using Base = std::vector<DERIVEDSTATUS>;
  using Base::Base;

  // for IO
  GenericTrackedStatusVector() {}

  /** Conversion to a standard STL container */
  operator std::vector<DERIVEDSTATUS>() const {
    return std::vector<DERIVEDSTATUS>(this->begin(), this->end());
  }

  This& operator+=(const GenericTrackedStatusVector& rhs) {
    this->insert(this->end(), rhs.begin(), rhs.end());
    return *this;
  }
};

/// @brief A vector of LandmarkStatus
typedef GenericTrackedStatusVector<LandmarkStatus> StatusLandmarkVector;
/// @brief A vector of KeypointStatus
typedef GenericTrackedStatusVector<KeypointStatus> StatusKeypointVector;

/**
 * @brief Map of key to an estimate containting a reference frame
 *
 * @tparam Key
 * @tparam Estimate
 */
template <typename Key, typename Estimate>
using EstimateMap = gtsam::FastMap<Key, ReferenceFrameValue<Estimate>>;

/// @brief Map of object ids to ReferenceFrameValue's of motions
using MotionEstimateMap = EstimateMap<ObjectId, Motion3>;

/**
 * @brief Generic mapping of Object Id -> FrameId -> Value within a nested
 * gtsam::FastMap structure. This is a common datastrcture used for object's to
 * store temporal information about each object.
 *
 * We call it ObjectCentric map as we order by ObjectId first.
 *
 * @tparam VALUE Value type to be stored
 */
template <typename VALUE>
class GenericObjectCentricMap
    : public gtsam::FastMap<ObjectId, gtsam::FastMap<FrameId, VALUE>> {
 public:
  using Base = gtsam::FastMap<ObjectId, gtsam::FastMap<FrameId, VALUE>>;
  using NestedBase = gtsam::FastMap<FrameId, VALUE>;

  using This = GenericObjectCentricMap<VALUE>;
  using Value = VALUE;

  using Base::at;
  using Base::Base;  // all the stl map stuff
  using Base::exists;
  using Base::insert2;

  /** Conversion to a gtsam::FastMap container */
  operator Base() const { return Base(this->begin(), this->end()); }

  operator typename Base::Base() const {
    return typename Base::Base(this->begin(), this->end());
  }

  /**
   * @brief Handy insert function allowing direct insertion to the nested map
   * structure.
   *
   * @param object_id
   * @param frame_id
   * @param value
   * @return true
   * @return false
   */
  bool insert22(ObjectId object_id, FrameId frame_id, const Value& value) {
    if (!this->exists(object_id)) {
      this->insert2(object_id, NestedBase{});
    }

    NestedBase& frame_map = this->at(object_id);
    return frame_map.insert2(frame_id, value);
  }

  bool exists(ObjectId object_id, FrameId frame_id) const {
    static size_t out_of_range_flag;
    return existsImpl(object_id, frame_id, out_of_range_flag);
  }

  const Value& at(ObjectId object_id, FrameId frame_id) const {
    return atImpl(this, object_id, frame_id);
  }

  Value& at(ObjectId object_id, FrameId frame_id) {
    return atImpl(const_cast<const This*>(this), object_id, frame_id);
  }

  /**
   * @brief Collect all objects that appear in the query frame, as well as their
   * value.
   *
   * @param frame_id FrameId
   * @return gtsam::FastMap<ObjectId, Value>
   */
  gtsam::FastMap<ObjectId, Value> collectByFrame(FrameId frame_id) const {
    gtsam::FastMap<ObjectId, Value> object_map;
    for (const auto& [object_id, frame_map] : *this) {
      if (frame_map.exists(frame_id)) {
        object_map.insert2(object_id, frame_map.at(frame_id));
      }
    }
    return object_map;
  }

 private:
  template <typename Container>
  static auto& atImpl(Container* container, ObjectId object_id,
                      FrameId frame_id) {
    size_t out_of_range_flag;
    const bool result = existsImpl(object_id, frame_id, out_of_range_flag);
    if (result) {
      CHECK_EQ(out_of_range_flag, 2u);
      return container->at(object_id)[frame_id];
    } else {
      std::stringstream ss;
      ss << "Index out of range: "
         << ((out_of_range_flag == 0) ? " object id " : " frame id")
         << " missing. Full query - (object id " << object_id << ", frame id "
         << frame_id << ").";
      throw std::out_of_range(ss.str());
    }
  }

  /**
   * @brief Helper function to determine if the query exists.
   * Operates like a regular exists function but also sets out_of_range_flag
   * to indicate which query (object_id or frame_id) is out of range:
   * out_of_range_flag = 0, object_id out of range
   * out_of_range_flag = 1, frame_id out of range
   * out_of_range_flag = 2 both queries exist and the function should return
   * true
   *
   * @param object_id
   * @param frame_id
   * @param out_of_range_flag
   * @return true
   * @return false
   */
  bool existsImpl(ObjectId object_id, FrameId frame_id,
                  size_t& out_of_range_flag) const {
    if (!this->exists(object_id)) {
      out_of_range_flag = 0;
      return false;
    }

    const auto& frame_map = this->at(object_id);
    if (!frame_map.exists(frame_id)) {
      out_of_range_flag = 1;
      return false;
    } else {
      out_of_range_flag = 2;
      return true;
    }
  }
};

/// @brief Map of object poses per object per frame
using ObjectPoseMap = GenericObjectCentricMap<gtsam::Pose3>;

// Optional string that can be modified directly (similar to old-stype
// boost::optional) to access the mutable reference the internal string must be
// accessed with get()
//  e.g. optional->get() = "updated string value";
// This is to overcome the fact that the stdlib does not support
// std::optional<T&> directly
using OptionalString = std::optional<std::reference_wrapper<std::string>>;

template <typename T>
std::string to_string(const T& t);

template <typename Input, typename Output>
bool convert(const Input&, Output&);

// TODO: adds delimiter at end of string too!
template <class Container>
inline std::string container_to_string(const Container& container,
                                       const std::string& delimiter = " ") {
  std::stringstream ss;
  for (const auto& c : container) {
    ss << c << delimiter;
  }
  return ss.str();
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

}  // namespace dyno

template <typename T, int D>
struct gtsam::traits<dyno::MeasurementWithCovariance<T, D>>
    : public gtsam::Testable<dyno::MeasurementWithCovariance<T, D>> {};
