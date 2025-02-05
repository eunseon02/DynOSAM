/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <gtsam/base/FastSet.h>
#include <pcl/common/centroid.h>  //for compute centroid

#include <memory>
#include <set>
#include <type_traits>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/common/PointCloudProcess.hpp"
#include "dynosam/common/Types.hpp"

namespace dyno {

// forward declare map
template <typename MEASUREMENT>
class Map;

struct InvalidMapException : public DynosamException {
  InvalidMapException()
      : DynosamException(
            "The Map could not be accessed as it is no longer valid. Has the "
            "object gone out of scope?") {}
};

template <typename MEASUREMENT>
class MapNodeBase {
 public:
  MapNodeBase(const std::shared_ptr<Map<MEASUREMENT>>& map) : map_ptr_(map) {}
  virtual ~MapNodeBase() = default;
  virtual int getId() const = 0;
  virtual std::string toString() const = 0;

  // TODO: not used
  template <typename ValueType>
  std::optional<ValueType> queryMap(gtsam::Key key) const;

 protected:
  std::shared_ptr<Map<MEASUREMENT>> map_ptr_;
};

// TODO: unused due to circular dependancies
template <typename NODE>
struct IsMapNode {
  using UnderlyingType =
      typename NODE::element_type;  // how to check NODE is a shared_ptr?
  static_assert(std::is_base_of_v<MapNodeBase, UnderlyingType>);
};

template <typename NODE>
struct MapNodePtrComparison {
  bool operator()(const NODE& f1, const NODE& f2) const {
    return f1->getId() < f2->getId();
  }
};

// TODO:assume Node is a shared_ptr to a MapNodeType -> enforce later!!!
template <typename NODE>
class FastMapNodeSet
    : public std::set<
          NODE, MapNodePtrComparison<NODE>,
          typename gtsam::internal::FastDefaultAllocator<NODE>::type> {
 public:
  typedef std::set<NODE, MapNodePtrComparison<NODE>,
                   typename gtsam::internal::FastDefaultAllocator<NODE>::type>
      Base;

  using Base::Base;  // Inherit the set constructors

  FastMapNodeSet() = default;  ///< Default constructor

  /** Constructor from a iterable container, passes through to base class */
  template <typename INPUTCONTAINER>
  explicit FastMapNodeSet(const INPUTCONTAINER& container)
      : Base(container.begin(), container.end()) {}

  /** Copy constructor from another FastMapNodeSet */
  FastMapNodeSet(const FastMapNodeSet<NODE>& x) : Base(x) {}

  /** Copy constructor from the base set class */
  FastMapNodeSet(const Base& x) : Base(x) {}

  template <typename Index = int>
  std::vector<Index> collectIds() const {
    std::vector<Index> ids;
    for (const auto& node : *this) {
      // ducktyping for node to have a getId function
      ids.push_back(static_cast<Index>(node->getId()));
    }

    return ids;
  }

  template <typename Index = int>
  Index getFirstIndex() const {
    const NODE& node = *Base::cbegin();
    return getIndexSafe<Index>(node);
  }

  template <typename Index = int>
  Index getLastIndex() const {
    const NODE& node = *Base::crbegin();
    return getIndexSafe<Index>(node);
  }

  template <typename Index = int>
  typename Base::iterator find(Index index) {
    return std::find_if(Base::begin(), Base::cend(), [index](const NODE& node) {
      return getIndexSafe<Index>(node) == index;
    });
  }

  template <typename Index = int>
  typename Base::const_iterator find(Index index) const {
    return std::find_if(Base::cbegin(), Base::cend(),
                        [index](const NODE& node) {
                          return getIndexSafe<Index>(node) == index;
                        });
  }

  template <typename Index = int>
  bool exists(Index index) const {
    return this->find(index) != this->end();
  }

  void merge(const FastMapNodeSet<NODE>& other) {
    Base::insert(other.begin(), other.end());
  }

  bool continuousAscendingIndex() const {
    auto ids = collectIds();
    if (ids.size() < 2) return true;

    bool is_continuous = true;
    for (size_t i = 1; i < ids.size(); i++) {
      const auto prev = ids.at(i - 1);
      const auto curr = ids.at(i);
      is_continuous &= (prev + 1 == curr);
    }
    return is_continuous;
  }

 private:
  template <typename Index>
  static inline Index getIndexSafe(const NODE& node) {
    return static_cast<Index>(node->getId());
  }
};

// forward delcare structs

template <typename MEASUREMENT>
struct FrameNode;
template <typename MEASUREMENT>
struct ObjectNode;
template <typename MEASUREMENT>
struct LandmarkNode;

template <typename MEASUREMENT>
using FrameNodePtr = std::shared_ptr<FrameNode<MEASUREMENT>>;
template <typename MEASUREMENT>
using ObjectNodePtr = std::shared_ptr<ObjectNode<MEASUREMENT>>;
template <typename MEASUREMENT>
using LandmarkNodePtr = std::shared_ptr<LandmarkNode<MEASUREMENT>>;

// TODO: using std::set, prevents the same object being added multiple times
// while this is what we want, it may hide a bug, as we never want this
// situation to actually occur
template <typename MEASUREMENT>
using FrameNodePtrSet = FastMapNodeSet<FrameNodePtr<MEASUREMENT>>;
template <typename MEASUREMENT>
using LandmarkNodePtrSet = FastMapNodeSet<LandmarkNodePtr<MEASUREMENT>>;
template <typename MEASUREMENT>
using ObjectNodePtrSet = FastMapNodeSet<ObjectNodePtr<MEASUREMENT>>;

template <typename ValueType>
class StateQuery : public std::optional<ValueType> {
 public:
  using Base = std::optional<ValueType>;
  enum Status { VALID, NOT_IN_MAP, WAS_IN_MAP, INVALID_MAP };
  gtsam::Key key_;
  Status status_;

  // TODO: not needed!
  StateQuery() {}
  StateQuery(gtsam::Key key, const ValueType& v) : key_(key), status_(VALID) {
    Base::emplace(v);
  }
  StateQuery(gtsam::Key key, Status status) : key_(key), status_(status) {}

  const ValueType& get() const {
    if (!Base::has_value())
      throw DynosamException("StateQuery has no value for query type " +
                             type_name<ValueType>() + " with key" +
                             DynoLikeKeyFormatter(key_));
    return Base::value();
  }

  bool isValid() const { return status_ == VALID; }

  static StateQuery InvalidMap() {
    return StateQuery(gtsam::Key{}, INVALID_MAP);
  }
  static StateQuery NotInMap(gtsam::Key key) {
    return StateQuery(key, NOT_IN_MAP);
  }
  static StateQuery WasInMap(gtsam::Key key) {
    return StateQuery(key, WAS_IN_MAP);
  }
};

/**
 * @brief Safe getter to a StateQuery object.
 * If the StateQuery is successful, result is set to the value of the query and
 * true is returned. Else, result is set to to the default value and false is
 * returned.
 *
 * @tparam ValueType
 * @param result
 * @param query
 * @param default_value
 * @return true
 * @return false
 */
template <typename ValueType>
bool getSafeQuery(ValueType& result, const StateQuery<ValueType>& query,
                  const ValueType& default_value) {
  if (query) {
    result = query.get();
    return true;
  } else {
    result = default_value;
    return false;
  }
}

struct InvalidLandmarkException : public DynosamException {
  InvalidLandmarkException(TrackletId tracklet_id,
                           const std::string& reason = std::string())
      : DynosamException("Landmark with tracklet id" +
                         std::to_string(tracklet_id) + " is invalid" +
                         (reason.empty() ? "." : " with reason " + reason)) {}
};

struct MissingLandmarkException : InvalidLandmarkException {
  MissingLandmarkException(TrackletId tracklet_id, FrameId frame_id,
                           bool is_static)
      : InvalidLandmarkException(
            tracklet_id, (is_static ? "static" : "dynamic") +
                             std::string(" landmark is missing from frame ") +
                             std::to_string(frame_id)) {}
};

template <typename MEASUREMENT>
class FrameNode : public MapNodeBase<MEASUREMENT> {
 public:
  using Base = MapNodeBase<MEASUREMENT>;
  using This = FrameNode<MEASUREMENT>;
  DYNO_POINTER_TYPEDEFS(This)

  FrameNode(const std::shared_ptr<Map<MEASUREMENT>>& map)
      : MapNodeBase<MEASUREMENT>(map) {}

  FrameId frame_id;
  // TODO: check consistency between dynamic lmks existing in the values, and a
  // motion existing all dynamic lmks landmarks that have observations at this
  // frame
  LandmarkNodePtrSet<MEASUREMENT> dynamic_landmarks;
  // all static landmarks that have observations at this frame
  LandmarkNodePtrSet<MEASUREMENT> static_landmarks;
  // this means that we have a point measurement on this object at this frame,
  // not necessarily that we have a motion at this frame
  ObjectNodePtrSet<MEASUREMENT> objects_seen;

  /// @brief Optional initial camera pose in world, provided by the front-end
  std::optional<gtsam::Pose3> X_world;
  /// @brief Optional initial object motions in the world, provided by the
  /// front-end
  std::optional<MotionEstimateMap> motions;

  /**
   * @brief Returns the frame_id
   *
   * @return int
   */
  int getId() const override;
  std::string toString() const override {
    std::stringstream ss;
    ss << "Frame Id: " << frame_id << "\n";
    ss << "Objects seen: "
       << container_to_string(objects_seen.template collectIds<ObjectId>())
       << "\n";
    ss << "Num dynamic points: " << numDynamicPoints() << "\n";
    ss << "Num static points: " << numStaticPoints();
    return ss.str();
  }

  bool objectObserved(ObjectId object_id) const;
  // TODO: test
  bool objectObservedInPrevious(ObjectId object_id) const;

  // object appears in both this and previous frame so we expect a motion from
  // k-1 to k
  // TODO: test
  bool objectMotionExpected(ObjectId object_id) const;

  gtsam::Key makePoseKey() const;

  gtsam::Key makeObjectMotionKey(ObjectId object_id) const;
  gtsam::Key makeObjectPoseKey(ObjectId object_id) const;

  /// @brief Const LandmarkNodePtr with corresponding Measurement value
  using LandmarkMeasurementPair =
      std::pair<const LandmarkNodePtr<MEASUREMENT>, MEASUREMENT>;

  // vector of measurements taken at this frame
  std::vector<LandmarkMeasurementPair> getStaticMeasurements() const;
  std::vector<LandmarkMeasurementPair> getDynamicMeasurements() const;
  std::vector<LandmarkMeasurementPair> getDynamicMeasurements(
      ObjectId object_id) const;

  // static and dynamic
  StatusLandmarkEstimates getAllLandmarkEstimates() const;

  inline size_t numObjects() const { return objects_seen.size(); }
  inline size_t numDynamicPoints() const { return dynamic_landmarks.size(); }
  inline size_t numStaticPoints() const { return static_landmarks.size(); }

 private:
  /// @brief
  using GenericGetLandmarkEstimateFunc =
      std::function<StateQuery<Landmark>(LandmarkNodePtr<MEASUREMENT>)>;
};

template <typename MEASUREMENT>
class ObjectNode : public MapNodeBase<MEASUREMENT> {
 public:
  using Base = MapNodeBase<MEASUREMENT>;
  using This = ObjectNode<MEASUREMENT>;
  DYNO_POINTER_TYPEDEFS(This)

  ObjectNode(const std::shared_ptr<Map<MEASUREMENT>>& map)
      : MapNodeBase<MEASUREMENT>(map) {}

  ObjectId object_id;
  // all tracklets
  LandmarkNodePtrSet<MEASUREMENT> dynamic_landmarks;

  /**
   * @brief Returns the object_id
   *
   * @return int
   */
  int getId() const override;
  std::string toString() const override {
    std::stringstream ss;
    ss << "Object Id: " << getId() << "\n";
    ss << "First seen: " << getFirstSeenFrame() << " last seen "
       << getLastSeenFrame() << "\n";
    ss << "Seen continuously: " << std::boolalpha
       << getSeenFrames().continuousAscendingIndex() << "\n";
    ss << "Total landmarks: " << dynamic_landmarks.size();
    return ss.str();
  }

  inline FrameId getFirstSeenFrame() const {
    return getSeenFrames().template getFirstIndex<FrameId>();
  }
  inline FrameId getLastSeenFrame() const {
    return getSeenFrames().template getLastIndex<FrameId>();
  }

  // this could take a while?
  // for all the lmks we have for this object, find the frames of those lmks
  FrameNodePtrSet<MEASUREMENT> getSeenFrames() const;
  FrameIds getSeenFrameIds() const;
  // The landmarks might have been seen at multiple frames but we know this is
  // the subset of lmks at this requested frame
  LandmarkNodePtrSet<MEASUREMENT> getLandmarksSeenAtFrame(
      FrameId frame_id) const;

  /// @brief A pair of Const LandmarkNodePtr's
  using LandmarkNodePair = std::pair<const LandmarkNodePtr<MEASUREMENT>,
                                     const LandmarkNodePtr<MEASUREMENT>>;
};

struct InvalidLandmarkQuery : public DynosamException {
  InvalidLandmarkQuery(gtsam::Key key, const std::string& string)
      : DynosamException("Landmark estimate query failed with key " +
                         DynoLikeKeyFormatter(key) + ", reason: " + string) {}
};

template <typename MEASUREMENT>
class LandmarkNode : public MapNodeBase<MEASUREMENT> {
 public:
  using Base = MapNodeBase<MEASUREMENT>;
  using This = LandmarkNode<MEASUREMENT>;

  // Map of measurements, via the frame this measurement was seen in
  using Measurements = gtsam::FastMap<FrameNodePtr<MEASUREMENT>, MEASUREMENT>;
  DYNO_POINTER_TYPEDEFS(This)

  LandmarkNode(const std::shared_ptr<Map<MEASUREMENT>>& map)
      : MapNodeBase<MEASUREMENT>(map) {}
  virtual ~LandmarkNode() = default;

  TrackletId tracklet_id;
  ObjectId object_id;  // will this change ever?

  /**
   * @brief Returns the tracklet_id
   *
   * @return int
   */
  int getId() const override;

  std::string toString() const override {
    std::stringstream ss;
    if (isStatic()) {
      ss << "Static point: ";
    } else {
      ss << "Dynamic point (" << object_id << "): ";
    }
    // tracklet ID
    ss << getId() << "\n";
    ss << "Num obs: " << numObservations();
    return ss.str();
  }

  ObjectId getObjectId() const;

  // simply checks the background label
  // dangernous if the label changes as now it will attempt to retrieve a static
  // point
  bool isStatic() const;
  size_t numObservations() const;

  inline const FrameNodePtrSet<MEASUREMENT>& getSeenFrames() const {
    return frames_seen_;
  }

  FrameIds getSeenFrameIds() const;

  inline const Measurements& getMeasurements() const { return measurements_; }

  bool seenAtFrame(FrameId frame_id) const;
  // should exist if also at frame
  bool hasMeasurement(FrameId frame_id) const;
  const MEASUREMENT& getMeasurement(FrameNodePtr<MEASUREMENT> frame_node) const;
  const MEASUREMENT& getMeasurement(FrameId frame_id) const;

  // TODO: test
  void add(FrameNodePtr<MEASUREMENT> frame_node,
           const MEASUREMENT& measurement);

  // THROWS exception when wrong type... is there a cleaner way to handle this?
  // TODO: there are some tests
  //  StateQuery<Landmark> getStaticLandmarkEstimate() const;
  //  StateQuery<Landmark> getDynamicLandmarkEstimate(FrameId frame_id) const;

  gtsam::Key makeStaticKey() const;
  gtsam::Key makeDynamicKey(FrameId frame_id) const;
  DynamicPointSymbol makeDynamicSymbol(FrameId frame_id) const;

  // bool appendStaticLandmarkEstimate(StatusLandmarkEstimates& estimates)
  // const; bool appendDynamicLandmarkEstimate(StatusLandmarkEstimates&
  // estimates, FrameId frame_id) const;

 private:
 protected:
  FrameNodePtrSet<MEASUREMENT> frames_seen_;
  Measurements measurements_;
};

}  // namespace dyno

#include "dynosam/common/MapNodes-inl.hpp"
