/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/common/Types.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/frontend/vision/Frame.hpp"


#include <gtsam/base/FastSet.h>
#include <type_traits>
#include <set>
#include <memory>

namespace dyno {

//forward declare map
class Map;

struct InvalidMapException : public DynosamException {
    InvalidMapException() : DynosamException("The Map could not be accessed as it is no longer valid. Has the object gone out of scope?") {}
};

class MapNodeBase {
public:
    MapNodeBase(const std::shared_ptr<Map>& map) : map_ptr_(map) {}
    virtual ~MapNodeBase() = default;
    virtual int getId() const = 0;

    template<typename ValueType>
    std::optional<ValueType> queryMap(gtsam::Key key) const;


protected:
    std::weak_ptr<Map> map_ptr_;
};

//TODO: unused due to circular dependancies
template<typename NODE>
struct IsMapNode {
    using UnderlyingType = typename NODE::element_type; //how to check NODE is a shared_ptr?
    static_assert(std::is_base_of_v<MapNodeBase, UnderlyingType>);
};


template<typename NODE>
struct MapNodePtrComparison {
    bool operator() (const NODE& f1, const NODE& f2) const {
        return f1->getId() < f2->getId();
    }
};

//TODO:assume Node is a shared_ptr to a MapNodeType -> enforce later!!!
template<typename NODE>
class FastMapNodeSet : public std::set<NODE, MapNodePtrComparison<NODE>,
        typename gtsam::internal::FastDefaultAllocator<NODE>::type> {
    public:
        typedef std::set<NODE, MapNodePtrComparison<NODE>,
            typename gtsam::internal::FastDefaultAllocator<NODE>::type> Base;


        using Base::Base;  // Inherit the set constructors

        FastMapNodeSet() = default; ///< Default constructor

        /** Constructor from a iterable container, passes through to base class */
        template<typename INPUTCONTAINER>
        explicit FastMapNodeSet(const INPUTCONTAINER& container)
        : Base(container.begin(), container.end()) {}

        /** Copy constructor from another FastMapNodeSet */
        FastMapNodeSet(const FastMapNodeSet<NODE>& x)
        : Base(x) {}

        /** Copy constructor from the base set class */
        FastMapNodeSet(const Base& x)
        : Base(x) {}

        template<typename Index = int>
        std::vector<Index> collectIds() const {
            std::vector<Index> ids;
            for (const auto& node : *this) {
                //ducktyping for node to have a getId function
                ids.push_back(static_cast<Index>(node->getId()));
            }

            return ids;
        }

        template<typename Index = int>
        Index getFirstIndex() const {
            const NODE& node = *Base::cbegin();
            return getIndexSafe<Index>(node);
        }

        template<typename Index = int>
        Index getLastIndex() const {
            const NODE& node = *Base::crbegin();
            return getIndexSafe<Index>(node);
        }

        template<typename Index = int>
        typename Base::iterator find(Index index) {
            return std::find_if(
                Base::begin(),
                Base::cend(),
                [index](const NODE& node) { return getIndexSafe<Index>(node) == index;});
        }

        template<typename Index = int>
        typename Base::const_iterator find(Index index) const {
            return std::find_if(
                Base::cbegin(),
                Base::cend(),
                [index](const NODE& node) { return getIndexSafe<Index>(node) == index;});
        }

        void merge(const FastMapNodeSet<NODE>& other) {
            Base::insert(other.begin(), other.end());
        }

    private:
        template<typename Index>
        static inline Index getIndexSafe(const NODE& node) {
            return static_cast<Index>(node->getId());
        }

};


//forward delcare structs
struct FrameNode;
struct ObjectNode;
struct LandmarkNode;

using FrameNodePtr = std::shared_ptr<FrameNode>;
using ObjectNodePtr = std::shared_ptr<ObjectNode>;
using LandmarkNodePtr = std::shared_ptr<LandmarkNode>;

//TODO: using std::set, prevents the same object being added multiple times
//while this is what we want, it may hide a bug, as we never want this situation to actually occur
using FrameNodePtrSet = FastMapNodeSet<FrameNodePtr>;
using LandmarkNodePtrSet = FastMapNodeSet<LandmarkNodePtr>;
using ObjectNodePtrSet = FastMapNodeSet<ObjectNodePtr>;

template<typename ValueType>
class StateQuery : public std::optional<ValueType> {
public:
    using Base = std::optional<ValueType>;
    enum Status { VALID, NOT_IN_MAP, WAS_IN_MAP, INVALID_MAP };
    gtsam::Key key_;
    Status status_;


    StateQuery(gtsam::Key key, const ValueType& v) : key_(key), status_(VALID) { Base::emplace(v); }
    StateQuery(gtsam::Key key, Status status) : key_(key), status_(status) {}

    const ValueType& get() const {
        if (!Base::has_value()) throw DynosamException("StateQuery has no value for query type " + type_name<ValueType>() + " with key" + DynoLikeKeyFormatter(key_));
        return Base::value();
    }

    bool isValid() const { return status_ == VALID; }

    static StateQuery InvalidMap() { return StateQuery(gtsam::Key{}, INVALID_MAP); }
    static StateQuery NotInMap(gtsam::Key key) { return StateQuery(key, NOT_IN_MAP); }
    static StateQuery WasInMap(gtsam::Key key) { return StateQuery(key, WAS_IN_MAP); }

};


class FrameNode : public MapNodeBase {

public:
    DYNO_POINTER_TYPEDEFS(FrameNode)

    FrameNode(const std::shared_ptr<Map>& map) : MapNodeBase(map) {}

    FrameId frame_id;
    //TODO: check consistency between dynamic lmks existing in the values, and a motion existing
    //all dynamic lmks landmarks that have observations at this frame
    LandmarkNodePtrSet dynamic_landmarks;
    //all static landmarks that have observations at this frame
    LandmarkNodePtrSet static_landmarks;
    //this means that we have a point measurement on this object at this frame,
    //not necessarily that we have a motion at this frame
    ObjectNodePtrSet objects_seen;

    int getId() const override;
    bool objectObserved(ObjectId object_id) const;

    const Frame& getTrackedFrame() const;

    //get pose estimate
    StateQuery<gtsam::Pose3> getPoseEstimate() const;
    //get object motion estimate
    StateQuery<gtsam::Pose3> getObjectMotionEstimate(ObjectId object_id) const;
    //get dynamic point estimate (at this frame)??
    StateQuery<Landmark> getDynamicLandmarkEstimate(TrackletId tracklet_id) const;

    //TODO: need testing
    //all in this frame
    StatusLandmarkEstimates getAllDynamicLandmarkEstimates() const;
    StatusLandmarkEstimates getDynamicLandmarkEstimates(ObjectId object_id) const;


private:
};

class ObjectNode  : public MapNodeBase {

public:
    DYNO_POINTER_TYPEDEFS(ObjectNode)

    ObjectNode(const std::shared_ptr<Map>& map) : MapNodeBase(map) {}

    ObjectId object_id;
    //all tracklets
    LandmarkNodePtrSet dynamic_landmarks;

    //returns object_id
    int getId() const override;

    //this could take a while?
    //for all the lmks we have for this object, find the frames of those lmks
    FrameNodePtrSet getSeenFrames() const;
    FrameIds getSeenFrameIds() const;
    //The landmarks might have been seen at multiple frames but we know this is the subset of lmks at
    //this requested frame
    LandmarkNodePtrSet getLandmarksSeenAtFrame(FrameId frame_id) const;

};

struct InvalidLandmarkQuery : public DynosamException {
    InvalidLandmarkQuery(gtsam::Key key, const std::string& string)
        : DynosamException("Landmark estimate query failed with key " + DynoLikeKeyFormatter(key) + ", reason: " + string) {}
};

//could either be static or dynamic. Is there a better way to handle this?
//Use inheritance?
class LandmarkNode : public MapNodeBase {
public:
    DYNO_POINTER_TYPEDEFS(LandmarkNode)

    LandmarkNode(const std::shared_ptr<Map>& map) : MapNodeBase(map) {}

    TrackletId tracklet_id;
    ObjectId object_id; //will this change ever?
    FrameNodePtrSet frames_seen;

    //returns tracklet id
    int getId() const override;
    ObjectId getObjectId() const;
    bool isStatic() const;

    size_t numObservations() const;

    StateQuery<Landmark> getStaticLandmarkEstimate() const;
};


} //dyno
