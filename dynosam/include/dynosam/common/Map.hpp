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
#include "dynosam/backend/BackendDefinitions.hpp" //for all the chr's used in the keys
#include "dynosam/utils/GtsamUtils.hpp"

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <boost/concept/usage.hpp>
#include <boost/concept/assert.hpp>

#include <type_traits>

namespace dyno {

struct MapNodeBase {
    virtual ~MapNodeBase() = default;
    virtual int getId() const = 0;
};

template<typename NODE>
struct IsMapNode {
    using UnderlyingType = typename NODE::element_type; //how to check NODE is a shared_ptr?
    static_assert(std::is_base_of_v<MapNodeBase, UnderlyingType>);
};

// template<typename NODE>
// struct MapNodePtrComparison : public IsMapNode<NODE> {
//     bool operator() (const NODE& f1, const NODE& f2) const {
//         return f1->getId() < f2->getId();
//     }
// };

//no static checks as above becuase of declaration order of pointers in the map
template<typename NODE>
struct MapNodePtrComparison {
    bool operator() (const NODE& f1, const NODE& f2) const {
        return f1->getId() < f2->getId();
    }
};


//TODO:assume Node is a shared_ptr to a MapNodeType -> enforce later!!!
template<typename NODE>
class FastMapNodeSet : public std::set<NODE, MapNodePtrComparison<NODE>> {
    public:
        typedef std::set<NODE, MapNodePtrComparison<NODE>> Base;

        using Base::Base;  // Inherit the set constructors

        FastMapNodeSet() = default; ///< Default constructor

        /** Constructor from a iterable container, passes through to base class */
        template<typename INPUTCONTAINER>
        explicit FastMapNodeSet(const INPUTCONTAINER& container) :
            Base(container.begin(), container.end()) {
        }

        /** Copy constructor from another FastMapNodeSet */
        FastMapNodeSet(const FastMapNodeSet<NODE>& x) :
        Base(x) {
        }

        /** Copy constructor from the base set class */
        FastMapNodeSet(const Base& x) :
        Base(x) {
        }

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

struct FrameNode;
struct ObjectNode;
struct LandmarkNode;

using FrameNodePtr = std::shared_ptr<FrameNode>;
using ObjectNodePtr = std::shared_ptr<ObjectNode>;
using LandmarkNodePtr = std::shared_ptr<LandmarkNode>;

using FrameNodePtrSet = FastMapNodeSet<FrameNodePtr>;
using LandmarkNodePtrSet = FastMapNodeSet<LandmarkNodePtr>;
using ObjectNodePtrSet = FastMapNodeSet<ObjectNodePtr>;


//TODO: using std::set, prevents the same thign being added multiple times
//while this is what we want, it may hide a bug, as we never want this situation toa ctually occur
struct FrameNode : public MapNodeBase {
    DYNO_POINTER_TYPEDEFS(FrameNode)

    FrameId frame_id;
    //set prevents the same feature being added multiple times
    LandmarkNodePtrSet dynamic_landmarks;
    LandmarkNodePtrSet static_landmarks;
    ObjectNodePtrSet objects_seen;

    inline int getId() const override{
        return (int)frame_id;
    }

    bool objectObserved(ObjectId object_id) const {
        return objects_seen.find(object_id) != objects_seen.end();
    }

    //get pose estimate
    std::optional<gtsam::Pose3> getPoseEstimate() const;
    //get object motion estimate
    std::optional<gtsam::Pose3> getObjectMotionEstimate(ObjectId object_id);
    //get dynamic point estimate (at this frame)??
    std::optional<Landmark> getStaticLandmarkEstimate(TrackletId tracklet_id) const;
    std::optional<Landmark> getDynamicLandmarkEstimate(TrackletId tracklet_id) const;

};


struct LandmarkNode : public MapNodeBase {
    DYNO_POINTER_TYPEDEFS(LandmarkNode)

    TrackletId tracklet_id;
    ObjectId object_id; //will this change ever?
    FrameNodePtrSet frames_seen;

    inline int getId() const override {
        return (int)tracklet_id;
    }

    bool isStatic() const {
        return object_id == background_label;
    }

    size_t numObservations() const {
        return frames_seen.size();
    }

};

struct ObjectNode  : public MapNodeBase {
    DYNO_POINTER_TYPEDEFS(ObjectNode)

    ObjectId object_id;
    //all tracklets
    LandmarkNodePtrSet dynamic_landmarks;


    inline int getId() const override {
        return (int)object_id;
    }

    //this could take a while?
    //for all the lmks we have for this object, find the frames of those lmks
    FrameNodePtrSet getSeenFrames() const {
        FrameNodePtrSet seen_frames;
        for(const auto& lmks : dynamic_landmarks) {
            seen_frames.merge(lmks->frames_seen);
        }
        return seen_frames;
    }

    FrameIds getSeenFrameIds() const {
        return getSeenFrames().collectIds<FrameId>();
    }

    //The landmarks might have been seen at multiple frames but we know this is the subset of lmks at
    //this requested frame
    LandmarkNodePtrSet getLandmarksSeenAtFrame(FrameId frame_id) const {
        LandmarkNodePtrSet seen_lmks;

        for(const auto& lmk : dynamic_landmarks) {
            //all frames this lmk was seen in
            const FrameNodePtrSet& frames = lmk->frames_seen;
            //lmk was observed at this frame
            if(frames.find(frame_id) != frames.end()) {
                seen_lmks.insert(lmk);
            }
        }
        return seen_lmks;


    }


};


class Map {

public:
    DYNO_POINTER_TYPEDEFS(Map)

    void updateObservations(const StatusKeypointMeasurements& keypoint_measurements);

    bool frameExists(FrameId frame_id) const;
    bool landmarkExists(TrackletId tracklet_id) const;
    bool objectExists(ObjectId object_id) const;

    //should these not all be const?!!! dont want to modify the values
    ObjectNode::Ptr getObject(ObjectId object_id);
    FrameNode::Ptr getFrame(FrameId frame_id);
    LandmarkNode::Ptr getLandmark(TrackletId tracklet_id);

    TrackletIds getStaticTrackletsByFrame(FrameId frame_id) const;

    size_t numObjectsSeen() const;



private:
    void addOrUpdateMapStructures(TrackletId tracklet_id, FrameId frame_id, ObjectId object_id, bool is_static);

//TODO: for now
private:


    gtsam::FastMap<FrameId, FrameNode::Ptr> frames_;
    gtsam::FastMap<TrackletId, LandmarkNode::Ptr> landmarks_;
    gtsam::FastMap<ObjectId, ObjectNode::Ptr> objects_;


};


}
