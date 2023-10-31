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

#include "dynosam/common/Types.hpp"
// #include "dynosam/common/S"

#include <map>
#include <vector>

namespace dyno {

enum KeyPointType {
    STATIC,
    DYNAMIC
};

//! Expected label for the background in a semantic or motion mask
constexpr static ObjectId background_label = 0u;

struct functional_keypoint {

    template<typename T = int>
    static inline T u(const Keypoint& kp) {
        return static_cast<T>(kp(0));
    }

    template<typename T = int>
    static inline int v(const Keypoint& kp) {
        return static_cast<T>(kp(1));
    }
};


// /**
//  * @brief A 3D point on the surface of a scene
//  *
//  */
// class Point {
// public:
//     DYNO_POINTER_TYPEDEFS(Point)

//     TrackletId tracklet_id_; //should match observing features
//     std::vector<FrameId> observing_frames_; //! which frames saw this keypoint (should also loosly mean age)
//     Landmark landmark_; //! 3d position of the point in the world frame
// }




/**
 * @brief 2D tracking and id information for a feature observation at a single frame
 *
 */
class Feature {

public:
    DYNO_POINTER_TYPEDEFS(Feature)

    constexpr static TrackletId invalid_id = -1; //!can refer to and invalid id label (of type int)
                                          //! including tracklet id, instance and tracking label

    constexpr static auto invalid_depth = NaN; //! nan is used to indicate the absense of a depth value (since we have double)

    Keypoint keypoint_; //! u,v keypoint at this frame (frame_id)
    Keypoint predicted_keypoint_; //from optical flow
    size_t age_;
    KeyPointType type_; //! starts STATIC
    TrackletId tracklet_id_; //starts invalid
    FrameId frame_id_;
    bool inlier_; //! Starts as inlier
    ObjectId instance_label_; //! instance label as provided by the input mask
    ObjectId tracking_label_; //! object tracking label that should indicate the same tracked object between frames

    Depth depth_; //! Depth as provided by a depth image (not Z). Initalised as invalud_depth (NaN)

    Feature() :
        keypoint_(),
        predicted_keypoint_(),
        age_(0u),
        type_(KeyPointType::STATIC),
        tracklet_id_(invalid_id),
        frame_id_(0u),
        inlier_(true),
        instance_label_(invalid_id),
        tracking_label_(invalid_id),
        depth_(invalid_depth) {}


    /**
     * @brief If the feature is valid - a combination of inlier and if the tracklet Id != -1
     *
     * To make a feature invalid, set tracklet_id == -1
     *
     * @return true
     * @return false
     */
    inline bool usable() const {
        return inlier_ && tracklet_id_ != invalid_id;
    }

    inline bool isStatic() const {
        return type_ == KeyPointType::STATIC;
    }

    inline void markInvalid() {
        tracklet_id_ = invalid_id;
    }

    inline bool hasDepth() const {
        return !std::isnan(depth_);
    }
};

using FeaturePtrs = std::vector<Feature::Ptr>;
using FeaturePair = std::pair<Feature::Ptr, Feature::Ptr>; //! Pair of feature (shared) pointers
using FeaturePairs = std::vector<FeaturePair>; //! Vector of feature pairs

//contains a set of features (usually per frame) that can be quickly accessed by tracklet id or iteterated over
class FeatureContainer {
public:
    using TrackletToFeatureMap = std::unordered_map<TrackletId, Feature::Ptr>;
    //! need to define iterator, value_type and reference to satisfy iterator traits
    using iterator = FeaturePtrs::iterator;
    using value_type = Feature::Ptr;
    using reference = Feature::Ptr&;
    using const_iterator = FeaturePtrs::const_iterator;

    FeatureContainer();
    FeatureContainer(const FeaturePtrs feature_vector);

    void add(Feature::Ptr feature);
    TrackletIds collectTracklets(bool only_usable = true) const;

    void markOutliers(const TrackletIds outliers);

    size_t size() const;

    Feature::Ptr getByTrackletId(TrackletId tracklet_id) const;
    Feature::Ptr at(size_t i) const;

    bool exists(TrackletId tracklet_id) const;

    //vector begin
    inline iterator begin() { return feature_vector_.begin(); }
    inline const_iterator begin() const { return feature_vector_.cbegin(); }

    //vector end
    inline iterator end() { return feature_vector_.end(); }
    inline const_iterator end() const { return feature_vector_.cend(); }


private:
    TrackletToFeatureMap feature_map_;
    FeaturePtrs feature_vector_;
};



}
