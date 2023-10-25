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

//v inefficient to store in vectors
struct FindFeatureByTrackletId
{
  TrackletId tracklet_id_;
  FindFeatureByTrackletId(TrackletId tracklet_id) : tracklet_id_(tracklet_id)
  {
  }
  FindFeatureByTrackletId(const Feature::Ptr& feature) : FindFeatureByTrackletId(feature->tracklet_id_)
  {
  }
  bool operator()(const Feature& f) const
  {
    return f.tracklet_id_ == tracklet_id_;
  }

  bool operator()(const Feature::Ptr& f) const
  {
    return f->tracklet_id_ == tracklet_id_;
  }
};

}
