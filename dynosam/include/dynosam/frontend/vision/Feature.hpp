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
#include "dynosam/common/StructuredContainers.hpp"
#include "dynosam/utils/Numerical.hpp"

#include <map>
#include <vector>
#include <type_traits>

namespace dyno {

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
    OpticalFlow measured_flow_; //! Observed optical flow that. The predicted keypoint is calculated as keypoint + flow
    Keypoint predicted_keypoint_; //from optical flow
    size_t age_;
    KeyPointType type_; //! starts STATIC
    TrackletId tracklet_id_; //starts invalid
    FrameId frame_id_;
    bool inlier_; //! Starts as inlier
    ObjectId instance_label_; //! instance label as provided by the input mask
    ObjectId tracking_label_; //! object tracking label that should indicate the same tracked object between frames

    Depth depth_; //! Depth as provided by a depth image (not Z). Initalised as invalid_depth (NaN)

    Feature() :
        keypoint_(),
        measured_flow_(),
        predicted_keypoint_(),
        age_(0u),
        type_(KeyPointType::STATIC),
        tracklet_id_(invalid_id),
        frame_id_(0u),
        inlier_(true),
        instance_label_(invalid_id),
        tracking_label_(invalid_id),
        depth_(invalid_depth) {}

    bool operator==(const Feature& other) const {
        return gtsam::equal_with_abs_tol(keypoint_, other.keypoint_) &&
               gtsam::equal_with_abs_tol(measured_flow_, other.measured_flow_) &&
               gtsam::equal_with_abs_tol(predicted_keypoint_, other.predicted_keypoint_) &&
               age_ == other.age_ &&
               type_ == other.type_ &&
               tracklet_id_ == other.tracklet_id_ &&
               frame_id_ == other.frame_id_ &&
               inlier_ == other.inlier_ &&
               instance_label_ == other.instance_label_ &&
               tracking_label_ == other.tracking_label_ &&
               fpEqual(depth_, other.depth_);
    }

    static Keypoint CalculatePredictedKeypoint(const Keypoint& keypoint, const OpticalFlow& measured_flow) {
        return keypoint + measured_flow;
    }

    /**
     * @brief Sets the measured optical flow (which should start at the features keypoint)
     * and updates the predicted keypoint using the flow: predicted_keypoint_ = keypoint_ + measured_flow_;
     *
     * Uses the internal Feature::keypoint_ value
     *
     * @param measured_flow
     */
    void setPredictedKeypoint(const OpticalFlow& measured_flow) {
        measured_flow_ = measured_flow;
        predicted_keypoint_ = CalculatePredictedKeypoint(keypoint_, measured_flow_);
    }



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

    inline static bool IsUsable(const Feature::Ptr& f) {
        return f->usable();
    }

    inline static bool IsNotNull(const Feature::Ptr& f) {
        return f != nullptr;
    }

};




using FeaturePtrs = std::vector<Feature::Ptr>;
using FeaturePair = std::pair<Feature::Ptr, Feature::Ptr>; //! Pair of feature (shared) pointers
using FeaturePairs = std::vector<FeaturePair>; //! Vector of feature pairs

//some typedefs and trait types
namespace internal {

/// @brief Alias for checking if the parsed iterator has a value_type equivalent to Feature::Ptr,
/// in other words, that the template iterates over Feature Ptr's.
/// @tparam Iter
template<typename Iter>
using is_feature_ptr_iterator = std::is_same<Feature::Ptr, typename Iter::value_type>;

template<typename Iter>
using enable_if_feature_ptr_iterator = typename std::enable_if<is_feature_ptr_iterator<Iter>::value, void>::type;

} //internal


//contains a set of features (usually per frame) that can be quickly accessed by tracklet id or iteterated over
class FeatureContainer {
public:
    using TrackletToFeatureMap = std::map<TrackletId, Feature::Ptr>;

    //this should satisfy the constraints for a filter_iterator_base
    template<typename MapIterator, typename MappedType>
    struct vector_iterator_base {
        using iterator_type = MapIterator;

        using value_type = MappedType;
        using reference = value_type&;
        using pointer = value_type*;

        iterator_type it_;
        vector_iterator_base(iterator_type it) : it_(it) {}

        reference operator*() { return it_->second; }
        reference operator->() { return it_->second; }

        bool operator==(const vector_iterator_base& other) const {
            return it_ == other.it_;
        }
        bool operator!=(const vector_iterator_base& other) const { return it_ != other.it_; }

        bool operator==(const iterator_type& other) const {
            return it_ == other;
        }
        bool operator!=(const iterator_type& other) const { return it_ != other; }

        vector_iterator_base& operator++() {
            ++it_;
            return *this;
        }

    };

    using vector_iterator = vector_iterator_base<TrackletToFeatureMap::iterator, Feature::Ptr>;
    using const_vector_iterator = vector_iterator_base<TrackletToFeatureMap::const_iterator, const Feature::Ptr>;

    //! need to define iterator, value_type and reference to satisfy iterator traits
    using iterator = vector_iterator;
    using pointer = iterator::pointer;

    using const_iterator = const_vector_iterator;
    using const_pointer = const_iterator::pointer;

    using value_type = Feature::Ptr;
    using reference = Feature::Ptr&;
    using const_reference = const Feature::Ptr&;
    using difference_type = std::ptrdiff_t;

    //MUST be defined after the iterator typedefs (using for FeatureContainer), e.g interator, pointer etc... for the filter_iterator definition to work
    using FilterIterator = internal::filter_iterator<FeatureContainer>;
    using ConstFilterIterator = internal::filter_const_iterator<FeatureContainer>;


    FeatureContainer();
    FeatureContainer(const FeaturePtrs feature_vector);

    //take copy of the feature
    void add(const Feature& feature);
    void add(Feature::Ptr feature);
    void remove(TrackletId tracklet_id);
    void clear();

    TrackletIds collectTracklets(bool only_usable = true) const;

    // //iterator (can be a filter iterator) where the iterator must iterate over Feature::Ptr's
    // template<typename Iter, typename = std::enable_if_t<internal::is_feature_ptr_iterator<Iter>>
    // TrackletIds collectTracklets(Iter iter) const;



    void markOutliers(const TrackletIds outliers);

    size_t size() const;

    Feature::Ptr getByTrackletId(TrackletId tracklet_id) const;

    // //this can totally give us nullptr's if we have removed this index
    // Feature::Ptr at(size_t i) const;

    bool exists(TrackletId tracklet_id) const;

//    //vector begin
//     inline iterator begin() { return feature_vector_.begin(); }
//     inline const_iterator begin() const { return feature_vector_.cbegin(); }

//     //vector end
//     inline iterator end() { return feature_vector_.end(); }
//     inline const_iterator end() const { return feature_vector_.cend(); }

    //vector beginc
    inline vector_iterator begin() { return vector_iterator(feature_map_.begin()); }
    inline const_vector_iterator begin() const { return const_vector_iterator(feature_map_.cbegin()); }

    //vector end
    inline vector_iterator end() { return vector_iterator(feature_map_.end()); }
    inline const_vector_iterator end() const { return const_vector_iterator(feature_map_.cend()); }

    FilterIterator beginUsable();
    FilterIterator beginUsable() const;
    // ConstFilterIterator beginUsable() const;




private:
    TrackletToFeatureMap feature_map_;
};

// template<typename Iter, typename = std::enable_if_t<internal::is_feature_ptr_iterator<Iter>>
// TrackletIds FeatureContainer::collectTracklets(Iter iter) const {
//     TrackletIds tracklets;
//     for(Feature::Ptr feature : iter) {
//         tracklets.push_back(feature->tracklet_id_);
//     }

//     return tracklets;
// }

/// @brief filter iterator over a FeatureContainer class
using FeatureFilterIterator = FeatureContainer::FilterIterator;
using ConstFeatureFilterIterator = FeatureContainer::ConstFilterIterator;


} //dyno

//add iterator traits so we can use smart thigns on the FeatureFilterIterator like std::count, std::distance...
template<>
struct std::iterator_traits<dyno::FeatureFilterIterator> : public dyno::internal::filter_iterator_detail<dyno::FeatureFilterIterator::pointer> {};

template<>
struct std::iterator_traits<dyno::ConstFeatureFilterIterator> : public dyno::internal::filter_iterator_detail<dyno::FeatureFilterIterator::pointer> {};

template<>
struct std::iterator_traits<dyno::FeatureContainer::vector_iterator> : public dyno::internal::filter_iterator_detail<dyno::FeatureContainer::vector_iterator::pointer> {};

template<>
struct std::iterator_traits<dyno::FeatureContainer::const_vector_iterator> : public dyno::internal::filter_iterator_detail<dyno::FeatureContainer::const_vector_iterator::pointer> {};

template<>
struct std::iterator_traits<dyno::FeatureContainer> : public dyno::internal::filter_iterator_detail<dyno::FeatureContainer::pointer> {};

template<>
struct std::iterator_traits<const dyno::FeatureContainer> : public dyno::internal::filter_iterator_detail<dyno::FeatureContainer::const_pointer> {};

template<>
struct std::iterator_traits<dyno::FeaturePtrs> : public dyno::internal::filter_iterator_detail<dyno::FeaturePtrs::pointer> {};

template<>
struct std::iterator_traits<const dyno::FeaturePtrs> : public dyno::internal::filter_iterator_detail<dyno::FeaturePtrs::const_pointer> {};
