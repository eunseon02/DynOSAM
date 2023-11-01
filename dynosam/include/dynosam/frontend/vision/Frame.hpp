/*
 *   Copyright (c) 2023 Jesse Morris (jesse.morris@sydney.edu.au)
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
#include "dynosam/common/ImageContainer.hpp"
#include "dynosam/common/StructuredContainers.hpp"
#include "dynosam/common/Camera.hpp"
#include "dynosam/common/DynamicObjects.hpp"
#include "dynosam/frontend/vision/Feature.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"

#include <functional>



namespace dyno {

//should this be here?
using TrackingInputImages = ImageContainerSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>;

class Frame {

public:
    DYNO_POINTER_TYPEDEFS(Frame)
    DYNO_DELETE_COPY_CONSTRUCTORS(Frame)

    // struct all_feature_container {
    //     using iterator = FeatureContainer::iterator;
    //     using value_type = FeatureContainer::value_type;
    //     using reference = FeatureContainer::reference;

    //     FeatureContainer& static_features_; //! ref
    //     FeatureContainer& dynamic_features_; //! ref
    //     iterator it_;

    //     all_feature_container(const FeatureContainer& static_features, const FeatureContainer& dynamic_features)
    //         :   static_features_(static_features), dynamic_features_(dynamic_features), it_(static_features.begin()) {}

    //     reference operator*() { return *it_; }
    //     reference operator->() { return *it_; }

    //     bool operator==(const all_feature_container& other) const {
    //         return it_ == other.it_;
    //     }
    //     bool operator!=(const all_feature_container& other) const { return it_ != other.it_; }

    //     bool operator==(const iterator& other) const {
    //         return it_ == other;
    //     }
    //     bool operator!=(const iterator& other) const { return it_ != other; }


    //     all_feature_container& operator++() {
    //         ++it_;
    //         if(it_ == static_features_.end()) {
    //             it_ = dynamic_features_.begin();
    //         }
    //         return *this;
    //     }

    //     iterator end() { return dynamic_features_.end(); }

    // };


    const FrameId frame_id_;
    const Timestamp timestamp_;
    Camera::Ptr camera_;
    const TrackingInputImages tracking_images_;

    gtsam::Pose3 T_world_camera_ = gtsam::Pose3::Identity();

    FeatureContainer static_features_;
    FeatureContainer dynamic_features_;

    static ObjectId global_object_id;

    //semantic instance label to object observation (by the actual observations in the image)
    std::map<ObjectId, DynamicObjectObservation> object_observations_;

    Frame(
        FrameId frame_id,
        Timestamp timestamp,
        Camera::Ptr camera,
        const TrackingInputImages& tracking_images,
        const FeatureContainer& static_features,
        const FeatureContainer& dynamic_features);

    //inliers and outliers
    inline size_t numStaticFeatures() const { return static_features_.size(); }

    //TODO: test
    inline size_t numStaticUsableFeatures() {
        auto iter = usableStaticFeaturesBegin();
        return static_cast<size_t>(std::distance(iter.begin(), iter.end()));
    }

    inline size_t numDynamicFeatures() const { return dynamic_features_.size(); }
    //TODO: test
    inline size_t numDynamicUsableFeatures() {
        auto iter = usableDynamicFeaturesBegin();
        return static_cast<size_t>(std::distance(iter.begin(), iter.end()));
    }

    //inliers and outliers
    inline size_t numTotalFeatures() const { return numStaticFeatures() +  numDynamicFeatures(); }


    bool exists(TrackletId tracklet_id) const;
    Feature::Ptr at(TrackletId tracklet_id) const;

    FeaturePtrs collectFeatures(TrackletIds tracklet_ids) const;

    Landmark backProjectToCamera(TrackletId tracklet_id) const;
    Landmark backProjectToWorld(TrackletId tracklet_id) const;

    void updateDepths(const ImageWrapper<ImageType::Depth>& depth, double max_static_depth, double max_dynamic_depth);


    //TODO: this really needs testing
    void moveObjectToStatic(ObjectId instance_label);
    //TODO: testing
    //also updates all the tracking_labels of the features associated with this object
    void updateObjectTrackingLabel(const DynamicObjectObservation& observation, ObjectId new_tracking_label);



    void getCorrespondences(AbsolutePoseCorrespondences& correspondences, const Frame& previous_frame, KeyPointType kp_type) const;
    void getCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, KeyPointType kp_type) const;

    //special iterator types
    FeatureFilterIterator usableStaticFeaturesBegin();
    FeatureFilterIterator usableDynamicFeaturesBegin();

protected:
    void getStaticCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame) const;
    void getDynamicCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame) const;

private:
    static void updateDepthsFeatureContainer(FeatureContainer& container, const ImageWrapper<ImageType::Depth>& depth, double max_depth);

    //based on the current set of dynamic features
    void constructDynamicObservations();



};






} //dyno
