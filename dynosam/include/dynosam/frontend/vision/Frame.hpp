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

#include "dynosam/frontend/vision/UndistortRectifier.hpp"

#include <functional>



namespace dyno {


class Frame {

public:
    DYNO_POINTER_TYPEDEFS(Frame)

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
    bool isFeatureUsable(TrackletId tracklet_id) const;

    //can be inliers or outliers
    FeaturePtrs collectFeatures(TrackletIds tracklet_ids) const;

    Landmark backProjectToCamera(TrackletId tracklet_id) const;
    Landmark backProjectToWorld(TrackletId tracklet_id) const;

    void updateDepths(const ImageWrapper<ImageType::Depth>& depth, double max_static_depth, double max_dynamic_depth);


    //TODO: this really needs testing
    void moveObjectToStatic(ObjectId instance_label);
    //TODO: testing
    //also updates all the tracking_labels of the features associated with this object
    void updateObjectTrackingLabel(const DynamicObjectObservation& observation, ObjectId new_tracking_label);

    template<typename RefType, typename CurType>
    using ConstructCorrespondanceFunc = std::function<
        /* In the form previous frame, previous feature, current feature*/
        TrackletCorrespondance<RefType, CurType>(const Frame&, const Feature::Ptr&, const Feature::Ptr&)>;

    ConstructCorrespondanceFunc<Landmark, Keypoint> landmarkWorldKeypointCorrespondance() const;
    ConstructCorrespondanceFunc<Landmark, gtsam::Vector3> landmarkWorldProjectedBearingCorrespondance() const;

    template<typename RefType, typename CurType>
    bool getCorrespondences(
        GenericCorrespondences<RefType, CurType>& correspondences,
        const Frame& previous_frame,
        KeyPointType kp_type,
        const ConstructCorrespondanceFunc<RefType, CurType>& func) const;

    // void getCorrespondences(AbsolutePoseCorrespondences& correspondences, const Frame& previous_frame, KeyPointType kp_type) const;
    bool getCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, KeyPointType kp_type) const;


    template<typename RefType, typename CurType>
    bool getDynamicCorrespondences(
        GenericCorrespondences<RefType, CurType>& correspondences,
        const Frame& previous_frame,
        ObjectId object_id,
        const ConstructCorrespondanceFunc<RefType, CurType>& func) const;

    //via the object id in the object_observations_ map
    //searches for instance via object instance
    //all in world
    // bool getDynamicCorrespondences(AbsolutePoseCorrespondences& correspondences, const Frame& previous_frame, ObjectId object_id) const;
    bool getDynamicCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame, ObjectId object_id) const;

     //points in world frame
    // void convertCorrespondencesWorld(AbsolutePoseCorrespondences& absolute_pose_correspondences, const FeaturePairs& correspondences, const Frame& previous_frame) const;


    //special iterator types
    FeatureFilterIterator usableStaticFeaturesBegin();
    FeatureFilterIterator usableStaticFeaturesBegin() const;

    FeatureFilterIterator usableDynamicFeaturesBegin();
    FeatureFilterIterator usableDynamicFeaturesBegin() const;


protected:
    //these do not do distortion or projection along the ray
    bool getStaticCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame) const;
    bool getDynamicCorrespondences(FeaturePairs& correspondences, const Frame& previous_frame) const;


private:
    static void updateDepthsFeatureContainer(FeatureContainer& container, const ImageWrapper<ImageType::Depth>& depth, double max_depth);

    //based on the current set of dynamic features
    void constructDynamicObservations();

private:
    UndistorterRectifier::Ptr undistorter_;

};

template<typename RefType, typename CurType>
bool Frame::getCorrespondences(
    GenericCorrespondences<RefType, CurType>& correspondences,
    const Frame& previous_frame,
    KeyPointType kp_type,
    const ConstructCorrespondanceFunc<RefType, CurType>& func) const
{
    correspondences.clear();
    FeaturePairs feature_correspondences;
    const bool result = getCorrespondences(feature_correspondences, previous_frame, kp_type);
     if(!result) {
        return false;
    }

    for(const auto& pair : feature_correspondences) {
        const Feature::Ptr& prev_feature = pair.first;
        const Feature::Ptr& curr_feature = pair.second;

        CHECK(prev_feature);
        CHECK(curr_feature);

        CHECK_EQ(prev_feature->tracklet_id_, curr_feature->tracklet_id_);
        CHECK_EQ(prev_feature->frame_id_, previous_frame.frame_id_);
        CHECK_EQ(curr_feature->frame_id_, frame_id_);

        correspondences.push_back(func(previous_frame, prev_feature, curr_feature));
    }
    return true;
}

template<typename RefType, typename CurType>
bool Frame::getDynamicCorrespondences(
    GenericCorrespondences<RefType, CurType>& correspondences,
    const Frame& previous_frame,
    ObjectId object_id,
    const ConstructCorrespondanceFunc<RefType, CurType>& func) const
{
    FeaturePairs feature_correspondences;
    const bool result = getDynamicCorrespondences(feature_correspondences, previous_frame, object_id);
    if(!result) {
        return false;
    }

    //unncessary but just for sanity check
    for(const auto& feature_pairs : feature_correspondences) {
        const Feature::Ptr& prev_feature = feature_pairs.first;
        const Feature::Ptr& curr_feature = feature_pairs.second;

        CHECK_EQ(feature_pairs.first->instance_label_, object_id);
        CHECK(feature_pairs.first->usable());

        CHECK_EQ(feature_pairs.second->instance_label_, object_id);
        CHECK(feature_pairs.second->usable());

        CHECK_EQ(feature_pairs.second->tracklet_id_, feature_pairs.first->tracklet_id_);

        correspondences.push_back(func(previous_frame, prev_feature, curr_feature));
    }
    return true;
}






} //dyno
