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

#include "dynosam/frontend/vision/FeatureTrackerBase.hpp"
#include "dynosam/common/Camera.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/ORBextractor.hpp"
#include "dynosam/frontend/vision/OccupancyGrid2D.hpp"
#include "dynosam/common/Types.hpp"

#include "dynosam/frontend/anms/NonMaximumSuppression.h"


#include <opencv4/opencv2/opencv.hpp>

namespace cv {
  class CLAHE;
  class SparsePyrLKOpticalFlow;
  class FastFeatureDetector;
}

namespace dyno {


class StaticFeatureTracker : public FeatureTrackerBase {
public:
    DYNO_POINTER_TYPEDEFS(StaticFeatureTracker)
    StaticFeatureTracker(const FrontendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue);

    virtual ~StaticFeatureTracker() {}
    virtual FeatureContainer trackStatic(Frame::Ptr previous_frame, const ImageContainer& image_container, FeatureTrackerInfo& tracker_info) = 0;

};

//currently assumes flow that is k to k + 1 (gross!!)
class ExternalFlowFeatureTracker : public StaticFeatureTracker {

public:
    ExternalFlowFeatureTracker(const FrontendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue);
    FeatureContainer trackStatic(Frame::Ptr previous_frame, const ImageContainer& image_container, FeatureTrackerInfo& tracker_info) override;


private:
    Feature::Ptr constructStaticFeature(const ImageContainer& image_container, const Keypoint& kp, size_t age, TrackletId tracklet_id, FrameId frame_id) const;


private:
    OccupandyGrid2D static_grid_; //! Grid used to feature bin static features
    ORBextractor::UniquePtr orb_detector_{nullptr};

};



class KltFeatureTracker : public StaticFeatureTracker {

public:
    KltFeatureTracker(const FrontendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue);
    FeatureContainer trackStatic(Frame::Ptr previous_frame, const ImageContainer& image_container, FeatureTrackerInfo& tracker_info) override;

    /**
     * @brief Outputs a CLAHE equalized greyscale image from the input RGB, which will be used to detect and track features
     *
     * @param image_container
     * @param equialized_greyscale
     */
    void equalizeImage(const ImageContainer& image_container, cv::Mat& equialized_greyscale) const;

    /**
     * @brief Detects features on the input image using the feature detector.
     *
     * The features are then spaced out using adaptive non-maxima-supression and refined using subpixel refinement (via cornerSubPix).
     *
     * The number of tracked points indicates how many currently tracked points we currently have; this is used to calculate how many more features
     * we need to reach the minimum number features per frame.
     *
     * @param processed_img
     * @param number_tracked
     * @param mask
     * @return std::vector<cv::Point2f>
     */
    std::vector<cv::Point2f> detectRawFeatures(const cv::Mat& processed_img, int number_tracked, const cv::Mat& mask = cv::Mat());

    //image container associated with the processed image
    bool detectFeatures(
        const cv::Mat& processed_img,
        const ImageContainer& image_container,
        const FeatureContainer& current_features,
        FeatureContainer& new_features);


    /**
     * @brief Tracks features using KLT tracker between the previous image and the current one.
     *
     * Once tracked, geometric outlier rejection is used to sample the inliers and, if not enough features are tracked,
     * new features are detected to be tracked in the previous frame.
     *
     * The input previous_features, is a container of INLIER features tracked from the previous frame and will be used to set the KLT tracker.
     * The new tracks (and detections) are dumped into tracked_features and the features that were poorly tracked from the previous frame are identified
     * in outlier_previous_features; this vector corresponds to features in previous_features.
     *
     * @param current_processed_img
     * @param previous_processed_img
     * @param image_container
     * @param previous_features
     * @param tracked_features
     * @param outlier_previous_features
     * @param tracker_info
     * @return true
     * @return false
     */
    bool trackPoints(
        const cv::Mat& current_processed_img,
        const cv::Mat& previous_processed_img,
        const ImageContainer& image_container,
        const FeatureContainer& previous_features,
        FeatureContainer& tracked_features,
        TrackletIds& outlier_previous_features,
        FeatureTrackerInfo& tracker_info);

    //use homography + ransac to mark inliers with tracking
    cv::Mat geometricVerification(const std::vector<cv::Point2f>& good_old, const std::vector<cv::Point2f>& good_new) const;

    /**
     * @brief A more concise static feature constructor that the one in ExternalFlowFeatureTracker.
     *
     * Expects the current kp to already be checked (ie. lies within the image, lies on a static pixel etc).
     * This function does the handling of the tracklet ids (ie. incrementing the id in the TrackletIdManager).
     *
     * If the feature tracked age is greater than max_feature_track_age_, we relabel it with a new tracklet id to prevent
     * massive observation growth in the backend.
     *
     * Function will also update the measured flow in the previous feature (silly old API...)
     *
     * As a result should never return nullptr
     *
     * @param kp_current
     * @param previous_feature
     * @param tracklet_id
     * @param frame_id
     * @return Feature::Ptr
     */
    Feature::Ptr constructStaticFeatureFromPrevious(const Keypoint& kp_current, Feature::Ptr previous_feature, const TrackletId tracklet_id, const FrameId frame_id) const;

    /**
     * @brief Construct a new static feature with a unique tracklet id
     *
     * @param kp_current
     * @param frame_id
     * @return Feature::Ptr
     */
    Feature::Ptr constructNewStaticFeature(const Keypoint& kp_current, const FrameId frame_id) const;

private:
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Ptr<cv::Feature2D> feature_detector_;

    NonMaximumSuppression::UniquePtr non_maximum_supression_;

    //Parameters to go in params eventually

    // Number features to detect
    int max_nr_keypoints_before_anms_ = 2000;
    double quality_level_ = 0.001;
    //also used for the mask
    int min_distance_btw_tracked_and_detected_features_ = 8;
    int block_size_ = 3;
    bool use_harris_corner_detector_ = false;
    double k_ = 0.04;

    int max_features_per_frame_; // Threshold for the number of features to keep tracking
    //! We relabel the track as a new track for any features longer that this
    size_t max_feature_track_age_ = 25;


};


} // dyno
