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

#include "dynosam/common/Camera.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/ORBextractor.hpp"
#include "dynosam/frontend/vision/Feature.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"
#include "dynosam/frontend/vision/OccupancyGrid2D.hpp"


#include <opencv4/opencv2/opencv.hpp>

namespace dyno {

struct PerObjectStatus {
    ObjectId object_id;
    size_t num_previous_track{0}; //number of (inlier) features tracked in the previous frame that MAY be used
    size_t num_track{0}; //actual number of features tracked (ie. used) from the previous frame - does not include newly sampled points!
    size_t num_sampled{0}; //num new points sampled and added to the set of features
    size_t num_outside_shrunken_image{0}; //sampled or tracked
    size_t num_zero_flow{0}; //sampled or tracked
    size_t num_tracked_with_different_label{0}; //number of points tracked from previous frame where the current label is different
    size_t num_tracked_with_background_label{0}; //number of points tracked from previous frame wehre current label is the background
    //would be nice to have some histogram data about each tracked point etc...

    PerObjectStatus(ObjectId id) : object_id(id) {}
};

// template<>
// inline std::string to_string(const PerObjectStatus& object_info) {

// }


struct FeatureTrackerInfo {
    FrameId frame_id;

    //static track info
    size_t static_track_optical_flow;
    size_t static_track_detections;


    inline PerObjectStatus& getObjectStatus(ObjectId object_id) {
        if(!dynamic_track.exists(object_id)) {
            dynamic_track.insert2(object_id, PerObjectStatus(object_id));
        }

        return dynamic_track.at(object_id);
    }

    gtsam::FastMap<ObjectId, PerObjectStatus> dynamic_track;
};

template<>
inline std::string to_string(const FeatureTrackerInfo& info) {
    std::stringstream ss;
    ss << "FeatureTrackerInfo: \n"
       << " - frame id: " << info.frame_id << "\n"
       << "\t- # optical flow: " << info.static_track_optical_flow << "\n"
       << "\t- # detections: " << info.static_track_detections << "\n";

    for(const auto& [object_id, object_status] : info.dynamic_track) {
        ss << "\t- Object: " << object_id << ": \n";
        ss << "\t\t - num_track " << object_status.num_track << "\n";
        ss << "\t\t - num_sampled " << object_status.num_sampled << "\n";

        if(VLOG_IS_ON(20)) {
            ss << "\t\t - num_previous_track " << object_status.num_previous_track << "\n";
            ss << "\t\t - num_outside_shrunken_image " << object_status.num_outside_shrunken_image << "\n";
            ss << "\t\t - num_zero_flow " << object_status.num_zero_flow << "\n";
            ss << "\t\t - num_tracked_with_different_label " << object_status.num_tracked_with_different_label << "\n";
            ss << "\t\t - num_tracked_with_background_label " << object_status.num_tracked_with_background_label << "\n";
        }
    };
    return ss.str();

}


class FeatureTracker
{
public:
    DYNO_POINTER_TYPEDEFS(FeatureTracker)

    //and camera?
    //does no processing with any depth
    //if depth is a problem should be handled aftererds and separately
    FeatureTracker(const FrontendParams& params, Camera::Ptr camera, ImageDisplayQueue* display_queue = nullptr);
    virtual ~FeatureTracker() {}

    //note: MOTION MASK!!
    Frame::Ptr track(FrameId frame_id, Timestamp timestamp, const TrackingInputImages& tracking_images);

    cv::Mat computeImageTracks(const Frame& previous_frame, const Frame& current_frame) const;

    /**
     * @brief Get the previous frame.
     *
     * Will be null on the first call of track
     *
     * @return Frame::Ptr
     */
    inline Frame::Ptr getPreviousFrame() { return previous_tracked_frame_; }
    inline const Frame::ConstPtr getPreviousFrame() const { return previous_tracked_frame_; }

    /**
     * @brief Get the most recent frame that has been tracked.
     * After a call to track, this will be the frame that is returned and will track features between getPreviousFrame() to this frame
     *
     * @return Frame::Ptr
     */
    inline Frame::Ptr getCurrentFrame() { return previous_frame_; }

    inline const FeatureTrackerInfo& getTrackerInfo() { return info_; }


protected:

    void trackStatic(FrameId frame_id, const TrackingInputImages& tracking_images, FeatureContainer& static_features, size_t& n_optical_flow,
                   size_t& n_new_tracks);
    void trackDynamic(FrameId frame_id, const TrackingInputImages& tracking_images, FeatureContainer& dynamic_features);

    void propogateMask(TrackingInputImages& tracking_images);

    inline bool isWithinShrunkenImage(const Keypoint& kp) {
        CHECK(!initial_computation_);
        const auto shrunken_row = params_.shrink_row;
        const auto shrunken_col = params_.shrink_col;

        const int predicted_col = functional_keypoint::u(kp);
        const int predicted_row = functional_keypoint::v(kp);

        const auto image_rows = img_size_.height;
        const auto image_cols = img_size_.width;
        return (predicted_row > shrunken_row && predicted_row < (image_rows - shrunken_row) &&
          predicted_col > shrunken_col && predicted_col < (image_cols - shrunken_col));

    }

private:
    void computeImageBounds(const cv::Size& size, int& min_x, int& max_x, int& min_y, int& max_y) const;

    Feature::Ptr constructStaticFeature(const TrackingInputImages& tracking_images, const Keypoint& kp, size_t age, TrackletId tracklet_id,
                                      FrameId frame_id) const;

protected:
    const FrontendParams params_;
    const cv::Size img_size_;
    Camera::Ptr camera_;
    ImageDisplayQueue* display_queue_;

private:
    Frame::Ptr previous_frame_{ nullptr }; //! The frame that will be used as the previous frame next time track is called. After track, this is actually the frame that track() returns
    Frame::Ptr previous_tracked_frame_{nullptr}; //! The frame that has just beed used to track on a new frame is created.
    ORBextractor::UniquePtr feature_detector_{nullptr};

    FeatureTrackerInfo info_;

    OccupandyGrid2D static_grid_; //! Grid used to feature bin static features

    size_t tracklet_count = 0;
    bool initial_computation_{ true };

};

}
