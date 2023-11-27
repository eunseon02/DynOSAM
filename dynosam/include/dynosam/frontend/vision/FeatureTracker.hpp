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

#include <opencv4/opencv2/opencv.hpp>

namespace dyno {

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
    Frame::Ptr track(FrameId frame_id, Timestamp timestamp, const TrackingInputImages& tracking_images, size_t& n_optical_flow, size_t& n_new_tracks);

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


protected:

    void trackStatic(FrameId frame_id, const TrackingInputImages& tracking_images, FeatureContainer& static_features, size_t& n_optical_flow,
                   size_t& n_new_tracks);
    void trackDynamic(FrameId frame_id, const TrackingInputImages& tracking_images, FeatureContainer& dynamic_features);

    void propogateMask(TrackingInputImages& tracking_images);

private:
    void computeImageBounds(const cv::Size& size, int& min_x, int& max_x, int& min_y, int& max_y) const;
    // bool posInGrid(const Keypoint& kp, int& pos_x, int& pos_y) const;

    Feature::Ptr constructStaticFeature(const TrackingInputImages& tracking_images, const Keypoint& kp, size_t age, TrackletId tracklet_id,
                                      FrameId frame_id) const;

protected:
    const FrontendParams params_;
    Camera::Ptr camera_;
    ImageDisplayQueue* display_queue_;

private:
    Frame::Ptr previous_frame_{ nullptr }; //! The frame that will be used as the previous frame next time track is called. After track, this is actually the frame that track() returns
    Frame::Ptr previous_tracked_frame_{nullptr}; //! The frame that has just beed used to track on a new frame is created.
    ORBextractor::UniquePtr feature_detector_{nullptr};

    size_t tracklet_count = 0;
    bool initial_computation_{ true };

    cv::Size img_size_;  // set on first computation

};

}
