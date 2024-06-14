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

#include "dynosam/frontend/vision/ObjectTracker.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"

namespace dyno {

cv::Mat ObjectTracker::track(const cv::Mat& masks, FrameId frame_id) {

    const ObjectIds instance_labels = vision_tools::getObjectLabels(masks);
    std::vector<byte_track::DetectionBase::Ptr> input_detections;
    for(auto instance_label : instance_labels) {
        cv::Rect bounding_box;
        if(vision_tools::findObjectBoundingBox(masks, instance_label, bounding_box)) {
            input_detections.push_back(std::make_shared<byte_track::Detection>(bounding_box, 1.0));
        }
        else {
            VLOG(30) << "Could not find bb for instance label " << instance_label << " at frame " << frame_id << " and therefore could not track!";
        }
    }

    auto object_tracks = impl_tracker_.update(input_detections, frame_id);
    CHECK_EQ(object_tracks.size(), input_detections.size());

    ObjectIds old_labels = instance_labels;
    ObjectIds new_labels; //the new tracked labels from the EKF
    for(const auto& track : object_tracks) {
        new_labels.push_back(track->get_track_id());
    }

    //update tracking labels
    cv::Mat updated_masks;
    vision_tools::relabelMasks(masks, updated_masks, old_labels, new_labels);
    return updated_masks;

}

}
