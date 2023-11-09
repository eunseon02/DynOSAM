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


#include "dynosam/pipeline/PipelinePayload.hpp"

#include <opencv4/opencv2/opencv.hpp> //for cv::Rect
#include <gtsam/geometry/Pose3.h> //for Pose3

#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/visualizer/ColourMap.hpp"

namespace dyno {


struct ObjectPoseGT {
    DYNO_POINTER_TYPEDEFS(ObjectPoseGT)

    FrameId frame_id_;
    ObjectId object_id_;
    gtsam::Pose3 L_camera_; //!object pose in camera frame
    cv::Rect bounding_box_; //!box of detection on image plane

    /// @brief 3D object 'dimensions' in meters. Not all datasets will contain. Used to represent a 3D bounding box.
    /// Expected order is {width, height, length}
    std::optional<gtsam::Vector3> object_dimensions_;

    /**
     * @brief Draws the object label and bounding box on the provided image
     *
     * @param img
     */
    inline void drawBoundingBox(cv::Mat& img) const {
        const cv::Scalar colour = ColourMap::getObjectColour(object_id_, true);
        const std::string label = "Object: " + std::to_string(object_id_);
        utils::drawLabel(img, label, colour, bounding_box_);
    }
};

struct GroundTruthInputPacket : public PipelinePayload {
    DYNO_POINTER_TYPEDEFS(GroundTruthInputPacket)

    //must have a default constructor for dataset loading and IO
    GroundTruthInputPacket() {}

    GroundTruthInputPacket(Timestamp timestamp, FrameId id, const gtsam::Pose3 X, const std::vector<ObjectPoseGT>& poses)
        : PipelinePayload(timestamp), frame_id_(id), X_world_(X), object_poses_(poses) {}

    FrameId frame_id_;
    gtsam::Pose3 X_world_; //camera pose in world frame
    std::vector<ObjectPoseGT> object_poses_;
};



} //dyno
