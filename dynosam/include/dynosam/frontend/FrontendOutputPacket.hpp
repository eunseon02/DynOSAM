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
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"

namespace dyno {

struct FrontendOutputPacketBase {
public:
    DYNO_POINTER_TYPEDEFS(FrontendOutputPacketBase)


public:
    const FrontendType frontend_type_;
    const StatusKeypointMeasurements static_keypoint_measurements_;
    const StatusKeypointMeasurements dynamic_keypoint_measurements_;
    const gtsam::Pose3 T_world_camera_;
    const Frame frame_;
    const cv::Mat debug_image_;

    FrontendOutputPacketBase(
        const FrontendType frontend_type,
        const StatusKeypointMeasurements& static_keypoint_measurements,
        const StatusKeypointMeasurements& dynamic_keypoint_measurements,
        const gtsam::Pose3& T_world_camera,
        const Frame& frame,
        const cv::Mat& debug_image = cv::Mat()
    )
    :   frontend_type_(frontend_type),
        static_keypoint_measurements_(static_keypoint_measurements),
        dynamic_keypoint_measurements_(dynamic_keypoint_measurements),
        T_world_camera_(T_world_camera),
        frame_(frame),
        debug_image_(debug_image)
    {}

    virtual ~FrontendOutputPacketBase() {}

    // FrontendInputPacketBase::ConstPtr input_; //for reference and possible display
    // Frame::Ptr frame_;
    // LandmarkMap tracked_landmarks;

    // //for now
    // std::map<ObjectId, gtsam::Pose3> object_poses_;
    //should this be here?
    // std::vector<ImageToDisplay> debug_images;
};


} //dyno
