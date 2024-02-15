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
#include "dynosam/frontend/FrontendOutputPacket.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/logger/Logger.hpp"


#include <glog/logging.h>

namespace dyno {

//forward from Tracker
struct FeatureTrackerInfo;

struct RGBDInstanceOutputPacket : public FrontendOutputPacketBase {

public:
    DYNO_POINTER_TYPEDEFS(RGBDInstanceOutputPacket)

    const StatusLandmarkEstimates static_landmarks_; //! in the camera frame
    const StatusLandmarkEstimates dynamic_landmarks_; //! in the camera frame
    const MotionEstimateMap estimated_motions_; //! Estimated motions in the world frame
    const ObjectPoseMap propogated_object_poses_; //! Propogated poses using the esimtate from the frontend
    const gtsam::Pose3Vector camera_poses_; //! Vector of ego-motion poses (drawn everytime)

     RGBDInstanceOutputPacket(
        const StatusKeypointMeasurements& static_keypoint_measurements,
        const StatusKeypointMeasurements& dynamic_keypoint_measurements,
        const StatusLandmarkEstimates& static_landmarks,
        const StatusLandmarkEstimates& dynamic_landmarks,
        const gtsam::Pose3 T_world_camera,
        const Frame& frame,
        const MotionEstimateMap& estimated_motions,
        const ObjectPoseMap propogated_object_poses = {},
        const gtsam::Pose3Vector camera_poses = {},
        const cv::Mat& debug_image = cv::Mat(),
        const GroundTruthInputPacket::Optional& gt_packet = std::nullopt
    )
    :
    FrontendOutputPacketBase(
        FrontendType::kRGBD,
        static_keypoint_measurements,
        dynamic_keypoint_measurements,
        T_world_camera,
        frame,
        debug_image,
        gt_packet),
    static_landmarks_(static_landmarks),
    dynamic_landmarks_(dynamic_landmarks),
    estimated_motions_(estimated_motions),
    propogated_object_poses_(propogated_object_poses),
    camera_poses_(camera_poses)
    {
        //they need to be the same size as we expect a 1-to-1 relation between the keypoint and the landmark (which acts as an initalisation point)
        CHECK_EQ(static_landmarks_.size(), static_keypoint_measurements_.size());
        CHECK_EQ(dynamic_landmarks_.size(), dynamic_keypoint_measurements_.size());
    }
};


//write to file on destructor
//TODO: currently FrontendLogger really is RGBDLogger?
class RGBDFrontendLogger : public EstimationModuleLogger {
public:
    DYNO_POINTER_TYPEDEFS(RGBDFrontendLogger)
    RGBDFrontendLogger() : EstimationModuleLogger("frontend") {}
};

} //dymo
