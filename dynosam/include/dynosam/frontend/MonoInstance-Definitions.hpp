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

#include <glog/logging.h>

namespace dyno {


/**
 * @brief Should provide camera pose estimation (can be up to scale, TODO: how to indicate this?),
 * object motion estimation (may only be the rotation)
 * as well as the 2D observations (static and dynamic)
 *
 */
struct MonocularInstanceOutputPacket : public FrontendOutputPacketBase {

public:
    DYNO_POINTER_TYPEDEFS(MonocularInstanceOutputPacket)

    const MotionEstimateMap estimated_motions_; //! Estimated motions in the world frame

    MonocularInstanceOutputPacket(
        const StatusKeypointMeasurements& static_keypoint_measurements,
        const StatusKeypointMeasurements& dynamic_keypoint_measurements,
        const gtsam::Pose3 T_world_camera,
        const Frame& frame,
        const MotionEstimateMap& estimated_motions,
        const cv::Mat& debug_image = cv::Mat(),
        const GroundTruthInputPacket::Optional& gt_packet = std::nullopt
    )
    :
    FrontendOutputPacketBase(
        FrontendType::kMono,
        static_keypoint_measurements,
        dynamic_keypoint_measurements,
        T_world_camera,
        frame,
        debug_image,
        gt_packet),
    estimated_motions_(estimated_motions)
    {
    }


};

} //dymo
