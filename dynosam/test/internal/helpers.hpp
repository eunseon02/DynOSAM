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
#include "dynosam/common/Camera.hpp"

#include <gtest/gtest.h>
#include <ament_index_cpp/get_package_prefix.hpp>


/**
 * @brief gets the full path to the installation directory of the test data which is expected to be at dynosam/test/data
 *
 * The full path will be the ROS install directory of this data after building
 *
 * @return std::string
 */
inline std::string getTestDataPath() {
  return ament_index_cpp::get_package_prefix("dynosam") + "/test/data";
}

namespace dyno_testing {

using namespace dyno;

inline StatusKeypointMeasurement makeStatusKeypointMeasurement(TrackletId tracklet_id, ObjectId object_id, FrameId frame_id, const Keypoint& keypoint = Keypoint()) {
  KeyPointType kp_type;
  if(object_id == background_label) {
    kp_type = KeyPointType::STATIC;
  }
  else {
    kp_type = KeyPointType::DYNAMIC;
  }
  return KeypointStatus(keypoint, frame_id, tracklet_id, object_id, kp_type);
}

inline void compareLandmarks(const Landmarks& lmks_1,
                        const Landmarks& lmks_2,
                        const float& tol = 1e-9) {
    ASSERT_EQ(lmks_1.size(), lmks_2.size());
    for (size_t i = 0u; i < lmks_1.size(); i++) {
      const auto& lmk_1 = lmks_1[i];
      const auto& lmk_2 = lmks_2[i];
      EXPECT_TRUE(gtsam::assert_equal(lmk_1, lmk_2, tol));
    }
  }

inline void compareKeypoints(const Keypoints& lmks_1,
                        const Keypoints& lmks_2,
                        const float& tol =  1e-9) {
    ASSERT_EQ(lmks_1.size(), lmks_2.size());
    for (size_t i = 0u; i < lmks_1.size(); i++) {
      const auto& lmk_1 = lmks_1[i];
      const auto& lmk_2 = lmks_2[i];
      EXPECT_TRUE(gtsam::assert_equal(lmk_1, lmk_2, tol));
    }
  }


inline CameraParams makeDefaultCameraParams() {
  CameraParams::IntrinsicsCoeffs intrinsics(4);
  CameraParams::DistortionCoeffs distortion(4);
  intrinsics.at(0) = 721.5377;  // fx
  intrinsics.at(1) = 721.5377;  // fy
  intrinsics.at(2) = 609.5593;  // u0
  intrinsics.at(3) = 172.8540;  // v0
  return CameraParams(intrinsics, distortion, cv::Size(640, 480), "radtan");
}

inline Camera makeDefaultCamera() {
  return Camera(makeDefaultCameraParams());
}

} //dyno_testing
