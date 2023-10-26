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

#include "dynosam/common/Camera.hpp"

#include <glog/logging.h>

namespace dyno {

Camera::Camera(const CameraParams& camera_params) : camera_params_(camera_params)
{
    camera_impl_ = std::make_unique<CameraImpl>(
        gtsam::Pose3::Identity(), //this should be the body pose cam from the calibration but currently we want everything in camera frame,
        camera_params_.constructGtsamCalibration<gtsam::Cal3DS2>()
    );

    CHECK(camera_impl_);
}

void Camera::project(const Landmark& lmk, Keypoint* kpt) const {
  CHECK_NOTNULL(kpt);
  // I believe this project call in gtsam is quite inefficient as it goes
  // through a cascade of calls... Only useful if you want gradients I guess...
  *kpt = camera_impl_->project2(lmk);
}

void Camera::project(const Landmarks& lmks, Keypoints* kpts) const {
  CHECK_NOTNULL(kpts)->clear();
  const auto& n_lmks = lmks.size();
  kpts->resize(n_lmks);
  // Can be greatly optimized with matrix mult or vectorization
  for (size_t i = 0u; i < n_lmks; i++) {
    project(lmks[i], &(*kpts)[i]);
  }
}

bool Camera::isKeypointContained(const Keypoint& kpt, Depth depth) const {
    return isKeypointContained(kpt) && depth > 0.0;
}

bool Camera::isKeypointContained(const Keypoint& kpt) const {
    return kpt(0) >= 0.0 && kpt(0) < camera_params_.ImageWidth() &&
           kpt(1) >= 0.0 && kpt(1) < camera_params_.ImageHeight();
}

void Camera::backProject(const Keypoints& kps, const Depths& depths, Landmarks* lmks) const {
    CHECK_NOTNULL(lmks)->clear();
  lmks->reserve(kps.size());
  CHECK_EQ(kps.size(), depths.size());
  CHECK(camera_impl_);
  for (size_t i = 0u; i < kps.size(); i++) {
    Landmark lmk;
    backProject(kps[i], depths[i], &lmk);
    lmks->push_back(lmk);
  }
}
void Camera::backProject(const Keypoint& kp, const Depth& depth, Landmark* lmk) const {
    CHECK(lmk);
    *lmk = camera_impl_->backproject(kp, depth);
}

//TODO: no tests
void Camera::backProject(const Keypoint& kp, const Depth& depth, Landmark* lmk, const gtsam::Pose3& X_world) const {
  CHECK(lmk);
  Landmark lmk_c;
  backProject(kp, depth, &lmk_c);
  *lmk = X_world * lmk_c;
}

bool Camera::isLandmarkContained(const Landmark& lmk, Keypoint* keypoint) const {
    Keypoint kp;
    project(lmk, &kp);

    const bool result = isKeypointContained(kp);
    if(keypoint) *keypoint = kp;
    return result;
}







} //dyno
