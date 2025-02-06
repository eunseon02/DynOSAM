/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include "dynosam/common/Camera.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"

namespace dyno {

/**
 * @brief Singleton class to manage a global tracklet id for all trackers to
 * ensure they are unique
 *
 */
class TrackletIdManager {
 public:
  DYNO_POINTER_TYPEDEFS(TrackletIdManager)

  static TrackletIdManager& instance() {
    if (!instance_) {
      instance_.reset(new TrackletIdManager());
    }
    return *instance_;
  }

  inline TrackletId getTrackletIdCount() const { return tracklet_count_; }
  inline void incrementTrackletIdCount() { tracklet_count_++; }

 private:
  TrackletIdManager() = default;
  TrackletId tracklet_count_{0};  //! Global TrackletId

  static std::unique_ptr<TrackletIdManager> instance_;
};

class FeatureTrackerBase {
 public:
  FeatureTrackerBase(const TrackerParams& params, Camera::Ptr camera,
                     ImageDisplayQueue* display_queue);

  cv::Mat computeImageTracks(const Frame& previous_frame,
                             const Frame& current_frame,
                             bool debug = false) const;

 protected:
  /**
   * @brief Checks if a keypoint is within an image, taking into account the
   * shrink row/col values in the params. If these values are zero, it just
   * checks that the keypoint is within the image size, as given by the camera
   * parameters.
   *
   * @param kp
   * @return true
   * @return false
   */
  bool isWithinShrunkenImage(const Keypoint& kp) const;

 protected:
  const TrackerParams params_;
  const cv::Size img_size_;  //! Expected image size from the camera

  Camera::Ptr camera_;
  ImageDisplayQueue* display_queue_;
};

}  // namespace dyno
