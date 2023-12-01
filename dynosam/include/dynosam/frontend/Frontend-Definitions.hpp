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

namespace dyno {

/**
 * @brief Determines which frontend module to load.
 *
 *
 *
 */
enum class FrontendType {
    kRGBD = 0,
    kMono = 1
};



/**
 * @brief Metadata of a keypoint. Includes type (static/dynamic) and label.
 *
 * Label may be background at which point the KeyPointType should be background_label
 *
 */
struct KeypointStatus {
  const KeyPointType kp_type_;
  const ObjectId label_; //! Will be 0 if background

  KeypointStatus(KeyPointType kp_type, ObjectId label) : kp_type_(kp_type), label_(label) {}

  inline bool isStatic() const {
    const bool is_static = (kp_type_ == KeyPointType::STATIC);
    {
      //sanity check
      if(is_static) CHECK_EQ(label_, background_label) << "Keypoint Type is STATIC but label is not background label (" << background_label << ")";
    }
    return is_static;
  }

  inline static KeypointStatus Static() {
    return KeypointStatus(KeyPointType::STATIC, background_label);
  }

  inline static KeypointStatus Dynamic(ObjectId label) {
    CHECK(label != background_label);
    return KeypointStatus(KeyPointType::DYNAMIC, label);
  }
};

/// @brief A pair relating a tracklet ID with an observed keypoint
using KeypointMeasurement = std::pair<TrackletId, Keypoint>;
/// @brief A pair relating a Keypoint measurement (TrackletId + Keypoint) with a status - inidicating the keypoint type and the object label
using StatusKeypointMeasurement = std::pair<KeypointStatus, KeypointMeasurement>;
/// @brief A vector of StatusKeypointMeasurements
using StatusKeypointMeasurements = std::vector<StatusKeypointMeasurement>;

}
