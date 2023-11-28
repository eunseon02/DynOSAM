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

#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <unordered_map>

namespace dyno {

using SymbolChar = unsigned char;
static constexpr SymbolChar kPoseSymbolChar = 'X';
static constexpr SymbolChar kObjectMotionSymbolChar = 'H';
static constexpr SymbolChar kStaticLandmarkSymbolChar = 'l';
static constexpr SymbolChar kDynamicLandmarkSymbolChar = 'm';


using CalibrationType = gtsam::Cal3DS2; //TODO: really need to check that this one matches the calibration in the camera!!

enum class LandmarkType { SMART, PROJECTION };
using Slot = long int;

using SmartProjectionFactor = gtsam::SmartProjectionPoseFactor<CalibrationType>;
using SmartProjectionFactorParams = gtsam::SmartProjectionParams;

/// @brief Map of tracklet id to slot. If slot is -1, it measns the factor is not in the map yet.
/// Used for both smart and projection factors
using TrackletIdSlotMap =
    std::unordered_map<TrackletId, Slot>>;

using TrackletIdSmartFactorMap =
    std::unordered_map<TrackletId, SmartProjectionFactor::shared_ptr>;

using TrackletIdLabelMap = std::unordered_map<TrackletId, ObjectId>;
using LandmarkMap = std::unordered_map<TrackletId, Landmark>;
using TrackletIdToTypeMap = std::unordered_map<TrackletId, LandmarkType>;


} //dyno
