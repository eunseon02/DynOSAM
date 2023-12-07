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
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/backend/DynamicPointSymbol.hpp"

#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <gtsam/inference/Symbol.h>

#include <variant>
#include <unordered_map>

namespace dyno {

using SymbolChar = unsigned char;
static constexpr SymbolChar kPoseSymbolChar = 'X';
static constexpr SymbolChar kObjectMotionSymbolChar = 'H';
static constexpr SymbolChar kStaticLandmarkSymbolChar = 'l';
static constexpr SymbolChar kDynamicLandmarkSymbolChar = 'm';

inline gtsam::Key H(unsigned char label, std::uint64_t j) {return gtsam::LabeledSymbol(kObjectMotionSymbolChar, label, j);}

inline gtsam::Symbol CameraPoseSymbol(FrameId frame_id) { return gtsam::Symbol(kPoseSymbolChar, frame_id); }
inline gtsam::Symbol ObjectMotionSymbol(FrameId frame_id) { return gtsam::Symbol(kObjectMotionSymbolChar, frame_id); }
inline gtsam::Symbol StaticLandmarkSymbol(TrackletId tracklet_id) { return gtsam::Symbol(kStaticLandmarkSymbolChar, tracklet_id); }
inline gtsam::Symbol DynamicLandmarkSymbol(FrameId frame_id, TrackletId tracklet_id) { return gtsam::Symbol(kDynamicLandmarkSymbolChar, tracklet_id); }


inline gtsam::Key ObjectMotionSymbol(ObjectId object_label, FrameId frame_id)
{
  unsigned char label = object_label + '0';
  return H(label, static_cast<std::uint64_t>(frame_id));
}

/**
 * @brief Construct an object motion symbol with the current frame id.
 *
 * A motion takes us from k-1 to k so we index using the k-1 frame id.
 * This function takes the current frame id (k) which is normally what we work with and constructs the motion symbol
 * using k-1 as the index.
 *
 * This does assume that we're tracking the object frame to frame
 *
 * @param object_label
 * @param current_frame_id
 * @return gtsam::Key
 */
inline gtsam::Key ObjectMotionSymbolFromCurrentFrame(ObjectId object_label, FrameId current_frame_id) {
    CHECK(current_frame_id > 0) << "Current frame Id must be at least 1 so that we can index from the previous frame!";
    return ObjectMotionSymbol(object_label, current_frame_id - 1u);
}

std::string DynoLikeKeyFormatter(gtsam::Key);

using CalibrationType = gtsam::Cal3DS2; //TODO: really need to check that this one matches the calibration in the camera!!


using Slot = long int;

constexpr static Slot UninitialisedSlot = -1; //! Inidicates that a factor is not in the graph or uninitialised

using SmartProjectionFactor = gtsam::SmartProjectionPoseFactor<CalibrationType>;
using GenericProjectionFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, CalibrationType>;

using SmartProjectionFactorParams = gtsam::SmartProjectionParams;

/**
 * @brief Metadata of a landmark. Includes type (static/dynamic) and label.
 *
 * Label may be background at which point the KeyPointType should be background_label
 * Also includes information on how the landamrk was estimated, age etc...
 *
 */
struct LandmarkStatus {

    enum Method { MEASURED, TRIANGULATED, OPTIMIZED };

    ObjectId label_; //! Will be 0 if background
    Method method_; //! How the landmark was constructed

};


/// @brief A pair relating a tracklet ID with an estiamted landmark
using LandmarkEstimate = std::pair<TrackletId, Landmark>;
/// @brief A pair relating a Landmark estimate (TrackletId + Landmark) with a status - inidicating type and the object label
using StatusLandmarkEstimate = std::pair<LandmarkStatus, LandmarkEstimate>;
/// @brief A vector of StatusLandmarkEstimate
using StatusLandmarkEstimates = std::vector<StatusLandmarkEstimate>;

/**
 * @brief Definition of a tracklet - a tracket point seen at multiple frames.
 *
 *
 * @tparam MEASUREMENT
 */
template<typename MEASUREMENT>
class DynamicObjectTracklet : public gtsam::FastMap<FrameId, MEASUREMENT> {

public:

    using Measurement = MEASUREMENT;
    using This = DynamicObjectTracklet<Measurement>;
    using Base = gtsam::FastMap<FrameId, Measurement>;

    DYNO_POINTER_TYPEDEFS(This)

    DynamicObjectTracklet(TrackletId tracklet_id) : tracklet_id_(tracklet_id) {}

    void add(FrameId frame_id, const Measurement& measurement) {
        this->insert({frame_id, measurement});
    }

    inline TrackletId getTrackletId() const { return tracklet_id_; }

private:
    TrackletId tracklet_id_;

};


template<typename MEASUREMENT>
class DynamicObjectTrackletManager {

public:
    using DynamicObjectTrackletM = DynamicObjectTracklet<MEASUREMENT>;

    using ObjectToTrackletIdMap = gtsam::FastMap<ObjectId, TrackletIds>;

    /// @brief TrackletId to a DynamicObjectTracklet for quick access
    using TrackletMap = gtsam::FastMap<TrackletId, typename DynamicObjectTrackletM::Ptr>;

    DynamicObjectTrackletManager() {}

    void add(ObjectId object_id, const TrackletId tracklet_id, const FrameId frame_id, const MEASUREMENT& measurement) {
        typename DynamicObjectTrackletM::Ptr tracklet = nullptr;
        if(trackletExists(tracklet_id)) {
            tracklet = tracklet_map_.at(tracklet_id);
        }
        else {
            //must be a completely new tracklet as we dont have an tracklet associated with it
            tracklet = std::make_shared<DynamicObjectTrackletM>(tracklet_id);
            CHECK(tracklet);

            tracklet_map_.insert({tracklet_id, tracklet});
        }

        CHECK(tracklet);
        CHECK(!tracklet->exists(frame_id)) << "Attempting to add tracklet measurement with object id "
            << object_id << " and tracklet_id " << tracklet_id
            << " to frame " << frame_id << " but an entry already exists at this frame";

        //if new object
        if(!object_trackletid_map_.exists(object_id)) {
            object_trackletid_map_.insert({object_id, TrackletIds{}});
        }

        auto& tracklet_ids = object_trackletid_map_.at(object_id);
        tracklet_ids.push_back(tracklet_id);

        CHECK(trackletExists(tracklet->getTrackletId()));

        tracklet->add(frame_id, measurement);
    }


    TrackletIds getPerObjectTracklets(const ObjectId object_id) {
        checkAndThrow(objectExists(object_id), "getPerObjectTracklets query failed as object id id does not exist. Offedning Id " + std::to_string(object_id));
        return object_trackletid_map_.at(object_id);
    }

    DynamicObjectTrackletM& getByTrackletId(const TrackletId tracklet_id) {
        checkAndThrow(trackletExists(tracklet_id), "DynamicObjectTracklet query failed as tracklet id does not exist. Offedning Id " + std::to_string(tracklet_id));
        return *tracklet_map_.at(tracklet_id);
    }

    inline bool objectExists(const ObjectId object_id) {
        return object_trackletid_map_.exists(object_id);
    }

    inline bool trackletExists(const TrackletId tracklet_id) const {
        return tracklet_map_.exists(tracklet_id);
    }

private:
    ObjectToTrackletIdMap object_trackletid_map_; //!! object ids to tracklet ids
    TrackletMap tracklet_map_;

};




} //dyno
