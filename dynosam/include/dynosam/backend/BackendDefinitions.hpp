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
#include "dynosam/logger/Logger.hpp"
#include "dynosam/common/GroundTruthPacket.hpp"

#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/LabeledSymbol.h>

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
inline gtsam::Symbol StaticLandmarkSymbol(TrackletId tracklet_id) { return gtsam::Symbol(kStaticLandmarkSymbolChar, tracklet_id); }
inline DynamicPointSymbol DynamicLandmarkSymbol(FrameId frame_id, TrackletId tracklet_id) { return DynamicPointSymbol(kDynamicLandmarkSymbolChar, tracklet_id, frame_id); }


inline gtsam::Key ObjectMotionSymbol(ObjectId object_label, FrameId frame_id)
{
  unsigned char label = object_label + '0';
  return H(label, static_cast<std::uint64_t>(frame_id));
}

bool checkIfLabeledSymbol(gtsam::Key key);

bool reconstructMotionInfo(gtsam::Key key, ObjectId& object_label, FrameId& frame_id);


//TODO: depricate!!
/**
 * @brief Construct an object motion symbol with the current frame id (k).
 *
 * A motion takes us from k-1 to k so we index using the k frame id.
 * This function takes the current frame id (k) which is normally what we work with and constructs the motion symbol
 * using k as the index such that the motion takes us from the k-1 to k.
 *
 * This does assume that we're tracking the object frame to frame
 *
 * @param object_label
 * @param current_frame_id
 * @return gtsam::Key
 */
inline gtsam::Key ObjectMotionSymbolFromCurrentFrame(ObjectId object_label, FrameId current_frame_id) {
    CHECK(current_frame_id > 0) << "Current frame Id must be at least 1 so that we can index from the previous frame!";
    return ObjectMotionSymbol(object_label, current_frame_id);
}

std::string DynoLikeKeyFormatter(gtsam::Key);

std::string DynoLikeKeyFormatterVerbose(gtsam::Key);

using CalibrationType = gtsam::Cal3DS2; //TODO: really need to check that this one matches the calibration in the camera!!

using Slot = long int;

constexpr static Slot UninitialisedSlot = -1; //! Inidicates that a factor is not in the graph or uninitialised

using SmartProjectionFactor = gtsam::SmartProjectionPoseFactor<CalibrationType>;
using GenericProjectionFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, CalibrationType>;

using SmartProjectionFactorParams = gtsam::SmartProjectionParams;


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

    FrameId getFirstFrame() const {
        return this->begin()->first;;
    }

private:
    TrackletId tracklet_id_;

};


class DebugInfo {
public:
    int num_static_factors = 0; //num new static factors added
    int num_new_static_points = 0;

    struct ObjectInfo {
        int num_dynamic_factors = 0;
        int num_new_dynamic_points = 0;
        int num_motion_factors = 0;
        bool smoothing_factor_added{false};
    };

    gtsam::FastMap<ObjectId, ObjectInfo> object_info{};

    bool odometry_factor_added {false};

    double update_static_time = 0;
    double update_dynamic_time = 0;
    double optimize_time = 0;

    double error_before = 0;
    double error_after = 0;

    size_t num_factors = 0;
    size_t num_values = 0;

    int num_elements_in_matrix = 0;
    int num_zeros_in_matrix = 0;

};

//TODO: currently FrontendLogger really is RGBDLogger?
class BackendLogger : public EstimationModuleLogger {
public:
    DYNO_POINTER_TYPEDEFS(BackendLogger)
    BackendLogger() : EstimationModuleLogger("backend") {}
};



//TODO: depricate
class DynamicObjectManager {
public:
    using ObjectToTrackletIdMap = gtsam::FastMap<ObjectId, TrackletIds>;
    using FrameToObjects = gtsam::FastMap<FrameId, ObjectIds>;
    using FrameToTracklets = gtsam::FastMap<FrameId, TrackletIds>;
    using TrackletIdTObject = gtsam::FastMap<TrackletId, ObjectId>;

    virtual void add(ObjectId object_id, const TrackletId tracklet_id, const FrameId frame_id) {
        addToStructure(object_tracklet_map_, object_id, tracklet_id);
        addToStructure(frame_to_objects_map_, frame_id, object_id);
        addToStructure(frame_to_tracklets_map_, frame_id, tracklet_id);

        //sanity check that tracklet Id only every appears on the same object
        //assumes that the tracklet is correcrly associated with the first object label association
        if(tracklet_to_object_map_.exists(tracklet_id)) {
            CHECK_EQ(tracklet_to_object_map_.at(tracklet_id), object_id);
        }
        else {
            tracklet_to_object_map_.insert({tracklet_id, object_id});
        }
    }

    ObjectIds getPerFrameObjects(const FrameId frame_id) const {
        // checkAndThrow(frameExists(frame_id), "getPerFrameObjects query failed as frame id id does not exist. Offedning Id " + std::to_string(frame_id));
        if(!frameExists(frame_id)) {
            return ObjectIds{};
        }
        return frame_to_objects_map_.at(frame_id);
    }

    TrackletIds getPerObjectTracklets(const ObjectId object_id) {
        checkAndThrow(objectExists(object_id), "getPerObjectTracklets query failed as object id id does not exist. Offedning Id " + std::to_string(object_id));
        return object_tracklet_map_.at(object_id);
    }

    // TrackletIds getPerFrameObjectTracklets(const FrameId frame_id, const ObjectId object_id) const {
    //     //All the tracklets for this object (over multiple frames)
    //     TrackletIds all_object_tracklets = getPerObjectTracklets(object_id);
    //     //all the tracklets in this frame (all objects inclusive)
    //     TrackletIds per_frame_tracklets = frame_to_tracklets_map_.at(frame_id);

    //     //we want the tracklets for this object that appear in the current frame


    // }

    inline bool frameExists(const FrameId frame_id) const {
        return frame_to_objects_map_.exists(frame_id);
    }

    inline bool objectExists(const ObjectId object_id) {
        return object_tracklet_map_.exists(object_id);
    }

    inline bool trackletExists(const TrackletId tracklet_id) const {
        return tracklet_to_object_map_.exists(tracklet_id);
    }

    inline ObjectId getObjectIdByTracklet(const TrackletId tracklet_id) const {
        checkAndThrow(tracklet_to_object_map_.exists(tracklet_id), "getObjectIdByTracklet query failed as tracklet id does not exist. Offedning Id " + std::to_string(tracklet_id));
        return tracklet_to_object_map_.at(tracklet_id);
    }

    //slow as we do an actual saerch through map
    //output will be sorted lowest to highest
    FrameIds getFramesPerObject(const ObjectId object_id) const {
        std::set<FrameId, std::less<FrameId>> all_frames;
        for(const auto&[frame_id, object_ids] : frame_to_objects_map_) {
            if(std::find(object_ids.begin(), object_ids.end(), object_id) != object_ids.end()) {
                all_frames.insert(frame_id);
            }
        }

        //frame_to_objects_map_ should be sorted becasue gtsam::FasMap uses std::less (as does std::set)
        return FrameIds(all_frames.begin(), all_frames.end());
    }

protected:
    /**
     * @brief Add to map of key to some vector of values.
     *
     * If key does not exist, a new value vector will be created and value added to it.
     * Value is only added to the vector in the map if it is not already present, such that all values are unique
     * and the vector acts like a set (why not use a set then, silly?)
     *
     * @tparam Key
     * @tparam Value
     * @tparam Allocator
     * @param map
     * @param key
     * @param value
     */
    template<typename Key, typename Value, typename Allocator>
    void addToStructure(gtsam::FastMap<Key, std::vector<Value, Allocator>>& map, Key key, Value value) {
        using VectorValue = std::vector<Value, Allocator>;
        //if new
        if(!map.exists(key)) {
            map.insert({key, VectorValue{}});
        }
        auto& vector_values = map.at(key);

        //only add if value already not present
        if(std::find(vector_values.begin(), vector_values.end(), value) == vector_values.end()) {
            vector_values.push_back(value);
        }
    }

protected:
    ObjectToTrackletIdMap object_tracklet_map_; //! object ids to tracklet ids
    FrameToObjects frame_to_objects_map_;
    FrameToTracklets frame_to_tracklets_map_;
    TrackletIdTObject tracklet_to_object_map_;



};


template<typename MEASUREMENT>
class DynamicObjectTrackletManager : public DynamicObjectManager {

public:
    using DynamicObjectTrackletM = DynamicObjectTracklet<MEASUREMENT>;

    /// @brief TrackletId to a DynamicObjectTracklet for quick access
    using TrackletMap = gtsam::FastMap<TrackletId, typename DynamicObjectTrackletM::Ptr>;

    DynamicObjectTrackletManager() {}

    void add(ObjectId object_id, const TrackletId tracklet_id, const FrameId frame_id, const MEASUREMENT& measurement) {
        DynamicObjectManager::add(object_id, tracklet_id, frame_id);
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



        CHECK(trackletExists(tracklet->getTrackletId()));

        tracklet->add(frame_id, measurement);


        //sanity check that tracklet Id only every appears on the same object
        //assumes that the tracklet is correcrly associated with the first object label association
        if(tracklet_to_object_map_.exists(tracklet_id)) {
            CHECK_EQ(tracklet_to_object_map_.at(tracklet_id), object_id);
        }
        else {
            tracklet_to_object_map_.insert({tracklet_id, object_id});
        }
    }

    DynamicObjectTrackletM& getByTrackletId(const TrackletId tracklet_id) {
        checkAndThrow(trackletExists(tracklet_id), "DynamicObjectTracklet query failed as tracklet id does not exist. Offedning Id " + std::to_string(tracklet_id));
        return *tracklet_map_.at(tracklet_id);
    }

    inline bool trackletExists(const TrackletId tracklet_id) const {
        return tracklet_map_.exists(tracklet_id);
    }

private:
    TrackletMap tracklet_map_;
};





} //dyno
