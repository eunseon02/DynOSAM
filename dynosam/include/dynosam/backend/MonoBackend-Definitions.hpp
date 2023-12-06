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

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/common/Exceptions.hpp"

namespace dyno {

/**
 * @brief
 * SMART: Indicates that the tracklet has a smart factor associated with it but has not yet been triangulated. It may be in the graph (check by looking at the slot)
 * PROJECTION: Indicates that the tracklet has been triangulated and is now a projection factor in the graph
 *
 */
enum ProjectionFactorType {
    SMART,
    PROJECTION
};

struct ProjectionFactorStatus {
    TrackletId tracklet_id_;
    ProjectionFactorType pf_type_;
    ObjectId object_id_;

    ProjectionFactorStatus(TrackletId tracklet_id, ProjectionFactorType pf_type, ObjectId object_id) : tracklet_id_(tracklet_id), pf_type_(pf_type), object_id_(object_id) {}
};

using TrackletIdToProjectionStatus = std::unordered_map<TrackletId, ProjectionFactorStatus>;


// Will be deleted from the smart projection factor map once the smart factor is converted to a projection factor
//if SlotIndex is -1, means that the factor has not been inserted yet in the graph
class SmartProjectionFactorMap : public gtsam::FastMap<TrackletId, std::pair<SmartProjectionFactor::shared_ptr, Slot>> {
public:

    inline bool exists(const TrackletId tracklet_id) const {
        const auto& it = this->find(tracklet_id);
        return it != this->end();
    }

    inline void add(TrackletId tracklet_id, SmartProjectionFactor::shared_ptr smart_factor, Slot slot) {
        //what if already exists?
        auto pair = std::make_pair(smart_factor, slot);
        this->insert({tracklet_id, pair});
    }


    inline SmartProjectionFactor::shared_ptr getSmartFactor(TrackletId tracklet_id) {
        checkAndThrow(exists(tracklet_id), "Cannot get smart factor from smart projection map as it does not exist. Offending tracklet: " + std::to_string(tracklet_id));
        return this->at(tracklet_id).first;
    }

    inline Slot& getSlot(TrackletId tracklet_id) {
        checkAndThrow(exists(tracklet_id), "Cannot get slot smart projection map as it does not exist. Offending tracklet: " + std::to_string(tracklet_id));
        return this->at(tracklet_id).second;
    }

    inline Slot getSlot(TrackletId tracklet_id) const {
        checkAndThrow(exists(tracklet_id), "Cannot get slot smart projection map as it does not exist. Offending tracklet: " + std::to_string(tracklet_id));
        return this->at(tracklet_id).second;
    }

};




} //dyno
