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

inline gtsam::Symbol CameraPoseSymbol(FrameId frame_id) { return gtsam::Symbol(kPoseSymbolChar, frame_id); }
inline gtsam::Symbol ObjectMotionSymbol(FrameId frame_id) { return gtsam::Symbol(kObjectMotionSymbolChar, frame_id); }


using CalibrationType = gtsam::Cal3DS2; //TODO: really need to check that this one matches the calibration in the camera!!


/**
 * @brief
 * SMART: Indicates that the tracklet has a smart factor associated with it but has not yet been triangulated. It may be in the graph (check by looking at the slot)
 * PROJECTION: Indicates that the tracklet has been triangulated and is not a projection factor in the graph
 *
 */
enum class ProjectionFactorType {
    SMART,
    PROJECTION
};
using Slot = long int;

constexpr static Slot UninitialisedSlot = -1; //! Inidicates that a factor is not in the graph or uninitialised

using SmartProjectionFactor = gtsam::SmartProjectionPoseFactor<CalibrationType>;
using GenericProjectionFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, CalibrationType>;

using SmartProjectionFactorParams = gtsam::SmartProjectionParams;

using ProjectionFactorVariant = std::variant<SmartProjectionFactor::shared_ptr, ProjectionFactor::shared_ptr>;

template<typename Visitor>
class ProjectionFactorVisitor {
public:
    ProjectionFactorVisitor(Visitor& visitor) : visitor_(visitor) {}
    ProjectionFactorVisitor() : visitor_(std::nullopt) {}

    template<typename SpecificVariant>
    auto&& operator(SpecificVariant&& variant) const {
        return std::visit()
    }
private:
    template<typename V>
    auto&& doVisit(V variant) const {
        if(visitor_) {
            Visitor& visitor = visitor_.value();
            return std::visit(visitor, variant);
        }
        else {
            return std::visit(Visitor{}, variant);
        }

    }

private:
    std::optional<Visitor&> visitor_;

};


struct ProjectionFactorStatus {

    TrackletId tracklet_id_;
    ProjectionFactorVariant projection_factor_;
    ObjectId object_id_;
    ProjectionFactorType type_;
    Slot slot_;

    /**
     * @brief Construct a new Projection Factor Status from a smart projection factor
     *
     * We know that with a smart projection factor we will not have an initial measurement yet so the factor will not be in the map.
     * The ProjectionFactorType is initalized as SMART and the slot is initalized as UninitialisedSlot.
     *
     * @param tracklet_id const TrackletId
     * @param projection_factor SmartProjectionFactor::shared_ptr
     * @param object_id ObjectId
     */
    ProjectionFactorStatus(const TrackletId tracklet_id, SmartProjectionFactor::shared_ptr projection_factor, ObjectId object_id)
    : tracklet_id_(tracklet_id), projection_factor_(projection_factor), object_id_(object_id), type_(ProjectionFactorType::SMART), slot_(UninitialisedSlot) {}

};


class ProjectionFactorStatusMap : public std::unordered_map<TrackletId, ProjectionFactorStatus> {
public:
    ProjectionFactorStatusMap();

    inline bool exists(const TrackletId tracklet_id) const {
        const auto& it = this->find(tracklet_id);
        return it != this->end();
    }

    inline void add(const ProjectionFactorStatus& projection_factor_status) {
        //what if already exists?
        this->insert({projection_factor_status.tracklet_id_, projection_factor_status})
    }

    inline ObjectId& getObjectIdProperty(const TrackletId tracklet_id) {
        return this->at(tracklet_id).object_id_;
    }

    inline ObjectId getObjectIdProperty(const TrackletId tracklet_id) const {
        return this->at(tracklet_id).object_id_;
    }

    inline ProjectionFactorType& getFactorTypeProperty(const TrackletId tracklet_id) {
        return this->at(tracklet_id).type_;
    }

    inline ProjectionFactorType getFactorTypeProperty(const TrackletId tracklet_id) const {
        return this->at(tracklet_id).type_;
    }

    inline ProjectionFactorType& getFactorTypeProperty(const TrackletId tracklet_id) {
        return this->at(tracklet_id).type_;
    }



};


} //dyno
