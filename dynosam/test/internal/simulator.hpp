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

#include <gtsam/geometry/Pose3.h>

namespace dyno_testing {

using namespace dyno;

class ScenarioBody {

public:
    DYNO_POINTER_TYPEDEFS(ScenarioBody)

    virtual ~ScenarioBody() {}

    virtual gtsam::Pose3 pose(double t) const = 0;  ///< pose at time t
    virtual gtsam::Pose3 motion_l(double t) const = 0;  ///< motion local frame from t-1 to t, in ^{t-1}L_{t-1}
    virtual gtsam::Pose3 motion_w(double t) const; ///< motion in world frame from t-1 to t

    virtual double entersScenario() const = 0;
    virtual double leavesScenario() const = 0;

    gtsam::Rot3 rotation(double t) const { return pose(t).rotation(); }
    gtsam::Vector3 translation(double t) const { return pose(t).translation(); }
};


class Scenario {

public:
    Scenario(ScenarioBody::Ptr camera_body) : camera_body_(camera_body) {}

    void addObjectBody(ObjectId object_id, ScenarioBody::Ptr object_body) {
        object_bodies_.insert2(object_id, object_body);
    }

    gtsam::Pose3 cameraPose(double t) const { camera_body_->pose(t); }

    ObjectIds getObjectIds(double t) const {
        ObjectIds object_ids;
        for(const auto&[object_id, obj] : object_bodies_) {
            if(objectInScenario(object_id, t)) object_ids.push_back(object_id);
        }

        return object_ids;
    }

    bool objectInScenario(ObjectId object_id, double t) const {
        if(object_bodies_.exists(object_id)) {
            const auto& object = object_bodies_.at(object_id);

            return t >= object->entersScenario() && t < object->leavesScenario();
        }
        return false;
    }

protected:
    ScenarioBody::Ptr camera_body_;
    gtsam::FastMap<ObjectId, ScenarioBody::Ptr> object_bodies_;

};


} // namespace dyno_testing
