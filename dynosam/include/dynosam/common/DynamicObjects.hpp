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

//eh gross that we need this here - both should be in vision
// #include "dynosam/frontend/vision/Vision-Definitions.hpp"


namespace dyno {

class Camera;

struct DynamicObjectObservation {

    TrackletIds object_features_; //! Tracklet id's of object features within the frame. Does not indicate usability
    ObjectId tracking_label_; //tracking id (not necessarily instance label???), -1 if not tracked yet
    ObjectId instance_label_; //this shoudl really be constant and part of the constructor as we get this straight from the input image and will never change
    bool marked_as_moving_{false};

    DynamicObjectObservation() : object_features_(), tracking_label_(-1) {}
    DynamicObjectObservation(const TrackletIds& object_features, ObjectId tracking_label) : object_features_(object_features), tracking_label_(tracking_label) {}

};

using DynamicObjectObservations = std::vector<DynamicObjectObservation>;


// class TemporalDynamic3DObject {

// public:
//     TemporalDynamic3DObject() {}

//     TemporalDynamic3DObject(ObjectId object_id, const GenericCorrespondences<Landmark, Landmark>& world_correspondences);

// private:
//     ObjectId object_id_;
//     GenericCorrespondences<Landmark, Landmark>& world_correspondences_;

// };





}
