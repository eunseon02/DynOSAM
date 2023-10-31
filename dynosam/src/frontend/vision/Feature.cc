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

#include "dynosam/frontend/vision/Feature.hpp"

#include <glog/logging.h>

namespace dyno {

FeatureContainer::FeatureContainer() : feature_map_(), feature_vector_() {}

FeatureContainer::FeatureContainer(const FeaturePtrs feature_vector) {
    for(size_t i = 0; i < feature_vector.size(); i++) {
        add(feature_vector.at(i));
    }
}

void FeatureContainer::add(Feature::Ptr feature) {
    CHECK(!exists(feature->tracklet_id_));

    feature_map_[feature->tracklet_id_] = feature;
    feature_vector_.push_back(feature);
}

TrackletIds FeatureContainer::collectTracklets(bool only_usable) const {
    TrackletIds tracklets;
    for(const auto& feature : *this) {
        if(only_usable && feature->usable()) {
            tracklets.push_back(feature->tracklet_id_);
        }
        else {
            tracklets.push_back(feature->tracklet_id_);
        }
    }

    return tracklets;
}

FeatureContainer::FilterIterator FeatureContainer::usableIterator() {
    return FilterIterator(feature_vector_, [](const Feature::Ptr& f) -> bool { return f->usable(); });
}

 void FeatureContainer::markOutliers(const TrackletIds outliers) {
    for(TrackletId tracklet_id : outliers) {
        CHECK(exists(tracklet_id));

        getByTrackletId(tracklet_id)->inlier_ = false;
    }
 }

size_t FeatureContainer::size() const {
    CHECK(feature_map_.size() == feature_vector_.size());
    return feature_vector_.size();
}

Feature::Ptr FeatureContainer::getByTrackletId(TrackletId tracklet_id) const {
    if(!exists(tracklet_id)) {
        return nullptr;
    }
    return feature_map_.at(tracklet_id);
}

Feature::Ptr FeatureContainer::at(size_t i) const {
    return feature_vector_.at(i);
}

bool FeatureContainer::exists(TrackletId tracklet_id) const {
    return feature_map_.find(tracklet_id) != feature_map_.end();
}

} //dyno
