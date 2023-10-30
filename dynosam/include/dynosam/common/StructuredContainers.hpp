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

#include <unordered_map>
#include <vector>
#include <memory>

#include <glog/logging.h>

namespace dyno {

// template<typename T>
// class TrackletContainer {
// public:
//     using Value = T;
//     using This = TrackletContainer<T>;
//     using ValuePtr = std::shared_ptr<Value>;
//     using TrackletMap = std::unordered_map<TrackletId, ValuePtr>;
//     using ValuePtrVector = std::vector<ValuePtr>;

//     using ValuePtrVectorIterator = typename ValuePtrVector::iterator;
//     using ValuePtrVectorConstIterator = typename ValuePtrVector::const_iterator;

//     TrackletContainer() : tracklet_value_map_(), value_vector_() {}
//     TrackletContainer(const ValuePtrVector& value_vector) {
//         for(size_t i = 0; i < value_vector.size(); i++) {
//             This::add(value_vector.at(i));
//         }
//     }

//     virtual ~TrackletContainer() = default;

//     virtual void add(TrackletId tracklet_id, ValuePtr value) {
//         CHECK(!exists(tracklet_id));
//         tracklet_value_map_[tracklet_id] = value;
//         value_vector_.push_back(value);
//     }

//     virtual TrackletIds collectTracklets() const {
//         TrackletIds tracklets;
//         for(const auto& [tracklet_id, value] : tracklet_value_map_) {
//             tracklets.push_back(tracklet_id);
//         }
//         return tracklets;
//     }

//     size_t size() const {
//         CHECK(tracklet_value_map_.size() == value_vector_.size());
//         return value_vector_.size();
//     }

//     bool exists(TrackletId tracklet_id) const {
//         return tracklet_value_map_.find(tracklet_id) != tracklet_value_map_.end();
//     }

//     ValuePtr getByTrackletId(TrackletId tracklet_id) const {
//         if(!exists(tracklet_id)) {
//             return nullptr;
//         }
//         return tracklet_value_map_.at(tracklet_id);
//     }

//     ValuePtr at(size_t i) const {
//         return value_vector_.at(i);
//     }


//     //vector begin
//     ValuePtrVectorIterator begin() { return value_vector_.begin(); }
//     ValuePtrVectorConstIterator begin() const { return value_vector_.cbegin(); }

//     //vector end
//     ValuePtrVectorIterator end() { return value_vector_.end(); }
//     ValuePtrVectorConstIterator end() const { return value_vector_.cend(); }

// protected:
//     TrackletMap tracklet_value_map_;
//     ValuePtrVector value_vector_;

// };

} //dyno
