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
#include <functional>

#include <glog/logging.h>

namespace dyno {
namespace internal {

// T is the pointer to the type that is expected to be iterated over
// such that we have struct iterator_traits<T*> or struct iterator_traits<const T*>;
// see https://en.cppreference.com/w/cpp/iterator/iterator_traits

//usually defined as filter_iterator_detail<Container::pointer> could also be <filter_iterator_detail<Iter::point>
template<typename T>
struct filter_iterator_detail {
    //! naming conventions to match those required by iterator traits
    using value_type = typename std::iterator_traits<T>::value_type;
    using reference_type = typename std::iterator_traits<T>::reference;
    using pointer = typename std::iterator_traits<T>::pointer;
    using difference_type = typename std::iterator_traits<T>::difference_type;
    using iterator_category = std::forward_iterator_tag; //i guess? only forward is defined (++iter) right now
};

//container is the actual container type and Iter is the iterator type (eg container::iterator or container::const_iterator)
//in this way, filter_iterator can act as either a regular iterator or a const iterator
//we expect container to also be iterable
//ref type should also be BaseDetail::reference_type?
//check to ensure container types and iter types are the same?
//container must at least contain (const and non const versions of end and begin which must return actual iterators which satisfy the
//forward iterator category)
//container must also at least have the definitions for Container::iterator, Container::const_iterator, Container::const_reference
//and Iter must be a properly defined iterator such that it satisfies all the conditions for a forward_iterator (e.f Iter::value_type, Iter::reference etc...)
template<typename Container, typename Iter, typename FilterFunction = std::function<bool(typename Container::const_reference)>>
struct filter_iterator_base : public filter_iterator_detail<typename Iter::pointer> {
public:

    using BaseDetail = filter_iterator_detail<typename Iter::pointer>;
    using iterator = Iter;
    using container = Container;
    using typename BaseDetail::value_type;
    using typename BaseDetail::reference_type;
    using typename BaseDetail::pointer;
    using typename BaseDetail::difference_type;
    using typename BaseDetail::iterator_category;

    filter_iterator_base(Container& container, const FilterFunction& filter_func) : filter_iterator_base(container, filter_func,container.begin()) {}

    reference_type operator*() { return *it_; }
    reference_type operator->() { return *it_; }

    bool operator==(const filter_iterator_base& other) const {
        return it_ == other.it_;
    }
    bool operator!=(const filter_iterator_base& other) const { return it_ != other.it_; }

    bool operator==(const iterator& other) const {
        return it_ == other;
    }
    bool operator!=(const iterator& other) const { return it_ != other; }

    // inline size_t size() const {
    //     return std::distance(begin(), end());
    // }

    Container& getContainer() { return container_; }
    const Container& getContainer() const { return container_; }

    bool operator()(typename Container::const_reference arg) const {
        return filter_func_(arg);
    }


    //preincrement (++iter)
    filter_iterator_base& operator++() {
        do {
            ++it_;
        }
        while(is_invalid());
        return *this;
    }

    //TODO??
    filter_iterator_base& operator++(int x) {
        do {
            it_+=x;
        }
        while(is_invalid());
        return *this;
    }

    //allows the iterator to be used as a enhanced for loop
    filter_iterator_base begin() { return filter_iterator_base(container_, filter_func_, container_.begin()); }
    filter_iterator_base end() { return filter_iterator_base(container_, filter_func_, container_.end()); }

    const filter_iterator_base begin() const { return filter_iterator_base(container_, filter_func_, container_.begin()); }
    const filter_iterator_base end() const { return filter_iterator_base(container_, filter_func_, container_.end()); }

private:
    bool is_invalid() const {
        return it_ != container_.end() && !filter_func_(*it_);
    }

    void find_next_valid() {
        while(is_invalid()) {
            this->operator++();
        }
    }

protected:
    filter_iterator_base(Container& container, const FilterFunction& filter_func, iterator it)
        :   container_(container), filter_func_(filter_func), it_(it)
        {  find_next_valid(); }

protected:
    Container& container_; //! reference to container
    FilterFunction filter_func_;
    mutable iterator it_;

};

template<typename Container>
using filter_iterator = filter_iterator_base<Container, typename Container::iterator>;

template<typename Container>
using filter_const_iterator = filter_iterator_base<Container, typename Container::const_iterator>;


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


} //internal

} //dyno
