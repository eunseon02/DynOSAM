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

#include "dynosam/common/ImageContainer.hpp"

namespace dyno {

template<typename... ImageTypes>
template<typename... SubsetImageTypes>
ImageContainerSubset<SubsetImageTypes...> ImageContainerSubset<ImageTypes...>::makeSubset(const bool clone) const {
    //the tuple of image wrappers to construct for the subset class
    using Subset = ImageContainerSubset<SubsetImageTypes...>;
    using WrappedImageTypesTupleSubset = typename Subset::WrappedImageTypesTuple;
    static constexpr size_t SubN = Subset::N;

    //wrappers to populate from the current set? what about copying etc
    WrappedImageTypesTupleSubset subset_wrappers;
    for (size_t i = 0; i < SubN; i++) {
        //get the type at the new subset
        internal::select_apply<SubN>(i, [&](auto I) {
            using SubType = typename Subset::ImageTypeStruct<I>;
            //find this image type in the current container as the idnex may be different
            constexpr size_t current_index = This::Index<SubType>();
            //with the index i of the subset wrapper (corresponding with type SubType)
            //get the image wrapper in this container using the request (sub) type
            //TODO:do we want cv::Mat data to be const (refer counter) or at some point clone? See ImageType where the cv::mat is stored?
            auto current_wrapped_image = std::get<current_index>(image_storage_);

            //if invalid but request, throw runtime error becuase whats the point of requesting an invalid image
            if(!current_wrapped_image.exists()) {
                throw std::runtime_error(
                    "Error constructing ImageContainerSubset - requested image type " + type_name<SubType>() + " is empty in parent container");
            }

            if(clone) {
                std::get<I>(subset_wrappers) = current_wrapped_image.clone();
            }
            else {
                std::get<I>(subset_wrappers) = current_wrapped_image;
            }
        });
    }

    //this is going to do a bunch of copies?
    auto construct_subset = [&](auto&&... args) { return Subset(args...); };
    return std::apply(construct_subset, subset_wrappers);
}

template<typename... ImageTypes>
void ImageContainerSubset<ImageTypes...>::validateSetup() const {
    cv::Size required_size;

    for (size_t i = 0; i < N; i++) {
        //get the type at the new subset
        internal::select_apply<N>(i, [&](auto stream_index) {
            using ImageTypeStruct = typename This::ImageTypeStruct<stream_index>;
            //only check if exists?
            if(This::exists<ImageTypeStruct>()) {
                //if required size is empty, then first check
                //set required size to be the image size and then compare against
                //this size on future iterations
                const bool is_first = required_size.empty();
                if(is_first) {
                    required_size = This::get<ImageTypeStruct>().size();
                }
                else {
                    //compare against required size
                    const cv::Size incoming_size = This::get<ImageTypeStruct>().size();
                    if(incoming_size != required_size) {
                        throw InvalidImageContainerException(
                            "Non-empty images were not all the same size. First image (type: " +
                            type_name<This::ImageTypeStruct<0u>>() + ") was of size " + to_string(required_size) +
                            " and other image (type: " + type_name<ImageTypeStruct>() + ") was of size " + to_string(incoming_size)
                        );
                    }
                }
            }
        });
    }
}

}
