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

#include <dynosam/common/Types.hpp>

#include "rclcpp/time.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/parameter.hpp"
#include "rcl_interfaces/msg/parameter.hpp"

#include "rosidl_runtime_cpp/traits.hpp"

#include <type_traits>

namespace dyno {

namespace internal {

template<typename Msg, typename = int>
struct HasMsgHeader : std::false_type{};

template <typename Msg>
struct HasMsgHeader<Msg, decltype((void) Msg::header, 0)> : std::true_type { };


template<typename Msg, typename = int>
struct HasChildFrame : std::false_type{};

template <typename Msg>
struct HasChildFrame<Msg, decltype((void) Msg::child_frame_id, 0)> : std::true_type { };


}


namespace utils {


/**
 * @brief Convert ROS time to timestamp
 *
 * @param time
 * @return Timestamp time in seconds
 */
Timestamp fromRosTime(const rclcpp::Time& time);

/**
 * @brief Converts Timestamp (in seconds) to ROS time
 *
 * @param timestamp
 * @return rclcpp::Time
 */
rclcpp::Time toRosTime(Timestamp timestamp);

/**
 * @brief Converts from input type to output type where output type is a ROS message with a header.
 * OUTPUT type MUST be a ROS message type
 *
 * Checks if the OUTPUT type has a variable called header and then sets the timestamp and frame id
 * Checks if the OUTPUT type has a variable called child_frame_id and, if child_frame_id provided, sets the value
 *
 */
template<typename INPUT, typename OUTPUT,
    typename = std::enable_if_t<rosidl_generator_traits::is_message<OUTPUT>::value>>
bool convertWithHeader(const INPUT& input, OUTPUT& output, Timestamp timestamp, const std::string& frame_id, std::optional<std::string> child_frame_id = std::nullopt) {
    if(!dyno::convert<INPUT, OUTPUT>(input, output)) return false;

    //does actually have a header
    if constexpr (internal::HasMsgHeader<OUTPUT>::value) {
        output.header.stamp = toRosTime(timestamp);
        output.header.frame_id = frame_id;
    }

    //if has child frame id
    if constexpr (internal::HasChildFrame<OUTPUT>::value) {
        if(child_frame_id) {
            output.child_frame_id = child_frame_id.value();
        }
    }

    return true;
}


} //utils
} //dyno
