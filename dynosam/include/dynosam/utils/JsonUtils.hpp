/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include <nlohmann/json.hpp>
#include <optional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>

#include <opencv4/opencv2/opencv.hpp>
#include <gtsam/geometry/Pose3.h>

using json = nlohmann::json;

namespace dyno {

// wrapper to write a type t that is json seralizable to an open ofstream
//with a set width
template<typename T>
std::ofstream& writeJson(std::ofstream& os, const T& t) {
    //T must be json seralizable
    const json j = t;
    os << std::setw(4) << j;
    return os;
};

}


/**
 * @brief Provide direct specalisations of nlohmann::adl_serializer
 * for gtsam/opencv types that we want to seralize. These are not declared in the header file
 * as they are template specalisations. They will directly be found by the json library.
 * We write them as specalisations and not free functions as per the guidelines of the library
 * (essentially to avoid namespace hijaking...)
 *
 */


namespace nlohmann {
    /**
     * @brief ADL (argument-dependant lookup) seralizer for std::optional types.
     * We want an std::optional seralizer as we frequently use this type
     * within our data structs so this saves having to write additional boiler-plate code.
     *
     * A free to_json/from_json within the dyno (or type specific namespace) must be available
     * for T (see basic example or the one that already exists for the GroundTruthPackets)
     * must exist.
     *
     * Taken directly from the nlohmann::json readme.
     *
     * @tparam T
     */
    template <typename T>
    struct adl_serializer<std::optional<T>> {
        static void to_json(json& j, const std::optional<T>& opt) {
            if (opt == std::nullopt) {
                j = nullptr;
            } else {
              j = *opt; // this will call adl_serializer<T>::to_json which will
                        // find the free function to_json in T's namespace!
            }
        }

        static void from_json(const json& j, std::optional<T>& opt) {
            if (j.is_null()) {
                opt = std::nullopt;
            } else {
                opt = j.template get<T>(); // same as above, but with
                                           // adl_serializer<T>::from_json
            }
        }
    };

    /**
     * @brief General template on Eigen. Somewhat experimental...
     *
     * We would LOVE to be able to template on Derived and specalise on
     * Eigen::MatrixBase<Derived> but for reaons (outlined in this extensiive PR from ma boi:
     * https://github.com/nlohmann/json/issues/3267) we cannot, at least with our version of nlohmann's library.
     * For later jesse or users: I installed the json lib with apt install in April 2024 so maybe this apt has not
     * been updated for a while and the souce code has a bug fix to this issue.
     *
     * NOTE: i imagine this will also fail for dynamic sized arrays....
     *
     * @tparam Scalar
     * @tparam Rows
     * @tparam Cols
     */
    template<typename Scalar, int Rows, int Cols>
    struct adl_serializer<Eigen::Matrix<Scalar, Rows, Cols>> {
        static_assert(Rows != Eigen::Dynamic, "Not implemented for dynamic rows");
        static_assert(Cols != Eigen::Dynamic, "Not implemented for dynamic cols");

        static void to_json(json& j, const Eigen::Matrix<Scalar, Rows, Cols>& matrix) {
            for (int row = 0; row < matrix.rows(); ++row) {
                json column = json::array();
                for (int col = 0; col < matrix.cols(); ++col) {
                    column.push_back(matrix(row, col));
                }
                j.push_back(column);
            }
        }

        static void from_json(const json& j, Eigen::Matrix<Scalar, Rows, Cols>& matrix) {
            for (std::size_t row = 0; row < j.size(); ++row) {
                const auto& jrow = j.at(row);
                for (std::size_t col = 0; col < jrow.size(); ++col) {
                    const auto& value = jrow.at(col);
                    value.get_to(matrix(row, col));
                }
            }
        }
    };


    template <>
    struct adl_serializer<gtsam::Pose3> {
        static void to_json(json& j, const gtsam::Pose3& pose) {
            j["tx"] = pose.x();
            j["ty"] = pose.y();
            j["tz"] = pose.z();

            const auto& rotation = pose.rotation();
            const auto& quaternion = rotation.toQuaternion();

            j["qx"] = quaternion.x();
            j["qy"] = quaternion.y();
            j["qz"] = quaternion.z();
            j["qw"] = quaternion.w();

        }

        static void from_json(const json& j, gtsam::Pose3& pose) {
            gtsam::Point3 translation(
                j["tx"].template get<double>(),
                j["ty"].template get<double>(),
                j["tz"].template get<double>()
            );

            //Note: w, x, y, z order
            gtsam::Rot3 rotation(
                j["qw"].template get<double>(),
                j["qx"].template get<double>(),
                j["qy"].template get<double>(),
                j["qz"].template get<double>()
            );

            pose = gtsam::Pose3(rotation, translation);
        }
    };

    template <typename T>
    struct adl_serializer<cv::Rect_<T>> {
        static void to_json(json& j, const cv::Rect_<T>& rect) {
            j["x"] = rect.x;
            j["y"] = rect.y;
            j["width"] = rect.width;
            j["height"] = rect.height;
        }

        static void from_json(const json& j, cv::Rect_<T>& rect) {
            T x = j["x"].template get<T>();
            T y = j["y"].template get<T>();
            T width = j["width"].template get<T>();
            T height = j["height"].template get<T>();

            rect = cv::Rect_<T>(x, y, width, height);
        }
    };
}
