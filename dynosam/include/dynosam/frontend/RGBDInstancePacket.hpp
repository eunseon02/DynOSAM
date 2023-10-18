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

// #pragma once

// #include "dynosam/frontend/FrontendInputPacket.hpp"
// #include "dynosam/utils/OpenCVUtils.hpp"

// #include <glog/logging.h>

// namespace dyno {

// struct RGBDInstancePacket : public InputImagePacketBase {

//     DYNO_POINTER_TYPEDEFS(RGBDInstancePacket)

//     const cv::Mat depth_;
//     const cv::Mat instance_mask_;

//     RGBDInstancePacket(
//         const FrameId frame_id,
//         const Timestamp timestamp,
//         const cv::Mat& rgb,
//         const cv::Mat& optical_flow,
//         const cv::Mat& depth,
//         const cv::Mat& instance_mask
//     )
//     :   InputImagePacketBase(
//             frame_id,
//             timestamp,
//             rgb,
//             optical_flow),
//         depth_(depth),
//         instance_mask_(instance_mask)
//     {
//         CHECK(depth_.type() == CV_64F) << "The provided depth image does not have datatype CV_64F";
//         CHECK(!depth_.empty());
//         CHECK(!instance_mask_.empty());
//     }


//     virtual void draw(cv::Mat& img) const override {
//         // draw each portion of the inputs
//         cv::Mat rgb, depth, flow, mask;

//         rgb_.copyTo(rgb);
//         CHECK(rgb.channels() == 3) << "Expecting rgb in frame to gave 3 channels";

//         depth_.copyTo(depth);
//         // expect depth in float 32
//         depth.convertTo(depth, CV_8UC1);

//         optical_flow_.copyTo(flow);

//         instance_mask_.copyTo(mask);

//         // canot display the original ones so these needs special treatment...
//         cv::Mat flow_viz, mask_viz;
//         utils::flowToRgb(flow, flow_viz);
//         utils::semanticMaskToRgb(rgb, mask, mask_viz);

//         cv::Mat top_row = utils::concatenateImagesHorizontally(rgb, depth);
//         cv::Mat bottom_row = utils::concatenateImagesHorizontally(flow_viz, mask_viz);
//         cv::Mat input_images = utils::concatenateImagesVertically(top_row, bottom_row);

//         // reisize images to be the original image size
//         cv::resize(input_images, input_images, cv::Size(rgb.cols, rgb.rows), 0, 0, cv::INTER_LINEAR);

//         input_images.copyTo(img);
//     }

// };

// }
