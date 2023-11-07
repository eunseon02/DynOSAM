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

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {

struct ColourMap {

constexpr static int COLOURS_MAX = 150;


static inline cv::Scalar getRainbowColour(size_t index) {
    CHECK(index < COLOURS_MAX);
    return cv::Scalar(
        colmap[index][0] * 255.0,
        colmap[index][1] * 255.0,
        colmap[index][2] * 255.0,
        255.0);
}

//if use_opencv_conventtion then colours are reversed to be BGRA
static inline cv::Scalar convertConention(ObjectId label, bool use_opencv_convention = false) {
  cv::Scalar colour = getObjectColour();
  if(use_opencv_convention) {
    return cv::Scalar(colour(2), colour(1), colour(0), colour(3));
  }
  return colour;
}

static inline cv::Scalar getObjectColour(ObjectId label) {
    while (label > 25)
  {
    label = label / 2.0;
  }
  switch (label)
  {
    case 0:
      return cv::Scalar(0, 0, 255, 255);  // red
    case 1:
      return cv::Scalar(128, 0, 128, 255);  // 255, 165, 0
    case 2:
      return cv::Scalar(255, 255, 0, 255);
    case 3:
      return cv::Scalar(0, 255, 0, 255);  // 255,255,0
    case 4:
      return cv::Scalar(255, 0, 0, 255);  // 255,192,203
    case 5:
      return cv::Scalar(0, 255, 255, 255);
    case 6:
      return cv::Scalar(128, 0, 128, 255);
    case 7:
      return cv::Scalar(255, 255, 255, 255);
    case 8:
      return cv::Scalar(255, 228, 196, 255);
    case 9:
      return cv::Scalar(180, 105, 255, 255);
    case 10:
      return cv::Scalar(165, 42, 42, 255);
    case 11:
      return cv::Scalar(35, 142, 107, 255);
    case 12:
      return cv::Scalar(45, 82, 160, 255);
    case 13:
      return cv::Scalar(0, 0, 255, 255);  // red
    case 14:
      return cv::Scalar(255, 165, 0, 255);
    case 15:
      return cv::Scalar(0, 255, 0, 255);
    case 16:
      return cv::Scalar(255, 255, 0, 255);
    case 17:
      return cv::Scalar(255, 192, 203, 255);
    case 18:
      return cv::Scalar(0, 255, 255, 255);
    case 19:
      return cv::Scalar(128, 0, 128, 255);
    case 20:
      return cv::Scalar(255, 255, 255, 255);
    case 21:
      return cv::Scalar(255, 228, 196, 255);
    case 22:
      return cv::Scalar(180, 105, 255, 255);
    case 23:
      return cv::Scalar(165, 42, 42, 255);
    case 24:
      return cv::Scalar(35, 142, 107, 255);
    case 25:
      return cv::Scalar(45, 82, 160, 255);
    case 41:
      return cv::Scalar(60, 20, 220, 255);
    default:
      LOG(WARNING) << "Label is " << label;
      return cv::Scalar(60, 20, 220, 255);
  }
}

static constexpr double colmap[COLOURS_MAX][3] = { { 0, 0, 0.5263157895 },
                                 { 0, 0, 0.5526315789 },
                                 { 0, 0, 0.5789473684 },
                                 { 0, 0, 0.6052631579 },
                                 { 0, 0, 0.6315789474 },
                                 { 0, 0, 0.6578947368 },
                                 { 0, 0, 0.6842105263 },
                                 { 0, 0, 0.7105263158 },
                                 { 0, 0, 0.7368421053 },
                                 { 0, 0, 0.7631578947 },
                                 { 0, 0, 0.7894736842 },
                                 { 0, 0, 0.8157894737 },
                                 { 0, 0, 0.8421052632 },
                                 { 0, 0, 0.8684210526 },
                                 { 0, 0, 0.8947368421 },
                                 { 0, 0, 0.9210526316 },
                                 { 0, 0, 0.9473684211 },
                                 { 0, 0, 0.9736842105 },
                                 { 0, 0, 1 },
                                 { 0, 0.0263157895, 1 },
                                 { 0, 0.0526315789, 1 },
                                 { 0, 0.0789473684, 1 },
                                 { 0, 0.1052631579, 1 },
                                 { 0, 0.1315789474, 1 },
                                 { 0, 0.1578947368, 1 },
                                 { 0, 0.1842105263, 1 },
                                 { 0, 0.2105263158, 1 },
                                 { 0, 0.2368421053, 1 },
                                 { 0, 0.2631578947, 1 },
                                 { 0, 0.2894736842, 1 },
                                 { 0, 0.3157894737, 1 },
                                 { 0, 0.3421052632, 1 },
                                 { 0, 0.3684210526, 1 },
                                 { 0, 0.3947368421, 1 },
                                 { 0, 0.4210526316, 1 },
                                 { 0, 0.4473684211, 1 },
                                 { 0, 0.4736842105, 1 },
                                 { 0, 0.5, 1 },
                                 { 0, 0.5263157895, 1 },
                                 { 0, 0.5526315789, 1 },
                                 { 0, 0.5789473684, 1 },
                                 { 0, 0.6052631579, 1 },
                                 { 0, 0.6315789474, 1 },
                                 { 0, 0.6578947368, 1 },
                                 { 0, 0.6842105263, 1 },
                                 { 0, 0.7105263158, 1 },
                                 { 0, 0.7368421053, 1 },
                                 { 0, 0.7631578947, 1 },
                                 { 0, 0.7894736842, 1 },
                                 { 0, 0.8157894737, 1 },
                                 { 0, 0.8421052632, 1 },
                                 { 0, 0.8684210526, 1 },
                                 { 0, 0.8947368421, 1 },
                                 { 0, 0.9210526316, 1 },
                                 { 0, 0.9473684211, 1 },
                                 { 0, 0.9736842105, 1 },
                                 { 0, 1, 1 },
                                 { 0.0263157895, 1, 0.9736842105 },
                                 { 0.0526315789, 1, 0.9473684211 },
                                 { 0.0789473684, 1, 0.9210526316 },
                                 { 0.1052631579, 1, 0.8947368421 },
                                 { 0.1315789474, 1, 0.8684210526 },
                                 { 0.1578947368, 1, 0.8421052632 },
                                 { 0.1842105263, 1, 0.8157894737 },
                                 { 0.2105263158, 1, 0.7894736842 },
                                 { 0.2368421053, 1, 0.7631578947 },
                                 { 0.2631578947, 1, 0.7368421053 },
                                 { 0.2894736842, 1, 0.7105263158 },
                                 { 0.3157894737, 1, 0.6842105263 },
                                 { 0.3421052632, 1, 0.6578947368 },
                                 { 0.3684210526, 1, 0.6315789474 },
                                 { 0.3947368421, 1, 0.6052631579 },
                                 { 0.4210526316, 1, 0.5789473684 },
                                 { 0.4473684211, 1, 0.5526315789 },
                                 { 0.4736842105, 1, 0.5263157895 },
                                 { 0.5, 1, 0.5 },
                                 { 0.5263157895, 1, 0.4736842105 },
                                 { 0.5526315789, 1, 0.4473684211 },
                                 { 0.5789473684, 1, 0.4210526316 },
                                 { 0.6052631579, 1, 0.3947368421 },
                                 { 0.6315789474, 1, 0.3684210526 },
                                 { 0.6578947368, 1, 0.3421052632 },
                                 { 0.6842105263, 1, 0.3157894737 },
                                 { 0.7105263158, 1, 0.2894736842 },
                                 { 0.7368421053, 1, 0.2631578947 },
                                 { 0.7631578947, 1, 0.2368421053 },
                                 { 0.7894736842, 1, 0.2105263158 },
                                 { 0.8157894737, 1, 0.1842105263 },
                                 { 0.8421052632, 1, 0.1578947368 },
                                 { 0.8684210526, 1, 0.1315789474 },
                                 { 0.8947368421, 1, 0.1052631579 },
                                 { 0.9210526316, 1, 0.0789473684 },
                                 { 0.9473684211, 1, 0.0526315789 },
                                 { 0.9736842105, 1, 0.0263157895 },
                                 { 1, 1, 0 },
                                 { 1, 0.9736842105, 0 },
                                 { 1, 0.9473684211, 0 },
                                 { 1, 0.9210526316, 0 },
                                 { 1, 0.8947368421, 0 },
                                 { 1, 0.8684210526, 0 },
                                 { 1, 0.8421052632, 0 },
                                 { 1, 0.8157894737, 0 },
                                 { 1, 0.7894736842, 0 },
                                 { 1, 0.7631578947, 0 },
                                 { 1, 0.7368421053, 0 },
                                 { 1, 0.7105263158, 0 },
                                 { 1, 0.6842105263, 0 },
                                 { 1, 0.6578947368, 0 },
                                 { 1, 0.6315789474, 0 },
                                 { 1, 0.6052631579, 0 },
                                 { 1, 0.5789473684, 0 },
                                 { 1, 0.5526315789, 0 },
                                 { 1, 0.5263157895, 0 },
                                 { 1, 0.5, 0 },
                                 { 1, 0.4736842105, 0 },
                                 { 1, 0.4473684211, 0 },
                                 { 1, 0.4210526316, 0 },
                                 { 1, 0.3947368421, 0 },
                                 { 1, 0.3684210526, 0 },
                                 { 1, 0.3421052632, 0 },
                                 { 1, 0.3157894737, 0 },
                                 { 1, 0.2894736842, 0 },
                                 { 1, 0.2631578947, 0 },
                                 { 1, 0.2368421053, 0 },
                                 { 1, 0.2105263158, 0 },
                                 { 1, 0.1842105263, 0 },
                                 { 1, 0.1578947368, 0 },
                                 { 1, 0.1315789474, 0 },
                                 { 1, 0.1052631579, 0 },
                                 { 1, 0.0789473684, 0 },
                                 { 1, 0.0526315789, 0 },
                                 { 1, 0.0263157895, 0 },
                                 { 1, 0, 0 },
                                 { 0.9736842105, 0, 0 },
                                 { 0.9473684211, 0, 0 },
                                 { 0.9210526316, 0, 0 },
                                 { 0.8947368421, 0, 0 },
                                 { 0.8684210526, 0, 0 },
                                 { 0.8421052632, 0, 0 },
                                 { 0.8157894737, 0, 0 },
                                 { 0.7894736842, 0, 0 },
                                 { 0.7631578947, 0, 0 },
                                 { 0.7368421053, 0, 0 },
                                 { 0.7105263158, 0, 0 },
                                 { 0.6842105263, 0, 0 },
                                 { 0.6578947368, 0, 0 },
                                 { 0.6315789474, 0, 0 },
                                 { 0.6052631579, 0, 0 },
                                 { 0.5789473684, 0, 0 },
                                 { 0.5526315789, 0, 0 } };

};

} //dyno
