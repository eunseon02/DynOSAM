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

//if use_opencv_convention then colours are reversed to be BGRA
//expects colour to be RGBA
static inline cv::Scalar RGBA2BGRA(cv::Scalar colour, bool use_opencv_convention = false) {
  if(use_opencv_convention) {
    return cv::Scalar(colour(2), colour(1), colour(0), colour(3));
  }
  return colour;
}

static inline cv::Scalar getObjectColour(ObjectId label, bool use_opencv_convention = false) {
  while (label > 25)
  {
    label = label / 2.0;
  }

  cv::Scalar colour;
  switch (label)
  {
    case 0:
      colour = cv::Scalar(0, 0, 255, 255);  // red
      break;
    case 1:
      colour = cv::Scalar(128, 0, 128, 255);  // 255, 165, 0
      break;
    case 2:
      colour =  cv::Scalar(10, 255, 0, 255);
      break;
    case 3:
      colour =  cv::Scalar(0, 255, 0, 255);  // 255,255,0
      break;
    case 4:
      colour =  cv::Scalar(255, 0, 0, 255);  // 255,192,203
      break;
    case 5:
      colour =  cv::Scalar(0, 255, 255, 255);
      break;
    case 6:
      colour =  cv::Scalar(128, 0, 128, 255);
      break;
    case 7:
      colour =  cv::Scalar(255, 255, 255, 255);
       break;
    case 8:
      colour =  cv::Scalar(255, 228, 196, 255);
      break;
    case 9:
      colour =  cv::Scalar(180, 105, 255, 255);
       break;
    case 10:
      colour =  cv::Scalar(165, 42, 42, 255);
       break;
    case 11:
      colour =  cv::Scalar(35, 142, 107, 255);
       break;
    case 12:
      colour =  cv::Scalar(45, 82, 160, 255);
       break;
    case 13:
      colour =  cv::Scalar(0, 0, 255, 255);  // red
       break;
    case 14:
      colour =  cv::Scalar(255, 165, 0, 255);
       break;
    case 15:
      colour =  cv::Scalar(0, 255, 0, 255);
       break;
    case 16:
      colour =  cv::Scalar(255, 255, 0, 255);
       break;
    case 17:
      colour =  cv::Scalar(255, 192, 203, 255);
       break;
    case 18:
      colour =  cv::Scalar(0, 255, 255, 255);
       break;
    case 19:
      colour =  cv::Scalar(128, 0, 128, 255);
       break;
    case 20:
      colour =  cv::Scalar(125, 255, 75, 255);
       break;
    case 21:
      colour =  cv::Scalar(255, 228, 196, 255);
       break;
    case 22:
      colour =  cv::Scalar(180, 105, 255, 255);
       break;
    case 23:
      colour =  cv::Scalar(165, 42, 42, 255);
       break;
    case 24:
      colour =  cv::Scalar(35, 142, 107, 255);
       break;
    case 25:
      colour =  cv::Scalar(45, 82, 160, 255);
       break;
    case 41:
      colour =  cv::Scalar(60, 20, 220, 255);
       break;
    default:
      LOG(WARNING) << "Label is " << label;
      colour =  cv::Scalar(60, 20, 220, 255);
       break;
  }

  return RGBA2BGRA(colour, use_opencv_convention);
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
