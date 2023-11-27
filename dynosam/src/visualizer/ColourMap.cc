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

#include "dynosam/visualizer/ColourMap.hpp"


namespace dyno {

cv::Scalar ColourMap::HSV2RGB(cv::Scalar hsv, bool use_opencv_convention) {
    //determine if input is 0-1 or 0-255. The converter algorithm needs the input colour
    //to be 0-1 but we want to return the rgb colour in the same form as the input
    //we dont have a great way of doing this so we just check if any of the SV values > 1
    //if they're not the colour is basically black anyway

    cv::Scalar hsv_norm = hsv;

    const bool requires_normalization =  GET_SATURATION(hsv) > 1 || GET_VALUE(hsv) > 1; // if true, range is 0-255
    if(requires_normalization) {
        hsv_norm(1) /= 255.0;
        hsv_norm(2) /= 255.0;
    }

    double hh, p, q, t, ff;
    long i;
    cv::Scalar out;

    if(hsv_norm(1) <= 0.0) {       // < is bogus, just shuts up warnings
        out = cv::Scalar(GET_VALUE(hsv), GET_VALUE(hsv), GET_VALUE(hsv)); //value
        out(3) = 255.0; //set Alpha

        if(requires_normalization) {
            out(0) *= 255.0;
            out(1) *= 255.0;
            out(2) *= 255.0;
        }
        return RGBA2BGRA(out, use_opencv_convention);
    }
    hh = GET_HUE(hsv_norm); //hue
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = GET_VALUE(hsv_norm) * (1.0 - GET_SATURATION(hsv_norm));
    q = GET_VALUE(hsv_norm) * (1.0 - (GET_SATURATION(hsv_norm) * ff));
    t = GET_VALUE(hsv_norm) * (1.0 - (GET_SATURATION(hsv_norm) * (1.0 - ff)));

    switch(i) {
    case 0:
        GET_R(out) = GET_VALUE(hsv_norm);
        GET_G(out)= t;
        GET_B(out) = p;
        break;
    case 1:
        GET_R(out) = q;
        GET_G(out) =  GET_VALUE(hsv_norm);
        GET_B(out) = p;
        break;
    case 2:
        GET_R(out) = p;
        GET_G(out) =  GET_VALUE(hsv_norm);
        GET_B(out) = t;
        break;

    case 3:
        GET_R(out) = p;
        GET_G(out) = q;
        GET_B(out) =  GET_VALUE(hsv_norm);
        break;
    case 4:
        GET_R(out) = t;
        GET_G(out) = p;
        GET_B(out) =  GET_VALUE(hsv_norm);
        break;
    case 5:
    default:
        GET_R(out) = GET_VALUE(hsv_norm);
        GET_G(out) = p;
        GET_B(out) = q;
        break;
    }

    if(requires_normalization) {
        out(0) *= 255.0;
        out(1) *= 255.0;
        out(2) *= 255.0;
        out(3) = 255.0; //set Alpha
    }

    return RGBA2BGRA(out, use_opencv_convention);


}

} //dyno
