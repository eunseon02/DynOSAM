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

#include <string>

namespace dyno {

struct FrontendParams {

// tracking points params
  int max_tracking_points_bg = 800;
  int max_tracking_points_obj = 800;

  // scene flow thresholds
  double scene_flow_magnitude = 0.12;
  double scene_flow_percentage = 0.3;

  // depth thresholds
  double depth_background_thresh = 40.0;
  double depth_obj_thresh = 25.0;

  double depth_scale_factor = 256.0;

  double base_line = 387.5744; //for now?

  // ORB detector params
  int n_features = 1200;
  double scale_factor = 1.2;
  int n_levels = 8;
  int init_threshold_fast = 20;
  int min_threshold_fast = 7;

  static FrontendParams fromYaml(const std::string& file_path);

};

}
