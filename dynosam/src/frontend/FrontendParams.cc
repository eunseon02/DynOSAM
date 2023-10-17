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

#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/utils/YamlParser.hpp"

#include <string>

namespace dyno {

FrontendParams FrontendParams::fromYaml(const std::string& file_path) {
    YamlParser yaml_parser(file_path);

    FrontendParams params;
    yaml_parser.getYamlParam("MaxTrackPointBG", &params.max_tracking_points_bg);
    yaml_parser.getYamlParam("MaxTrackPointOBJ", &params.max_tracking_points_obj);

    yaml_parser.getYamlParam("SFMgThres", &params.scene_flow_magnitude);
    yaml_parser.getYamlParam("SFDsThres", &params.scene_flow_percentage);

    yaml_parser.getYamlParam("ThDepthBG", &params.depth_background_thresh);
    yaml_parser.getYamlParam("ThDepthOBJ", &params.depth_obj_thresh);

    yaml_parser.getYamlParam("DepthMapFactor", &params.depth_scale_factor);

    yaml_parser.getYamlParam("ORBextractor.nFeatures", &params.n_features);
    yaml_parser.getYamlParam("ORBextractor.scaleFactor", &params.scale_factor);
    yaml_parser.getYamlParam("ORBextractor.nLevels", &params.n_levels);
    yaml_parser.getYamlParam("ORBextractor.iniThFAST", &params.init_threshold_fast);
    yaml_parser.getYamlParam("ORBextractor.minThFAST", &params.min_threshold_fast);

    return params;

}


} //dyno
