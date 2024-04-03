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
    yaml_parser.getYamlParam("max_nr_static_points", &params.max_tracking_points_bg, params.max_tracking_points_bg);
    yaml_parser.getYamlParam("max_nr_object_points", &params.max_tracking_points_obj, params.max_tracking_points_obj);
    yaml_parser.getYamlParam("cell_size", &params.cell_size, params.cell_size);

    yaml_parser.getYamlParam("scene_flow_mag_threshold", &params.scene_flow_magnitude, params.scene_flow_magnitude);
    yaml_parser.getYamlParam("scene_flow_dist_threshold", &params.scene_flow_percentage, params.scene_flow_percentage);

    yaml_parser.getYamlParam("max_depth_static", &params.depth_background_thresh, params.depth_background_thresh);
    yaml_parser.getYamlParam("max_depth_object", &params.depth_obj_thresh, params.depth_obj_thresh);

    yaml_parser.getYamlParam("n_features", &params.n_features, params.n_features);
    yaml_parser.getYamlParam("scale_factor", &params.scale_factor, params.scale_factor);
    yaml_parser.getYamlParam("n_levels", &params.n_levels, params.n_levels);
    yaml_parser.getYamlParam("init_threshold_fast", &params.init_threshold_fast, params.init_threshold_fast);
    yaml_parser.getYamlParam("min_threshold_fast", &params.min_threshold_fast, params.min_threshold_fast);

    yaml_parser.getYamlParam("shrink_row", &params.shrink_row, params.shrink_row);
    yaml_parser.getYamlParam("shrink_col", &params.shrink_col, params.shrink_col);

    yaml_parser.getNestedYamlParam(
        "tracker", "ransac_use_2point_mono",
        &params.ransac_use_2point_mono,
        params.ransac_use_2point_mono);

    yaml_parser.getNestedYamlParam(
        "tracker", "ransac_randomize",
        &params.ransac_randomize,
        params.ransac_randomize);

    yaml_parser.getNestedYamlParam(
        "tracker", "ransac_threshold_mono",
        &params.ransac_threshold_mono,
        params.ransac_threshold_mono);

    yaml_parser.getNestedYamlParam(
        "tracker", "optimize_2d2d_pose_from_inliers",
        &params.optimize_2d2d_pose_from_inliers,
        params.optimize_2d2d_pose_from_inliers);

    yaml_parser.getNestedYamlParam(
        "tracker", "use_ego_motion_pnp",
        &params.use_ego_motion_pnp,
        params.use_ego_motion_pnp);

    yaml_parser.getNestedYamlParam(
        "tracker", "use_object_motion_pnp",
        &params.use_object_motion_pnp,
        params.use_object_motion_pnp);

    yaml_parser.getNestedYamlParam(
        "tracker", "ransac_threshold_pnp",
        &params.ransac_threshold_pnp,
        params.ransac_threshold_pnp);

    yaml_parser.getNestedYamlParam(
        "tracker", "optimize_3d2d_pose_from_inliers",
        &params.optimize_3d2d_pose_from_inliers,
        params.optimize_3d2d_pose_from_inliers);

    yaml_parser.getNestedYamlParam(
        "tracker", "ransac_threshold_stereo",
        &params.ransac_threshold_stereo,
        params.ransac_threshold_stereo);

    yaml_parser.getNestedYamlParam(
        "tracker", "optimize_3d3d_pose_from_inliers",
        &params.optimize_3d3d_pose_from_inliers,
        params.optimize_3d3d_pose_from_inliers);

    yaml_parser.getNestedYamlParam(
        "tracker", "ransac_iterations",
        &params.ransac_iterations,
        params.ransac_iterations);

    yaml_parser.getNestedYamlParam(
        "tracker", "ransac_probability",
        &params.ransac_probability,
        params.ransac_probability);


    return params;

}


} //dyno
