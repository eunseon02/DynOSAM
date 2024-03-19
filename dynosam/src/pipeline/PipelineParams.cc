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

#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam/utils/YamlParser.hpp"

namespace dyno {

DynoParams::DynoParams(const std::string& params_folder_path) {

    //currently just camera params
    camera_params_ = CameraParams::fromYamlFile(params_folder_path + "CameraParams.yaml");

    frontend_params_ = FrontendParams::fromYaml(params_folder_path + "FrontendParams.yaml");

    YamlParser pipeline_parser(params_folder_path + "PipelineParams.yaml");
    pipeline_parser.getYamlParam("parallel_run", &parallel_run_);

    pipeline_parser.getYamlParam("data_provider_type", &data_provider_type_);

    int frontend_type_i;
    pipeline_parser.getYamlParam("frontend_type", &frontend_type_i);
    frontend_type_ = static_cast<FrontendType>(frontend_type_i);

    int optimizer_type_i;
    pipeline_parser.getYamlParam("optimizer_type", &optimizer_type_i);
    optimizer_type_ = static_cast<OptimizerType>(optimizer_type_i);

}


}
