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

#include "dynosam/pipeline/Pipeline-Definitions.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam/utils/YamlParser.hpp"
#include "dynosam/common/Types.hpp"


namespace dyno {


decltype(DynoParams::kPipelineFilename) constexpr DynoParams::kPipelineFilename;
decltype(DynoParams::kFrontendFilename) constexpr DynoParams::kFrontendFilename;
decltype(DynoParams::kBackendFilename) constexpr DynoParams::kBackendFilename;
decltype(DynoParams::kCameraFilename) constexpr DynoParams::kCameraFilename;


DynoParams::DynoParams(const std::string& params_folder_path)
    : DynoParams(params_folder_path + "/" + kPipelineFilename,
                 params_folder_path + "/" + kFrontendFilename,
                 params_folder_path + "/" + kBackendFilename,
                 params_folder_path + "/" + kCameraFilename)   {}

DynoParams::DynoParams(
               const std::string& pipeline_params_filepath,
               const std::string& frontend_params_filepath,
               const std::string& backend_params_filepath,
               const std::string& cam_params_filepath)
    :   PipelineParams("DynosamParams")
{
// //currently just camera params
    // camera_params_ = CameraParams::fromYamlFile(params_folder_path + "CameraParams.yaml");

    // YamlParser pipeline_parser(params_folder_path + "PipelineParams.yaml");
    // pipeline_parser.getYamlParam("parallel_run", &parallel_run_);

    // pipeline_parser.getYamlParam("data_provider_type", &data_provider_type_);

    // int frontend_type_i;
    // pipeline_parser.getYamlParam("frontend_type", &frontend_type_i);
    // frontend_type_ = static_cast<FrontendType>(frontend_type_i);

}

bool DynoParams::fromYamlFile(const std::string&) { return true; }
std::string DynoParams::toString() const {
    return "string";
}

bool DynoParams::equals(const PipelineParams& obj, double tol) const {
    return true;
}


} //dyno
