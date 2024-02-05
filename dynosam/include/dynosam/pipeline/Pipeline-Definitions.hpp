// /*
//  *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
//  *   All rights reserved.

//  *   Permission is hereby granted, free of charge, to any person obtaining a copy
//  *   of this software and associated documentation files (the "Software"), to deal
//  *   in the Software without restriction, including without limitation the rights
//  *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  *   copies of the Software, and to permit persons to whom the Software is
//  *   furnished to do so, subject to the following conditions:

//  *   The above copyright notice and this permission notice shall be included in all
//  *   copies or substantial portions of the Software.

//  *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  *   SOFTWARE.
//  */

// #pragma once

// #include "dynosam/pipeline/PipelineParams.hpp"
// #include "dynosam/frontend/FrontendParams.hpp"
// #include "dynosam/backend/BackendParams.hpp"
// #include "dynosam/common/CameraParams.hpp"
// #include "dynosam/frontend/Frontend-Definitions.hpp"

// namespace dyno {

// class DynoParams : public PipelineParams {


// public:

//     DynoParams(const std::string& params_folder_path);


//     DynoParams(const std::string& pipeline_params_filepath,
//                const std::string& frontend_params_filepath,
//                const std::string& backend_params_filepath,
//                const std::string& cam_params_filepath);

//     //does nothing as everything should be initalsied in the constructor as we have no default
//     bool fromYamlFile(const std::string&) override;
//     std::string toString() const override;
//     bool equals(const PipelineParams& obj, double tol = 1e-9) const override;


// public:
//     static constexpr char kPipelineFilename[] = "PipelineParams.yaml";
//     static constexpr char kCameraFilename[] = "CameraParams.yaml";
//     static constexpr char kFrontendFilename[] = "FrontendParams.yaml";
//     static constexpr char kBackendFilename[] = "BackendParams.yaml";

//     FrontendParams frontend_params_;
//     BackendParams backend_params_;
//     CameraParams camera_params_;

//     int data_provider_type_; //Kitti, VirtualKitti, Online...

//     FrontendType frontend_type_ = FrontendType::kRGBD;


//     //! Pipeline level params
//     bool parallel_run_{true};

// };


// } //dyno
