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

#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/pipeline/PipelineManager.hpp"
#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include "dynosam/visualizer/OpenCVFrontendDisplay.hpp"

#include "dynosam/common/Camera.hpp"

#include <glog/logging.h>

int main(int argc, char* argv[]) {

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;
    FLAGS_log_prefix = 1;

    dyno::CameraParams camera_params = dyno::CameraParams::fromYamlFile("some_file.yaml");
    dyno::Camera::Ptr camera = std::make_shared<dyno::Camera>(camera_params);


    auto data_loader = std::make_unique<dyno::KittiDataLoader>("/root/data/kitti/0000");
    auto frontend_module = std::make_shared<dyno::RGBDInstanceFrontendModule>(dyno::FrontendParams(), camera);
    auto frontend_display = std::make_shared<dyno::OpenCVFrontendDisplay>();

    dyno::DynoPipelineManager pipeline(std::move(data_loader), frontend_module, frontend_display);
    pipeline.spin();



}
