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
#include "dynosam/dataprovider/DataProviderModule.hpp"
#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include "dynosam/utils/Spinner.hpp"

#include "dynosam/frontend/FrontendPipeline.hpp"
#include <glog/logging.h>

int main(int argc, char* argv[]) {

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;
    FLAGS_log_prefix = 1;


    dyno::DataProviderModule data_module("rbgd-instance provider module");
    dyno::KittiDataLoader data_loader("/root/data/kitti/0000");

    data_loader.registerInputImagesCallback(std::bind(&dyno::DataProviderModule::fillInputPacketQueue, &data_module, std::placeholders::_1));

    dyno::FrontendPipeline::InputQueue frontend_input_queue;
    data_module.registerOutputQueue(&frontend_input_queue);

    auto frontend_module = std::make_shared<dyno::RGBDInstanceFrontendModule>();
    dyno::FrontendPipeline frontend_pipeline("frontend-pipeline", &frontend_input_queue, frontend_module);

     //for now in this order as the frontend spinner is blocking?
    auto frontend_spinner = std::make_unique<dyno::Spinner>(std::bind(&dyno::FrontendPipeline::spin, &frontend_pipeline), "frontend-pipeline-spinner");
    auto data_spinner = std::make_unique<dyno::Spinner>(std::bind(&dyno::DataProviderModule::spin, &data_module), "data-provider-spinner");

    data_loader.spin();


}
