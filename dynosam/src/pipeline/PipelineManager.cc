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

#include "dynosam/pipeline/PipelineManager.hpp"
#include <glog/logging.h>

namespace dyno {

DynoPipelineManager::DynoPipelineManager(DataProvider::UniquePtr data_loader,  FrontendModule::Ptr frontend_module, FrontendDisplay::Ptr frontend_display)
    :   data_loader_(std::move(data_loader))

{
    CHECK(frontend_module);
    CHECK(data_loader_);
    CHECK(frontend_display);

    //TODO: factories for different loaders etc later
    data_provider_module_ = std::make_unique<DataProviderModule>("data-provider");
    data_loader_->registerInputImagesCallback(std::bind(&dyno::DataProviderModule::fillInputPacketQueue, data_provider_module_.get(), std::placeholders::_1));
    data_provider_module_->registerOutputQueue(&frontend_input_queue_);


    frontend_pipeline_ = std::make_unique<FrontendPipeline>("frontend-pipeline", &frontend_input_queue_, frontend_module);
    frontend_pipeline_->registerOutputQueue(&frontend_output_queue_);

    frontend_viz_pipeline_ = std::make_unique<FrontendVizPipeline>(&frontend_output_queue_, frontend_display);

}

DynoPipelineManager::~DynoPipelineManager() {}

void DynoPipelineManager::spin(bool parallel_run) {
    if(parallel_run) {
        LOG(INFO) << "Running PipelineManager with parallel_run=true";

        frontend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::FrontendPipeline::spin, frontend_pipeline_.get()), "frontend-pipeline-spinner");
        data_provider_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::DataProviderModule::spin, data_provider_module_.get()), "data-provider-spinner");
        frontend_viz_pipeline_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::FrontendVizPipeline::spin, frontend_viz_pipeline_.get()), "frontend-display-spinner");
    }
    else {
        LOG(FATAL) << "Not implemented";
    }

    data_loader_->spin();
}



} //dyno
