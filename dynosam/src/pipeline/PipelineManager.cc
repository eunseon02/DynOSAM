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
#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include <glog/logging.h>

//TODO: for now should be in params
DEFINE_int32(frontend_type, 0, "Type of parser to use:\n "
                              "0: RGBDInstance");

namespace dyno {

DynoPipelineManager::DynoPipelineManager(const DynoParams& params, DataProvider::UniquePtr data_loader, FrontendDisplay::Ptr frontend_display)
    :   params_(params),
        data_loader_(std::move(data_loader)),
        displayer_(&display_queue_, false)

{
    LOG(INFO) << "Starting DynoPipelineManager";

    CHECK(data_loader_);
    CHECK(frontend_display);

    //TODO: factories for different loaders etc later
    data_interface_ = std::make_unique<DataInterfacePipeline>(params.parallel_run_);
    data_loader_->registerImageContainerCallback(
        std::bind(&dyno::DataInterfacePipeline::fillImageContainerQueue, data_interface_.get(), std::placeholders::_1)
    );

    //ground truth
    data_loader_->registerGroundTruthPacketCallback(
        std::bind(&dyno::DataInterfacePipeline::addGroundTruthPacket, data_interface_.get(), std::placeholders::_1)
    );

    //preprocessing
    data_interface_->registerImageContainerPreprocessor(
        std::bind(&dyno::DataProvider::imageContainerPreprocessor, data_loader_.get(), std::placeholders::_1)
    );

    data_interface_->registerOutputQueue(&frontend_input_queue_);

    FrontendModule::Ptr frontend = nullptr;

    const CameraParams& camera_params = params_.camera_params_;
    //eventually from actual params
    switch (FLAGS_frontend_type)
    {
        case 0: {
            LOG(INFO) << "Making RGBDInstance frontend";
            Camera::Ptr camera = std::make_shared<Camera>(camera_params);
            frontend = std::make_shared<RGBDInstanceFrontendModule>(params.frontend_params_, camera, &display_queue_);
        }   break;

        default: {
            LOG(FATAL) << "Not implemented!";
        }  break;
    }


    frontend_pipeline_ = std::make_unique<FrontendPipeline>("frontend-pipeline", &frontend_input_queue_, frontend);
    frontend_pipeline_->registerOutputQueue(&frontend_output_queue_);

    frontend_viz_pipeline_ = std::make_unique<FrontendVizPipeline>(&frontend_output_queue_, frontend_display);

    launchSpinners();

}

DynoPipelineManager::~DynoPipelineManager() {
    shutdownPipelines();
    shutdownSpinners();
}


void DynoPipelineManager::shutdownSpinners() {
    if(frontend_pipeline_spinner_) frontend_pipeline_spinner_->shutdown();

    if(data_provider_spinner_) data_provider_spinner_->shutdown();

    if(frontend_viz_pipeline_spinner_) frontend_viz_pipeline_spinner_->shutdown();
}

void DynoPipelineManager::shutdownPipelines() {
    display_queue_.shutdown();
    frontend_pipeline_->shutdown();
    data_interface_->shutdown();
    frontend_viz_pipeline_->shutdown();
}

bool DynoPipelineManager::spin() {

    if(data_loader_->spin() || frontend_pipeline_->isWorking()) {
        spinViz(); //for now
        //a later problem!
        return true;
    }
    return false;

}

bool DynoPipelineManager::spinViz() {
    // if()
    displayer_.process(); //when enabled this gives a segafault when the process ends. when commented out the program just waits at thee end
    return true;
}


void DynoPipelineManager::launchSpinners() {
    LOG(INFO) << "Running PipelineManager with parallel_run=true";
    frontend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::FrontendPipeline::spin, frontend_pipeline_.get()), "frontend-pipeline-spinner");
    data_provider_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::DataInterfacePipeline::spin, data_interface_.get()), "data-interface-spinner");
    frontend_viz_pipeline_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::FrontendVizPipeline::spin, frontend_viz_pipeline_.get()), "frontend-display-spinner");
}



} //dyno
