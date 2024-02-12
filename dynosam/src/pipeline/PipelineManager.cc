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
#include "dynosam/frontend/MonoInstanceFrontendModule.hpp"
#include "dynosam/backend/MonoBatchBackendModule.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/backend/MonoBackendModule.hpp"
#include "dynosam/backend/RGBDBackendModule.hpp"
#include "dynosam/utils/TimingStats.hpp"

#include <glog/logging.h>


namespace dyno {

DynoPipelineManager::DynoPipelineManager(const DynoParams& params, DataProvider::UniquePtr data_loader, FrontendDisplay::Ptr frontend_display, BackendDisplay::Ptr backend_display)
    :   params_(params),
        data_loader_(std::move(data_loader)),
        displayer_(&display_queue_, params.parallel_run_)

{
    LOG(INFO) << "Starting DynoPipelineManager";

    CHECK(data_loader_);
    CHECK(frontend_display);

    //TODO: we should not set starting from here as this should not be a property of the DataProvider (e.g this doesnt make sense for an online system)
    //see comment in DataProvider header

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
    BackendModule::Ptr backend = nullptr;

    Map::Ptr map = Map::create();

    const CameraParams& camera_params = params_.camera_params_;
    //eventually from actual params
    switch (params_.frontend_type_)
    {
        case FrontendType::kRGBD: {
            LOG(INFO) << "Making RGBDInstance frontend";
            Camera::Ptr camera = std::make_shared<Camera>(camera_params);
            frontend = std::make_shared<RGBDInstanceFrontendModule>(params.frontend_params_, camera, &display_queue_);
            backend = std::make_shared<RGBDBackendModule>(params_.backend_params_, camera, map, &display_queue_);
        }   break;
        case FrontendType::kMono: {
            LOG(INFO) << "Making MonoInstance frontend";
            Camera::Ptr camera = std::make_shared<Camera>(camera_params);
            frontend = std::make_shared<MonoInstanceFrontendModule>(params.frontend_params_, camera, &display_queue_);
            backend = std::make_shared<MonoBatchBackendModule>(params.backend_params_, camera, &display_queue_);
        }   break;

        default: {
            LOG(FATAL) << "Not implemented!";
        }  break;
    }


    frontend_pipeline_ = std::make_unique<FrontendPipeline>("frontend-pipeline", &frontend_input_queue_, frontend);
    frontend_pipeline_->registerOutputQueue(&frontend_viz_input_queue_);
    frontend_pipeline_->parallelRun(params.parallel_run_);

    if(backend) {
        backend_pipeline_ = std::make_unique<BackendPipeline>("backend-pipeline", &backend_input_queue_, backend);
        backend_pipeline_->parallelRun(params.parallel_run_);
        //also register connection between front and back
        frontend_pipeline_->registerOutputQueue(&backend_input_queue_);

        backend_pipeline_->registerOutputQueue(&backend_output_queue_);
    }

    if(backend && backend_display) {
        backend_viz_pipeline_ = std::make_unique<BackendVizPipeline>("backend-viz-pipeline", &backend_output_queue_, backend_display);
    }

    frontend_viz_pipeline_ = std::make_unique<FrontendVizPipeline>("frontend-viz-pipeline", &frontend_viz_input_queue_, frontend_display);

    launchSpinners();

}

DynoPipelineManager::~DynoPipelineManager() {
    shutdownPipelines();
    shutdownSpinners();
}


void DynoPipelineManager::shutdownSpinners() {
    if(frontend_pipeline_spinner_) frontend_pipeline_spinner_->shutdown();

    if(backend_pipeline_spinner_) backend_pipeline_spinner_->shutdown();

    if(data_provider_spinner_) data_provider_spinner_->shutdown();

    if(frontend_viz_pipeline_spinner_) frontend_viz_pipeline_spinner_->shutdown();

    if(backend_viz_pipeline_spinner_) backend_viz_pipeline_spinner_->shutdown();
}

void DynoPipelineManager::shutdownPipelines() {
    display_queue_.shutdown();
    frontend_pipeline_->shutdown();

    if(backend_pipeline_) backend_pipeline_->shutdown();

    data_interface_->shutdown();

    if(frontend_viz_pipeline_) frontend_viz_pipeline_->shutdown();
    if(backend_viz_pipeline_) backend_viz_pipeline_->shutdown();
}

bool DynoPipelineManager::spin() {

    utils::TimingStatsCollector timer("pipeline_spin");
    if(data_loader_->spin() || frontend_pipeline_->isWorking()) {
        if(!params_.parallel_run_) {
            frontend_pipeline_->spinOnce();
            if(backend_pipeline_) backend_pipeline_->spinOnce();
        }
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
    LOG(INFO) << "Running PipelineManager with parallel_run=" << params_.parallel_run_;

    if(params_.parallel_run_) {
        frontend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::FrontendPipeline::spin, frontend_pipeline_.get()), "frontend-pipeline-spinner");

        if(backend_pipeline_)
            backend_pipeline_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::BackendPipeline::spin, backend_pipeline_.get()), "backend-pipeline-spinner");

    }

    data_provider_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::DataInterfacePipeline::spin, data_interface_.get()), "data-interface-spinner");

    if(frontend_viz_pipeline_)
        frontend_viz_pipeline_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::FrontendVizPipeline::spin, frontend_viz_pipeline_.get()), "frontend-display-spinner");

    if(backend_viz_pipeline_)
        backend_viz_pipeline_spinner_ = std::make_unique<dyno::Spinner>(std::bind(&dyno::BackendVizPipeline::spin, backend_viz_pipeline_.get()), "backend-display-spinner");
}



} //dyno
