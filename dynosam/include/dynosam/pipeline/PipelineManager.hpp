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

#pragma once

#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam/dataprovider/DataInterfacePipeline.hpp"
#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/frontend/FrontendPipeline.hpp"
#include "dynosam/backend/BackendPipeline.hpp"
#include "dynosam/visualizer/VisualizerPipelines.hpp"
#include "dynosam/utils/Spinner.hpp"
#include "dynosam/common/Types.hpp"

#include "dynosam/backend/Optimizer.hpp"
#include "dynosam/backend/BatchOptimizer.hpp"
#include "dynosam/backend/IncrementalOptimizer.hpp"



namespace dyno {


class DynoPipelineManager {

public:
    DYNO_POINTER_TYPEDEFS(DynoPipelineManager)

    //why are some unique and some shared?? silly silly
    DynoPipelineManager(const DynoParams& params, DataProvider::Ptr data_loader, FrontendDisplay::Ptr frontend_display, BackendDisplay::Ptr backend_display);
    ~DynoPipelineManager();

    /**
     * @brief Spins the whole pipeline by spinning the data provider.
     * If in sequential mode, it will return for each spin
     * If in parallel mode, it will not return until the pipeline is shutdown.
     *
     * The return value indicates the data provider modules state: false if finished or shutdown,
     * true its working nominally
     *
     * @return true
     * @return false
     */
    virtual bool spin();

    virtual bool spinViz();

private:
    void launchSpinners();
    void shutdownSpinners();
    void shutdownPipelines();

    template<typename Measurement>
    typename Optimizer<Measurement>::Ptr createOptimizer(OptimizerType optimizer_type) const;

    void loadPipelines(const CameraParams& camera_params, FrontendDisplay::Ptr frontend_display, BackendDisplay::Ptr backend_display);

private:
    const DynoParams params_;
    const bool use_offline_frontend_;

    //! Use the pipeline base version of the frontend pipeline
    //! This allows us to have different pipelines to provide data to the backend (e.g. load seralized data)
    //! which have different templates
    PipelineBase::UniquePtr frontend_pipeline_{nullptr};
    FrontendPipeline::InputQueue frontend_input_queue_;
    FrontendPipeline::OutputQueue frontend_viz_input_queue_;

    ///TODO: another hack to get the final size of the dataset
    //depending on use_offline_frontend_ this either gets the size from the dataset
    //of from the FrontendOfflinePipeline
    //currently only works for RGB!!
    std::function<FrameId()> get_dataset_size_;

    BackendPipeline::UniquePtr backend_pipeline_{nullptr};
    FrontendPipeline::OutputQueue backend_input_queue_;
    BackendPipeline::OutputQueue backend_output_queue_;


    //Data-provider pointers
    DataInterfacePipeline::UniquePtr data_interface_;
    DataProvider::Ptr data_loader_;

    //Display and Viz
    FrontendVizPipeline::UniquePtr frontend_viz_pipeline_;
    BackendVizPipeline::UniquePtr backend_viz_pipeline_;
    ImageDisplayQueue display_queue_;
    OpenCVImageDisplayQueue displayer_;


    //Threaded spinners
    Spinner::UniquePtr data_provider_spinner_;
    Spinner::UniquePtr frontend_pipeline_spinner_;
    Spinner::UniquePtr backend_pipeline_spinner_;
    Spinner::UniquePtr frontend_viz_pipeline_spinner_;
    Spinner::UniquePtr backend_viz_pipeline_spinner_;


};

//TODO: refactor!!
template<typename Measurement>
typename Optimizer<Measurement>::Ptr DynoPipelineManager::createOptimizer(OptimizerType optimizer_type) const {
    if(optimizer_type == OptimizerType::kIncremental) {
        LOG(INFO) << "Constructing incremental optimizer";
        return std::make_shared<IncrementalOptimizer<Measurement>>();
        // return std::make_shared<ISAMOptimizer<Measurement>>();
    }
    else if(optimizer_type == OptimizerType::kBatch) {
        LOG(INFO) << "Constructing batch optimizer";

        BatchOptimizerParams batch_params;
        CHECK(get_dataset_size_) << "dataset size function must be set - right now only works with RBG!!";
        batch_params.get_last_frame = get_dataset_size_;

        return std::make_shared<BatchOptimizer<Measurement>>(batch_params);
    }
}



} //dyno
