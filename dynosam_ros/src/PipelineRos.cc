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

#include "dynosam_ros/PipelineRos.hpp"
#include "dynosam_ros/FrontendDisplayRos.hpp"
#include "dynosam_ros/BackendDisplayRos.hpp"


#include <dynosam/dataprovider/DataProviderFactory.hpp>
#include <dynosam/dataprovider/DataProviderUtils.hpp>
#include <dynosam/pipeline/PipelineParams.hpp>
#include <dynosam/visualizer/OpenCVFrontendDisplay.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "rclcpp/parameter.hpp"
#include "rcl_interfaces/msg/parameter.hpp"


namespace dyno {

DynoPipelineManagerRos::DynoPipelineManagerRos(const rclcpp::NodeOptions& options) : Node("dynosam", options)
{
    RCLCPP_INFO_STREAM(this->get_logger(), "Starting DynoPipelineManagerRos");

    const std::string params_path = getParamsPath();
    const std::string dataset_path = getDatasetPath();

    RCLCPP_INFO_STREAM(this->get_logger(), "Loading Dyno VIO params from: " << params_path);
    RCLCPP_INFO_STREAM(this->get_logger(), "Loading dataset from: " << dataset_path);

    dyno::DynoParams params(params_path);

    dyno::DataProvider::UniquePtr data_loader = dyno::DataProviderFactory::Create(dataset_path, params_path, static_cast<dyno::DatasetType>(params.data_provider_type_));
     RCLCPP_INFO_STREAM(this->get_logger(), "Constructed data loader");

    auto frontend_display = std::make_shared<dyno::FrontendDisplayRos>(this->create_sub_node("frontend_viz"));
    auto backend_display = std::make_shared<dyno::BackendDisplayRos>(this->create_sub_node("backend_viz"));


    pipeline_ = std::make_unique<DynoPipelineManager>(params, std::move(data_loader), frontend_display, backend_display);
}


std::string DynoPipelineManagerRos::getParamsPath() {
     return searchForPathWithParams("params_folder_path", "dynosam/params",
        "Path to the folder containing the yaml files with the DynoVIO parameters.");
}

std::string DynoPipelineManagerRos::getDatasetPath() {
    return searchForPathWithParams("dataset_path", "dataset", "Path to the dataset.");
}

std::string DynoPipelineManagerRos::searchForPathWithParams(const std::string& param_name, const std::string& default_path, const std::string& description) {
    auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};
    param_desc.description = description;

    this->declare_parameter(param_name, default_path, param_desc);

    std::string path;
    if(this->get_parameter(param_name, path)) {
        return path;
    }
    else {
        throw std::runtime_error("ROS param `" + param_name + "` expected but not found");
    }

    throwExceptionIfPathInvalid(path);
    return path;
}

}
