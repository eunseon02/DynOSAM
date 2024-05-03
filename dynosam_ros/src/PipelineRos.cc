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

DynoNode::DynoNode(const std::string& node_name, const rclcpp::NodeOptions& options) : Node(node_name, options)
{
    RCLCPP_INFO_STREAM(this->get_logger(), "Starting DynoNode");
    auto params_path = getParamsPath();
    RCLCPP_INFO_STREAM(this->get_logger(), "Loading Dyno VO params from: " << params_path);
    dyno_params_ = std::make_unique<DynoParams>(params_path);
}

dyno::DataProvider::Ptr DynoNode::createDataProvider() {
    auto params_path = getParamsPath();
    auto dataset_path = getDatasetPath();
    auto dyno_params = getDynoParams();

    RCLCPP_INFO_STREAM(this->get_logger(), "Loading dataset from: " << dataset_path);

    dyno::DataProvider::Ptr data_loader = dyno::DataProviderFactory::Create(dataset_path, params_path, static_cast<dyno::DatasetType>(dyno_params.data_provider_type_));
    RCLCPP_INFO_STREAM(this->get_logger(), "Constructed data loader");
    return data_loader;
}



std::string DynoNode::searchForPathWithParams(const std::string& param_name, const std::string& default_path, const std::string& description) {
    // check if we've alrady declared this param
    if (!this->has_parameter(param_name)) {
        auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};
        param_desc.description = description;

        this->declare_parameter(param_name, default_path, param_desc);
    }

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


DynoPipelineManagerRos::DynoPipelineManagerRos(const rclcpp::NodeOptions& options) : DynoNode("dynosam", options)
{
    RCLCPP_INFO_STREAM(this->get_logger(), "Starting DynoPipelineManagerRos");

    auto params = getDynoParams();

    DisplayParams display_params{};
    auto frontend_display = std::make_shared<dyno::FrontendDisplayRos>(display_params, this->create_sub_node("frontend_viz"));
    auto backend_display = std::make_shared<dyno::BackendDisplayRos>(display_params, this->create_sub_node("backend_viz"));

    auto data_loader = createDataProvider();
    pipeline_ = std::make_unique<DynoPipelineManager>(params, data_loader, frontend_display, backend_display);
}


}
