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


#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executor.hpp"

#include "dynosam_ros/PipelineRos.hpp"
#include <glog/logging.h>



int main(int argc, char* argv[]) {

    auto args = rclcpp::init_and_remove_ros_arguments(argc, argv);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;
    FLAGS_log_prefix = 1;

    rclcpp::NodeOptions options;
    options.arguments(args);

    rclcpp::executors::SingleThreadedExecutor exec;
    auto ros_pipeline = std::make_shared<dyno::DynoPipelineManagerRos>();

    exec.add_node(ros_pipeline);
    while(rclcpp::ok()) {
        if(!ros_pipeline->spinOnce()) {
            break;
        }
        exec.spin_some();
    }

    ros_pipeline.reset();


}
