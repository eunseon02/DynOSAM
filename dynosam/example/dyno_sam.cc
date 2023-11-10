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
#include "dynosam/visualizer/OpenCVFrontendDisplay.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"

#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"

#include <glog/logging.h>
#include <gflags/gflags.h>


DEFINE_string(path_to_kitti, "/root/data/kitti", "Path to KITTI dataset");
//TODO: (jesse) many better ways to do this with ros - just for now
DEFINE_string(params_folder_path, "dynosam/params", "Path to the folder containing the yaml files with the VIO parameters.");

int main(int argc, char* argv[]) {

    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;
    FLAGS_log_prefix = 1;

    // dyno::DynoParams params(FLAGS_params_folder_path);

    // auto data_loader = std::make_unique<dyno::KittiDataLoader>(FLAGS_path_to_kitti, dyno::KittiDataLoader::MaskType::SEMANTIC_INSTANCE);
    // auto frontend_display = std::make_shared<dyno::OpenCVFrontendDisplay>();

    // dyno::DynoPipelineManager pipeline(params, std::move(data_loader), frontend_display);
    // while(pipeline.spin()) {};

    dyno::VirtualKittiDataLoader d("/root/data/virtual_kitti", "Scene01", "clone");
    d.setCallback([](dyno::FrameId frame, dyno::Timestamp timestamp, cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion, dyno::GroundTruthInputPacket gt_packet) {
        LOG(INFO) << "Frame " << frame << " ts " << timestamp;

        cv::Mat flow_viz;
        dyno::utils::flowToRgb(optical_flow, flow_viz);

        cv::Mat mask_viz;
        // dyno::utils::semanticMaskToRgb(rgb, motion, mask_viz);

        cv::imshow("RGB", rgb);
        cv::imshow("OF", flow_viz);
        // cv::imshow("Motion", mask_viz);

        cv::waitKey(1);
        return true;
    });

    while(d.spin()) {}




}
