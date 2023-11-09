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

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"

#include "dynosam/frontend/FrontendInputPacket.hpp"

namespace dyno {


//depth, motion mask and gt
using VirtualKittiDatasetProvider = DynoDatasetProvider<cv::Mat, cv::Mat, GroundTruthInputPacket>;

class VirtualKittiDataLoader : public VirtualKittiDatasetProvider {

public:

    //expect to be the top level where the folders undearneath are in the form vkitti_2.0.3_depth... (or as in the download...)
    VirtualKittiDataLoader(const fs::path& dataset_path, const std::string& scene, const std::string& scene_type);
    // bool spin() override;

private:

    const std::string v_depth_folder = "vkitti_2.0.3_depth";
    const std::string v_forward_flow_folder = "vkitti_2.0.3_forwardFlow";
    const std::string v_forward_scene_flow_folder = "vkitti_2.0.3_forwardSceneFlow";
    const std::string v_instance_segmentation_folder = "vkitti_2.0.3_instanceSegmentation";
    const std::string v_rgb_folder = "vkitti_2.0.3_rgb";
    const std::string v_text_gt_folder = "vkitti_2.0.3_textgt";
};

} //dyno
