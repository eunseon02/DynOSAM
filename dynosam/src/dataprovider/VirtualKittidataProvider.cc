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

#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"
#include "dynosam/dataprovider/DataProviderUtils.hpp"

#include <glog/logging.h>

namespace dyno {

class GenericVirtualKittiImageLoader : public dyno::DataFolder<cv::Mat> {
public:
    std::string getFolderName() const override { return ""; }
};

class VirtualKittiRGBDataFolder : public GenericVirtualKittiImageLoader {

public:
    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << idx;
        const std::string file_path = (std::string)getAbsolutePath()  + "/rgb_" + ss.str() + ".jpg";

        return cv::Mat()
    }
};

class VirtualKittiForwardFlowDataFolder : public GenericVirtualKittiImageLoader  {

public:
    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << idx;
        const std::string file_path = (std::string)getAbsolutePath()  + "/flow_" + ss.str() + ".png";
        return cv::Mat()
    }
};

class VirtualKittiDepthDataFolder : public GenericVirtualKittiImageLoader  {

public:
    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << idx;
        const std::string file_path = (std::string)getAbsolutePath()  + "/depth_" + ss.str() + ".jpg";
        return cv::Mat()
    }
};


class VirtualKittiMotionSegFolder : public GenericVirtualKittiImageLoader  {

public:
    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << idx;
        const std::string file_path = (std::string)getAbsolutePath()  + "/instancegt_" + ss.str() + ".png";
        return cv::Mat()
    }
};

class VirtualKittiTextGtLaoder {

public:
    DYNO_POINTER_TYPEDEFS(VirtualKittiTextGtLaoder)

    VirtualKittiTextGtLaoder(const std::string& file_path)
    :  bbox_path(file_path + "/bbox.txt"),
       colors_path(file_path + "/colors.txt"),
       extrinsic_path(file_path + "/extrinsic.txt"),
       info_path(file_path + "/info.txt"),
       intrinsic_path(file_path + "/intrinsic.txt"),
       pose_path(file_path + "/pose.txt")  {

        throwExceptionIfPathInvalid(bbox_path);
        throwExceptionIfPathInvalid(colors_path);
        throwExceptionIfPathInvalid(extrinsic_path);
        throwExceptionIfPathInvalid(info_path);
        throwExceptionIfPathInvalid(intrinsic_path);
        throwExceptionIfPathInvalid(pose_path);

    }

private:
    std::vector<ObjectPoseGT> object_poses_; //! all object poses to be sorted later

private:
    const std::string bbox_path;
    const std::string colors_path;
    const std::string extrinsic_path;
    const std::string info_path;
    const std::string intrinsic_path;
    const std::string pose_path;

    std::vector<gtsam::Pose3> camera_poses; //! camera poses in the world frame


}

class VirtualKittiMotionSegFolder : public dyno::DataFolder<cv::Mat>  {

public:
    VirtualKittiMotionSegFolder(VirtualKittiTextGtLaoder::Ptr gt_loader): gt_loader_(CHECK_NOTNULL(gt_loader)) {}

    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << idx;
        const std::string file_path = (std::string)getAbsolutePath()  + "/instancegt_" + ss.str() + ".png";
        return cv::Mat()
    }

    VirtualKittiTextGtLaoder::Ptr gt_loader_;

};


//everything current assumes camera0!! This includes loading things like extrinsics...
VirtualKittiDataLoader::VirtualKittiDataLoader(const fs::path& dataset_path,  const std::string& scene, const std::string& scene_type) {

    const std::string path = dataset_path;

    LOG(INFO) << "Starting VirtualKittiDataLoader with path " << path << " requested scene " << scene << " and scene type " << scene_type;
    const std::string depth_folder = path + "/" + v_depth_folder + "/" + scene + "/" + scene_type + "/frames/depth/Camera_0";
    const std::string forward_flow_folder = path + "/" + v_forward_flow_folder + "/" + scene + "/" + scene_type + "/frames/forwardFlow/Camera_0";
    const std::string forward_scene_flow_folder = path + "/" + v_forward_scene_flow_folder + "/" + scene + "/" + scene_type + "/frames/forwardsceneFlow/Camera_0";
    const std::string instance_segmentation_folder = path + "/" + v_instance_segmentation_folder + "/" + scene + "/" + scene_type + "/frames/instanceSegmentation/Camera_0";
    const std::string rgb_folder = path + "/" + v_rgb_folder + "/" + scene + "/" + scene_type + "/frames/rgb/Camera_0";
    const std::string text_gt_folder = path + "/" + v_text_gt_folder + "/" + scene + "/" + scene_type;

    throwExceptionIfPathInvalid(depth_folder);
    throwExceptionIfPathInvalid(forward_flow_folder);
    throwExceptionIfPathInvalid(forward_scene_flow_folder);
    throwExceptionIfPathInvalid(instance_segmentation_folder);
    throwExceptionIfPathInvalid(rgb_folder);
    throwExceptionIfPathInvalid(text_gt_folder);

}

bool VirtualKittiDataLoader::spin() {
    return false;
}

} //dyno
