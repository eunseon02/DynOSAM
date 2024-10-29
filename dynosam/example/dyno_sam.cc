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

#include "dynosam/backend/MonoBackendTools.hpp"

#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/common/Camera.hpp"
#include "dynosam/common/ImageContainer.hpp"

#include "dynosam/visualizer/ColourMap.hpp"

#include <Eigen/Dense>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <png++/png.hpp>


DEFINE_string(path_to_kitti, "/root/data/kitti", "Path to KITTI dataset");
//TODO: (jesse) many better ways to do this with ros - just for now
DEFINE_string(params_folder_path, "dynosam/params", "Path to the folder containing the yaml files with the VIO parameters.");

// int main(int argc, char* argv[]) {

//     // using namespace dyno;

//     // const std::string path = "/root/data/vdo_slam/kitti/kitti-step/panoptic_maps/train/0000/000000.png";

//     // png::image<png::rgb_pixel> index_image(path);

//     // cv::Size size(index_image.get_width(), index_image.get_height());
//     // cv::Mat class_segmentation(size, ImageType::ClassSegmentation::OpenCVType);

//     // for (size_t y = 0; y < index_image.get_height(); ++y)
//     // {
//     //     for (size_t x = 0; x < index_image.get_width(); ++x)
//     //     {
//     //         const auto& pixel = index_image.get_pixel(x, y);
//     //         const int red = (int)pixel.red;
//     //         const int green = (int)pixel.green;
//     //         const int blue = (int)pixel.blue;

//     //         if(red == 0) {
//     //             class_segmentation.at<int>(y, x) = ImageType::ClassSegmentation::Labels::Road;
//     //         }
//     //         else if(red == 12) {
//     //             class_segmentation.at<int>(y, x) = ImageType::ClassSegmentation::Labels::Rider;
//     //         }


//     //         // semantic_image.at<cv::Vec3b>(y, x)[0] = blue;
//     //         // semantic_image.at<cv::Vec3b>(y, x)[1] = green;
//     //         // semantic_image.at<cv::Vec3b>(y, x)[2] = red;
//     //     }
//     // }

//     // cv::Mat road_viz = dyno::ImageType::ClassSegmentation::toRGB(
//     //         class_segmentation
//     //     );

//     // cv::imshow("Seg", road_viz);
//     // cv::waitKey(0);

//     google::ParseCommandLineFlags(&argc, &argv, true);
//     google::InitGoogleLogging(argv[0]);
//     FLAGS_logtostderr = 1;
//     FLAGS_colorlogtostderr = 1;
//     FLAGS_log_prefix = 1;

//     dyno::DynoParams dyno_params("/home/user/dev_ws/src/DynOSAM/dynosam/params/");

//     auto camera = std::make_shared<dyno::Camera>(dyno_params.camera_params_);

//     // auto data_loader = std::make_unique<dyno::KittiDataLoader>(FLAGS_path_to_kitti, dyno::KittiDataLoader::MaskType::SEMANTIC_INSTANCE);
//     // auto frontend_display = std::make_shared<dyno::OpenCVFrontendDisplay>();

//     // dyno::DynoPipelineManager pipeline(params, std::move(data_loader), frontend_display);
//     // while(pipeline.spin()) {};
//     dyno::VirtualKittiDataLoader::Params params;
//     params.scene = "Scene01";
//     params.scene_type = "clone";
//     params.mask_type = dyno::MaskType::MOTION;

//     // dyno::KittiDataLoader::Params params;
//     // params.base_line = 388.1822;
//     // params.mask_type = dyno::MaskType::MOTION;

//     // dyno::KittiDataLoader d("/root/data/vdo_slam/kitti/kitti/0020", params);

//     // cv::Mat previous_optical_flow;
//     // cv::Mat previous_class_segmentation;
//     // cv::Mat previous_motion_mask;
//     // gtsam::Pose3 previous_cam_pose;


//     dyno::FeatureTracker::UniquePtr tracker= std::make_unique<dyno::FeatureTracker>(dyno_params.frontend_params_, camera);

//     dyno::VirtualKittiDataLoader d("/root/data/virtual_kitti", params);
//     d.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp, cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion, cv::Mat class_semantics, dyno::GroundTruthInputPacket gt_packet) {
//         LOG(INFO) << "Frame " << frame_id << " ts " << timestamp;
//         using namespace dyno;
//         dyno::ImageContainer::Ptr image_container = ImageContainer::Create(
//                 timestamp,
//                 frame_id,
//                 ImageWrapper<ImageType::RGBMono>(rgb),
//                 ImageWrapper<ImageType::Depth>(depth),
//                 ImageWrapper<ImageType::OpticalFlow>(optical_flow),
//                 ImageWrapper<ImageType::MotionMask>(motion),
//                 ImageWrapper<ImageType::ClassSegmentation>(class_semantics));


//         // cv::Mat flow_viz;
//         // dyno::utils::flowToRgb(optical_flow, flow_viz);

//         // cv::Mat mask_viz;
//         // dyno::utils::semanticMaskToRgb(rgb, motion, mask_viz);


//         // cv::Mat road_viz = dyno::ImageType::ClassSegmentation::toRGB(
//         //     class_semantics
//         // );

//         cv::imshow("RGB", rgb);
//         cv::imshow("OF", ImageType::OpticalFlow::toRGB(optical_flow));
//         cv::imshow("Motion", ImageType::MotionMask::toRGB(motion));
//         cv::imshow("Depth", ImageType::Depth::toRGB(depth));


//         cv::waitKey(1);

//         return true;
//     });

//     while(d.spin()) {}




// }


#include "dynosam/dataprovider/ClusterSlamDataProvider.hpp"
#include "dynosam/dataprovider/OMDDataProvider.hpp"

// int main(int argc, char* argv[]) {

//     using namespace dyno;
//     google::ParseCommandLineFlags(&argc, &argv, true);
//     google::InitGoogleLogging(argv[0]);
//     FLAGS_logtostderr = 1;
//     FLAGS_colorlogtostderr = 1;
//     FLAGS_log_prefix = 1;

//     ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-L2");
//     // OMDDataLoader loader("/root/data/omm/swinging_4_unconstrained");
//     auto camera = std::make_shared<Camera>(*loader.getCameraParams());
//     auto tracker = std::make_shared<FeatureTracker>(FrontendParams(), camera);

//     loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp, cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion, dyno::GroundTruthInputPacket gt_packet) -> bool {

//         LOG(INFO) << frame_id << " " << timestamp;

//         cv::imshow("RGB", rgb);
//         cv::imshow("OF", ImageType::OpticalFlow::toRGB(optical_flow));
//         cv::imshow("Motion", ImageType::MotionMask::toRGB(motion));
//         cv::imshow("Depth", ImageType::Depth::toRGB(depth));

//         ImageContainer::Ptr container = ImageContainer::Create(
//                     timestamp,
//                     frame_id,
//                     ImageWrapper<ImageType::RGBMono>(rgb),
//                     ImageWrapper<ImageType::Depth>(depth),
//                     ImageWrapper<ImageType::OpticalFlow>(optical_flow),
//                     ImageWrapper<ImageType::MotionMask>(motion));

//         // auto frame = tracker->track(frame_id, timestamp, *container);
//         // Frame::Ptr previous_frame = tracker->getPreviousFrame();
//         // if(previous_frame) {
//         //     cv::imshow("Tracking", tracker->computeImageTracks(*previous_frame, *frame));

//         // }



//         cv::waitKey(0);
//         return true;
//     });

//     while(loader.spin()) {}


// }

#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"

int main(int argc, char* argv[]) {

    using namespace dyno;
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;
    FLAGS_log_prefix = 1;

    KittiDataLoader::Params params;
    // KittiDataLoader loader("/root/data/vdo_slam/kitti/kitti/0000/", params);
    // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-L1");
    OMDDataLoader loader("/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/");

    auto camera = std::make_shared<Camera>(*loader.getCameraParams());
    auto tracker = std::make_shared<FeatureTracker>(FrontendParams(), camera);

    loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp, cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion, GroundTruthInputPacket) -> bool {
    // loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp, cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion, gtsam::Pose3, GroundTruthInputPacket) -> bool {

        LOG(INFO) << frame_id << " " << timestamp;

        cv::Mat of_viz, motion_viz, depth_viz;
        of_viz = ImageType::OpticalFlow::toRGB(optical_flow);
        motion_viz = ImageType::MotionMask::toRGB(motion);
        depth_viz = ImageType::Depth::toRGB(depth);

         ImageContainer::Ptr container = ImageContainer::Create(
                    timestamp,
                    frame_id,
                    ImageWrapper<ImageType::RGBMono>(rgb),
                    ImageWrapper<ImageType::Depth>(depth),
                    ImageWrapper<ImageType::OpticalFlow>(optical_flow),
                    ImageWrapper<ImageType::MotionMask>(motion));


        // cv::Mat boarder_mask;
        // vision_tools::computeObjectMaskBoundaryMask(
        //     motion,
        //     boarder_mask,
        //     8
        // );

        // cv::Scalar red = dyno::Color::red();

        // const ObjectIds instance_labels = vision_tools::getObjectLabels(motion);
        // for(const auto object_id : instance_labels) {
        //     std::vector<std::vector<cv::Point>> detected_contours;
        //     vision_tools::findObjectBoundingBox(motion, object_id,detected_contours);

        //     cv::drawContours(boarder_mask, detected_contours, -1, red, 8);
        // }

        // cv::imshow("Mask with boarder", boarder_mask);

        // cv::imshow("RGB", rgb);
        // cv::imshow("OF", of_viz);
        cv::imshow("Motion", motion_viz);
        // cv::waitKey(1);
        // cv::imshow("Depth", depth_viz);

        auto frame = tracker->track(frame_id, timestamp, *container);
        Frame::Ptr previous_frame = tracker->getPreviousFrame();

        // // motion_viz = ImageType::MotionMask::toRGB(frame->image_container_.get<ImageType::MotionMask>());
        // // // cv::imshow("Motion", motion_viz);

        cv::Mat tracking;
        if(previous_frame) {
            tracking = tracker->computeImageTracks(*previous_frame, *frame, false);

        }
        if(!tracking.empty())
            cv::imshow("Tracking", tracking);



        LOG(INFO) << to_string(tracker->getTrackerInfo());
        const std::string path = "/root/results/misc/";
        if ((char)cv::waitKey(0) == 's') {
            LOG(INFO) << "Saving...";
            // cv::imwrite(path + "omd_su4_rgb.png", rgb);
            // cv::imwrite(path + "omd_su4_of.png", of_viz);
            // cv::imwrite(path + "omd_su4_motion.png", motion_viz);
            // cv::imwrite(path + "omd_su4_depth.png", depth_viz);
            cv::imwrite(path + "omd_tracking.png", tracking);




        }
        cv::waitKey(1);



        return true;
    });

    while(loader.spin()) {}


}


// #include "dynosam/dataprovider/ProjectAriaDataProvider.hpp"
// #include "dynosam/frontend/vision/VisionTools.hpp"

// int main(int argc, char* argv[]) {

//     using namespace dyno;
//     google::ParseCommandLineFlags(&argc, &argv, true);
//     google::InitGoogleLogging(argv[0]);
//     FLAGS_logtostderr = 1;
//     FLAGS_colorlogtostderr = 1;
//     FLAGS_log_prefix = 1;

//     // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-S1");
//     ProjectARIADataLoader loader("/root/data/zed/acfr_3_moving_medium/");

//     loader.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp, cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion) -> bool {

//         LOG(INFO) << frame_id << " " << timestamp;

//         cv::imshow("RGB", rgb);
//         cv::imshow("OF", ImageType::OpticalFlow::toRGB(optical_flow));
//         cv::imshow("Motion", ImageType::MotionMask::toRGB(motion));
//         cv::imshow("Depth", ImageType::Depth::toRGB(depth));

//         cv::Mat shrunk_mask;
//         vision_tools::shrinkMask(motion, shrunk_mask, 20);
//         cv::imshow("Shrunk Motion", ImageType::MotionMask::toRGB(shrunk_mask));

//         cv::waitKey(1);
//         return true;
//     });

//     while(loader.spin()) {}


// }
