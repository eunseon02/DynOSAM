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

#include <Eigen/Dense>

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

    dyno::DynoParams dyno_params("/home/user/dev_ws/src/DynOSAM/dynosam/params/");

    auto camera = std::make_shared<dyno::Camera>(dyno_params.camera_params_);

    // auto data_loader = std::make_unique<dyno::KittiDataLoader>(FLAGS_path_to_kitti, dyno::KittiDataLoader::MaskType::SEMANTIC_INSTANCE);
    // auto frontend_display = std::make_shared<dyno::OpenCVFrontendDisplay>();

    // dyno::DynoPipelineManager pipeline(params, std::move(data_loader), frontend_display);
    // while(pipeline.spin()) {};
    dyno::VirtualKittiDataLoader::Params params;
    params.scene = "Scene01";
    params.scene_type = "clone";
    params.mask_type = dyno::MaskType::MOTION;

    // dyno::KittiDataLoader::Params params;
    // params.base_line = 388.1822;
    // params.mask_type = dyno::MaskType::MOTION;

    // dyno::KittiDataLoader d("/root/data/vdo_slam/kitti/kitti/0020", params);

    cv::Mat previous_optical_flow;
    cv::Mat previous_class_segmentation;
    cv::Mat previous_motion_mask;
    gtsam::Pose3 previous_cam_pose;


    dyno::FeatureTracker::UniquePtr tracker= std::make_unique<dyno::FeatureTracker>(dyno_params.frontend_params_, camera);

    dyno::VirtualKittiDataLoader d("/root/data/virtual_kitti", params);
    d.setCallback([&](dyno::FrameId frame_id, dyno::Timestamp timestamp, cv::Mat rgb, cv::Mat optical_flow, cv::Mat depth, cv::Mat motion, cv::Mat class_semantics, dyno::GroundTruthInputPacket gt_packet) {
        LOG(INFO) << "Frame " << frame_id << " ts " << timestamp;
        using namespace dyno;
        dyno::ImageContainer::Ptr image_container = ImageContainer::Create(
                timestamp,
                frame_id,
                ImageWrapper<ImageType::RGBMono>(rgb),
                ImageWrapper<ImageType::Depth>(depth),
                ImageWrapper<ImageType::OpticalFlow>(optical_flow),
                ImageWrapper<ImageType::MotionMask>(motion),
                ImageWrapper<ImageType::ClassSegmentation>(class_semantics));

        auto tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
        size_t n_optical_flow, n_new_tracks;
        Frame::Ptr frame =  tracker->track(frame_id, timestamp, tracking_images, n_optical_flow, n_new_tracks);

        auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();
        frame->updateDepths(image_container->getImageWrapper<ImageType::Depth>(), dyno_params.frontend_params_.depth_background_thresh, dyno_params.frontend_params_.depth_obj_thresh);

        Frame::Ptr previous_frame = tracker->getPreviousFrame();
        const gtsam::Matrix3 K_inv = camera->getParams().getCameraMatrixEigen().inverse();


        if(previous_frame) {
            cv::Mat tracking_img = tracker->computeImageTracks(*previous_frame, *frame);

            auto feature_points_on_ground_itr = FeatureFilterIterator(
                const_cast<FeatureContainer&>(frame->static_features_),
                [&class_semantics](const Feature::Ptr& f) -> bool {
                    bool feature_is_road =
                        functional_keypoint::at<int>(f->keypoint_, class_semantics) == ImageType::ClassSegmentation::Labels::Road;

                    return Feature::IsUsable(f) && feature_is_road;
                }
            );

            Landmarks static_ground_points;
            Keypoints kps;
            for(Feature::Ptr f_on_road : feature_points_on_ground_itr) {
                Landmark lmk;
                CHECK(f_on_road->hasDepth());
                camera->backProject(f_on_road->keypoint_, f_on_road->depth_, &lmk, gt_packet.X_world_);
                static_ground_points.push_back(lmk);
                kps.push_back(f_on_road->keypoint_);

            }

            //size of points on ground
            size_t m = static_ground_points.size();
            gtsam::Matrix A = gtsam::Matrix::Zero(m, 3);
            gtsam::Matrix b = gtsam::Matrix::Zero(m, 1);

            for(size_t i = 0; i < m; i++) {
                const Landmark& lmk = static_ground_points.at(i);
                A(i, 0) = lmk(0);
                A(i, 1) = lmk(2);
                A(i, 2) = 1.0;

                b(i, 0) = lmk(1);
            }

            //plane coeffs -> Ax + Bz + C = y, because in camera convention y axis is normal to the ground usually
            gtsam::Matrix plane_coeffs = A.colPivHouseholderQr().solve(b);
            LOG(INFO) << "X= " << plane_coeffs;

            const auto object_labels = vision_tools::getObjectLabels(motion);
            for(ObjectId object_id : object_labels) {
                gtsam::Point2Vector observation_prev;
                gtsam::Point2Vector observation_curr;
                gtsam::Point3Vector triangulated_points;
                //Z coordinates of triangulated poitns in the camera frame
                Depths z_camera;
                std::optional<double> depth_opt = mono_backend_tools::estimateDepthFromRoad(
                    previous_cam_pose,
                    gt_packet.X_world_,
                    camera,
                    previous_motion_mask,
                    motion,
                    previous_class_segmentation,
                    class_semantics,
                    previous_optical_flow,
                    object_id,
                    observation_prev,
                    observation_curr,
                    triangulated_points,
                    z_camera
                );

                for(const Keypoint& kp : observation_curr) {
                    gtsam::Point3 kp_hom(kp(0), kp(1), 1);
                    gtsam::Point3 ray_cam = K_inv * kp_hom;
                    gtsam::Point3 ray_world = gt_packet.X_world_.rotation() * ray_cam;
                    ray_world.normalize();

                    const double A = plane_coeffs(0);
                    const double B = plane_coeffs(1);
                    const double C = plane_coeffs(2);

                    const gtsam::Point3 camera_center = gt_packet.X_world_.translation();
                    //depth of the kp
                    const double lambda = (-C - A* camera_center(0) + camera_center(1) - B*camera_center(2))/(A * ray_world(0) - ray_world(1) + B * ray_world(2));

                    const double expected_depth = functional_keypoint::at<Depth>(kp, image_container->getDepth());
                    LOG(INFO) << lambda << " " <<expected_depth;

                }
            }


            // for(size_t i = 0; i < m; i++) {
            //     const Landmark& lmk = static_ground_points.at(i);
            //     const Keypoint& kp = kps.at(i);

            //     gtsam::Point3 kp_hom(kp(0), kp(1), 1);
            //     gtsam::Point3 ray_cam = K_inv * kp_hom;
            //     gtsam::Point3 ray_world = gt_packet.X_world_.rotation() * ray_cam;
            //     ray_world.normalize();

            //     const double A = plane_coeffs(0);
            //     const double B = plane_coeffs(1);
            //     const double C = plane_coeffs(2);

            //     const gtsam::Point3 camera_center = gt_packet.X_world_.translation();

            //     //depth of the kp
            //     const double lambda = (-C - A* camera_center(0) + camera_center(1) - B*camera_center(2))/(A * ray_world(0) - ray_world(1) + B * ray_world(2));
            //     // const double execpted_lambda = gtsam::Point3 (lmk - gt_packet.X_world_.translation()).norm();
            //     // LOG(INFO) << lambda << " " <<execpted_lambda;



            // }

            cv::imshow("Tracks", tracking_img);

        }


        cv::Mat flow_viz;
        dyno::utils::flowToRgb(optical_flow, flow_viz);

        cv::Mat mask_viz;
        dyno::utils::semanticMaskToRgb(rgb, motion, mask_viz);


        cv::Mat road_viz = dyno::ImageType::ClassSegmentation::toRGB(
            class_semantics
        );

        cv::imshow("RGB", rgb);
        cv::imshow("OF", flow_viz);
        cv::imshow("Motion", mask_viz);
        cv::imshow("Road semantics", road_viz);

        previous_optical_flow = optical_flow;
        previous_class_segmentation = class_semantics;
        previous_motion_mask = motion;
        previous_cam_pose = gt_packet.X_world_;

        cv::waitKey(1);

        return true;
    });

    while(d.spin()) {}




}
