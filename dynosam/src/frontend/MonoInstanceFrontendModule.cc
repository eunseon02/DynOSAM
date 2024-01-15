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

#include "dynosam/frontend/MonoInstanceFrontendModule.hpp"
#include "dynosam/frontend/MonoInstance-Definitions.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {


MonoInstanceFrontendModule::MonoInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : FrontendModule(frontend_params, display_queue),
      camera_(camera)
    {
    CHECK_NOTNULL(camera_);
    tracker_ = std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);
}

FrontendModule::ImageValidationResult MonoInstanceFrontendModule::validateImageContainer(const ImageContainer::Ptr& image_container) const {
    return ImageValidationResult(image_container->hasMotionMask(), "Motion mask is required");
}


FrontendModule::SpinReturn MonoInstanceFrontendModule::boostrapSpin(FrontendInputPacketBase::ConstPtr input) {
    ImageContainer::Ptr image_container = input->image_container_;

    TrackingInputImages tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
    size_t n_optical_flow, n_new_tracks;
    Frame::Ptr frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), tracking_images, n_optical_flow, n_new_tracks);

    CHECK(image_container->hasDepth()) << "Needs depth for initial testing in backend";
    auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();
    frame->updateDepths(image_container->getImageWrapper<ImageType::Depth>(), base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

    return {State::Nominal, nullptr};

}


FrontendModule::SpinReturn MonoInstanceFrontendModule::nominalSpin(FrontendInputPacketBase::ConstPtr input) {

    ImageContainer::Ptr image_container = input->image_container_;

    TrackingInputImages tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
    //TODO: if we have a motion mask we should mark all the objects "as_moving" and remove this from the dynamic tracking, we should not do both
    size_t n_optical_flow, n_new_tracks;
    Frame::Ptr frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), tracking_images, n_optical_flow, n_new_tracks);

    CHECK(image_container->hasDepth()) << "Needs depth for initial testing in backend";
    auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();
    frame->updateDepths(image_container->getImageWrapper<ImageType::Depth>(), base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

    Frame::Ptr previous_frame = tracker_->getPreviousFrame();
    CHECK(previous_frame);

    AbsolutePoseCorrespondences correspondences;
    {
        utils::TimingStatsCollector track_dynamic_timer("frame_correspondences");
        frame->getCorrespondences(correspondences, *previous_frame, KeyPointType::STATIC, frame->landmarkWorldKeypointCorrespondance());
    }

    LOG(INFO) << "Done correspondences";

    AbsoluteCameraMotionSolver camera_motion_solver(base_params_, camera_->getParams());
    AbsoluteCameraMotionSolver::MotionResult camera_pose_result;
    {
        utils::TimingStatsCollector track_dynamic_timer("solve_camera_pose");
        camera_pose_result = camera_motion_solver.solve(correspondences);
    }

    if(camera_pose_result.valid()) {
        frame->T_world_camera_ = camera_pose_result.get();
    }
    else {
        LOG(ERROR) << "Unable to solve camera pose for frame " << frame->frame_id_;
        frame->T_world_camera_ = gtsam::Pose3::Identity();
    }

    TrackletIds tracklets = frame->static_features_.collectTracklets();
    CHECK_GE(tracklets.size(), correspondences.size()); //tracklets shoudl be more (or same as) correspondances as there will be new points untracked

    frame->static_features_.markOutliers(camera_pose_result.outliers_); //do we need to mark innliers? Should start as inliers

    //removed - instead rely on motion tracking label
    // {
    //     utils::TimingStatsCollector track_dynamic_timer("tracking_dynamic");
    //     vision_tools::trackDynamic(base_params_,*previous_frame, frame);
    // }

    AbsoluteObjectMotionSolver<Landmark, Keypoint> object_motion_solver(base_params_, camera_->getParams(), frame->T_world_camera_);

    DecompositionRotationEstimates motion_estimates;
    ObjectIds poor_tracking_ids;
    LOG(INFO) << "Num obj observations " << frame->object_observations_.size();
    for(const auto& [object_id, observations] : frame->object_observations_) {
        utils::TimingStatsCollector object_motion_timer("solve_object_motion");
        AbsolutePoseCorrespondences dynamic_correspondences;

        LOG(INFO) << "Num point observations " << observations.object_features_.size() << " for object instance " << object_id;

        // //get the corresponding feature pairs
        //some weird times where this doesnt work when it should
        //eg object 4 (kitti seq 04 between frames 3-4), where clearly both frames (3 & 4) track this object
        //but no correspondences are found! Mayne they are all different tracks - why are these features not propogated??
        bool result = frame->getDynamicCorrespondences(
            dynamic_correspondences,
            *previous_frame,
            object_id,
            frame->landmarkWorldKeypointCorrespondance());

        if(!result) {
            poor_tracking_ids.push_back(object_id);
            LOG(WARNING) << "Could not get corresponding feature pairs for object instance " << object_id;
            continue;
        }



        const AbsoluteObjectMotionSolver<Landmark, Keypoint>::MotionResult object_motion_result = object_motion_solver.solve(
            dynamic_correspondences);

        if(object_motion_result.valid()) {
            LOG(INFO) << "Solved motion with " << dynamic_correspondences.size() << " correspondences for object instance " << object_id;

            frame->dynamic_features_.markOutliers(object_motion_result.outliers_);

            ReferenceFrameEstimate<EssentialDecompositionResult> estimate(
                EssentialDecompositionResult(
                    gtsam::Rot3::Identity(),
                    gtsam::Rot3::Identity(),
                    gtsam::traits<gtsam::Vector3>::Identity()), ReferenceFrame::WORLD);

            motion_estimates.insert({object_id, estimate});

            if(!input->optional_gt_.has_value()) {
                LOG(WARNING) << "Cannot update object pose because no gt!";
                continue;
            }

            const GroundTruthInputPacket gt_packet = input->optional_gt_.value();
            CHECK_EQ(gt_packet.frame_id_, frame->frame_id_);
        }
        else {
            LOG(WARNING) << "Could not solve motion object instance " << object_id;

        }
    }

    for(ObjectId object_id : poor_tracking_ids) {
        //go over all dynamic objects and remove this points (make outliers)
        // const auto& observation = frame->object_observations_.at(object_id);
        // for(TrackletId tracks : observation.tracking_label_) {
        //     frame->
        // }
        //moving t
    }

    LOG(INFO) << motion_estimates.size();

    cv::Mat tracking_img = tracker_->computeImageTracks(*previous_frame, *frame);
    if(display_queue_) display_queue_->push(ImageToDisplay("tracks", tracking_img));

    auto output = constructOutput(*frame, motion_estimates, tracking_img, input->optional_gt_);
    return {State::Nominal, output};

    // //1. camera pose estimation (2D-2D)
    // RelativePoseCorrespondences correspondences;
    // //this does not create proper bearing vectors (at leas tnot for 3d-2d pnp solve)
    // //bearing vectors are also not undistorted atm!!
    // {
    //     utils::TimingStatsCollector track_dynamic_timer("mono_frame_correspondences");
    //     frame->getCorrespondences(correspondences, *previous_frame, KeyPointType::STATIC, frame->imageKeypointCorrespondance());
    // }

    // LOG(INFO) << "Done 2D-2D correspondences";

    // RelativeCameraMotionSolver camera_motion_solver(base_params_, camera_->getParams());
    // RelativeCameraMotionSolver::MotionResult camera_pose_result;
    // {
    //     utils::TimingStatsCollector track_dynamic_timer("solve_camera_pose");
    //     camera_pose_result = camera_motion_solver.solve(correspondences);
    // }


    // if(camera_pose_result.valid()) {
    //     const gtsam::Pose3 T_world_camera = previous_frame->T_world_camera_ * camera_pose_result.get();
    //     frame->T_world_camera_ = T_world_camera;
    // }
    // else {
    //     LOG(ERROR) << "Unable to solve camera pose for frame " << frame->frame_id_;
    //     frame->T_world_camera_ = gtsam::Pose3::Identity();

    // }


    // RelativeObjectMotionSolver relative_object_motion_solver(base_params_, camera_->getParams());
    // DecompositionRotationEstimates motion_estimates;
    // // //2. recover object motion (rotation)
    // for(const auto& [object_id, observations] : frame->object_observations_) {
    //     utils::TimingStatsCollector object_motion_timer("solve_object_motion");
    //     RelativePoseCorrespondences dynamic_correspondences;

    //     //get the corresponding feature pairs
    //     bool result = frame->getDynamicCorrespondences(
    //         dynamic_correspondences,
    //         *previous_frame,
    //         object_id,
    //         frame->imageKeypointCorrespondance());

    //     if(!result) {
    //         continue;
    //     }


    //     LOG(INFO) << "Solving motion with " << dynamic_correspondences.size() << " correspondences for object instance " << object_id;

    //     const RelativeObjectMotionSolver::MotionResult object_motion_result = relative_object_motion_solver.solve(
    //         dynamic_correspondences);

    //     if(object_motion_result.valid()) {
    //         frame->dynamic_features_.markOutliers(object_motion_result.outliers_);

    //         ReferenceFrameEstimate<EssentialDecompositionResult> estimate(object_motion_result.value(), ReferenceFrame::WORLD);
    //         motion_estimates.insert({object_id, estimate});

    //         if(!input->optional_gt_.has_value()) {
    //             LOG(WARNING) << "Cannot update object pose because no gt!";
    //             continue;
    //         }

    //         // const GroundTruthInputPacket gt_packet = input->optional_gt_.value();
    //         // CHECK_EQ(gt_packet.frame_id_, frame->frame_id_);
    //         // logAndPropogateObjectPoses(per_frame_object_poses, gt_packet, object_motion_result.get(), object_id);
    //     }
    // }




    // LOG(INFO) << motion_estimates.size();

    // cv::Mat tracking_img = tracker_->computeImageTracks(*previous_frame, *frame);
    // if(display_queue_) display_queue_->push(ImageToDisplay("tracks", tracking_img));

    // auto output = constructOutput(*frame, motion_estimates, tracking_img, input->optional_gt_);
    // return {State::Nominal, output};

}


MonocularInstanceOutputPacket::Ptr MonoInstanceFrontendModule::constructOutput(
        const Frame& frame,
        const DecompositionRotationEstimates& estimated_motions,
        const cv::Mat& debug_image,
        const GroundTruthInputPacket::Optional& gt_packet)
{
    StatusKeypointMeasurements static_keypoint_measurements;
    for(const Feature::Ptr& f : frame.usableStaticFeaturesBegin()) {
        const TrackletId tracklet_id = f->tracklet_id_;
        const Keypoint kp = f->keypoint_;
        CHECK(f->isStatic());
        CHECK(Feature::IsUsable(f));

        KeypointStatus status = KeypointStatus::Static();
        KeypointMeasurement measurement = std::make_pair(tracklet_id, kp);

        static_keypoint_measurements.push_back(std::make_pair(status, measurement));
    }

    StatusKeypointMeasurements dynamic_keypoint_measurements;
    for(const auto& [object_id, obs] : frame.object_observations_) {
        CHECK_EQ(object_id, obs.instance_label_);
        // CHECK(obs.marked_as_moving_); //TODO: need to comment back in!!

        for(const TrackletId tracklet : obs.object_features_) {
            if(frame.isFeatureUsable(tracklet)) {
                const Feature::Ptr f = frame.at(tracklet);
                CHECK(!f->isStatic());
                CHECK_EQ(f->instance_label_, object_id);

                const TrackletId tracklet_id = f->tracklet_id_;
                const Keypoint kp = f->keypoint_;

                KeypointStatus status = KeypointStatus::Dynamic(object_id);
                KeypointMeasurement measurement = std::make_pair(tracklet_id, kp);

                dynamic_keypoint_measurements.push_back(std::make_pair(status, measurement));
            }
        }

    }

    return std::make_shared<MonocularInstanceOutputPacket>(
        static_keypoint_measurements,
        dynamic_keypoint_measurements,
        frame.T_world_camera_,
        frame,
        estimated_motions,
        debug_image,
        gt_packet
    );
}




} //dyno
