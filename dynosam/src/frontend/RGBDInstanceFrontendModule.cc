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

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {


RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : FrontendModule(frontend_params, display_queue),
      camera_(camera),
      motion_solver_(frontend_params, camera->getParams())
{
    CHECK_NOTNULL(camera_);
    tracker_ = std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);
}

bool RGBDInstanceFrontendModule::validateImageContainer(const ImageContainer::Ptr& image_container) const {
    return image_container->hasDepth();
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::boostrapSpin(FrontendInputPacketBase::ConstPtr input) {
    ImageContainer::Ptr image_container = input->image_container_;

    //if we only have instance semgentation (not motion) then we need to make a motion mask out of the semantic mask
    //we cannot do this for the first frame so we will just treat the semantic mask and the motion mask
    //and then subsequently elimate non-moving objects later on
    TrackingInputImages tracking_images;
    if(image_container->hasSemanticMask()) {
        CHECK(!image_container->hasMotionMask());

        auto intermediate_tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::SemanticMask>();
        tracking_images = TrackingInputImages(
            intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
            intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
            ImageWrapper<ImageType::MotionMask>(
                intermediate_tracking_images.get<ImageType::SemanticMask>()
            )
        );
    }
    else {
        tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
    }


    size_t n_optical_flow, n_new_tracks;
    Frame::Ptr frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), tracking_images, n_optical_flow, n_new_tracks);

    auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();
    frame->updateDepths(image_container->getImageWrapper<ImageType::Depth>(), base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

    LOG(INFO) << "In RGBD instance module frontend boostrap";

    return {State::Nominal, nullptr};
}


FrontendModule::SpinReturn RGBDInstanceFrontendModule::nominalSpin(FrontendInputPacketBase::ConstPtr input) {
    ImageContainer::Ptr image_container = input->image_container_;
    LOG(INFO) << "In RGBD instance module frontend nominal";

    //if we only have instance semgentation (not motion) then we need to make a motion mask out of the semantic mask
    //we cannot do this for the first frame so we will just treat the semantic mask and the motion mask
    //and then subsequently elimate non-moving objects later on
    TrackingInputImages tracking_images;
    if(image_container->hasSemanticMask()) {
        CHECK(!image_container->hasMotionMask());

        auto intermediate_tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::SemanticMask>();
        tracking_images = TrackingInputImages(
            intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
            intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
            ImageWrapper<ImageType::MotionMask>(
                intermediate_tracking_images.get<ImageType::SemanticMask>()
            )
        );
    }
    else {
        tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
    }

    size_t n_optical_flow, n_new_tracks;
    LOG(INFO) << "Beginning tracking on frame " << input->getFrameId();
    Frame::Ptr frame = nullptr;
    {
        utils::TimingStatsCollector tracking_timer("tracking_timer");
        frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), tracking_images, n_optical_flow, n_new_tracks);

    }
    CHECK(frame);

    Frame::Ptr previous_frame = tracker_->getPreviousFrame();
    CHECK(previous_frame);

    LOG(INFO) << "Done tracking on frame " << input->getFrameId();

    auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();

    {
        utils::TimingStatsCollector update_depths_timer("depth_updater");
        frame->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

    }

    AbsolutePoseCorrespondences correspondences;
    //this does not create proper bearing vectors (at leas tnot for 3d-2d pnp solve)
    //bearing vectors are also not undistorted atm!!
    //TODO: change to use landmarkWorldProjectedBearingCorrespondance and then change motion solver to take already projected bearing vectors
    {
        utils::TimingStatsCollector track_dynamic_timer("frame_correspondences");
        frame->getCorrespondences(correspondences, *previous_frame, KeyPointType::STATIC, frame->landmarkWorldKeypointCorrespondance());
    }

    LOG(INFO) << "Done correspondences";

    MotionResult camera_pose_result;
    {
        utils::TimingStatsCollector track_dynamic_timer("solve_camera_pose");
        camera_pose_result = motion_solver_.solveCameraPose(correspondences);
    }

    // LOG(INFO) << "Done correspondences";

    if(camera_pose_result.valid()) {
        frame->T_world_camera_ = camera_pose_result.get();
    }
    else {
        LOG(ERROR) << "Unable to solve camera pose for frame " << frame->frame_id_;
        frame->T_world_camera_ = gtsam::Pose3::Identity();

    }
    // LOG(INFO) << "Solved camera pose";

    TrackletIds tracklets = frame->static_features_.collectTracklets();
    CHECK_GE(tracklets.size(), correspondences.size()); //tracklets shoudl be more (or same as) correspondances as there will be new points untracked

    frame->static_features_.markOutliers(camera_pose_result.outliers_); //do we need to mark innliers? Should start as inliers

    {
        utils::TimingStatsCollector track_dynamic_timer("tracking_dynamic");
        vision_tools::trackDynamic(base_params_,*previous_frame, frame);
    }

    MotionEstimateMap motion_estimates;
    std::map<ObjectId, gtsam::Pose3> per_frame_object_poses; //for viz
    for(const auto& [object_id, observations] : frame->object_observations_) {
        utils::TimingStatsCollector object_motion_timer("solve_object_motion");
        AbsolutePoseCorrespondences dynamic_correspondences;

        // auto func = [&](const Frame& previous_frame, const Feature::Ptr& previous_feature, const Feature::Ptr& current_feature) {
        //     const Frame& current_frame = *frame;

        //     const Landmark&

        //     return TrackletCorrespondance<Landmark, Landmark>()
        // }

        // GenericCorrespondences<Landmark, Landmark> dynamic_correspondences;
        //   bool result = frame->getDynamicCorrespondences(
        //     dynamic_correspondences,
        //     *previous_frame_,
        //     object_id,
        //     [&]());

        // //get the corresponding feature pairs
        bool result = frame->getDynamicCorrespondences(
            dynamic_correspondences,
            *previous_frame,
            object_id,
            frame->landmarkWorldKeypointCorrespondance());

        if(!result) {
            continue;
        }


        LOG(INFO) << "Solving motion with " << dynamic_correspondences.size() << " correspondences for object instance " << object_id;

        const MotionResult object_motion_result = motion_solver_.solveObjectMotion(
            dynamic_correspondences,
            frame->T_world_camera_);

        if(object_motion_result.valid()) {
            frame->dynamic_features_.markOutliers(object_motion_result.outliers_);

            ReferenceFrameEstimate<Motion3> estimate(object_motion_result, ReferenceFrame::WORLD);
            motion_estimates.insert({object_id, estimate});

            if(!input->optional_gt_.has_value()) {
                LOG(WARNING) << "Cannot update object pose because no gt!";
                continue;
            }

            const GroundTruthInputPacket gt_packet = input->optional_gt_.value();
            CHECK_EQ(gt_packet.frame_id_, frame->frame_id_);
            logAndPropogateObjectPoses(per_frame_object_poses, gt_packet, object_motion_result.get(), object_id);
        }
    }

    cv::Mat tracking_img = tracker_->computeImageTracks(*previous_frame, *frame);
    if(display_queue_) display_queue_->push(ImageToDisplay("tracks", tracking_img));


    auto output = constructOutput(*frame, motion_estimates, tracking_img, per_frame_object_poses, input->optional_gt_);
    return {State::Nominal, output};
}


RGBDInstanceOutputPacket::Ptr RGBDInstanceFrontendModule::constructOutput(
    const Frame& frame,
    const MotionEstimateMap& estimated_motions,
    const cv::Mat& debug_image,
    const std::map<ObjectId, gtsam::Pose3>& propogated_object_poses,
    const GroundTruthInputPacket::Optional& gt_packet)
{
    StatusKeypointMeasurements static_keypoint_measurements;
    Landmarks static_landmarks;
    for(const Feature::Ptr& f : frame.usableStaticFeaturesBegin()) {
        const TrackletId tracklet_id = f->tracklet_id_;
        const Keypoint kp = f->keypoint_;
        const Landmark lmk = frame.backProjectToWorld(tracklet_id);
        CHECK(f->isStatic());
        CHECK(Feature::IsUsable(f));

        KeypointStatus status = KeypointStatus::Static();
        KeypointMeasurement measurement = std::make_pair(tracklet_id, kp);

        static_keypoint_measurements.push_back(std::make_pair(status, measurement));
        static_landmarks.push_back(lmk);
    }


    StatusKeypointMeasurements dynamic_keypoint_measurements;
    Landmarks dynamic_landmarks;
    for(const auto& [object_id, obs] : frame.object_observations_) {
        CHECK_EQ(object_id, obs.instance_label_);
        CHECK(obs.marked_as_moving_);

        for(const TrackletId tracklet : obs.object_features_) {
            if(frame.isFeatureUsable(tracklet)) {
                const Feature::Ptr f = frame.at(tracklet);
                CHECK(!f->isStatic());
                CHECK_EQ(f->instance_label_, object_id);

                const TrackletId tracklet_id = f->tracklet_id_;
                const Keypoint kp = f->keypoint_;
                const Landmark lmk = frame.backProjectToWorld(tracklet_id);

                KeypointStatus status = KeypointStatus::Dynamic(object_id);
                KeypointMeasurement measurement = std::make_pair(tracklet_id, kp);

                dynamic_keypoint_measurements.push_back(std::make_pair(status, measurement));
                dynamic_landmarks.push_back(lmk);
            }
        }

    }

    return std::make_shared<RGBDInstanceOutputPacket>(
        static_keypoint_measurements,
        dynamic_keypoint_measurements,
        static_landmarks,
        dynamic_landmarks,
        frame.T_world_camera_,
        frame,
        estimated_motions,
        propogated_object_poses,
        debug_image,
        gt_packet
    );
}


void RGBDInstanceFrontendModule::logAndPropogateObjectPoses(std::map<ObjectId, gtsam::Pose3>& per_frame_object_poses, const GroundTruthInputPacket& gt_packet, const gtsam::Pose3& prev_H_world_curr, ObjectId object_id) {
    auto it = std::find_if(gt_packet.object_poses_.begin(), gt_packet.object_poses_.end(),
                    [=](const ObjectPoseGT& gt_object) { return gt_object.object_id_ == object_id; });

    if(it == gt_packet.object_poses_.end()) {
        LOG(WARNING) << "Object Id " << object_id << " cannot be found in the gt packet";
        return;
    }

    const ObjectPoseGT& object_gt_packet = *it;

    if(object_poses_.find(object_id) == object_poses_.end()) {
        object_poses_[object_id] = gt_packet.X_world_ * object_gt_packet.L_camera_;
    }
    else {
        //propogate
        const gtsam::Pose3& prev_L_world = object_poses_[object_id];
        const gtsam::Pose3 curr_L_world = prev_H_world_curr * prev_L_world;
        object_poses_[object_id] = curr_L_world;
    }

    per_frame_object_poses[object_id] = object_poses_[object_id];
}

} //dyno
