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
#include "dynosam/frontend/RGBDInstancePacket.hpp"
#include "dynosam/utils/SafeCast.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {


RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : FrontendModule(frontend_params, display_queue),
      camera_(camera),
      rgbd_processor_(frontend_params, camera),
      motion_solver_(frontend_params, camera->getParams())
{
    CHECK_NOTNULL(camera_);
    tracker_ = std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);
}

bool RGBDInstanceFrontendModule::validateImageContainer(const ImageContainer::Ptr& image_container) const {
    return image_container->hasDepth() && image_container->hasSemanticMask();
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
    vision_tools::disparityToDepth(base_params_, depth_image_wrapper, depth_image_wrapper);
    frame->updateDepths(image_container->getImageWrapper<ImageType::Depth>(), base_params_.depth_background_thresh, base_params_.depth_obj_thresh);
    // rgbd_processor_.updateDepth(frame, image_container->getImageWrapper<ImageType::Depth>());

    LOG(INFO) << "In RGBD instance module frontend boostrap";
    previous_frame_ = frame;

    return {State::Nominal, nullptr};
}


FrontendModule::SpinReturn RGBDInstanceFrontendModule::nominalSpin(FrontendInputPacketBase::ConstPtr input) {
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

    if(display_queue_) {
        cv::Mat depth_disp;
        const cv::Mat& depth = image_container->getDepth();
        rgbd_processor_.disparityToDepth(depth, depth_disp);

        depth_disp.convertTo(depth_disp, CV_8UC1);
        display_queue_->push(ImageToDisplay("depth", depth_disp)); //eh something here no work
    }

    size_t n_optical_flow, n_new_tracks;
    Frame::Ptr frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), tracking_images, n_optical_flow, n_new_tracks);

    auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();
    vision_tools::disparityToDepth(base_params_, depth_image_wrapper, depth_image_wrapper);
    frame->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);


    AbsolutePoseCorrespondences correspondences;
    //this does not create proper bearing vectors (at leas tnot for 3d-2d pnp solve)
    //bearing vectors are also not undistorted atm!!
    rgbd_processor_.getCorrespondences(correspondences, *previous_frame_, *frame, KeyPointType::STATIC);

    LOG(INFO) << "Gotten correspondances, solving camera pose";

    // //also update the inliers outliers?
    TrackletIds inliers, outliers;
    frame->T_world_camera_ = motion_solver_.solveCameraPose(correspondences, inliers, outliers);
    // LOG(INFO) << "Solved camera pose";

    TrackletIds tracklets = frame->static_features_.collectTracklets();
    CHECK_GE(tracklets.size(), correspondences.size()); //tracklets shoudl be more (or same as) correspondances as there will be new points untracked

    frame->static_features_.markOutliers(outliers); //do we need to mark innliers? Should start as inliers

    cv::Mat moving_object_mat;
    rgbd_processor_.updateMovingObjects(*previous_frame_, frame, moving_object_mat);

    if(display_queue_) display_queue_->push(ImageToDisplay("moving objects", moving_object_mat));

    cv::Mat tracking_img = tracker_->computeImageTracks(*previous_frame_, *frame);
    if(display_queue_) display_queue_->push(ImageToDisplay("tracks", tracking_img));

    LandmarkMap landmark_map; //map of current features in world frame
    for(const Feature::Ptr& feature : frame->static_features_) {
        if(feature->usable()) {
            CHECK(feature->hasDepth());

            Landmark lmk_world;
            camera_->backProject(
                feature->keypoint_, feature->depth_, &lmk_world, frame->T_world_camera_
            );

            landmark_map.insert({feature->tracklet_id_, lmk_world});
        }
    }



    LOG(INFO) << "In RGBD instance module frontend nominal";
    previous_frame_ = frame;


    auto output = std::make_shared<FrontendOutputPacketBase>();
    output->input_ = input;
    output->frame_ = frame;
    output->tracked_landmarks = landmark_map;
    return {State::Nominal, output};
}

} //dyno
