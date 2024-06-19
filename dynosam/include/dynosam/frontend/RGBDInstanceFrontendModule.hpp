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

#include "dynosam/common/Camera.hpp"
#include "dynosam/frontend/FrontendModule.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/ObjectTracker.hpp"


// #include "dynosam/common/byte_tracker/ByteTracker.hpp"

namespace dyno {

class RGBDInstanceFrontendModule : public FrontendModule {

public:
    RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue);
    ~RGBDInstanceFrontendModule();

    using SpinReturn = FrontendModule::SpinReturn;

private:
    Camera::Ptr camera_;
    EgoMotionSolver motion_solver_;
    ObjectMotionSovler object_motion_solver_;
    FeatureTracker::UniquePtr tracker_;
    RGBDFrontendLogger::UniquePtr logger_;

private:

    ImageValidationResult validateImageContainer(const ImageContainer::Ptr& image_container) const override;
    SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
    SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;

    /**
     * @brief Solves PnP between frame_k-1 and frame_k using the tracked correspondances
     * to estimate the frame of the current camera
     *
     * the pose of the Frame::Ptr (frame_k) is updated, and the features marked as outliers
     * by PnP are set as outliers.
     *
     * Depending on FrontendParams::use_ego_motion_pnp, a differnet solver will be used to estimate the pose
     *
     * @param frame_k
     * @param frame_k_1
     * @return true
     * @return false
     */
    bool solveCameraMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1);


    bool solveObjectMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1, ObjectId object_id, MotionEstimateMap& motion_estimates);

    RGBDInstanceOutputPacket::Ptr constructOutput(
        const Frame& frame,
        const MotionEstimateMap& estimated_motions,
        const gtsam::Pose3& T_world_camera,
        const GroundTruthInputPacket::Optional& gt_packet = std::nullopt,
        const DebugImagery::Optional& debug_imagery = std::nullopt);

    //if FLAGS_use_byte_tracker is true, runs the Byte Tracking algorithm on the input image and generates a global track
    //which is ued to update the motion mask such that each pixel value is set to the global track for that object
    void objectTrack(TrackingInputImages& tracking_images, FrameId frame_id);

    //determines if the objects in the current frame are static/dynamic by looking at the scene flow
    //depth must be provided for each feature
    //current camera pose must be calculated and set in the current_frame
    //this will remove non-dynamic objects from the Frame::object_observations, is this what we want?
    //TODO: maybe better to keep them but to check if theyre dynamic later on!!!
    void determineDynamicObjects(const Frame& previous_frame, Frame::Ptr current_frame, bool used_semantic_mask);

    //updates the object_poses_ map which is then sent to the viz via the output
    //this updates the estimated trajectory of the object from a starting pont using the
    //object motion to propofate the object pose
    void propogateObjectPoses(const MotionEstimateMap& motion_estimates, FrameId frame_id);
    //updates object poses
    // void propogateObjectPose(const gtsam::Pose3& prev_H_world_curr, ObjectId object_id, FrameId frame_id);


    //used when we want to seralize the output to json via the FLAGS_save_frontend_json flag
    std::map<FrameId, RGBDInstanceOutputPacket::Ptr> output_packet_record_;


    UndistorterRectifier::Ptr undistorter_; //TODO: lots of places this is used!! Streamline so we do undistortion once!!

    ObjectTracker object_tracker_;



};


} //dyno
