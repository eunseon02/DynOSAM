/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/utils/Metrics.hpp"

#include <glog/logging.h>

namespace dyno {

FrontendLogger::FrontendLogger() {
    object_motion_errors_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "t_err", "r_err"));


    object_points_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "tracklet_id",
            "x_world", "y_world", "z_world"));

    camera_pose_errors_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "t_abs_err", "r_abs_err",
            "t_rel_err", "r_rel_err"));

    camera_pose_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "x", "y", "z",
            "roll", "pitch" , "yaw"));

    object_pose_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "x", "y", "z",
            "roll", "pitch" , "yaw"));

}

FrontendLogger::~FrontendLogger() {
    LOG(INFO) << "Writing out frontend logger...";

    OfstreamWrapper::WriteOutCsvWriter(*object_motion_errors_csv_, "frontend_motion_error_log.csv");
    OfstreamWrapper::WriteOutCsvWriter(*camera_pose_errors_csv_, "frontend_camera_pose_error_log.csv");
    OfstreamWrapper::WriteOutCsvWriter(*camera_pose_csv_, "frontend_camera_pose_log.csv");
    OfstreamWrapper::WriteOutCsvWriter(*object_pose_csv_, "frontend_object_pose_log.csv");
    OfstreamWrapper::WriteOutCsvWriter(*object_points_csv_, "frontend_object_points_log.csv");
}

void FrontendLogger::logObjectMotion(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const MotionEstimateMap& motion_estimates) {
    if(!gt_packets.exists(frame_id)) {
        LOG(WARNING) << "No gt packet at frame id " << frame_id << ". Unable to log frontend object motions";
        return;
    }

    const GroundTruthInputPacket& gt_packet = gt_packets.at(frame_id);

    for(const auto& [object_id, motions] : motion_estimates) {

        //TODO: -1, -1 if can not find gt motion? This will message with average!!
        utils::TRErrorPair motion_error(-1, -1);
        ObjectPoseGT object_pose_gt;
        if(!gt_packet.getObject(object_id, object_pose_gt)) {
            LOG(ERROR) << "Could not find gt object at frame " << frame_id << " for object Id" << object_id;
        }
        else {
            const gtsam::Pose3& estimate = motions;
            motion_error = utils::TRErrorPair::CalculatePoseError(estimate, *object_pose_gt.prev_H_current_world_);
        }

        //frame_id, object_id, t_err, r_error
        *object_motion_errors_csv_ << frame_id << object_id << motion_error.translation_ << motion_error.rot_;

    }
}

void FrontendLogger::logObjectPose(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const ObjectPoseMap& propogated_poses) {
    //assume object poses get logged in frame order!!!
    for(const auto&[object_id, poses_map] : propogated_poses) {
        //do not draw if in current frame
        if(!poses_map.exists(frame_id)) {
            continue;
        }

        const gtsam::Pose3& pose = poses_map.at(frame_id);
        const auto rot = pose.rotation();
        *object_pose_csv_ << frame_id << object_id << pose.x() << pose.y() << pose.z() << rot.roll() << rot.pitch() << rot.yaw();
    }
}


void FrontendLogger::logCameraPose(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const gtsam::Pose3& T_world_camera, std::optional<const gtsam::Pose3> T_world_camera_k_1) {
    if(!gt_packets.exists(frame_id)) {
        LOG(WARNING) << "No gt packet at frame id " << frame_id << ". Unable to log frontend object motions";
        return;
    }

    const GroundTruthInputPacket& gt_packet_k = gt_packets.at(frame_id);
    const gtsam::Pose3 gt_T_world_camera_k = gt_packet_k.X_world_;

    utils::TRErrorPair absolute_pose_error = utils::TRErrorPair::CalculatePoseError(T_world_camera,gt_T_world_camera_k);

    utils::TRErrorPair relative_pose_error(0, 0);

    const FrameId frame_id_k_1 = frame_id - 1;
    //if we have a previous gt packet AND a provided estimate
    if(gt_packets.exists(frame_id_k_1) && T_world_camera_k_1) {
        const GroundTruthInputPacket& gt_packet_k_1 = gt_packets.at(frame_id_k_1);
        const gtsam::Pose3 gt_T_world_camera_k_1 = gt_packet_k_1.X_world_;

        relative_pose_error = utils::TRErrorPair::CalculateRelativePoseError(T_world_camera_k_1.value(), T_world_camera, gt_T_world_camera_k_1, gt_T_world_camera_k);
    }


    //frame_id, t_abs_err, r_abs_error, t_rel_error, r_rel_error
    *camera_pose_errors_csv_ << frame_id << absolute_pose_error.translation_ << absolute_pose_error.rot_ << relative_pose_error.translation_ << relative_pose_error.rot_;

    const auto rot = T_world_camera.rotation();
    *camera_pose_csv_ << frame_id << T_world_camera.x() << T_world_camera.y() << T_world_camera.z() << rot.roll() << rot.pitch() << rot.yaw();

}

void FrontendLogger::logObjectPoints(FrameId frame_id, const gtsam::Pose3& T_world_camera_k, const StatusLandmarkEstimates& dynamic_landmarks) {
    for(const auto& status_lmks : dynamic_landmarks) {
        const TrackletId tracklet_id = status_lmks.tracklet_id_;
        const Landmark& lmk_camera = status_lmks.value_;
        ObjectId object_id = status_lmks.label_;

        //assume measurement is in camera frame
        Landmark lmk_world = T_world_camera_k * lmk_camera;

        *object_points_csv_ << frame_id << object_id << tracklet_id << lmk_world(0) << lmk_world(1) << lmk_world(2);
    }
}

}
