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

#include "dynosam/backend/RGBDBackendModule.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"

#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <glog/logging.h>

namespace dyno {

RGBDBackendModule::RGBDBackendModule(const BackendParams& backend_params, Camera::Ptr camera, Map3d::Ptr map, ImageDisplayQueue* display_queue)
    : BackendModule(backend_params, camera, display_queue), map_(CHECK_NOTNULL(map))
{
    gtsam::ISAM2Params isam_params;
    isam_params.findUnusedFactorSlots = false; //this is very important rn as we naively keep track of slots
    isam_params.relinearizeThreshold = 0.01;

    smoother_ = std::make_unique<gtsam::ISAM2>(isam_params);

    //TODO: functioanlise and streamline with BackendModule
    static_point_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.static_point_noise_sigma_);
    dynamic_point_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.dynamic_point_noise_sigma_);

    if(backend_params.use_robust_kernals_) {
        static_point_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), static_point_noise_);

        dynamic_point_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), dynamic_point_noise_);
    }
}

RGBDBackendModule::~RGBDBackendModule() {}

RGBDBackendModule::SpinReturn
RGBDBackendModule::boostrapSpin(BackendInputPacket::ConstPtr input) {
    RGBDInstanceOutputPacket::ConstPtr rgbd_output = safeCast<BackendInputPacket, RGBDInstanceOutputPacket>(input);
    checkAndThrow((bool)rgbd_output, "Failed to cast BackendInputPacket to RGBDInstanceOutputPacket in RGBDBackendModule");

    return rgbdBoostrapSpin(rgbd_output);
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::nominalSpin(BackendInputPacket::ConstPtr input) {
    RGBDInstanceOutputPacket::ConstPtr rgbd_output = safeCast<BackendInputPacket, RGBDInstanceOutputPacket>(input);
    checkAndThrow((bool)rgbd_output, "Failed to cast BackendInputPacket to RGBDInstanceOutputPacket in RGBDBackendModule");

    return rgbdNominalSpin(rgbd_output);
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::rgbdBoostrapSpin(RGBDInstanceOutputPacket::ConstPtr input) {

    const FrameId frame_k = input->getFrameId();
    //estimate of pose from the frontend
    const gtsam::Pose3 T_world_cam_k_frontend = input->T_world_camera_;

    {
        utils::TimingStatsCollector("map.update_observations");
        map_->updateObservations(input->static_landmarks_);
        map_->updateObservations(input->dynamic_landmarks_);
    }

    gtsam::Values new_values;
    gtsam::NonlinearFactorGraph new_factors;

    addInitialPose(T_world_cam_k_frontend, frame_k, new_values, new_factors);
    //must optimzie to update the map
    optimize(frame_k, new_values, new_factors);

    return {State::Nominal, nullptr};
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::rgbdNominalSpin(RGBDInstanceOutputPacket::ConstPtr input) {

    const FrameId frame_k = input->getFrameId();
    const FrameId frame_k_1 = frame_k - 1u;
    //estimate of pose from the frontend
    const gtsam::Pose3 T_world_cam_k_frontend = input->T_world_camera_;
    {
        utils::TimingStatsCollector("map.update_observations");
        map_->updateObservations(input->static_landmarks_);
        map_->updateObservations(input->dynamic_landmarks_);
    }

    gtsam::Values new_values;
    gtsam::NonlinearFactorGraph new_factors;
    addOdometry(T_world_cam_k_frontend, frame_k, frame_k_1, new_values, new_factors);
    updateStaticObservations(T_world_cam_k_frontend, frame_k, new_values, new_factors);

    optimize(frame_k, new_values, new_factors);

    const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_k);
    CHECK_EQ(frame_node_k->frame_id, frame_k);
    auto static_measurements_k = frame_node_k->getStaticMeasurements();

    LandmarkMap static_lmks;
    // for(const auto&[lmk_node, measurement] : static_measurements_k) {
    //     Landmark lmk_world = T_world_cam_k_frontend * measurement;
    //     static_lmks.insert2(lmk_node->getId(), lmk_world);
    // }

    StatusLandmarkEstimates estimates = map_->getFullStaticMap();
    for(const auto& status : estimates) {
        static_lmks.insert2(status.tracklet_id_, status.value_);
    }

    auto backend_output = std::make_shared<BackendOutputPacket>();
    backend_output->timestamp_ = input->getTimestamp();
    backend_output->T_world_camera_ = T_world_cam_k_frontend;
    backend_output->static_lmks_ = static_lmks;

    return {State::Nominal, backend_output};
}

void RGBDBackendModule::addInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
    new_factors.addPrior(CameraPoseSymbol(frame_id_k), T_world_camera, initial_pose_prior_);
}

void RGBDBackendModule::addOdometry(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, FrameId frame_id_k_1, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    //add state
    new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);

    StateQuery<gtsam::Pose3> pose_query = map_->getPoseEstimate(frame_id_k_1);
    if(pose_query) {
        LOG(INFO) << "Adding odom between " << frame_id_k_1 << " and " << frame_id_k;
        const gtsam::Pose3 T_world_camera_k_1 = pose_query.get();
        const gtsam::Pose3 odom = T_world_camera_k_1.inverse() * T_world_camera;

        factor_graph_tools::addBetweenFactor(frame_id_k_1, frame_id_k, odom, odometry_noise_, new_factors);
    }
}


void RGBDBackendModule::updateStaticObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors) {
    const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_id_k);
    CHECK_NOTNULL(frame_node_k);

    constexpr static size_t kMinObservations = 3u;
    for(const LandmarkNode3d::Ptr& lmk_node : frame_node_k->static_landmarks) {
        const gtsam::Key point_key = lmk_node->makeStaticKey();
        //check if lmk node is already in map (which should mean it is equivalently in isam)
        if(map_->exists(point_key)) {
            //3d point in camera frame
            const Landmark& measured = lmk_node->getMeasurement(frame_id_k);
            new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                frame_node_k->makePoseKey(), //pose key for this frame
                point_key,
                measured,
                static_point_noise_
            );

            // LOG(INFO) << "Adding new factors for static point " << DynoLikeKeyFormatter(point_key) << " at frame " << frame_id_k;
        }
        else {
            //see if we have enough observations to add this lmk
            if(lmk_node->numObservations() < kMinObservations) { continue;}

            FrameNodePtrSet<Landmark> seen_frames = lmk_node->getSeenFrames();
            for(const FrameNode3d::Ptr& seen_frame : seen_frames) {
                //check for sanity that the seen frames are <= than the current frame
                CHECK_LE(seen_frame->getId(), frame_id_k);
                const Landmark& measured = lmk_node->getMeasurement(seen_frame);
                new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                    seen_frame->makePoseKey(), //pose key at previous frames
                    point_key,
                    measured,
                    static_point_noise_
                );
            }

            //add initial value
            //pick the one in this frame
            const Landmark& measured = lmk_node->getMeasurement(frame_id_k);
            const Landmark lmk_world = T_world_camera * measured;
            new_values.insert(point_key, lmk_world);

            // LOG(INFO) << "Adding new value for static point " << DynoLikeKeyFormatter(point_key) << " at frame " << frame_id_k;
        }
    }
}


void RGBDBackendModule::optimize(FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    gtsam::ISAM2UpdateParams isam_update_params;

    gtsam::Values old_state = map_->getValues();
    gtsam::NonlinearFactorGraph old_graph = map_->getGraph();

    gtsam::ISAM2Result result = smoother_->update(new_factors, new_values, isam_update_params);

    gtsam::Values estimate = smoother_->calculateBestEstimate();
    gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();

    double error_before = old_graph.error(old_state);
    double error_after = old_graph.error(estimate);

    LOG(INFO) << "Optimization Errors:\n"
             << " - Error before :" << error_before
             << '\n'
             << " - Error after  :" << error_after;

    map_->updateEstimates(estimate, graph, frame_id_k);
}



} //dyno
