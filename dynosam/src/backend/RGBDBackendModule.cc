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
#include "dynosam/logger/Logger.hpp"


#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"

#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <glog/logging.h>
#include <gflags/gflags.h>

DECLARE_string(output_path); //defined in logger/logger.h

namespace dyno {

RGBDBackendModule::RGBDBackendModule(const BackendParams& backend_params, Camera::Ptr camera, Map3d::Ptr map, ImageDisplayQueue* display_queue)
    : BackendModule(backend_params, camera, display_queue), map_(CHECK_NOTNULL(map))
{
    gtsam::ISAM2Params isam_params;
    isam_params.findUnusedFactorSlots = false; //this is very important rn as we naively keep track of slots
    // isam_params.relinearizeThreshold = 0.01;

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
    LOG(INFO) << "Running backend " << frame_k;
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
    //  map_->updateEstimates(new_values_, new_factors_, frame_k);

    return {State::Nominal, nullptr};
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::rgbdNominalSpin(RGBDInstanceOutputPacket::ConstPtr input) {

    const FrameId frame_k = input->getFrameId();
    LOG(INFO) << "Running backend " << frame_k;
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
    updateStaticObservations(T_world_cam_k_frontend, frame_k, new_values, new_factors_);

    // gtsam::Values new_dyn_values;
    // gtsam::NonlinearFactorGraph new_dyn_factors;
    updateDynamicObservations(T_world_cam_k_frontend, frame_k, new_values, new_factors);

    optimize(frame_k, new_values, new_factors);
    // map_->updateEstimates(new_values_, new_factors_, frame_k);

    //TODO: update debug info

    auto backend_output = std::make_shared<BackendOutputPacket>();
    backend_output->timestamp_ = input->getTimestamp();
    // backend_output->T_world_camera_ = map_->getPoseEstimate(frame_k).get();
    // backend_output->static_landmarks_ = map_->getFullStaticMap();
    // backend_output->dynamic_landmarks_ = map_->getD

    debug_info_ = DebugInfo();

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

void RGBDBackendModule::updateDynamicObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors) {
    const FrameId frame_id_k_1 = frame_id_k - 1u;

    const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_id_k);
    const FrameNode3d::Ptr frame_node_k_1 = map_->getFrame(frame_id_k_1);
    CHECK_NOTNULL(frame_node_k_1);

    constexpr static size_t kMinObservations = 3u;

    //for each object
    for(const ObjectNode3d::Ptr& object_node : frame_node_k->objects_seen) {


        DebugInfo::ObjectInfo object_debug_info;

        const ObjectId object_id = object_node->getId();

        //first check that object exists in the previous frame
        if(!frame_node_k->objectMotionExpected(object_id)) {
           continue;
        }

        // possibly the longest call?
        // landmarks on this object seen at frame k
        LandmarkNodePtrSet<Landmark> seen_lmks_k = object_node->getLandmarksSeenAtFrame(frame_id_k);

        //if we dont have at least 3 observations of this object in this frame AND the previous frame
        if(seen_lmks_k.size() < 3u || object_node->getLandmarksSeenAtFrame(frame_id_k_1).size() < 3) {
            continue;
        }

        //iterate over each lmk we have on this object
        for(const LandmarkNode3d::Ptr& obj_lmk_node : seen_lmks_k) {
            CHECK_EQ(obj_lmk_node->getObjectId(), object_id);

            //see if we have enough observations to add this lmk
            if(obj_lmk_node->numObservations() < kMinObservations) { continue;}

            TrackletId tracklet_id = obj_lmk_node->getId();

            // LOG(INFO) << "Tracklet exists " << is_dynamic_tracklet_in_map_.exists(tracklet_id);

            //if does not exist, we need to go back and all the previous measurements & factors & motions
            if(!is_dynamic_tracklet_in_map_.exists(tracklet_id)) {
                //could sanity check that no object motion exists for the previous frame?

                //add the points/motions from the past
                FrameNodePtrSet<Landmark> seen_frames = obj_lmk_node->getSeenFrames();
                // CHECK_EQ(frame_id_k - kMinObservations, seen_frames.getFirstIndex<FrameId>()) << container_to_string(seen_frames.collectIds<FrameId>());
                // CHECK_EQ(frame_id_k, seen_frames.getLastIndex<FrameId>()) <<  container_to_string(seen_frames.collectIds<FrameId>());
                //assert frame observations are continuous?

                //start at the first frame we want to start adding points in as we know have seen them enough times
                //start from +1, becuase the motion index is k-1 to k and there is no motion k-2 to k-1
                //but the first poitns we want to add are at k-1
                const FrameId starting_motion_frame = seen_frames.getFirstIndex<FrameId>() + 1u; //as we index the motion from k

                auto starting_motion_frame_itr = seen_frames.find(starting_motion_frame);
                CHECK(starting_motion_frame_itr != seen_frames.end());

                std::stringstream ss;
                ss << "Going back to add point on object " << object_id << " at frames\n";

                for(auto seen_frames_itr = starting_motion_frame_itr; seen_frames_itr != seen_frames.end(); seen_frames_itr++) {
                    auto seen_frames_itr_prev = seen_frames_itr;
                    std::advance(seen_frames_itr_prev, -1);
                    CHECK(seen_frames_itr_prev != seen_frames.end());

                    FrameNode3d::Ptr query_frame_node_k = *seen_frames_itr;
                    FrameNode3d::Ptr query_frame_node_k_1 = *seen_frames_itr_prev;


                    CHECK_EQ(query_frame_node_k->frame_id, query_frame_node_k_1->frame_id + 1u);

                    //add points UP TO AND INCLUDING the current frame
                    if(query_frame_node_k->frame_id > frame_id_k) {
                        break;
                    }



                    //point needs to be be in k and k-1 -> we have validated the object exists in these two frames
                    //but not the points
                    CHECK(obj_lmk_node->seenAtFrame(query_frame_node_k->frame_id));
                    if(!obj_lmk_node->seenAtFrame(query_frame_node_k_1->frame_id)) {
                        LOG(WARNING) << "Tracklet " << tracklet_id << " on object " << object_id << " seen at " << query_frame_node_k->frame_id << " but not " << query_frame_node_k_1->frame_id;
                        break;
                    } //this miay mean this this point never gets added?

                    ss << query_frame_node_k_1->frame_id << " " << query_frame_node_k->frame_id << "\n";

                    const Landmark& measured_k = obj_lmk_node->getMeasurement(query_frame_node_k);
                    const Landmark& measured_k_1 = obj_lmk_node->getMeasurement(query_frame_node_k_1);

                    gtsam::Key object_point_key_k = obj_lmk_node->makeDynamicKey(query_frame_node_k->frame_id);
                    gtsam::Key object_point_key_k_1 = obj_lmk_node->makeDynamicKey(query_frame_node_k_1->frame_id);

                    //this assumes we add all the points in order and have continuous frames (which we should have?)
                    if(seen_frames_itr == starting_motion_frame_itr) {
                        //on first iteration we should add both values at query_k and query_k_1
                        //otherwise we just need to add k, as k_1 will be added from the previous iteration
                        new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                            query_frame_node_k_1->makePoseKey(), //pose key at previous frames
                            object_point_key_k_1,
                            measured_k_1,
                            dynamic_point_noise_
                        );
                        object_debug_info.num_dynamic_factors++;

                        const Landmark lmk_world_k_1 = T_world_camera * measured_k_1;
                        new_values.insert(object_point_key_k_1, lmk_world_k_1);
                        object_debug_info.num_new_dynamic_points++;
                    }

                    //previous point must be added by the previous iteration
                    CHECK(new_values.exists(object_point_key_k_1));

                    new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                        query_frame_node_k->makePoseKey(), //pose key at previous frames
                        object_point_key_k,
                        measured_k,
                        dynamic_point_noise_
                    );
                    object_debug_info.num_dynamic_factors++;

                    const Landmark lmk_world_k = T_world_camera * measured_k;
                    new_values.insert(object_point_key_k, lmk_world_k);
                    object_debug_info.num_new_dynamic_points++;

                    const gtsam::Key object_motion_key_k = query_frame_node_k->makeObjectMotionKey(object_id);
                    //check in map as well, because this could be a new point on an old object, meaning the motion already exists
                    if(!map_->exists(object_motion_key_k, new_values)) {
                        //make new object motion
                        new_values.insert(object_motion_key_k, gtsam::Pose3::Identity());
                        LOG(INFO) << "Adding value " << DynoLikeKeyFormatterVerbose(object_motion_key_k);
                    }

                    new_point_factors.emplace_shared<LandmarkMotionTernaryFactor>(
                        object_point_key_k_1,
                        object_point_key_k,
                        object_motion_key_k,
                        landmark_motion_noise_
                    );
                    object_debug_info.num_motion_factors++;

                }

                // LOG(INFO) << ss.str();
                is_dynamic_tracklet_in_map_.insert2(tracklet_id, true);
            }
            else {
                //these tracklets should already be in the graph so we should only need to add the new measurements from this frame
                //check that we have previous point for this frame
                gtsam::Key object_point_key_k = obj_lmk_node->makeDynamicKey(frame_id_k);
                gtsam::Key object_point_key_k_1 = obj_lmk_node->makeDynamicKey(frame_id_k_1);

                const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(object_id);

                CHECK(map_->exists(object_point_key_k_1));
                CHECK(!map_->exists(object_point_key_k, new_values));

                const Landmark measured_k = obj_lmk_node->getMeasurement(frame_id_k);
                new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                    frame_node_k->makePoseKey(), //pose key at previous frames
                    object_point_key_k,
                    measured_k,
                    dynamic_point_noise_
                );
                object_debug_info.num_dynamic_factors++;

                new_point_factors.emplace_shared<LandmarkMotionTernaryFactor>(
                    object_point_key_k_1,
                    object_point_key_k,
                    object_motion_key_k,
                    landmark_motion_noise_
                );
                object_debug_info.num_motion_factors++;

                const Landmark lmk_world_k = T_world_camera * measured_k;
                new_values.insert(object_point_key_k, lmk_world_k);
                object_debug_info.num_new_dynamic_points++;

                //should not have to check the map because this is an old point
                if(!new_values.exists(object_motion_key_k)) {
                    //make new object motion -> this can happen if all the points on the object are well tracked from the previous motion
                    new_values.insert(object_motion_key_k, gtsam::Pose3::Identity());
                    LOG(INFO) << "Adding value " << DynoLikeKeyFormatterVerbose(object_motion_key_k);
                }
            }
        }

        debug_info_.object_info.insert2(object_id, object_debug_info);

    }

    for(const auto&[object_id, object_info] : debug_info_.object_info) {
        std::stringstream ss;
        ss << "Object id debug info: " << object_id << "\n";
        ss << "Num point factors: " << object_info.num_dynamic_factors << "\n";
        ss << "Num point variables: " << object_info.num_new_dynamic_points << "\n";
        ss << "Num motion factors: " << object_info.num_motion_factors << "\n";

        LOG(INFO) << ss.str();
    }
}


void RGBDBackendModule::optimize(FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    gtsam::ISAM2UpdateParams isam_update_params;

    LOG(INFO) << "Starting optimization for " << frame_id_k;

    gtsam::Values old_state = map_->getValues();

    try {
        gtsam::ISAM2Result result = smoother_->update(new_factors, new_values, isam_update_params);
    }
    catch(const gtsam::ValuesKeyAlreadyExists& e) {
        new_factors.saveGraph(FLAGS_output_path + "/isam2_graph.dot", DynoLikeKeyFormatter);
        LOG(FATAL) << "gtsam::ValuesKeyAlreadyExists thrown near key " << DynoLikeKeyFormatterVerbose(e.key());
    }
    catch(const gtsam::ValuesKeyDoesNotExist& e) {
        new_factors.saveGraph(FLAGS_output_path + "/isam2_graph.dot", DynoLikeKeyFormatter);
        LOG(FATAL) << "gtsam::ValuesKeyDoesNotExist thrown near key " << DynoLikeKeyFormatterVerbose(e.key());
    }


    // gtsam::Values estimate = smoother_->calculateBestEstimate();
    gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();
    // gtsam::Values all_values = smoother_->

    gtsam::Values all_values = map_->getValues();
    all_values.insert_or_assign(new_values);

    // double error_before = graph.error(old_state);
    // double error_after = graph.error(estimate);

    // LOG(INFO) << "Optimization Errors:\n"
    //          << " - Error before :" << error_before
    //          << '\n'
    //          << " - Error after  :" << error_after;

    // map_->updateEstimates(estimate, graph, frame_id_k);
    map_->updateEstimates(all_values, graph, frame_id_k);
}


void RGBDBackendModule::saveGraph(const std::string& file) {
    gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();
    graph.saveGraph(getOutputFilePath(file), DynoLikeKeyFormatter);
}
void RGBDBackendModule::saveTree(const std::string& file) {
    smoother_->saveGraph(getOutputFilePath(file), DynoLikeKeyFormatter);
}



} //dyno
