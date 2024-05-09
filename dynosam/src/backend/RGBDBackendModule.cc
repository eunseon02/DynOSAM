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

#include "dynosam/common/Flags.hpp"

#include "dynosam/backend/IncrementalOptimizer.hpp"


#include "dynosam/factors/LandmarkQuadricFactor.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/ObjectKinematicFactor.hpp"

#include <gtsam/base/debug.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <glog/logging.h>
#include <gflags/gflags.h>



namespace dyno {

RGBDBackendModule::RGBDBackendModule(const BackendParams& backend_params, Map3d::Ptr map, RGBDOptimizer::Ptr optimizer, const UpdaterType& updater_type, ImageDisplayQueue* display_queue)
    : Base(backend_params, map, optimizer, display_queue)
{
    CHECK_NOTNULL(map);
    CHECK_NOTNULL(map_);

    //TODO: functioanlise and streamline with BackendModule
    static_point_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.static_point_noise_sigma_);
    dynamic_point_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.dynamic_point_noise_sigma_);
    //set in base!
    // landmark_motion_noise_ =  gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.motion_ternary_factor_noise_sigma_);

    if(backend_params.use_robust_kernals_) {
        static_point_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), static_point_noise_);

        dynamic_point_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), dynamic_point_noise_);

        //TODO: not k_huber_3d_points_ not just used for 3d points
        landmark_motion_noise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(backend_params.k_huber_3d_points_), landmark_motion_noise_);

    }

    CHECK_NOTNULL(static_point_noise_);
    CHECK_NOTNULL(dynamic_point_noise_);
    CHECK_NOTNULL(landmark_motion_noise_);

    if(updater_type == UpdaterType::MotionInWorld) {
        updater_ = std::make_unique<UpdateImplInWorld>(this);
        LOG(INFO) << "Using UpdateImplInWorld";
    }
    else {
        CHECK(false) << "Not implemented";
    }

    // updater_ = std::make_unique<UpdateImplInWorldPrimitives>(this);
    CHECK_NOTNULL(updater_);

    if(backend_params.use_logger_) {
        logger_ = std::make_unique<BackendLogger>(updater_->name());
        LOG(INFO) << "Creating backend logger with name " << logger_->moduleName();
    }


}

RGBDBackendModule::~RGBDBackendModule() {
    LOG(INFO) << "Destructing RGBDBackendModule";

    // //TODO: for now - this module does not know if we have poses in the graph
    // if(logger_) {
    //     gtsam::FastMap<ObjectId, gtsam::Vector3> estimated_radii;
    //     for(ObjectId object_id : map_->getObjectIds()) {
    //         gtsam::Vector3 P;
    //         if(map_->safeGet(P, ObjectQuadricSymbol(object_id))) {
    //             estimated_radii.insert2(object_id, P);
    //         }
    //     }

    //     if(!estimated_radii.empty()) {
    //         logger_->logEllipsoids(estimated_radii);
    //     }
    // }

}

RGBDBackendModule::SpinReturn
RGBDBackendModule::boostrapSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) {

    const FrameId frame_k = input->getFrameId();
    CHECK_EQ(spin_state_.frame_id, frame_k);
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

    updater_->setInitialPose(T_world_cam_k_frontend, frame_k, new_values, new_factors);
    //must optimzie to update the map
    optimizer_->update(spin_state_, new_values, new_factors, map_);
    // optimize(frame_k, new_values, new_factors);
    map_->updateEstimates(new_values, new_factors, frame_k);

    return {State::Nominal, nullptr};
}

RGBDBackendModule::SpinReturn
RGBDBackendModule::nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) {

    const FrameId frame_k = input->getFrameId();
    CHECK_EQ(spin_state_.frame_id, frame_k);

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
    updater_->updateOdometry(T_world_cam_k_frontend, frame_k, frame_k_1, new_values, new_factors);
    updater_->updateStaticObservations(frame_k, T_world_cam_k_frontend, new_values, new_factors, debug_info_);

    // gtsam::Values new_dyn_values;
    // gtsam::NonlinearFactorGraph new_dyn_factors;
    updater_->updateDynamicObservations(frame_k, T_world_cam_k_frontend, new_values, new_factors, debug_info_);

    gtsam::NonlinearFactorGraph full_graph = map_->getGraph();
    full_graph += new_factors;

    //update the map with the new values
    //the map will insert or assign these new values which MAY then be updated from the state
    //NOTE: do we want optimzied vs not optimized values in the graph?

    const bool optimized = optimizer_->update(spin_state_, new_values, new_factors, map_);

    //NOTE: assumes frames start at 0
    static FrameId last_optimized_frame = optimizer_->getLastOptimizedState().frame_id;

    if(optimized) {
        const auto& esimtates = optimizer_->getValues();
        // const auto& graph
        map_->updateEstimates(esimtates, full_graph, frame_k);

        if(logger_) {
            //only logs this the frame specified - in batch case we want to update all!!
            for(FrameId frame_id = last_optimized_frame; frame_id <= frame_k; frame_id++) {
                //the updater is the only thing that knows EXACTLY which values it put into the map!!
                //hmmm is this problematic when we create the final output
                //the map should also know which values it has (e.g. if we have object pose?)
                updater_->logBackendFromMap(frame_id, map_, *logger_);
            }
        }

        last_optimized_frame = optimizer_->getLastOptimizedState().frame_id;
        CHECK_EQ(last_optimized_frame, frame_k);

        // optimizer_->getFactors().saveGraph(getOutputFilePath("batch_output.dot"), DynoLikeKeyFormatter);
    }
    else {
        //must update the map with the new values anyway
        map_->updateEstimates(new_values, full_graph, frame_k);
    }


    // if(optimizer_->shouldOptimize(frame_k)) {
    //     gtsam::Values estimate;
    //     gtsam::NonlinearFactorGraph graph;
    //     std::tie(estimate, graph) = optimizer_->optimize();
    //     // TODO: depending on if (fixed-lag)incremental or batch, the graph returned from optimize may or MAY not be the full graph
    //     //NOTE: this also doesnt allow us to query the map for when the last time we updated the (optimized) values
    //     //as we need these values in the map for when we call updateStaticObservations/updateDynamicObservations
    //     //maybe break into update initial values and update estimates?
    //     map_->updateEstimates(estimate, full_graph, frame_k);
    //     // must be called after the map update as it uses this to get all the info
    //     //currently only propofate object pose when optimize?
    //     //can only run with ground truth!!!???!!
    //     // auto composed_poses = getObjectPoses(FLAGS_init_object_pose_from_gt);

    //     // // //only logs this the frame specified - in batch case we want to update all!!
    //     // for(FrameId frame_id = last_optimized_frame; frame_id <= frame_k; frame_id++)
    //     //     logBackendFromMap(frame_id, composed_poses);

    //     last_optimized_frame = frame_k;
    // }
    // else {
    //     gtsam::Values values = map_->getValues();
    //     values.insert_or_assign(new_values);

    //     map_->updateEstimates(values, full_graph, frame_k);
    // }


    auto backend_output = std::make_shared<BackendOutputPacket>();
    backend_output->timestamp_ = input->getTimestamp();
    backend_output->frame_id_ = input->getFrameId();
    backend_output->T_world_camera_ = map_->getPoseEstimate(frame_k).get();
    backend_output->static_landmarks_ = map_->getFullStaticMap();
    backend_output->dynamic_landmarks_ = map_->getDynamicMap(frame_k);
    // backend_output->composed_object_poses = updateObjectPoses(frame_k, input);

    debug_info_ = DebugInfo();
    new_object_measurements_per_frame_.clear();

    return {State::Nominal, backend_output};
}

// void RGBDBackendModule::addInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
//     new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
//     new_factors.addPrior(CameraPoseSymbol(frame_id_k), T_world_camera, initial_pose_prior_);
// }

// void RGBDBackendModule::addOdometry(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, FrameId frame_id_k_1, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
//     //add state
//     new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
//     timestamp_map_[CameraPoseSymbol(frame_id_k)] = frame_id_k;

//     StateQuery<gtsam::Pose3> pose_query = map_->getPoseEstimate(frame_id_k_1);
//     if(pose_query) {
//         LOG(INFO) << "Adding odom between " << frame_id_k_1 << " and " << frame_id_k;
//         const gtsam::Pose3 T_world_camera_k_1 = pose_query.get();
//         const gtsam::Pose3 odom = T_world_camera_k_1.inverse() * T_world_camera;

//         factor_graph_tools::addBetweenFactor(frame_id_k_1, frame_id_k, odom, odometry_noise_, new_factors);
//     }
// }


// void RGBDBackendModule::updateStaticObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors) {
//     const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_id_k);
//     CHECK_NOTNULL(frame_node_k);

//     constexpr static size_t kMinObservations = 2u;

//     //TODO: debug stuff
//     int num_points = 0;

//     LOG(INFO) << "Looping over " <<  frame_node_k->static_landmarks.size() << " static lmks for frame " << frame_id_k;
//     for(const LandmarkNode3d::Ptr& lmk_node : frame_node_k->static_landmarks) {


//         const gtsam::Key point_key = lmk_node->makeStaticKey();
//         //check if lmk node is already in map (which should mean it is equivalently in isam)
//         if(map_->exists(point_key)) {
//             //3d point in camera frame
//             const Landmark& measured = lmk_node->getMeasurement(frame_id_k);
//             new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                 frame_node_k->makePoseKey(), //pose key for this frame
//                 point_key,
//                 measured,
//                 static_point_noise_
//             );

//             debug_info_.num_static_factors++;

//         }
//         else {
//             //see if we have enough observations to add this lmk
//             if(lmk_node->numObservations() < kMinObservations) { continue;}

//             num_points++;

//             // if(num_points > 5) {break;}

//             //this condition should only run once per tracklet (ie.e the first time the tracklet has enough observations)
//             //we gather the tracklet observations and then initalise it in the new values
//             //these should then get added to the map
//             //and map_->exists() should return true for all other times
//             FrameNodePtrSet<Landmark> seen_frames = lmk_node->getSeenFrames();
//             for(const FrameNode3d::Ptr& seen_frame : seen_frames) {
//                 //check for sanity that the seen frames are <= than the current frame
//                 CHECK_LE(seen_frame->getId(), frame_id_k);
//                 const Landmark& measured = lmk_node->getMeasurement(seen_frame);
//                 new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                     seen_frame->makePoseKey(), //pose key at previous frames
//                     point_key,
//                     measured,
//                     static_point_noise_
//                 );
//             }

//             //add initial value
//             //pick the one in this frame
//             const Landmark& measured = lmk_node->getMeasurement(frame_id_k);
//             const Landmark lmk_world = T_world_camera * measured;
//             new_values.insert(point_key, lmk_world);
//             timestamp_map_[point_key] = frame_id_k;

//             debug_info_.num_new_static_points++;

//         }
//     }

//     LOG(INFO) << "Num new static points: " << debug_info_.num_new_static_points << "\n" << "Num new static factors " << debug_info_.num_static_factors;
// }

// void RGBDBackendModule::updateDynamicObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors) {
//     const FrameId frame_id_k_1 = frame_id_k - 1u;

//     const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_id_k);
//     const FrameNode3d::Ptr frame_node_k_1 = map_->getFrame(frame_id_k_1);
//     CHECK_NOTNULL(frame_node_k_1);

//     constexpr static size_t kMinObservations = 3u;

//     //for each object
//     for(const ObjectNode3d::Ptr& object_node : frame_node_k->objects_seen) {


//         DebugInfo::ObjectInfo object_debug_info;

//         const ObjectId object_id = object_node->getId();

//         //first check that object exists in the previous frame
//         if(!frame_node_k->objectMotionExpected(object_id)) {
//            continue;
//         }

//         // possibly the longest call?
//         // landmarks on this object seen at frame k
//         LandmarkNodePtrSet<Landmark> seen_lmks_k = object_node->getLandmarksSeenAtFrame(frame_id_k);

//         //if we dont have at least 3 observations of this object in this frame AND the previous frame
//         if(seen_lmks_k.size() < 3u || object_node->getLandmarksSeenAtFrame(frame_id_k_1).size() < 3) {
//             continue;
//         }

//         //TODO: debug
//         int num_lmks = 0;


//         //iterate over each lmk we have on this object
//         for(const LandmarkNode3d::Ptr& obj_lmk_node : seen_lmks_k) {
//             CHECK_EQ(obj_lmk_node->getObjectId(), object_id);

//             //see if we have enough observations to add this lmk
//             if(obj_lmk_node->numObservations() < kMinObservations) { continue;}

//             num_lmks++;

//             // if(num_lmks > 5) {
//             //     break;
//             // }

//             TrackletId tracklet_id = obj_lmk_node->getId();

//             // LOG(INFO) << "Tracklet exists " << is_dynamic_tracklet_in_map_.exists(tracklet_id);

//             //if does not exist, we need to go back and all the previous measurements & factors & motions
//             if(!is_dynamic_tracklet_in_map_.exists(tracklet_id)) {
//                 //could sanity check that no object motion exists for the previous frame?

//                 //add the points/motions from the past
//                 FrameNodePtrSet<Landmark> seen_frames = obj_lmk_node->getSeenFrames();
//                 // CHECK_EQ(frame_id_k - kMinObservations, seen_frames.getFirstIndex<FrameId>()) << container_to_string(seen_frames.collectIds<FrameId>());
//                 // CHECK_EQ(frame_id_k, seen_frames.getLastIndex<FrameId>()) <<  container_to_string(seen_frames.collectIds<FrameId>());
//                 //assert frame observations are continuous?

//                 //start at the first frame we want to start adding points in as we know have seen them enough times
//                 //start from +1, becuase the motion index is k-1 to k and there is no motion k-2 to k-1
//                 //but the first poitns we want to add are at k-1
//                 const FrameId starting_motion_frame = seen_frames.getFirstIndex<FrameId>() + 1u; //as we index the motion from k

//                 auto starting_motion_frame_itr = seen_frames.find(starting_motion_frame);
//                 CHECK(starting_motion_frame_itr != seen_frames.end());

//                 std::stringstream ss;
//                 ss << "Going back to add point on object " << object_id << " at frames\n";

//                 //iterate over k-N to k (inclusive) and all all
//                 for(auto seen_frames_itr = starting_motion_frame_itr; seen_frames_itr != seen_frames.end(); seen_frames_itr++) {
//                     auto seen_frames_itr_prev = seen_frames_itr;
//                     std::advance(seen_frames_itr_prev, -1);
//                     CHECK(seen_frames_itr_prev != seen_frames.end());

//                     FrameNode3d::Ptr query_frame_node_k = *seen_frames_itr;
//                     FrameNode3d::Ptr query_frame_node_k_1 = *seen_frames_itr_prev;


//                     CHECK_EQ(query_frame_node_k->frame_id, query_frame_node_k_1->frame_id + 1u);

//                     //add points UP TO AND INCLUDING the current frame
//                     if(query_frame_node_k->frame_id > frame_id_k) {
//                         break;
//                     }

//                     //point needs to be be in k and k-1 -> we have validated the object exists in these two frames
//                     //but not the points
//                     CHECK(obj_lmk_node->seenAtFrame(query_frame_node_k->frame_id));
//                     if(!obj_lmk_node->seenAtFrame(query_frame_node_k_1->frame_id)) {
//                         LOG(WARNING) << "Tracklet " << tracklet_id << " on object " << object_id << " seen at " << query_frame_node_k->frame_id << " but not " << query_frame_node_k_1->frame_id;
//                         break;
//                     } //this miay mean this this point never gets added?

//                     ss << query_frame_node_k_1->frame_id << " " << query_frame_node_k->frame_id << "\n";

//                     const Landmark& measured_k = obj_lmk_node->getMeasurement(query_frame_node_k);
//                     const Landmark& measured_k_1 = obj_lmk_node->getMeasurement(query_frame_node_k_1);

//                     //this should DEFINITELY be in the map, as long as we update the values in the map everyy time
//                     StateQuery<gtsam::Pose3> T_world_camera_k_1_query = query_frame_node_k_1->getPoseEstimate();
//                     CHECK(T_world_camera_k_1_query) << "Failed cam pose query at frame " << query_frame_node_k_1->frame_id
//                         << ". This may happen if the map_ is not updated every iteration OR something is wrong with the tracking...";
//                     const gtsam::Pose3 T_world_camera_k_1 = T_world_camera_k_1_query.get();

//                     StateQuery<gtsam::Pose3> T_world_camera_k_query = query_frame_node_k->getPoseEstimate();
//                     gtsam::Pose3 T_world_camera_k;
//                     //querying the map at the current frame ie. the last iteration in this loop as we havent added
//                     //the pose to the map yet!!
//                     if(!T_world_camera_k_query) {
//                         //this SHOULD happen when query_frame_node_k->frame_id == frame_id_k and at no other time
//                         CHECK_EQ(query_frame_node_k->frame_id, frame_id_k);
//                         //if this is okay, just set the pose to be the argument, since this should be the pose at frame_k
//                         T_world_camera_k = T_world_camera;
//                     }
//                     else {
//                         //we are not at the last frame of this loop yet so query_frame_node_k->frame_id < frame_id_k
//                         CHECK_LT(query_frame_node_k->frame_id, frame_id_k);
//                         T_world_camera_k = T_world_camera_k_query.get();
//                     }


//                     // const gtsam::Pose3 T_world_camera_k = T_world_camera_k_query.get();

//                     gtsam::Key object_point_key_k = obj_lmk_node->makeDynamicKey(query_frame_node_k->frame_id);
//                     gtsam::Key object_point_key_k_1 = obj_lmk_node->makeDynamicKey(query_frame_node_k_1->frame_id);

//                     //this assumes we add all the points in order and have continuous frames (which we should have?)
//                     if(seen_frames_itr == starting_motion_frame_itr) {
//                         //on first iteration we should add both values at query_k and query_k_1
//                         //otherwise we just need to add k, as k_1 will be added from the previous iteration
//                         new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                             query_frame_node_k_1->makePoseKey(), //pose key at previous frames
//                             object_point_key_k_1,
//                             measured_k_1,
//                             dynamic_point_noise_
//                         );
//                         object_debug_info.num_dynamic_factors++;

//                         const Landmark lmk_world_k_1 = T_world_camera_k_1 * measured_k_1;
//                         new_values.insert(object_point_key_k_1, lmk_world_k_1);
//                         timestamp_map_[object_point_key_k_1] = query_frame_node_k_1->frame_id;
//                         object_debug_info.num_new_dynamic_points++;
//                     }

//                     //previous point must be added by the previous iteration
//                     CHECK(new_values.exists(object_point_key_k_1));

//                     new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                         query_frame_node_k->makePoseKey(), //pose key at this (in the iteration) frames
//                         object_point_key_k,
//                         measured_k,
//                         dynamic_point_noise_
//                     );
//                     object_debug_info.num_dynamic_factors++;

//                     const Landmark lmk_world_k = T_world_camera_k * measured_k;
//                     new_values.insert(object_point_key_k, lmk_world_k);
//                     timestamp_map_[object_point_key_k] = query_frame_node_k->frame_id;
//                     object_debug_info.num_new_dynamic_points++;

//                     const gtsam::Key object_motion_key_k = query_frame_node_k->makeObjectMotionKey(object_id);
//                     //check in map as well, because this could be a new point on an old object, meaning the motion already exists
//                     if(!map_->exists(object_motion_key_k, new_values)) {
//                         //make new object motion
//                         new_values.insert(object_motion_key_k, gtsam::Pose3::Identity());
//                         //at time k or k-1 since a motion is between frames?
//                         timestamp_map_[object_motion_key_k] = query_frame_node_k->frame_id;
//                         LOG(INFO) << "Adding value " << DynoLikeKeyFormatterVerbose(object_motion_key_k);
//                     }

//                     new_point_factors.emplace_shared<LandmarkMotionTernaryFactor>(
//                         object_point_key_k_1,
//                         object_point_key_k,
//                         object_motion_key_k,
//                         landmark_motion_noise_
//                     );
//                     object_debug_info.num_motion_factors++;

//                     if(FLAGS_use_smoothing_factor) {
//                         //motion key at previous frame
//                         const gtsam::Symbol object_motion_key_k_1 = query_frame_node_k_1->makeObjectMotionKey(object_id);
//                         bool smoothing_added =
//                             safeAddObjectSmoothingFactor(object_motion_key_k_1, object_motion_key_k, new_values, new_point_factors);

//                         if(smoothing_added) {
//                             object_debug_info.smoothing_factor_added = true;
//                         }

//                     }


//                 }
//                 is_dynamic_tracklet_in_map_.insert2(tracklet_id, true);
//             }
//             else {
//                 //these tracklets should already be in the graph so we should only need to add the new measurements from this frame
//                 //check that we have previous point for this frame
//                 gtsam::Key object_point_key_k = obj_lmk_node->makeDynamicKey(frame_id_k);
//                 gtsam::Key object_point_key_k_1 = obj_lmk_node->makeDynamicKey(frame_id_k_1);

//                 const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(object_id);

//                 CHECK(map_->exists(object_point_key_k_1));
//                 CHECK(!map_->exists(object_point_key_k, new_values));

//                 const Landmark measured_k = obj_lmk_node->getMeasurement(frame_id_k);
//                 new_point_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//                     frame_node_k->makePoseKey(), //pose key at previous frames
//                     object_point_key_k,
//                     measured_k,
//                     dynamic_point_noise_
//                 );
//                 object_debug_info.num_dynamic_factors++;

//                 new_point_factors.emplace_shared<LandmarkMotionTernaryFactor>(
//                     object_point_key_k_1,
//                     object_point_key_k,
//                     object_motion_key_k,
//                     landmark_motion_noise_
//                 );
//                 object_debug_info.num_motion_factors++;

//                 const Landmark lmk_world_k = T_world_camera * measured_k;
//                 new_values.insert(object_point_key_k, lmk_world_k);
//                 timestamp_map_[object_point_key_k] = frame_id_k;
//                 object_debug_info.num_new_dynamic_points++;

//                 //should not have to check the map because this is an old point
//                 if(!new_values.exists(object_motion_key_k)) {
//                     //make new object motion -> this can happen if all the points on the object are well tracked from the previous motion
//                     new_values.insert(object_motion_key_k, gtsam::Pose3::Identity());
//                     timestamp_map_[object_motion_key_k] = frame_id_k;
//                     LOG(INFO) << "Adding value " << DynoLikeKeyFormatterVerbose(object_motion_key_k);
//                 }

//                 //
//                 if(FLAGS_use_smoothing_factor && frame_node_k_1->objectObserved(object_id)) {
//                     //motion key at previous frame
//                     const gtsam::Symbol object_motion_key_k_1 = frame_node_k_1->makeObjectMotionKey(object_id);
//                     bool smoothing_added =
//                         safeAddObjectSmoothingFactor(object_motion_key_k_1, object_motion_key_k, new_values, new_point_factors);

//                     if(smoothing_added) {
//                         object_debug_info.smoothing_factor_added = true;
//                     }

//                 }
//             }
//         }

//         debug_info_.object_info.insert2(object_id, object_debug_info);

//     }

//     for(const auto&[object_id, object_info] : debug_info_.object_info) {
//         std::stringstream ss;
//         ss << "Object id debug info: " << object_id << "\n";
//         ss << "Num point factors: " << object_info.num_dynamic_factors << "\n";
//         ss << "Num point variables: " << object_info.num_new_dynamic_points << "\n";
//         ss << "Num motion factors: " << object_info.num_motion_factors << "\n";
//         ss << "Smoothing factor added: " << std::boolalpha <<  object_info.smoothing_factor_added << "\n";

//         LOG(INFO) << ss.str();
//     }
// }


// void RGBDBackendModule::optimize(FrameId frame_id_k, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors) {
//     DynoISAM2UpdateParams isam_update_params;
//     // isam_update_params.

//     isam_update_params.setOrderingFunctions([=](const DynoISAM2UpdateParams& updateParams,
//                                         const DynoISAM2Result& result,
//                                         const gtsam::KeySet& affectedKeysSet,
//                                         const gtsam::VariableIndex& affectedFactorsVarIndex) -> gtsam::Ordering {
//         // affectedFactorsVarIndex.print("VariableIndex: ", DynoLikeKeyFormatterVerbose);
//         gtsam::FastMap<gtsam::Key, int> constrainedKeys;
//         std::vector<std::pair<FrameId, gtsam::Key>> ordered_dynamic_lmks;
//         std::set<FrameId> dynamic_lmk_frames_set;

//         //points, motion, pose (in index order)

//         for(const gtsam::Key key : affectedKeysSet) {
//             auto chr = DynoChrExtractor(key);

//             if(chr == kDynamicLandmarkSymbolChar) {
//                 auto frame_id = DynamicPointSymbol(key).frameId();
//                 dynamic_lmk_frames_set.insert(frame_id);
//                 ordered_dynamic_lmks.push_back(std::make_pair(frame_id, key));
//             }
//         }

//         int max_group = dynamic_lmk_frames_set.size();

//         gtsam::FastMap<FrameId, int> frame_ordering;

//         for(const auto& frame_key : ordered_dynamic_lmks) {
//             int index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_key.first));
//             // CHECK(index >= 0);
//             // //index tells use the order, starting at 0, with the first frame being what we want to have last in the ordering
//             // constrainedKeys.insert2(frame_key.second, index);
//             frame_ordering.insert2(frame_key.first, index);
//             // LOG(INFO) << "Adding group: " << index << " for key " << DynoLikeKeyFormatterVerbose(frame_key.second);
//         }

//         //last frame
//         const auto last_frame = frame_id_k;

//         //frame ordering
//         //frame_index * 3 (number of variabels we want to order; points, motion, pose) + offset
//         //ordering but by frame
//         // for(const gtsam::Key key : affectedKeysSet) {
//         //     auto chr = DynoChrExtractor(key);

//         //     int group = 0;

//         //     if(chr == kPoseSymbolChar) {
//         //         FrameId frame_id = gtsam::Symbol(key).index();
//         //         int frame_index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_id));
//         //         //2 offset for pose
//         //         group = frame_index * 3 + 2;
//         //     }
//         //     else if(chr == kObjectMotionSymbolChar) {
//         //         ObjectId object_label;
//         //         FrameId frame_id;
//         //         CHECK(reconstructMotionInfo(key, object_label, frame_id));
//         //         int frame_index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_id));
//         //         //1 offset for motion
//         //         group = frame_index * 3 + 0;
//         //     }
//         //     else if(chr == kDynamicLandmarkSymbolChar) {
//         //         auto frame_id = DynamicPointSymbol(key).frameId();
//         //         int frame_index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_id));
//         //         //0 offset for points
//         //         group = frame_index * 3 + 1;
//         //     }

//         //     LOG(INFO) << "Adding group: " << group << " for key " << DynoLikeKeyFormatterVerbose(key);
//         //     constrainedKeys.insert2(key, group);
//         // }

//         //dont assign static, so start groups at 1
//         //eliminate static, then dynamic points BUT NOT THE LAST POINT, then motions, then the last dynamic points, then poses
//         //IDEA: eliminate all the dynamic points before the motions (except the alst ones)
//         //so that we can write the motions in terms of the poses?
//         // for(const gtsam::Key key : affectedKeysSet) {
//         //     auto chr = DynoChrExtractor(key);

//         //     int group = 0;

//         //     if(chr == kPoseSymbolChar) {
//         //         FrameId frame_id = gtsam::Symbol(key).index();
//         //         if(frame_id == last_frame) {
//         //             group = 4; //put alst
//         //         }
//         //         else {
//         //             group = 0; //put with the other non-last dynamic points?
//         //         }
//         //     }
//         //     else if(chr == kObjectMotionSymbolChar) {
//         //         ObjectId object_label;
//         //         FrameId frame_id;
//         //         CHECK(reconstructMotionInfo(key, object_label, frame_id));
//         //         if(frame_id == last_frame) {
//         //             //I think this should be after the last dynamic point key...
//         //             group = 3; //put last
//         //         }
//         //         else {
//         //             //put as early as possible so we can try and marginalize it out!!
//         //             group = 0;
//         //         }

//         //     }
//         //     else if(chr == kDynamicLandmarkSymbolChar) {
//         //         auto frame_id = DynamicPointSymbol(key).frameId();

//         //         if(frame_id == last_frame) {
//         //             group = 4;
//         //         }
//         //         else {
//         //             group = 0;
//         //         }

//         //     }
//         //     // else if(chr == kStaticLandmarkSymbolChar) {


//         //     // }

//         //     LOG(INFO) << "Adding group: " << group << " for key " << DynoLikeKeyFormatterVerbose(key);
//         //     constrainedKeys.insert2(key, group);
//         // }



//         return Ordering::ColamdConstrained(affectedFactorsVarIndex, constrainedKeys);
//     });

//     //create an ordering putting motions last?
//     gtsam::FastMap<gtsam::Key, int> constrainedKeys;
//     //for each object

//     // for(const auto& [keys, value] : new_values) {
//     //     constrainedKeys.insert2(keys, 1);
//     // }

//     //very slow - find motion factors index's and mark them as newly affected keys
//     // gtsam::NonlinearFactorGraph old_graph = smoother_->getFactorsUnsafe();

//     // gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> affected_motion_factors;
//     // for(size_t i = 0; i < old_graph.size(); i++) {
//     //     auto factor = old_graph.at(i);
//     //     auto motion_factor = boost::dynamic_pointer_cast<LandmarkMotionTernaryFactor>(
//     //         factor;
//     //     );

//     //     if(motion_factor) {
//     //         //check if it contains a new key
//     //         if(motion_factor.find())
//     //     }

//     // }

//     //allows keys to be re-ordered
//     gtsam::FastList<gtsam::Key> extraReelimKeys;
//     // gtsam::KeyVector marginalize_keys;

//     // std::vector<std::pair<FrameId, gtsam::Key>> ordered_dynamic_lmks;
//     // std::set<FrameId> dynamic_lmk_frames_set;

//     // for(const gtsam::Key key : new_values.keys()) {
//     //     auto chr = DynoChrExtractor(key);

//     //     if(chr == kDynamicLandmarkSymbolChar) {
//     //         auto frame_id = DynamicPointSymbol(key).frameId();
//     //         dynamic_lmk_frames_set.insert(frame_id);
//     //         ordered_dynamic_lmks.push_back(std::make_pair(frame_id, key));
//     //     }
//     // }

//     // int max_group = dynamic_lmk_frames_set.size();

//     // for(const auto& frame_key : ordered_dynamic_lmks) {
//     //     int index = std::distance(dynamic_lmk_frames_set.begin(), dynamic_lmk_frames_set.find(frame_key.first));
//     //     CHECK(index >= 0);
//     //     //index tells use the order, starting at 0, with the first frame being what we want to have last in the ordering
//     //     constrainedKeys.insert2(frame_key.second, index);
//     //     LOG(INFO) << "Adding group: " << index << " for key " << DynoLikeKeyFormatterVerbose(frame_key.second);
//     // }

//     // for(const gtsam::Key key : new_values.keys()) {
//     //     auto chr = DynoChrExtractor(key);

//     //     if(chr == kPoseSymbolChar || chr == kObjectMotionSymbolChar) {
//     //         constrainedKeys.insert2(key, max_group);
//     //         LOG(INFO) << "Adding group: " << max_group << " for key " << DynoLikeKeyFormatterVerbose(key);
//     //     }
//     // }

//     // // //eliminate motions last? (group < dynamic key)
//     // const FrameNode3d::Ptr frame_node_k = map_->getFrame(frame_id_k);
//     // const FrameNode3d::Ptr frame_node_k_1 = map_->getFrame(frame_id_k - 1u);
//     // for(const ObjectNode3d::Ptr& object_node : frame_node_k->objects_seen) {
//     //     const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(object_node->getId());

//     //     if(new_values.exists(object_motion_key_k)) {
//     //         constrainedKeys[object_motion_key_k] = 1;
//     //     }

//     //     // if(frame_node_k_1) {
//     //     //     const gtsam::Key object_motion_key_k_1 = frame_node_k_1->makeObjectMotionKey(object_node->getId());
//     //     //     if(map_->exists(object_motion_key_k_1)) {
//     //     //         constrainedKeys.insert2(object_motion_key_k_1, 0);
//     //     //         marginalize_keys.push_back(object_motion_key_k_1);
//     //     //     }

//     //     // }
//     // }

//     // for(const auto& dyn_lmk_node : frame_node_k->dynamic_landmarks) {
//     //     const gtsam::Key key = dyn_lmk_node->makeDynamicKey(frame_node_k->getId());


//     //     //eliminate previous keys first
//     //     if(frame_id_k > 1) {
//     //         const gtsam::Key previous_key = dyn_lmk_node->makeDynamicKey(frame_node_k->getId() - 1);
//     //         // if(map_->exists(previous_key)) {
//     //             //should be in map but right now we dont have a way of checking if the keys have been deleted in the map
//     //             //once we get an estimate update from the smoother
//     //         // if(smoother_->valueExists(previous_key)) {
//     //         //     //have to put these keys are the start
//     //         //     constrainedKeys.insert2(previous_key, 0);
//     //         //     old_dynamic_points.push_back(previous_key);
//     //         //     // extraReelimKeys.push_back(previous_key);
//     //         // }
//     //     }


//     //     // if(new_values.exists(key)) {
//     //     //     constrainedKeys.insert2(key, 1);
//     //     // }
//     // }

//     // //further up the tree than the other two?
//     // constrainedKeys.insert2(CameraPoseSymbol(frame_id_k), 1);

//     // if(frame_id_k > 1) {
//     //     constrainedKeys.insert2(CameraPoseSymbol(frame_id_k -1), 0);
//     // }
//     // // gtsam::FastList<gtsam::Key> no_relin_keys;

//     // //make sure only the poses between the last motions are reliminated
//     // if((int)frame_id_k - 1 > 1) {
//     //     for(FrameId f = 0; f < frame_id_k - 1; f++) {
//     //         no_relin_keys.push_back(CameraPoseSymbol(f));
//     //     }
//     // }
//     // std::cout << "No relin keys: ";
//     // for (const auto key : no_relin_keys) {
//     //     std::cout << DynoLikeKeyFormatter(key) << " ";
//     // }
//     // std::cout << std::endl;


//     // Mark additional keys between the constrainted keys and the leaves
//     //from fixed lag smoother
//     std::function<void(const gtsam::Key&,const gtsam::ISAM2Clique::shared_ptr&, std::set<gtsam::Key>&)> recursiveMarkAffectedKeys =
//     [&recursiveMarkAffectedKeys](const gtsam::Key& key, const gtsam::ISAM2Clique::shared_ptr& clique, std::set<gtsam::Key>& additionalKeys) -> void{
//             // Check if the separator keys of the current clique contain the specified key
//         if (std::find(clique->conditional()->beginParents(),
//             clique->conditional()->endParents(), key)
//             != clique->conditional()->endParents()) {

//             // Mark the frontal keys of the current clique
//             for(gtsam::Key i: clique->conditional()->frontals()) {
//                 additionalKeys.insert(i);
//             }

//             // Recursively mark all of the children
//             for(const gtsam::ISAM2Clique::shared_ptr& child: clique->children) {
//                 recursiveMarkAffectedKeys(key, child, additionalKeys);
//             }
//         }
//         // If the key was not found in the separator/parents, then none of its children can have it either
//     };

//     //these keys should already be in the smoother and should now be at the leaves of the tree
//     // std::set<gtsam::Key> additionalKeys;
//     // for(gtsam::Key key : marginalize_keys) {
//     //     gtsam::ISAM2Clique::shared_ptr clique = smoother_->operator[](key);
//     //     for(const gtsam::ISAM2Clique::shared_ptr& child: clique->children) {
//     //         recursiveMarkAffectedKeys(key, child, additionalKeys);
//     //     }
//     // }
//     // gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(), additionalKeys.end());

//     // isam_update_params.constrainedKeys = constrainedKeys;
//     // isam_update_params.extraReelimKeys = additionalMarkedKeys;
//     // isam_update_params.
//     // isam_update_params.noRelinKeys = no_relin_keys;

//     LOG(INFO) << "Starting optimization for " << frame_id_k;

//     gtsam::Values old_state = map_->getValues();
//     gtsam::Values all_values = old_state;
//     all_values.insert_or_assign(new_values);

//     try {
//         // gtsam::IncrementalFixedLagSmoother::Result fl_result = smoother_->update(new_factors, new_values, timestamp_map_);
//         // gtsam::ISAM2Result result = smoother_->getISAM2Result();
//         utils::TimingStatsCollector("backend.update");
//         smoother_result_ = smoother_->update(new_factors, new_values, isam_update_params);
//         // result = smoother_->update();
//         // result = smoother_->update();

//         // smoother_->print("isam2 ", DynoLikeKeyFormatter);
//         smoother_result_.print();
//         LOG(INFO) << "Num total vars: " << all_values.size() << " num new vars " << new_values.size();
//         //keys that were part of the bayes tree that is removed before conversion to a factor graph
//         LOG(INFO) << "Num marked keys: " << smoother_result_.markedKeys.size();
//         LOG(INFO) << "Num total factors: " << smoother_->getFactorsUnsafe().size() << " num new factors " << new_factors.size();
//         // LOG(INFO) << "Num total factors: " << smoother_->getFactors().size() << " num new factors " << new_factors.size();

//         // auto detailed_results = result.details();
//         // if(detailed_results) {
//         //     std::stringstream ss;
//         //     ss << "Keys re-eliminated: ";
//         //     for(const auto& [key, variable_status] : detailed_results->variableStatus) {
//         //         if(variable_status.isReeliminated) {
//         //             ss << DynoLikeKeyFormatter(key) << " ";
//         //         }
//         //     }

//         //     LOG(INFO) << ss.str();
//         // }

//         // bool did_batch = result.
//     }
//     catch(const gtsam::ValuesKeyAlreadyExists& e) {
//         new_factors.saveGraph(getOutputFilePath("isam2_graph.dot"), DynoLikeKeyFormatter);
//         LOG(FATAL) << "gtsam::ValuesKeyAlreadyExists thrown near key " << DynoLikeKeyFormatterVerbose(e.key());
//     }
//     catch(const gtsam::ValuesKeyDoesNotExist& e) {
//         new_factors.saveGraph(getOutputFilePath("isam2_graph.dot"), DynoLikeKeyFormatter);
//         LOG(FATAL) << "gtsam::ValuesKeyDoesNotExist thrown near key " << DynoLikeKeyFormatterVerbose(e.key());
//     }

//     timestamp_map_.clear();

//     gtsam::KeySet marginalize_keys = factor_graph_tools::travsersal::getLeafKeys(*smoother_);
//     gtsam::PrintKeySet(marginalize_keys, "Marginalize keys", dyno::DynoLikeKeyFormatter);


//     gtsam::Values estimate = smoother_->calculateBestEstimate();
//     gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();

//     LOG(INFO) << "Starting marginalization...";
//     // Marginalize out any needed variables
//     //only works in certain ordering situations.....
//     //does marginalization reuslt in null ptrs in the graph?
//     if (marginalize_keys.size() > 1) {
//         gtsam::FastList<gtsam::Key> leafKeys(marginalize_keys.begin(),
//             marginalize_keys.end());
//         //remove poses?
//         auto is_pose = [](auto key) {
//             auto chr = DynoChrExtractor(key);
//             return chr == kPoseSymbolChar || chr == kStaticLandmarkSymbolChar;
//             // return false;
//         };
//         // leafKeys.erase(std::remove_if(leafKeys.begin(), leafKeys.end(), is_pose), leafKeys.end());

//         // smoother_->marginalizeLeaves(leafKeys);
//     }

//     LOG(INFO) << "finished marginalization...";

//     // gtsam::Values estimate = smoother_->calculateBestEstimate();
//     // gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();
//     // gtsam::Values all_values = smoother_->

//     double error_before = graph.error(all_values);
//     double error_after = graph.error(estimate);

//     LOG(INFO) << "Optimization Errors:\n"
//              << " - Error before :" << error_before
//              << '\n'
//              << " - Error after  :" << error_after;

//     map_->updateEstimates(estimate, graph, frame_id_k);
//     // map_->updateEstimates(all_values, graph, frame_id_k);
// }

void RGBDBackendModule::UpdateImpl::setInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    auto initial_pose_prior = parent_->initial_pose_prior_;

    new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
    new_factors.addPrior(CameraPoseSymbol(frame_id_k), T_world_camera, initial_pose_prior);
}


void RGBDBackendModule::UpdateImpl::updateOdometry(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, FrameId frame_id_k_1, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    new_values.insert(CameraPoseSymbol(frame_id_k), T_world_camera);
    auto odometry_noise = parent_->odometry_noise_;

    auto map = parent_->getMap();
    StateQuery<gtsam::Pose3> pose_query = map->getPoseEstimate(frame_id_k_1);
    if(pose_query) {
        LOG(INFO) << "Adding odom between " << frame_id_k_1 << " and " << frame_id_k;
        const gtsam::Pose3 T_world_camera_k_1 = pose_query.get();
        const gtsam::Pose3 odom = T_world_camera_k_1.inverse() * T_world_camera;

        factor_graph_tools::addBetweenFactor(frame_id_k_1, frame_id_k, odom, odometry_noise, new_factors);
    }
}



void RGBDBackendModule::UpdateImplInWorld::updateStaticObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info) {
    auto map = parent_->getMap();
    const auto& params = parent_->base_params_;
    const FrameNode3d::Ptr frame_node_k = map->getFrame(frame_id_k);
    CHECK_NOTNULL(frame_node_k);

    auto static_point_noise = parent_->static_point_noise_;

    LOG(INFO) << "Looping over " <<  frame_node_k->static_landmarks.size() << " static lmks for frame " << frame_id_k;
    for(const LandmarkNode3d::Ptr& lmk_node : frame_node_k->static_landmarks) {


        const gtsam::Key point_key = lmk_node->makeStaticKey();
        //check if lmk node is already in map (which should mean it is equivalently in isam)
        if(map->exists(point_key)) {
            //3d point in camera frame
            const Landmark& measured = lmk_node->getMeasurement(frame_id_k);
            new_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                frame_node_k->makePoseKey(), //pose key for this frame
                point_key,
                measured,
                static_point_noise
            );

            if(debug_info) debug_info->num_static_factors++;

        }
        else {
            //see if we have enough observations to add this lmk
            if(lmk_node->numObservations() < params.min_static_obs_) { continue;}

            // num_points++;

            // if(num_points > 5) {break;}

            //this condition should only run once per tracklet (ie.e the first time the tracklet has enough observations)
            //we gather the tracklet observations and then initalise it in the new values
            //these should then get added to the map
            //and map_->exists() should return true for all other times
            FrameNodePtrSet<Landmark> seen_frames = lmk_node->getSeenFrames();
            for(const FrameNode3d::Ptr& seen_frame : seen_frames) {
                //check for sanity that the seen frames are <= than the current frame
                CHECK_LE(seen_frame->getId(), frame_id_k);
                const Landmark& measured = lmk_node->getMeasurement(seen_frame);
                new_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                    seen_frame->makePoseKey(), //pose key at previous frames
                    point_key,
                    measured,
                    static_point_noise
                );
            }

            //add initial value
            //pick the one in this frame
            const Landmark& measured = lmk_node->getMeasurement(frame_id_k);
            const Landmark lmk_world = T_world_camera * measured;
            new_values.insert(point_key, lmk_world);
            // timestamp_map_[point_key] = frame_id_k;

            if(debug_info) debug_info->num_new_static_points++;

        }
    }

    if(debug_info) LOG(INFO) << "Num new static points: " << debug_info->num_new_static_points << "\n" << "Num new static factors " << debug_info->num_static_factors;
}

void RGBDBackendModule::UpdateImplInWorld::updateDynamicObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info) {
    auto map = parent_->getMap();
    const auto& params = parent_->base_params_;
    const FrameId frame_id_k_1 = frame_id_k - 1u;

    const FrameNode3d::Ptr frame_node_k = map->getFrame(frame_id_k);
    const FrameNode3d::Ptr frame_node_k_1 = map->getFrame(frame_id_k_1);
    CHECK_NOTNULL(frame_node_k_1);

    //collect noise models to be used
    auto landmark_motion_noise = parent_->landmark_motion_noise_;
    auto dynamic_point_noise = parent_->dynamic_point_noise_;

    //! At least three points on the object are required to solve
    //! otherise the system is indeterminate
    constexpr static size_t kMinNumberPoints = 3u;

    //! get reference to dynamic tracklet map
    auto& is_dynamic_tracklet_in_map = parent_->is_dynamic_tracklet_in_map_;

    //for each object
    for(const ObjectNode3d::Ptr& object_node : frame_node_k->objects_seen) {

        DebugInfo::ObjectInfo object_debug_info;

        const ObjectId object_id = object_node->getId();
        //which frames object measurements will be added as factors to the map
        std::set<FrameId> frames_added;

        //first check that object exists in the previous frame
        if(!frame_node_k->objectMotionExpected(object_id)) {
           continue;
        }

        // possibly the longest call?
        // landmarks on this object seen at frame k
        LandmarkNodePtrSet<Landmark> seen_lmks_k = object_node->getLandmarksSeenAtFrame(frame_id_k);

        //if we dont have at least 3 observations of this object in this frame AND the previous frame
        if(seen_lmks_k.size() < kMinNumberPoints || object_node->getLandmarksSeenAtFrame(frame_id_k_1).size() < kMinNumberPoints) {
            continue;
        }

        //TODO: debug
        int num_lmks = 0;


        //iterate over each lmk we have on this object
        for(const LandmarkNode3d::Ptr& obj_lmk_node : seen_lmks_k) {
            CHECK_EQ(obj_lmk_node->getObjectId(), object_id);

            //see if we have enough observations to add this lmk
            if(obj_lmk_node->numObservations() < params.min_dynamic_obs_) { continue;}

            num_lmks++;

            // if(num_lmks > 5) {
            //     break;
            // }

            TrackletId tracklet_id = obj_lmk_node->getId();

            //if does not exist, we need to go back and all the previous measurements & factors & motions
            if(!is_dynamic_tracklet_in_map.exists(tracklet_id)) {
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

                //iterate over k-N to k (inclusive) and all all
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

                    //this should DEFINITELY be in the map, as long as we update the values in the map everyy time
                    StateQuery<gtsam::Pose3> T_world_camera_k_1_query = query_frame_node_k_1->getPoseEstimate();
                    CHECK(T_world_camera_k_1_query) << "Failed cam pose query at frame " << query_frame_node_k_1->frame_id
                        << ". This may happen if the map_ is not updated every iteration OR something is wrong with the tracking...";
                    const gtsam::Pose3 T_world_camera_k_1 = T_world_camera_k_1_query.get();

                    StateQuery<gtsam::Pose3> T_world_camera_k_query = query_frame_node_k->getPoseEstimate();
                    gtsam::Pose3 T_world_camera_k;
                    //querying the map at the current frame ie. the last iteration in this loop as we havent added
                    //the pose to the map yet!!
                    if(!T_world_camera_k_query) {
                        //this SHOULD happen when query_frame_node_k->frame_id == frame_id_k and at no other time
                        CHECK_EQ(query_frame_node_k->frame_id, frame_id_k);
                        //if this is okay, just set the pose to be the argument, since this should be the pose at frame_k
                        T_world_camera_k = T_world_camera;
                    }
                    else {
                        //we are not at the last frame of this loop yet so query_frame_node_k->frame_id < frame_id_k
                        CHECK_LT(query_frame_node_k->frame_id, frame_id_k);
                        T_world_camera_k = T_world_camera_k_query.get();
                    }


                    // const gtsam::Pose3 T_world_camera_k = T_world_camera_k_query.get();

                    gtsam::Key object_point_key_k = obj_lmk_node->makeDynamicKey(query_frame_node_k->frame_id);
                    gtsam::Key object_point_key_k_1 = obj_lmk_node->makeDynamicKey(query_frame_node_k_1->frame_id);

                    //this assumes we add all the points in order and have continuous frames (which we should have?)
                    if(seen_frames_itr == starting_motion_frame_itr) {
                        //on first iteration we should add both values at query_k and query_k_1
                        //otherwise we just need to add k, as k_1 will be added from the previous iteration
                        new_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                            query_frame_node_k_1->makePoseKey(), //pose key at previous frames
                            object_point_key_k_1,
                            measured_k_1,
                            dynamic_point_noise
                        );
                        object_debug_info.num_dynamic_factors++;
                        //update frames added with previous frame
                        //very important to capture the first frame the object is seen
                        frames_added.insert(query_frame_node_k_1->frame_id);

                        const Landmark lmk_world_k_1 = T_world_camera_k_1 * measured_k_1;
                        new_values.insert(object_point_key_k_1, lmk_world_k_1);
                        object_debug_info.num_new_dynamic_points++;
                    }

                    //previous point must be added by the previous iteration
                    CHECK(new_values.exists(object_point_key_k_1));

                    new_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                        query_frame_node_k->makePoseKey(), //pose key at this (in the iteration) frames
                        object_point_key_k,
                        measured_k,
                        dynamic_point_noise
                    );
                    object_debug_info.num_dynamic_factors++;
                    frames_added.insert(query_frame_node_k->frame_id);

                    const Landmark lmk_world_k = T_world_camera_k * measured_k;
                    new_values.insert(object_point_key_k, lmk_world_k);
                    object_debug_info.num_new_dynamic_points++;

                    const gtsam::Key object_motion_key_k = query_frame_node_k->makeObjectMotionKey(object_id);
                    //check in map as well, because this could be a new point on an old object, meaning the motion already exists
                    if(!map->exists(object_motion_key_k, new_values)) {
                        //make new object motion
                        new_values.insert(object_motion_key_k, gtsam::Pose3::Identity());
                        //at time k or k-1 since a motion is between frames?
                        // timestamp_map_[object_motion_key_k] = query_frame_node_k->frame_id;
                        LOG(INFO) << "Adding value " << DynoLikeKeyFormatterVerbose(object_motion_key_k);
                    }

                    new_factors.emplace_shared<LandmarkMotionTernaryFactor>(
                        object_point_key_k_1,
                        object_point_key_k,
                        object_motion_key_k,
                        landmark_motion_noise
                    );
                    object_debug_info.num_motion_factors++;

                    if(FLAGS_use_smoothing_factor) {
                        //motion key at previous frame
                        const gtsam::Symbol object_motion_key_k_1 = query_frame_node_k_1->makeObjectMotionKey(object_id);
                        bool smoothing_added =
                            parent_->safeAddObjectSmoothingFactor(object_motion_key_k_1, object_motion_key_k, new_values, new_factors);

                        if(smoothing_added) {
                            object_debug_info.smoothing_factor_added = true;
                        }

                    }


                }
                is_dynamic_tracklet_in_map.insert2(tracklet_id, true);
            }
            else {
                //these tracklets should already be in the graph so we should only need to add the new measurements from this frame
                //check that we have previous point for this frame
                gtsam::Key object_point_key_k = obj_lmk_node->makeDynamicKey(frame_id_k);
                gtsam::Key object_point_key_k_1 = obj_lmk_node->makeDynamicKey(frame_id_k_1);

                const gtsam::Key object_motion_key_k = frame_node_k->makeObjectMotionKey(object_id);

                CHECK(map->exists(object_point_key_k_1));
                CHECK(!map->exists(object_point_key_k, new_values));

                const Landmark measured_k = obj_lmk_node->getMeasurement(frame_id_k);
                new_factors.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
                    frame_node_k->makePoseKey(), //pose key at previous frames
                    object_point_key_k,
                    measured_k,
                    dynamic_point_noise
                );
                object_debug_info.num_dynamic_factors++;
                frames_added.insert(frame_node_k->frame_id);

                new_factors.emplace_shared<LandmarkMotionTernaryFactor>(
                    object_point_key_k_1,
                    object_point_key_k,
                    object_motion_key_k,
                    landmark_motion_noise
                );
                object_debug_info.num_motion_factors++;

                const Landmark lmk_world_k = T_world_camera * measured_k;
                new_values.insert(object_point_key_k, lmk_world_k);
                // timestamp_map_[object_point_key_k] = frame_id_k;
                object_debug_info.num_new_dynamic_points++;

                //should not have to check the map because this is an old point
                if(!new_values.exists(object_motion_key_k)) {
                    //make new object motion -> this can happen if all the points on the object are well tracked from the previous motion
                    new_values.insert(object_motion_key_k, gtsam::Pose3::Identity());
                    // timestamp_map_[object_motion_key_k] = frame_id_k;
                    LOG(INFO) << "Adding value " << DynoLikeKeyFormatterVerbose(object_motion_key_k);
                }

                //
                if(FLAGS_use_smoothing_factor && frame_node_k_1->objectObserved(object_id)) {
                    //motion key at previous frame
                    const gtsam::Symbol object_motion_key_k_1 = frame_node_k_1->makeObjectMotionKey(object_id);
                    bool smoothing_added =
                        parent_->safeAddObjectSmoothingFactor(object_motion_key_k_1, object_motion_key_k, new_values, new_factors);

                    if(smoothing_added) {
                        object_debug_info.smoothing_factor_added = true;
                    }

                }
            }
        }

        if(debug_info) debug_info->object_info.insert2(object_id, object_debug_info);

        if(frames_added.size() > 0) {
            //update internal datastructure indicating for which frames new factors/values were added
            parent_->new_object_measurements_per_frame_.insert2(object_id, frames_added);
        }

    }


    if(debug_info) {
        for(const auto&[object_id, object_info] : debug_info->object_info) {
            std::stringstream ss;
            ss << "Object id debug info: " << object_id << "\n";
            ss << "Num point factors: " << object_info.num_dynamic_factors << "\n";
            ss << "Num point variables: " << object_info.num_new_dynamic_points << "\n";
            ss << "Num motion factors: " << object_info.num_motion_factors << "\n";
            ss << "Smoothing factor added: " << std::boolalpha <<  object_info.smoothing_factor_added << "\n";

            LOG(INFO) << ss.str();
        }
    }
}

void RGBDBackendModule::UpdateImplInWorld::logBackendFromMap(FrameId frame_k, RGBDMap::Ptr map, BackendLogger& logger) {
    const auto& gt_packet_map = parent_->gt_packet_map_;

    const ObjectIds object_ids = map->getObjectIds();
    ObjectPoseMap composed_object_pose_map;

    for(auto object_id : object_ids) {
        const auto& object_node = map->getObject(object_id);
        composed_object_pose_map.insert2(
            //composed poses
            object_id, object_node->computeComposedPoseMap(gt_packet_map, FLAGS_init_object_pose_from_gt));
    }

    //get MotionestimateMap
    const MotionEstimateMap motions = map->getMotionEstimates(frame_k);
    logger.logObjectMotion(gt_packet_map, frame_k, motions);

    StateQuery<gtsam::Pose3> X_k_query = map->getPoseEstimate(frame_k);

    if(X_k_query) {
        logger.logCameraPose(gt_packet_map, frame_k, X_k_query.get());
    }
    else {
        LOG(WARNING) << "Could not log camera pose estimate at frame " << frame_k;
    }


    logger.logObjectPose(gt_packet_map, frame_k, composed_object_pose_map);
}


void RGBDBackendModule::UpdateImplInWorldPrimitives::updateDynamicObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info) {
    UpdateImplInWorld::updateDynamicObservations(frame_id_k, T_world_camera, new_values, new_factors, debug_info);

    auto map = parent_->getMap();

    //update with new_values from the base function
    //TODO: bit of a hack as this then gets called outside this class
    //but it means all the iniitail lmk values get put into the map and we can query them
    //dont want to (re)set the graph in the map so just replace it with the existing graph
    map->updateEstimates(new_values, map->getGraph(), frame_id_k);

    const auto& params = parent_->base_params_;
    const FrameId frame_id_k_1 = frame_id_k - 1u;
    const FrameNode3d::Ptr frame_node_k = map->getFrame(frame_id_k);

    //for each object in this frame
    //for each object
    for(const ObjectNode3d::Ptr& object_node : frame_node_k->objects_seen) {
        const ObjectId object_id = object_node->getId();
        LOG(INFO) << "Object pose/quadric update for " << object_id;
        const gtsam::Symbol quadric_sym = ObjectQuadricSymbol(object_id);
        // CHECK(parent_->new_object_measurements_per_frame_.exists(object_id));
        //if no measurements added, skip
        if(!parent_->new_object_measurements_per_frame_.exists(object_id)) {
            LOG(INFO) << "Skipping object pose update " << object_id;
            continue;
        }
        //need to check if we have already added poses to this object
        //this will occur since we have a min number of frames an object is seen
        //before going back to add all the object points
        //we need to make sure we go back and all the poses for these points
        const std::set<FrameId> new_measurement_frames = parent_->new_object_measurements_per_frame_.at(object_id);
        CHECK(!new_measurement_frames.empty());
        //check this frame is in the list of measurements
        CHECK(new_measurement_frames.find(frame_id_k) != new_measurement_frames.end());

        std::set<FrameId> frames_to_add_for;

        for(auto itr = new_measurement_frames.begin(); itr != new_measurement_frames.end(); itr++) {
            const FrameId frame_id = *itr;
            const FrameNode3d::Ptr frame_node = map->getFrame(frame_id);
            CHECK(frame_node);
            //check there is no inconsistencies and that this object is seen at this frame in the map
            CHECK(frame_node->objectObserved(object_id));

            // //check if new object -> initalise P
            // //lots of ways to do this - we could check for the existance of P in the values
            const FrameId current_frame_id = frame_id;
            // const gtsam::Key current_pose_key = ObjectPoseSymbol(object_id, current_frame_id);

            if(current_frame_id < 1) {continue;}


            //firstly we check for a motion between k-1 and k
            //the map may not have the motion yet so we need to check in the new_values as well
            if(object_node->hasMotionEstimate(current_frame_id) || new_values.exists(ObjectMotionSymbol(object_id, current_frame_id))) {
                frames_to_add_for.insert(current_frame_id);

                const FrameId previous_frame_id = frame_id - 1u;
                const FrameNode3d::Ptr previous_frame_node = map->getFrame(previous_frame_id);
                if(previous_frame_node) {
                    //check there is no inconsistencies and that this object is seen at this frame in the map
                    CHECK(previous_frame_node->objectObserved(object_id));
                    frames_to_add_for.insert(previous_frame_id);
                }
            }
        }

        std::stringstream ss;
        LOG(INFO) << "new_measurement_frames " << container_to_string(new_measurement_frames);
        //TODO: this is wrong - we seem to have extra frames!!!?
        LOG(INFO) << "frames_to_add_for " << container_to_string(frames_to_add_for);

        for(FrameId frame_id : frames_to_add_for) {

            const gtsam::Key object_pose_key = ObjectPoseSymbol(object_id, frame_id);
            ss << "Adding object pose " << DynoLikeKeyFormatter(object_pose_key) << " at frame " << frame_id;
            if(map->exists(object_pose_key, new_values)) {
                //skip if pose (and factors) has already been added
                continue;
            }

            //TODO:
            //for now
            {
                const FrameNode3d::Ptr frame_node = map->getFrame(frame_id);
                const StatusLandmarkEstimates& dynamic_lmks = frame_node->getDynamicLandmarkEstimates(object_id);
                //convert to point cloud -> should be a map with only one map in it
                CloudPerObject object_clouds = groupObjectCloud(
                    dynamic_lmks,
                    map->at<gtsam::Pose3>(frame_node->makePoseKey(), new_values)
                );

                if(object_clouds.size() == 0) {
                    LOG(FATAL) << "Cannot object clouds from dynamic landmarks of " << object_id << " and frame " << frame_id << "!! "
                        << " # Dynamic lmks in the map for this object at this frame was " << dynamic_lmks.size();
                }
                CHECK_EQ(object_clouds.size(), 1);
                CHECK(object_clouds.exists(object_id));

                const auto dynamic_point_cloud = object_clouds.at(object_id);
                pcl::PointXYZ centroid;
                pcl::computeCentroid(dynamic_point_cloud, centroid);

                gtsam::Point3 translation = pclPointToGtsam(centroid);
                new_values.insert(object_pose_key, gtsam::Pose3(gtsam::Rot3::Identity(), translation));

                if(object_node->getFirstSeenFrame() == frame_id) {
                    //first time the object is seen
                    //sanity check -> Parmaeters should not exist yet!!
                    CHECK(!map->exists(quadric_sym, new_values));
                    new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                        object_pose_key,
                        new_values.at<gtsam::Pose3>(object_pose_key),
                        gtsam::noiseModel::Isotropic::Sigma(6u, 0.05)
                    );
                }

            }

            int num_quadric_factors = 0;

            //also assume we need to add all the factors for the points already in the map!
            const LandmarkNodePtrSet<Landmark> seen_lmks = object_node->getLandmarksSeenAtFrame(frame_id);
            for(const LandmarkNode3d::Ptr& obj_lmk_node : seen_lmks) {
                gtsam::Key object_point_key = obj_lmk_node->makeDynamicKey(frame_id);
                if(map->exists(object_point_key, new_values)) {
                    new_factors.emplace_shared<LandmarkQuadricFactor>(
                        object_point_key,
                        object_pose_key,
                        quadric_sym,
                        gtsam::noiseModel::Isotropic::Sigma(1u, 1.5)
                    );


                    num_quadric_factors++;
                }
            }

            ss << " and " << num_quadric_factors << " quadric factors\n";

            // add OKF
            if(frame_id > 0) {
                const gtsam::Key previous_object_pose_key = ObjectPoseSymbol(object_id, frame_id-1);
                const gtsam::Key object_motion_key = ObjectMotionSymbol(object_id, frame_id);
                if(map->exists(previous_object_pose_key, new_values) && map->exists(object_motion_key, new_values)) {
                    ss << "Adding OKF for  " << object_id << " at frame " << frame_id;
                    new_factors.emplace_shared<ObjectKinematicFactor>(
                            object_motion_key,
                            previous_object_pose_key,
                            object_pose_key,
                            gtsam::noiseModel::Isotropic::Sigma(6u, 0.01)
                        );

                }
            }
        }

        //add P if this object is new
        if(!map->exists(quadric_sym, new_values)) {
            //TODO:
            new_values.insert(quadric_sym, gtsam::Vector3(1, 1, 1));
        }

        LOG(INFO) << ss.str();

    }
    //check if motion exists between k and k-1 -> if exists initalise L and k-1 and/or k
    //for each point add LandmarkQuadric factor
}

void RGBDBackendModule::UpdateImplInWorldPrimitives::logBackendFromMap(FrameId frame_k, RGBDMap::Ptr map, BackendLogger& logger) {
    //TODO: so much repeated code
    const auto& gt_packet_map = parent_->gt_packet_map_;

    const ObjectIds object_ids = map->getObjectIds();
    ObjectPoseMap estimated_object_pose_map;

    for(auto object_id : object_ids) {
        //TODO: only get objects that are in this frame!!!?
        const auto& object_node = map->getObject(object_id);
        estimated_object_pose_map.insert2(
            //composed poses
            object_id, object_node->computeEstimatedPoseMap());
    }

    //get MotionestimateMap
    const MotionEstimateMap motions = map->getMotionEstimates(frame_k);
    logger.logObjectMotion(gt_packet_map, frame_k, motions);

    StateQuery<gtsam::Pose3> X_k_query = map->getPoseEstimate(frame_k);

    if(X_k_query) {
        logger.logCameraPose(gt_packet_map, frame_k, X_k_query.get());
    }
    else {
        LOG(WARNING) << "Could not log camera pose estimate at frame " << frame_k;
    }


    logger.logObjectPose(gt_packet_map, frame_k, estimated_object_pose_map);
}


// void RGBDBackendModule::UpdateImplInWorldPrimitives::updateDynamicObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info) {



//TODO: these functions can go in base
void RGBDBackendModule::saveGraph(const std::string& file) {

    //TODO: must be careful as there could be inconsistencies between the graph in the optimzier,
    //and the graph in the map
    gtsam::NonlinearFactorGraph graph = map_->getGraph();
    // gtsam::NonlinearFactorGraph graph = smoother_->getFactorsUnsafe();
    graph.saveGraph(getOutputFilePath(file), DynoLikeKeyFormatter);
}
void RGBDBackendModule::saveTree(const std::string& file) {
    auto incremental_optimizer = safeCast<Optimizer<Landmark>, IncrementalOptimizer<Landmark>>(optimizer_);
    if(incremental_optimizer) {
        auto smoother = incremental_optimizer->getSmoother();
        smoother.saveGraph(getOutputFilePath(file), DynoLikeKeyFormatter);
        // smoother.getISAM2().saveGraph(getOutputFilePath(file), DynoLikeKeyFormatter);
    }
}



} //dyno
