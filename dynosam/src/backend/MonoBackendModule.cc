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

#include "dynosam/backend/MonoBackendModule.hpp"
#include "dynosam/backend/MonoBackendTools.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/frontend/MonoInstance-Definitions.hpp"

#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"

#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/utils/Numerical.hpp"

#include <gtsam/inference/Key.h>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>

DEFINE_bool(run_as_graph_file_only, true, "If true values will be saved to a graph file to for unit testing.");
DEFINE_string(backend_graph_file, "/root/results/DynoSAM/mono_backend_graph.g2o", "Path to write graph file to");

namespace dyno {

MonoBackendModule::MonoBackendModule(const BackendParams& backend_params, Camera::Ptr camera)
    :   BackendModule(backend_params),
        camera_(CHECK_NOTNULL(camera))
{

    const auto& camera_params = camera_->getParams();
    gtsam_calibration_ = boost::make_shared<Camera::CalibrationType>(camera_params.constructGtsamCalibration<Camera::CalibrationType>());

    CHECK(gtsam_calibration_);
    setFactorParams(backend_params);

    gtsam::ISAM2Params isam_params;
    isam_params.findUnusedFactorSlots = false; //this is very important rn as we naively keep track of slots
    isam_params.relinearizeThreshold = 0.01;

    smoother_ = std::make_unique<gtsam::ISAM2>(isam_params);
}

MonoBackendModule::~MonoBackendModule() {
    // std::ofstream graph_logger;
    // LOG(INFO) << "Writing to " << FLAGS_backend_graph_file;
    // graph_logger.open(FLAGS_backend_graph_file);
    // graph_logger << fg_ss_.str();
    // graph_logger.close();
    // state_graph_.saveGraph("/root/results/DynoSAM/mono_backend_graph.dot");
    smoother_->saveGraph("/root/results/DynoSAM/mono_backend_isam_graph.dot");
}


MonoBackendModule::SpinReturn MonoBackendModule::boostrapSpin(BackendInputPacket::ConstPtr input) {
    MonocularInstanceOutputPacket::ConstPtr mono_output = safeCast<BackendInputPacket, MonocularInstanceOutputPacket>(input);
    checkAndThrow((bool)mono_output, "Failed to cast BackendInputPacket to MonocularInstanceOutputPacket in MonoBackendModule");

    return monoBoostrapSpin(mono_output);

}
MonoBackendModule::SpinReturn MonoBackendModule::nominalSpin(BackendInputPacket::ConstPtr input) {
    MonocularInstanceOutputPacket::ConstPtr mono_output = safeCast<BackendInputPacket, MonocularInstanceOutputPacket>(input);
    checkAndThrow((bool)mono_output, "Failed to cast BackendInputPacket to MonocularInstanceOutputPacket in MonoBackendModule");

    return monoNominalSpin(mono_output);
}


MonoBackendModule::SpinReturn MonoBackendModule::monoBoostrapSpin(MonocularInstanceOutputPacket::ConstPtr input) {

    //1. Triangulate initial map (static)
    //2. Triangulation function for object points (requires two frames min)
    CHECK(input);
    //check cameras the same?
    TrackletIds new_smart_factors;
    TrackletIds updated_smart_factors;
    StatusKeypointMeasurements new_projection_measurements;

    const gtsam::Pose3 T_world_camera_measured = input->T_world_camera_;
    const gtsam::Pose3 T_world_camera_gt = input->gt_packet_->X_world_;
    const FrameId current_frame_id =  input->getFrameId();

    if(FLAGS_run_as_graph_file_only) {
        buildGraphWithDepth(input);
        // return {State::Nominal, nullptr};



    }

    // gtsam::Values new_values;
    // gtsam::NonlinearFactorGraph new_factors;

    // addInitialPose(T_world_camera_gt, current_frame_id, new_values, new_factors);
    // updateSmartStaticObservations(input->static_keypoint_measurements_, current_frame_id, new_smart_factors, updated_smart_factors, new_projection_measurements);

    // optimize(current_frame_id, new_smart_factors, TrackletIds{}, TrackletIds{}, new_values, new_factors);

    return {State::Nominal, nullptr};
}

MonoBackendModule::SpinReturn MonoBackendModule::monoNominalSpin(MonocularInstanceOutputPacket::ConstPtr input) {
    CHECK(input);

    TrackletIds new_smart_factors;
    TrackletIds updated_smart_factors;
    StatusKeypointMeasurements new_projection_measurements;

    const gtsam::Pose3 T_world_camera_measured = input->T_world_camera_;

    const gtsam::Pose3 T_world_camera_gt = input->gt_packet_->X_world_;

    const FrameId current_frame_id =  input->getFrameId();
    LOG(INFO) << "Running backend on frame " << current_frame_id;

    if(FLAGS_run_as_graph_file_only) {
        buildGraphWithDepth(input);
        // return {State::Nominal, nullptr};
    }

    // gtsam::Values new_values;
    // gtsam::NonlinearFactorGraph new_factors;

    // addOdometry(T_world_camera_gt, current_frame_id, current_frame_id-1, new_values, new_factors);
    // updateSmartStaticObservations(input->static_keypoint_measurements_, current_frame_id, new_smart_factors, updated_smart_factors, new_projection_measurements);

    // gtsam::Values new_and_current_values(state_);
    // new_and_current_values.insert(new_values);

    // TrackletIds tracklets_triangulated;
    // TrackletIds only_updated_smart_factors;

    // tryTriangulateExistingSmartStaticFactors(updated_smart_factors, new_and_current_values, tracklets_triangulated, only_updated_smart_factors, new_values, new_factors);
    // addStaticProjectionMeasurements(current_frame_id, new_projection_measurements, new_factors);

    // LOG(INFO) << "New smart factors " << new_smart_factors.size() << " updated smart factors " << updated_smart_factors.size() << " new projections " << new_projection_measurements.size() << " num triangulated " << tracklets_triangulated.size();

    // optimize(current_frame_id, new_smart_factors, only_updated_smart_factors, tracklets_triangulated, new_values, new_factors);
    // optimize(current_frame_id, new_smart_factors, updated_smart_factors, TrackletIds{}, new_values, new_factors);

    // updateDynamicObjectTrackletMap(input);

    // const gtsam::Pose3 previous_pose = state_.at<gtsam::Pose3>(CameraPoseSymbol(current_frame_id-1));
    // const gtsam::Pose3 current_pose = state_.at<gtsam::Pose3>(CameraPoseSymbol(current_frame_id));

    // LOG(ERROR) << "Estimated motions " <<  input->estimated_motions_.size();

    // StatusLandmarkEstimates all_dynamic_object_triangulation;

    // for(const auto& [object_id, ref_estimate] : input->estimated_motions_) {
    //     const EssentialDecompositionResult& decomposition_result = ref_estimate;

    //     StatusLandmarkEstimates dynamic_object_triangulation;
    //     gtsam::Rot3 R_motion;

    //     if(!attemptObjectTriangulation(
    //         current_frame_id,
    //         current_frame_id-1,
    //         object_id,
    //         decomposition_result,
    //         current_pose,
    //         previous_pose,
    //         dynamic_object_triangulation,
    //         R_motion)) { continue; }


    //     // //copy to all dynamic triangulation
    //     all_dynamic_object_triangulation.insert(
    //         all_dynamic_object_triangulation.begin(), dynamic_object_triangulation.begin(), dynamic_object_triangulation.end());

    // }

    LandmarkMap lmk_map;
    for(const auto&[tracklet_id, status] : tracklet_to_status_map_) {
        if(status.pf_type_ == ProjectionFactorType::PROJECTION) {
            //assume static only atm
            const auto lmk_symbol = StaticLandmarkSymbol(tracklet_id);
            if(!state_.exists(lmk_symbol)) {
                continue;
            }

            CHECK(state_.exists(lmk_symbol));
            const gtsam::Point3 lmk = state_.at<gtsam::Point3>(lmk_symbol);
            lmk_map.insert({tracklet_id, lmk});
        }
    }

    StatusLandmarkEstimates all_dynamic_object_triangulation;
    gtsam::FastMap<ObjectId, gtsam::Pose3Vector> object_poses_composed_;
    for(const auto& [object_id, ref_estimate] : input->estimated_motions_) {
        object_poses_composed_.insert({object_id, gtsam::Pose3Vector{}});

        FrameIds all_frames = do_tracklet_manager_.getFramesPerObject(object_id);
        for(size_t i = 1; i < all_frames.size(); i++) {
            FrameId frame_id = all_frames.at(i);
            gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, frame_id);

            if(!state_.exists(motion_symbol)) {
                continue;
            }

            if(i == 1) {
                ObjectPoseGT object_pose_gt;
                auto gt_frame_packet = gt_packet_map_.at(frame_id-1);
                if(!gt_frame_packet.getObject(object_id, object_pose_gt)) {
                    LOG(FATAL) << "Coudl not find gt object at frame " << frame_id-1 << " for object Id" << object_id;
                }
                CHECK_EQ(object_poses_composed_.at(object_id).size(), 0u);
                object_poses_composed_.at(object_id).push_back(object_pose_gt.L_world_);
            }
            else {
                gtsam::Pose3 motion = state_.at<gtsam::Pose3>(motion_symbol);
                gtsam::Pose3 object_pose = motion * (object_poses_composed_.at(object_id).back()); //object pose at the current frame, composed from the previous motion
                object_poses_composed_.at(object_id).push_back(object_pose);
            }
        }

        LOG(INFO) << "Added " <<  object_poses_composed_.at(object_id).size() << " composed poses for object " << object_id;

        //this is a lot of iteration for what should be a straighforward lookup
        const TrackletIds tracklet_ids = do_tracklet_manager_.getPerObjectTracklets(object_id);

        for(TrackletId tracklet_id : tracklet_ids) {
            CHECK(do_tracklet_manager_.trackletExists(tracklet_id));

            const DynamicObjectTracklet<Keypoint>& tracklet = do_tracklet_manager_.getByTrackletId(tracklet_id);

            for(const auto& [frame_id, measurement] : tracklet) {
                (void)measurement;
                DynamicPointSymbol dynamic_point_symbol = DynamicLandmarkSymbol(frame_id, tracklet_id);

                if(state_.exists(dynamic_point_symbol)) {
                    const Landmark lmk = state_.at<Landmark>(dynamic_point_symbol);
                    auto lmk_estimate = std::make_pair(tracklet_id, lmk);

                    LandmarkStatus status;
                    status.label_ = object_id;
                    status.method_ = LandmarkStatus::Method::OPTIMIZED;

                    all_dynamic_object_triangulation.push_back(std::make_pair(status, lmk_estimate));
                }

            }

        }

    }

    auto backend_output = std::make_shared<BackendOutputPacket>();
    backend_output->timestamp_ = input->getTimestamp();
    // backend_output->T_world_camera_ = state_.at<gtsam::Pose3>(CameraPoseSymbol(current_frame_id));
    backend_output->static_lmks_ = lmk_map;
    backend_output->object_poses_composed_ = object_poses_composed_;

    if(state_.exists(CameraPoseSymbol(current_frame_id))) {
        backend_output->T_world_camera_ = state_.at<gtsam::Pose3>(CameraPoseSymbol(current_frame_id));
    }

    backend_output->dynamic_lmks_ = all_dynamic_object_triangulation;

    return {State::Nominal, backend_output};
}


void MonoBackendModule::addInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors) {
    //add state
    new_values.insert(CameraPoseSymbol(frame_id), T_world_camera);
    new_factors.addPrior(CameraPoseSymbol(frame_id), T_world_camera, initial_pose_prior_);
}

void MonoBackendModule::addOdometry(const gtsam::Pose3& T_world_camera, FrameId frame_id, FrameId prev_frame_id, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
    //add state
    new_values.insert(CameraPoseSymbol(frame_id), T_world_camera);

    const gtsam::Symbol prev_pose_symbol = CameraPoseSymbol(prev_frame_id);
    //if prev_pose_symbol is in state, use this to construct the btween factor
    if(state_.exists(prev_pose_symbol)) {
        LOG(INFO) << "Adding odom between " << prev_frame_id << " and " << frame_id;
        const gtsam::Pose3 prev_pose = state_.at<gtsam::Pose3>(prev_pose_symbol);
        const gtsam::Pose3 odom = prev_pose.inverse() * T_world_camera;

        factor_graph_tools::addBetweenFactor(prev_frame_id, frame_id, odom, odometry_noise_, new_factors);
    }
}



void MonoBackendModule::updateSmartStaticObservations(
        const StatusKeypointMeasurements& measurements,
        const FrameId frame_id,
        TrackletIds& new_smart_factors,
        TrackletIds& updated_smart_factors,
        StatusKeypointMeasurements& new_projection_measurements)
{

    std::set<TrackletId> only_new_smart_factors;

    for(const StatusKeypointMeasurement& static_measurement : measurements) {
        const KeypointStatus& status = static_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = static_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::STATIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        auto it = tracklet_to_status_map_.find(tracklet_id);
        if(it == tracklet_to_status_map_.end()) {

            double rank_tolerance = 1.0;
            //! max distance to triangulate point in meters
            double landmark_distance_threshold = 20.0;
            //! max acceptable reprojection error // before tuning: 3
            double outlier_rejection = 8.0;
            double retriangulation_threshold = 1.0e-3;

            static_projection_params_.setRankTolerance(rank_tolerance);
            static_projection_params_.setLandmarkDistanceThreshold(
                landmark_distance_threshold);
            static_projection_params_.setRetriangulationThreshold(retriangulation_threshold);
            static_projection_params_.setDynamicOutlierRejectionThreshold(outlier_rejection);
            //! EPI: If set to true, will refine triangulation using LM.
            static_projection_params_.setEnableEPI(true);
            static_projection_params_.setLinearizationMode(gtsam::HESSIAN);
            static_projection_params_.setDegeneracyMode(gtsam::ZERO_ON_DEGENERACY);
            static_projection_params_.throwCheirality = true;
            static_projection_params_.verboseCheirality = true;

            // if the TrackletIdToProjectionStatus does not have this tracklet, then it should be the first time we have seen
            // it and therefore, should not be in any of the other data structures
            SmartProjectionFactor::shared_ptr smart_factor =
                factor_graph_tools::constructSmartProjectionFactor(
                    static_smart_noise_,
                    gtsam_calibration_,
                    static_projection_params_
                );

            ProjectionFactorStatus projection_status(tracklet_id, ProjectionFactorType::SMART, object_id);
            tracklet_to_status_map_.insert({tracklet_id, projection_status});

            CHECK(!smart_factor_map_.exists(tracklet_id)) << "Smart factor with tracklet id " << tracklet_id
                << " exists in the smart factor map but not in the tracklet_to_status_map";

            smart_factor_map_.add(tracklet_id, smart_factor, UninitialisedSlot);
            new_smart_factors.push_back(tracklet_id);
            only_new_smart_factors.insert(tracklet_id);
        }
        //this measurement is not new and is therefore a smart factor already in the map OR is a projection factor
        //sanity check that the object label is still the same for the tracked object
        CHECK_EQ(object_id, tracklet_to_status_map_.at(tracklet_id).object_id_);

        const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);
        const ProjectionFactorType factor_type = tracklet_to_status_map_.at(tracklet_id).pf_type_;
        if(factor_type ==  ProjectionFactorType::SMART) {
            CHECK(smart_factor_map_.exists(tracklet_id)) << "Factor has been marked as smart but does not exist in the smart factor map";
            //sanity check that we dont have a point for this factor yet
            CHECK(!state_.exists(lmk_symbol)) << "Factor has been marked as smart a lmk value exists for it";

            //update smart factor. This may be a new one or an existing factor
            SmartProjectionFactor::shared_ptr smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);
            factor_graph_tools::addSmartProjectionMeasurement(smart_factor, kp, frame_id);

            //append to list of updated smart factors only if this factor is also not new
            //we want the list of updated_smart_factors to only contain the factors updated and not also the new ones
            if(only_new_smart_factors.find(tracklet_id) == only_new_smart_factors.end()) {
                updated_smart_factors.push_back(tracklet_id);
            }


        }
        else if(factor_type == ProjectionFactorType::PROJECTION) {
            CHECK(!smart_factor_map_.exists(tracklet_id)) << "Factor has been marked as projection but exists in the smart factor map. It should have been removed.";
            //sanity check that we DO have a point for this factor yet
            CHECK(state_.exists(lmk_symbol)) << "Factor has been marked as projection but there is no lmk in the state vector.";
            new_projection_measurements.push_back(static_measurement);
        }

    }
}


// void MonoBackendModule::addNewSmartStaticFactors(const TrackletIds& new_smart_factors,  gtsam::NonlinearFactorGraph& factors) {
//     for(const TrackletId tracklet_id : new_smart_factors) {
//         CHECK(smart_static_factor_map_.exists(tracklet_id)) << "Smart factor with tracklet id " << tracklet_id
//                 << " has been marked as a new factor but is not in the smart_static_factor_map";
//         SmartProjectionFactor::shared_ptr smart_factor = smart_static_factor_map_.getSmartFactor(tracklet_id);

//         Slot& slot = smart_static_factor_map_.getSlot(tracklet_id);
//         CHECK_EQ(slot, -1) << "Smart factor with tracklet id " << tracklet_id
//                 << " has been marked as a new factor but slot is not -1";

//         size_t current_slot = factors.size();
//         factors.push_back(smart_factor);

//         slot = current_slot; //update slot
//     }
// }

void MonoBackendModule::tryTriangulateExistingSmartStaticFactors(const TrackletIds& updated_smart_factors, const gtsam::Values& new_and_current_state, TrackletIds& triangulated_tracklets,  TrackletIds& only_updated_smart_factors, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_projection_factors) {
    for(const TrackletId tracklet_id : updated_smart_factors) {
        CHECK(smart_factor_map_.exists(tracklet_id)) << "Smart factor with tracklet id " << tracklet_id
                << " has been updated but is not in the smart_static_factor_map";

        const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);
        CHECK(!new_and_current_state.exists(lmk_symbol) && !state_.exists(lmk_symbol)) << "Factor has been marked as smart a lmk value exists for it";


        SmartProjectionFactor::shared_ptr smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);
        if(smart_factor->size() > 2u) {
            //try and triangulate the point using the new and current values. This should contain the current pose
            gtsam::TriangulationResult triangulation_result;
            try {
                triangulation_result = smart_factor->point(new_and_current_state);
            }
            catch(const gtsam::CheiralityException& e) {
               continue; //just dont add...?
            }
            if(triangulation_result) {
                CHECK(triangulation_result.valid());
                CHECK(!smart_factor->isDegenerate());
                CHECK(!smart_factor->isFarPoint());
                CHECK(!smart_factor->isOutlier());
                CHECK(!smart_factor->isPointBehindCamera());
                const gtsam::Point3 point = triangulation_result.value();
                convertSmartToProjectionFactor(tracklet_id, point, new_values, new_projection_factors);

                triangulated_tracklets.push_back(tracklet_id);
            }
            else {
                only_updated_smart_factors.push_back(tracklet_id);
            }
        }
        // else {
        //     // only_updated_smart_factors.push_back(tracklet_id);
        // }
    }
}

bool MonoBackendModule::convertSmartToProjectionFactor(const TrackletId smart_factor_to_convert, const gtsam::Point3& lmk_world, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_projection_factors) {
    const TrackletId tracklet_id = smart_factor_to_convert;
    SmartProjectionFactor::shared_ptr smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);
    CHECK(smart_factor->point().valid());
    const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);


    Slot slot = smart_factor_map_.getSlot(tracklet_id);
    CHECK(slot != UninitialisedSlot) << "Trying to delete and convert smart factor with tracklet id " << tracklet_id << " but the slot is -1";

    {
        // SmartProjectionFactor::shared_ptr smart = boost::dynamic_pointer_cast<SmartProjectionFactor>(factors[slot]);
        // CHECK(smart);
        // CHECK(smart == smart_factor);
        // CHECK(smart->equals(*smart_factor));
    }

    std::stringstream ss;
    ss << "Adding projection factor for lmk: " << gtsam::DefaultKeyFormatter(lmk_symbol) << " ";
    new_values.insert(lmk_symbol, lmk_world);
    for (size_t i = 0; i < smart_factor->keys().size(); i++) {
        const gtsam::Symbol pose_symbol = gtsam::Symbol(smart_factor->keys().at(i));
        const auto& measured = smart_factor->measured().at(i);

        ss << " Pose Key: " << gtsam::DefaultKeyFormatter(pose_symbol) ;

        new_projection_factors.emplace_shared<GenericProjectionFactor>(
            measured,
            static_projection_noise_,
            pose_symbol,
            lmk_symbol,
            gtsam_calibration_
        );
    }

    LOG(INFO) << ss.str();

    //update status
    tracklet_to_status_map_.at(tracklet_id).pf_type_ = ProjectionFactorType::PROJECTION;

    return true;
}


void MonoBackendModule::addStaticProjectionMeasurements(const FrameId frame_id, const StatusKeypointMeasurements& new_projection_measurements, gtsam::NonlinearFactorGraph& new_projection_factors) {
    for(const StatusKeypointMeasurement& static_measurement : new_projection_measurements) {
        const KeypointStatus& status = static_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = static_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::STATIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        CHECK(!smart_factor_map_.exists(tracklet_id)) << "Measurement with tracklet id " << tracklet_id
                    << " has been marked as a projection factor but is in the smart_static_factor_map";
        CHECK_EQ(tracklet_to_status_map_.at(tracklet_id).pf_type_, ProjectionFactorType::PROJECTION);
        new_projection_factors.emplace_shared<GenericProjectionFactor>(
                kp,
                static_smart_noise_,
                CameraPoseSymbol(frame_id),
                StaticLandmarkSymbol(tracklet_id),
                gtsam_calibration_
            );

    }
}


void MonoBackendModule::optimize(FrameId frame_id, const TrackletIds& new_smart_factors, const TrackletIds& updated_smart_factors, const TrackletIds& triangulated_tracklets, const gtsam::Values& new_values, const gtsam::NonlinearFactorGraph& new_factors) {


    //guarantee that (static) smart factors will be at the start
    gtsam::NonlinearFactorGraph factors_to_add;
    for(TrackletId new_smart_factor_tracklet_ids : new_smart_factors) {
        Slot slot = smart_factor_map_.getSlot(new_smart_factor_tracklet_ids);
        CHECK(slot == UninitialisedSlot);
        factors_to_add.push_back(smart_factor_map_.getSmartFactor(new_smart_factor_tracklet_ids));
    }

    CHECK_EQ(factors_to_add.size(), new_smart_factors.size());
    factors_to_add.push_back(new_factors); //the other new factors

    gtsam::ISAM2UpdateParams isam_update_params;

    //for updated smart factors
    //indicate which new affected keys have been made. We assume this is the pose at the current frame
    gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> new_affected_keys;
    //this should be the only key affected
    gtsam::Key pose_key = CameraPoseSymbol(frame_id);
    //some sanity check - check key is in new_values
    CHECK(new_values.exists(pose_key));
    for(TrackletId updated_smart_factor_tracklet_ids : updated_smart_factors) {
        Slot slot = smart_factor_map_.getSlot(updated_smart_factor_tracklet_ids);
        CHECK(slot != UninitialisedSlot);

        {
            //some sanity check - check key is in new_values
            auto factor = smart_factor_map_.getSmartFactor(updated_smart_factor_tracklet_ids);
            CHECK(factor->find(pose_key) != factor->end());
            CHECK_EQ(factor, smoother_->getFactorsUnsafe().at(slot));
        }
        new_affected_keys.insert({slot, gtsam::KeySet{pose_key}});
    }
    isam_update_params.newAffectedKeys = new_affected_keys;

    //the triangulated tracklets are now the factors that should be removed from the smoother
    //we can them also remove the tracklet from the smart_factor_map_ as we no longer need to track the slot anymore
    //however we may need to track the projection factor slot if we want to remove it later (ie. cheirality exception)
    gtsam::FactorIndices factor_indicies_to_remove;
    for(TrackletId triangulated_tracklet_ids : triangulated_tracklets) {
        Slot slot = smart_factor_map_.getSlot(triangulated_tracklet_ids);
        factor_indicies_to_remove.push_back(slot);
        smart_factor_map_.erase(triangulated_tracklet_ids);
    }
    // isam_update_params.removeFactorIndices = factor_indicies_to_remove;

    LOG(INFO) << "Running update";
    gtsam::ISAM2Result result;
    try {
        result = smoother_->update(factors_to_add, new_values, isam_update_params);
    }
    catch(const gtsam::IndeterminantLinearSystemException& e) {
        auto factors = smoother_->getFactorsUnsafe();
        factors.saveGraph("/root/results/DynoSAM/mono_backend_graph.dot", DynoLikeKeyFormatter);


        std::vector<gtsam::NonlinearFactor::shared_ptr> associated_factors;
        factor_graph_tools::getAssociatedFactors(associated_factors, factors, e.nearbyVariable());
        gtsam::NonlinearFactorGraph associated_graph(associated_factors);

        std::stringstream ss;
        ss << "/root/results/DynoSAM/mono_backend_graph_failure_" << DynoLikeKeyFormatter(e.nearbyVariable()) << ".dot";
        associated_graph.saveGraph(ss.str());
        LOG(ERROR) << "Num associated factors " << associated_graph.size();
        LOG(FATAL) << "called after throwing an instance of 'gtsam::IndeterminantLinearSystemException', variable " << DynoLikeKeyFormatter(e.nearbyVariable());

    }
    catch(const gtsam::ValuesKeyAlreadyExists& e) {
        auto factors = smoother_->getFactorsUnsafe();
        factors.saveGraph("/root/results/DynoSAM/mono_backend_graph.dot", DynoLikeKeyFormatter);
        LOG(FATAL) << "called after throwing an instance of 'gtsam::ValuesKeyAlreadyExists', key already exists " << DynoLikeKeyFormatter(e.key());

    }
    catch(const gtsam::CheiralityException& e) {
        auto factors = smoother_->getFactorsUnsafe();
        // factors.saveGraph("/root/results/DynoSAM/mono_backend_graph.dot", DynoLikeKeyFormatter);
        // LOG(FATAL) << "called after throwing an instance of 'gtsam::CheiralityException', variable " << DynoLikeKeyFormatter(e.nearbyVariable());

        std::vector<gtsam::NonlinearFactor::shared_ptr> associated_factors;
        factor_graph_tools::getAssociatedFactors(associated_factors, factors, e.nearbyVariable());

        //laziest thing possible but try and find the factors -> smart factors might still be here?

    }

    //recover slots for new smart factors
    const gtsam::FactorIndices& new_factors_indices = result.newFactorsIndices;

    //iterate over the first N new_factor_indices since we know this correspondes with new smatrt factors
    for(size_t i = 0; i < new_smart_factors.size(); i++) {
        Slot slot = new_factors_indices.at(i);
        TrackletId tracklet_id = new_smart_factors.at(i);

        //update slot
        smart_factor_map_.getSlot(tracklet_id) = slot;
    }

    state_ = smoother_->calculateEstimate();

}



void MonoBackendModule::updateDynamicObjectTrackletMap(MonocularInstanceOutputPacket::ConstPtr input) {
    const FrameId current_frame_id =  input->getFrameId();
    const auto& dynamic_measurements = input->dynamic_keypoint_measurements_;

    for(const StatusKeypointMeasurement& dynamic_measurement : dynamic_measurements) {
        const KeypointStatus& status = dynamic_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = dynamic_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::DYNAMIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        do_tracklet_manager_.add(object_id, tracklet_id, current_frame_id, kp);
    }
}


bool MonoBackendModule::attemptObjectTriangulation(
        FrameId current_frame,
        FrameId previous_frame,
        ObjectId object_id,
        const EssentialDecompositionResult&  motion_estimate,
        const gtsam::Pose3& T_world_camera_curr,
        const gtsam::Pose3& T_world_camera_prev,
        StatusLandmarkEstimates& triangulated_values,
        gtsam::Rot3& R_motion)
{

    int count = 0;
    triangulated_values.clear();

    gtsam::Point3Vector lmks_world;
    TrackletIds triangulated_tracklets;


    CHECK(do_tracklet_manager_.objectExists(object_id));

    const TrackletIds tracklet_ids = do_tracklet_manager_.getPerObjectTracklets(object_id);
    gtsam::Point2Vector observation_curr, observation_prev;
    for(TrackletId tracklet_id : tracklet_ids) {
        CHECK(do_tracklet_manager_.trackletExists(tracklet_id));

        if(count > 100) {
            break;
        }

        DynamicObjectTracklet<Keypoint>& tracklet = do_tracklet_manager_.getByTrackletId(tracklet_id);

        if(tracklet.size() < 2u) { continue; }

        //check if in prev and current frame
        if(tracklet.exists(current_frame) && tracklet.exists(previous_frame)) {
            const Keypoint& current_measurement = tracklet.at(current_frame);
            observation_curr.push_back(current_measurement);

            const Keypoint& prev_measurement = tracklet.at(previous_frame);
            observation_prev.push_back(prev_measurement);

            triangulated_tracklets.push_back(tracklet_id);

            count++;
        }
    }

    LOG(WARNING) << count << " tracket points for obj " << object_id;
    CHECK(observation_curr.size() == observation_prev.size());

    if(observation_curr.size() < 3u) {
        return false;
    }

    const gtsam::Matrix3 K = gtsam_calibration_->K();
    const gtsam::Matrix3 R1 = motion_estimate.R1_.matrix();
    const gtsam::Matrix3 R2 = motion_estimate.R2_.matrix();

    const gtsam::Matrix3 R_world_camera_curr = T_world_camera_curr.rotation().matrix();
    const gtsam::Matrix3 R_world_camera_prev = T_world_camera_prev.rotation().matrix();

    //at this point we need to compute W_R where R is the rotation component of H
    //R1/R2 actually corresponds to the second half of the essential matrix constraint which, when motion is addeded, becomes
    // X_{k}^R{-1} * {WH^}_R * X_{k-1}^R = R1/R2
    //and we just want {WH^}_R
    const gtsam::Matrix3 world_H_R1 = R_world_camera_curr * R1 * R_world_camera_prev.inverse();
    const gtsam::Matrix3 world_H_R2 = R_world_camera_curr * R2 * R_world_camera_prev.inverse();

    gtsam::Point3Vector triangulated_points_R1 =
        mono_backend_tools::triangulatePoint3Vector(
            T_world_camera_prev,
            T_world_camera_curr,
            K,
            observation_prev,
            observation_curr,
            world_H_R1
        );

    gtsam::Point3Vector triangulated_points_R2 =
        mono_backend_tools::triangulatePoint3Vector(
            T_world_camera_prev,
            T_world_camera_curr,
            K,
            observation_prev,
            observation_curr,
            world_H_R2
        );


    const double sigma_R1 = calculateStandardDeviation(triangulated_points_R1);
    const double sigma_R2 = calculateStandardDeviation(triangulated_points_R2);

    int R1_points_visible = 0, R2_points_visible = 0;

    for(const gtsam::Point3& pt : triangulated_points_R1) {
        gtsam::Point3 pt_prev_camera = T_world_camera_prev.inverse() * pt;
        if(camera_->isLandmarkContained(pt_prev_camera)) {
            R1_points_visible++;
        }
    }

     for(const gtsam::Point3& pt : triangulated_points_R2) {
        gtsam::Point3 pt_prev_camera = T_world_camera_prev.inverse() * pt;
        if(camera_->isLandmarkContained(pt_prev_camera)) {
            R2_points_visible++;
        }
    }

    LOG(INFO) << "sigmaR1 " << sigma_R1 << " sigma r2 " << sigma_R2;
    LOG(INFO) << "R1_points_visible " << R1_points_visible << " R2_points_visible " << R2_points_visible;

    // if(sigma_R1 < sigma_R2) {
    //     lmks_world = triangulated_points_R1;
    //     R_motion = gtsam::Rot3(world_H_R1);
    // }
    // else {
    //     lmks_world = triangulated_points_R2;
    //     R_motion = gtsam::Rot3(world_H_R2);
    // }


    if(R2_points_visible < R1_points_visible && R2_points_visible > 0) {
        lmks_world = triangulated_points_R1;
        R_motion = gtsam::Rot3(world_H_R1);
    }
    else if(R1_points_visible < R2_points_visible && R1_points_visible > 0) {
        lmks_world = triangulated_points_R2;
        R_motion = gtsam::Rot3(world_H_R2);
    }
    else if(R2_points_visible > 0 && R1_points_visible > 0) {
        if(sigma_R1 < sigma_R2) {
            lmks_world = triangulated_points_R1;
            R_motion = gtsam::Rot3(world_H_R1);
        }
        else {
            lmks_world = triangulated_points_R2;
            R_motion = gtsam::Rot3(world_H_R2);
        }
    }
    else {
        return false;
    }


    CHECK(lmks_world.size() == triangulated_tracklets.size());
    for(size_t i = 0; i < lmks_world.size(); i++) {
        const TrackletId tracklet_id = triangulated_tracklets.at(i);
        const Landmark lmk = lmks_world.at(i);
        auto lmk_estimate = std::make_pair(tracklet_id, lmk);

        LandmarkStatus status;
        status.label_ = object_id;
        status.method_ = LandmarkStatus::Method::TRIANGULATED;

        triangulated_values.push_back(std::make_pair(status, lmk_estimate));
    }


    return true;
}


void MonoBackendModule::addInitalObjectValues(
        FrameId current_frame_id,
        ObjectId object_id,
        const StatusLandmarkEstimates& triangulated_values,
        const gtsam::Pose3& prev_H_world_current,
        gtsam::Values& new_values,
        gtsam::NonlinearFactorGraph& factors)
{
    const gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, current_frame_id);

    new_values.insert(motion_symbol, prev_H_world_current);

    for(const StatusLandmarkEstimate& estimate : triangulated_values) {
        const LandmarkStatus& status = estimate.first;
        CHECK(status.label_ != background_label);

        const LandmarkEstimate& le = estimate.second;

        const Landmark lmk = le.second;
        const TrackletId tracklet_id = le.first;


        // const gtsam::Symbol sym = DynamicLandmarkSymbol(tracklet_id);
        // new_values.insert(sym, lmk);

        //add motion factor
        // factors.emplace_shared<LandmarkMotionTernaryFactor>(

        // )


    }

}


// void MonoBackendModule::updateStaticObservations(
//         const StatusKeypointMeasurements& measurements,
//         const FrameId frame_id,
//         gtsam::Values& new_point_values,
//         std::vector<SmartProjectionFactor::shared_ptr>& new_smart_factors,
//         std::vector<SmartProjectionFactor::shared_ptr>& new_projection_factors,
//         TrackletIds& smart_factors_to_convert) {


//     size_t num_triangulated = 0u;
//     size_t num_smart_measurements = 0u;
//     size_t num_projection_measurements = 0;

//     //check that new values has the camera pose from this frame in it. It should as we need it to try and triangulate the points
//     CHECK(new_values.exists(CameraPoseSymbol(frame_id)));

//     const gtsam::Symbol pose_symbol = CameraPoseSymbol(frame_id);
//     //the new and current values at the start of this function
//     gtsam::Values new_and_current_values(state_);
//     new_and_current_values.insert(new_values);

//     for(const StatusKeypointMeasurement& static_measurement : measurements) {
//         const KeypointStatus& status = static_measurement.first;
//         const KeyPointType kp_type = status.kp_type_;
//         const KeypointMeasurement& measurement = static_measurement.second;

//         const ObjectId object_id = status.label_;

//         CHECK(kp_type == KeyPointType::STATIC);
//         const TrackletId tracklet_id = measurement.first;
//         const Keypoint& kp = measurement.second;

//         auto it = tracklet_to_status_map_.find(tracklet_id);
//         if(it == tracklet_to_status_map_.end()) {
//             // if the TrackletIdToProjectionStatus does not have this tracklet, then it should be the first time we have seen
//             // it and therefore, should not be in any of the other data structures

//             SmartProjectionFactor::shared_ptr smart_factor =
//                 factor_graph_tools::constructSmartProjectionFactor(
//                     static_smart_noise_,
//                     gtsam_calibration_,
//                     static_projection_params_
//                 );

//             ProjectionFactorStatus projection_status(tracklet_id, ProjectionFactorType::SMART, object_id);
//             tracklet_to_status_map_.insert({tracklet_id, projection_status});

//             CHECK(!smart_static_factor_map_.exists(tracklet_id)) << "Smart factor with tracklet id " << tracklet_id
//                 << " exists in the smart factor map but not in the tracklet_to_status_map";

//             smart_static_factor_map_.add(tracklet_id, smart_factor, UninitialisedSlot);
//         }


//         //sanity check that the object label is still the same for the tracked object
//         CHECK_EQ(object_id, tracklet_to_status_map_.at(tracklet_id).object_id_);

//         const ProjectionFactorType factor_type = tracklet_to_status_map_.at(tracklet_id).pf_type_;
//         const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);

//         if(factor_type ==  ProjectionFactorType::SMART) {
//             CHECK(smart_static_factor_map_.exists(tracklet_id)) << "Factor has been marked as smart but does not exist in the smart factor map";
//             //sanity check that we dont have a point for this factor yet
//             CHECK(!new_values.exists(lmk_symbol) && !state_.exists(lmk_symbol)) << "Factor has been marked as smart a lmk value exists for it";


//             SmartProjectionFactor::shared_ptr smart_factor = smart_static_factor_map_.getSmartFactor(tracklet_id);
//             factor_graph_tools::addSmartProjectionMeasurement(smart_factor, kp, frame_id);

//             Slot slot = smart_static_factor_map_.getSlot(tracklet_id);
//             if(slot == UninitialisedSlot) {
//                 //factor is not in graph yet
//                 //we dont know what the slot is going to be yet. The slot will get updated when we actually add everything to the state graph
//                 new_smart_factors.push_back(tracklet_id);
//             }
//             else {
//                 //check the factor is in the current graph?
//                 //only init if in graph?

//                 //TODO:do we need to check if min size
//                 //try and triangulate the point using the new and current values. This should contain the current pose
//                 gtsam::TriangulationResult triangulation_result = smart_factor->point(new_and_current_values);

//                 if(triangulation_result) {
//                     smart_factors_to_convert.push_back(tracklet_id);

//                     //add initial value to new values
//                     const gtsam::Point3 lmk_initial = *triangulation_result;
//                     new_point_values.insert(lmk_symbol, lmk_initial);

//                     num_triangulated++;
//                 }
//             }

//             num_smart_measurements++;
//         }
//         else if(factor_type == ProjectionFactorType::PROJECTION) {
//             CHECK(!smart_static_factor_map_.exists(tracklet_id)) << "Factor has been marked as projection but exists in the smart factor map. It should have been removed.";
//             //sanity check that we DO have a point for this factor yet
//             CHECK(state_.exists(lmk_symbol)) << "Factor has been marked as projection but there is no lmk in the state vector.";

//             new_factors.emplace_shared<GenericProjectionFactor>(
//                 kp,
//                 static_smart_noise_,
//                 pose_symbol,
//                 lmk_symbol,
//                 gtsam_calibration_
//             );

//             num_projection_measurements++;
//         }

//     }

//     LOG(INFO) << "Num smart " << num_smart_measurements << " num triangulated " << num_triangulated << " num projected " << num_projection_measurements;
// }


// void MonoBackendModule::convertAndDeleteSmartFactors(const gtsam::Values& new_values, const TrackletIds& smart_factors_to_convert, gtsam::NonlinearFactorGraph& new_factors) {
//     for(const TrackletId tracklet : smart_factors_to_convert) {
//         //expect these factors to be in the graph
//         Slot slot = smart_static_factor_map_.getSlot(tracklet);
//         CHECK(slot != UninitialisedSlot) << "Trying to delete and convert smart factor with tracklet id " << tracklet << " but the slot is -1";

//         //can we check that this is the factor we want?
//         SmartProjectionFactor::shared_ptr smart = boost::dynamic_pointer_cast<SmartProjectionFactor>(state_graph_[slot]);
//         CHECK(smart);

//         const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet);
//         //also check that the 3d point of this triangulated factor is is new values
//         CHECK(new_values.exists(lmk_symbol));

//         //TODO: this will NOT work with incremental as we need to actually remove by slot!!
//         state_graph_.remove(slot);




//         //iterate over all keys in the factor and add them as projection factors
//         for (size_t i = 0; i < smart->keys().size(); i++) {
//             const gtsam::Symbol& pose_symbol = gtsam::Symbol(smart->keys().at(i));
//             const auto& measured = smart->measured().at(i);

//             new_factors.emplace_shared<GenericProjectionFactor>(
//                 measured,
//                 static_smart_noise_,
//                 pose_symbol,
//                 lmk_symbol,
//                 gtsam_calibration_
//             );
//         }
//     }
// }

// void MonoBackendModule::addToStatesStructures(const gtsam::Values& new_values, const gtsam::NonlinearFactorGraph& new_factors, const TrackletIds& new_smart_factors) {
//     state_.insert(new_values);
//     state_graph_ += new_factors;

//     for(TrackletId tracklet_id : new_smart_factors) {
//         //these should all be in the smart_static_factor_map_
//         //note: reference
//         Slot& slot = smart_static_factor_map_.getSlot(tracklet_id);
//         auto smart_factor = smart_static_factor_map_.getSmartFactor(tracklet_id);
//         CHECK(slot == UninitialisedSlot);

//         size_t current_slot = state_graph_.size();
//         state_graph_.push_back(smart_factor);

//         slot = current_slot;
//     }
// }


void MonoBackendModule::setFactorParams(const BackendParams& backend_params) {
    //set static projection smart noise
    static_smart_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, backend_params.smart_projection_noise_sigma_);
    auto huber =
        gtsam::noiseModel::mEstimator::Huber::Create(0.00001, gtsam::noiseModel::mEstimator::Base::ReweightScheme::Block);
    static_projection_noise_ = gtsam::noiseModel::Robust::Create(huber, static_smart_noise_);

    CHECK(static_smart_noise_);

    gtsam::Vector6 odom_sigmas;
    odom_sigmas.head<3>().setConstant(backend_params.odometry_rotation_sigma_);
    odom_sigmas.tail<3>().setConstant(
        backend_params.odometry_translation_sigma_);
    odometry_noise_ = gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);
    CHECK(odometry_noise_);

    initial_pose_prior_ =  gtsam::noiseModel::Isotropic::Sigma(6u, 0.0001);
    CHECK(initial_pose_prior_);

    landmark_motion_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, backend_params.motion_ternary_factor_noise_sigma_);
    CHECK(landmark_motion_noise_);

}


void MonoBackendModule::buildGraphWithDepth(MonocularInstanceOutputPacket::ConstPtr input) {
    const FrameId current_frame_id = input->getFrameId();
    const auto& input_tracking_images = input->frame_.tracking_images_;
    const gtsam::Pose3 T_world_camera = input->T_world_camera_;

    CHECK(input->gt_packet_);
    const GroundTruthInputPacket gt_packet = input->gt_packet_.value();

    gt_packet_map_.insert({current_frame_id, gt_packet});

    gtsam::Pose3 T_world_camera_gt = gt_packet.X_world_;

    gtsam::Key pose_symbol = CameraPoseSymbol(current_frame_id);

    static bool is_first = true;

    //the pose we will actually use
    const gtsam::Pose3 cam_pose = T_world_camera_gt;

    size_t num_new_static_points = 0;

    LOG(INFO) << "Running buildGraphWithDepth on frame " <<  current_frame_id;

    if(is_first) {
        //camera pose
        addInitialPose(cam_pose, current_frame_id, new_values_, new_factors_);

    }
    else {
        LOG(INFO) << "Adding odom";
        new_values_.insert(CameraPoseSymbol(current_frame_id), cam_pose);

        const gtsam::Symbol prev_pose_symbol = CameraPoseSymbol(current_frame_id-1);
        gtsam::Pose3 previous_pose;
        //if prev_pose_symbol is in state, use this to construct the btween factor
        if(state_.exists(prev_pose_symbol)) {
            previous_pose = state_.at<gtsam::Pose3>(prev_pose_symbol);
        }
        else if(new_values_.exists(prev_pose_symbol)) {
            previous_pose = new_values_.at<gtsam::Pose3>(prev_pose_symbol);
        }
        else {
            LOG(FATAL) << "Shoudl have prev pose";
        }

        const gtsam::Pose3 odom = previous_pose.inverse() * cam_pose;

        factor_graph_tools::addBetweenFactor(current_frame_id-1, current_frame_id, odom, odometry_noise_, new_factors_);
        // addOdometry(cam_pose, current_frame_id, current_frame_id-1, new_values_, new_factors_);
    }

    static gtsam::FastMap<TrackletId, std::vector<GenericProjectionFactor::shared_ptr>> static_factor_map;
    static gtsam::FastMap<TrackletId, bool> static_in_graph_static_factor_map;
    static gtsam::FastMap<gtsam::Symbol, gtsam::Point3> static_initial_points;

    static gtsam::FastMap<TrackletId, bool> marked_for_cherality;


    int static_count = 0;
    //static points
    for(const StatusKeypointMeasurement& static_measurement : input->static_keypoint_measurements_) {



        const KeypointStatus& status = static_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = static_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::STATIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);

        //if tracklet has been marked as bad lmk stop tracking it (not ideal as we would actually want to keep tracking it
        //to try and triangulate later. )
        if(marked_for_cherality.exists(tracklet_id)) {
            continue;
        }


        const Landmark lmk_cam =  input->frame_.backProjectToCamera(tracklet_id);
        //only works when T_world_camera in the frame is correct (ie, monocular will have a lot of drift)
        // const Landmark lmk_world = input->frame_.backProjectToWorld(tracklet_id);

        //check that in front of camera?
        if(!camera_->isLandmarkContained(lmk_cam)) {
            continue;
        }

        const Landmark lmk_world = cam_pose * lmk_cam;



        if(!static_initial_points.exists(lmk_symbol)) {
            // new_values_.insert(lmk_symbol, lmk_world);
            static_initial_points.insert({lmk_symbol, lmk_world});
            // num_new_static_points++;

            ProjectionFactorStatus projection_status(tracklet_id, ProjectionFactorType::PROJECTION, object_id);
            //just so we can then add it to the output map which checks for projection
            tracklet_to_status_map_.insert({tracklet_id, projection_status});


        }


        auto projection_factor = boost::make_shared<GenericProjectionFactor>(
            kp,
            static_smart_noise_,
            pose_symbol,
            lmk_symbol,
            gtsam_calibration_,
            true, true
        );


        //new tracklet Id
        if(!static_in_graph_static_factor_map.exists(tracklet_id)) {
            static_in_graph_static_factor_map.insert({tracklet_id, false});
        }


        //if factor is not in graph just add to factor map
        if(!static_in_graph_static_factor_map.at(tracklet_id)) {
            //check if new factor
             if(!static_factor_map.exists(tracklet_id)) {
                static_factor_map.insert({tracklet_id, std::vector<GenericProjectionFactor::shared_ptr>{projection_factor}});
            }
            else {
                static_factor_map.at(tracklet_id).push_back(projection_factor);
            }
        }
        //if factor is in map add to new factors as we will add to map
        else {
            // CHECK(smoother_->valueExists(lmk_symbol));
            CHECK(new_values_.exists(lmk_symbol));
            new_factors_.push_back(projection_factor);
        }


        //check if we can now add factor and value to map
        //add all the collected projection factors and the initial value
        CHECK(static_initial_points.exists(lmk_symbol));
        if(!static_in_graph_static_factor_map.at(tracklet_id) && static_factor_map.at(tracklet_id).size() >= 2u) {
            static_in_graph_static_factor_map.at(tracklet_id) = true;

            for(auto f : static_factor_map.at(tracklet_id)) {
                new_factors_.push_back(f);
            }

            new_values_.insert(lmk_symbol, lmk_world);

            static auto kPointPrior = gtsam::noiseModel::Isotropic::Sigma(3, 0.3);
            // new_factors_.addPrior(lmk_symbol, lmk_world, kPointPrior);
        }

        static_count++;

    }

    static gtsam::FastMap<TrackletId, std::vector<GenericProjectionFactor::shared_ptr>> dynamic_factor_map;
    static gtsam::FastMap<ObjectId, std::vector<LandmarkMotionTernaryFactor::shared_ptr>> dynamic_motion_factor_map;
    static gtsam::FastMap<TrackletId, bool> dynamic_in_graph_factor_map;
    static gtsam::FastMap<TrackletId, gtsam::Values> dynamic_initial_points;



    //dynamic measurements
    const auto& dynamic_measurements = input->dynamic_keypoint_measurements_;
    std::set<TrackletId> set_tracklets;
    for(const StatusKeypointMeasurement& dynamic_measurement : dynamic_measurements) {

        // if(dynamic_count > 50) {
        //     break;
        // }

        const KeypointStatus& status = dynamic_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = dynamic_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::DYNAMIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        set_tracklets.insert(tracklet_id);

        do_tracklet_manager_.add(object_id, tracklet_id, current_frame_id, kp);

        // LOG_IF(INFO, tracklet_id < 5560 && tracklet_id > 5550) << "adding dynamic tracklet " << tracklet_id << " to object " << object_id;

        //TODO: check if missaocation happens in frontend or as part of the the do_tracklet_manager_

    }

    CHECK(set_tracklets.size() == dynamic_measurements.size());

    const size_t min_dynamic_obs = 3u;

    for(const auto& [object_id, ref_estimate] : input->estimated_motions_) {
        //this will grow massively overtime?
        LOG(INFO) << "Looking at objects " << object_id;
        const TrackletIds tracklet_ids = do_tracklet_manager_.getPerObjectTracklets(object_id);

        {
            std::set<TrackletId> tracklet_set(tracklet_ids.begin(), tracklet_ids.end());
            CHECK_EQ(tracklet_set.size(), tracklet_ids.size());
        }

        //we need frame to at least be 1 (so we can backwards index the motion from 0) so skip all processinig
        //until the current frame is more than the min obs
        //dynamic measurements are still updated every frame by updating the dynamic object tracklet manager
        if(current_frame_id <= min_dynamic_obs) {
            continue;
        }


        //describes a motion from previous frame to current frame
        gtsam::Symbol current_motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, current_frame_id);

        int dynamic_count = 0;
        TrackletIds new_well_tracked_points;
        TrackletIds existing_well_tracked_points;


        for(TrackletId tracklet_id : tracklet_ids) {
            CHECK(do_tracklet_manager_.trackletExists(tracklet_id));

            // if(dynamic_count > 30) {
            //     break;
            // }

            // LOG(INFO) << "Tracklet id " << tracklet_id << " for object id " << object_id;

            DynamicObjectTracklet<Keypoint>& tracklet = do_tracklet_manager_.getByTrackletId(tracklet_id);

            DynamicPointSymbol current_dynamic_point_symbol = DynamicLandmarkSymbol(current_frame_id, tracklet_id);
            DynamicPointSymbol previous_dynamic_point_symbol = DynamicLandmarkSymbol(current_frame_id - 1u, tracklet_id);

            if(!tracklet.exists(current_frame_id)) { continue; }

            //   new tracklet Id
            if(!dynamic_in_graph_factor_map.exists(tracklet_id)) {
                dynamic_in_graph_factor_map.insert({tracklet_id, false});
            }

            if(dynamic_in_graph_factor_map.at(tracklet_id)) {
                // LOG_IF(INFO, tracklet_id < 5560 && tracklet_id > 5550) << "Dynamic tracklet " << tracklet_id << " is in graph at frame " <<  current_frame_id;
                //the previous point should be in the graph
                // CHECK(smoother_->valueExists(previous_dynamic_point_symbol)) << DynoLikeKeyFormatter(previous_dynamic_point_symbol);
                CHECK(new_values_.exists(previous_dynamic_point_symbol)) << DynoLikeKeyFormatter(previous_dynamic_point_symbol);
                existing_well_tracked_points.push_back(tracklet_id);
            }
            else {
                CHECK(!new_values_.exists(previous_dynamic_point_symbol)) << DynoLikeKeyFormatter(previous_dynamic_point_symbol);
                //not in graph so lets see if well tracked
                bool is_well_tracked = true;
                for(size_t i = current_frame_id - min_dynamic_obs; i <= current_frame_id; i++) {
                    //for each dynamic point, check if we have a previous point that is also tracked
                    is_well_tracked &= tracklet.exists(i);
                }

                if(is_well_tracked) {
                    // LOG_IF(INFO, tracklet_id < 5560 && tracklet_id > 5550) << "Dynamic tracklet " << tracklet_id << " is well tracked at frame " <<  current_frame_id;
                    new_well_tracked_points.push_back(tracklet_id);
                }

            }

            dynamic_count++;

        }

        LOG(INFO) << new_well_tracked_points.size() << " newly tracked for dynamic object and " << existing_well_tracked_points.size() << " existing tracks " << object_id;
        for(TrackletId tracked_id : new_well_tracked_points) {
            DynamicObjectTracklet<Keypoint>& tracklet = do_tracklet_manager_.getByTrackletId(tracked_id);


            const size_t starting_frame_points = current_frame_id - min_dynamic_obs;
            const size_t starting_frame_motion = starting_frame_points + 1;
        //     //up to and including the current frame
            for(size_t frame = starting_frame_points; frame <= current_frame_id; frame++) {
                DynamicPointSymbol dynamic_point_symbol = DynamicLandmarkSymbol(frame, tracked_id);

                // LOG_IF(INFO, tracked_id < 5560 && tracked_id > 5550) << "Adding new tracks at frame " << frame << " tracklet id " << tracked_id << " " << DynoLikeKeyFormatter(dynamic_point_symbol);
                CHECK(tracklet.exists(frame));
                gtsam::Symbol cam_symbol = CameraPoseSymbol(frame);

                auto gt_frame_packet = gt_packet_map_.at(frame);

                Landmark lmk_world;

                //if first point then use gt
                if(frame == starting_frame_points) {
                    lmk_world = cam_pose * input->frame_.backProjectToCamera(tracked_id);
                }
                else {
                    //propoate from gt motion
                    ObjectPoseGT object_pose_gt;
                    auto gt_frame_packet = gt_packet_map_.at(frame);
                    if(!gt_frame_packet.getObject(object_id, object_pose_gt)) {
                        LOG(FATAL) << "Coudl not find gt object at frame " << frame << " for object Id " << object_id << " and packet " << gt_frame_packet;
                    }

                    const auto prev_dynamic_point_symbol = DynamicLandmarkSymbol(frame-1u, tracked_id);
                    CHECK(object_pose_gt.prev_H_current_world_);
                    CHECK(new_values_.exists(prev_dynamic_point_symbol));

                    gtsam::Pose3 prev_H_current_world_ = object_pose_gt.prev_H_current_world_.value(); //TODO: perterb?

                    Landmark lmk_world_prev = new_values_.at<Landmark>(prev_dynamic_point_symbol);
                    lmk_world = prev_H_current_world_ * lmk_world_prev;
                }


                // CHECK(!smoother_->valueExists(dynamic_point_symbol));
                CHECK(!new_values_.exists(dynamic_point_symbol));
                new_values_.insert(dynamic_point_symbol, lmk_world);

                const Keypoint& kp = tracklet.at(frame);
                auto projection_factor = boost::make_shared<GenericProjectionFactor>(
                    kp,
                    //note: using static noise
                    static_smart_noise_,
                    cam_symbol,
                    dynamic_point_symbol,
                    gtsam_calibration_
                );

                new_factors_.push_back(projection_factor);
            }

            //add motion -> start from first index + 1 so we can index from the current frame
            for(size_t frame = starting_frame_motion; frame <= current_frame_id; frame++) {
                // LOG_IF(INFO, tracked_id < 5560 && tracked_id > 5550) << "Adding new motion from at frame " << frame;
                DynamicPointSymbol current_dynamic_point_symbol = DynamicLandmarkSymbol(frame, tracked_id);
                DynamicPointSymbol previous_dynamic_point_symbol = DynamicLandmarkSymbol(frame - 1u, tracked_id);

                CHECK(new_values_.exists(current_dynamic_point_symbol));
                CHECK(new_values_.exists(previous_dynamic_point_symbol));

                // CHECK(!smoother_->valueExists(current_dynamic_point_symbol));
                // CHECK(!smoother_->valueExists(previous_dynamic_point_symbol));

                //motion that takes us from previous to current
                gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, frame);
                // // CHECK(!smoother_->valueExists(motion_symbol));
                // CHECK(!new_values_.exists(motion_symbol));
                if(!new_values_.exists(motion_symbol)) {

                    ObjectPoseGT object_pose_gt;
                    auto gt_frame_packet = gt_packet_map_.at(frame);
                    if(!gt_frame_packet.getObject(object_id, object_pose_gt)) {
                        LOG(FATAL) << "Coudl not find gt object at frame " << frame << " for object Id" << object_id;
                    }

                    CHECK(object_pose_gt.prev_H_current_world_);


                    // new_values_.insert(motion_symbol, object_pose_gt.prev_H_current_world_.value());
                    new_values_.insert(motion_symbol, gtsam::Pose3::Identity());
                }

                auto motion_factor = boost::make_shared<LandmarkMotionTernaryFactor>(
                    previous_dynamic_point_symbol,
                    current_dynamic_point_symbol,
                    motion_symbol,
                    landmark_motion_noise_
                );

                new_factors_.push_back(motion_factor);
            }

            dynamic_in_graph_factor_map.at(tracked_id) = true;

        }

        // LOG(INFO) << "Adding existing tracks";
        for(TrackletId tracked_id : existing_well_tracked_points) {
            DynamicObjectTracklet<Keypoint>& tracklet = do_tracklet_manager_.getByTrackletId(tracked_id);
            DynamicPointSymbol current_dynamic_point_symbol = DynamicLandmarkSymbol(current_frame_id, tracked_id);
            DynamicPointSymbol previous_dynamic_point_symbol = DynamicLandmarkSymbol(current_frame_id - 1u, tracked_id);

            const Landmark lmk_world = cam_pose * input->frame_.backProjectToCamera(tracked_id);

            CHECK(!new_values_.exists(current_dynamic_point_symbol));
            CHECK(new_values_.exists(previous_dynamic_point_symbol));
            // CHECK(!smoother_->valueExists(current_dynamic_point_symbol));
            // CHECK(smoother_->valueExists(previous_dynamic_point_symbol));
            new_values_.insert(current_dynamic_point_symbol, lmk_world);

            const Keypoint& kp = tracklet.at(current_frame_id);
            auto projection_factor = boost::make_shared<GenericProjectionFactor>(
                kp,
                //note: using static noise
                static_smart_noise_,
                pose_symbol,
                current_dynamic_point_symbol,
                gtsam_calibration_
            );

            new_factors_.push_back(projection_factor);

            //motion that takes us from previous to current
            gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, current_frame_id);
            // CHECK(!new_values_.exists(motion_symbol));
            // CHECK(!smoother_->valueExists(motion_symbol));

            CHECK(new_values_.exists(ObjectMotionSymbolFromCurrentFrame(object_id, current_frame_id-1u)));

            if(!new_values_.exists(motion_symbol)) {

                ObjectPoseGT object_pose_gt;
                if(!gt_packet.getObject(object_id, object_pose_gt)) {
                    LOG(FATAL) << "Coudl not find gt object at frame " << current_frame_id << " for object Id" << object_id;
                }

                CHECK(object_pose_gt.prev_H_current_world_);


                // new_values_.insert(motion_symbol, object_pose_gt.prev_H_current_world_.value());
                new_values_.insert(motion_symbol, gtsam::Pose3::Identity());
            }

            auto motion_factor = boost::make_shared<LandmarkMotionTernaryFactor>(
                previous_dynamic_point_symbol,
                current_dynamic_point_symbol,
                motion_symbol,
                landmark_motion_noise_
            );

            new_factors_.push_back(motion_factor);
        }
    }


    int num_opt_attemts = 0;
    const int max_opt_attempts = 100;
    gtsam::Values initial_values = new_values_;
    bool solved = false;
    std::function<void()> handle_cheriality_optimize = [&]() -> void {
        LOG(INFO) << "Running graph optimziation with values " << new_values_.size() << " and factors " << new_factors_.size() << ". Opt attempts=" << num_opt_attemts;

        if(solved) {
            return;
        }

        num_opt_attemts++;
        if(num_opt_attemts >= max_opt_attempts) {
            LOG(FATAL) << "Max opt attempts reached";
        }

        gtsam::LevenbergMarquardtParams lm_params;
        lm_params.setMaxIterations(100);
        lm_params.verbosityLM = gtsam::LevenbergMarquardtParams::VerbosityLM::SUMMARY;

        lm_params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;
        // // // params.setRelativeErrorTol(-std::numeric_limits<double>::max());
        // // // params.setAbsoluteErrorTol(-std::numeric_limits<double>::max());
        // gtsam::LevenbergMarquardtOptimizer opt(graph, initial, params);


        try {

            double error_before = new_factors_.error(initial_values);
            LOG(INFO) << "Error before: " << error_before;

            gtsam::LevenbergMarquardtOptimizer opt(new_factors_, initial_values, lm_params);

            state_ = opt.optimize();
            new_values_ = initial_values; //update the new_values to be ther ones that were used as the initial values for the optimziation that solved

            double error_after = new_factors_.error(state_);
            LOG(INFO) << "Error after " << error_after;

            solved = true;
            return;

        }
        catch(const gtsam::CheiralityException& e) {
            const gtsam::Key nearby_variable = e.nearbyVariable();
            gtsam::FactorIndices associated_factors; //use idnex's as using the iterators will rearrange the graph everytime, therefore making the other iterators incorrect
            factor_graph_tools::getAssociatedFactors(associated_factors, new_factors_, nearby_variable);

            gtsam::Symbol static_symbol(nearby_variable);
            CHECK(static_symbol.chr() == kStaticLandmarkSymbolChar);

            TrackletId nearby_tracklet = (TrackletId)static_symbol.index();
            marked_for_cherality.insert2(nearby_tracklet, true);


            LOG(WARNING) << "gtsam::CheiralityException throw at variable " << DynoLikeKeyFormatter(nearby_variable) << ". Removing factors: " << associated_factors.size();

            for(gtsam::FactorIndex index: associated_factors) {
                //do not erase (this will reannrange the factors)
                new_factors_.remove(index);
            }

            //even if failure update the current set of values as these should be closer to the optimal so hopefully less recursive iterations
            // initial_values = opt.values();
            //Apparently also removing the variable results in a ValueKeyNotExists exception, even though all the factors have been
            //removed. Maybe the factor graph/Values keeps a cache of variables?
            initial_values.erase(nearby_variable);

            //make new graph with no non-null factors in it
            //graph.error will segfault with null values in it!
            gtsam::NonlinearFactorGraph replacemenent_graph;
            for(auto f : new_factors_) {
                if(f) replacemenent_graph.push_back(f);
            }

            // new_factors_.clear();
            new_factors_ = replacemenent_graph;

            handle_cheriality_optimize();
            return;

        }
    };

    if(current_frame_id % 30 == 0) {
        handle_cheriality_optimize();
    }


    if(is_first) {
        is_first = false;
    }




}


} //dyno
