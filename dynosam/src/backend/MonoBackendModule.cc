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
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/frontend/MonoInstance-Definitions.hpp"

#include "dynosam/utils/GtsamUtils.hpp"

#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_bool(run_as_graph_file_only, false, "If true values will be saved to a graph file to for unit testing.");
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
}

MonoBackendModule::~MonoBackendModule() {
    // std::ofstream graph_logger;
    // LOG(INFO) << "Writing to " << FLAGS_backend_graph_file;
    // graph_logger.open(FLAGS_backend_graph_file);
    // graph_logger << fg_ss_.str();
    // graph_logger.close();
    state_graph_.saveGraph("/root/results/DynoSAM/mono_backend_graph.dot");
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
    const FrameId current_frame_id =  input->getFrameId();

    if(FLAGS_run_as_graph_file_only) {
        saveAllToGraphFile(input);
        return {State::Nominal, nullptr};
    }

    addInitialPose(T_world_camera_measured, current_frame_id, state_, state_graph_);
    updateSmartStaticObservations(input->static_keypoint_measurements_, current_frame_id, new_smart_factors, updated_smart_factors, new_projection_measurements);
    addNewSmartStaticFactors(new_smart_factors, state_graph_);
    // updateStaticObservations(input->static_keypoint_measurements_, current_frame_id, new_point_values, new_smart_factors, new_projection_factors, smart_factors_to_convert);
    // convertAndDeleteSmartFactors(new_values, smart_factors_to_convert, new_factors);
    // addToStatesStructures(new_values, new_factors, new_smart_factors);
    return {State::Nominal, nullptr};
}

MonoBackendModule::SpinReturn MonoBackendModule::monoNominalSpin(MonocularInstanceOutputPacket::ConstPtr input) {
    CHECK(input);

    TrackletIds new_smart_factors;
    TrackletIds updated_smart_factors;
    StatusKeypointMeasurements new_projection_measurements;

    const gtsam::Pose3 T_world_camera_measured = input->T_world_camera_;
    const FrameId current_frame_id =  input->getFrameId();

    if(FLAGS_run_as_graph_file_only) {
        saveAllToGraphFile(input);
        return {State::Nominal, nullptr};
    }

    addOdometry(T_world_camera_measured, current_frame_id, current_frame_id-1, state_, state_graph_);

    //process should be
    //1. make new smart factors for new measurements
    //2. Iterate over smart factors and update or convert
    //3. Update projection factors
    updateSmartStaticObservations(input->static_keypoint_measurements_, current_frame_id, new_smart_factors, updated_smart_factors, new_projection_measurements);
    addNewSmartStaticFactors(new_smart_factors, state_graph_);

    gtsam::Values new_and_current_values(state_);
    tryTriangulateExistingSmartStaticFactors(updated_smart_factors, new_and_current_values,state_, state_graph_);
    addStaticProjectionMeasurements(current_frame_id, new_projection_measurements, state_graph_);
    // updateStaticObservations(input->static_keypoint_measurements_, current_frame_id, new_point_values, new_smart_factors, new_projection_factors, smart_factors_to_convert);
    // convertAndDeleteSmartFactors(new_values, smart_factors_to_convert, new_factors);
    // addToStatesStructures(new_values, new_factors, new_smart_factors);
    return {State::Nominal, nullptr};
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

            //update smart factor
            SmartProjectionFactor::shared_ptr smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);
            factor_graph_tools::addSmartProjectionMeasurement(smart_factor, kp, frame_id);
            //this will also include new factors
            //this might be problematic when we get to isam2 -> we need this information to tell isam2 factors have new measurements
            //but we dont want to do this with the new factors (as they will get added that iteration anyway)
            updated_smart_factors.push_back(tracklet_id);

        }
        else if(factor_type == ProjectionFactorType::PROJECTION) {
            CHECK(!smart_factor_map_.exists(tracklet_id)) << "Factor has been marked as projection but exists in the smart factor map. It should have been removed.";
            //sanity check that we DO have a point for this factor yet
            CHECK(state_.exists(lmk_symbol)) << "Factor has been marked as projection but there is no lmk in the state vector.";
            new_projection_measurements.push_back(static_measurement);
        }

    }
}


void MonoBackendModule::addNewSmartStaticFactors(const TrackletIds& new_smart_factors,  gtsam::NonlinearFactorGraph& factors) {
    for(const TrackletId tracklet_id : new_smart_factors) {
        CHECK(smart_factor_map_.exists(tracklet_id)) << "Smart factor with tracklet id " << tracklet_id
                << " has been marked as a new factor but is not in the smart_factor_map";
        SmartProjectionFactor::shared_ptr smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);

        Slot& slot = smart_factor_map_.getSlot(tracklet_id);
        CHECK_EQ(slot, -1) << "Smart factor with tracklet id " << tracklet_id
                << " has been marked as a new factor but slot is not -1";

        size_t current_slot = factors.size();
        factors.push_back(smart_factor);

        slot = current_slot; //update slot
    }
}

void MonoBackendModule::tryTriangulateExistingSmartStaticFactors(const TrackletIds& updated_smart_factors, const gtsam::Values& new_and_current_state, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& factors) {
    for(const TrackletId tracklet_id : updated_smart_factors) {
        CHECK(smart_factor_map_.exists(tracklet_id)) << "Smart factor with tracklet id " << tracklet_id
                << " has been updated but is not in the smart_factor_map";

        const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);
        CHECK(!new_and_current_state.exists(lmk_symbol) && !state_.exists(lmk_symbol)) << "Factor has been marked as smart a lmk value exists for it";


        SmartProjectionFactor::shared_ptr smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);
        if(smart_factor->size() >= 2u) {
            //try and triangulate the point using the new and current values. This should contain the current pose
          gtsam::TriangulationResult triangulation_result = smart_factor->point(new_and_current_state);
          if(triangulation_result) {
            CHECK(triangulation_result.valid());
            const gtsam::Point3 point = triangulation_result.value();
            convertSmartToProjectionFactor(tracklet_id, point, new_values, factors);
          }
        }
    }
}

bool MonoBackendModule::convertSmartToProjectionFactor(const TrackletId smart_factor_to_convert, const gtsam::Point3& lmk_world, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& factors) {
    const TrackletId tracklet_id = smart_factor_to_convert;
    SmartProjectionFactor::shared_ptr smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);
    CHECK(smart_factor->point().valid());
    const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);


    Slot slot = smart_factor_map_.getSlot(tracklet_id);
    CHECK(slot != UninitialisedSlot) << "Trying to delete and convert smart factor with tracklet id " << tracklet_id << " but the slot is -1";

    {
        SmartProjectionFactor::shared_ptr smart = boost::dynamic_pointer_cast<SmartProjectionFactor>(factors[slot]);
        CHECK(smart);
        CHECK(smart == smart_factor);
        CHECK(smart->equals(*smart_factor));
    }
    //TODO: this will NOT work with incremental as we need to actually remove by slot!!
    factors.remove(slot);

    new_values.insert(lmk_symbol, lmk_world);
    for (size_t i = 0; i < smart_factor->keys().size(); i++) {
        const gtsam::Symbol& pose_symbol = gtsam::Symbol(smart_factor->keys().at(i));
        const auto& measured = smart_factor->measured().at(i);

        factors.emplace_shared<GenericProjectionFactor>(
            measured,
            static_smart_noise_,
            pose_symbol,
            lmk_symbol,
            gtsam_calibration_
        );
    }

    //remvoe from smart_factor_map_
    smart_factor_map_.erase(tracklet_id);

    //update status
    tracklet_to_status_map_.at(tracklet_id).pf_type_ = ProjectionFactorType::PROJECTION;

    return true;
}


void MonoBackendModule::addStaticProjectionMeasurements(const FrameId frame_id, const StatusKeypointMeasurements& new_projection_measurements, gtsam::NonlinearFactorGraph& factors) {
    for(const StatusKeypointMeasurement& static_measurement : new_projection_measurements) {
        const KeypointStatus& status = static_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = static_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::STATIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        CHECK(!smart_factor_map_.exists(tracklet_id)) << "Measurement with tracklet id " << tracklet_id
                    << " has been marked as a projection factor but is in the smart_factor_map";
        CHECK_EQ(tracklet_to_status_map_.at(tracklet_id).pf_type_, ProjectionFactorType::PROJECTION);
        factors.emplace_shared<GenericProjectionFactor>(
                kp,
                static_smart_noise_,
                CameraPoseSymbol(frame_id),
                StaticLandmarkSymbol(tracklet_id),
                gtsam_calibration_
            );

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

//             CHECK(!smart_factor_map_.exists(tracklet_id)) << "Smart factor with tracklet id " << tracklet_id
//                 << " exists in the smart factor map but not in the tracklet_to_status_map";

//             smart_factor_map_.add(tracklet_id, smart_factor, UninitialisedSlot);
//         }


//         //sanity check that the object label is still the same for the tracked object
//         CHECK_EQ(object_id, tracklet_to_status_map_.at(tracklet_id).object_id_);

//         const ProjectionFactorType factor_type = tracklet_to_status_map_.at(tracklet_id).pf_type_;
//         const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);

//         if(factor_type ==  ProjectionFactorType::SMART) {
//             CHECK(smart_factor_map_.exists(tracklet_id)) << "Factor has been marked as smart but does not exist in the smart factor map";
//             //sanity check that we dont have a point for this factor yet
//             CHECK(!new_values.exists(lmk_symbol) && !state_.exists(lmk_symbol)) << "Factor has been marked as smart a lmk value exists for it";


//             SmartProjectionFactor::shared_ptr smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);
//             factor_graph_tools::addSmartProjectionMeasurement(smart_factor, kp, frame_id);

//             Slot slot = smart_factor_map_.getSlot(tracklet_id);
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
//             CHECK(!smart_factor_map_.exists(tracklet_id)) << "Factor has been marked as projection but exists in the smart factor map. It should have been removed.";
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
//         Slot slot = smart_factor_map_.getSlot(tracklet);
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
//         //these should all be in the smart_factor_map_
//         //note: reference
//         Slot& slot = smart_factor_map_.getSlot(tracklet_id);
//         auto smart_factor = smart_factor_map_.getSmartFactor(tracklet_id);
//         CHECK(slot == UninitialisedSlot);

//         size_t current_slot = state_graph_.size();
//         state_graph_.push_back(smart_factor);

//         slot = current_slot;
//     }
// }


void MonoBackendModule::setFactorParams(const BackendParams& backend_params) {
    //set static projection smart noise
    static_smart_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, backend_params.smart_projection_noise_sigma_);
    CHECK(static_smart_noise_);

    gtsam::Vector6 odom_sigmas;
    odom_sigmas.head<3>().setConstant(backend_params.odometry_rotation_sigma_);
    odom_sigmas.tail<3>().setConstant(
        backend_params.odometry_translation_sigma_);
    odometry_noise_ = gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);
    CHECK(odometry_noise_);

    initial_pose_prior_ =  gtsam::noiseModel::Isotropic::Sigma(6u, 0.0001);
    CHECK(initial_pose_prior_);
}


void MonoBackendModule::saveAllToGraphFile(MonocularInstanceOutputPacket::ConstPtr input) {
    const FrameId frame_id = input->getFrameId();
    const auto& input_tracking_images = input->frame_.tracking_images_;
    const gtsam::Pose3 pose = input->T_world_camera_;

    CHECK(input->gt_packet_);
    GroundTruthInputPacket gt_packet = input->gt_packet_.value();
    const gtsam::Pose3 gt_pose = gt_packet.X_world_;




    gtsam::Key camera_key = CameraPoseSymbol(frame_id);

    if(previous_input_ == nullptr) {
        const gtsam::Point3 p = gt_pose.translation();
        const auto q = gt_pose.rotation().toQuaternion();

        state_graph_.addPrior(CameraPoseSymbol(frame_id), gt_pose, initial_pose_prior_);
        fg_ss_ << "EDGE_SE3_PRIOR " << camera_key << " ";
        fg_ss_ << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " ";

        utils::saveNoiseModelAsUpperTriangular(fg_ss_, static_cast<const gtsam::noiseModel::Gaussian&>(*initial_pose_prior_));
        fg_ss_ << "\n";
    }
    else {

        const gtsam::Pose3 prev_pose = previous_input_->T_world_camera_;
        gtsam::Key prev_camera_key = CameraPoseSymbol(previous_input_->getFrameId());

        CHECK(previous_input_->gt_packet_);
        GroundTruthInputPacket prev_gt_packet = previous_input_->gt_packet_.value();
        const gtsam::Pose3 prev_gt_pose = prev_gt_packet.X_world_;

        const gtsam::Pose3 gt_odom = prev_gt_pose.inverse() * gt_pose;
        const gtsam::Point3 p = gt_odom.translation();
        const auto q = gt_odom.rotation().toQuaternion();
        fg_ss_ << "EDGE_SE3:QUAT " << prev_camera_key << " " << camera_key << " " << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " ";
        utils::saveNoiseModelAsUpperTriangular(fg_ss_, static_cast<const gtsam::noiseModel::Gaussian&>(*odometry_noise_));
        fg_ss_ << "\n";

        factor_graph_tools::addBetweenFactor(previous_input_->getFrameId(), frame_id, gt_odom, odometry_noise_, state_graph_);
    }

    {
        //write camera pose
        const gtsam::Point3 p = gt_pose.translation();
        const auto q = gt_pose.rotation().toQuaternion();
        fg_ss_ << "VERTEX_SE3:QUAT " << camera_key << " " << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        state_.insert(CameraPoseSymbol(frame_id), gt_pose);
    }

    //lets use the state vector to track initalisation values
    const StatusKeypointMeasurements& static_measurements = input->static_keypoint_measurements_;
    for(const StatusKeypointMeasurement& static_measurement : static_measurements) {
        const KeypointStatus& status = static_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = static_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::STATIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        gtsam::Key static_point_key = StaticLandmarkSymbol(tracklet_id);
        if(!state_.exists(static_point_key)) {
            Landmark lmk_w = gt_pose * input->frame_.backProjectToCamera(tracklet_id);
            state_.insert(static_point_key, lmk_w);
            fg_ss_ << "VERTEX_TRACKXYZ " << static_point_key << " " << lmk_w(0) << " " << lmk_w(1) << " " << lmk_w(2) << "\n";
        }

        fg_ss_ << "EDGE_2D_PROJECTION " << camera_key << " " << static_point_key << " " << kp(0) << " " << kp(1);
        utils::saveNoiseModelAsUpperTriangular(fg_ss_, static_cast<const gtsam::noiseModel::Gaussian&>(*static_smart_noise_));
        fg_ss_ << "\n";

        state_graph_.emplace_shared<GenericProjectionFactor>(
                kp,
                static_smart_noise_,
                camera_key,
                static_point_key,
                gtsam_calibration_
            );
    }

    previous_input_ = input;

}


} //dyno
