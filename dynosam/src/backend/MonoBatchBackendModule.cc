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

#include "dynosam/backend/MonoBatchBackendModule.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"

#include "dynosam/utils/GtsamUtils.hpp"

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>

DEFINE_bool(use_gt_camera_pose, true, "If true gt camera pose (with noise) will be used as initalisation.");
DEFINE_double(dp_init_sigma_perturb, 0.1, "Amount of noise to add to each dynamic point when initalising it.");


namespace dyno {

MonoBatchBackendModule::MonoBatchBackendModule(const BackendParams& backend_params, Camera::Ptr camera)
    :   BackendModule(backend_params, camera) {


    pose_init_func_ = [=](FrameId frame_id, const gtsam::Pose3& measurement, const GroundTruthInputPacket& gt_packet) {
        CHECK_EQ(frame_id, gt_packet.frame_id_);

        if(FLAGS_use_gt_camera_pose) {
            return gt_packet.X_world_;
        }
        else {
            return measurement;
        }
    };


    dynamic_point_init_func_ = [=](
        FrameId frame_id, TrackletId tracklet_id, ObjectId object_id, const Keypoint& measurement, const GroundTruthInputPacket& gt_packet) {

        const auto& camera_params = camera_->getParams();

        ObjectPoseGT object_pose_gt;
        CHECK(gt_packet.getObject(object_id, object_pose_gt));

        //for now, just perturb with noise from gt pose
        auto gt_pose_camera = utils::perturbWithNoise(
            object_pose_gt.L_camera_,
            FLAGS_dp_init_sigma_perturb);

        //Z coordinate in camera frame
        const double Z = gt_pose_camera.z();

        //Convert (distorted) image coordinates uv to intrinsic coordinates xy
        gtsam::Point2 pc = gtsam_calibration_->calibrate(measurement);
        //projection equation x = f(X/Z) and we have normalized coordinates, x, and now Z
        //rearranging for X (and then also Y) -> X = x/f * Z
        const double X = pc(0) * Z;
        const double Y = pc(1) * Z;

        //gt for camera pose?
        return gt_packet.X_world_ * gtsam::Vector3(X, Y, Z); //back into world frame

    };

    motion_init_func_ = [=](
        FrameId frame_id, ObjectId object_id, const GroundTruthInputPacket& gt_packet
    )
    {
        return gtsam::Pose3::Identity();
    };


}

MonoBatchBackendModule::~MonoBatchBackendModule() {}

MonoBatchBackendModule::SpinReturn MonoBatchBackendModule::boostrapSpin(BackendInputPacket::ConstPtr input) {
    MonocularInstanceOutputPacket::ConstPtr mono_output = safeCast<BackendInputPacket, MonocularInstanceOutputPacket>(input);
    checkAndThrow((bool)mono_output, "Failed to cast BackendInputPacket to MonocularInstanceOutputPacket in MonoBackendModule");

    return monoBoostrapSpin(mono_output);

}
MonoBatchBackendModule::SpinReturn MonoBatchBackendModule::nominalSpin(BackendInputPacket::ConstPtr input) {
    MonocularInstanceOutputPacket::ConstPtr mono_output = safeCast<BackendInputPacket, MonocularInstanceOutputPacket>(input);
    checkAndThrow((bool)mono_output, "Failed to cast BackendInputPacket to MonocularInstanceOutputPacket in MonoBackendModule");

    return monoNominalSpin(mono_output);
}

MonoBatchBackendModule::SpinReturn MonoBatchBackendModule::monoBoostrapSpin(MonocularInstanceOutputPacket::ConstPtr input) {

    const FrameId current_frame_id = input->getFrameId();
    CHECK(input->gt_packet_);
    const GroundTruthInputPacket gt_packet = input->gt_packet_.value();
    gt_packet_map_.insert({current_frame_id, gt_packet});

    const gtsam::Pose3 T_world_camera = pose_init_func_(current_frame_id, input->T_world_camera_, gt_packet);
    gtsam::Key current_pose_symbol = CameraPoseSymbol(current_frame_id);
    gtsam::Key previous_pose_symbol = CameraPoseSymbol(current_frame_id - 1u);

    //add state
    new_values_.insert(current_pose_symbol, T_world_camera);
    new_factors_.addPrior(current_pose_symbol, T_world_camera, initial_pose_prior_);

    return {State::Nominal, nullptr};
}

MonoBatchBackendModule::SpinReturn MonoBatchBackendModule::monoNominalSpin(MonocularInstanceOutputPacket::ConstPtr input) {
    CHECK(input);
    const FrameId current_frame_id =  input->getFrameId();
    LOG(INFO) << "Running backend on frame " << current_frame_id;
    CHECK(input->gt_packet_);
    const GroundTruthInputPacket gt_packet = input->gt_packet_.value();
    gt_packet_map_.insert({current_frame_id, gt_packet});

    const gtsam::Pose3 T_world_camera = pose_init_func_(current_frame_id, input->T_world_camera_, gt_packet);
    gtsam::Key current_pose_symbol = CameraPoseSymbol(current_frame_id);
    gtsam::Key previous_pose_symbol = CameraPoseSymbol(current_frame_id - 1u);


    LOG(INFO) << "Adding odom";
    new_values_.insert(current_pose_symbol, T_world_camera);
    // new_factors_.addPrior(current_pose_symbol, T_world_camera, initial_pose_prior_);
    gtsam::Pose3 previous_pose;
    //if prev_pose_symbol is in state, use this to construct the btween factor
    if(state_.exists(previous_pose_symbol)) {
        previous_pose = state_.at<gtsam::Pose3>(previous_pose_symbol);
    }
    else if(new_values_.exists(previous_pose_symbol)) {
        previous_pose = new_values_.at<gtsam::Pose3>(previous_pose_symbol);
    }
    else {
        LOG(FATAL) << "Shoudl have prev pose";
    }

    const gtsam::Pose3 odom = previous_pose.inverse() * T_world_camera;
    factor_graph_tools::addBetweenFactor(current_frame_id-1u, current_frame_id, odom, odometry_noise_, new_factors_);

    runStaticUpdate(current_frame_id, current_pose_symbol, input->static_keypoint_measurements_);
    runDynamicUpdate(current_frame_id, current_pose_symbol, input->dynamic_keypoint_measurements_, input->estimated_motions_);

    int num_opt_attemts = 0;
    const int max_opt_attempts = 300;
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
            // marked_for_cherality.insert2(nearby_tracklet, true);


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

    if(current_frame_id % 15 == 0) {
        handle_cheriality_optimize();
    }


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

    StatusLandmarkEstimates initial_dynamic_lmks;
    auto extracted_dynamic_points = new_values_.extract<gtsam::Point3>(gtsam::Symbol::ChrTest(kDynamicLandmarkSymbolChar));
    for(const auto&[key, point] : extracted_dynamic_points) {
        DynamicPointSymbol dps(key);

        const TrackletId tracklet_id = dps.trackletId();
        const auto object_id = do_tracklet_manager_.getObjectIdByTracklet(tracklet_id);

        auto estimate = std::make_pair(tracklet_id, point);

        LandmarkStatus status;
        status.label_ = object_id;
        status.method_ = LandmarkStatus::Method::TRIANGULATED;

        initial_dynamic_lmks.push_back(std::make_pair(status, estimate));

    }



    StatusLandmarkEstimates all_dynamic_object_triangulation;
    auto extracted_dynamic_points_opt = state_.extract<gtsam::Point3>(gtsam::Symbol::ChrTest(kDynamicLandmarkSymbolChar));
    for(const auto&[key, point] : extracted_dynamic_points_opt) {
        DynamicPointSymbol dps(key);

        const TrackletId tracklet_id = dps.trackletId();
        const FrameId frame_id = dps.frameId();

        if(frame_id % 10 != 0) {continue;}
        const auto object_id = do_tracklet_manager_.getObjectIdByTracklet(tracklet_id);

        auto estimate = std::make_pair(tracklet_id, point);

        LandmarkStatus status;
        status.label_ = object_id;
        status.method_ = LandmarkStatus::Method::TRIANGULATED;

        all_dynamic_object_triangulation.push_back(std::make_pair(status, estimate));

    }

    gtsam::FastMap<ObjectId, gtsam::Pose3Vector> object_poses_composed_;
    for(const auto& [object_id, ref_estimate] : input->estimated_motions_) {
        object_poses_composed_.insert({object_id, gtsam::Pose3Vector{}});

        FrameIds all_frames = do_tracklet_manager_.getFramesPerObject(object_id);
        LOG(INFO) << "Adding output for object " << object_id << " appearing at frame " << container_to_string(all_frames);

        //TODO: we should do it like this, but if there is a problem with the tracking or somegthing (see object 4 (kitti seq 04 between frames 3-4))
        // the object data is not tracked properly and the frames the objects are seen in do not correspond with the motions added here
        // for(size_t i = 1; i < all_frames.size(); i++) {
        //     //assumes all frames contain a continuous set of frames (e.g. 2, 3, 4...)
        //     FrameId frame_id = all_frames.at(i);
        //     gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, frame_id);

        //     if(!state_.exists(motion_symbol)) {
        //         continue;
        //     }
        //     LOG(INFO) << i << " " << frame_id;

        //     if(i == 1) {
        //         ObjectPoseGT object_pose_gt;
        //         CHECK(gt_packet_map_.exists(frame_id - 1));
        //         auto gt_frame_packet = gt_packet_map_.at(frame_id-1);
        //         if(!gt_frame_packet.getObject(object_id, object_pose_gt)) {
        //             LOG(FATAL) << "Could not find gt object at frame " << frame_id-1 << " for object Id" << object_id;
        //         }
        //         CHECK_EQ(object_poses_composed_.at(object_id).size(), 0u);
        //         object_poses_composed_.at(object_id).push_back(object_pose_gt.L_world_);
        //     }
        //     else {
        //         gtsam::Pose3 motion = state_.at<gtsam::Pose3>(motion_symbol);
        //         CHECK_GT(object_poses_composed_.at(object_id).size(), 0u);
        //         gtsam::Pose3 object_pose = motion * (object_poses_composed_.at(object_id).back()); //object pose at the current frame, composed from the previous motion
        //         // object_poses_composed_.at(object_id).push_back(object_pose);
        //     }
        // }

        FrameIds frame_ids; //the actual frame ids the object appears (in the graph) in -> this SHOULD be the same as seen from the frontend but in some cases this is different (unsure why yet!)
        for(size_t i = 1; i < all_frames.size(); i++) {
            //assumes all frames contain a continuous set of frames (e.g. 2, 3, 4...)
            FrameId frame_id = all_frames.at(i);
            gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, frame_id);
            if(state_.exists(motion_symbol)) {

                //sanity check
                if(frame_ids.size() > 0) {
                    //check in order
                    auto prev_frame_id = frame_ids.back();
                    CHECK_EQ(prev_frame_id + 1, frame_id);
                }

                frame_ids.push_back(frame_id);
            }

        }
        LOG(INFO) << "Adding output for object " << object_id << " which are in graph for frames " << container_to_string(frame_ids);

        //we can start this from 0 as we iterated over the "all seen frames" starting at 1
        for(size_t i = 0; i < frame_ids.size(); i++) {
            FrameId frame_id = frame_ids.at(i);
            gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, frame_id);

            if(i == 0) {
                ObjectPoseGT object_pose_gt;
                CHECK(gt_packet_map_.exists(frame_id - 1));
                auto gt_frame_packet = gt_packet_map_.at(frame_id-1);
                if(!gt_frame_packet.getObject(object_id, object_pose_gt)) {
                    LOG(FATAL) << "Could not find gt object at frame " << frame_id-1 << " for object Id" << object_id;
                }
                CHECK_EQ(object_poses_composed_.at(object_id).size(), 0u);
                object_poses_composed_.at(object_id).push_back(object_pose_gt.L_world_);
            }
            else {
                gtsam::Pose3 motion = state_.at<gtsam::Pose3>(motion_symbol);
                CHECK_GT(object_poses_composed_.at(object_id).size(), 0u);
                gtsam::Pose3 object_pose = motion * (object_poses_composed_.at(object_id).back()); //object pose at the current frame, composed from the previous motion
                object_poses_composed_.at(object_id).push_back(object_pose);
            }
        }

        LOG(INFO) << "Added " <<  object_poses_composed_.at(object_id).size() << " composed poses for object " << object_id;

        // //this is a lot of iteration for what should be a straighforward lookup
        // const TrackletIds tracklet_ids = do_tracklet_manager_.getPerObjectTracklets(object_id);

        // for(TrackletId tracklet_id : tracklet_ids) {
        //     CHECK(do_tracklet_manager_.trackletExists(tracklet_id));

        //     const DynamicObjectTracklet<Keypoint>& tracklet = do_tracklet_manager_.getByTrackletId(tracklet_id);

        //     for(const auto& [frame_id, measurement] : tracklet) {
        //         (void)measurement;
        //         DynamicPointSymbol dynamic_point_symbol = DynamicLandmarkSymbol(frame_id, tracklet_id);

        //         if(state_.exists(dynamic_point_symbol)) {
        //             const Landmark lmk = state_.at<Landmark>(dynamic_point_symbol);
        //             // const Landmark lmk = new_values_.at<Landmark>(dynamic_point_symbol);
        //             auto lmk_estimate = std::make_pair(tracklet_id, lmk);

        //             LandmarkStatus status;
        //             status.label_ = object_id;
        //             status.method_ = LandmarkStatus::Method::OPTIMIZED;

        //             all_dynamic_object_triangulation.push_back(std::make_pair(status, lmk_estimate));
        //         }

        //     }

        // }

        LOG(INFO) << "Done adding points";

    }

    gtsam::Pose3Vector optimized_poses;
    auto extracted_poses = state_.extract<gtsam::Pose3>(gtsam::Symbol::ChrTest(kPoseSymbolChar));
    for(const auto&[key, pose] : extracted_poses) {
        (void)key;
        optimized_poses.push_back(pose);
    }

    auto backend_output = std::make_shared<BackendOutputPacket>();
    backend_output->timestamp_ = input->getTimestamp();
    backend_output->initial_dynamic_lmks_ = initial_dynamic_lmks;
    // backend_output->T_world_camera_ = state_.at<gtsam::Pose3>(CameraPoseSymbol(current_frame_id));
    backend_output->static_lmks_ = lmk_map;
    backend_output->object_poses_composed_ = object_poses_composed_;

    if(state_.exists(CameraPoseSymbol(current_frame_id))) {
        backend_output->T_world_camera_ = state_.at<gtsam::Pose3>(CameraPoseSymbol(current_frame_id));
    }
    backend_output->dynamic_lmks_ = all_dynamic_object_triangulation;
    backend_output->optimized_poses_ = optimized_poses;

    return {State::Nominal, backend_output};
}



void MonoBatchBackendModule::runStaticUpdate(
    FrameId current_frame_id,
    gtsam::Key pose_key,
    const StatusKeypointMeasurements& static_keypoint_measurements)
{



    //static points
    for(const StatusKeypointMeasurement& static_measurement : static_keypoint_measurements) {
        const KeypointStatus& status = static_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = static_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::STATIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        const gtsam::Symbol lmk_symbol = StaticLandmarkSymbol(tracklet_id);
        //first attempt to triangulate on this point before seeing if we can turn it ino a 3d lmk
        //if it does not exist here it cannot be in the graph as a projection factor
        if(!static_smart_factor_map.exists(tracklet_id))
        {

            double rank_tolerance = 1.0;
            //! max distance to triangulate point in meters
            double landmark_distance_threshold = 10.0;
            //! max acceptable reprojection error // before tuning: 3
            double outlier_rejection = 2;
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
            static_projection_params_.throwCheirality = false;
            static_projection_params_.verboseCheirality = false;

            // if the TrackletIdToProjectionStatus does not have this tracklet, then it should be the first time we have seen
            // it and therefore, should not be in any of the other data structures
            SmartProjectionFactor::shared_ptr smart_factor =
                factor_graph_tools::constructSmartProjectionFactor(
                    static_smart_noise_,
                    gtsam_calibration_,
                    static_projection_params_
                );

            static_smart_factor_map.insert2(tracklet_id, smart_factor);
        }

        //get smart factor and see if we can triangulate
        SmartProjectionFactor::shared_ptr smart_factor = static_smart_factor_map.at(tracklet_id);
        factor_graph_tools::addSmartProjectionMeasurement(smart_factor, kp, current_frame_id);



        //now check if we have already added this to the data structures via the values structure
        //if not, this means we should try and triangulate via the smart factor

        //not in graph
        if(!new_values_.exists(lmk_symbol)) {
            try {
                gtsam::TriangulationResult triangulation_result = smart_factor->point(new_values_);
                if(triangulation_result.valid()) {
                    CHECK(smart_factor->point());
                    CHECK(!smart_factor->isDegenerate());
                    CHECK(!smart_factor->isFarPoint());
                    CHECK(!smart_factor->isOutlier());
                    CHECK(!smart_factor->isPointBehindCamera());

                    new_values_.insert(lmk_symbol, triangulation_result.get());
                    // static_count++;


                    ProjectionFactorStatus projection_status(tracklet_id, ProjectionFactorType::PROJECTION, object_id);
                    //just so we can then add it to the output map which checks for projection
                    tracklet_to_status_map_.insert({tracklet_id, projection_status});

                    //convert old measurements to projection factors
                    //iterate over all keys in the factor and add them as projection factors
                    for (size_t i = 0; i < smart_factor->keys().size(); i++) {
                        const gtsam::Symbol& pose_symbol = gtsam::Symbol(smart_factor->keys().at(i));
                        const auto& measured = smart_factor->measured().at(i);

                        new_factors_.emplace_shared<GenericProjectionFactor>(
                            measured,
                            static_smart_noise_,
                            pose_symbol,
                            lmk_symbol,
                            gtsam_calibration_
                        );
                    }

                }
            }
            catch(const gtsam::CheiralityException&) {}

        }
        else {
            //already in graph
            auto projection_factor = boost::make_shared<GenericProjectionFactor>(
                kp,
                static_smart_noise_,
                pose_key,
                lmk_symbol,
                gtsam_calibration_,
                false, false
            );
            new_factors_.push_back(projection_factor); //do we need to check the chariality of this?
        }



    }
}

void MonoBatchBackendModule::runDynamicUpdate(
    FrameId current_frame_id,
    gtsam::Key pose_key,
    const StatusKeypointMeasurements& dynamic_keypoint_measurements,
    const DecompositionRotationEstimates& estimated_motions)
{
    //dynamic measurements
    std::set<TrackletId> set_tracklets;
    for(const StatusKeypointMeasurement& dynamic_measurement : dynamic_keypoint_measurements) {

        const KeypointStatus& status = dynamic_measurement.first;
        const KeyPointType kp_type = status.kp_type_;
        const KeypointMeasurement& measurement = dynamic_measurement.second;

        const ObjectId object_id = status.label_;

        CHECK(kp_type == KeyPointType::DYNAMIC);
        const TrackletId tracklet_id = measurement.first;
        const Keypoint& kp = measurement.second;

        set_tracklets.insert(tracklet_id);

        do_tracklet_manager_.add(object_id, tracklet_id, current_frame_id, kp);

    }

    CHECK(set_tracklets.size() == dynamic_keypoint_measurements.size());
    const size_t min_dynamic_obs = 3u;

    for(const auto& [object_id, ref_estimate] : estimated_motions) {
        //this will grow massively overtime?
        LOG(INFO) << "Looking at object " << object_id;
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

        int dynamic_count = 0;
        //new points to add to the graph - no points with this tracklet id should be added yet
        TrackletIds new_well_tracked_points;
        //points to add to the graph that previous existed (were modelled) at proceeding timesteps
        TrackletIds existing_well_tracked_points;

        for(TrackletId tracklet_id : tracklet_ids) {
            CHECK(do_tracklet_manager_.trackletExists(tracklet_id));

            // if(dynamic_count > 300) {
            //     break;
            // }

            DynamicObjectTracklet<Keypoint>& tracklet = do_tracklet_manager_.getByTrackletId(tracklet_id);

            const DynamicPointSymbol current_dynamic_point_symbol = DynamicLandmarkSymbol(current_frame_id, tracklet_id);
            const DynamicPointSymbol previous_dynamic_point_symbol = DynamicLandmarkSymbol(current_frame_id - 1u, tracklet_id);

            if(!tracklet.exists(current_frame_id)) { continue; }

            //   new tracklet Id
            if(!dynamic_in_graph_factor_map.exists(tracklet_id)) {
                dynamic_in_graph_factor_map.insert({tracklet_id, false});
            }

            if(dynamic_in_graph_factor_map.at(tracklet_id)) {
                //the previous point should be in the graph
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
            //up to and including the current frame
            for(size_t frame = starting_frame_points; frame <= current_frame_id; frame++) {
                DynamicPointSymbol dynamic_point_symbol = DynamicLandmarkSymbol(frame, tracked_id);
                const Keypoint& kp = tracklet.at(frame);


                CHECK(tracklet.exists(frame));
                gtsam::Symbol cam_symbol = CameraPoseSymbol(frame);

                //get object gt for the current frame
                ObjectPoseGT object_pose_gt;
                auto gt_frame_packet = gt_packet_map_.at(frame);
                if(!gt_frame_packet.getObject(object_id, object_pose_gt)) {
                    LOG(FATAL) << "Coudl not find gt object at frame " << frame << " for object Id " << object_id << " and packet " << gt_frame_packet;
                }

                Landmark lmk_world = dynamic_point_init_func_(
                    frame,
                    tracked_id,
                    object_id,
                    kp,
                    gt_frame_packet
                );

                if(frame == starting_frame_points) {
                    //add prior on first time this point is seen? This will add a prior on every (new) point which we dont want
                    new_factors_.addPrior(dynamic_point_symbol, lmk_world, gtsam::noiseModel::Isotropic::Sigma(3, 10));
                }

                // CHECK(!smoother_->valueExists(dynamic_point_symbol));
                CHECK(!new_values_.exists(dynamic_point_symbol));
                new_values_.insert(dynamic_point_symbol, lmk_world);

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

                auto gt_frame_packet = gt_packet_map_.at(frame);

                //motion that takes us from previous to current
                gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, frame);
                // // CHECK(!smoother_->valueExists(motion_symbol));
                // CHECK(!new_values_.exists(motion_symbol));
                if(!new_values_.exists(motion_symbol)) {

                    new_values_.insert(motion_symbol, motion_init_func_(
                        frame, object_id, gt_frame_packet
                    ));
                    safeAddConstantObjectVelocityFactor(frame, object_id, new_values_, new_factors_);
                    LOG(INFO) << "Added motion symbol " << DynoLikeKeyFormatter(motion_symbol);

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

        LOG(INFO) << "Updating " << existing_well_tracked_points.size() << " well tracked points";
        for(TrackletId tracked_id : existing_well_tracked_points) {
            const DynamicObjectTracklet<Keypoint>& tracklet = do_tracklet_manager_.getByTrackletId(tracked_id);
            const Keypoint& kp = tracklet.at(current_frame_id);

            const DynamicPointSymbol current_dynamic_point_symbol = DynamicLandmarkSymbol(current_frame_id, tracked_id);
            const DynamicPointSymbol previous_dynamic_point_symbol = DynamicLandmarkSymbol(current_frame_id - 1u, tracked_id);

            auto gt_frame_packet = gt_packet_map_.at(current_frame_id);
            const Landmark lmk_world = dynamic_point_init_func_(
                    current_frame_id,
                    tracked_id,
                    object_id,
                    kp,
                    gt_frame_packet
            );

            CHECK(!new_values_.exists(current_dynamic_point_symbol));
            CHECK(new_values_.exists(previous_dynamic_point_symbol));
            new_values_.insert(current_dynamic_point_symbol, lmk_world);

            auto projection_factor = boost::make_shared<GenericProjectionFactor>(
                kp,
                //note: using static noise
                static_smart_noise_,
                pose_key,
                current_dynamic_point_symbol,
                gtsam_calibration_
            );

            new_factors_.push_back(projection_factor);

            //motion that takes us from previous to current
            gtsam::Symbol motion_symbol = ObjectMotionSymbolFromCurrentFrame(object_id, current_frame_id);
            CHECK(new_values_.exists(ObjectMotionSymbolFromCurrentFrame(object_id, current_frame_id-1u)));

            if(!new_values_.exists(motion_symbol)) {

                new_values_.insert(motion_symbol, motion_init_func_(
                        current_frame_id, object_id, gt_frame_packet
                ));
                safeAddConstantObjectVelocityFactor(current_frame_id, object_id, new_values_, new_factors_);
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

}

bool MonoBatchBackendModule::safeAddConstantObjectVelocityFactor(FrameId current_frame, ObjectId object_id, const gtsam::Values& values, gtsam::NonlinearFactorGraph& factors) {
    //motion to take us from k-2 to k-1
    const auto prev_object_motion_key = ObjectMotionSymbol(object_id, current_frame-1u);
    //motion to take us from k-1 to k
    const auto curr_object_motion_key = ObjectMotionSymbol(object_id, current_frame);

    if(values.exists(prev_object_motion_key) && values.exists(curr_object_motion_key)) {
        CHECK(object_smoothing_noise_);
        CHECK_EQ(object_smoothing_noise_->dim(), 6u);

        factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            prev_object_motion_key,
            curr_object_motion_key,
            gtsam::Pose3::Identity(),
            object_smoothing_noise_
        );
        return true;
    }
    else {
        return false;
    }
}



} //dyno
