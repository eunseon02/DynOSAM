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
#pragma once

#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/common/Map.hpp"

#include "dynosam/backend/DynoISAM2.hpp"
#include "dynosam/common/Flags.hpp"


#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/ISAM2.h>

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

namespace dyno {


using RGBDBackendModuleTraits = BackendModuleTraits<RGBDInstanceOutputPacket, Landmark>;

class RGBDBackendModule : public BackendModuleType<RGBDBackendModuleTraits> {

public:
    using Base = BackendModuleType<RGBDBackendModuleTraits>;
    using RGBDMap = Base::MapType;
    using RGBDOptimizer = Base::OptimizerType;

    enum UpdaterType {
        MotionInWorld,
    };

    RGBDBackendModule(const BackendParams& backend_params, RGBDMap::Ptr map, RGBDOptimizer::Ptr optimizer, const UpdaterType& updater_type, ImageDisplayQueue* display_queue = nullptr);
    ~RGBDBackendModule();

    using SpinReturn = Base::SpinReturn;

    //TODO: move to optimizer and put into pipeline manager where we know the type and bind write output to shutdown procedure
    void saveGraph(const std::string& file = "rgbd_graph.dot");
    void saveTree(const std::string& file = "rgbd_bayes_tree.dot");

    std::tuple<gtsam::Values, gtsam::NonlinearFactorGraph>
    constructGraph(FrameId from_frame, FrameId to_frame, bool set_initial_camera_pose_prior);

//TODO: for now
public:
    SpinReturn boostrapSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;
    SpinReturn nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;

    struct UpdateObservationParams {
        //! If true, vision related updated will backtrack to the start of a new tracklet and all the measurements to the graph
        //! should make false in batch case where we want to be explicit about which frames are added!
        bool do_backtrack = false;
        mutable DebugInfo::Optional debug_info{}; //TODO debug info should go into a map per frame? in UpdateObservationResult

    };

    struct ConstructGraphOptions : public UpdateObservationParams {
        FrameId from_frame{0u};
        FrameId to_frame{0u};
        bool set_initial_camera_pose_prior = true;
    };

    struct UpdateObservationResult {
        std::set<FrameId> frames_updated;
        std::set<ObjectId> objects_affected;

        inline UpdateObservationResult& operator+=(const UpdateObservationResult& oth) {
            frames_updated.insert(oth.frames_updated.begin(), oth.frames_updated.end());
            objects_affected.insert(oth.objects_affected.begin(), oth.objects_affected.end());
            return *this;
        }
    };

    class Updater {

    public:

        DYNO_POINTER_TYPEDEFS(Updater)

        RGBDBackendModule* parent_;
        Updater(RGBDBackendModule* parent) : parent_(CHECK_NOTNULL(parent)) {}

        //  //adds pose to the new values and a prior on this pose to the new_factors
        void setInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values);

        //TODO: specify noise model
        void setInitialPosePrior(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::NonlinearFactorGraph& new_factors);

        void addOdometry(FrameId from_frame, FrameId to_frame, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);

        void addOdometry(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors);

        UpdateObservationResult updateStaticObservations(
            FrameId from_frame,
            FrameId to_frame,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        UpdateObservationResult updateDynamicObservations(
            FrameId from_frame,
            FrameId to_frame,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);


        UpdateObservationResult updateStaticObservations(
            FrameId frame_id_k,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        UpdateObservationResult updateDynamicObservations(
            FrameId frame_id_k,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        //log everything all frames!!
        void logBackendFromMap(BackendLogger& logger);

        // //we need a separate way of tracking if a dynamic tracklet is in the map, since each point is modelled uniquely
        // //simply used as an O(1) lookup, the value is not actually used. If the key exists, we assume that the tracklet is in the map
        gtsam::FastMap<gtsam::Key, bool> is_other_values_in_map; //! the set of (static related) values managed by this updater. Allows checking if values have already been added over successifve function calls
        gtsam::FastMap<TrackletId, bool> is_dynamic_tracklet_in_map_; //! thr set of dynamic points that have been added by this updater. We use a separate map containing the tracklets as the keys are non-unique

        auto getMap() { return parent_->getMap(); }

    };

public:
    bool buildSlidingWindowOptimisation(FrameId frame_k, gtsam::Values& optimised_values, double& error_before, double& error_after);


public:
    // std::unique_ptr<DynoISAM2> smoother_;
    // DynoISAM2Result smoother_result_;
    // std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;
    // UpdateImpl::UniquePtr updater_;
    Updater::UniquePtr new_updater_;
    FrameId first_frame_id_; //the first frame id that is received

    //logger here!!
    BackendLogger::UniquePtr logger_{nullptr};
    gtsam::FastMap<FrameId, gtsam::Pose3> initial_camera_poses_; //! Camera poses as estimated from the frontend per frame


    //base backend module does not correctly share properties between mono and rgbd (i.e static_pixel_noise_ is in backend module but is not used in this class)
    //TODO: really need a rgbd and mono base class for this reason
    gtsam::SharedNoiseModel static_point_noise_; //! 3d isotropic pixel noise on static points
    gtsam::SharedNoiseModel dynamic_point_noise_; //! 3d isotropic pixel noise on dynamic points

    DebugInfo debug_info_;



};

} //dyno
