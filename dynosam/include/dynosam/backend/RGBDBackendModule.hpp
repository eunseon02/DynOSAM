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

//TODO: for now
public:
    SpinReturn boostrapSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;
    SpinReturn nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;

    // //TODO: taken from MonoBackendModule
    //  //adds pose to the new values and a prior on this pose to the new_factors
    // void addInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);

    // //T_world_camera is the estimate from the frontend and should be associated with the curr_frame_id
    // void addOdometry(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, FrameId frame_id_k_1, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);

    // void updateStaticObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors);
    // void updateDynamicObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors);



    // void optimize(FrameId frame_id_k, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors);

    // const ObjectPoseMap& updateObjectPoses(FrameId frame_id_k, const RGBDInstanceOutputPacket::ConstPtr input);

    //helper factor graph functions
    //call should happen on every frame
    // const gtsam::FastMap<ObjectId, gtsam::Pose3>& updateInitialObjectPoses(FrameId frame_id_k, const RGBDInstanceOutputPacket::ConstPtr input);

    // const ObjectPoseMap& updateObjectPoses(FrameId frame_id_k, const RGBDInstanceOutputPacket::ConstPtr input);

    struct UpdateImpl {
        DYNO_POINTER_TYPEDEFS(UpdateImpl)

        RGBDBackendModule* parent_;
        UpdateImpl(RGBDBackendModule* parent) : parent_(CHECK_NOTNULL(parent)) {}

        virtual std::string name() const = 0;

        //TODO: (jesse) shouldn't these all be const?
        virtual void setInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);
        virtual void updateOdometry(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, FrameId frame_id_k_1, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);

        virtual void updateStaticObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info) = 0;
        virtual void updateDynamicObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info) = 0;

        virtual void logBackendFromMap(FrameId frame_k, RGBDMap::Ptr map, BackendLogger& logger) = 0;

    };

    struct UpdateImplInWorld : public UpdateImpl {
        UpdateImplInWorld(RGBDBackendModule* parent) : UpdateImpl(parent) {}

        void updateStaticObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info = {}) override;
        virtual void updateDynamicObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info = {}) override;
        virtual void logBackendFromMap(FrameId frame_k, RGBDMap::Ptr map, BackendLogger& logger) override;

        inline std::string name() const override { return "rgbd_motion_world"; }
    };


    struct UpdateImplInWorldPrimitives : public UpdateImplInWorld {
        UpdateImplInWorldPrimitives(RGBDBackendModule* parent) : UpdateImplInWorld(parent) {}

        virtual void updateDynamicObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info = {}) override;
        virtual void logBackendFromMap(FrameId frame_k, RGBDMap::Ptr map, BackendLogger& logger) override;

        inline std::string name() const override { return "rgbd_motion_world_primitive"; }
    };

    // struct UpdateImplInWorldObjectPoses : public UpdateImplInWorldPrimitives {
    //     UpdateImplInWorldObjectPoses(RGBDBackendModule* parent) : UpdateImplInWorldPrimitives(parent) {}

    //     virtual void updateDynamicObservations(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors, DebugInfo::Optional debug_info = {}) override;

    //     inline std::string name() const override { return "rgbd_motion_world_object_poses"; }
    // };

public:
    // std::unique_ptr<DynoISAM2> smoother_;
    // DynoISAM2Result smoother_result_;
    // std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;
    UpdateImpl::UniquePtr updater_;

    //logger here!!
    BackendLogger::UniquePtr logger_{nullptr};



    using KeyTimestampMap = gtsam::IncrementalFixedLagSmoother::KeyTimestampMap;
    KeyTimestampMap timestamp_map_;
    // gtsam::Values new_values_;
    // gtsam::NonlinearFactorGraph new_factors_;

    gtsam::FastMap<ObjectId, gtsam::Pose3> initial_object_poses_; //constructed from the input

    //! A mapping of object to the frame id of when the initial object pose was set
    //! Used to ensure consistency when propogating object poses
    gtsam::FastMap<ObjectId, FrameId> initial_object_poses_set_;
    ObjectPoseMap composed_object_poses_;

    //base backend module does not correctly share properties between mono and rgbd (i.e static_pixel_noise_ is in backend module but is not used in this class)
    //TODO: really need a rgbd and mono base class for this reason
    gtsam::SharedNoiseModel static_point_noise_; //! 3d isotropic pixel noise on static points
    gtsam::SharedNoiseModel dynamic_point_noise_; //! 3d isotropic pixel noise on dynamic points

    //we need a separate way of tracking if a dynamic tracklet is in the map, since each point is modelled uniquely
    //simply used as an O(1) lookup, the value is not actually used. If the key exists, we assume that the tracklet is in the map
    gtsam::FastMap<TrackletId, bool> is_dynamic_tracklet_in_map_;

    //internal datastructure to store which objects were added (to the values and now appear in the map)
    //at THIS frame and for which frames this object is new
    //in the case that the object already has a history, the set should contain only one ite, which is this frame
    //in the case that the object has just been added, it will be added for N frames, since we go back and add object points
    //this is cleared everry spin
    //generally this refers to which frame object measurements will be added as factors to the map
    //but in the world-centric case, this also related to values
    gtsam::FastMap<ObjectId, std::set<FrameId>> new_object_measurements_per_frame_;

    DebugInfo debug_info_;



};

} //dyno
