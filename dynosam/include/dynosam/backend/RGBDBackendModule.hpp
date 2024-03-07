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

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/ISAM2.h>

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

namespace dyno {


using RGBDBackendModuleTraits = BackendModuleTraits<RGBDInstanceOutputPacket, Landmark>;

//TODO: nice to do some tempalting so that the module knows the expected input type to cast to which also knows the measurement type!!!
//so we can use this to define the map<M> type
class RGBDBackendModule : public BackendModuleType<RGBDBackendModuleTraits> {

public:
    using Base = BackendModuleType<RGBDBackendModuleTraits>;
    RGBDBackendModule(const BackendParams& backend_params, Map3d::Ptr map, ImageDisplayQueue* display_queue = nullptr);
    ~RGBDBackendModule();

    using SpinReturn = Base::SpinReturn;

    void saveGraph(const std::string& file = "rgbd_graph.dot");
    void saveTree(const std::string& file = "rgbd_bayes_tree.dot");

//TODO: for now
public:
    SpinReturn boostrapSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;
    SpinReturn nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;

    //TODO: taken from MonoBackendModule
     //adds pose to the new values and a prior on this pose to the new_factors
    void addInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);

    //T_world_camera is the estimate from the frontend and should be associated with the curr_frame_id
    void addOdometry(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, FrameId frame_id_k_1, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);

    void updateStaticObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors);
    void updateDynamicObservations(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_point_factors);



    void optimize(FrameId frame_id_k, const gtsam::Values& new_values,  const gtsam::NonlinearFactorGraph& new_factors);

    const ObjectPoseMap& updateObjectPoses(FrameId frame_id_k, const RGBDInstanceOutputPacket::ConstPtr input);

    //helper factor graph functions




public:
    std::unique_ptr<DynoISAM2> smoother_;
    // std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;

    using KeyTimestampMap = gtsam::IncrementalFixedLagSmoother::KeyTimestampMap;
    KeyTimestampMap timestamp_map_;
    // gtsam::Values new_values_;
    // gtsam::NonlinearFactorGraph new_factors_;

    ObjectPoseMap composed_object_poses_;

    //base backend module does not correctly share properties between mono and rgbd (i.e static_pixel_noise_ is in backend module but is not used in this class)
    //TODO: really need a rgbd and mono base class for this reason
    gtsam::SharedNoiseModel static_point_noise_; //! 3d isotropic pixel noise on static points
    gtsam::SharedNoiseModel dynamic_point_noise_; //! 3d isotropic pixel noise on dynamic points

    //we need a separate way of tracking if a dynamic tracklet is in the map, since each point is modelled uniquely
    //simply used as an O(1) lookup, the value is not actually used. If the key exists, we assume that the tracklet is in the map
    gtsam::FastMap<TrackletId, bool> is_dynamic_tracklet_in_map_;

    DebugInfo debug_info_;



};

} //dyno
