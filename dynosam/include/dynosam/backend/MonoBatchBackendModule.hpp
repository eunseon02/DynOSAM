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
#include "dynosam/backend/MonoBackend-Definitions.hpp"
#include "dynosam/common/Camera.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"


#include "dynosam/frontend/MonoInstance-Definitions.hpp" // for MonocularInstanceOutputPacket

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace dyno {

class MonoBatchBackendModule : public BackendModule {

public:
    MonoBatchBackendModule(const BackendParams& backend_params, Camera::Ptr camera);
    ~MonoBatchBackendModule();

    using SpinReturn = BackendModule::SpinReturn;

    //arg 2 is measurement from frontend
    using PoseInitFunc = std::function<gtsam::Pose3(FrameId, const gtsam::Pose3&, const GroundTruthInputPacket&)>;
    //pose3 is the camera pose being initalised in the graph at this frame id
    using DynamicPointInitFunc = std::function<gtsam::Point3(
        FrameId, TrackletId, ObjectId, const Keypoint&, const gtsam::Pose3&, const GroundTruthInputPacket&)>;
    //frame is is k so the motion should go from k-1 to k
    using MotionInitFunc = std::function<gtsam::Pose3(FrameId, ObjectId, const GroundTruthInputPacket&)>;

    //! Initalisation of the values in the graph
    PoseInitFunc pose_init_func_;
    DynamicPointInitFunc dynamic_point_init_func_;
    MotionInitFunc motion_init_func_;

    class DebugInfo {
    public:
        FrameId frame_id;
        std::set<ObjectId> objects_motions_added; //with frame =k, so motions are k-1 to k
        gtsam::FastMap<ObjectId, int> num_scale_priors_added_per_object_;
        gtsam::FastMap<ObjectId, int> num_initial_priors_added_per_object_;
        gtsam::FastMap<ObjectId, int> num_newly_tracked_object_points_;
        gtsam::FastMap<ObjectId, int> num_existing_tracked_object_points_;

        inline void incrementNumScalePriors(ObjectId object_id) { incrementMap(num_scale_priors_added_per_object_, object_id);}
        inline void incrementInitialPriors(ObjectId object_id) { incrementMap(num_initial_priors_added_per_object_, object_id);}
        inline void incrementNewlyTrackedObjectPoints(ObjectId object_id) { incrementMap(num_newly_tracked_object_points_, object_id);}
        inline void incrementExistingTrackedObjectPoints(ObjectId object_id) { incrementMap(num_existing_tracked_object_points_, object_id);}

        friend std::ostream& operator<<(std::ostream& os, const DebugInfo& info) {

            auto print_map =[](const gtsam::FastMap<ObjectId, int>& fast_map) {
                std::stringstream ss;
                for(const auto& [object_id, num] : fast_map) {
                    ss << std::setw(6) << "object id = " << object_id << " count = " << num << "\n";
                }
                return ss.str();
            };

            os << "frame: " << info.frame_id << "\n";
            os << "object motions added " << container_to_string(info.objects_motions_added) << "\n";
            os << "Num scale priors:\n" << print_map(info.num_scale_priors_added_per_object_);
            os << "Num initial priors:\n" << print_map(info.num_initial_priors_added_per_object_);
            os << "Num newly tracked points:\n" << print_map(info.num_newly_tracked_object_points_);
            os << "Num existing tracked points:\n" << print_map(info.num_existing_tracked_object_points_);
            return os;
        }


    private:
        void incrementMap(gtsam::FastMap<ObjectId, int>& fast_map, ObjectId object_id) {
            if(fast_map.exists(object_id)) {
                fast_map.at(object_id)++;
            }
            else {
                fast_map.insert2(object_id, 1);
            }
        }


    };

private:
    //copied from MonoBackendModule -> should synthesise classes together once implementation
    //is finalised
    SpinReturn boostrapSpin(BackendInputPacket::ConstPtr input) override;
    SpinReturn nominalSpin(BackendInputPacket::ConstPtr input) override;

    SpinReturn monoBoostrapSpin(MonocularInstanceOutputPacket::ConstPtr input);
    SpinReturn monoNominalSpin(MonocularInstanceOutputPacket::ConstPtr input);

    gtsam::FastMap<FrameId, DebugInfo> debug_infos_; // should represent debug information for all the frames included in the optimziation

    void runStaticUpdate(
        FrameId current_frame_id,
        gtsam::Key pose_key,
        const StatusKeypointMeasurements& static_keypoint_measurements);

    void runDynamicUpdate(
        FrameId current_frame_id,
        gtsam::Key pose_key,
        const StatusKeypointMeasurements& dynamic_keypoint_measurements,
        const DecompositionRotationEstimates& estimated_motions);

    void checkForScalePriors(
        FrameId current_frame_id,
        const DecompositionRotationEstimates& estimated_motions);

    bool safeAddConstantObjectVelocityFactor(FrameId current_frame, ObjectId object_id, const gtsam::Values& values, gtsam::NonlinearFactorGraph& factors);

    DynamicObjectTrackletManager<Keypoint> do_tracklet_manager_;

    gtsam::Values new_values_;
    gtsam::NonlinearFactorGraph new_factors_;

    gtsam::Values state_;
    gtsam::FastMap<FrameId,GroundTruthInputPacket> gt_packet_map_;
    gtsam::FastMap<FrameId,MonocularInstanceOutputPacket::ConstPtr> input_packet_map_;

    gtsam::SharedNoiseModel robust_static_pixel_noise_; //! Robust 2d isotropic pixel noise on static points (for projection factors)
    gtsam::SharedNoiseModel robust_dynamic_pixel_noise_; //! Robust 2d isotropic pixel noise on dynamic points (for projection factors)



    TrackletIdToProjectionStatus tracklet_to_status_map_; //only used for static points currently
    SmartProjectionFactorMap smart_factor_map_;

    //! Static map structures
    gtsam::FastMap<TrackletId, SmartProjectionFactor::shared_ptr> static_smart_factor_map;

    //! Dynamic map structures
    gtsam::FastMap<TrackletId, std::vector<GenericProjectionFactor::shared_ptr>> dynamic_factor_map;
    gtsam::FastMap<ObjectId, std::vector<LandmarkMotionTernaryFactor::shared_ptr>> dynamic_motion_factor_map;
    gtsam::FastMap<TrackletId, bool> dynamic_in_graph_factor_map;
    //updated when a motion variable is added and we update the last time a motion was added for this object
    gtsam::FastMap<ObjectId, FrameId> object_in_graph_;
    gtsam::FastMap<TrackletId, gtsam::Values> dynamic_initial_points;



};

} //dyno
