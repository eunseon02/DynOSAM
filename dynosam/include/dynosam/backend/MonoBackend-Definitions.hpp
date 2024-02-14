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

#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/utils/Metrics.hpp"
#include "dynosam/logger/Logger.hpp"

namespace dyno {

/**
 * @brief
 * SMART: Indicates that the tracklet has a smart factor associated with it but has not yet been triangulated. It may be in the graph (check by looking at the slot)
 * PROJECTION: Indicates that the tracklet has been triangulated and is now a projection factor in the graph
 *
 */
enum ProjectionFactorType {
    SMART,
    PROJECTION
};

enum ObjectDegeneracyType {
    TOO_SMALL,
    DEGENERATE_TRIANGULATION,
    OK
};

struct ProjectionFactorStatus {
    TrackletId tracklet_id_;
    ProjectionFactorType pf_type_;
    ObjectId object_id_;

    ProjectionFactorStatus(TrackletId tracklet_id, ProjectionFactorType pf_type, ObjectId object_id) : tracklet_id_(tracklet_id), pf_type_(pf_type), object_id_(object_id) {}
};

using TrackletIdToProjectionStatus = std::unordered_map<TrackletId, ProjectionFactorStatus>;


// Will be deleted from the smart projection factor map once the smart factor is converted to a projection factor
//if SlotIndex is -1, means that the factor has not been inserted yet in the graph
class SmartProjectionFactorMap : public gtsam::FastMap<TrackletId, std::pair<SmartProjectionFactor::shared_ptr, Slot>> {
public:

    inline bool exists(const TrackletId tracklet_id) const {
        const auto& it = this->find(tracklet_id);
        return it != this->end();
    }

    inline void add(TrackletId tracklet_id, SmartProjectionFactor::shared_ptr smart_factor, Slot slot) {
        //what if already exists?
        auto pair = std::make_pair(smart_factor, slot);
        this->insert({tracklet_id, pair});
    }


    inline SmartProjectionFactor::shared_ptr getSmartFactor(TrackletId tracklet_id) {
        checkAndThrow(exists(tracklet_id), "Cannot get smart factor from smart projection map as it does not exist. Offending tracklet: " + std::to_string(tracklet_id));
        return this->at(tracklet_id).first;
    }

    inline Slot& getSlot(TrackletId tracklet_id) {
        checkAndThrow(exists(tracklet_id), "Cannot get slot smart projection map as it does not exist. Offending tracklet: " + std::to_string(tracklet_id));
        return this->at(tracklet_id).second;
    }

    inline Slot getSlot(TrackletId tracklet_id) const {
        checkAndThrow(exists(tracklet_id), "Cannot get slot smart projection map as it does not exist. Offending tracklet: " + std::to_string(tracklet_id));
        return this->at(tracklet_id).second;
    }

};


//TODO: depricate
class MonoDebugInfo {
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

        friend std::ostream& operator<<(std::ostream& os, const MonoDebugInfo& info) {

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

using MonoDebugInfos = std::vector<MonoDebugInfo>;
using MonoDebugInfoMap = gtsam::FastMap<FrameId, MonoDebugInfo>;


//TODO: tidy up!! MonoBackendLogger?
struct BackendLogger {


    void log(const std::string& name, const gtsam::Values& values, const GroundTruthPacketMap& packet_map) {

        CsvWriter object_motion(CsvHeader(
            "frame_id",
            "object_id",
            "x", "y" ,"z", "r", "p", "y",
            "t_err", "r_err"));

        CsvWriter object_motion_summary(CsvHeader(
            "object_id",
            "avg_t_err", "avg_r_err",
            "num_frame"));

        auto extracted_motions = values.extract<gtsam::Pose3>(gtsam::Symbol::ChrTest(kObjectMotionSymbolChar));

        using namespace dyno::utils;

        gtsam::FastMap<ObjectId, TRErrorPairVector> avg_errors;

        for(const auto&[key, motion] : extracted_motions) {
            ObjectId object_id;
            FrameId frame_id;

            if(reconstructMotionInfo(key, object_id, frame_id)) {
                CHECK(packet_map.exists(frame_id));
                const GroundTruthInputPacket& gt_packet = packet_map.at(frame_id);

                ObjectPoseGT object_pose_gt;
                if(!gt_packet.getObject(object_id, object_pose_gt)) {
                    LOG(ERROR) << "Could not find gt object at frame " << frame_id << " for object Id" << object_id;
                    continue;
                }

                CHECK(object_pose_gt.prev_H_current_world_);

                TRErrorPair motion_error = TRErrorPair::CalculatePoseError(motion, *object_pose_gt.prev_H_current_world_);

                const double x = motion.x();
                const double y = motion.y();
                const double z = motion.z();

                const gtsam::Rot3 rot = motion.rotation();
                const gtsam::Vector3 rpy = rot.rpy();
                const double rr = rpy(0);
                const double rp = rpy(1);
                const double ry = rpy(2);

                if(!avg_errors.exists(object_id)) {
                    avg_errors.insert2(object_id, TRErrorPairVector{});
                }

                avg_errors.at(object_id).push_back(motion_error);

                object_motion << frame_id
                            << object_id
                            << x << y << z
                            << rr << rp << ry
                            << motion_error.translation_
                            << motion_error.rot_;



            }
        }

        {
            const std::string file_name = "/" + name + "_motion_log.csv";
            OfstreamWrapper::WriteOutCsvWriter(object_motion, file_name);
        }

        {
            for(const auto&[object_id, avg_error] : avg_errors) {
                const TRErrorPair avg = avg_error.average();

                object_motion_summary << object_id << avg.translation_ << avg.rot_ << avg_error.size();
            }

            const std::string file_name = "/" + name + "_motion_summary_log.csv";
            OfstreamWrapper::WriteOutCsvWriter(object_motion_summary, file_name);
        }
    }

    void log(const std::string& name, const GroundTruthPacketMap& packet_map, const gtsam::FastMap<ObjectId, gtsam::FastMap<FrameId, gtsam::Pose3>>& composed_object_poses) {

        CsvWriter object_pose(CsvHeader(
            "frame_id",
            "object_id",
            "t_abs_err", "r_abs_err",
            "t_rel_err", "r_rel_err"));

        CsvWriter object_pose_summary(CsvHeader(
            "object_id",
            "avg_t_abs_err", "avg_r_abs_err",
            "avg_t_rel_err", "avg_r_rel_err",
            "num_frame"));

        using namespace dyno::utils;

        gtsam::FastMap<ObjectId, TRErrorPairVector> avg_abs_errors;
        gtsam::FastMap<ObjectId, TRErrorPairVector> avg_rel_errors;

        for(const auto&[object_id, poses_map] : composed_object_poses) {
            for(const auto&[frame_id, pose] : poses_map) {

                double t_abs_err = 0, r_abs_err = 0, t_rel_err = 0, r_rel_err = 0;

                const GroundTruthInputPacket& gt_packet = packet_map.at(frame_id);
                ObjectPoseGT object_pose_gt;
                if(!gt_packet.getObject(object_id, object_pose_gt)) {
                    LOG(ERROR) << "Could not find gt object at frame " << frame_id << " for object Id" << object_id;
                    continue;
                }

                const auto& gt_pose = object_pose_gt.L_world_;
                const auto& estimated_pose = pose;
                TRErrorPair absolute_pose_error = TRErrorPair::CalculatePoseError(estimated_pose,gt_pose);
                t_abs_err = absolute_pose_error.translation_;
                r_abs_err = absolute_pose_error.rot_;

                //get previous pose for relative error
                if(packet_map.exists(frame_id - 1) && poses_map.exists(frame_id - 1)) {
                    const GroundTruthInputPacket& gt_packet_prev = packet_map.at(frame_id - 1);
                    const gtsam::Pose3 previous_estimated_pose = poses_map.at(frame_id - 1);

                    const ObjectPoseGT* object_pose_gt_curr;
                    const ObjectPoseGT* object_pose_gt_prev;
                    if(!gt_packet_prev.findAssociatedObject(object_id, gt_packet, &object_pose_gt_prev, &object_pose_gt_curr)) {continue;}

                    CHECK_NOTNULL(object_pose_gt_curr);
                    CHECK_NOTNULL(object_pose_gt_prev);

                    CHECK_EQ(object_pose_gt_prev->frame_id_, frame_id - 1);
                    CHECK_EQ(object_pose_gt_curr->frame_id_, frame_id);

                    const auto& gt_pose_prev = object_pose_gt_prev->L_world_;
                    TRErrorPair relative_pose_error = TRErrorPair::CalculateRelativePoseError(previous_estimated_pose, estimated_pose, gt_pose_prev, gt_pose);
                    t_rel_err = relative_pose_error.translation_;
                    r_rel_err = relative_pose_error.rot_;

                    if(!avg_rel_errors.exists(object_id)) {
                        avg_rel_errors.insert2(object_id, TRErrorPairVector{});
                    }
                    avg_rel_errors.at(object_id).push_back(relative_pose_error);
                }

                if(!avg_abs_errors.exists(object_id)) {
                    avg_abs_errors.insert2(object_id, TRErrorPairVector{});
                }
                avg_abs_errors.at(object_id).push_back(absolute_pose_error);

                object_pose << frame_id << object_id << t_abs_err << r_abs_err << t_rel_err << r_rel_err;
            }
        }

        {
            const std::string file_name = "/" + name + "_object_pose_error_log.csv";
            OfstreamWrapper::WriteOutCsvWriter(object_pose, file_name);
        }

        {
            for(const auto&[object_id, avg_abs_error] : avg_abs_errors) {
                const TRErrorPair avg_abs = avg_abs_error.average();

                double rel_t_error = -1;
                double rel_r_error = -1;

                //we will not have a object id for all relative errors
                if(avg_rel_errors.exists(object_id)) {
                    const TRErrorPair avg_rel = avg_rel_errors.at(object_id).average();
                    rel_t_error = avg_rel.translation_;
                    rel_r_error = avg_rel.rot_;
                }

                object_pose_summary << object_id
                    << avg_abs.translation_ << avg_abs.rot_
                    << rel_t_error << rel_r_error
                    << avg_abs_error.size();
            }

            const std::string file_name = "/" + name + "_object_pose_error_summary_log.csv";
            OfstreamWrapper::WriteOutCsvWriter(object_pose_summary, file_name);
        }
    }

    void log(const std::string& name, const MonoDebugInfoMap& infos) {
        CsvWriter debug_info(CsvHeader(
            "frame_id",
            "object_id",
            "num_scale_priors",
            "num_initial_priors"));

        for(const auto&[frame_id, info] : infos) {
            for(ObjectId object_id : info.objects_motions_added) {
                int num_scale_priors = 0, num_initial_priors = 0;

                if(info.num_scale_priors_added_per_object_.exists(object_id)) {
                    num_scale_priors = info.num_scale_priors_added_per_object_.at(object_id);
                }

                if(info.num_initial_priors_added_per_object_.exists(object_id)) {
                    num_initial_priors = info.num_initial_priors_added_per_object_.at(object_id);
                }

                debug_info << frame_id << object_id << num_scale_priors << num_initial_priors;
            }
        }

        const std::string file_name = "/" + name + "_debug_info_log.csv";
        OfstreamWrapper::WriteOutCsvWriter(debug_info, file_name);
    }

};


} //dyno
