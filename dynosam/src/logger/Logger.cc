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

#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/Metrics.hpp"
#include "dynosam/utils/Statistics.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>


DEFINE_string(output_path, "./", "Path where to store dynosam's log output.");

namespace dyno {

namespace fs = std::filesystem;

std::string getOutputFilePath(const std::string& file_name) {
  fs::path out_path(FLAGS_output_path);
  return out_path / file_name;
}


void writeStatisticsSamplesToFile(const std::string& file_name) {
  utils::Statistics::WriteAllSamplesToCsvFile(getOutputFilePath(file_name));
}

void writeStatisticsSummaryToFile(const std::string& file_name) {
  utils::Statistics::WriteSummaryToCsvFile(getOutputFilePath(file_name));
}


// This constructor will directly open the log file when called.
OfstreamWrapper::OfstreamWrapper(const std::string& filename,
                                 const bool& open_file_in_append_mode)
    : OfstreamWrapper(filename, FLAGS_output_path, open_file_in_append_mode) {}


OfstreamWrapper::OfstreamWrapper(const std::string& filename,
                const std::string& output_path,
                const bool& open_file_in_append_mode)
  : filename_(filename), output_path_(output_path) {
    openLogFile(open_file_in_append_mode);
}

// This destructor will directly close the log file when the wrapper is
// destructed. So no need to explicitly call .close();
OfstreamWrapper::~OfstreamWrapper() {
  VLOG(20) << "Closing output file: " << filename_.c_str();
  ofstream_.flush();
  ofstream_.close();
}

void OfstreamWrapper::closeAndOpenLogFile() {
  ofstream_.flush();
  ofstream_.close();
  CHECK(!filename_.empty());
  OpenFile(output_path_ + '/' + filename_, &ofstream_, false);
}


bool OfstreamWrapper::WriteOutCsvWriter(const CsvWriter& csv, const std::string& filename) {
    //set append mode to false as we never want to write over the top of a csv file as this
    //will upset the header
    OfstreamWrapper ofsw(filename, false);
    return csv.write(ofsw.ofstream_);
}

void OfstreamWrapper::openLogFile(bool open_file_in_append_mode) {
  CHECK(!filename_.empty());
  CHECK(!output_path_.empty());
  LOG(INFO) << "Opening output file: " << filename_.c_str();
  OpenFile((std::string)getFilePath(),
           &ofstream_,
           open_file_in_append_mode);
}


fs::path OfstreamWrapper::getFilePath() const {
  fs::path fs_out_path(output_path_);
  if(!fs::exists(fs_out_path))  throw std::runtime_error("OfstreamWrapper - Output path does not exist: " + output_path_);

  return fs_out_path / fs::path(filename_);
}


EstimationModuleLogger::EstimationModuleLogger(const std::string& module_name)
  : module_name_(module_name),
    object_motion_errors_file_name_(module_name_ + "_object_motion_error_log.csv"),
    object_pose_errors_file_name_(module_name_ + "_object_pose_errors_log.csv"),
    object_pose_file_name_(module_name_ + "_object_pose_log.csv"),
    object_bbx_file_name_(module_name_ + "_object_bbx_log.csv"),
    camera_pose_errors_file_name_(module_name_ + "_camera_pose_error_log.csv"),
    camera_pose_file_name_(module_name_ + "_camera_pose_log.csv"),
    map_points_file_name_(module_name_ + "_map_points_log.csv")
{
  object_motion_errors_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "t_err", "r_err"));

  object_pose_errors_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "t_abs_err", "r_abs_err",
            "t_rel_err", "r_rel_err"));


  camera_pose_errors_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "t_abs_err", "r_abs_err",
            "t_rel_err", "r_rel_err"));

  camera_pose_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "x", "y", "z",
            "roll", "pitch" , "yaw"));

  object_pose_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "x", "y", "z",
            "roll", "pitch" , "yaw"));

  map_points_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "tracklet_id",
            "x_world", "y_world", "z_world"));

  object_pose_errors_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "t_abs_err", "r_abs_err",
            "t_rel_err", "r_rel_err"));

  object_bbx_csv_ = std::make_unique<CsvWriter>(CsvHeader(
            "frame_id",
            "object_id",
            "min_bbx_x", "min_bbx_y", "min_bbx_z",
            "max_bbx_x", "max_bbx_y", "max_bbx_z",
            "px", "py", "pz",
            "qw", "qx", "qy", "qz"));

}

EstimationModuleLogger::~EstimationModuleLogger() {
  LOG(INFO) << "Writing out " << module_name_ << " logger...";

  OfstreamWrapper::WriteOutCsvWriter(*object_motion_errors_csv_, object_motion_errors_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*object_pose_errors_csv_, object_pose_errors_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*object_pose_csv_, object_pose_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*object_bbx_csv_, object_bbx_file_name_);

  OfstreamWrapper::WriteOutCsvWriter(*camera_pose_errors_csv_, camera_pose_errors_file_name_);
  OfstreamWrapper::WriteOutCsvWriter(*camera_pose_csv_, camera_pose_file_name_);

  OfstreamWrapper::WriteOutCsvWriter(*map_points_csv_, map_points_file_name_);
}

void EstimationModuleLogger::logObjectMotion(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const MotionEstimateMap& motion_estimates) {
  if(!gt_packets.exists(frame_id)) {
        LOG(WARNING) << "No gt packet at frame id " << frame_id << ". Unable to log frontend object motions";
        return;
    }

    const GroundTruthInputPacket& gt_packet = gt_packets.at(frame_id);

    for(const auto& [object_id, motions] : motion_estimates) {

        //TODO: -1, -1 if can not find gt motion? This will message with average!!
        utils::TRErrorPair motion_error(-1, -1);
        ObjectPoseGT object_pose_gt;
        if(!gt_packet.getObject(object_id, object_pose_gt)) {
            LOG(ERROR) << "Could not find gt object at frame " << frame_id << " for object Id" << object_id;
        }
        else {
            const gtsam::Pose3& estimate = motions;
            motion_error = utils::TRErrorPair::CalculatePoseError(estimate, *object_pose_gt.prev_H_current_world_);
        }

        //frame_id, object_id, t_err, r_error
        *object_motion_errors_csv_ << frame_id << object_id << motion_error.translation_ << motion_error.rot_;

    }
}

//assume poses are in world?
void EstimationModuleLogger::logObjectPose(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const ObjectPoseMap& propogated_poses) {
  //assume object poses get logged in frame order!!!
  for(const auto&[object_id, poses_map] : propogated_poses) {
      //do not draw if in current frame
      if(!poses_map.exists(frame_id)) {
          VLOG(10) << "Cannot log object pose (id=" << object_id << ") for frame " << frame_id << " as it does not exist in the map";
          continue;
      }

      const gtsam::Pose3& L_world_k = poses_map.at(frame_id);
      const auto rot = L_world_k.rotation();
      *object_pose_csv_ << frame_id << object_id << L_world_k.x() << L_world_k.y() << L_world_k.z() << rot.roll() << rot.pitch() << rot.yaw();

      const FrameId frame_id_k_1 = frame_id - 1u;
      if(!poses_map.exists(frame_id_k_1)) {
          continue;
      }

      const gtsam::Pose3& L_world_k_1 = poses_map.at(frame_id_k_1);

      if(!gt_packets.exists(frame_id) || !gt_packets.exists(frame_id_k_1)) {
        LOG(WARNING) << "No gt packet at frame id " << frame_id << " or previous frame. Unable to log object pose errors";
        return;
      }

      const GroundTruthInputPacket& gt_packet_k = gt_packets.at(frame_id);
      const GroundTruthInputPacket& gt_packet_k_1 = gt_packets.at(frame_id_k_1);
      //check object exists in this frame
      const ObjectPoseGT* object_gt_k;
      const ObjectPoseGT* object_gt_k_1;
      if(gt_packet_k.findAssociatedObject(object_id, gt_packet_k_1, &object_gt_k, &object_gt_k_1)) {
        CHECK_NOTNULL(object_gt_k);
        CHECK_NOTNULL(object_gt_k_1);
        //get gt poses
        const auto& gt_L_world_k = object_gt_k->L_world_;
        const auto& gt_L_world_k_1 = object_gt_k_1->L_world_;

        utils::TRErrorPair absolute_pose_error = utils::TRErrorPair::CalculatePoseError(L_world_k, gt_L_world_k);
        utils::TRErrorPair relative_pose_error = utils::TRErrorPair::CalculateRelativePoseError(L_world_k_1, L_world_k, gt_L_world_k_1, gt_L_world_k);

         //frame_id, object_id, t_abs_err, r_abs_error, t_rel_error, r_rel_error
        *object_pose_errors_csv_ << frame_id << object_id << absolute_pose_error.translation_ << absolute_pose_error.rot_ << relative_pose_error.translation_ << relative_pose_error.rot_;

      }

  }
}

void EstimationModuleLogger::logCameraPose(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const gtsam::Pose3& T_world_camera, std::optional<const gtsam::Pose3> T_world_camera_k_1) {
  if(!gt_packets.exists(frame_id)) {
        LOG(WARNING) << "No gt packet at frame id " << frame_id << ". Unable to log object motions";
        return;
    }

    const GroundTruthInputPacket& gt_packet_k = gt_packets.at(frame_id);
    const gtsam::Pose3 gt_T_world_camera_k = gt_packet_k.X_world_;

    utils::TRErrorPair absolute_pose_error = utils::TRErrorPair::CalculatePoseError(T_world_camera,gt_T_world_camera_k);

    utils::TRErrorPair relative_pose_error(0, 0);

    const FrameId frame_id_k_1 = frame_id - 1;
    //if we have a previous gt packet AND a provided estimate
    if(gt_packets.exists(frame_id_k_1) && T_world_camera_k_1) {
        const GroundTruthInputPacket& gt_packet_k_1 = gt_packets.at(frame_id_k_1);
        const gtsam::Pose3 gt_T_world_camera_k_1 = gt_packet_k_1.X_world_;
        relative_pose_error = utils::TRErrorPair::CalculateRelativePoseError(T_world_camera_k_1.value(), T_world_camera, gt_T_world_camera_k_1, gt_T_world_camera_k);
    }


    //frame_id, t_abs_err, r_abs_error, t_rel_error, r_rel_error
    *camera_pose_errors_csv_ << frame_id << absolute_pose_error.translation_ << absolute_pose_error.rot_ << relative_pose_error.translation_ << relative_pose_error.rot_;

    const auto rot = T_world_camera.rotation();
    *camera_pose_csv_ << frame_id << T_world_camera.x() << T_world_camera.y() << T_world_camera.z() << rot.roll() << rot.pitch() << rot.yaw();
}

void EstimationModuleLogger::logPoints(FrameId frame_id, const gtsam::Pose3& T_world_local_k, const StatusLandmarkEstimates& landmarks) {
  for(const auto& status_lmks : landmarks) {
      const TrackletId tracklet_id = status_lmks.trackletId();
      ObjectId object_id = status_lmks.objectId();
      Landmark lmk_world = status_lmks.value();

      if(status_lmks.referenceFrame() == ReferenceFrame::LOCAL) {
        lmk_world = T_world_local_k * status_lmks.value();
      }
      else if(status_lmks.referenceFrame() == ReferenceFrame::OBJECT) {
        throw DynosamException("Cannot log object point in the object reference frame");
      }

      *map_points_csv_ << frame_id << object_id << tracklet_id << lmk_world(0) << lmk_world(1) << lmk_world(2);
  }

}

void EstimationModuleLogger::logObjectBbxes(FrameId frame_id, const BbxPerObject& object_bbxes){
  for(const auto&[object_id, this_object_bbx] : object_bbxes) {
    const gtsam::Quaternion& q = this_object_bbx.orientation_.toQuaternion();
    *object_bbx_csv_ << frame_id << object_id << this_object_bbx.min_bbx_point_.x() << this_object_bbx.min_bbx_point_.y() << this_object_bbx.min_bbx_point_.z()
                                              << this_object_bbx.max_bbx_point_.x() << this_object_bbx.max_bbx_point_.y() << this_object_bbx.max_bbx_point_.z()
                                              << this_object_bbx.bbx_position_.x() << this_object_bbx.bbx_position_.y() << this_object_bbx.bbx_position_.z()
                                              << q.w() << q.x() << q.y() << q.z();
  }
}

} //dyno
