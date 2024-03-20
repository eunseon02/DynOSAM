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

#include "dynosam/utils/CsvWriter.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/common/PointCloudProcess.hpp"

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include <glog/logging.h>




namespace dyno {

/**
 * @brief Constructs a file path (using the file name) where the root of the path is
 * defined by the FLAGS_output_path param.
 *
 * This flag defines the output folder of all output log/debug files.
 * The resulting file path with then be FLAGS_output_path/file_name
 *
 * @param file_name
 * @return std::string
 */
std::string getOutputFilePath(const std::string& file_name);

/**
 * @brief Writes all the statistics to a csv file on the path defined by getOutputFilePath(file_name).
 *
 * The output csv file has a header "label, samples"
 * Where ALL recorded samples per label are saved to file.
 *
 * Internally uses Statistics::WriteAllSamplesToCsvFile.
 *
 * Default file name is statistics_samples.csv.
 *
 * @param file_name
 * @return true
 * @return false
 */
void writeStatisticsSamplesToFile(const std::string& file_name = "statistics_samples.csv");

/**
 * @brief Writes a summary of the statistics to a csv file on the path defined by getOutputFilePath(file_name).
 *
 * Internally uses Statistics::WriteSummaryToCsvFile.
 *
 * Default file name is statistics.csv.
 *
 * @param file_name
 * @return true
 * @return false
 */
void writeStatisticsSummaryToFile(const std::string& file_name = "statistics.csv");

// Open files with name output_filename, and checks that it is valid
static inline void OpenFile(const std::string& output_filename,
                            std::ofstream* output_file,
                            bool append_mode = false) {
  CHECK_NOTNULL(output_file);
  output_file->open(output_filename.c_str(),
                    append_mode ? std::ios_base::app : std::ios_base::out);
  output_file->precision(20);
  CHECK(output_file->is_open()) << "Cannot open file: " << output_filename;
  CHECK(output_file->good()) << "File in bad state: " << output_filename;
}

// Wrapper for std::ofstream to open/close it when created/destructed.
class OfstreamWrapper {
 public:
  OfstreamWrapper(const std::string& filename,
                  const bool& open_file_in_append_mode = false);
  virtual ~OfstreamWrapper();
  void closeAndOpenLogFile();

  static bool WriteOutCsvWriter(const CsvWriter& csv, const std::string& filename);

 public:
  std::ofstream ofstream_;
  const std::string filename_;
  const std::string output_path_;
  const bool open_file_in_append_mode_ = false;

 protected:
  void openLogFile(const std::string& output_file_name,
                   bool open_file_in_append_mode = false);
};



class EstimationModuleLogger {

public:
  DYNO_POINTER_TYPEDEFS(EstimationModuleLogger)
  EstimationModuleLogger(const std::string& module_name);
  //write to file on destructor
  virtual ~EstimationModuleLogger();

  //logs to motion errors
  virtual void logObjectMotion(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const MotionEstimateMap& motion_estimates);

  //logs to object pose errors and the object pose itself (to a differnet file)
  virtual void logObjectPose(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const ObjectPoseMap& propogated_poses);

  //logs to camera pose errors and the camera pose itself (to a differnet file)
  virtual void logCameraPose(const GroundTruthPacketMap& gt_packets, FrameId frame_id, const gtsam::Pose3& T_world_camera, std::optional<const gtsam::Pose3> T_world_camera_k_1);

  virtual void logPoints(FrameId frame_id, const gtsam::Pose3& T_world_local_k, const StatusLandmarkEstimates& landmarks);

  //logs to object bounding boxes
  virtual void logObjectBbxes(FrameId frame_id, const BbxPerObject& object_bbxes);

protected:
  const std::string module_name_;

  const std::string object_motion_errors_file_name_;
  const std::string object_pose_errors_file_name_;
  const std::string object_pose_file_name_;
  const std::string object_bbx_file_name_;

  const std::string camera_pose_errors_file_name_;
  const std::string camera_pose_file_name_;

  const std::string map_points_file_name_;

  CsvWriter::UniquePtr object_motion_errors_csv_;
  CsvWriter::UniquePtr object_pose_errors_csv_;
  CsvWriter::UniquePtr object_pose_csv_;
  CsvWriter::UniquePtr object_bbx_csv_;

  CsvWriter::UniquePtr camera_pose_errors_csv_;
  CsvWriter::UniquePtr camera_pose_csv_;

  CsvWriter::UniquePtr map_points_csv_;


};


} //dyno
