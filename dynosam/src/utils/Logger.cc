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

#include "dynosam/utils/Logger.hpp"

#include <gflags/gflags.h>

DEFINE_string(output_path, "./", "Path where to store DynoSAM's log output.");


namespace dyno {

/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
// This constructor will directly open the log file when called.
OfstreamWrapper::OfstreamWrapper(const std::string& filename,
                                 const bool& open_file_in_append_mode)
    : filename_(filename), output_path_(FLAGS_output_path) {
  openLogFile(filename, open_file_in_append_mode);
}

// This destructor will directly close the log file when the wrapper is
// destructed. So no need to explicitly call .close();
OfstreamWrapper::~OfstreamWrapper() {
  LOG(INFO) << "Closing output file: " << filename_.c_str();
  ofstream_.close();
}

void OfstreamWrapper::closeAndOpenLogFile() {
  ofstream_.close();
  CHECK(!filename_.empty());
  OpenFile(output_path_ + '/' + filename_, &ofstream_, false);
}

void OfstreamWrapper::openLogFile(const std::string& output_file_name,
                                  bool open_file_in_append_mode) {
  CHECK(!output_file_name.empty());
  LOG(INFO) << "Opening output file: " << output_file_name.c_str();
  OpenFile(output_path_ + '/' + output_file_name,
           &ofstream_,
           open_file_in_append_mode);
}


GroundTruthLogger::GroundTruthLogger()
    :   gt_pose_("camera_pose_gt.csv"),
        gt_object_motion_("object_motion_gt.csv"),
        gt_object_pose_("object_pose_gt.csv") {}

void GroundTruthLogger::log(const GroundTruthInputPacket& gt_packet) {

}

FrontendLogger::FrontendLogger()
    :   vo_error_("frontend_vo_error.csv"),
        estimated_pose_("frontend_camera_pose.csv"),
        object_motion_error_("frontend_object_motion_error.csv"),
        estimated_object_motion_("frontend_object_motion_error.csv") {}

void FrontendLogger::log(const Frame& frame, const GroundTruthInputPacket& gt_packet) {

}


} //dyno
