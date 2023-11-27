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

#include <glog/logging.h>

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>


namespace dyno {

//forward declare
class Frame;
class GroundTruthInputPacket;

template<typename T>
std::string toLogString(const T& t);

/* -------------------------------------------------------------------------- */
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

 public:
  std::ofstream ofstream_;
  const std::string filename_;
  const std::string output_path_;
  const bool open_file_in_append_mode = false;

  bool is_header_written_ = false;

 protected:
  void openLogFile(const std::string& output_file_name,
                   bool open_file_in_append_mode = false);
};


class GroundTruthLogger {
public:
    GroundTruthLogger();

    void log(const GroundTruthInputPacket& gt_packet);

private:
    OfstreamWrapper gt_pose_; //! gt camera pose
    OfstreamWrapper gt_object_motion_; //! gt object motion
    OfstreamWrapper gt_object_pose_; //! gt object motion
};


class FrontendLogger {
public:
    FrontendLogger();

    void log(const Frame& frame, const GroundTruthInputPacket& gt_packet);


private:
    OfstreamWrapper vo_error_; //! visual odom error
    OfstreamWrapper estimated_pose_; //! estimated camera pose

    OfstreamWrapper object_motion_error_; //! object motion error (k-1 to k)
    OfstreamWrapper estimated_object_motion_; //! estimated object motion

};


} //dyno
