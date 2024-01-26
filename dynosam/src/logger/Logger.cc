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
#include <gflags/gflags.h>
#include <glog/logging.h>


DEFINE_string(output_path, "./", "Path where to store dynosam's log output.");

namespace dyno {

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

void OfstreamWrapper::openLogFile(const std::string& output_file_name,
                                  bool open_file_in_append_mode) {
  CHECK(!output_file_name.empty());
  LOG(INFO) << "Opening output file: " << output_file_name.c_str();
  OpenFile(output_path_ + '/' + output_file_name,
           &ofstream_,
           open_file_in_append_mode);
}



}
