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

#include "dynosam/common/Types.hpp"
#include "dynosam/common/Flags.hpp"
#include <gflags/gflags.h>

#include <string>
#include <cxxabi.h>

//common glags used in multiple modules
DEFINE_bool(init_object_pose_from_gt, false , "If true, then the viz pose from the frontend/backend will start from the gt");
DEFINE_bool(save_frontend_json, false, "If true, then the output of the frontend will be saved as a binarized json file");
DEFINE_bool(frontend_from_file, false, "If true, the frontend will try and load all the data from a seralized json file");


namespace dyno {


template<>
std::string to_string(const ReferenceFrame& reference_frame) {
  switch (reference_frame)
  {
  case ReferenceFrame::LOCAL:
    return "local";
  case ReferenceFrame::GLOBAL:
    return "global";
  case ReferenceFrame::OBJECT:
    return "object";
  default:
    return "Unknown reference frame";
  }
};

std::string demangle(const char* name)
{
  // by default set to the original mangled name
  std::string demangled_name = std::string(name);

  // g++ version of demangle
  char* demangled = nullptr;
  int status = -1;  // some arbitrary value to eliminate the compiler warning
  demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);

  demangled_name = (status == 0) ? std::string(demangled) : std::string(name);
  std::free(demangled);

  return demangled_name;
}


} //dyno
