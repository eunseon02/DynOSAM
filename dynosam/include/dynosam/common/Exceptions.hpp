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

#include <stdexcept>
#include <string>
#include <functional>

namespace dyno {

struct DynosamException : public std::runtime_error {
DynosamException(const std::string& what) : std::runtime_error(what) {}
};

template<typename Exception = DynosamException>
inline void checkAndThrow(bool condition) {
  if (!condition) {
    throw Exception("");
  }
}


template<typename Exception = DynosamException>
inline void checkAndThrow(bool condition, const std::string& error_message) {
  if (!condition) {
    throw Exception(error_message);
  }
}

template<typename Exception, typename... EArgs>
inline void checkAndThrow(bool condition, EArgs&& ...eargs) {
  if (!condition) {
    throw Exception(std::forward<EArgs>(eargs)...);
  }
}

} //dyno
