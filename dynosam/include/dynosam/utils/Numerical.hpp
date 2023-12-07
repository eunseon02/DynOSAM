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


#include <gtsam/base/Vector.h> //for fpequal
#include <gtsam/base/Matrix.h>

namespace dyno  {

inline bool fpEqual(double a, double b, double tol=1e-9) {
    return gtsam::fpEqual(a, b, tol);
}

inline bool is_zero(double a) {
    return fpEqual(a, 0.0);
}

//Jesse: make iterator T so we can work on any iterable type? I think this will also make Eigen compatable
template<typename T>
inline bool equals_with_abs_tol(const std::vector<T>& vec1, const std::vector<T>& vec2, double tol = 1e-9) {
    if(vec1.size() != vec2.size()) return false;

    const size_t m = vec1.size();
    for(size_t i = 0; i < m; i++) {
        if(!fpEqual(vec1[i], vec2[i], tol)) {
            return false;
        }
    }

    return true;
}

template<typename T>
inline T computeCentroid(const std::vector<T>& vec) {
  T sum{};
  for(const T& t : vec) { sum += t; }
  return sum/static_cast<double>(vec.size());
}

template<typename T>
inline T computeCentroid(const std::vector<T, Eigen::aligned_allocator<T>>& vec) {
  T sum{};
  for(const T& t : vec) { sum += t; }
  return sum/static_cast<double>(vec.size());
}

template<typename T>
inline double calculateStandardDeviation(const std::vector<T, Eigen::aligned_allocator<T>>& vec) {
  const T mean = computeCentroid<T>(vec);

  double sum = 0;
  for(const T& t : vec) {
    T sub = t.colwise() - mean;
    sum += sub.norm();
  }

  return std::sqrt(sum/static_cast<double>(vec.size()));
}


inline bool saveMatrixAsUpperTriangular(std::ostream& os, const gtsam::Matrix& matrix)
{
  const size_t rows = matrix.rows();
  const size_t cols = matrix.cols();

  if (rows != cols)
  {
    throw std::runtime_error("Attempting to save matrix as upper triangular but input size was not square");
  }

  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = i; j < cols; j++)
    {
      os << " " << matrix(i, j);
    }
  }
  return os.good();
}


} //dyno
