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

#include "dynosam/utils/Numerical.hpp"

#include <eigen3/Eigen/Core> //must be included before opencv
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <opencv4/opencv2/opencv.hpp>


namespace opengv {
typedef Eigen::Matrix<double, 3, 4> transformation_t;
}


namespace dyno
{
namespace utils
{
gtsam::Pose3 cvMatToGtsamPose3(const cv::Mat& H);
// Converts a rotation matrix and translation vector from opencv to gtsam
// pose3
gtsam::Pose3 cvMatsToGtsamPose3(const cv::Mat& R, const cv::Mat& T);

cv::Mat gtsamPose3ToCvMat(const gtsam::Pose3& pose);

/* ------------------------------------------------------------------------ */
// Converts a 3x3 rotation matrix from opencv to gtsam Rot3
gtsam::Rot3 cvMatToGtsamRot3(const cv::Mat& R);

// Converts a 3x1 OpenCV matrix to gtsam Point3
gtsam::Point3 cvMatToGtsamPoint3(const cv::Mat& cv_t);
cv::Mat gtsamPoint3ToCvMat(const gtsam::Point3& point);

/**
 * @brief Converts a vector of 16 elements listing the elements of a 4x4 3D pose
*  matrix by rows into a pose3 in gtsam
 *
 */
gtsam::Pose3 poseVectorToGtsamPose3(const std::vector<double>& vector_pose);





template <typename T=double>
inline gtsam::Point2 cvPointToGtsam(const cv::Point_<T>& point)
{
  return gtsam::Point2(static_cast<double>(point.x), static_cast<double>(point.y));
}

template <typename T=double>
gtsam::Point2Vector cvPointsToGtsam(const std::vector<cv::Point_<T>>& points) {
  gtsam::Point2Vector gtsam_points;
  for (const auto& p : points)
  {
    gtsam_points.push_back(cvPointToGtsam<T>(p));
  }
  return gtsam_points;
}

template<typename T=double>
inline cv::Point_<T> gtsamPointToCv(const gtsam::Point2& point) {
  return cv::Point_<T>(static_cast<T>(point(0)), static_cast<T>(point(1)));
}

template<typename T=double>
std::vector<cv::Point_<T>> gtsamPointsToCv(const gtsam::Point2Vector& points) {
  std::vector<cv::Point_<T>> cv_points;
  for(const auto& p : points) {
    cv_points.push_back(gtsamPointToCv<T>(p));
  }
  return cv_points;
}

//  converts an opengv transformation (3x4 [R t] matrix) to a gtsam::Pose3
gtsam::Pose3 openGvTfToGtsamPose3(const opengv::transformation_t& RT);


template <class T>
static bool getEstimateOfKey(const gtsam::Values& state, const gtsam::Key& key, T* estimate)
{
  if (state.exists(key))
  {
    *CHECK_NOTNULL(estimate) = state.at<T>(key);
    return true;
  }
  else
  {
    return false;
  }
}

inline bool saveNoiseModelAsUpperTriangular(std::ostream& os, const gtsam::noiseModel::Gaussian& noise_model) {
  const gtsam::Matrix info = noise_model.information();
  return saveMatrixAsUpperTriangular(os, info);
}


} //utils
} //dyno

#include "dynosam/utils/GtsamUtils-inl.hpp"
