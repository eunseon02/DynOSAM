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

#include "dynosam/common/CameraParams.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/Numerical.hpp"    //for equals
#include "dynosam/utils/OpenCVUtils.hpp"  //for cv equals
#include "dynosam/utils/GtsamUtils.hpp"  //for cv equals
#include "dynosam/utils/YamlParser.hpp"

#include <glog/logging.h>
#include <memory>

namespace dyno
{

template<>
std::string to_string(const DistortionModel& distortion) {
  switch (distortion)
  {
  case DistortionModel::NONE:
    return "None";
    break;
  case DistortionModel::RADTAN:
    return "Radtan";
    break;
  case DistortionModel::EQUIDISTANT:
    return "Equidistant";
    break;
  case DistortionModel::FISH_EYE:
    return "Fish-eye";
    break;
  default:
    return "Unknown distortion model";
    break;
  }
}

CameraParams::CameraParams(const IntrinsicsCoeffs& intrinsics, const DistortionCoeffs& distortion, const cv::Size& image_size,
              const std::string& distortion_model, const gtsam::Pose3& T_robot_camera)
  : CameraParams(intrinsics, distortion, image_size, CameraParams::stringToDistortion(distortion_model, "pinhole"), T_robot_camera) {}

CameraParams::CameraParams(const IntrinsicsCoeffs& intrinsics, const DistortionCoeffs& distortion, const cv::Size& image_size,
              const DistortionModel& distortion_model, const gtsam::Pose3& T_robot_camera)
:   intrinsics_(intrinsics)
  , distortion_coeff_(distortion)
  , image_size_(image_size)
  , distortion_model_(distortion_model)
  , T_robot_camera_(T_robot_camera)
{
  CHECK_EQ(intrinsics_.size(), 4u) << "Intrinsics must be of length 4 - [fx fy cu cv]";
  CHECK_GT(distortion_coeff_.size(), 0u);

  CameraParams::convertDistortionVectorToMatrix(distortion_coeff_, &D_);
  CameraParams::convertIntrinsicsVectorToMatrix(intrinsics_, &K_);

  cv::cv2eigen(K_, K_eigen_);
  K_.copyTo(P_);
}

CameraParams CameraParams::fromYamlFile(const std::string& file_path) {
  YamlParser yaml_parser(file_path);

  std::string camera_id;
  yaml_parser.getYamlParam("camera_id", &camera_id);
  CHECK(!camera_id.empty()) << "Camera id cannot be empty.";
  VLOG(1) << "Parsing camera parameters for: " << camera_id;

  cv::Size image_size;
  parseImgSize(yaml_parser, &image_size);

  gtsam::Pose3 T_robot_camera;
  parseBodyPoseCam(yaml_parser, &T_robot_camera);

  IntrinsicsCoeffs intrinsics;
  parseCameraIntrinsics(yaml_parser, &intrinsics);

  DistortionCoeffs distortion;
  parseCameraDistortion(yaml_parser, &distortion);

  DistortionModel model;
  parseDistortionModel(yaml_parser, &model);


  return CameraParams(intrinsics, distortion, image_size, model, T_robot_camera);
}

void CameraParams::convertDistortionVectorToMatrix(const DistortionCoeffs& distortion_coeffs,
                                                  cv::Mat* distortion_coeffs_mat)
{
  *distortion_coeffs_mat = cv::Mat::zeros(1, distortion_coeffs.size(), CV_64F);
  for (int k = 0; k < distortion_coeffs_mat->cols; k++)
  {
    distortion_coeffs_mat->at<double>(0, k) = distortion_coeffs[k];
  }
}

void CameraParams::convertIntrinsicsVectorToMatrix(const IntrinsicsCoeffs& intrinsics, cv::Mat* camera_matrix)
{
  *camera_matrix = cv::Mat::eye(3, 3, CV_64F);
  camera_matrix->at<double>(0, 0) = intrinsics[0];
  camera_matrix->at<double>(1, 1) = intrinsics[1];
  camera_matrix->at<double>(0, 2) = intrinsics[2];
  camera_matrix->at<double>(1, 2) = intrinsics[3];
}

DistortionModel CameraParams::stringToDistortion(const std::string& distortion_model, const std::string& camera_model)
{
  std::string lower_case_distortion_model = distortion_model;
  std::string lower_case_camera_model = camera_model;

  std::transform(lower_case_distortion_model.begin(), lower_case_distortion_model.end(),
                 lower_case_distortion_model.begin(), ::tolower);
  std::transform(lower_case_camera_model.begin(), lower_case_camera_model.end(), lower_case_camera_model.begin(),
                 ::tolower);

  if (lower_case_camera_model == "pinhole")
  {
    if (lower_case_distortion_model == "none")
    {
      return DistortionModel::NONE;
    }
    else if ((lower_case_distortion_model == "plumb_bob") || (lower_case_distortion_model == "radial-tangential") ||
             (lower_case_distortion_model == "radtan"))
    {
      return DistortionModel::RADTAN;
    }
    else if (lower_case_distortion_model == "equidistant")
    {
      return DistortionModel::EQUIDISTANT;
    }
    else if (lower_case_distortion_model == "kannala_brandt")
    {
      return DistortionModel::FISH_EYE;
    }
    else
    {
      LOG(ERROR) << "Unrecognized distortion model for pinhole camera. Valid "
                    "pinhole distortion model options are 'none', 'radtan', "
                    "'equidistant', 'fish eye'.";
    }
  }
  else
  {
    LOG(ERROR) << "Unrecognized camera model. Valid camera models are 'pinhole'";
  }
  // Return no distortion model if invalid model is provided
  return DistortionModel::NONE;
}

bool CameraParams::equals(const CameraParams& other, double tol) const
{
  return dyno::equals_with_abs_tol(intrinsics_, other.intrinsics_, tol) &&
         dyno::equals_with_abs_tol(distortion_coeff_, other.distortion_coeff_, tol) &&
         (image_size_.width == other.image_size_.width && image_size_.height == other.image_size_.height) &&
         distortion_model_ == other.distortion_model_ &&
         T_robot_camera_.equals(other.T_robot_camera_, tol) &&
         utils::compareCvMatsUpToTol(K_, other.K_, tol) &&
         utils::compareCvMatsUpToTol(D_, other.D_);  // NOTE: do not comapre with P matrix as currently not calculated
}

const std::string CameraParams::toString() const
{
  std::stringstream out;
  out << "\nIntrinsics: \n- fx " << fx() << "\n- fy " << fy() << "\n- cu " << cu()
      << "\n- cv " << cv() << "\nimage_size: \n- width: " << ImageWidth() << "\n- height: " << ImageHeight()
      << "\n- K: " << K_ << '\n'
      << "- Distortion Model: " << to_string(distortion_model_) << '\n'
      << "- D: " << D_ << '\n'
      << "- P: " << P_ << '\n';

  return out.str();
}

void CameraParams::parseDistortionModel(const YamlParser& yaml_parser, DistortionModel* model) {
  CHECK_NOTNULL(model);
  std::string distortion_model, camera_model;
  yaml_parser.getYamlParam("distortion_model", &distortion_model);
  yaml_parser.getYamlParam("camera_model", &camera_model);
  *model = stringToDistortion(distortion_model, camera_model);

  if(*model == DistortionModel::NONE) {
    throw InvalidCameraCalibration("Invalid distortion model/camera model combination when constructing camera params from yaml file - "
      " distortion model: " + distortion_model + ", camera model: " + camera_model);
  }

}


void CameraParams::parseImgSize(const YamlParser& yaml_parser,
                                cv::Size* image_size) {
  CHECK_NOTNULL(image_size);
  std::vector<int> resolution;
  yaml_parser.getYamlParam("resolution", &resolution);
  CHECK_EQ(resolution.size(), 2);
  *image_size = cv::Size(resolution[0], resolution[1]);
}



// void CameraParams::parseFrameRate(const YamlParser& yaml_parser,
//                                   double* frame_rate) {
//   CHECK_NOTNULL(frame_rate);
//   int rate = 0;
//   yaml_parser.getYamlParam("rate_hz", &rate);
//   CHECK_GT(rate, 0u);
//   *frame_rate = 1 / static_cast<double>(rate);
// }

void CameraParams::parseBodyPoseCam(const YamlParser& yaml_parser,
                                    gtsam::Pose3* body_Pose_cam) {
  CHECK_NOTNULL(body_Pose_cam);
  // int n_rows = 0;
  // yaml_parser.getNestedYamlParam("T_BS", "rows", &n_rows);
  // CHECK_EQ(n_rows, 4);
  // int n_cols = 0;
  // yaml_parser.getNestedYamlParam("T_BS", "cols", &n_cols);
  // CHECK_EQ(n_cols, 4);
  std::vector<double> vector_pose;
  yaml_parser.getNestedYamlParam("T_BS", "data", &vector_pose);
  *body_Pose_cam = utils::poseVectorToGtsamPose3(vector_pose);
}

void CameraParams::parseCameraIntrinsics(const YamlParser& yaml_parser,
                                         IntrinsicsCoeffs* intrinsics) {
  CHECK_NOTNULL(intrinsics);
  std::vector<double> intrinsics_v;
  yaml_parser.getYamlParam("intrinsics", &intrinsics_v);
  CHECK_EQ(intrinsics_v.size(), 4u);
  intrinsics->resize(4u);
  // Move elements from one to the other.
  std::copy_n(std::make_move_iterator(intrinsics_v.begin()),
              intrinsics->size(),
              intrinsics->begin());
}

void CameraParams::parseCameraDistortion(const YamlParser& yaml_parser, DistortionCoeffs* distortion) {
  CHECK_NOTNULL(distortion);
  yaml_parser.getYamlParam("distortion_coefficients", distortion);
}


template<>
gtsam::Cal3DS2 CameraParams::constructGtsamCalibration<gtsam::Cal3DS2>() const {
  static const auto requested_calibration_name = type_name<gtsam::Cal3DS2>(); //only used for debug so seems waste to allocate everytime


  if(distortion_model_ != DistortionModel::RADTAN) {
    throw InvalidCameraCalibration("Requested gtsam calibration was " + requested_calibration_name +
    " which is unsupported by this camera model: " + to_string(distortion_model_));
  }

  if(distortion_coeff_.size() < 4u) {
    throw InvalidCameraCalibration("Distortion coefficients have size <4 for camera params with distortion model"
      + to_string(distortion_model_) + " and requested gtsam calibration of type " + requested_calibration_name);
  }

  constexpr static double skew = 0.0;
  return gtsam::Cal3DS2(fx(), fy(), skew, cu(), cv(), distortion_coeff_.at(0), distortion_coeff_.at(1), distortion_coeff_.at(2), distortion_coeff_.at(3));
}


template<>
gtsam::Cal3Fisheye CameraParams::constructGtsamCalibration<gtsam::Cal3Fisheye>() const {
  static const auto requested_calibration_name = type_name<gtsam::Cal3Fisheye>(); //only used for debug so seems waste to allocate everytime


  //jesse: not sure if gtsam well supports equidistant models!
  if(distortion_model_ != DistortionModel::FISH_EYE && distortion_model_ != DistortionModel::EQUIDISTANT) {
    throw InvalidCameraCalibration("Requested gtsam calibration was " + requested_calibration_name +
    " which is unsupported by this camera model: " + to_string(distortion_model_));
  }

  if(distortion_coeff_.size() < 4u) {
    throw InvalidCameraCalibration("Distortion coefficients have size <4 for camera params with distortion model"
      + to_string(distortion_model_) + " and requested gtsam calibration of type " + requested_calibration_name);
  }

  constexpr static double skew = 0.0;
  return gtsam::Cal3Fisheye(fx(), fy(), skew, cu(), cv(), distortion_coeff_.at(0), distortion_coeff_.at(1), distortion_coeff_.at(2), distortion_coeff_.at(3));
}

}  // namespace dyno
