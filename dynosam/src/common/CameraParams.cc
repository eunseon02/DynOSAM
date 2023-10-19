#include "dynosam/common/camera/CameraParams.hpp"
#include "dynosam/utils/Numerical.h"    //for equals
#include "dynosam/utils/OpenCVUtils.hpp"  //for cv equals

#include <glog/logging.h>
#include <memory>

namespace dyno
{
CameraParam::CameraParam(const IntrinsicsCoeffs& intrinsics, const DistortionCoeffs& distortion,
                         const cv::Size& image_size, const std::string& distortion_model, const HardwareId& hardware_id,
                         const gtsam::Pose3& T_R_C, const std::string& frame, const std::string& name,
                         const std::string& optics)
  : intrinsics_(intrinsics)
  , distortion_coeff_(distortion)
  , image_size_(image_size)
  , distortion_model_(CameraParam::stringToDistortion(distortion_model, "pinhole"))
  , hardware_id_(hardware_id)
  , T_R_C_(T_R_C)
  , frame_(frame)
  , name_(name)
  , optics_(optics)
{
  CHECK_EQ(intrinsics_.size(), 4u) << "Intrinsics must be of length 4 - [fx fy cu cv]";
  CHECK_GT(distortion_coeff_.size(), 0u);

  CameraParam::convertDistortionVectorToMatrix(distortion_coeff_, &D_);
  CameraParam::convertIntrinsicsVectorToMatrix(intrinsics_, &K_);
}

void CameraParam::convertDistortionVectorToMatrix(const DistortionCoeffs& distortion_coeffs,
                                                  cv::Mat* distortion_coeffs_mat)
{
  *distortion_coeffs_mat = cv::Mat::zeros(1, distortion_coeffs.size(), CV_64F);
  for (int k = 0; k < distortion_coeffs_mat->cols; k++)
  {
    distortion_coeffs_mat->at<double>(0, k) = distortion_coeffs[k];
  }
}

void CameraParam::convertIntrinsicsVectorToMatrix(const IntrinsicsCoeffs& intrinsics, cv::Mat* camera_matrix)
{
  *camera_matrix = cv::Mat::eye(3, 3, CV_64F);
  camera_matrix->at<double>(0, 0) = intrinsics[0];
  camera_matrix->at<double>(1, 1) = intrinsics[1];
  camera_matrix->at<double>(0, 2) = intrinsics[2];
  camera_matrix->at<double>(1, 2) = intrinsics[3];
}

DistortionModel CameraParam::stringToDistortion(const std::string& distortion_model, const std::string& camera_model)
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

bool CameraParam::equals(const CameraParam& other, double tol) const
{
  return utils::equal_with_abs_tol(intrinsics_, other.intrinsics_, tol) &&
         utils::equal_with_abs_tol(distortion_coeff_, other.distortion_coeff_, tol) &&
         (image_size_.width == other.image_size_.width && image_size_.height == other.image_size_.height) &&
         distortion_model_ == other.distortion_model_ && hardware_id_ == other.hardware_id_ &&
         T_R_C_.equals(other.T_R_C_, tol) && frame_ == other.frame_ && name_ == other.name_ &&
         utils::compareCvMatsUpToTol(K_, other.K_, tol) &&
         utils::compareCvMatsUpToTol(D_, other.D_);  // NOTE: do not comapre with P matrix as currently not calculated
}

const std::string CameraParam::toString() const
{
  std::stringstream out;
  out << "\nHardware ID: " << hardware_id_ << "\nIntrinsics: \n- fx " << fx() << "\n- fy " << fy() << "\n- cu " << cu()
      << "\n- cv " << cv() << "\nimage_size: \n- width: " << ImageWidth() << "\n- height: " << ImageHeight()
      << "\n- K: " << K_ << '\n'
      << "- link: " << frame_ << "\n"
      << "- name: " << name_ << "\n"
      << "- Distortion Model: " << distortionToString(distortion_model_) << '\n'
      << "- D: " << D_ << '\n'
      << "- P: " << P_ << '\n';

  return out.str();
}

}  // namespace dyno
