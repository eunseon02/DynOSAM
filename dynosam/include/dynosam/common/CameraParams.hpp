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

#include "dynosam/common/Types.hpp"
#include "dynosam/utils/YamlParser.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3Fisheye.h>
#include <eigen3/Eigen/Core>
#include <exception>

namespace dyno
{

class InvalidCameraCalibration : public std::runtime_error {
public:
  InvalidCameraCalibration(const std::string& what) : std::runtime_error(what) {}
};


enum class DistortionModel {
  NONE,
  RADTAN,
  EQUIDISTANT,
  FISH_EYE
};



class CameraParams
{
public:
  DYNO_POINTER_TYPEDEFS(CameraParams)

  using DistortionCoeffs = std::vector<double>;
  // fu, fv, cu, cv
  using IntrinsicsCoeffs = std::vector<double>;
  // eg pinhole
  using CameraModel = std::string;

  /**
   * @brief Constructs the parameters for a specific camera.
   *
   * Distortion model expects a string as either "none", "plumb_bob", "radial-tangential", "radtan", "equidistant" or
   * "kanna_brandt". The corresponding DistortionModel (enum) will then be assigned.
   *
   * @param intrinsics const IntrinsicsCoeffs& Coefficients for the intrinsics matrix. Should be in the form [fu fv cu
   * cv]
   * @param distortion const DistortionCoeffs& Coefficients for the camera distortion.
   * @param image_size const cv::Size& Image width and height
   * @param distortion_model_ const std::string& Expected distortion model to be used for this camera.
   * @param T_robot_camera const gtsam::Pose3& The camera extrinsics describing the transformation of the camera in the robot frame
   */
  CameraParams(const IntrinsicsCoeffs& intrinsics, const DistortionCoeffs& distortion, const cv::Size& image_size,
              const std::string& distortion_model, const gtsam::Pose3& T_robot_camera = gtsam::Pose3::Identity());

  CameraParams(const IntrinsicsCoeffs& intrinsics, const DistortionCoeffs& distortion, const cv::Size& image_size,
              const DistortionModel& distortion_model, const gtsam::Pose3& T_robot_camera = gtsam::Pose3::Identity());

  virtual ~CameraParams() = default;


  static CameraParams fromYamlFile(const std::string& file_path);

  inline double fx() const
  {
    return intrinsics_[0];
  }
  inline double fy() const
  {
    return intrinsics_[1];
  }
  inline double cu() const
  {
    return intrinsics_[2];
  }
  inline double cv() const
  {
    return intrinsics_[3];
  }
  inline int ImageWidth() const
  {
    return image_size_.width;
  }
  inline int ImageHeight() const
  {
    return image_size_.height;
  }

  inline const cv::Size& imageSize() const {
    return image_size_;
  }

  inline cv::Mat getCameraMatrix() const
  {
    return K_;
  }

  inline cv::Mat getDistortionCoeffs() const
  {
    return D_;
  }

  inline gtsam::Pose3 getExtrinsics() const
  {
    return T_robot_camera_;
  }



  static void convertDistortionVectorToMatrix(const DistortionCoeffs& distortion_coeffs,
                                              cv::Mat* distortion_coeffs_mat);

  static void convertIntrinsicsVectorToMatrix(const IntrinsicsCoeffs& intrinsics, cv::Mat* camera_matrix);

  /** Taken from: https://github.com/ethz-asl/image_undistort
   * @brief stringToDistortion
   * @param distortion_model
   * @param camera_model
   * @return actual distortion model enum class
   */
  static DistortionModel stringToDistortion(const std::string& distortion_model, const std::string& camera_model);

  bool equals(const CameraParams& other, double tol = 1e-9) const;

  const std::string toString() const;

  //specalisations for gtsam::Cal3Fisheye and gtsam::Cal3DS2 are provided
  template<typename CALIBRATION>
  CALIBRATION constructGtsamCalibration() const;

private:

  static void parseDistortionModel(const YamlParser& yaml_parser, DistortionModel* model);
  static void parseImgSize(const YamlParser& yaml_parser, cv::Size* image_size);
  static void parseBodyPoseCam(const YamlParser& yaml_parser,
                               gtsam::Pose3* body_Pose_cam);
  static void parseCameraIntrinsics(const YamlParser& yaml_parser,
                                    IntrinsicsCoeffs* intrinsics);
  static void parseCameraDistortion(const YamlParser& yaml_parser,
                                    DistortionCoeffs* distortion);


private:
  // updates cv Mat P
  // for now only works if FISH_EYE
  //   void estimateNewMatrixForDistortion();

  //! fu, fv, cu, cv
  const IntrinsicsCoeffs intrinsics_;
  const DistortionCoeffs distortion_coeff_;
  cv::Size image_size_;

  //! Distortion parameters
  const DistortionModel distortion_model_;

  const gtsam::Pose3 T_robot_camera_;  //! Transform of the camera frame to the robot frame

  //! OpenCV structures: needed to compute the undistortion map.
  //! 3x3 camera matrix K (last row is {0,0,1})
  //! stored as a CV_64F (double)
  cv::Mat K_;

  //! New camera matrix constructed from
  //! estimateNewCameraMatrixForUndistortRectify
  cv::Mat P_;

  cv::Mat D_;
};





}  // namespace dyno
