#pragma once



#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3Fisheye.h>

namespace dyno
{
class CameraParam
{
public:

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
   * @param intrinsics_ const IntrinsicsCoeffs& Coefficients for the intrinsics matrix. Should be in the form [fu fv cu
   * cv]
   * @param distortion_ const DistortionCoeffs& Coefficients for the camera distortion.
   * @param image_size_ const cv::Size& Image width and height
   * @param distortion_model_ const std::string& Expected distortion model to be used for this camera.
   * @param hardware_id_ const HardwareId& Associated hardware ID of this camera. Should be unique and is used
   *  to associate these parameters with a physical camera
   * @param T_R_C_ const gtsam::Pose3& The camera extrinsics describing the pose of the camera realative to the robot
   * body frame (eg. body frame to camera frame)
   * @param frame_ const std::string& frame_ The frame of this camera (usually extracted from the ROS tf tree).
   * @param name_ const std::string& name_ Human readaible name for this camera. Defaults to empty.
   */
  CameraParam(const IntrinsicsCoeffs& intrinsics, const DistortionCoeffs& distortion, const cv::Size& image_size,
              const std::string& distortion_model, const gtsam::Pose3& T_R_C);

  virtual ~CameraParam() = default;

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
    return T_R_C_;
  }

  inline std::string getCameraName() const
  {
    return name_;
  }

  inline std::string getOpticsType() const
  {
    return optics_;
  }

  inline void rescaleIntrinsics(double width, double height)
  {
    double x_scale = static_cast<double>(width) / static_cast<double>(image_size_.width);
    double y_scale = static_cast<double>(height) / static_cast<double>(image_size_.height);
    intrinsics_.at(0) *= x_scale;
    intrinsics_.at(1) *= y_scale;
    intrinsics_.at(2) *= x_scale;
    intrinsics_.at(3) *= y_scale;
    convertIntrinsicsVectorToMatrix(intrinsics_, &K_);
    image_size_.width = width;
    image_size_.height = height;
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

  bool equals(const CameraParam& other, double tol = 1e-9) const;

  const std::string toString() const;

public:
  // updates cv Mat P
  // for now only works if FISH_EYE
  //   void estimateNewMatrixForDistortion();

  //! fu, fv, cu, cv
  IntrinsicsCoeffs intrinsics_;
  const DistortionCoeffs distortion_coeff_;
  cv::Size image_size_;

  //! Distortion parameters
  const DistortionModel distortion_model_;

  const gtsam::Pose3 T_R_C_;  // transform of the camera from the robot frame (base_link) -> the camera wrt the base
                              // link

  //! OpenCV structures: needed to compute the undistortion map.
  //! 3x3 camera matrix K (last row is {0,0,1})
  //! stored as a CV_64F (double)
  cv::Mat K_;

  //! New camera matrix constructed from
  //! estimateNewCameraMatrixForUndistortRectify
  cv::Mat P_;

  cv::Mat D_;
};

using CameraParams = std::vector<CameraParam>;

}  // namespace dyno

