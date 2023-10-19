
#include <gtsam/geometry/Cal3Fisheye.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <thread>

using namespace dyno;

TEST(testCameraParams, basicConstruction)
{
  // Intrinsics.
  const std::vector<double> intrinsics_expected = { 458.654, 457.296, 367.215, 248.375 };
  //   // Distortion coefficients.
  const std::vector<double> distortion_expected = { -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05 };

  const cv::Size size_expected(752, 480);

  const std::string expected_distortion_model = "equidistant";
  const HardwareId hardware_id = 0;

  // Sensor extrinsics wrt. the body-frame.
  gtsam::Rot3 R_expected(0.0148655429818, -0.999880929698, 0.00414029679422, 0.999557249008, 0.0149672133247,
                         0.025715529948, -0.0257744366974, 0.00375618835797, 0.999660727178);
  gtsam::Point3 T_expected(-0.0216401454975, -0.064676986768, 0.00981073058949);
  gtsam::Pose3 pose_expected(R_expected, T_expected);

  const std::string frame = "cam_0";
  const std::string name = "Top Left";
  const std::string optics = "cil222";

  CameraParam params(intrinsics_expected, distortion_expected, size_expected, expected_distortion_model, hardware_id,
                     pose_expected, frame, name, optics);

  EXPECT_EQ(size_expected.width, params.ImageWidth());
  EXPECT_EQ(size_expected.height, params.ImageHeight());

  for (int c = 0u; c < 4u; c++)
  {
    EXPECT_DOUBLE_EQ(intrinsics_expected[c], params.intrinsics_[c]);
  }
  EXPECT_DOUBLE_EQ(intrinsics_expected[0], params.K_.at<double>(0, 0));
  EXPECT_DOUBLE_EQ(intrinsics_expected[1], params.K_.at<double>(1, 1));
  EXPECT_DOUBLE_EQ(intrinsics_expected[2], params.K_.at<double>(0, 2));
  EXPECT_DOUBLE_EQ(intrinsics_expected[3], params.K_.at<double>(1, 2));

  EXPECT_DOUBLE_EQ(intrinsics_expected[0], params.fx());
  EXPECT_DOUBLE_EQ(intrinsics_expected[1], params.fy());
  EXPECT_DOUBLE_EQ(intrinsics_expected[2], params.cu());
  EXPECT_DOUBLE_EQ(intrinsics_expected[3], params.cv());
  //   EXPECT_EQ(cam_params.intrinsics_.size(), 4u);
  gtsam::Cal3Fisheye gtsam_calib = utils::cameraParamsToGtsamCalibration<gtsam::Cal3Fisheye>(params);

  EXPECT_DOUBLE_EQ(intrinsics_expected[0], gtsam_calib.fx());
  EXPECT_DOUBLE_EQ(intrinsics_expected[1], gtsam_calib.fy());
  EXPECT_DOUBLE_EQ(0u, gtsam_calib.skew());
  EXPECT_DOUBLE_EQ(intrinsics_expected[2], gtsam_calib.px());
  EXPECT_DOUBLE_EQ(intrinsics_expected[3], gtsam_calib.py());

  EXPECT_TRUE(assert_equal(pose_expected, params.T_R_C_));

  for (int c = 0u; c < 4u; c++)
  {
    EXPECT_DOUBLE_EQ(distortion_expected[c], params.D_.at<double>(c));
  }
  EXPECT_EQ(params.D_.rows, 1u);
  EXPECT_EQ(params.D_.cols, 4u);
  EXPECT_DOUBLE_EQ(distortion_expected[0], gtsam_calib.k1());
  EXPECT_DOUBLE_EQ(distortion_expected[1], gtsam_calib.k2());
  EXPECT_DOUBLE_EQ(distortion_expected[2], gtsam_calib.k3());
  EXPECT_DOUBLE_EQ(distortion_expected[3], gtsam_calib.k4());
}

TEST(testCameraParams, convertDistortionVectorToMatrix)
{
  std::vector<double> distortion_coeffs;

  // 4 distortion params
  distortion_coeffs = { 1.0, -2.0, 1.3, 10 };
  cv::Mat distortion_coeffs_mat;
  CameraParam::convertDistortionVectorToMatrix(distortion_coeffs, &distortion_coeffs_mat);
  EXPECT_EQ(distortion_coeffs_mat.cols, distortion_coeffs.size());
  EXPECT_EQ(distortion_coeffs_mat.rows, 1u);
  for (size_t i = 0u; i < distortion_coeffs.size(); i++)
  {
    EXPECT_EQ(distortion_coeffs_mat.at<double>(0, i), distortion_coeffs.at(i));
  }

  // 5 distortion params
  distortion_coeffs = { 1, 1.2f, 3u, 4l, 5.34 };  //! randomize types as well
  CameraParam::convertDistortionVectorToMatrix(distortion_coeffs, &distortion_coeffs_mat);
  EXPECT_EQ(distortion_coeffs_mat.cols, distortion_coeffs.size());
  EXPECT_EQ(distortion_coeffs_mat.rows, 1u);
  for (size_t i = 0u; i < distortion_coeffs.size(); i++)
  {
    EXPECT_EQ(distortion_coeffs_mat.at<double>(0u, i), distortion_coeffs.at(i));
  }

  // n distortion params
  distortion_coeffs = { 1.0, 1.2, 3.2, 4.3, 5.34, 10203, 1818.9, 1.9 };
  CameraParam::convertDistortionVectorToMatrix(distortion_coeffs, &distortion_coeffs_mat);
  EXPECT_EQ(distortion_coeffs_mat.cols, distortion_coeffs.size());
  EXPECT_EQ(distortion_coeffs_mat.rows, 1u);
  for (size_t i = 0u; i < distortion_coeffs.size(); i++)
  {
    EXPECT_EQ(distortion_coeffs_mat.at<double>(0u, i), distortion_coeffs.at(i));
  }
}

