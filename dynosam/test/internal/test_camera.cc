#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace dyno;

TEST(Camera, project)
{
  Landmarks lmks;
  lmks.push_back(Landmark(0.0, 0.0, 1.0));
  lmks.push_back(Landmark(0.0, 0.0, 2.0));
  lmks.push_back(Landmark(0.0, 1.0, 2.0));
  lmks.push_back(Landmark(0.0, 10.0, 20.0));
  lmks.push_back(Landmark(1.0, 0.0, 2.0));

  CameraParam::IntrinsicsCoeffs intrinsics(4);
  CameraParam::DistortionCoeffs distortion(4);

  intrinsics.at(0) = 1.0;  // fx
  intrinsics.at(1) = 1.0;  // fy
  intrinsics.at(2) = 3.0;  // u0
  intrinsics.at(3) = 2.0;  // v0
  Keypoints expected_kpts;
  expected_kpts.push_back(Keypoint(intrinsics.at(2), intrinsics.at(3)));
  expected_kpts.push_back(Keypoint(intrinsics.at(2), intrinsics.at(3)));
  expected_kpts.push_back(Keypoint(3.0, 1.0 / 2.0 + 2.0));
  expected_kpts.push_back(Keypoint(3.0, 1.0 / 2.0 + 2.0));
  expected_kpts.push_back(Keypoint(1.0 / 2.0 + 3.0, 2.0));

  CameraParam camera_params(intrinsics, distortion, cv::Size(640, 480), "none", 1, gtsam::Pose3::identity(),
                            "base_link");

  Camera camera(camera_params);

  Keypoints actual_kpts;
  EXPECT_NO_THROW(camera.project(lmks, &actual_kpts));
  testing::compareKeypoints(expected_kpts, actual_kpts, 0.0001f);
}

TEST(Camera, backProjectSingleSimple)
{
  // Easy test first, back-project keypoint at the center of the image with
  // a given depth.
  CameraParam::IntrinsicsCoeffs intrinsics(4);
  CameraParam::DistortionCoeffs distortion(4);
  intrinsics.at(0) = 721.5377;  // fx
  intrinsics.at(1) = 721.5377;  // fy
  intrinsics.at(2) = 609.5593;  // u0
  intrinsics.at(3) = 172.8540;  // v0
  CameraParam camera_params(intrinsics, distortion, cv::Size(640, 480), "none", 1, gtsam::Pose3::identity(),
                            "base_link");

  Camera camera(camera_params);

  Keypoint kpt(camera_params.cu(), camera_params.cv());
  Landmark actual_lmk;
  double depth = 2.0;
  camera.backProject(kpt, depth, &actual_lmk);

  Landmark expected_lmk(0.0, 0.0, depth);
  EXPECT_NEAR(expected_lmk.x(), actual_lmk.x(), 0.0001);
  EXPECT_NEAR(expected_lmk.y(), actual_lmk.y(), 0.0001);
  EXPECT_NEAR(expected_lmk.z(), actual_lmk.z(), 0.0001);
}

TEST(Camera, backProjectMultipleSimple)
{
  // Easy test first, back-project keypoints at the center of the image with
  // different depths.
  CameraParam::IntrinsicsCoeffs intrinsics(4);
  CameraParam::DistortionCoeffs distortion(4);
  intrinsics.at(0) = 721.5377;  // fx
  intrinsics.at(1) = 721.5377;  // fy
  intrinsics.at(2) = 609.5593;  // u0
  intrinsics.at(3) = 172.8540;  // v0
  CameraParam camera_params(intrinsics, distortion, cv::Size(640, 480), "none", 1, gtsam::Pose3::identity(),
                            "base_link");

  Camera camera(camera_params);

  Keypoint kpt(camera_params.cu(), camera_params.cv());
  // Create 3 keypoints centered at image with different depths
  Keypoints kpts(3, kpt);
  Depths depths = { 2.0, 3.0, 4.5 };
  Landmarks actual_lmks;
  camera.backProject(kpts, depths, &actual_lmks);

  Landmarks expected_lmks;
  for (const auto& depth : depths)
  {
    expected_lmks.push_back(Landmark(0.0, 0.0, depth));
  }

  testing::compareLandmarks(actual_lmks, expected_lmks, 0.0001);
}

TEST(Camera, backProjectSingleTopLeft)
{
  // Back-project keypoint at the center of the image with a given depth.
  CameraParam::IntrinsicsCoeffs intrinsics(4);
  CameraParam::DistortionCoeffs distortion(4);

  double fx = 30.9 / 2.2;
  double fy = 12.0 / 23.0;
  double cu = 390.8;
  double cv = 142.2;

  intrinsics.at(0) = fx;  // fx
  intrinsics.at(1) = fy;  // fy
  intrinsics.at(2) = cu;  // u0
  intrinsics.at(3) = cv;  // v0
  CameraParam camera_params(intrinsics, distortion, cv::Size(640, 480), "none", 1, gtsam::Pose3::identity(),
                            "base_link");

  Camera camera(camera_params);

  Landmark actual_lmk;
  double depth = 2.0;
  Keypoint kpt(0.0, 0.0);  // Top-left corner
  camera.backProject(kpt, depth, &actual_lmk);

  Landmark expected_lmk(depth / fx * (-cu), depth / fy * (-cv), depth);
  EXPECT_NEAR(expected_lmk.x(), actual_lmk.x(), 0.0001);
  EXPECT_NEAR(expected_lmk.y(), actual_lmk.y(), 0.0001);
  EXPECT_NEAR(expected_lmk.z(), actual_lmk.z(), 0.0001);
}
