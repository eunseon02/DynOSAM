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

#include "dynosam/backend/MonoBackendTools.hpp"
#include "dynosam/utils/Numerical.hpp"
#include <gtsam/geometry/triangulation.h>
#include <opencv2/core/eigen.hpp>
#include <dynosam/utils/Accumulator.hpp>

#include "dynosam/common/ImageContainer.hpp"

namespace dyno {

namespace mono_backend_tools {


/**
 * @brief Triangulation of a point cluster with rotation compensated
 * Solves for the 3D positions (approximation) of a point cluster at the previous timestep given the 2D observations in two camera frames
 *
 */

gtsam::Point3Vector triangulatePoint3Vector(
  const gtsam::Pose3& X_world_camera_prev,
  const gtsam::Pose3& X_world_camera_curr,
  const gtsam::Matrix3& intrinsic,
  const gtsam::Point2Vector& observation_prev,
  const gtsam::Point2Vector& observation_curr,
  const gtsam::Matrix3& obj_rotation){

  const gtsam::Matrix3 X_world_camera_prev_rot = X_world_camera_prev.rotation().matrix();
  const gtsam::Point3 X_world_camera_prev_trans = X_world_camera_prev.translation();
  const gtsam::Matrix3 X_world_camera_curr_rot = X_world_camera_curr.rotation().matrix();
  const gtsam::Point3 X_world_camera_curr_trans = X_world_camera_curr.translation();

  gtsam::Matrix3 K_inv = intrinsic.inverse();

  gtsam::Matrix3 project_inv_prev = X_world_camera_prev_rot*K_inv;
  gtsam::Matrix3 project_inv_curr = X_world_camera_curr_rot*K_inv;

  // std::cout << observation_prev.size() << std::endl;
  // for (int i = 0; i < observation_prev.size(); i++){
  //   std::cout << "Obv prev " << i << "\n" << observation_prev[i] << std::endl;
  // }

  gtsam::Point2 centroid_2d_prev = computeCentroid(observation_prev);
  gtsam::Point3 centroid_2d_homo(centroid_2d_prev.x(), centroid_2d_prev.y(), 1.0);
  gtsam::Point3 centroid_3d_prev = K_inv*centroid_2d_homo;
  gtsam::Point3 coeffs_scale = (obj_rotation - Eigen::Matrix3d::Identity())*centroid_3d_prev;

  gtsam::Point3 results_single = obj_rotation*X_world_camera_prev_trans - X_world_camera_curr_trans;

  // std::cout << coeffs_scale << std::endl;
  // std::cout << results_single << std::endl;

  int n_points = observation_prev.size();
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Zero(3*n_points, 1+2*n_points);
  Eigen::VectorXd results(3*n_points);
  // std::cout << "Coeffs initial\n" << coeffs << std::endl;
  for (int i_points = 0; i_points < n_points; i_points++){
    gtsam::Point3 this_obv_prev(observation_prev[i_points].x(), observation_prev[i_points].y(), 1.0);
    gtsam::Point3 this_obv_curr(observation_curr[i_points].x(), observation_curr[i_points].y(), 1.0);

    gtsam::Point3 coeffs_prev = obj_rotation*(project_inv_prev*this_obv_prev);
    gtsam::Point3 coeffs_curr = project_inv_curr*this_obv_curr;

    // coeffs << coeffs_curr.x(), -coeffs_prev.x(),
    //           coeffs_curr.y(), -coeffs_prev.y(),
    //           coeffs_curr.z(), -coeffs_prev.z();
    coeffs.block<3, 1>(3*i_points, 0) = coeffs_scale;
    coeffs.block<3, 1>(3*i_points, 1+2*i_points) = coeffs_curr;
    coeffs.block<3, 1>(3*i_points, 2+2*i_points) = -coeffs_prev;

    // Eigen::Vector3d results(results_gtsam.x(), results_gtsam.y(), results_gtsam.z());
    results.block<3, 1>(3*i_points, 0) = results_single;
  }

  // std::cout << "Coeffs constructed\n" << coeffs << std::endl;

  gtsam::Point3Vector points_world;
  Eigen::VectorXd depths_lstsq = coeffs.colPivHouseholderQr().solve(results);

  // std::cout << depths_lstsq << std::endl;

  for (int i_points = 0; i_points < n_points; i_points++){
    gtsam::Point3 this_obv_prev(observation_prev[i_points].x(), observation_prev[i_points].y(), 1.0);
    gtsam::Point3 this_obv_curr(observation_curr[i_points].x(), observation_curr[i_points].y(), 1.0);
    Eigen::Vector2d depths(depths_lstsq(1+2*i_points), depths_lstsq(2+2*i_points));

    gtsam::Point3 this_point_curr = K_inv*(depths(0)*this_obv_curr);
    gtsam::Point3 this_point_prev = K_inv*(depths(1)*this_obv_prev);

    gtsam::Point3 this_point_world = X_world_camera_prev_rot*this_point_prev + X_world_camera_prev_trans;
    points_world.push_back(this_point_world);
  }



  return points_world;

}


/**
 * @brief Triangulation of a point cluster with rotation compensated, and not expanding H_world into H_L
 * Solves for the 3D positions (approximation) of a point cluster at the previous timestep given the 2D observations in two camera frames
 *
 */

gtsam::Point3Vector triangulatePoint3VectorNonExpanded(
  const gtsam::Pose3& X_world_camera_prev,
  const gtsam::Pose3& X_world_camera_curr,
  const gtsam::Matrix3& intrinsic,
  const gtsam::Point2Vector& observation_prev,
  const gtsam::Point2Vector& observation_curr,
  const gtsam::Matrix3& obj_rotation){

  const gtsam::Matrix3 X_world_camera_prev_rot = X_world_camera_prev.rotation().matrix();
  const gtsam::Point3 X_world_camera_prev_trans = X_world_camera_prev.translation();
  const gtsam::Matrix3 X_world_camera_curr_rot = X_world_camera_curr.rotation().matrix();
  const gtsam::Point3 X_world_camera_curr_trans = X_world_camera_curr.translation();

  gtsam::Matrix3 K_inv = intrinsic.inverse();

  gtsam::Matrix3 project_inv_prev = X_world_camera_prev_rot*K_inv;
  gtsam::Matrix3 project_inv_curr = X_world_camera_curr_rot*K_inv;

  // std::cout << observation_prev.size() << std::endl;
  // for (int i = 0; i < observation_prev.size(); i++){
  //   std::cout << "Obv prev " << i << "\n" << observation_prev[i] << std::endl;
  // }

  gtsam::Point3 results = obj_rotation*X_world_camera_prev_trans - X_world_camera_curr_trans;

  // std::cout << coeffs_scale << std::endl;
  // std::cout << results_single << std::endl;

  int n_points = observation_prev.size();
  gtsam::Point3Vector points_world;
  for (int i_points = 0; i_points < n_points; i_points++){
    gtsam::Point3 this_obv_prev(observation_prev[i_points].x(), observation_prev[i_points].y(), 1.0);
    gtsam::Point3 this_obv_curr(observation_curr[i_points].x(), observation_curr[i_points].y(), 1.0);

    gtsam::Point3 coeffs_prev = obj_rotation*(project_inv_prev*this_obv_prev);
    gtsam::Point3 coeffs_curr = project_inv_curr*this_obv_curr;

    Eigen::MatrixXd coeffs(3, 2);
    coeffs << coeffs_curr.x(), -coeffs_prev.x(),
              coeffs_curr.y(), -coeffs_prev.y(),
              coeffs_curr.z(), -coeffs_prev.z();

    Eigen::VectorXd depths_lstsq = coeffs.colPivHouseholderQr().solve(results);

    gtsam::Point3 this_point_curr = K_inv*(depths_lstsq(0)*this_obv_curr);
    gtsam::Point3 this_point_prev = K_inv*(depths_lstsq(1)*this_obv_prev);

    gtsam::Point3 this_point_world = X_world_camera_prev_rot*this_point_prev + X_world_camera_prev_trans;
    points_world.push_back(this_point_world);
  }

  // std::cout << "Coeffs constructed\n" << coeffs << std::endl;


  // std::cout << depths_lstsq << std::endl;

  return points_world;

}


/**
 * @brief Compute point coorindates in the camera frame from their depths and pixels
 *
 */

gtsam::Point3Vector depthsToPoints(const gtsam::Matrix3& intrinsic, const Eigen::VectorXd& depths, const gtsam::Point2Vector& observations){
  CHECK_EQ(depths.size(), observations.size());

  gtsam::Matrix3 K_inv = intrinsic.inverse();

  int n_points = depths.size();
  gtsam::Point3Vector points_camera;
  for (int i_points = 0; i_points < n_points; i_points++){
    gtsam::Point3 this_obv_homo(observations[i_points].x(), observations[i_points].y(), 1.0);
    gtsam::Point3 this_point = K_inv*(depths(i_points)*this_obv_homo);
    points_camera.push_back(this_point);
  }

  return points_camera;
}


/**
 * @brief Compute point depths in the camera frame using camera intrinsics
 * points - 3D point coordinates in the camera frame
 *
 */

Eigen::VectorXd pointsToDepths(const gtsam::Matrix3& intrinsic, const gtsam::Point3Vector& points){
  int n_points = points.size();
  Eigen::VectorXd depths(n_points);

  for (int i_points = 0; i_points < n_points; i_points++){
    gtsam::Point3 this_obv_nonnorm = intrinsic*points[i_points];
    depths[i_points] = this_obv_nonnorm.z();
  }

  return depths;
}


/**
 * @brief Use the standard deviation of the depths of a point cluster to check whether this cluster is reasonable
 * std_thres - a tune-able threshold for the standard deviation
 *
 */

bool checkClusterViaStd(const double std_thres, const Eigen::VectorXd& depths){
  int n_depths = depths.size();
  double standard_deviation = 1000.0;
  double sum = 0.0;
  for (int i_depths = 0; i_depths < n_depths; i_depths++){
    sum += depths[i_depths];
  }

  double mean = sum / n_depths;

  for (int i_depths = 0; i_depths < n_depths; i_depths++){
    standard_deviation += pow(depths[i_depths] - mean, 2);
  }

  standard_deviation = sqrt(standard_deviation / n_depths);

  return (standard_deviation <= std_thres);
}

/**
 * @brief Estimate the depth of the object using the background pixels below the object
 *
 * returns std::nullopt when the object is not found on this frame
 */

std::optional<double> estimateDepthFromRoad(const gtsam::Pose3& X_world_camera_prev,
  const gtsam::Pose3& X_world_camera_curr,
  const Camera::Ptr camera,
  const cv::Mat& prev_semantic_mask,
  const cv::Mat& curr_semantic_mask,
  const ImageWrapper<ImageType::ClassSegmentation>& prev_class_seg,
  const ImageWrapper<ImageType::ClassSegmentation>& curr_class_seg,
  const cv::Mat& prev_optical_flow,
  const ObjectId obj_id,
  gtsam::Point2Vector& observation_prev,
  gtsam::Point2Vector& observation_curr,
  gtsam::Point3Vector& triangualted_points,
  Depths& z_camera){



  ObjectIds prev_object_ids = vision_tools::getObjectLabels(prev_semantic_mask);

  //check if object id is in the previous semantic mask
  if(std::find(prev_object_ids.begin(), prev_object_ids.end(), obj_id) == prev_object_ids.end()) {
    LOG(WARNING) << "Could not find object id " << obj_id << " in previous semantic mask";
    return std::optional<double>{};
  }

  cv::Mat obj_mask = (prev_semantic_mask == obj_id);

  // cv::imshow("Object Mask", obj_mask);

  // cv::Mat obj_mask_curr = (curr_semantic_mask == obj_id);
  // cv::imshow("Object Mask Current", obj_mask_curr);

  // cv::Mat dilated_obj_mask;
  // cv::Mat dilate_element = cv::getStructuringElement(cv::MORPH_RECT,
  //                                                   cv::Size(1, 11)); // a rectangle of 1*5
  // cv::dilate(obj_mask, dilated_obj_mask, dilate_element, cv::Point(0, 10)); // defining anchor point so it only erode down
  // // cv::imshow("Object Mask Dilated", dilated_obj_mask);

  // std::vector<std::vector<cv::Point> > contours;
  // std::vector<cv::Vec4i> hierarchy;
  // cv::findContours(dilated_obj_mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

  //NOTE: while using virtual kitti assume perfect masks so no need for dilation!!!
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(obj_mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

  if (contours.size()>0){
    // std::cout << "There are " << contours.size() << " contours of obj " << obj_id << " we can find on the previous image.\n";
  }

  if (contours.size() < 1){
    return std::optional<double>{};
  }

  // cv::Mat contours_viz = cv::Mat::zeros(obj_mask.size(), CV_8UC3);
  // for (int i_contour = 0; i_contour < contours.size(); i_contour++){
  //   for (int i_point = 0; i_point < contours[i_contour].size(); i_point++){
  //     // std::cout << "Contour point at " << contours[i_contour][i_point].x << ", " << contours[i_contour][i_point].y << std::endl;
  //     contours_viz.at<uchar>(contours[i_contour][i_point].y, contours[i_contour][i_point].x) = 255;
  //   }
  // }
  // for( size_t i = 0; i< contours.size(); i++ ){
  //   cv::Scalar color = cv::Scalar(0, 0, 255);
  //   drawContours( contours_viz, contours, (int)i, color, 1, cv::LINE_8, hierarchy, 0 );
  // }
  // cv::imshow("Contours " + std::to_string(obj_id), contours_viz);
  // cv::waitKey(1);

  cv::Mat ground_pixel_prev_viz = cv::Mat::zeros(obj_mask.size(), CV_8U);
  cv::Mat ground_pixel_viz = cv::Mat::zeros(obj_mask.size(), CV_8U);


  int n_points = 0;
  for (int i_contour = 0; i_contour < contours.size(); i_contour++){
    n_points += contours[i_contour].size();
  }
  std::vector<cv::Point> full_contour;
  full_contour.reserve(n_points);
  for (int i_contour = 0; i_contour < contours.size(); i_contour++){
    full_contour.insert(full_contour.end(), contours[i_contour].begin(), contours[i_contour].end());
  }
  // n_points = full_contour.size();
  std::vector<Keypoint> ground_pixels_prev;
  std::vector<Keypoint> ground_pixels_curr;
  for (int i_points = 0; i_points < n_points; i_points++){
    // Feature::Ptr this_obj_feature = prev_frame.at(obj_tracklets[i_points]);
    // Keypoint this_pixel = this_obj_feature->keypoint_;
    Keypoint this_pixel = gtsam::Point2(full_contour[i_points].x, full_contour[i_points].y);

    int u = functional_keypoint::u(this_pixel);
    const cv::Mat& this_semantic_col = prev_semantic_mask.col(u);
    int i_row = 0;
    // Search upwards up to the contour until we hit the descired object patch
    for (i_row = prev_semantic_mask.rows-1; i_row >= functional_keypoint::v(this_pixel); i_row--){
      //check for road semantics
      int v = i_row+1;
      const Keypoint kp_prev(u, v);
      //TODO: fix auto casting of the at function here! The template cannot diffirentiate between cv::Mat and the ImageWrapper becuase of the explicit cast
      bool prev_is_road =
        functional_keypoint::at<int>(kp_prev, prev_class_seg) == ImageType::ClassSegmentation::Labels::Road;

      if (this_semantic_col.at<ObjectId>(i_row) == obj_id && prev_is_road){
         // Verify that the previous pixel is a background pixel - in the case of an object being blocked by another one below it
        // Also make sure the pixel is not already at the bottom of the image - does not have a previous pixel
        if (i_row < prev_semantic_mask.rows-1 && this_semantic_col.at<ObjectId>(i_row+1) == 0){
          double flow_xe = static_cast<double>(prev_optical_flow.at<cv::Vec2f>(v, u)[0]);
          double flow_ye = static_cast<double>(prev_optical_flow.at<cv::Vec2f>(v, u)[1]);

          OpticalFlow flow(flow_xe, flow_ye);
          Keypoint ground_pixel_curr = Feature::CalculatePredictedKeypoint(Keypoint(u, v), flow);
          bool curr_is_road =
            functional_keypoint::at<int>(ground_pixel_curr, curr_class_seg) == ImageType::ClassSegmentation::Labels::Road;
          if (curr_semantic_mask.at<ObjectId>(ground_pixel_curr.y(), ground_pixel_curr.x()) == 0 && curr_is_road){
            ground_pixels_prev.push_back(Keypoint(u, v));
            ground_pixel_prev_viz.at<uchar>(v, u) = 255;
            ground_pixels_curr.push_back(ground_pixel_curr);

            ground_pixel_viz.at<uchar>(ground_pixel_curr.y(), ground_pixel_curr.x()) = 255;
          }
        }
      }
    }
  }

  // std::cout << "Passed ground pixel search.\n";

  // cv::imshow("Ground pixels prev", ground_pixel_prev_viz);
  cv::imshow("Ground pixels", ground_pixel_viz);
  cv::waitKey(1);

  int n_ground_pixels = ground_pixels_prev.size();
  std::vector<gtsam::Point3> ground_points;
  std::vector<gtsam::Pose3> cam_poses({X_world_camera_prev, X_world_camera_curr});

  const auto& camera_params = camera->getParams();
  auto gtsam_calibration = camera_params.constructGtsamCalibration<Camera::CalibrationType>();

  gtsam::CameraSet<Camera::CameraImpl> cameras = {
    Camera::CameraImpl(X_world_camera_prev, gtsam_calibration),
    Camera::CameraImpl(X_world_camera_curr, gtsam_calibration)
  };


  double depth_sum = 0.0;
  int num_degenerate_points = 0;

  //this is not depth - its Z (in camera frame!!)
  utils::Accumulatord depth_accumulator;
  gtsam::Point2Vector observation_prev_pre_filter;
  gtsam::Point2Vector observation_curr_pre_filter;
  gtsam::Point3Vector triangualted_points_pre_filter;

  for (int i_ground_pixels = 0; i_ground_pixels < n_ground_pixels; i_ground_pixels++){
    gtsam::Point2Vector measurments({ground_pixels_prev[i_ground_pixels], ground_pixels_curr[i_ground_pixels]});
    gtsam::TriangulationResult this_point_result;
    try {
      gtsam::TriangulationParameters triangulation_params;
      //values in the matrix > rankTolerance will add to the rank
      // triangulation_params.rankTolerance = 30.0;
      // triangulation_params.enableEPI = true;
      // triangulation_params.dynamicOutlierRejectionThreshold = 0.1;
      // triangulation_params.landmarkDistanceThreshold = 20;

      this_point_result = gtsam::triangulateSafe(cameras, measurments, triangulation_params);
      // gtsam::Point3 point = gtsam::triangulatePoint3(cameras, measurments, 10, false, nullptr, true);
      // size_t i = 0;
      // double maxReprojError = 0.0;
      // for(const auto& camera: cameras) {
      //   const gtsam::Pose3& pose = camera.pose();
      //   // verify that the triangulated point lies in front of all cameras
      //   // Only needed if this was not yet handled by exception
      //   const gtsam::Point3& p_local = pose.transformTo(point);
      //   if (p_local.z() <= 0)
      //     this_point_result = gtsam::TriangulationResult::BehindCamera();
      //   // Check reprojection error
      //   if (triangulation_params.dynamicOutlierRejectionThreshold > 0) {
      //     const auto& zi = measurments.at(i);
      //     gtsam::Point2 reprojectionError = camera.reprojectionError(point, zi);
      //     maxReprojError = std::max(maxReprojError, reprojectionError.norm());
      //   }
      //   i += 1;
      // }
      // if (triangulation_params.dynamicOutlierRejectionThreshold > 0
      //     && maxReprojError > triangulation_params.dynamicOutlierRejectionThreshold)
      //   this_point_result= gtsam::TriangulationResult::Outlier();

      // this_point_result = gtsam::TriangulationResult(point);
    }
    catch(const gtsam::TriangulationCheiralityException&) {
       continue;
    }
    catch(const gtsam::CheiralityException&) {
       continue;
    }
    catch(...) {continue;}

    if(this_point_result.degenerate()) {
      num_degenerate_points++;
    }

    if(!this_point_result) {continue; }

    gtsam::Point3 this_point = *this_point_result;
    ground_points.push_back(this_point);

    // TODO: Filter these points/depths and reject outliers
    gtsam::Point3 this_point_camera_curr = X_world_camera_curr.transformTo(this_point);
    observation_prev_pre_filter.push_back(ground_pixels_prev[i_ground_pixels]);
    observation_curr_pre_filter.push_back(ground_pixels_curr[i_ground_pixels]);
    triangualted_points_pre_filter.push_back(this_point);


    depth_accumulator.Add(this_point_camera_curr.z());

  }

  // LOG(INFO) << "num degenerate points " << num_degenerate_points << " out of " << n_ground_pixels << " with std " << depth_accumulator.StandardDeviation();
  //remove outliers, if value within +/- this value from the mean
  //must be bigger than one... dumbass
  const double kStdOutlierThreshold = 1.5;
  double obj_depth_mean = depth_accumulator.Mean();
  double obj_depth_std = depth_accumulator.StandardDeviation();

  const auto min_value = obj_depth_mean - kStdOutlierThreshold * obj_depth_std;
  const auto max_value = obj_depth_mean + kStdOutlierThreshold * obj_depth_std;

  utils::Accumulatord depth_accumulator_filtered;
  const auto& samples = depth_accumulator.GetAllSamples();
  //TODO: clean up and make function? How to know which indicies remain after filtering?
  for(size_t i = 0; i < samples.size(); i++) {
    auto depth = samples.at(i);
    if(depth > min_value && depth < max_value) {
      depth_accumulator_filtered.Add(depth);
      observation_prev.push_back(observation_prev_pre_filter.at(i));
      observation_curr.push_back(observation_curr_pre_filter.at(i));
      triangualted_points.push_back(triangualted_points_pre_filter.at(i));
      z_camera.push_back(depth);
    }
  }

  LOG(INFO) << "Old mean " << obj_depth_mean << " old std " << obj_depth_std
    << ". new mean " << depth_accumulator_filtered.Mean() << " new std " << depth_accumulator_filtered.StandardDeviation();

  return std::optional<double>(depth_accumulator_filtered.Mean());
}

/**
 * @brief Estimate the depth of the object using an approximate dimension of the object
 * Assuming the object is flat/facing the camera with a surface normal to the camera optical axis
 *
 * //TODO: Jesse - should be tracked object points not mask?
 *
 */

std::optional<double> estimateDepthFromDimension(const cv::Mat& semantic_mask,
                                  const Camera::Ptr camera,
                                  const ObjectId obj_id,
                                  const double obj_width){

  ObjectIds object_ids = vision_tools::getObjectLabels(semantic_mask);

  //check if object id is in the previous semantic mask
  if(std::find(object_ids.begin(), object_ids.end(), obj_id) == object_ids.end()) {
    LOG(WARNING) << "Could not find object id " << obj_id << " in semantic mask";
    return std::optional<double>{};
  }

  cv::Mat obj_mask = (semantic_mask == obj_id);
  cv::Mat close_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                    cv::Size(11, 11)); // a circle with diameter 11
  cv::Mat closed_mask;
  cv::morphologyEx(obj_mask, closed_mask, cv::MORPH_CLOSE, close_element);

  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(closed_mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

  if (contours.size() < 1){
    return std::optional<double>{};
  }

  int n_points = 0;
  for (int i_contour = 0; i_contour < contours.size(); i_contour++){
    n_points += contours[i_contour].size();
  }
  std::vector<cv::Point> full_contour;
  full_contour.reserve(n_points);
  for (int i_contour = 0; i_contour < contours.size(); i_contour++){
    full_contour.insert(full_contour.end(), contours[i_contour].begin(), contours[i_contour].end());
  }
  std::vector<cv::Point> contours_poly;
  cv::approxPolyDP(full_contour, contours_poly, 3.0, true); // epsilon = 3
  cv::Rect boundingbox = boundingRect(contours_poly);
  double obj_pixel_width = boundingbox.width;

  const auto& camera_params = camera->getParams();
  double obj_depth = obj_width*camera_params.fx()/obj_pixel_width; // assuming the object is normal to the optical axis, and aligned with x axis

  return std::optional<double>(obj_depth);
}


/**
 * @brief Check the ratio between the estimated scale of the object between frames
 * prev_points and curr_points are expected to be the same size and same order in point correspondences
 *
 */

bool validateScaleConsistency(const gtsam::Point3Vector& prev_points,
                              const gtsam::Point3Vector& curr_points,
                              double scale_ratio_thres){
  CHECK_EQ(prev_points.size(), curr_points.size());

  int n_points = prev_points.size();
  int n_pair = 0;
  double prev_distance_sum = 0.0;
  double curr_distance_sum = 0.0;

  for (int i = 0; i < n_points; i++){
    for (int j = i+1; j < n_points; j++){
      gtsam::Point3 prev_this_pair = prev_points[i] - prev_points[j];
      gtsam::Point3 curr_this_pair = curr_points[i] - curr_points[j];
      prev_distance_sum += prev_this_pair.norm();
      curr_distance_sum += curr_this_pair.norm();
      n_pair++;
    }
  }

  double prev_distance_average = prev_distance_sum/(double) n_pair;
  double curr_distance_average = curr_distance_sum/(double) n_pair;

  double scale_ratio = curr_distance_average / prev_distance_average;
  // We should allow the thres to be bi-directional.
  // i.e. prev can be bigger or smaller than curr scale, within a ratio thres
  // Now we expect scale ratio thres to be the lower bound
  if (scale_ratio_thres > 1){
    scale_ratio_thres = 1/scale_ratio_thres;
  }
  if(scale_ratio >= scale_ratio_thres || scale_ratio <= 1/scale_ratio_thres){
    return true;
  } else {
    return false;
  }
}

} //mono_backend_tools
} //dyno
