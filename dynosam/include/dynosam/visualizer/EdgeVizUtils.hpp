/*
 * Edge visualization utilities
 *
 * NOTE: The function definitions in `dynosam/src/visualizer/EdgeVizUtils.cc` are copied
 * verbatim from `dynosam/example/dyno_sam.cc` (namespace `edge_viz`) as requested.
 */
#pragma once

#include <Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "dynosam/backend/edge_map/localMap.hpp"

namespace edge_viz {

void setCameraParams(float fx_val, float fy_val, float cx_val, float cy_val);

void visualizeAssociationResult(const edge_map::localMapPtr& pLocalMap,
                                std::vector<std::vector<cv::Point3d>>& clusterClouds,
                                std::vector<cv::Vec3b>& clusterCloudColors);

void visualizeMergedLocalMap(const edge_map::localMapPtr& pLocalMap,
                             std::vector<std::vector<cv::Point3d>>& mergedClouds);

void getSlidingWindow(const edge_map::localMapPtr& pLocalMap,
                      std::vector<Eigen::Matrix4d>& sliding_window);

void saveEdgeKeyFrameTrajectory(const std::string& filename,
                                const edge_map::localMapPtr& pLocalMap);

}  // namespace edge_viz

