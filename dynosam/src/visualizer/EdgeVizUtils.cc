// Function definitions copied verbatim from `dynosam/example/dyno_sam.cc`
// (namespace `edge_viz`) as requested.

#include "dynosam/visualizer/EdgeVizUtils.hpp"

#include <glog/logging.h>

#include <fstream>
#include <iomanip>

#include <sophus/se3.hpp>

namespace edge_viz {

// Camera parameters for visualization (similar to localmapping.cc)
static float fx = 525.0f;
static float fy = 525.0f;
static float cx = 319.5f;
static float cy = 239.5f;

void setCameraParams(float fx_val, float fy_val, float cx_val, float cy_val) {
  fx = fx_val;
  fy = fy_val;
  cx = cx_val;
  cy = cy_val;
}

// Visualize association result (from localmapping.cc lines 27-78)
void visualizeAssociationResult(const dyno::localMapPtr& pLocalMap,
                                std::vector<std::vector<cv::Point3d>>& clusterClouds,
                                std::vector<cv::Vec3b>& clusterCloudColors) {
  clusterClouds.clear();
  clusterCloudColors.clear();

  int skipped_small = 0;
  int skipped_missing = 0;
  for (const auto& cluster : pLocalMap->mvEleEdgeClusters) {
    std::vector<unsigned int> edgeIDs = cluster.mvElementEdgeIDs;
    std::vector<int> edgeIdx;
    if (edgeIDs.size() < 5) {
      skipped_small++;
      continue;
    }
    bool all_found = true;
    for (size_t i = 0; i < edgeIDs.size(); ++i) {
      auto it = pLocalMap->mmElementID2index.find(edgeIDs[i]);
      if (it == pLocalMap->mmElementID2index.end()) {
        all_found = false;
        skipped_missing++;
        break;
      }
      int index = it->second;
      edgeIdx.push_back(index);
    }
    if (!all_found || edgeIdx.empty()) continue;

    // Edge cluster point cloud
    std::vector<cv::Point3d> clusterCloud;
    cv::Vec3b color = cluster.visColor;

    for (size_t i = 0; i < edgeIdx.size(); ++i) {
      int kf_edge_idx = pLocalMap->mvElementEdges[edgeIdx[i]].kf_edge_idx;
      int kf_id = pLocalMap->mvElementEdges[edgeIdx[i]].kf_id;
      
      // Safety check: verify kf_id exists in mmKFID2KFindex
      auto kf_idx_it = pLocalMap->mmKFID2KFindex.find(kf_id);
      if (kf_idx_it == pLocalMap->mmKFID2KFindex.end()) {
        skipped_missing++;
        continue;
      }
      int kf_idx = kf_idx_it->second;
      
      // Safety check: verify kf_idx is within bounds
      if (kf_idx < 0 || static_cast<size_t>(kf_idx) >= pLocalMap->mvKeyFrames.size()) {
        skipped_missing++;
        continue;
      }
      
      // Safety check: verify keyframe pointer is valid
      if (!pLocalMap->mvKeyFrames[kf_idx]) {
        skipped_missing++;
        continue;
      }
      
      // Safety check: verify kf_edge_idx is within bounds
      if (kf_edge_idx < 0 || static_cast<size_t>(kf_edge_idx) >= pLocalMap->mvKeyFrames[kf_idx]->mvEdges.size()) {
        skipped_missing++;
        continue;
      }

      Edge& edge = pLocalMap->mvKeyFrames[kf_idx]->mvEdges[kf_edge_idx];
      // Global pose of current map element edge
      Eigen::Matrix4d Trans_curr = pLocalMap->mvKeyFrames[kf_idx]->KF_pose_g.matrix();
      // Relative pose of current map element edge w.r.t. local map reference frame
      Eigen::Matrix4d Trans_ref_curr = Trans_curr;
      // Point cloud of single edge map element
      std::vector<cv::Point3d> cloud;
      // Calculate point cloud of single edge map element
      for (size_t j = 0; j < edge.mvPoints.size(); ++j) {
        orderedEdgePoint pt = edge.mvPoints[j];
        // Calculate 3D coordinates
        float x = (float(pt.x) - cx) / fx * pt.depth;
        float y = (float(pt.y) - cy) / fy * pt.depth;
        float z = pt.depth;
        Eigen::Vector4d points(x, y, z, 1);
        // Reproject to get new projection points
        points = Trans_ref_curr * points;
        cv::Point3d point(points.x(), points.y(), points.z());
        cloud.push_back(point);
      }
      clusterCloud.insert(clusterCloud.end(), cloud.begin(), cloud.end());
    }
    if (!clusterCloud.empty()) {
      clusterClouds.push_back(clusterCloud);
      clusterCloudColors.push_back(color);
    }
  }

  // Debug: Log filtering results
  static int call_count = 0;
  if (++call_count % 100 == 0) {
    LOG(INFO) << "visualizeAssociationResult: total_clusters=" << pLocalMap->mvEleEdgeClusters.size()
              << ", skipped_small=" << skipped_small << ", skipped_missing=" << skipped_missing
              << ", output_clusters=" << clusterClouds.size();
  }
}

// Visualize merged local map
void visualizeMergedLocalMap(const dyno::localMapPtr& pLocalMap,
                             std::vector<std::vector<cv::Point3d>>& mergedClouds,
                             std::vector<cv::Vec3b>& mergedCloudColors) {
  mergedClouds.clear();
  mergedCloudColors.clear();
  mergedClouds.reserve(pLocalMap->mvEleEdgeClusters.size());
  mergedCloudColors.reserve(pLocalMap->mvEleEdgeClusters.size());

  int total_clusters = pLocalMap->mvEleEdgeClusters.size();
  int merged = 0;
  int empty_merged = 0;

  for (const auto& cluster : pLocalMap->mvEleEdgeClusters) {
    if (cluster.mbMerged == false) continue;
    merged++;
    std::vector<cv::Point3d> merged_cloud = cluster.mvMergedCloud_ref;
    if (merged_cloud.empty()) {
      empty_merged++;
      continue;
    }
    mergedClouds.push_back(merged_cloud);
    mergedCloudColors.push_back(cluster.visColor);
  }

  // Debug: Log filtering results
  static int call_count = 0;
  if (++call_count % 100 == 0) {
    LOG(INFO) << "visualizeMergedLocalMap: total_clusters=" << total_clusters << ", merged=" << merged
              << ", empty_merged=" << empty_merged << ", output_clusters=" << mergedClouds.size();
  }
}

// Get sliding window poses (from localmapping.cc lines 107-115)
void getSlidingWindow(const dyno::localMapPtr& pLocalMap,
                      std::vector<Eigen::Matrix4d>& sliding_window) {
  sliding_window.clear();
  for (size_t i = 0; i < pLocalMap->mvKeyFrames.size(); ++i) {
    sliding_window.push_back(pLocalMap->mvKeyFrames[i]->KF_pose_g.matrix());
  }
}

// Save edge keyframe trajectory in TUM format
void saveEdgeKeyFrameTrajectory(const std::string& filename,
                                const dyno::localMapPtr& pLocalMap) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    LOG(WARNING) << "Failed to open edge keyframe trajectory file: " << filename;
    return;
  }

  for (size_t i = 0; i < pLocalMap->mvKeyFrames.size(); ++i) {
    const auto& kf = pLocalMap->mvKeyFrames[i];
    const Sophus::SE3d& pose = kf->KF_pose_g;
    const Eigen::Matrix4d& T = pose.matrix();

    // Extract translation
    double tx = T(0, 3);
    double ty = T(1, 3);
    double tz = T(2, 3);

    // Extract rotation matrix and convert to quaternion
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Quaterniond q(R);

    // TUM format: timestamp tx ty tz qx qy qz qw
    file << std::fixed << std::setprecision(6) << kf->KF_stamp << " " << tx << " " << ty << " "
         << tz << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
  }

  file.close();
  VLOG(10) << "Saved " << pLocalMap->mvKeyFrames.size() << " edge keyframe poses to: " << filename;
}

}  // namespace edge_viz

