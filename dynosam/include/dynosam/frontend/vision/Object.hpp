/**
* This file is part of OA-SLAM.
*
* Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
* (Inria, LORIA, Université de Lorraine)
* OA-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OA-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OA-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef OBJECT_H
#define OBJECT_H

#include "dynosam/frontend/vision/BboxUtile.hpp"

#include <random>
#include <memory>
#include <list>
#include <iostream>

#include <algorithm>

#include <Eigen/Dense>


#include "dynosam_common/Ellipse.hpp"
#include "dynosam_common/Ellipsoid.hpp"
#include "dynosam_common/Edge.hpp"

// Forward declaration
namespace dyno {
class KeyFrame;
}

namespace dyno {
    
class Object
{
    public:

        static unsigned int factory_id;

        Object(unsigned int cat, const BBox2& bbox, const Ellipse ell, double score, std::pair<float, float> depth_data, Eigen::Matrix3d K, 
              const Matrix34d& Rt, long unsigned int frame_idx, dyno::KeyFrame *kf);

        Object(const Ellipsoid& ellipsoid) : ellipsoid_(ellipsoid){
            id_ = 0;
        }

        void AddDetection(unsigned int cat, const BBox2& bbox, const Ellipse ell, double score, const Matrix34d& Rt, unsigned int frame_idx, dyno::KeyFrame* kf);

        // Returns a VALUE COPY of the ellipsoid (mutex is released after copy).
        // Do NOT store as const-reference; use Ellipsoid e = GetEllipsoid().
        Ellipsoid GetEllipsoid() const {
            std::unique_lock<std::mutex> lock(mutex_ellipsoid_);
            return ellipsoid_;
        }

        void SetEllipsoid(const Ellipsoid& ell) {
            std::unique_lock<std::mutex> lock(mutex_ellipsoid_);
            ellipsoid_ = ell;
        }

        unsigned int GetId() const {
            return id_;
        }

        void SetId(unsigned int id) {
            //std::cout<<"mapobj:"<<id_<<"set id to:"<<id<<std::endl;
            id_ = id;
        }

        size_t GetNbObservations() const {
            return N_;
        }

        bool operator<(const Object* rhs) const{
            return this->id_ < rhs->GetId();
        }

        cv::Scalar GetColor() const {
            return color_;
        }

        unsigned int GetCategoryId() const {
            return category_id_;
        }

        size_t GetLastObsFrameId() const {
            return last_obs_frame_id_;
        }

        bool GetFlagOptimized() const{
            return flag_optimized;
        }

        // Edge-SLAM: associated points are stored as world 3D points (from edge points).
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> GetAssociatedMapPoints() const {
            std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
            return associated_world_points_;
        }

        // std::vector<MapPoint*> GetFilteredAssociatedMapPoints(int threshold);

        // void InsertNewAsscoatedMapPoint(MapPoint* mp){
        //     std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
        //     associated_map_points_.insert(mp);
        // }

        // void OptimizeReconstruction(bool b_random_detections);

        void OptimizeReconstructionQuat(bool b_random_detections);

        /**
         * @brief Build per-object local map from associated edges and run
         *        clusterFittingProjection + pose optimization.
         *
         * Called from the processing thread (via tbb::parallel_for_each) after
         * the global sliding-window BA.  For each KF that observes this object,
         * edges with matching object_id are collected into a per-object localMap,
         * then the standard association → cluster → fitting → BA pipeline runs
         * independently.
         *
         * Thread-safe: creates its own localMap instance (no shared state).
         *
         * @param kfs  KeyFrames from the current sliding window
         */
        void OptimizeWithEdgePipeline(const std::vector<std::shared_ptr<dyno::KeyFrame>>& kfs);

        /**
         * @brief Refine the ellipsoid CENTER using 3D→2D edge reprojection (Gauss-Newton).
         *
         * For each (cluster 3D point, observing KF) pair:
         *   - Project the 3D cluster point into the KF image
         *   - Find the nearest observed 2D edge point in that KF
         *   - Accumulate the reprojection Jacobian and residual
         * Then solve the normal equations to update the ellipsoid center.
         * Axes and orientation are kept unchanged; only the center moves.
         *
         * @param merged_pts_world  BA-optimized 3D edge points (world frame,
         *                          from cluster.mvMergedCloud_ref)
         * @param kf_edge_obs       Pairs of (KeyFrame*, edge_idx in kf->mvEdges)
         *                          from the cluster's elementEdges
         * @param max_dist_px       Inlier threshold in pixels
         * @param max_iter          Maximum Gauss-Newton iterations
         */
        void RefineWithEdgeProjection(
            const std::vector<cv::Point3d>&                     merged_pts_world,
            const std::vector<std::pair<dyno::KeyFrame*, int>>& kf_edge_obs,
            float max_dist_px = 5.0f,
            int   max_iter    = 5);

        int GetObservationNumber(){
            return observed_kfs.size();
        }

        std::vector<dyno::KeyFrame*> GetObservations(){
            std::unique_lock<std::mutex> lock(mutex_add_detection_);
            return observed_kfs;
        }

        // Returns a copy of all observed bounding boxes (one per AddDetection call).
        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> GetObservedBboxes() const {
            std::unique_lock<std::mutex> lock(mutex_add_detection_);
            return bboxes_;
        }

        // Edge-SLAM: return associated WORLD points that lie inside the ellipsoid.
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        GetFilteredAssociatedMapPoints(int threshold);

        void SetBadFlag(){
            mbBad = true;
        }

        bool isBad(){
            return mbBad;
        }

        // Per-object merged edge clusters from OptimizeWithEdgePipeline.
        // Each inner vector is one cluster's polyline (world frame).
        void SetMergedEdgeClusters(std::vector<std::vector<cv::Point3d>> clusters) {
            std::lock_guard<std::mutex> lk(mutex_merged_edge_clusters_);
            merged_edge_clusters_ = std::move(clusters);
        }
        std::vector<std::vector<cv::Point3d>> GetMergedEdgeClusters() const {
            std::lock_guard<std::mutex> lk(mutex_merged_edge_clusters_);
            return merged_edge_clusters_;
        }

        // ── Anchor clusters (hierarchical edge representation) ─────────────
        // Anchor clusters are optimized 3D edge polylines from previous
        // optimization windows.  They serve as stable "super elementEdge"
        // references for new frames, limiting drift accumulation.
        struct AnchorCluster {
            std::vector<cv::Point3d> pts;  // world-frame ordered polyline
            int last_updated_kf_id = -1;   // KF_ID of the most recent update
        };

        void SetAnchorClusters(std::vector<AnchorCluster> anchors) {
            std::lock_guard<std::mutex> lk(mutex_merged_edge_clusters_);
            anchor_clusters_ = std::move(anchors);
        }
        std::vector<AnchorCluster> GetAnchorClusters() const {
            std::lock_guard<std::mutex> lk(mutex_merged_edge_clusters_);
            return anchor_clusters_;
        }
        // KF_ID of the last KF that was processed into anchor clusters.
        int GetLastAnchoredKFId() const {
            std::lock_guard<std::mutex> lk(mutex_merged_edge_clusters_);
            return last_anchored_kf_id_;
        }
        void SetLastAnchoredKFId(int id) {
            std::lock_guard<std::mutex> lk(mutex_merged_edge_clusters_);
            last_anchored_kf_id_ = id;
        }

        

    public:
        long unsigned int last_obs_frame_id_ = -1;
        std::pair<std::pair<long unsigned int, int>, double> last_obs_ids_and_max_iou; // for multiple match

        long unsigned int  mnFirstKFid = -1;
        long unsigned int  mnLastKFid = -1;

    protected:
        unsigned int category_id_;
        unsigned int id_;
        double last_obs_score_ = 0.0;
        cv::Scalar color_;
        Eigen::Matrix3d K_;
        bool flag_optimized;
        bool mbBad;
        size_t N_ = 0;

        std::vector<BBox2, Eigen::aligned_allocator<BBox2>> bboxes_ = std::vector<BBox2, Eigen::aligned_allocator<BBox2>>();
        std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>> Rts_ = std::vector<Matrix34d, Eigen::aligned_allocator<Matrix34d>>();

        std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>> ellipses_ = std::vector<Ellipse, Eigen::aligned_allocator<Ellipse>>();

        std::vector<dyno::KeyFrame*> observed_kfs;

        // Throttle expensive ellipsoid refinement: track last observation count we optimized at.
        // (Optimization is triggered from AddDetection.)
        std::size_t last_ellipsoid_opt_observation_count_{0};

        Ellipsoid ellipsoid_;

        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> associated_world_points_;

        // Edge cluster IDs associated with this object (from KeyFrames)
        // Maps KeyFrame ID -> (edge_id -> cluster_id)
        // This maintains edge cluster information across sliding window updates
        // Each edge in a KeyFrame can have a different cluster_id
        std::map<int, std::map<int, unsigned int>> associated_edge_cluster_ids_;

        mutable std::mutex mutex_ellipsoid_;
        mutable std::mutex mutex_associated_map_points_;
        mutable std::mutex mutex_add_detection_;
        mutable std::mutex mutex_merged_edge_clusters_;

        std::vector<std::vector<cv::Point3d>> merged_edge_clusters_;
        std::vector<AnchorCluster> anchor_clusters_;
        int last_anchored_kf_id_ = -1;

        Object() = delete;
};


}

#endif //  OBJECT_H
