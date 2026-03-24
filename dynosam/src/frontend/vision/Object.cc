/**
 * This file is part of OA-SLAM (ported into dynosam).
 */

#include "dynosam/frontend/vision/Object.hpp"
#include "dynosam/frontend/vision/ColorManager.hpp"
#include "dynosam/frontend/vision/Distance.hpp"
#include "dynosam/backend/edge_map/KeyFrame.hpp"
#include "dynosam/backend/edge_map/localMap.hpp"
#include "dynosam/backend/edge_map/Optimizer.hpp"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <glog/logging.h>
#include <random>
#include <mutex>

#include "dynosam/frontend/vision/OptimizerObjects.h"

namespace dyno 
{
    using Vector9 = Eigen::Matrix<double, 9, 1>;

    // GTSAM factor that penalizes the discrepancy between a projected ellipsoid
    // and a measured 2D ellipse using a 2D Gaussian Wasserstein distance.
    // State layout: [0:3) axes, [3:6) center, [6:9) SO(3) tangent (axis-angle)
    class EllipsoidProjectionFactorGtsam
        : public gtsam::NoiseModelFactor1<Vector9> {
     public:
      EllipsoidProjectionFactorGtsam(const Ellipse& meas,
                                     const Eigen::Matrix<double, 3, 4>& P,
                                     const gtsam::SharedNoiseModel& model,
                                     gtsam::Key key)
          : gtsam::NoiseModelFactor1<Vector9>(model, key),
            meas_(meas),
            P_(P) {}

      gtsam::Vector evaluateError(
          const Vector9& x,
          boost::optional<gtsam::Matrix&> H = boost::none) const override {
        // Compute Jacobian if requested using numerical differentiation
        if (H) {
          *H = gtsam::numericalDerivative11<gtsam::Vector1, Vector9>(
              std::bind(&EllipsoidProjectionFactorGtsam::computeErrorInternal,
                        this, std::placeholders::_1),
              x);
        }
        return computeErrorInternal(x);
      }

     private:
      gtsam::Vector computeErrorInternal(const Vector9& x) const {
        // State layout: [0:3) axes, [3:6) center, [6:9) SO(3) tangent (axis-angle)
        Eigen::Vector3d axes = x.segment<3>(0).cwiseAbs();
        Eigen::Vector3d center = x.segment<3>(3);
        Eigen::Vector3d w = x.segment<3>(6);

        // Check for invalid state
        if (axes.minCoeff() <= 0.0 || !std::isfinite(center.sum()) || !std::isfinite(w.sum())) {
          gtsam::Vector1 e;
          e << 1e6;  // Large error for invalid state
          return e;
        }

        // Soft constraint to prevent ellipsoid axes from exploding.
        const double max_axis_m = 10.0;  // meters
        double axes_penalty = 0.0;
        for (int i = 0; i < 3; ++i) {
          if (!std::isfinite(axes[i])) {
            axes_penalty += 1e6;
            continue;
          }
          if (axes[i] > max_axis_m) {
            const double diff = axes[i] - max_axis_m;
            axes_penalty += 100.0 * diff * diff;
          }
        }

        gtsam::Rot3 R = gtsam::Rot3::Expmap(w);
        Ellipsoid ellipsoid(axes, R.matrix(), center);

        Ellipse proj = ellipsoid.project(P_);
        double d = gaussian_wasserstein_2d(meas_, proj);
        
        // Check for invalid distance (NaN or Inf)
        if (!std::isfinite(d)) {
          d = 1e6;  // Large error for invalid distance
        }
        
        // Add axes penalty to total error
        d += axes_penalty;

        gtsam::Vector1 e;
        e << d;
        return e;
      }

      Ellipse meas_;
      Eigen::Matrix<double, 3, 4> P_;
    };

    // Weak prior to keep the solution near the current ellipsoid parameters.
    // This combats inherent scale/depth ambiguities in pure 2D-ellipse constraints.
    class EllipsoidStatePriorFactorGtsam
        : public gtsam::NoiseModelFactor1<Vector9> {
     public:
      EllipsoidStatePriorFactorGtsam(const Vector9& prior,
                                     const gtsam::SharedNoiseModel& model,
                                     gtsam::Key key)
          : gtsam::NoiseModelFactor1<Vector9>(model, key), prior_(prior) {}

      gtsam::Vector evaluateError(
          const Vector9& x,
          boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (H) {
          *H = gtsam::Matrix::Identity(9, 9);
        }
        // Prior on axes uses abs() to match main factor parameterization
        gtsam::Vector9 e = x - prior_;
        e.segment<3>(0) = x.segment<3>(0).cwiseAbs() - prior_.segment<3>(0).cwiseAbs();
        return e;
      }

     private:
      Vector9 prior_;
    };
    unsigned int Object::factory_id = 0;

    Object::Object(unsigned int cat, const BBox2& bb, const Ellipse ell, double score, std::pair<float, float> depth_data, Eigen::Matrix3d K,
            const Matrix34d& Rt, long unsigned int frame_idx, KeyFrame *kf){
        id_ = factory_id++;
        category_id_ = cat;
        N_ += 1;
        K_ = K;
        last_obs_score_ = score;
        last_obs_frame_id_ = frame_idx;
        flag_optimized = false;
        mbBad = false;

        bboxes_.push_back(bb);
        Rts_.push_back(Rt);

        color_ = RandomUniformColorGenerator::Generate();

        //RECONSTRUCT WITH DEPTH
        float avg_depth = depth_data.first;
        float diff_depth = depth_data.second;

        Eigen::Vector2d bb_center = bbox_center(bb);
        double u = (bb_center(0) - K_(0, 2)) / K_(0, 0);
        double v = (bb_center(1) - K_(1, 2)) / K_(1, 1);
        Eigen::Vector3d bb_center_cam(u*avg_depth, v*avg_depth, avg_depth);
        auto Rcw = Rt.block<3,3>(0,0);
        auto tcw = Rt.col(3);
        Eigen::Vector3d center_world = Rcw.transpose() * bb_center_cam + (-Rcw.transpose() * tcw);
        //ROTATION
        Eigen::Vector3d zc = bb_center_cam / bb_center_cam.norm();
        Eigen::Vector3d up_vec{0, -1, 0};
        Eigen::Vector3d xc = (-up_vec).cross(zc);
        xc = xc / xc.norm();
        Eigen::Vector3d yc = zc.cross(xc);
        Eigen::Matrix3d rot_cam;
        rot_cam.col(0) = xc;
        rot_cam.col(1) = yc;
        rot_cam.col(2) = zc;
        Eigen::Matrix3d rot_world = Rcw.transpose() * rot_cam;
        //AXES
        double width_in_img = bb[2] - bb[0];
        double height_in_img = bb[3] - bb[1];
        double width_in_world = (width_in_img/K_(0, 0))*avg_depth;
        double height_in_world = (height_in_img/K_(1, 1))*avg_depth;
        Eigen::Vector3d axes(0.5*width_in_world, 0.5*height_in_world, 0.5*diff_depth);
        ellipsoid_ = Ellipsoid(axes, rot_world, center_world);
        if(kf){
            mnFirstKFid = kf->KF_ID;
            mnLastKFid = kf->KF_ID;
            ellipses_.push_back(ell);
            observed_kfs.push_back(kf);
            std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
            const auto enc_list = kf->GetEdgeIndicesInBox(bb[0], bb[2], bb[1], bb[3]);

            // Rt is [R_cw | t_cw]; convert camera point -> world point
            const Eigen::Matrix3d Rcw = Rt.block<3,3>(0,0);
            const Eigen::Vector3d tcw = Rt.col(3);
            associated_world_points_.reserve(associated_world_points_.size() + enc_list.size());

            // 1) Depth filtering
            std::vector<double> depths;
            depths.reserve(enc_list.size());
            for (std::size_t enc : enc_list) {
                const int edge_id = static_cast<int>(enc / 100000);
                const int pt_idx  = static_cast<int>(enc % 100000);
                auto itEdge = kf->mmIndexMap.find(edge_id);
                if (itEdge == kf->mmIndexMap.end()) continue;
                const auto& edge = kf->mvEdges[itEdge->second];
                if (pt_idx < 0 || pt_idx >= static_cast<int>(edge.mvPoints.size())) continue;
                const auto& pt = edge.mvPoints[pt_idx];
                if (pt.z_3d > 0.0) depths.push_back(pt.z_3d);
            }

            double mean_depth = 0.0;
            double depth_var  = 0.0;
            if (!depths.empty()) {
                double sum = 0.0;
                for (double d : depths) sum += d;
                mean_depth = sum / depths.size();
                double sq_sum = 0.0;
                for (double d : depths) {
                    double diff = d - mean_depth;
                    sq_sum += diff * diff;
                }
                depth_var = sq_sum / depths.size();
            }
            const double depth_std = (depth_var > 0.0) ? std::sqrt(depth_var) : 0.0;
            const double kSigma = 2.0;
            const double fallback_eps = 0.15;

            // 2) depth 필터를 통과한 포인트만 object와 edge에 할당
            for (std::size_t enc : enc_list) {
                const int edge_id = static_cast<int>(enc / 100000);
                const int pt_idx  = static_cast<int>(enc % 100000);
                auto itEdge = kf->mmIndexMap.find(edge_id);
                if (itEdge == kf->mmIndexMap.end()) continue;
                const int edge_idx = itEdge->second;
                auto& edge = kf->mvEdges[edge_idx];
                if (pt_idx < 0 || pt_idx >= static_cast<int>(edge.mvPoints.size())) continue;
                const auto& pt = edge.mvPoints[pt_idx];

                if (pt.z_3d <= 0.0) continue;
                if (!depths.empty()) {
                    double tol = (depth_std > 1e-3) ? kSigma * depth_std : fallback_eps;
                    if (std::fabs(pt.z_3d - mean_depth) > tol) {
                        continue;
                    }
                }

                // Attach this edge to the current object for visualization:
                // - store object id on the edge
                // - store the object's color on the edge
                // - also update mmEdgeIndex2ObjectId for consistency
                edge.object_id = static_cast<int>(id_);
                edge.color = GetColor();
                kf->mmEdgeIndex2ObjectId[edge_idx] = static_cast<int>(id_);

                // pt.x_3d/y_3d/z_3d are in keyframe camera coordinates
                Eigen::Vector3d pc(pt.x_3d, pt.y_3d, pt.z_3d);
                Eigen::Vector3d pw = Rcw.transpose() * pc + (-Rcw.transpose() * tcw);
                associated_world_points_.push_back(pw);
            }
        }
    }

    void Object::AddDetection(unsigned int cat, const BBox2& bbox, const Ellipse ell, double score, const Matrix34d& Rt, unsigned int frame_idx, dyno::KeyFrame* kf){
        //todo more cat id...
        std::unique_lock<std::mutex> lock(mutex_add_detection_);
        last_obs_frame_id_ = frame_idx;
        last_obs_score_ = score;
        N_ += 1;
        bboxes_.push_back(bbox);
        Rts_.push_back(Rt);

        if(kf){
            mnLastKFid = kf->KF_ID;
            ellipses_.push_back(ell);
            observed_kfs.push_back(kf);
            // Edge-SLAM: 새 관측 bbox 안에 있는 edge 포인트를 다시 한 번 객체에 연결해 준다.
            {
                std::unique_lock<std::mutex> lock_pts(mutex_associated_map_points_);
                const auto enc_list =
                    kf->GetEdgeIndicesInBox(bbox[0], bbox[2], bbox[1], bbox[3]);
                std::vector<std::size_t> enc_list_for_object;
                enc_list_for_object.reserve(enc_list.size());

                // Keep only edges that were already assigned to this object from the
                // instance mask at KeyFrame creation time. This avoids bbox-only
                // reassignment bleeding across object boundaries.
                for (std::size_t enc : enc_list) {
                    const int edge_id = static_cast<int>(enc / 100000);
                    auto itEdge = kf->mmIndexMap.find(edge_id);
                    if (itEdge == kf->mmIndexMap.end()) continue;
                    const int edge_idx = itEdge->second;
                    auto itObj = kf->mmEdgeIndex2ObjectId.find(edge_idx);
                    if (itObj == kf->mmEdgeIndex2ObjectId.end()) continue;
                    if (itObj->second != static_cast<int>(id_)) continue;
                    enc_list_for_object.push_back(enc);
                }

                // Rt is [R_cw | t_cw]; convert camera point -> world point
                const Eigen::Matrix3d Rcw = Rt.block<3,3>(0,0);
                const Eigen::Vector3d tcw = Rt.col(3);
                associated_world_points_.reserve(
                    associated_world_points_.size() + enc_list_for_object.size());

                // 1) bbox 내부 edge 포인트들의 depth 통계 (새 detection 기준)
                std::vector<double> depths;
                depths.reserve(enc_list_for_object.size());
                for (std::size_t enc : enc_list_for_object) {
                    const int edge_id = static_cast<int>(enc / 100000);
                    const int pt_idx  = static_cast<int>(enc % 100000);
                    auto itEdge = kf->mmIndexMap.find(edge_id);
                    if (itEdge == kf->mmIndexMap.end()) continue;
                    const auto& edge = kf->mvEdges[itEdge->second];
                    if (pt_idx < 0 ||
                        pt_idx >= static_cast<int>(edge.mvPoints.size())) continue;
                    const auto& pt = edge.mvPoints[pt_idx];
                    if (pt.z_3d > 0.0) depths.push_back(pt.z_3d);
                }

                double mean_depth = 0.0;
                double depth_var  = 0.0;
                if (!depths.empty()) {
                    double sum = 0.0;
                    for (double d : depths) sum += d;
                    mean_depth = sum / depths.size();
                    double sq_sum = 0.0;
                    for (double d : depths) {
                        double diff = d - mean_depth;
                        sq_sum += diff * diff;
                    }
                    depth_var = sq_sum / depths.size();
                }
                const double depth_std = (depth_var > 0.0) ? std::sqrt(depth_var) : 0.0;
                const double kSigma = 2.0;
                const double fallback_eps = 0.15; // m 단위 허용 편차

                // 2) depth 필터를 통과한 포인트만 object와 edge에 재할당
                for (std::size_t enc : enc_list_for_object) {
                    const int edge_id = static_cast<int>(enc / 100000);
                    const int pt_idx  = static_cast<int>(enc % 100000);
                    auto itEdge = kf->mmIndexMap.find(edge_id);
                    if (itEdge == kf->mmIndexMap.end()) continue;
                    const int edge_idx = itEdge->second;
                    auto& edge = kf->mvEdges[edge_idx];
                    if (pt_idx < 0 ||
                        pt_idx >= static_cast<int>(edge.mvPoints.size())) continue;
                    const auto& pt = edge.mvPoints[pt_idx];

                    if (pt.z_3d <= 0.0) continue;
                    if (!depths.empty()) {
                        double tol = (depth_std > 1e-3) ? kSigma * depth_std : fallback_eps;
                        if (std::fabs(pt.z_3d - mean_depth) > tol) {
                            continue;
                        }
                    }

                    // Attach this edge to the current object for visualization:
                    // - store object id on the edge
                    // - store the object's color on the edge
                    edge.object_id = static_cast<int>(id_);
                    edge.color = GetColor();
                    kf->mmEdgeIndex2ObjectId[edge_idx] = static_cast<int>(id_);

                    // pt.x_3d/y_3d/z_3d are in keyframe camera coordinates
                    Eigen::Vector3d pc(pt.x_3d, pt.y_3d, pt.z_3d);
                    Eigen::Vector3d pw =
                        Rcw.transpose() * pc + (-Rcw.transpose() * tcw);
                    associated_world_points_.push_back(pw);
                }
            }

            // [DISABLED] Ellipsoid-based optimization replaced by edge-projection
            // refinement (RefineWithEdgeProjection) called after sliding-window BA.
            // constexpr std::size_t kEllipsoidOptStride = 5;
            // if (observed_kfs.size() > 2 &&
            //     observed_kfs.size() < 30 &&
            //     (observed_kfs.size() >= last_ellipsoid_opt_observation_count_ + kEllipsoidOptStride)) {
            //     OptimizeReconstructionQuat(true);
            //     flag_optimized = true;
            //     last_ellipsoid_opt_observation_count_ = observed_kfs.size();
            // }
        }
        
    }

    // void Object::OptimizeReconstruction(bool b_random_detections)
    // {
    //     const Ellipsoid& ellipsoid = this->GetEllipsoid();

    //     // std::cout << "===============================> Start ellipsoid optimization " << id_ << std::endl;
    //     typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolver_6_1;
    //     BlockSolver_6_1::LinearSolverType *linear_solver = new g2o::LinearSolverDense<BlockSolver_6_1::PoseMatrixType>();


    //     auto solver = new g2o::OptimizationAlgorithmLevenberg(
    //         new BlockSolver_6_1(linear_solver)
    //     );
    //     g2o::SparseOptimizer optimizer;
    //     optimizer.setAlgorithm(solver);
    //     optimizer.setVerbose(false);


    //     VertexEllipsoidNoRot* vertex = new VertexEllipsoidNoRot();
    //     // VertexEllipsoid* vertex = new VertexEllipsoid();
    //     vertex->setId(0);
    //     Eigen::Matrix<double, 6, 1> e;
    //     e << ellipsoid.GetAxes(), ellipsoid.GetCenter();
    //     vertex->setEstimate(e);
    //     optimizer.addVertex(vertex);

    
    //     std::vector<size_t> chosen_indexes; //We only choose part of the detections for optimization
    //     size_t N = bboxes_.size();
    //     size_t min_opt_N = 20;

    //     for(size_t i=0; i<N; i++){
    //         chosen_indexes.push_back(i);
    //     }

    //     if(b_random_detections && N > min_opt_N){
    //         random_shuffle(chosen_indexes.begin(), chosen_indexes.end());
    //         std::vector<size_t> tmp;
    //         for(size_t i=0; i<min_opt_N; i++){
    //             tmp.push_back(chosen_indexes[i]);
    //         }
    //         chosen_indexes = tmp;
    //     }

    //     auto it_bb = bboxes_.begin();
    //     auto it_Rt = Rts_.begin();
        
    //     for (auto i : chosen_indexes){//size_t i = 0; i < bboxes_.size() && it_bb != bboxes_.end() && it_Rt != Rts_.end(); ++i, ++it_bb, ++it_Rt) {
    //         it_bb = bboxes_.begin() + i;
    //         it_Rt = Rts_.begin() + i;
    //         Eigen::Matrix<double, 3, 4> P = K_ * (*it_Rt);
    //         //auto kf = observed_kfs[i];
    //         //Matrix34d Rt = cvToEigenMatrix<double, float, 3, 4>(kf->GetPose());
    //         //Eigen::Matrix<double, 3, 4> P = K_ * Rt;
    //         EdgeEllipsoidProjection *edge = new EdgeEllipsoidProjection(P, Ellipse::FromBbox(*it_bb), ellipsoid.GetOrientation());
    //         edge->setId(i);
    //         edge->setVertex(0, vertex);
    //         Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
    //         edge->setInformation(information_matrix);
    //         optimizer.addEdge(edge);
    //     }
        
    //     optimizer.initializeOptimization();
    //     optimizer.optimize(8);
    //     Eigen::Matrix<double, 6, 1> ellipsoid_est = vertex->estimate();

    //     Ellipsoid new_ellipsoid(ellipsoid_est.head(3), ellipsoid.GetOrientation(), ellipsoid_est.tail(3));
    //     this->SetEllipsoid(new_ellipsoid);
    // }

    void Object::OptimizeReconstructionQuat(bool b_random_detections)
    {
        const Ellipsoid& ellipsoid = this->GetEllipsoid();

        //std::cout << "===============================> Start ellipsoid optimization quat " << id_ << std::endl;
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 1>> BlockSolver;
        BlockSolver::LinearSolverType *linear_solver = new g2o::LinearSolverDense<BlockSolver::PoseMatrixType>();

        // std::cout << "Optim obj " << obj->GetTrack()->GetId()<< "\n";
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            new BlockSolver(linear_solver)
        );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);


        VertexEllipsoidQuat* vertex = new VertexEllipsoidQuat();
        vertex->setId(0);
        EllipsoidQuat ellipsoid_quat = EllipsoidQuat::FromEllipsoid(ellipsoid);
        vertex->setEstimate(ellipsoid_quat);
        optimizer.addVertex(vertex);

        //We only choose part of the detections for optimization
        std::vector<size_t> chosen_indexes; 
        size_t N = observed_kfs.size();
        size_t min_opt_N = 10;

        for(size_t i=0; i<N; i++){
            chosen_indexes.push_back(i);
        }

        if(b_random_detections && N > min_opt_N){
            random_shuffle(chosen_indexes.begin(), chosen_indexes.end());
            std::vector<size_t> tmp;
            for(size_t i=0; i<min_opt_N; i++){
                tmp.push_back(chosen_indexes[i]);
            }
            chosen_indexes = tmp;
        }

        auto it_ell = ellipses_.begin();

        for (auto i : chosen_indexes){
            it_ell = ellipses_.begin() + i;
            // Use the cached world->camera pose stored at detection time (Rt = [R_cw | t_cw]).
            // This avoids per-edge SE3 matrix extraction + inversion.
            if (i >= Rts_.size()) continue;
            const Matrix34d& Rt = Rts_[i];
            Eigen::Matrix<double, 3, 4> P = K_ * Rt;
            EdgeEllipsoidProjectionQuat *edge =
                new EdgeEllipsoidProjectionQuat(P, *it_ell, ellipsoid.GetOrientation());
            edge->setId(i);
            edge->setVertex(0, vertex);
            Eigen::Matrix<double, 1, 1> information_matrix = Eigen::Matrix<double, 1, 1>::Identity();
            edge->setInformation(information_matrix);
            // // Attach robust kernel to reduce sensitivity to outlier ellipses/projections.
            // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            // rk->setDelta(1.345);
            // edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
        }
        
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        EllipsoidQuat ellipsoid_quat_est = vertex->estimate();
        Ellipsoid new_ellipsoid = ellipsoid_quat_est.ToEllipsoid();
        SetEllipsoid(new_ellipsoid);
    }

    void Object::RefineWithEdgeProjection(
        const std::vector<cv::Point3d>&                     merged_pts_world,
        const std::vector<std::pair<dyno::KeyFrame*, int>>& kf_edge_obs,
        float max_dist_px,
        int   max_iter)
    {
        if (merged_pts_world.empty() || kf_edge_obs.empty()) return;

        // Current ellipsoid center (copy to avoid holding lock during iterations)
        Ellipsoid curr = GetEllipsoid();
        Eigen::Vector3d center = curr.GetCenter();
        if (!center.allFinite()) return;

        // Convert 3D points and filter NaN
        std::vector<Eigen::Vector3d,
                    Eigen::aligned_allocator<Eigen::Vector3d>> pts;
        pts.reserve(merged_pts_world.size());
        for (const auto& p : merged_pts_world) {
            Eigen::Vector3d ep(p.x, p.y, p.z);
            if (ep.allFinite()) pts.push_back(ep);
        }
        if (pts.empty()) return;

        // Relative positions fixed: r_i = pt_3d_i - center_0
        // When center moves to c: pt_3d_i(c) = c + r_i
        std::vector<Eigen::Vector3d,
                    Eigen::aligned_allocator<Eigen::Vector3d>> rel_pts;
        rel_pts.reserve(pts.size());
        for (const auto& p : pts) rel_pts.push_back(p - center);

        const double max_dist2 = static_cast<double>(max_dist_px * max_dist_px);

        // ── Gauss-Newton (Levenberg-Marquardt damping) ────────────────────────
        for (int iter = 0; iter < max_iter; ++iter) {
            Eigen::Matrix3d JtJ = Eigen::Matrix3d::Zero();
            Eigen::Vector3d Jtr = Eigen::Vector3d::Zero();
            int n_inliers = 0;

            for (const auto& [kf_ptr, edge_idx] : kf_edge_obs) {
                if (!kf_ptr) continue;
                if (edge_idx < 0 ||
                    edge_idx >= static_cast<int>(kf_ptr->mvEdges.size())) continue;
                const Edge& edge = kf_ptr->mvEdges[edge_idx];
                if (edge.mvPoints.empty()) continue;

                // World-to-camera transform from Sophus SE3d
                Sophus::SE3d T_cw = kf_ptr->KF_pose_g.inverse();
                const Eigen::Matrix3d R_cw = T_cw.rotationMatrix();
                const Eigen::Vector3d t_cw = T_cw.translation();
                const double fx = kf_ptr->mFx, fy = kf_ptr->mFy;
                const double cx = kf_ptr->mCx, cy = kf_ptr->mCy;

                for (const auto& r_i : rel_pts) {
                    // 3D point in world frame with current center
                    const Eigen::Vector3d pt_w = center + r_i;
                    // 3D point in camera frame
                    const Eigen::Vector3d pt_c = R_cw * pt_w + t_cw;
                    if (pt_c.z() < 0.1) continue;

                    const double iz  = 1.0 / pt_c.z();
                    const double iz2 = iz * iz;
                    const double u_proj = fx * pt_c.x() * iz + cx;
                    const double v_proj = fy * pt_c.y() * iz + cy;

                    // Nearest observed 2D edge point (inlier gate)
                    double min_d2 = max_dist2;
                    double obs_u = -1.0, obs_v = -1.0;
                    for (const auto& ep : edge.mvPoints) {
                        const double du = ep.x - u_proj;
                        const double dv = ep.y - v_proj;
                        const double d2 = du * du + dv * dv;
                        if (d2 < min_d2) {
                            min_d2 = d2;
                            obs_u  = ep.x;
                            obs_v  = ep.y;
                        }
                    }
                    if (obs_u < 0.0) continue;  // no inlier found

                    // Reprojection residual [u_proj - obs_u, v_proj - obs_v]
                    const Eigen::Vector2d res(u_proj - obs_u, v_proj - obs_v);

                    // Jacobian:  d(proj)/d(center) = J_proj(2×3) * R_cw(3×3)
                    //   J_proj = [[fx/z,    0,  -fx*X/z²],
                    //             [0,    fy/z,  -fy*Y/z²]]
                    Eigen::Matrix<double, 2, 3> J_proj;
                    J_proj << fx * iz,     0.0,    -fx * pt_c.x() * iz2,
                              0.0,         fy * iz, -fy * pt_c.y() * iz2;
                    const Eigen::Matrix<double, 2, 3> J = J_proj * R_cw;

                    JtJ += J.transpose() * J;
                    Jtr += J.transpose() * res;
                    ++n_inliers;
                }
            }

            if (n_inliers < 10) break;

            // Levenberg-Marquardt damping for numerical stability
            const double lambda = 1e-4 * JtJ.trace() / 3.0;
            JtJ.diagonal().array() += lambda;

            const Eigen::Vector3d delta = JtJ.ldlt().solve(-Jtr);
            if (!delta.allFinite() || delta.norm() > 2.0) break;

            center += delta;
            if (delta.norm() < 1e-4) break;
        }

        if (!center.allFinite()) return;

        // Keep axes + orientation, update center only
        SetEllipsoid(Ellipsoid(curr.GetAxes(), curr.GetOrientation(), center));

        VLOG(3) << "[RefineWithEdgeProjection] obj_id=" << id_
                << " center=[" << center.transpose() << "]";
    }

    void Object::OptimizeWithEdgePipeline(
        const std::vector<std::shared_ptr<dyno::KeyFrame>>& kfs)
    {
        const int obj_id = static_cast<int>(id_);
        if (kfs.size() < 2) return;

        // ── 0. Incremental sliding window ────────────────────────────────────
        // Only process KFs newer than last_anchored_kf_id_.
        // We also keep a small overlap (kOverlap) with the previous anchor
        // window so that new frames can associate to existing clusters.
        constexpr int kObjWindowSize = 15;  // max new KFs to process at once
        constexpr int kOverlap       = 3;   // # of old KFs to carry as context

        const int last_anchor_kf = GetLastAnchoredKFId();

        // Collect KFs: kOverlap old ones (for association context) + new ones
        std::vector<std::shared_ptr<dyno::KeyFrame>> src_kfs;
        src_kfs.reserve(kObjWindowSize + kOverlap);
        int new_kfs_count = 0;
        for (const auto& kf : kfs) {
            if (!kf) continue;
            if (kf->KF_ID <= last_anchor_kf) {
                // Only keep the most recent kOverlap old KFs as context
                src_kfs.push_back(kf);
                if (static_cast<int>(src_kfs.size()) > kOverlap) {
                    src_kfs.erase(src_kfs.begin());
                }
            } else {
                src_kfs.push_back(kf);
                ++new_kfs_count;
                if (new_kfs_count >= kObjWindowSize) break;
            }
        }
        // Need at least 2 KFs to do anything meaningful
        if (static_cast<int>(src_kfs.size()) < 2) return;
        // If no genuinely new KFs, nothing to update
        if (new_kfs_count == 0) return;

        // ── 1. Build per-object KeyFrames containing only this object's edges ──
        // For each sliding-window KF, create a lightweight copy that keeps only
        // the edges associated with this object.
        std::vector<std::shared_ptr<dyno::KeyFrame>> obj_kfs;
        obj_kfs.reserve(src_kfs.size());

        for (const auto& kf : src_kfs) {
            if (!kf) continue;

            // Collect edge indices belonging to this object
            std::vector<int> obj_edge_indices;
            for (int i = 0; i < static_cast<int>(kf->mvEdges.size()); ++i) {
                auto it = kf->mmEdgeIndex2ObjectId.find(i);
                if (it != kf->mmEdgeIndex2ObjectId.end() && it->second == obj_id) {
                    obj_edge_indices.push_back(i);
                } else if (kf->mvEdges[i].object_id == obj_id) {
                    obj_edge_indices.push_back(i);
                }
            }
            if (obj_edge_indices.empty()) continue;

            // Create a new KeyFrame with only this object's edges
            auto obj_kf = std::make_shared<dyno::KeyFrame>();
            obj_kf->KF_ID     = kf->KF_ID;
            obj_kf->KF_stamp  = kf->KF_stamp;
            obj_kf->KF_pose_g = kf->KF_pose_g;
            obj_kf->mFx = kf->mFx; obj_kf->mFy = kf->mFy;
            obj_kf->mCx = kf->mCx; obj_kf->mCy = kf->mCy;
            obj_kf->mWidth  = kf->mWidth;
            obj_kf->mHeight = kf->mHeight;
            obj_kf->mMatGray = kf->mMatGray;  // shallow copy OK (read-only)

            obj_kf->mvEdges.reserve(obj_edge_indices.size());
            for (int idx : obj_edge_indices) {
                obj_kf->mvEdges.push_back(kf->mvEdges[idx]);
                int new_idx = static_cast<int>(obj_kf->mvEdges.size()) - 1;
                int edge_id = kf->mvEdges[idx].edge_ID;
                obj_kf->mmIndexMap[edge_id] = new_idx;
                obj_kf->mmEdgeIndex2ObjectId[new_idx] = obj_id;
            }

            // Rebuild grid from scratch using only this object's edges.
            // The original kf->mGrid contains encoded entries for ALL edges,
            // but per-object KF's mmIndexMap only has this object's edges,
            // so GetEdgeIndicesInBox would fail to look up most entries.
            {
                constexpr int GRID_COLS = 64;
                constexpr int GRID_ROWS = 48;
                obj_kf->mGrid.clear();
                obj_kf->mGrid.resize(GRID_COLS);
                for (int gi = 0; gi < GRID_COLS; ++gi)
                    obj_kf->mGrid[gi].resize(GRID_ROWS);

                const float invW = (obj_kf->mWidth  > 0) ?
                    static_cast<float>(GRID_COLS) / static_cast<float>(obj_kf->mWidth)  : 1.0f;
                const float invH = (obj_kf->mHeight > 0) ?
                    static_cast<float>(GRID_ROWS) / static_cast<float>(obj_kf->mHeight) : 1.0f;

                for (size_t ei = 0; ei < obj_kf->mvEdges.size(); ++ei) {
                    const auto& edge = obj_kf->mvEdges[ei];
                    const int edge_id = edge.edge_ID;
                    for (int pi = 0; pi < static_cast<int>(edge.mvPoints.size()); ++pi) {
                        const auto& pt = edge.mvPoints[pi];
                        int gx = static_cast<int>(pt.x * invW);
                        int gy = static_cast<int>(pt.y * invH);
                        if (gx < 0) gx = 0; if (gx >= GRID_COLS) gx = GRID_COLS - 1;
                        if (gy < 0) gy = 0; if (gy >= GRID_ROWS) gy = GRID_ROWS - 1;
                        std::size_t enc = static_cast<std::size_t>(edge_id) * 100000 +
                                          static_cast<std::size_t>(pi);
                        obj_kf->mGrid[gx][gy].push_back(enc);
                    }
                }
            }

            // Also rebuild mMatSearch (used by edgeWiseCorrespondenceLocalMapping)
            obj_kf->constructSearchPlainParallel();

            obj_kfs.push_back(obj_kf);
        }

        if (obj_kfs.size() < 2) return;

        // ── 2. Run the standard edge pipeline on the per-object local map ──────
        auto obj_local_map = std::make_shared<dyno::localMap>();

        for (const auto& okf : obj_kfs) {
            obj_local_map->addFrame2LocalMap(okf);
        }

        if (obj_local_map->msState != dyno::localMap::State::INITIALIZED) return;
        if (obj_local_map->mvEleEdgeClusters.empty()) return;

        // Cluster fitting (merge multi-frame edge observations)
        // Use a lower minimum-element threshold for per-object maps
        // (fewer edges per object → smaller clusters are still valid)
        obj_local_map->clusterFittingProjection(/*min_elements=*/2);

        // Pose optimization (refine KF poses using merged edges)
        dyno::Optimizer::optimizeAllInvolvedKFs(obj_local_map);

        // ── 3. Collect optimized 3D edge points + KF observations ──────────────
        std::vector<cv::Point3d> all_merged_pts;
        std::vector<std::pair<dyno::KeyFrame*, int>> all_kf_obs;
        std::vector<std::vector<cv::Point3d>> per_cluster_clouds;

        for (const auto& cluster : obj_local_map->mvEleEdgeClusters) {
            if (!cluster.mbMerged || cluster.mvMergedCloud_ref.empty()) continue;

            per_cluster_clouds.push_back(cluster.mvMergedCloud_ref);

            all_merged_pts.insert(all_merged_pts.end(),
                                  cluster.mvMergedCloud_ref.begin(),
                                  cluster.mvMergedCloud_ref.end());

            for (unsigned int ele_id : cluster.mvElementEdgeIDs) {
                auto it_ele = obj_local_map->mmElementID2index.find(ele_id);
                if (it_ele == obj_local_map->mmElementID2index.end()) continue;
                const elementEdge& ele = obj_local_map->mvElementEdges[it_ele->second];
                auto it_kf = obj_local_map->mmKFID2KFindex.find(ele.kf_id);
                if (it_kf == obj_local_map->mmKFID2KFindex.end()) continue;
                dyno::KeyFrame* kf_ptr = obj_local_map->mvKeyFrames[it_kf->second].get();
                if (!kf_ptr) continue;
                all_kf_obs.emplace_back(kf_ptr, ele.kf_edge_idx);
            }
        }

        // ── 4. Refine ellipsoid center via reprojection ────────────────────────
        if (!all_merged_pts.empty() && !all_kf_obs.empty()) {
            RefineWithEdgeProjection(all_merged_pts, all_kf_obs);
        }

        // ── 5. Stitch new clusters into anchor_clusters_ ─────────────────────
        // For each new merged cluster, check if it is 3D-close to an existing
        // anchor cluster.  If so, replace/update the anchor with the newer
        // (more recently optimized) cloud; otherwise, add as a new anchor.
        constexpr double kStitchDistTh = 0.3;  // metres – tune to scene scale

        auto centroid = [](const std::vector<cv::Point3d>& pts) -> cv::Point3d {
            cv::Point3d c(0, 0, 0);
            for (const auto& p : pts) { c.x += p.x; c.y += p.y; c.z += p.z; }
            const double n = static_cast<double>(pts.size());
            return { c.x / n, c.y / n, c.z / n };
        };
        auto dist3d = [](const cv::Point3d& a, const cv::Point3d& b) -> double {
            double dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
            return std::sqrt(dx*dx + dy*dy + dz*dz);
        };

        // Robust principal direction via trimmed PCA of the point cloud.
        // Removes far outliers from centroid before covariance estimation.
        // Returns false if the cloud is not line-like enough.
        auto principalDirRobust = [&](const std::vector<cv::Point3d>& pts,
                                      Eigen::Vector3d* dir_out,
                                      double* linearity_out) -> bool {
            if (!dir_out || !linearity_out) return false;
            *dir_out = Eigen::Vector3d(1, 0, 0);
            *linearity_out = 0.0;
            if (pts.size() < 6) return false;

            constexpr size_t kMaxSamples = 200;
            constexpr double kTrimTopRatio = 0.15;   // trim top 15% farthest points
            constexpr size_t kMinInliers = 12;
            constexpr double kMinLinearity = 0.55;   // (lambda1-lambda2)/lambda1

            const size_t step = std::max<size_t>(1, pts.size() / kMaxSamples);
            std::vector<Eigen::Vector3d> samples;
            samples.reserve((pts.size() + step - 1) / step);
            for (size_t i = 0; i < pts.size(); i += step) {
                samples.emplace_back(pts[i].x, pts[i].y, pts[i].z);
            }
            if (samples.size() < kMinInliers) return false;

            Eigen::Vector3d c = Eigen::Vector3d::Zero();
            for (const auto& p : samples) c += p;
            c /= static_cast<double>(samples.size());

            std::vector<double> dist2;
            dist2.reserve(samples.size());
            for (const auto& p : samples) dist2.push_back((p - c).squaredNorm());

            const size_t keep_n =
                std::max(kMinInliers,
                         static_cast<size_t>(std::floor((1.0 - kTrimTopRatio) *
                                                        static_cast<double>(samples.size()))));
            if (keep_n > dist2.size()) return false;
            std::nth_element(dist2.begin(), dist2.begin() + (keep_n - 1), dist2.end());
            const double d2_th = dist2[keep_n - 1];

            std::vector<Eigen::Vector3d> inliers;
            inliers.reserve(keep_n);
            for (const auto& p : samples) {
                if ((p - c).squaredNorm() <= d2_th) inliers.push_back(p);
            }
            if (inliers.size() < kMinInliers) return false;

            Eigen::Vector3d c_in = Eigen::Vector3d::Zero();
            for (const auto& p : inliers) c_in += p;
            c_in /= static_cast<double>(inliers.size());

            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (const auto& p : inliers) {
                const Eigen::Vector3d v = p - c_in;
                cov += v * v.transpose();
            }
            cov /= static_cast<double>(inliers.size());

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
            if (solver.info() != Eigen::Success) return false;
            const auto evals = solver.eigenvalues();  // ascending
            const double lambda1 = std::max(1e-12, evals(2));
            const double lambda2 = std::max(0.0, evals(1));
            const double linearity = (lambda1 - lambda2) / lambda1;
            *linearity_out = linearity;
            if (linearity < kMinLinearity) return false;

            Eigen::Vector3d dir = solver.eigenvectors().col(2);
            if (!dir.allFinite() || dir.norm() < 1e-12) return false;
            dir.normalize();
            *dir_out = dir;
            return true;
        };

        // Get current anchor list (thread-safe copy)
        auto anchors = GetAnchorClusters();

        // Track the latest KF_ID in this batch
        int max_kf_id = last_anchor_kf;
        for (const auto& okf : obj_kfs) {
            if (okf && okf->KF_ID > max_kf_id) max_kf_id = okf->KF_ID;
        }

        for (const auto& cloud : per_cluster_clouds) {
            if (cloud.empty()) continue;
            const cv::Point3d c_new = centroid(cloud);
            Eigen::Vector3d d_new;
            double linearity_new = 0.0;
            if (!principalDirRobust(cloud, &d_new, &linearity_new)) {
                // Keep non-line-like clusters as new anchors instead of forcing
                // unstable angle-based matching.
                Object::AnchorCluster ac;
                ac.pts = cloud;
                ac.last_updated_kf_id = max_kf_id;
                anchors.push_back(std::move(ac));
                continue;
            }

            int best_idx = -1;
            double best_cost = std::numeric_limits<double>::infinity();
            constexpr float kAngleSimTh = 0.70f;     // abs(dot) threshold
            constexpr double kAngleWeight = 0.25;   // balance dist & angle (unitless)
            for (int ai = 0; ai < static_cast<int>(anchors.size()); ++ai) {
                if (anchors[ai].pts.empty()) continue;
                const cv::Point3d c_a = centroid(anchors[ai].pts);
                Eigen::Vector3d d_a;
                double linearity_a = 0.0;
                if (!principalDirRobust(anchors[ai].pts, &d_a, &linearity_a)) {
                    continue;
                }

                const double d = dist3d(c_new, c_a);
                const double dist_norm = d / kStitchDistTh;

                // Eigenvector sign is ambiguous; use abs(dot).
                const double angle_sim = std::abs(d_new.dot(d_a));  // [0,1]
                if (angle_sim < kAngleSimTh) continue;

                // Lower is better. dist_norm is >= 0.
                const double cost = dist_norm - kAngleWeight * angle_sim;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_idx = ai;
                }
            }

            if (best_idx >= 0) {
                // Update existing anchor with the freshly optimized cloud
                anchors[best_idx].pts = cloud;
                anchors[best_idx].last_updated_kf_id = max_kf_id;
            } else {
                // New anchor cluster
                Object::AnchorCluster ac;
                ac.pts = cloud;
                ac.last_updated_kf_id = max_kf_id;
                anchors.push_back(std::move(ac));
            }
        }

        SetAnchorClusters(anchors);
        SetLastAnchoredKFId(max_kf_id);

        // ── 6. Visualization: current window's merged clusters only ──────────
        // Use per_cluster_clouds (freshly optimized) for visualization so that
        // we show clean, recent results — not potentially drifted old anchors.
        if (!per_cluster_clouds.empty()) {
            SetMergedEdgeClusters(per_cluster_clouds);
        }

        // ── 7. Update associated_world_points_ from ALL anchor clusters ──────
        // Use all anchors so edge-projection association covers the full object
        // surface even when parts are not visible in the current window.
        std::vector<cv::Point3d> anchor_world_pts;
        for (const auto& ac : anchors) {
            for (const auto& p : ac.pts) anchor_world_pts.push_back(p);
        }
        if (!anchor_world_pts.empty()) {
            //TODO: test this
            // Guard against catastrophic shrink after a noisy optimization step.
            // If the new anchor-derived set collapses too much, keep previous
            // associated points so frame-to-frame association remains stable.
            constexpr size_t kMinKeepPts = 200;
            constexpr double kMinKeepRatio = 0.20;  // keep old if new < 20% of old

            std::lock_guard<std::mutex> lk(mutex_associated_map_points_);
            const size_t old_n = associated_world_points_.size();
            const size_t new_n = anchor_world_pts.size();

            bool accept_new = true;
            if (old_n >= kMinKeepPts) {
                const double ratio = static_cast<double>(new_n) /
                                     static_cast<double>(std::max<size_t>(1, old_n));
                if (new_n < kMinKeepPts && ratio < kMinKeepRatio) {
                    accept_new = false;
                }
            }

            if (accept_new) {
                associated_world_points_.clear();
                associated_world_points_.reserve(anchor_world_pts.size());
                for (const auto& pt : anchor_world_pts) {
                    associated_world_points_.emplace_back(pt.x, pt.y, pt.z);
                }
            }
        }

        // Debug log disabled to avoid high-frequency spam; enable with LOG/VLOG
        // if needed during tuning.
        // LOG(INFO) << "[OptimizeWithEdgePipeline] obj_id=" << obj_id
        //           << " new_kfs=" << new_kfs_count
        //           << " obj_kfs=" << obj_kfs.size()
        //           << " clusters=" << obj_local_map->mvEleEdgeClusters.size()
        //           << " anchors=" << anchors.size()
        //           << " merged_pts=" << all_merged_pts.size();
    }


}
