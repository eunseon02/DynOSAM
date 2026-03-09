/**
 * This file is part of OA-SLAM (ported into dynosam).
 */

#include "dynosam/frontend/vision/Object.hpp"
#include "dynosam/frontend/vision/ColorManager.hpp"
#include "dynosam/frontend/vision/Distance.hpp"
#include "dynosam/backend/edge_map/KeyFrame.hpp"

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

            for (std::size_t enc : enc_list) {
                const int edge_id = static_cast<int>(enc / 100000);
                const int pt_idx  = static_cast<int>(enc % 100000);
                auto itEdge = kf->mmIndexMap.find(edge_id);
                if (itEdge == kf->mmIndexMap.end()) continue;
                auto& edge = kf->mvEdges[itEdge->second];
                if (pt_idx < 0 || pt_idx >= static_cast<int>(edge.mvPoints.size())) continue;
                const auto& pt = edge.mvPoints[pt_idx];

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
            // std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
            // auto vIndices_in_box = kf->GetFeaturesInBox(bbox[0], bbox[2], bbox[1], bbox[3]);
            // for(auto i : vIndices_in_box){
            //     MapPoint* mp = kf->mvpMapPoints[i];
            //     if (mp) {
            //         associated_map_points_.insert(mp);
            //     }
            // }

            // Trigger local ellipsoid refinement once we have enough observations,
            // similar to the original OA-SLAM behavior.
            if (observed_kfs.size() > 2 && observed_kfs.size() < 30) {
                VLOG(1) << "[Object::AddDetection] Triggering ellipsoid optimization for object_id=" 
                        << id_ << ", observed_kfs=" << observed_kfs.size();
                OptimizeReconstructionQuat(true);
                flag_optimized = true;
            }
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
        
        // Log initial ellipsoid state
        Eigen::Vector3d axes_init = ellipsoid.GetAxes();
        Eigen::Vector3d center_init = ellipsoid.GetCenter();
        // VLOG(1) << "[Object::OptimizeReconstructionQuat] Starting optimization for object_id=" << id_
        //         << ", initial axes=[" << axes_init.transpose() << "]"
        //         << ", initial center=[" << center_init.transpose() << "]"
        //         << ", observed_kfs=" << observed_kfs.size();

        // Build initial 9D state: [axes (3), center (3), so3 tangent (3)]
        Vector9 x0;
        x0.segment<3>(0) = ellipsoid.GetAxes();
        x0.segment<3>(3) = ellipsoid.GetCenter();
        gtsam::Rot3 R0(ellipsoid.GetOrientation());
        auto axis_angle_pair = R0.axisAngle();
        Eigen::Vector3d axis = axis_angle_pair.first.unitVector();
        double angle = axis_angle_pair.second;
        x0.segment<3>(6) = axis * angle;

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values values;

        gtsam::Key key = gtsam::Symbol('E', id_);
        values.insert<Vector9>(key, x0);

        // Noise model: use robust noise model similar to g2o's Huber kernel
        // g2o version uses Identity information matrix with Huber robust kernel
        // GTSAM equivalent: use Huber m-estimator with robust noise model
        auto base_noise = gtsam::noiseModel::Isotropic::Sigma(1, 1.0);
        auto robust_noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(1.345),  // Huber threshold (same as g2o default)
            base_noise
        );

        // Add a weak prior to prevent runaway solutions (scale/depth ambiguity).
        // Sigmas are chosen to be permissive but still stabilizing.
        gtsam::Vector9 prior_sigmas;
        prior_sigmas.segment<3>(0) = (axes_init.cwiseAbs() * 0.5).cwiseMax(0.05);  // axes
        prior_sigmas.segment<3>(3).setConstant(1.0);                               // center (m)
        prior_sigmas.segment<3>(6).setConstant(0.5);                               // rotation tangent (rad)
        auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
        graph.emplace_shared<EllipsoidStatePriorFactorGtsam>(x0, prior_noise, key);

        // Choose subset of detections (same logic as original code)
        std::vector<size_t> chosen_indexes;
        size_t N = observed_kfs.size();
        size_t min_opt_N = 10;

        for (size_t i = 0; i < N; ++i) {
            chosen_indexes.push_back(i);
        }
        if (b_random_detections && N > min_opt_N) {
            std::shuffle(chosen_indexes.begin(), chosen_indexes.end(),
                         std::mt19937{std::random_device{}()});
            chosen_indexes.resize(min_opt_N);
        }

        // Inlier gating (match voom's behavior):
        // only keep measurements where the current ellipsoid projection has reasonable IoU
        // and the ellipsoid is in front of the camera (z>0).
        std::vector<size_t> inlier_indexes;
        inlier_indexes.reserve(chosen_indexes.size());
        double sum_iou_before = 0.0;

        const Eigen::Vector3d center_before = ellipsoid.GetCenter();
        const Eigen::Vector4d center_before_h(center_before[0], center_before[1], center_before[2], 1.0);

        for (auto i : chosen_indexes) {
            if (i >= observed_kfs.size() || i >= ellipses_.size() || i >= bboxes_.size()) {
                continue;  // Skip invalid indices
            }
            auto kf = observed_kfs[i];
            if (!kf) {
                continue;  // Skip null keyframes
            }

            // Convert Sophus::SE3d pose (camera -> world, T_wc) to world -> camera (T_cw)
            const Sophus::SE3d& pose_wc = kf->KF_pose_g;
            Eigen::Matrix4d T_wc = pose_wc.matrix();
            Eigen::Matrix4d T_cw = T_wc.inverse();

            Matrix34d Rt;
            Rt.block<3,3>(0,0) = T_cw.topLeftCorner<3,3>();
            Rt.col(3)          = T_cw.topRightCorner<3,1>();

            // Check that ellipsoid center is in front of the camera
            const double z_before = Rt.row(2).dot(center_before_h);
            if (!(z_before > 0.0)) {
                continue;
            }

            Eigen::Matrix<double, 3, 4> P = K_ * Rt;

            // Compute IoU between current projection and measured bbox for gating
            const Ellipse proj_before = ellipsoid.project(P);
            const double iou_before = bboxes_iou(proj_before.ComputeBbox(), bboxes_[i]);
            if (!(iou_before > 0.1)) {  // same threshold as voom checkOptimization()
                continue;
            }

            sum_iou_before += iou_before;
            inlier_indexes.push_back(i);

            const Ellipse& det_ell = ellipses_[i];
            graph.emplace_shared<EllipsoidProjectionFactorGtsam>(
                det_ell, P, robust_noise, key);
        }

        const double mean_iou_before =
            inlier_indexes.empty() ? 0.0 : (sum_iou_before / static_cast<double>(inlier_indexes.size()));

        if (graph.empty() || inlier_indexes.size() < 4) {
            LOG(ERROR) << "[Object::OptimizeReconstructionQuat] WARNING: Empty graph for object_id=" << id_;
            return;
        }

        // Compute initial error for logging
        double initial_error = graph.error(values);
        VLOG(1) << "[Object::OptimizeReconstructionQuat] Graph has " << graph.size() 
                << " factors, initial error=" << initial_error;

        gtsam::LevenbergMarquardtParams params;
        params.setVerbosityLM("ERROR");
        params.setMaxIterations(20);  // Increase iterations for better convergence
        params.setAbsoluteErrorTol(1e-5);
        params.setRelativeErrorTol(1e-5);
        // Note: GTSAM doesn't have setInitialLambda, lambda is adjusted automatically
        
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
        gtsam::Values result = optimizer.optimize();
        
        // Compute final error
        double final_error = graph.error(result);
        int iterations = optimizer.iterations();

        Vector9 x_opt = result.at<Vector9>(key);
        Eigen::Vector3d axes = x_opt.segment<3>(0).cwiseAbs();
        Eigen::Vector3d center = x_opt.segment<3>(3);
        Eigen::Vector3d w = x_opt.segment<3>(6);
        gtsam::Rot3 R_opt = gtsam::Rot3::Expmap(w);

        // Build optimized ellipsoid for post-check
        Ellipsoid ellipsoid_after(axes, R_opt.matrix(), center);

        // Post-check (match voom checkOptimization): compute mean IoU after optimization
        double sum_iou_after = 0.0;
        size_t inliers_after = 0;
        const Eigen::Vector4d center_after_h(center[0], center[1], center[2], 1.0);

        for (auto i : inlier_indexes) {
            auto kf = observed_kfs[i];
            if (!kf) continue;

            const Sophus::SE3d& pose_wc = kf->KF_pose_g;
            Eigen::Matrix4d T_wc = pose_wc.matrix();
            Eigen::Matrix4d T_cw = T_wc.inverse();

            Matrix34d Rt;
            Rt.block<3,3>(0,0) = T_cw.topLeftCorner<3,3>();
            Rt.col(3)          = T_cw.topRightCorner<3,1>();

            const double z_after = Rt.row(2).dot(center_after_h);
            if (!(z_after > 0.0)) continue;

            Eigen::Matrix<double, 3, 4> P = K_ * Rt;
            const Ellipse proj_after = ellipsoid_after.project(P);
            const double iou_after = bboxes_iou(proj_after.ComputeBbox(), bboxes_[i]);
            if (iou_after > 0.1) {
                sum_iou_after += iou_after;
                inliers_after++;
            }
        }

        const double mean_iou_after =
            (inliers_after == 0) ? 0.0 : (sum_iou_after / static_cast<double>(inliers_after));

        // Validate optimization result: check for reasonable ellipsoid size + IoU support
        // Typical object sizes: 0.1m to 5m for most objects
        const double max_axis = 10.0;  // Maximum axis length in meters
        const double min_axis = 0.01;  // Minimum axis length in meters
        const double max_axis_change_ratio = 5.0;  // Maximum allowed change ratio
        
        bool axes_valid = true;
        for (int i = 0; i < 3; ++i) {
            if (axes[i] > max_axis || axes[i] < min_axis) {
                axes_valid = false;
                break;
            }
            // Check if axes changed too dramatically
            if (axes_init[i] > 0.01 && std::abs(axes[i] / axes_init[i]) > max_axis_change_ratio) {
                axes_valid = false;
                break;
            }
        }
        
        // Check if error increased significantly (optimization diverged)
        bool error_valid = (final_error < initial_error * 10.0);  // Allow some increase but not too much

        // IoU-based validation (voom-like): require enough inliers and mean IoU above threshold
        const bool iou_valid = (inliers_after >= 4) && (mean_iou_after >= 0.1) && (mean_iou_after >= mean_iou_before * 0.8);
        
        if (!axes_valid || !error_valid || !iou_valid) {
            LOG(WARNING) << "[Object::OptimizeReconstructionQuat] Optimization result invalid for object_id=" << id_
                        << ", axes_valid=" << axes_valid << ", error_valid=" << error_valid << ", iou_valid=" << iou_valid
                        << ", axes=[" << axes.transpose() << "] (init=[" << axes_init.transpose() << "])"
                        << ", error: " << initial_error << " -> " << final_error
                        << ", mean_iou: " << mean_iou_before << " -> " << mean_iou_after
                        << " - Rejecting optimization result, keeping initial ellipsoid";
            return;  // Reject optimization result, keep original ellipsoid
        }

        // Log optimization results
        Eigen::Vector3d axes_change = axes - axes_init;
        Eigen::Vector3d center_change = center - center_init;
        VLOG(1) << "[Object::OptimizeReconstructionQuat] Optimization completed for object_id=" << id_
                << ", iterations=" << iterations
                << ", error: " << initial_error << " -> " << final_error 
                << " (reduction: " << ((initial_error - final_error) / initial_error * 100.0) << "%)"
                << ", axes=[" << axes.transpose() << "] (change: [" << axes_change.transpose() << "])"
                << ", center=[" << center.transpose() << "] (change: [" << center_change.transpose() << "])";

        Ellipsoid new_ellipsoid(axes, R_opt.matrix(), center);
        SetEllipsoid(new_ellipsoid);
    }


    // Edge-SLAM: filter associated WORLD points that fall inside the current ellipsoid.
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
    Object::GetFilteredAssociatedMapPoints(int /*threshold*/)
    {
        std::unique_lock<std::mutex> lock(mutex_associated_map_points_);
        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> filtered;
        filtered.reserve(associated_world_points_.size());

        for (const auto& pw : associated_world_points_) {
            if (ellipsoid_.IsInside(pw, 1.0)) {
                filtered.push_back(pw);
            }
        }
        return filtered;
    }


}
