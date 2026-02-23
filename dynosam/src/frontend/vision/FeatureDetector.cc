/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/frontend/vision/FeatureDetector.hpp"

#include <glog/logging.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for_each.h>
// #include <tbb/parallel_for.h>

#include <chrono>
#include <iomanip>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include "dynosam/frontend/vision/ORBextractor.hpp"
#include "dynosam_common/Cuda.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

#ifdef DYNO_CUDA_OPENCV_ENABLED
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace dyno {

#ifdef DYNO_CUDA_OPENCV_ENABLED

struct GFTTDetectorCUDA {};

template <>
FunctionalDetector::Ptr FunctionalDetector::Create<GFTTDetectorCUDA>(
    const TrackerParams& tracker_params) {
  LOG(INFO) << "Creating cv::cuda::createGoodFeaturesToTrackDetector";

  cv::Ptr<cv::cuda::CornersDetector> detector =
      cv::cuda::createGoodFeaturesToTrackDetector(
          CV_8UC1, tracker_params.max_nr_keypoints_before_anms,
          tracker_params.gfft_params.quality_level,
          tracker_params.min_distance_btw_tracked_and_detected_static_features,
          tracker_params.gfft_params.block_size,
          tracker_params.gfft_params.use_harris_corner_detector,
          tracker_params.gfft_params.k);

  // auto feature_detector_ = cv::GFTTDetector::create(
  //     tracker_params.max_nr_keypoints_before_anms,
  //     tracker_params.gfft_params.quality_level,
  //     tracker_params.min_distance_btw_tracked_and_detected_static_features,
  //     tracker_params.gfft_params.block_size,
  //     tracker_params.gfft_params.use_harris_corner_detector,
  //     tracker_params.gfft_params.k);
  auto functional_detector = [=](const cv::Mat& img, KeypointsCV& keypoints,
                                 const cv::Mat& mask) -> void {
    // Detect keypoints
    // Upload to GPU
    unsigned int width = img.size().width;
    unsigned int height = img.size().height;

    cv::cuda::GpuMat d_img(height, width, img.type());
    d_img.upload(img);  // simple and portable

    cv::cuda::GpuMat d_mask(height, width, mask.type());
    d_mask.upload(mask);  // simple and portable

    cv::cuda::GpuMat keypointsGPU;
    detector->detect(d_img, keypointsGPU, d_mask);

    std::vector<cv::Point2f> points;
    keypointsGPU.download(points);

    cv::KeyPoint::convert(points, keypoints);
  };

  return std::make_shared<FunctionalDetector>(functional_detector);
}
#endif

template <>
FunctionalDetector::Ptr FunctionalDetector::Create<cv::GFTTDetector>(
    const TrackerParams& tracker_params) {
  LOG(INFO) << "Creating cv::GFTTDetector";

  auto feature_detector_ = cv::GFTTDetector::create(
      tracker_params.max_nr_keypoints_before_anms,
      tracker_params.gfft_params.quality_level,
      tracker_params.min_distance_btw_tracked_and_detected_static_features,
      tracker_params.gfft_params.block_size,
      tracker_params.gfft_params.use_harris_corner_detector,
      tracker_params.gfft_params.k);
  auto functional_detector = [=](const cv::Mat& img, KeypointsCV& keypoints,
                                 const cv::Mat& mask) -> void {
    CHECK_NOTNULL(feature_detector_)->detect(img, keypoints, mask);
  };

  return std::make_shared<FunctionalDetector>(functional_detector);
}

template <>
FunctionalDetector::Ptr FunctionalDetector::Create<ORBextractor>(
    const TrackerParams& tracker_params) {
  LOG(INFO) << "Creating dyno::ORBextractor";

  auto orb_detector_ = std::make_shared<ORBextractor>(
      tracker_params.max_nr_keypoints_before_anms,
      tracker_params.orb_params.scale_factor,
      tracker_params.orb_params.n_levels,
      tracker_params.orb_params.init_threshold_fast,
      tracker_params.orb_params.min_threshold_fast);

  // NOTE that the mask is not used in this implementation
  auto functional_detector = [=](const cv::Mat& img, KeypointsCV& keypoints,
                                 const cv::Mat&) -> void {
    CHECK_NOTNULL(orb_detector_);
    // mask and descriptors are empty
    cv::Mat descriptors;
    orb_detector_->operator()(img, cv::Mat(), keypoints, descriptors);
  };

  return std::make_shared<FunctionalDetector>(functional_detector);
}

FunctionalDetector::Ptr FunctionalDetector::FactoryCreate(
    const TrackerParams& tracker_params) {
  using FDT = TrackerParams::FeatureDetectorType;
  switch (tracker_params.feature_detector_type) {
    case FDT::GFTT:
      return FunctionalDetector::Create<cv::GFTTDetector>(tracker_params);
    case FDT::ORB_SLAM_ORB:
      return FunctionalDetector::Create<ORBextractor>(tracker_params);
    case FDT::GFFT_CUDA: {
      // TODO: this should actually be a #ifdef because
      // Create<cv::cuda::FastFeatureDetector> is conditionally compiled
      if (utils::opencvCudaAvailable()) {
        return FunctionalDetector::Create<GFTTDetectorCUDA>(tracker_params);
      } else {
        LOG(WARNING) << "GFFT_CUDA selected but OPENCV CUDA not enabled. "
                        "Falling back to GFFT";
        return FunctionalDetector::Create<cv::GFTTDetector>(tracker_params);
      }
    }
    default:
      LOG(ERROR) << "Unknown Feature detection type!";
      return nullptr;
      break;
  }
}

SparseFeatureDetector::SparseFeatureDetector(
    const TrackerParams& tracker_params,
    const FeatureDetector::Ptr& feature_detector)
    : tracker_params_(tracker_params),
      feature_detector_(CHECK_NOTNULL(feature_detector)),
      clahe_(nullptr),
      non_maximum_supression_(nullptr),
      mbUseFixedThreshold(true),
      mpCanny_lower_bound(tracker_params.edge_coarse.cannyLow),
      mpCanny_higher_bound(tracker_params.edge_coarse.cannyHigh),
      mpAngle_bias(20.0f) {
  // Enable OpenCV optimizations (same as ROEVO)
  cv::setUseOptimized(true);
  cv::setNumThreads(0);
  
  if (tracker_params_.use_clahe_filter)
    clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));  // TODO: make params

  if (tracker_params.use_anms)
    non_maximum_supression_ = std::make_unique<AdaptiveNonMaximumSuppression>(
        tracker_params.anms_params.non_max_suppression_type);
}

void SparseFeatureDetector::detect(const cv::Mat& image, KeypointsCV& keypoints,
                                   int number_tracked,
                                   const cv::Mat& detection_mask) {
  cv::Mat processed_image = image.clone();

  // pre-process image if required
  if (clahe_) clahe_->apply(processed_image, processed_image);

  std::vector<cv::KeyPoint> raw_keypoints;
  {
    utils::ChronoTimingStats timer("feature_detector.detect");
    feature_detector_->detect(processed_image, raw_keypoints, detection_mask);
  }

  std::vector<cv::KeyPoint>& max_keypoints = raw_keypoints;
  if (tracker_params_.use_anms) {
    // calculate number of corners needed
    int nr_corners_needed =
        std::max(tracker_params_.max_features_per_frame - number_tracked, 0);

    static constexpr float tolerance = 0.1;

    const auto& anms_params = tracker_params_.anms_params;
    Eigen::MatrixXd binning_mask = anms_params.binning_mask;

    {
      utils::ChronoTimingStats timer("feature_detector.anms");
      max_keypoints = non_maximum_supression_->suppressNonMax(
          raw_keypoints, nr_corners_needed, tolerance, processed_image.cols,
          processed_image.rows, anms_params.nr_horizontal_bins,
          anms_params.nr_vertical_bins, binning_mask);
    }
  }

  if (tracker_params_.use_subpixel_corner_refinement &&
      max_keypoints.size() > 0u) {
    // Convert keypoints to points
    std::vector<cv::Point2f> points;
    cv::KeyPoint::convert(max_keypoints, points);

    const auto& subpixel_corner_refinement_params =
        tracker_params_.subpixel_corner_refinement_params;
    utils::ChronoTimingStats timer("feature_detector.sub_pix");
    cv::cornerSubPix(processed_image, points,
                     subpixel_corner_refinement_params.window_size,
                     subpixel_corner_refinement_params.zero_zone,
                     subpixel_corner_refinement_params.criteria);

    cv::KeyPoint::convert(points, max_keypoints);
  }

  keypoints = max_keypoints;
}

void SparseFeatureDetector::detectEdge(const cv::Mat& image, std::vector<Edge>& edges,
  const cv::Mat& detection_mask) {
    // Clear previous results to ensure clean state
    mvEdges.clear();
    mvEdgeClusters.clear();
    edges.clear();

    // Validate input image
    if (image.empty()) {
        LOG(ERROR) << "Input image is empty in detectEdge";
        return;
    }
    
    // Initialize width and height
    mWidth = image.cols;
    mHeight = image.rows;
    
    {
        utils::ChronoTimingStats timer("edge_detection.gradient");
        cv::Mat grad_x, grad_y;
        cv::Scharr(image, grad_x, CV_32F, 1, 0);
        cv::Scharr(image, grad_y, CV_32F, 0, 1);

        mMatGradMagnitude.create(image.size(), CV_32F);
        mMatGradAngle.create(image.size(), CV_32F);
        
        // Calculate gradient magnitude and direction: magnitude is the size, angle is the direction (0~360 degrees)
        // The last parameter: false means L1 norm gradient magnitude, true means L2 norm gradient magnitude
        cv::cartToPolar(grad_x, grad_y, mMatGradMagnitude, mMatGradAngle, true);
    }

    {
        utils::ChronoTimingStats timer("edge_detection.canny");
        if(mbUseFixedThreshold)
        {
            cv::Canny(image, mMatCanny, mpCanny_lower_bound, mpCanny_higher_bound, 3, true); 
        }else{
            cv::Mat binary;
            double otsu_thresh = cv::threshold(image, binary, 0, 255, cv::THRESH_OTSU);
            cv::Canny(image, mMatCanny, 0.5*otsu_thresh, otsu_thresh, 3, true);
        }
    }
    
    // Apply detection_mask to Canny result: only detect edges where mask != 0
    if (!detection_mask.empty()) {
        utils::ChronoTimingStats timer("edge_detection.mask");
        CHECK_EQ(detection_mask.type(), CV_8U);
        CHECK_EQ(mMatCanny.size(), detection_mask.size());
        cv::bitwise_and(mMatCanny, detection_mask, mMatCanny);
    }
    
    {
        utils::ChronoTimingStats timer("edge_detection.preprocess");
        int edge_count_before = cv::countNonZero(mMatCanny);
        preprocessCannyMat();
        int edge_count_after = cv::countNonZero(mMatCanny);
    }
    
    {
        utils::ChronoTimingStats timer("edge_detection.clustering");
        regionGrowthClusteringOCanny(mpAngle_bias, detection_mask);
    }
    
    {
        utils::ChronoTimingStats timer("edge_detection.ordered_edges");
        cvt2OrderedEdgesParallel();
    }
    
    VLOG(10) << "SparseFeatureDetector::detectEdge: detected " << mvEdges.size() << " edges";

    edges = mvEdges;
}

float SparseFeatureDetector::calcAngleBias(float angle_1, float angle_2)
{
    float res = fabs(angle_1 - angle_2);
    if(res > 180){
        res = 360 - res;
    }
    return res;
}


void SparseFeatureDetector::preprocessCannyMat()
{
    cv::Mat matBinary;
    mMatCanny.convertTo(matBinary, CV_8U, 1.0/255); // Directly convert to 0/1 values

    for(int i = 0; i < matBinary.rows; ++i) 
    {
        uint8_t* current = matBinary.ptr<uint8_t>(i);
        uint8_t* above = i > 0 ? matBinary.ptr<uint8_t>(i-1) : nullptr;
        uint8_t* below = i < matBinary.rows-1 ? matBinary.ptr<uint8_t>(i+1) : nullptr;
        
        for(int j = 0; j < matBinary.cols; ++j) 
        {
            if(current[j] == 0) continue; // Skip non-edge points
            
            int left = j > 0 ? current[j-1] : 0;
            int right = j < matBinary.cols-1 ? current[j+1] : 0;
            int up = above ? above[j] : 0;
            int down = below ? below[j] : 0;
            
            bool connected = (left > 0 && up > 0) || (right > 0 && up > 0) ||
                             (left > 0 && down > 0) || (right > 0 && down > 0);
            
            if(connected) {
                current[j] = 0;
            }
        }
    }
}

void SparseFeatureDetector::regionGrowthClusteringOCanny(float angle_Thres, const cv::Mat& detection_mask)
{
    cv::Mat labelMatTmp(mMatCanny.rows, mMatCanny.cols, CV_16UC1, cv::Scalar::all(65535));
    // Matrix to track if current point has been visited: 0 means not visited, 1 means visited
    cv::Mat visitedMat(mMatCanny.rows, mMatCanny.cols, CV_8UC1, cv::Scalar::all(0));
    // Clear edge clusters
    mvEdgeClusters.clear();
    
    // Pre-define image pointers
    uint8_t* canny_ptr = mMatCanny.data;
    uint16_t* label_ptr = (uint16_t*)labelMatTmp.data;
    uint8_t* visited_ptr = visitedMat.data;
    const float* angle_ptr = mMatGradAngle.ptr<float>(0);
    const int canny_step = mMatCanny.step;
    const int label_step = labelMatTmp.step / sizeof(uint16_t);
    const int visited_step = visitedMat.step;
    const int angle_step = mMatGradAngle.step / sizeof(float);
    const int rows = mMatCanny.rows;
    const int cols = mMatCanny.cols;
    const int width = mWidth;
    const int height = mHeight;

    int label_global = 0;
    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < cols; ++x)
        {
            // If this point is a Canny edge and hasn't been visited, use it as a seed to find a region through region growing
            if (visited_ptr[y * visited_step + x] != 0 || canny_ptr[y * canny_step + x] != 255)
                continue;
            
            // Create a new cluster
            std::vector<edgePoint> current_cluster;
            label_global++;
            // Update visit status of current position
            visited_ptr[y * visited_step + x] = 1;
            // Each point in a cluster gets an ID starting from 0
            int point_id = 0;

            // Create an edge point. Since this is the first point of the edge, set ID to 0 and parent to -1
            edgePoint curr_edge_point(cv::Point(x,y), point_id, -1);
            curr_edge_point.isRoot = true;
            point_id++;

            // Queue for breadth-first region growing traversal
            std::queue<edgePoint> open_list;
            open_list.push(curr_edge_point);

            // Region growing traversal based on graph search (breadth-first)
            while(!open_list.empty())
            {
                // Get front element from queue using front()
                edgePoint current_point = open_list.front();
                open_list.pop();

                const int cx = current_point.pixel.x;
                const int cy = current_point.pixel.y;

                // Pixels that can enter open_list must belong to the current cluster
                label_ptr[cy * label_step + cx] = label_global;
                current_cluster.push_back(current_point);

                // Expand neighboring regions of current pixel, add new pixels that meet conditions to open_list
                const float curr_angle = angle_ptr[cy * angle_step + cx];

                // Define a lambda method to expand neighborhood. The expanded neighborhood must satisfy:
                // ① Coordinates are within image bounds
                // ② Not visited yet
                // ③ Is a Canny edge point
                // ④ Gradient angle is continuous with its parent node
                auto check_and_push = [&](int nx, int ny) {
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height 
                        && visited_ptr[ny * visited_step + nx] == 0
                        && canny_ptr[ny * canny_step + nx] == 255) 
                    {
                        // Select points within image bounds that haven't been visited, get their gradient angle
                        float neigh_angle = angle_ptr[ny * angle_step + nx];
                        if (calcAngleBias(neigh_angle, curr_angle) < angle_Thres) 
                        {
                            // Points that meet angle requirements can be clustered, so assign ID and add to queue for further expansion
                            visited_ptr[ny * visited_step + nx] = 1;
                            // This point is expanded from current_point, so its parent ID is current_point.point_id
                            edgePoint neigh_pt(cv::Point(nx, ny), point_id, current_point.point_id);
                            open_list.push(neigh_pt);
                            point_id++;
                        }
                    }
                };
                // Expand to 8-neighborhood
                check_and_push(cx+1, cy);   // Right
                check_and_push(cx+1, cy+1); // Bottom-right
                check_and_push(cx,   cy+1); // Bottom
                check_and_push(cx-1, cy+1); // Bottom-left
                check_and_push(cx-1, cy);   // Left
                check_and_push(cx-1, cy-1); // Top-left
                check_and_push(cx,   cy-1); // Top
                check_and_push(cx+1, cy-1); // Top-right

            }
            
            // Region growth finished, a cluster is generated. Check cluster size, only keep large clusters
            if(current_cluster.size() > 9)
            {
                // Finish a cluster, now we have a complete current_cluster
                EdgeCluster edge(current_cluster);
                mvEdgeClusters.push_back(edge);
            }
        }
    }
}

void SparseFeatureDetector::cvt2OrderedEdgesParallel()
{
    // Parallel write, so directly resize instead of reverse
    mvEdges.resize(mvEdgeClusters.size());

    // Pre-fetch mMatGradAngle pointer (avoid repeated calls to at<>)
    const float* angle_ptr = mMatGradAngle.ptr<float>(0);
    const int angle_step = mMatGradAngle.step / sizeof(float);

    // Use TBB to process each edge cluster in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mvEdgeClusters.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i != range.end(); ++i) {
                Edge curr_edge(i);  // Each edge has its own independent ID
                const auto& cluster_points = mvEdgeClusters[i].organize();
                curr_edge.mvPoints.reserve(cluster_points.size());

                for (const auto& point : cluster_points) {
                    const int x = static_cast<int>(point.pixel.x);
                    const int y = static_cast<int>(point.pixel.y);

                    // angle(0~360) of the image gradient
                    const float angle = angle_ptr[y * angle_step + x];
                    
                    // an orderedEdgePoint is initially constructed by coordinate (x,y) and gradient angle
                    curr_edge.mvPoints.emplace_back(x, y, angle);
                }
                mvEdges[i] = std::move(curr_edge);  // Directly write to pre-allocated position
            }
        });
}

}  // namespace dyno
