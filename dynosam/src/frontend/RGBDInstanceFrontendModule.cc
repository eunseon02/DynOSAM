/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"

#include <glog/logging.h>

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/common/Flags.hpp"  //for common flags
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"

DEFINE_bool(use_frontend_logger, false,
            "If true, the frontend logger will be used");
DEFINE_bool(use_dynamic_track, true,
            "If true, the dynamic tracking will be used");

DEFINE_bool(log_projected_masks, false,
            "If true, projected masks will be saved at every frame");

namespace dyno {

RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(
    const FrontendParams& frontend_params, Camera::Ptr camera,
    ImageDisplayQueue* display_queue)
    : FrontendModule(frontend_params, display_queue),
      camera_(camera),
      motion_solver_(frontend_params.ego_motion_solver_params,
                     camera->getParams()) {
  // object_motion_solver_(frontend_params.object_motion_solver_params,
  //                       camera->getParams()) {

  CHECK_NOTNULL(camera_);
  tracker_ =
      std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);

  if (FLAGS_use_frontend_logger) {
    logger_ = std::make_unique<RGBDFrontendLogger>();
  }

  gtsam::ISAM2Params isam2_params;
  isam2_params.evaluateNonlinearError = true;

  ObjectMotionSovlerF2F::Params object_motion_solver_params =
      frontend_params.object_motion_solver_params;
  // add ground truth hook
  object_motion_solver_params.ground_truth_packets_request = [&]() {
    return this->getGroundTruthPackets();
  };

  object_motion_solver_ = std::make_unique<ObjectMotionSolverSAM>(
      object_motion_solver_params, camera->getParams(), isam2_params);
}

RGBDInstanceFrontendModule::~RGBDInstanceFrontendModule() {
  if (FLAGS_save_frontend_json) {
    LOG(INFO) << "Saving frontend output as json";
    const std::string file_path =
        getOutputFilePath(kRgbdFrontendOutputJsonFile);
    JsonConverter::WriteOutJson(output_packet_record_, file_path,
                                JsonConverter::Format::BSON);
  }
}

FrontendModule::ImageValidationResult
RGBDInstanceFrontendModule::validateImageContainer(
    const ImageContainer::Ptr& image_container) const {
  return ImageValidationResult(image_container->hasDepth(),
                               "Depth is required");
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::boostrapSpin(
    FrontendInputPacketBase::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;

  // if we only have instance semgentation (not motion) then we need to make a
  // motion mask out of the semantic mask we cannot do this for the first frame
  // so we will just treat the semantic mask and the motion mask and then
  // subsequently elimate non-moving objects later on
  TrackingInputImages tracking_images;
  if (image_container->hasSemanticMask()) {
    CHECK(!image_container->hasMotionMask());
    // TODO: some bug when going from semantic mask to motion mask as motion
    // mask is empty in the tracker after this process!!! its becuase we dont
    // actually use the tracking_images!!
    auto intermediate_tracking_images =
        image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow,
                                    ImageType::SemanticMask>();
    tracking_images = TrackingInputImages(
        intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
        intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
        ImageWrapper<ImageType::MotionMask>(
            intermediate_tracking_images.get<ImageType::SemanticMask>()));
  } else {
    tracking_images =
        image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow,
                                    ImageType::MotionMask>();
  }

  Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                                     *image_container);
  CHECK(frame->updateDepths());

  return {State::Nominal, nullptr};
}

FrontendModule::SpinReturn RGBDInstanceFrontendModule::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;
  // if we only have instance semgentation (not motion) then we need to make a
  // motion mask out of the semantic mask we cannot do this for the first frame
  // so we will just treat the semantic mask and the motion mask and then
  // subsequently elimate non-moving objects later on
  TrackingInputImages tracking_images;
  if (image_container->hasSemanticMask()) {
    CHECK(!image_container->hasMotionMask());

    auto intermediate_tracking_images =
        image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow,
                                    ImageType::SemanticMask>();
    tracking_images = TrackingInputImages(
        intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
        intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
        ImageWrapper<ImageType::MotionMask>(
            intermediate_tracking_images.get<ImageType::SemanticMask>()));
  } else {
    tracking_images =
        image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow,
                                    ImageType::MotionMask>();
  }

  Frame::Ptr frame = nullptr;
  {
    utils::TimingStatsCollector tracking_timer("tracking_timer");
    frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                            *image_container);
  }
  CHECK(frame);

  Frame::Ptr previous_frame = tracker_->getPreviousFrame();
  CHECK(previous_frame);

  LOG(INFO) << to_string(tracker_->getTrackerInfo());

  {
    utils::TimingStatsCollector update_depths_timer("depth_updater");
    frame->updateDepths();
  }
  // updates frame->T_world_camera_
  if (!solveCameraMotion(frame, previous_frame)) {
    LOG(ERROR) << "Could not solve for camera";
  }

  if (FLAGS_use_dynamic_track) {
    // TODO: bring back byte tracker??
    utils::TimingStatsCollector track_dynamic_timer("tracking_dynamic");
    vision_tools::trackDynamic(base_params_, *previous_frame, frame);
  }

  MotionEstimateMap motion_estimates;
  std::tie(motion_estimates, object_poses_) =
      object_motion_solver_->solve(frame, previous_frame);
  // solveObjectMotions(frame, previous_frame, motion_estimates);

  // update the object_poses trajectory map which will be send to the viz
  // propogateObjectPoses(motion_estimates, frame->getFrameId());

  if (logger_) {
    auto ground_truths = this->getGroundTruthPackets();
    logger_->logCameraPose(frame->getFrameId(), frame->getPose(),
                           ground_truths);
    logger_->logObjectMotion(frame->getFrameId(), motion_estimates,
                             ground_truths);
    logger_->logTrackingLengthHistogram(frame);
    logger_->logFrameIdToTimestamp(frame->getFrameId(), frame->getTimestamp());
  }

  DebugImagery debug_imagery;
  debug_imagery.tracking_image =
      tracker_->computeImageTracks(*previous_frame, *frame);
  if (display_queue_)
    display_queue_->push(
        ImageToDisplay("tracks", debug_imagery.tracking_image));

  debug_imagery.detected_bounding_boxes = frame->drawDetectedObjectBoxes();
  // use the tracking images from the frame NOT the input tracking images since
  // the feature tracking will do some modifications on the images
  // (particularily the mask during mask propogation)
  //  debug_imagery.input_images = frame->tracking_images_;
  debug_imagery.input_images = tracking_images;

  RGBDInstanceOutputPacket::Ptr output =
      constructOutput(*frame, motion_estimates, frame->T_world_camera_,
                      input->optional_gt_, debug_imagery);

  if (FLAGS_save_frontend_json)
    output_packet_record_.insert({output->getFrameId(), output});

  if (FLAGS_log_projected_masks)
    vision_tools::writeOutProjectMaskAndDepthMap(
        frame->image_container_.get<ImageType::Depth>(),
        frame->image_container_.get<ImageType::MotionMask>(),
        *frame->getCamera(), frame->getFrameId());

  if (logger_) {
    auto ground_truths = this->getGroundTruthPackets();
    logger_->logPoints(output->getFrameId(), output->T_world_camera_,
                       output->dynamic_landmarks_);
    // object_poses_ are in frontend module
    logger_->logObjectPose(output->getFrameId(), object_poses_, ground_truths);
    logger_->logObjectBbxes(output->getFrameId(), output->getObjectBbxes());
  }
  return {State::Nominal, output};
}

bool RGBDInstanceFrontendModule::solveCameraMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_k_1) {
  Pose3SolverResult result;
  if (base_params_.use_ego_motion_pnp) {
    result = motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k);
  } else {
    // TODO: untested
    LOG(FATAL) << "Not tested";
    // result = motion_solver_.geometricOutlierRejection3d3d(frame_k_1,
    // frame_k);
  }

  VLOG(15) << (base_params_.use_ego_motion_pnp ? "3D2D" : "3D3D")
           << "camera pose estimate at frame " << frame_k->frame_id_
           << (result.status == TrackingStatus::VALID ? " success "
                                                      : " failure ")
           << ":\n"
           << "- Tracking Status: " << to_string(result.status) << '\n'
           << "- Total Correspondences: "
           << result.inliers.size() + result.outliers.size() << '\n'
           << "\t- # inliers: " << result.inliers.size() << '\n'
           << "\t- # outliers: " << result.outliers.size() << '\n';

  if (result.status == TrackingStatus::VALID) {
    frame_k->T_world_camera_ = result.best_result;
    TrackletIds tracklets = frame_k->static_features_.collectTracklets();
    CHECK_GE(tracklets.size(),
             result.inliers.size() +
                 result.outliers.size());  // tracklets shoudl be more (or same
                                           // as) correspondances as there will
                                           // be new points untracked
    frame_k->static_features_.markOutliers(result.outliers);

    if (base_params_.refine_camera_pose_with_joint_of) {
      VLOG(10) << "Refining camera pose with joint of";
      OpticalFlowAndPoseOptimizer flow_optimizer(
          base_params_.object_motion_solver_params.joint_of_params);

      auto flow_opt_result = flow_optimizer.optimizeAndUpdate<CalibrationType>(
          frame_k_1, frame_k, result.inliers, result.best_result);
      frame_k->T_world_camera_ = flow_opt_result.best_result.refined_pose;
    }
    return true;
  } else {
    frame_k->T_world_camera_ = gtsam::Pose3::Identity();
    return false;
  }
}

RGBDInstanceOutputPacket::Ptr RGBDInstanceFrontendModule::constructOutput(
    const Frame& frame, const MotionEstimateMap& estimated_motions,
    const gtsam::Pose3& T_world_camera,
    const GroundTruthInputPacket::Optional& gt_packet,
    const DebugImagery::Optional& debug_imagery) {
  StatusKeypointMeasurements static_keypoint_measurements;
  StatusLandmarkEstimates static_landmarks;
  for (const Feature::Ptr& f : frame.usableStaticFeaturesBegin()) {
    const TrackletId tracklet_id = f->trackletId();
    const Keypoint kp = f->keypoint();
    Landmark lmk_camera;
    camera_->backProject(kp, f->depth(), &lmk_camera);
    CHECK(f->isStatic());
    CHECK(Feature::IsUsable(f));

    // dont include features that have only been seen once as we havent had a
    // chance to validate it yet
    if (f->age() < 1) {
      continue;
    }

    static_keypoint_measurements.push_back(
        KeypointStatus::Static(kp, frame.getFrameId(), tracklet_id));

    static_landmarks.push_back(LandmarkStatus::StaticInLocal(
        lmk_camera, frame.getFrameId(), tracklet_id,
        LandmarkStatus::Method::MEASURED));
  }

  StatusKeypointMeasurements dynamic_keypoint_measurements;
  StatusLandmarkEstimates dynamic_landmarks;
  for (const auto& [object_id, obs] : frame.object_observations_) {
    CHECK_EQ(object_id, obs.instance_label_);
    // TODO: add back in?
    //  CHECK(obs.marked_as_moving_);

    for (const TrackletId tracklet : obs.object_features_) {
      if (frame.isFeatureUsable(tracklet)) {
        const Feature::Ptr f = frame.at(tracklet);
        CHECK(!f->isStatic());
        CHECK_EQ(f->objectId(), object_id);

        // dont include features that have only been seen once as we havent had
        // a chance to validate it yet
        if (f->age() < 1) {
          continue;
        }

        const TrackletId tracklet_id = f->trackletId();
        const Keypoint kp = f->keypoint();
        Landmark lmk_camera;
        camera_->backProject(kp, f->depth(), &lmk_camera);

        dynamic_keypoint_measurements.push_back(KeypointStatus::Dynamic(
            kp, frame.frame_id_, tracklet_id, object_id));

        dynamic_landmarks.push_back(LandmarkStatus::DynamicInLocal(
            lmk_camera, frame.frame_id_, tracklet_id, object_id,
            LandmarkStatus::Method::MEASURED));
      }
    }
  }

  // update trajectory of camera poses to be visualised by the frontend viz
  // module
  camera_poses_.push_back(T_world_camera);

  return std::make_shared<RGBDInstanceOutputPacket>(
      static_keypoint_measurements, dynamic_keypoint_measurements,
      static_landmarks, dynamic_landmarks, T_world_camera, frame.timestamp_,
      frame.frame_id_, estimated_motions, object_poses_, camera_poses_, camera_,
      gt_packet, debug_imagery);
}

// void RGBDInstanceFrontendModule::propogateObjectPoses(
//     const MotionEstimateMap& motion_estimates, FrameId frame_id) {
//   gtsam::Point3Vector object_centroids_k_1, object_centroids_k;

//   for (const auto& [object_id, motion_estimate] : motion_estimates) {
//     const auto frame_k_1 = tracker_->getPreviousFrame();
//     const auto frame_k = tracker_->getCurrentFrame();

//     auto object_points = FeatureFilterIterator(
//         const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
//         [object_id, &frame_k](const Feature::Ptr& f) -> bool {
//           return Feature::IsUsable(f) && f->objectId() == object_id &&
//                  frame_k->exists(f->trackletId()) &&
//                  frame_k->isFeatureUsable(f->trackletId());
//         });

//     gtsam::Point3 centroid_k_1(0, 0, 0);
//     gtsam::Point3 centroid_k(0, 0, 0);
//     size_t count = 0;
//     for (const auto& feature : object_points) {
//       gtsam::Point3 lmk_k_1 =
//           frame_k_1->backProjectToCamera(feature->trackletId());
//       centroid_k_1 += lmk_k_1;

//       gtsam::Point3 lmk_k =
//       frame_k->backProjectToCamera(feature->trackletId()); centroid_k +=
//       lmk_k;

//       count++;
//     }

//     centroid_k_1 /= count;
//     centroid_k /= count;

//     centroid_k_1 = frame_k_1->getPose() * centroid_k_1;
//     centroid_k = frame_k->getPose() * centroid_k;

//     object_centroids_k_1.push_back(centroid_k_1);
//     object_centroids_k.push_back(centroid_k);
//   }

//   if (FLAGS_init_object_pose_from_gt) {
//     dyno::propogateObjectPoses(object_poses_, motion_estimates,
//                                object_centroids_k_1, object_centroids_k,
//                                frame_id, getGroundTruthPackets());
//   } else {
//     dyno::propogateObjectPoses(object_poses_, motion_estimates,
//                                object_centroids_k_1, object_centroids_k,
//                                frame_id);
//   }
// }

}  // namespace dyno
