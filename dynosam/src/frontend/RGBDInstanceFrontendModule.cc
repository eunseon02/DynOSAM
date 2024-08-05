// /*
//  *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
//  *   All rights reserved.

//  *   Permission is hereby granted, free of charge, to any person obtaining a copy
//  *   of this software and associated documentation files (the "Software"), to deal
//  *   in the Software without restriction, including without limitation the rights
//  *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  *   copies of the Software, and to permit persons to whom the Software is
//  *   furnished to do so, subject to the following conditions:

//  *   The above copyright notice and this permission notice shall be included in all
//  *   copies or substantial portions of the Software.

//  *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  *   SOFTWARE.
//  */

// #include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
// #include "dynosam/frontend/RGBDInstance-Definitions.hpp"
// #include "dynosam/frontend/vision/Vision-Definitions.hpp"
// #include "dynosam/frontend/vision/MotionSolver.hpp"

// #include "dynosam/common/DynamicObjects.hpp"

// #include "dynosam/utils/SafeCast.hpp"
// #include "dynosam/utils/TimingStats.hpp"
// #include "dynosam/logger/Logger.hpp"

// #include <opencv4/opencv2/opencv.hpp>
// #include <glog/logging.h>

// #include "dynosam/common/Flags.hpp" //for common flags

// DEFINE_bool(use_frontend_logger, false , "If true, the frontend logger will be used");

// namespace dyno {


// RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
//     : FrontendModule(frontend_params, display_queue),
//       camera_(camera),
//       motion_solver_(frontend_params, camera->getParams()),
//       object_motion_solver_(frontend_params, camera->getParams())
//     {
//     CHECK_NOTNULL(camera_);
//     tracker_ = std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);

//     //TODO: copies from Frame.cc -lots of places this is used!!!
//     const CameraParams& cam_params = camera->getParams();
//     cv::Mat P = cam_params.getCameraMatrix();
//     cv::Mat R = cv::Mat::eye(3,3,CV_32FC1);
//     undistorter_ = std::make_shared<UndistorterRectifier>(P, cam_params, R);

//     if(FLAGS_use_frontend_logger) {
//         logger_ = std::make_unique<RGBDFrontendLogger>();
//     }
//     LOG(INFO) << "Made RGBD frontend";
// }

// RGBDInstanceFrontendModule::~RGBDInstanceFrontendModule() {
//     if(FLAGS_save_frontend_json) {
//         LOG(INFO) << "Saving frontend output as json";
//         const std::string file_path = getOutputFilePath(kRgbdFrontendOutputJsonFile);
//         JsonConverter::WriteOutJson(output_packet_record_, file_path, JsonConverter::Format::BSON);
//     }

// }

// FrontendModule::ImageValidationResult
// RGBDInstanceFrontendModule::validateImageContainer(const ImageContainer::Ptr& image_container) const {
//     return ImageValidationResult(image_container->hasDepth(), "Depth is required");
// }

// FrontendModule::SpinReturn
// RGBDInstanceFrontendModule::boostrapSpin(FrontendInputPacketBase::ConstPtr input) {
//     LOG(INFO) << "Frontend boostrap";
//     Frame::Ptr frame = trackNewFrame(input);
//     RGBDInstanceOutputPacket::Ptr output = processFrame(frame, input->optional_gt_);
//     logOutputPacket(output);
//     return {State::Nominal, output};
// }


// FrontendModule::SpinReturn
// RGBDInstanceFrontendModule::nominalSpin(FrontendInputPacketBase::ConstPtr input) {
//     Frame::Ptr frame = trackNewFrame(input);
//     RGBDInstanceOutputPacket::Ptr output = processFrame(frame, input->optional_gt_);
//     logOutputPacket(output);
//     return {State::Nominal, output};
// }

// Frame::Ptr RGBDInstanceFrontendModule::trackNewFrame(FrontendInputPacketBase::ConstPtr input){
//     ImageContainer::Ptr image_container = input->image_container_;

//     objectTrack(image_container, input->getFrameId());

//     Frame::Ptr frame = nullptr;
//     {
//         utils::TimingStatsCollector tracking_timer("frontend.feature_tracker");
//         frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), *image_container);

//     }
//     CHECK(frame);

//     auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();
//     frame->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);
//     return frame;

// }

// RGBDInstanceOutputPacket::Ptr RGBDInstanceFrontendModule::processFrame(Frame::Ptr frame, GroundTruthInputPacket::Optional ground_truth) {
//     Frame::Ptr previous_frame = tracker_->getPreviousFrame();
//     if(!previous_frame) {
//         return constructOutput(*frame, MotionEstimateMap{}, frame->T_world_camera_, ground_truth);
//     }

//     //TODO: currently hack - the tracker now udpates the previous frame too (by resampling) so we need to udapte the depth too!
//     // auto depth_image_wrapper = previous_frame->image_container_.getImageWrapper<ImageType::Depth>();
//     // previous_frame->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

//     CHECK_EQ(previous_frame->frame_id_ + 1u, frame->frame_id_);
//     VLOG(20) << to_string(tracker_->getTrackerInfo());

//     auto image_container = frame->image_container_;

//     //updates frame->T_world_camera_
//     if(!solveCameraMotion(frame, previous_frame)) {
//         LOG(ERROR) << "Could not solve for camera";
//     }

//     //mark observations as moving or not
//     //if semantic mask is used, then use scene flow to try and determine if an object is moving or not!!
//     const bool is_semantic_mask = image_container.hasSemanticMask();
//     determineDynamicObjects(*previous_frame, frame, is_semantic_mask);

//     ObjectIds failed_object_tracks;

//     for(const auto& [object_id, observations] : frame->object_observations_) {

//         LOG(INFO) << "Solving motion for " << object_id << " with " << observations.numFeatures();

//         if(!solveObjectMotion(frame, previous_frame, object_id, frame->motion_estimates_)) {
//             VLOG(5) << "Could not solve motion for object " << object_id <<
//                 " from frame " << previous_frame->frame_id_ << " -> " << frame->frame_id_;
//             failed_object_tracks.push_back(object_id);
//         }

//     }
//     ///remove objects from the object observations list
//     //does not remove the features etc but stops the object being propogated to the backend
//     //as we loop over the object observations in the constructOutput function
//     for(auto object_id : failed_object_tracks) {
//         frame->object_observations_.erase(object_id);
//     }

//     //update the object_poses trajectory map which will be send to the viz
//     propogateObjectPoses(frame->motion_estimates_, frame->frame_id_);

//     DebugImagery debug_imagery;
//     debug_imagery.tracking_image = tracker_->computeImageTracks(*previous_frame, *frame);
//     if(display_queue_) display_queue_->push(ImageToDisplay("tracks", debug_imagery.tracking_image));

//     debug_imagery.detected_bounding_boxes = frame->drawDetectedObjectBoxes();
//     //use the tracking images from the frame NOT the input tracking images since the feature tracking
//     //will do some modifications on the images (particularily the mask during mask propogation)
//     //TODO: this really needs to be cloned but this is complex so do later!!
//     TrackingInputImages tracking_images;
//     if(image_container.hasSemanticMask()) {
//         CHECK(!image_container.hasMotionMask());

//         auto intermediate_tracking_images = image_container.makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::SemanticMask>();
//         tracking_images = TrackingInputImages(
//             intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
//             intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
//             ImageWrapper<ImageType::MotionMask>(
//                 intermediate_tracking_images.get<ImageType::SemanticMask>()
//             )
//         );
//     }
//     else {
//         tracking_images = image_container.makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
//     }

//     debug_imagery.input_images = tracking_images;

//     RGBDInstanceOutputPacket::Ptr output = constructOutput(*frame, frame->motion_estimates_, frame->T_world_camera_, ground_truth, debug_imagery);

//     return output;
// }

// void RGBDInstanceFrontendModule::logOutputPacket(const RGBDInstanceOutputPacket::Ptr& output) {
//     if(FLAGS_save_frontend_json) output_packet_record_.insert({output->getFrameId(), output});

//     if(logger_) {
//         logger_->logPoints(output->getFrameId(), output->T_world_camera_, output->dynamic_landmarks_);
//         //object_poses_ are in frontend module
//         logger_->logObjectPose(gt_packet_map_, output->getFrameId(), object_poses_);
//         logger_->logObjectBbxes(output->getFrameId(), output->getObjectBbxes());

//         logger_->logCameraPose(gt_packet_map_, output->getFrameId(), output->T_world_camera_);
//         logger_->logObjectMotion(gt_packet_map_, output->getFrameId(), output->estimated_motions_);
//     }

// }




// bool RGBDInstanceFrontendModule::solveCameraMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1) {

//     Pose3SolverResult result;
//     {
//     utils::TimingStatsCollector timer("frontend.solve_camera_motion");
//     if(base_params_.use_ego_motion_pnp) {
//         result = motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k);
//     }
//     else {
//         //TODO: untested
//         result = motion_solver_.geometricOutlierRejection3d3d(frame_k_1, frame_k);
//     }
//     }

//     VLOG(15) << (base_params_.use_ego_motion_pnp ? "3D2D" : "3D3D") << "camera pose estimate at frame " << frame_k->frame_id_
//         << (result.status == TrackingStatus::VALID ? " success " : " failure ") << ":\n"
//         << "- Tracking Status: "
//         << to_string(result.status) << '\n'
//         << "- Total Correspondences: " << result.inliers.size() + result.outliers.size() << '\n'
//         << "\t- # inliers: " << result.inliers.size() << '\n'
//         << "\t- # outliers: " << result.outliers.size() << '\n';

//     if(result.status == TrackingStatus::VALID) {
//         frame_k->T_world_camera_ = result.best_pose;
//             TrackletIds tracklets = frame_k->static_features_.collectTracklets();
//             CHECK_GE(tracklets.size(), result.inliers.size() + result.outliers.size()); //tracklets shoudl be more (or same as) correspondances as there will be new points untracked
//             //TODO: should also mark outliers on frame_k_1
//             frame_k->static_features_.markOutliers(result.outliers); //do we need to mark innliers? Should start as inliers

//             // gtsam::Pose3 refined_pose;
//             // gtsam::Point2Vector refined_flows;
//             // TrackletIds refined_inliers;
//             // //TODO: this function marks things as outliers internally but keeps the inlier as a out-arg
//             // //so if we're not careful there could be a mismatch!!
//             // motion_solver_.refineJointPoseOpticalFlow(
//             //     result,
//             //     frame_k_1,
//             //     frame_k,
//             //     refined_pose,
//             //     refined_flows,
//             //     refined_inliers
//             // );

//             // //original flow image that goes from k to k+1 (gross, im sorry!)
//             // const cv::Mat flow_image = frame_k->image_container_.get<ImageType::OpticalFlow>();

//             // //update flow and depth
//             // for(size_t i = 0; i < refined_inliers.size(); i++) {
//             //     TrackletId tracklet_id = refined_inliers.at(i);
//             //     gtsam::Point2 refined_flow = refined_flows.at(i);

//             //     const Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
//             //     Feature::Ptr feature_k = frame_k->at(tracklet_id);

//             //     const Keypoint kp_k_1 = feature_k_1->keypoint_;
//             //     Keypoint refined_keypoint = kp_k_1 + refined_flow;

//             //     //check boundaries?

//             //     //update keypoint!!
//             //     feature_k->keypoint_ = refined_keypoint;

//             //     //we now have to update the prediced keypoint using the original flow!!
//             //     //TODO: code copied from feature tracker
//             //     const int x = functional_keypoint::u(refined_keypoint);
//             //     const int y = functional_keypoint::v(refined_keypoint);
//             //     double flow_xe = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[0]);
//             //     double flow_ye = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[1]);
//             //     //the measured flow after the origin has been updated
//             //     OpticalFlow new_measured_flow(flow_xe, flow_ye);
//             //     feature_k->measured_flow_ = new_measured_flow;
//             //     // TODO: check predicted flow is within image
//             //     Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(refined_keypoint, new_measured_flow);
//             //     feature_k->predicted_keypoint_ = predicted_kp;


//             //     //woudl need to update the measured flow in frame_k_1 since, right now, flow is k-1 to k
//             //     //TODO:update depth
//             // }
//             // auto depth_image_wrapper = frame_k->image_container_.getImageWrapper<ImageType::Depth>();
//             // frame_k->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);
//             // // frame_k->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);
//             frame_k->T_world_camera_ = result.best_pose;
//             return true;
//     }
//     else {
//         frame_k->T_world_camera_ = gtsam::Pose3::Identity();
//         return false;
//     }

// }

// bool RGBDInstanceFrontendModule::solveObjectMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,  ObjectId object_id, MotionEstimateMap& motion_estimates) {

//     Pose3SolverResult result;

//     {
//     utils::TimingStatsCollector timer("frontend.solve_object_motion");
//     if(base_params_.use_object_motion_pnp) {
//         result = object_motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k,  frame_k->T_world_camera_, object_id);
//     }
//     else {
//         result = object_motion_solver_.geometricOutlierRejection3d3d(frame_k_1, frame_k,  frame_k->T_world_camera_, object_id);
//     }
//     }

//     VLOG(15) << (base_params_.use_object_motion_pnp ? "3D2D" : "3D3D") << " object motion estimate " << object_id << " at frame " << frame_k->frame_id_
//         << (result.status == TrackingStatus::VALID ? " success " : " failure ") << ":\n"
//         << "- Tracking Status: "
//         << to_string(result.status) << '\n'
//         << "- Total Correspondences: " << result.inliers.size() + result.outliers.size() << '\n'
//         << "\t- # inliers: " << result.inliers.size() << '\n'
//         << "\t- # outliers: " << result.outliers.size() << '\n';

//     //sanity check
//     //if valid, remove outliers and add to motion estimation
//     if(result.status == TrackingStatus::VALID) {
//         frame_k->dynamic_features_.markOutliers(result.outliers);

//         //TODO: we update depths here becuase in the geometricOutlierRejection3d2d the optical flow opt will move the
//         //keypoints and so we need to upadte the z values
//         //TODO: should not be separate from the optimisation so refactor API!!!
//         // auto depth_image_wrapper = frame_k->image_container_.getImageWrapper<ImageType::Depth>();
//         // frame_k->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

//         ReferenceFrameValue<Motion3> estimate(result.best_pose, ReferenceFrame::GLOBAL);

//         const std::lock_guard<std::mutex> lock(object_motion_mutex_);
//         motion_estimates.insert({object_id, estimate});
//         return true;
//     }
//     else {
//         return false;
//     }
// }

// void RGBDInstanceFrontendModule::objectTrack(ImageContainer::Ptr image_container, FrameId frame_id) {

//     if(FLAGS_use_byte_tracker) {
//         utils::TimingStatsCollector track_dynamic_timer("frontend.object_tracker_2d");
//         cv::Mat& original_mask = image_container->get<ImageType::MotionMask>();
//         original_mask = object_tracker_.track(image_container->get<ImageType::MotionMask>(), frame_id);
//     }

// }

// void RGBDInstanceFrontendModule::determineDynamicObjects(const Frame& previous_frame, Frame::Ptr current_frame, bool used_semantic_mask) {
//     auto& objects_by_instance_label = current_frame->object_observations_;
//     const auto& params = base_params_;

//     //a motion mask was used - just go through all the object observations and mark as dynamic!!
//     if(!used_semantic_mask) {
//         for(auto& [label, object_observation] : objects_by_instance_label) {
//             object_observation.marked_as_moving_ = true;
//         }
//         return;
//     }

//     auto& previous_dynamic_feature_container = previous_frame.dynamic_features_;
//     auto& current_dynamic_feature_container = current_frame->dynamic_features_;

//     ObjectIds instance_labels_to_remove;

//     utils::TimingStatsCollector track_dynamic_timer("frontend.scene_flow_motion_seg");
//     //a semantic mask was used - determine motion segmentation with optical flow!!
//     for(auto& [label, object_observation] : objects_by_instance_label) {
//         double sf_min=100, sf_max=0, sf_mean=0, sf_count=0;
//         std::vector<int> sf_range(10,0);

//         const size_t num_object_features = object_observation.object_features_.size();
//         // LOG(INFO) << "tracking object observation with instance label " << instance_label << " and " << num_object_features << " features";

//         int feature_pairs_valid = 0;
//         for(const TrackletId tracklet_id : object_observation.object_features_) {
//             if(previous_dynamic_feature_container.exists(tracklet_id)) {
//             CHECK(current_dynamic_feature_container.exists(tracklet_id));

//                 Feature::Ptr current_feature = current_dynamic_feature_container.getByTrackletId(tracklet_id);
//                 Feature::Ptr previous_feature = previous_dynamic_feature_container.getByTrackletId(tracklet_id);

//                 if(!previous_feature->usable()) {
//                     current_feature->markInvalid();
//                     continue;
//                 }

//                 //this can happen in situations such as the updateDepths when depths > thresh are marked invalud
//                 if(!current_feature->usable()) { continue;}

//                 CHECK(!previous_feature->isStatic());
//                 CHECK(!current_feature->isStatic());

//                 Landmark lmk_previous = previous_frame.backProjectToWorld(tracklet_id);
//                 Landmark lmk_current = current_frame->backProjectToWorld(tracklet_id);

//                 Landmark flow_world = lmk_current - lmk_previous ;
//                 double sf_norm = flow_world.norm();

//                 feature_pairs_valid++;

//                 if (sf_norm<params.scene_flow_magnitude)
//                     sf_count = sf_count+1;
//                 if(sf_norm<sf_min)
//                     sf_min = sf_norm;
//                 if(sf_norm>sf_max)
//                     sf_max = sf_norm;
//                 sf_mean = sf_mean + sf_norm;

//                 {
//                     if (0.0<=sf_norm && sf_norm<0.05)
//                         sf_range[0] = sf_range[0] + 1;
//                     else if (0.05<=sf_norm && sf_norm<0.1)
//                         sf_range[1] = sf_range[1] + 1;
//                     else if (0.1<=sf_norm && sf_norm<0.2)
//                         sf_range[2] = sf_range[2] + 1;
//                     else if (0.2<=sf_norm && sf_norm<0.4)
//                         sf_range[3] = sf_range[3] + 1;
//                     else if (0.4<=sf_norm && sf_norm<0.8)
//                         sf_range[4] = sf_range[4] + 1;
//                     else if (0.8<=sf_norm && sf_norm<1.6)
//                         sf_range[5] = sf_range[5] + 1;
//                     else if (1.6<=sf_norm && sf_norm<3.2)
//                         sf_range[6] = sf_range[6] + 1;
//                     else if (3.2<=sf_norm && sf_norm<6.4)
//                         sf_range[7] = sf_range[7] + 1;
//                     else if (6.4<=sf_norm && sf_norm<12.8)
//                         sf_range[8] = sf_range[8] + 1;
//                     else if (12.8<=sf_norm && sf_norm<25.6)
//                         sf_range[9] = sf_range[9] + 1;
//                 }

//             }

//         }

//         VLOG(10) << "Number feature pairs valid " << feature_pairs_valid << " out of " << num_object_features << " for label  " << label;

//         if (sf_count/num_object_features>params.scene_flow_percentage || num_object_features < 10u)
//         {
//             // label this object as static background
//             // LOG(INFO) << "Instance object " << instance_label << " to static for frame " << current_frame->frame_id_;
//             instance_labels_to_remove.push_back(label);
//             object_observation.marked_as_moving_ = false;
//         }
//         else {
//             // LOG(INFO) << "Instance object " << instance_label << " marked as dynamic";
//             object_observation.marked_as_moving_ = true;
//         }
//     }

//     //we do the removal after the iteration so as not to mess up the loop
//     for(const auto label : instance_labels_to_remove) {
//         VLOG(30) << "Removing label " << label;
//         //TODO: this is really really slow!!
//         current_frame->moveObjectToStatic(label);
//         // LOG(INFO) << "Done Removing label " << label;
//     }
// }




// RGBDInstanceOutputPacket::Ptr RGBDInstanceFrontendModule::constructOutput(
//     const Frame& frame,
//     const MotionEstimateMap& estimated_motions,
//     const gtsam::Pose3& T_world_camera,
//     const GroundTruthInputPacket::Optional& gt_packet,
//     const DebugImagery::Optional& debug_imagery)
// {
//     StatusKeypointMeasurements static_keypoint_measurements;
//     StatusLandmarkEstimates static_landmarks;
//     for(const Feature::Ptr& f : frame.usableStaticFeaturesBegin()) {
//         const TrackletId tracklet_id = f->tracklet_id_;
//         const Keypoint kp = f->keypoint_;
//         Landmark lmk_camera;
//         camera_->backProject(kp, f->depth_, &lmk_camera);
//         CHECK(f->isStatic());
//         CHECK(Feature::IsUsable(f));

//         //we really need static feature in the backend (and frame nodes are constructed of observations)
//         //and we need frame zero to be part of the backend so that the world frame is the same for both gt and estimation motions!

//         //dont include features that have only been seen once as we havent had a chance to validate it yet
//         // if(f->age_ < 1) {
//         //     continue;
//         // }


//         static_keypoint_measurements.push_back(
//             KeypointStatus::Static(
//                 kp,
//                 frame.frame_id_,
//                 tracklet_id
//             )
//         );

//         static_landmarks.push_back(
//             LandmarkStatus::StaticInLocal(
//                 lmk_camera,
//                 frame.frame_id_,
//                 tracklet_id,
//                 LandmarkStatus::Method::MEASURED
//             )
//         );
//     }


//     StatusKeypointMeasurements dynamic_keypoint_measurements;
//     StatusLandmarkEstimates dynamic_landmarks;
//     for(const auto& [object_id, obs] : frame.object_observations_) {
//         CHECK_EQ(object_id, obs.instance_label_);
//         //TODO: whats the logic here?
//         //on the first frame they will not be moving yet!
//         // CHECK(obs.marked_as_moving_);
//         if(!obs.marked_as_moving_) {continue;}

//         for(const TrackletId tracklet : obs.object_features_) {
//             if(frame.isFeatureUsable(tracklet)) {
//                 const Feature::Ptr f = frame.at(tracklet);
//                 CHECK(!f->isStatic());
//                 CHECK_EQ(f->instance_label_, object_id);

//                 //dont include features that have only been seen once as we havent had a chance to validate it yet
//                 if(f->age_ < 1) {
//                     continue;
//                 }

//                 const TrackletId tracklet_id = f->tracklet_id_;
//                 const Keypoint kp = f->keypoint_;
//                 Landmark lmk_camera;
//                 camera_->backProject(kp, f->depth_, &lmk_camera);

//                 dynamic_keypoint_measurements.push_back(
//                     KeypointStatus::Dynamic(
//                         kp,
//                         frame.frame_id_,
//                         tracklet_id,
//                         object_id
//                     )
//                 );

//                 dynamic_landmarks.push_back(
//                     LandmarkStatus::DynamicInLocal(
//                         lmk_camera,
//                         frame.frame_id_,
//                         tracklet_id,
//                         object_id,
//                         LandmarkStatus::Method::MEASURED
//                     )
//                 );
//             }
//         }

//     }

//     //update trajectory of camera poses to be visualised by the frontend viz module
//     camera_poses_.push_back(T_world_camera);

//     return std::make_shared<RGBDInstanceOutputPacket>(
//         static_keypoint_measurements,
//         dynamic_keypoint_measurements,
//         static_landmarks,
//         dynamic_landmarks,
//         T_world_camera,
//         frame.timestamp_,
//         frame.frame_id_,
//         estimated_motions,
//         object_poses_,
//         camera_poses_,
//         camera_,
//         gt_packet,
//         debug_imagery
//     );
// }


// void RGBDInstanceFrontendModule::propogateObjectPoses(const MotionEstimateMap& motion_estimates, FrameId frame_id) {

//     gtsam::Point3Vector object_centroids_k_1, object_centroids_k;

//     for(const auto& [object_id, motion_estimate] : motion_estimates) {
//         const auto frame_k_1 = tracker_->getPreviousFrame();
//         const auto frame_k = tracker_->getCurrentFrame();

//         auto object_points = FeatureFilterIterator(
//             const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
//             [object_id, &frame_k](const Feature::Ptr& f) -> bool {
//                 return Feature::IsUsable(f) && f->instance_label_ == object_id &&
//                     frame_k->exists(f->tracklet_id_) && frame_k->isFeatureUsable(f->tracklet_id_);
//             });

//         gtsam::Point3 centroid_k_1(0, 0, 0);
//         gtsam::Point3 centroid_k(0, 0, 0);
//         size_t count = 0;
//         for(const auto& feature : object_points) {
//             gtsam::Point3 lmk_k_1 = frame_k_1->backProjectToWorld(feature->tracklet_id_);
//             centroid_k_1 += lmk_k_1;

//             gtsam::Point3 lmk_k = frame_k->backProjectToWorld(feature->tracklet_id_);
//             centroid_k += lmk_k;

//             count++;
//         }

//         centroid_k_1 /= count;
//         centroid_k /= count;

//         object_centroids_k_1.push_back(centroid_k_1);
//         object_centroids_k.push_back(centroid_k);
//     }

//     if(FLAGS_init_object_pose_from_gt) {
//         dyno::propogateObjectPoses(
//             object_poses_,
//             motion_estimates,
//             object_centroids_k_1,
//             object_centroids_k,
//             frame_id,
//             gt_packet_map_);
//     }
//     else {
//         dyno::propogateObjectPoses(
//             object_poses_,
//             motion_estimates,
//             object_centroids_k_1,
//             object_centroids_k,
//             frame_id);
//     }

// }

// void RGBDInstanceFrontendModule::propogateObjectPose(const gtsam::Pose3& prev_H_world_curr, ObjectId object_id, FrameId frame_id) {

// }

// void RGBDInstanceFrontendModule::propogateObjectPose(const gtsam::Pose3& prev_H_world_curr, ObjectId object_id, FrameId frame_id) {
//     //start from previous frame -> if we have a motion we must have observations in k-1
//     const FrameId frame_id_k_1 = frame_id - 1;
//     if(!object_poses_.exists(object_id)) {

//         gtsam::Pose3 starting_pose;

//         //need gt pose for rotation even if not for translation
//         CHECK(gt_packet_map_.exists(frame_id_k_1)) << "Cannot initalise object poses for viz using gt as the ground truth does not exist for frame " << frame_id_k_1;
//         const GroundTruthInputPacket& gt_packet_k_1 = gt_packet_map_.at(frame_id_k_1);

//         ObjectPoseGT object_pose_gt_k;
//         if(!gt_packet_k_1.getObject(object_id, object_pose_gt_k)) {
//             LOG(ERROR) << "Object Id " << object_id <<  " cannot be found in the gt packet. Unable to initalise object starting point use gt pose. Skipping!";
//             return;
//             // throw std::runtime_error("Object Id " + std::to_string(object_id) + " cannot be found in the gt packet. Unable to initalise object starting point use gt pose");
//         }

//         starting_pose = object_pose_gt_k.L_world_;

//         //if not use gt translation, find centroid of object
//         if(!FLAGS_init_object_pose_from_gt) {
//             const auto frame_k_1 = tracker_->getPreviousFrame();
//             const auto frame_k = tracker_->getCurrentFrame();

//             auto object_points = FeatureFilterIterator(
//                 const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
//                 [object_id, &frame_k](const Feature::Ptr& f) -> bool {
//                     return Feature::IsUsable(f) && f->instance_label_ == object_id &&
//                     frame_k->exists(f->tracklet_id_) && frame_k->isFeatureUsable(f->tracklet_id_);
//                 });

//             gtsam::Point3 centroid(0, 0, 0);
//             size_t count = 0;
//             for(const auto& feature : object_points) {
//                 gtsam::Point3 lmk = frame_k_1->backProjectToWorld(feature->tracklet_id_);
//                 centroid += lmk;
//                 count++;
//             }

//             centroid /= count;
//             //update starting pose
//             starting_pose = gtsam::Pose3(object_pose_gt_k.L_world_.rotation(), centroid);


//         }
//         //new object
//         object_poses_.insert2(object_id , gtsam::FastMap<FrameId, gtsam::Pose3>{});
//         object_poses_.at(object_id).insert2(frame_id_k_1, starting_pose);
//     }

//     auto& pose_map = object_poses_.at(object_id);
//     if(!pose_map.exists(frame_id_k_1)) {
//         //this can happen if the object misses a frame (e.g becomes too small - this is problematic while we do not properly propogate out own tracking labels!!)
//         LOG(ERROR)<< "Previous frame " << frame_id_k_1 << " does not exist for object " << object_id << ". They should be in order?";
//         return;
//     }

//     const gtsam::Pose3& object_pose_k_1 = pose_map.at(frame_id_k_1);
//     gtsam::Pose3 object_pose_k = prev_H_world_curr * object_pose_k_1;
//     pose_map.insert2(frame_id, object_pose_k);
// }

// } //dyno


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

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"
#include "dynosam/logger/Logger.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

#include "dynosam/common/Flags.hpp" //for common flags

DEFINE_bool(use_frontend_logger, false , "If true, the frontend logger will be used");
DEFINE_bool(use_dynamic_track, true, "If true, the dynamic tracking will be used");

namespace dyno {


RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : FrontendModule(frontend_params, display_queue),
      camera_(camera),
      motion_solver_(frontend_params, camera->getParams()),
      object_motion_solver_(frontend_params, camera->getParams())
    {
    CHECK_NOTNULL(camera_);
    tracker_ = std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);

    if(FLAGS_use_frontend_logger) {
        logger_ = std::make_unique<RGBDFrontendLogger>();
    }
}

RGBDInstanceFrontendModule::~RGBDInstanceFrontendModule() {
    if(FLAGS_save_frontend_json) {
        LOG(INFO) << "Saving frontend output as json";
        const std::string file_path = getOutputFilePath(kRgbdFrontendOutputJsonFile);
        JsonConverter::WriteOutJson(output_packet_record_, file_path, JsonConverter::Format::BSON);
    }

}

FrontendModule::ImageValidationResult
RGBDInstanceFrontendModule::validateImageContainer(const ImageContainer::Ptr& image_container) const {
    return ImageValidationResult(image_container->hasDepth(), "Depth is required");
}

FrontendModule::SpinReturn
RGBDInstanceFrontendModule::boostrapSpin(FrontendInputPacketBase::ConstPtr input) {

    ImageContainer::Ptr image_container = input->image_container_;

    //if we only have instance semgentation (not motion) then we need to make a motion mask out of the semantic mask
    //we cannot do this for the first frame so we will just treat the semantic mask and the motion mask
    //and then subsequently elimate non-moving objects later on
    TrackingInputImages tracking_images;
    if(image_container->hasSemanticMask()) {
        CHECK(!image_container->hasMotionMask());

        auto intermediate_tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::SemanticMask>();
        tracking_images = TrackingInputImages(
            intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
            intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
            ImageWrapper<ImageType::MotionMask>(
                intermediate_tracking_images.get<ImageType::SemanticMask>()
            )
        );
    }
    else {
        tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
    }


    Frame::Ptr frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), *image_container);

    auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();
    frame->updateDepths(image_container->getImageWrapper<ImageType::Depth>(), base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

    return {State::Nominal, nullptr};
}


FrontendModule::SpinReturn
RGBDInstanceFrontendModule::nominalSpin(FrontendInputPacketBase::ConstPtr input) {
    ImageContainer::Ptr image_container = input->image_container_;
    //if we only have instance semgentation (not motion) then we need to make a motion mask out of the semantic mask
    //we cannot do this for the first frame so we will just treat the semantic mask and the motion mask
    //and then subsequently elimate non-moving objects later on
    TrackingInputImages tracking_images;
    if(image_container->hasSemanticMask()) {
        CHECK(!image_container->hasMotionMask());

        auto intermediate_tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::SemanticMask>();
        tracking_images = TrackingInputImages(
            intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
            intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
            ImageWrapper<ImageType::MotionMask>(
                intermediate_tracking_images.get<ImageType::SemanticMask>()
            )
        );
    }
    else {
        tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
    }

    Frame::Ptr frame = nullptr;
    {
        utils::TimingStatsCollector tracking_timer("tracking_timer");
        frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), *image_container);

    }
    CHECK(frame);

    Frame::Ptr previous_frame = tracker_->getPreviousFrame();
    CHECK(previous_frame);

    LOG(INFO) << to_string(tracker_->getTrackerInfo());

    auto depth_image_wrapper = image_container->getImageWrapper<ImageType::Depth>();

    {
        utils::TimingStatsCollector update_depths_timer("depth_updater");
        frame->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

    }
    //updates frame->T_world_camera_
    if(!solveCameraMotion(frame, previous_frame)) {
        LOG(ERROR) << "Could not solve for camera";
    }

    if(FLAGS_use_dynamic_track) {
        //TODO: bring back byte tracker??
        utils::TimingStatsCollector track_dynamic_timer("tracking_dynamic");
        vision_tools::trackDynamic(base_params_,*previous_frame, frame);
    }


    MotionEstimateMap motion_estimates;
    ObjectIds failed_object_tracks;
    for(const auto& [object_id, observations] : frame->object_observations_) {

        if(!solveObjectMotion(frame, previous_frame, object_id, motion_estimates)) {
            VLOG(5) << "Could not solve motion for object " << object_id <<
                " from frame " << previous_frame->frame_id_ << " -> " << frame->frame_id_;
            failed_object_tracks.push_back(object_id);
        }
    }

    ///remove objects from the object observations list
    //does not remove the features etc but stops the object being propogated to the backend
    //as we loop over the object observations in the constructOutput function
    for(auto object_id : failed_object_tracks) {
        frame->object_observations_.erase(object_id);
    }

    //update the object_poses trajectory map which will be send to the viz
    propogateObjectPoses(motion_estimates, frame->frame_id_);

    if(logger_) {
        logger_->logCameraPose(gt_packet_map_, frame->frame_id_, frame->T_world_camera_);
        logger_->logObjectMotion(gt_packet_map_, frame->frame_id_, motion_estimates);
        logger_->logTrackingLengthHistogram(frame);
    }

    DebugImagery debug_imagery;
    debug_imagery.tracking_image = tracker_->computeImageTracks(*previous_frame, *frame);
    if(display_queue_) display_queue_->push(ImageToDisplay("tracks", debug_imagery.tracking_image));

    debug_imagery.detected_bounding_boxes = frame->drawDetectedObjectBoxes();
    //use the tracking images from the frame NOT the input tracking images since the feature tracking
    //will do some modifications on the images (particularily the mask during mask propogation)
    // debug_imagery.input_images = frame->tracking_images_;
    debug_imagery.input_images = tracking_images;



    RGBDInstanceOutputPacket::Ptr output = constructOutput(*frame, motion_estimates, frame->T_world_camera_, input->optional_gt_, debug_imagery);

    if(FLAGS_save_frontend_json) output_packet_record_.insert({output->getFrameId(), output});

    if(logger_) {
        logger_->logPoints(output->getFrameId(), output->T_world_camera_, output->dynamic_landmarks_);
        //object_poses_ are in frontend module
        logger_->logObjectPose(gt_packet_map_, output->getFrameId(), object_poses_);
        logger_->logObjectBbxes(output->getFrameId(), output->getObjectBbxes());
    }
    return {State::Nominal, output};
}


bool RGBDInstanceFrontendModule::solveCameraMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1) {

    Pose3SolverResult result;
    if(base_params_.use_ego_motion_pnp) {
        result = motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k);
    }
    else {
        //TODO: untested
        result = motion_solver_.geometricOutlierRejection3d3d(frame_k_1, frame_k);
    }

    VLOG(15) << (base_params_.use_ego_motion_pnp ? "3D2D" : "3D3D") << "camera pose estimate at frame " << frame_k->frame_id_
          << (result.status == TrackingStatus::VALID ? " success " : " failure ") << ":\n"
          << "- Tracking Status: "
          << to_string(result.status) << '\n'
          << "- Total Correspondences: " << result.inliers.size() + result.outliers.size() << '\n'
          << "\t- # inliers: " << result.inliers.size() << '\n'
          << "\t- # outliers: " << result.outliers.size() << '\n';

    if(result.status == TrackingStatus::VALID) {

        frame_k->T_world_camera_ = result.best_pose;

        if(FLAGS_refine_with_optical_flow) {
            gtsam::Pose3 refined_pose;
            gtsam::Point2Vector refined_flows;
            TrackletIds refined_inliers;
            motion_solver_.refineJointPoseOpticalFlow(
                result,
                frame_k_1,
                frame_k,
                refined_pose,
                refined_flows,
                refined_inliers
            );

            // //original flow image that goes from k to k+1 (gross, im sorry!)
            const cv::Mat flow_image = frame_k->image_container_.get<ImageType::OpticalFlow>();
            const cv::Mat& motion_mask = frame_k->image_container_.get<ImageType::MotionMask>();

            auto camera = frame_k->camera_;

            //HACK: internally just mark as outlier if it is so!!
            //update flow and depth
            for(size_t i = 0; i < refined_inliers.size(); i++) {
                TrackletId tracklet_id = refined_inliers.at(i);
                gtsam::Point2 refined_flow = refined_flows.at(i);

                const Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
                Feature::Ptr feature_k = frame_k->at(tracklet_id);

                CHECK_EQ(feature_k->instance_label_, background_label);

                const Keypoint kp_k_1 = feature_k_1->keypoint_;
                Keypoint refined_keypoint = kp_k_1 + refined_flow;

                //check boundaries?
                if(!camera->isKeypointContained(refined_keypoint)) {
                    feature_k->inlier_ = false;
                    continue;
                }

                //update keypoint!!
                feature_k->keypoint_ = refined_keypoint;
                ObjectId predicted_label = functional_keypoint::at<ObjectId>(refined_keypoint, motion_mask);
                if(predicted_label != background_label) {
                    feature_k->inlier_ = false;
                    //TODO: other fields of the feature does not get updated? Inconsistencies as measured flow, predicted kp
                    //etc are no longer correct!!?
                    continue;
                }

                //we now have to update the prediced keypoint using the original flow!!
                //TODO: code copied from feature tracker
                const int x = functional_keypoint::u(refined_keypoint);
                const int y = functional_keypoint::v(refined_keypoint);
                double flow_xe = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[0]);
                double flow_ye = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[1]);
                //the measured flow after the origin has been updated
                OpticalFlow new_measured_flow(flow_xe, flow_ye);
                feature_k->measured_flow_ = new_measured_flow;
                // TODO: check predicted flow is within image
                Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(refined_keypoint, new_measured_flow);
                feature_k->predicted_keypoint_ = predicted_kp;


                //woudl need to update the measured flow in frame_k_1 since, right now, flow is k-1 to k
                //TODO:update depth
            }

            frame_k->T_world_camera_ = refined_pose;

        }
        TrackletIds tracklets = frame_k->static_features_.collectTracklets();
        CHECK_GE(tracklets.size(), result.inliers.size() + result.outliers.size()); //tracklets shoudl be more (or same as) correspondances as there will be new points untracked
        frame_k->static_features_.markOutliers(result.outliers); //do we need to mark innliers? Should start as inliers

        // update depths after fetures marked as outliers?
        auto depth_image_wrapper = frame_k->image_container_.getImageWrapper<ImageType::Depth>();
        frame_k->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);


        return true;
    }
    else {
        frame_k->T_world_camera_ = gtsam::Pose3::Identity();
        return false;
    }

}

bool RGBDInstanceFrontendModule::solveObjectMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1,  ObjectId object_id, MotionEstimateMap& motion_estimates) {

    Pose3SolverResult result;
    if(base_params_.use_object_motion_pnp) {
        result = object_motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k,  frame_k->T_world_camera_, object_id);
    }
    else {
        result = object_motion_solver_.geometricOutlierRejection3d3d(frame_k_1, frame_k,  frame_k->T_world_camera_, object_id);
    }

    VLOG(15) << (base_params_.use_object_motion_pnp ? "3D2D" : "3D3D") << " object motion estimate " << object_id << " at frame " << frame_k->frame_id_
          << (result.status == TrackingStatus::VALID ? " success " : " failure ") << ":\n"
          << "- Tracking Status: "
          << to_string(result.status) << '\n'
          << "- Total Correspondences: " << result.inliers.size() + result.outliers.size() << '\n'
          << "\t- # inliers: " << result.inliers.size() << '\n'
          << "\t- # outliers: " << result.outliers.size() << '\n';

    //sanity check
    //if valid, remove outliers and add to motion estimation
    if(result.status == TrackingStatus::VALID) {
        frame_k->dynamic_features_.markOutliers(result.outliers);

         //TODO: we update depths here becuase in the geometricOutlierRejection3d2d the optical flow opt will move the
//         //keypoints and so we need to upadte the z values
//         //TODO: should not be separate from the optimisation so refactor API!!!
        // auto depth_image_wrapper = frame_k->image_container_.getImageWrapper<ImageType::Depth>();
        // frame_k->updateDepths(depth_image_wrapper, base_params_.depth_background_thresh, base_params_.depth_obj_thresh);


        ReferenceFrameValue<Motion3> estimate(result.best_pose, ReferenceFrame::GLOBAL);
        motion_estimates.insert({object_id, estimate});
        return true;
    }
    else {
        return false;
    }
}


RGBDInstanceOutputPacket::Ptr RGBDInstanceFrontendModule::constructOutput(
    const Frame& frame,
    const MotionEstimateMap& estimated_motions,
    const gtsam::Pose3& T_world_camera,
    const GroundTruthInputPacket::Optional& gt_packet,
    const DebugImagery::Optional& debug_imagery)
{
    StatusKeypointMeasurements static_keypoint_measurements;
    StatusLandmarkEstimates static_landmarks;
    for(const Feature::Ptr& f : frame.usableStaticFeaturesBegin()) {
        const TrackletId tracklet_id = f->tracklet_id_;
        const Keypoint kp = f->keypoint_;
        Landmark lmk_camera;
        camera_->backProject(kp, f->depth_, &lmk_camera);
        CHECK(f->isStatic());
        CHECK(Feature::IsUsable(f));

        //dont include features that have only been seen once as we havent had a chance to validate it yet
        if(f->age_ < 1) {
            continue;
        }


        static_keypoint_measurements.push_back(
            KeypointStatus::Static(
                kp,
                frame.frame_id_,
                tracklet_id
            )
        );

        static_landmarks.push_back(
            LandmarkStatus::StaticInLocal(
                lmk_camera,
                frame.frame_id_,
                tracklet_id,
                LandmarkStatus::Method::MEASURED
            )
        );
    }


    StatusKeypointMeasurements dynamic_keypoint_measurements;
    StatusLandmarkEstimates dynamic_landmarks;
    for(const auto& [object_id, obs] : frame.object_observations_) {
        CHECK_EQ(object_id, obs.instance_label_);
        //TODO: add back in?
        // CHECK(obs.marked_as_moving_);

        for(const TrackletId tracklet : obs.object_features_) {
            if(frame.isFeatureUsable(tracklet)) {
                const Feature::Ptr f = frame.at(tracklet);
                CHECK(!f->isStatic());
                CHECK_EQ(f->instance_label_, object_id);

                //dont include features that have only been seen once as we havent had a chance to validate it yet
                if(f->age_ < 1) {
                    continue;
                }

                const TrackletId tracklet_id = f->tracklet_id_;
                const Keypoint kp = f->keypoint_;
                Landmark lmk_camera;
                camera_->backProject(kp, f->depth_, &lmk_camera);

                dynamic_keypoint_measurements.push_back(
                    KeypointStatus::Dynamic(
                        kp,
                        frame.frame_id_,
                        tracklet_id,
                        object_id
                    )
                );

                dynamic_landmarks.push_back(
                    LandmarkStatus::DynamicInLocal(
                        lmk_camera,
                        frame.frame_id_,
                        tracklet_id,
                        object_id,
                        LandmarkStatus::Method::MEASURED
                    )
                );
            }
        }

    }

    //update trajectory of camera poses to be visualised by the frontend viz module
    camera_poses_.push_back(T_world_camera);

    return std::make_shared<RGBDInstanceOutputPacket>(
        static_keypoint_measurements,
        dynamic_keypoint_measurements,
        static_landmarks,
        dynamic_landmarks,
        T_world_camera,
        frame.timestamp_,
        frame.frame_id_,
        estimated_motions,
        object_poses_,
        camera_poses_,
        camera_,
        gt_packet,
        debug_imagery
    );
}


// void RGBDInstanceFrontendModule::propogateObjectPoses(const MotionEstimateMap& motion_estimates, FrameId frame_id) {
//     for(const auto& [object_id, motion_estimate] : motion_estimates) {
//         propogateObjectPose(motion_estimate, object_id, frame_id);
//     }
// }

// void RGBDInstanceFrontendModule::propogateObjectPose(const gtsam::Pose3& prev_H_world_curr, ObjectId object_id, FrameId frame_id) {
//     //start from previous frame -> if we have a motion we must have observations in k-1
//     const FrameId frame_id_k_1 = frame_id - 1;
//     if(!object_poses_.exists(object_id)) {

//         gtsam::Pose3 starting_pose;

//         //need gt pose for rotation even if not for translation
//         CHECK(gt_packet_map_.exists(frame_id_k_1)) << "Cannot initalise object poses for viz using gt as the ground truth does not exist for frame " << frame_id_k_1;
//         const GroundTruthInputPacket& gt_packet_k_1 = gt_packet_map_.at(frame_id_k_1);

//         ObjectPoseGT object_pose_gt_k;
//         if(!gt_packet_k_1.getObject(object_id, object_pose_gt_k)) {
//             LOG(ERROR) << "Object Id " << object_id <<  " cannot be found in the gt packet. Unable to initalise object starting point use gt pose. Skipping!";
//             return;
//             // throw std::runtime_error("Object Id " + std::to_string(object_id) + " cannot be found in the gt packet. Unable to initalise object starting point use gt pose");
//         }

//         starting_pose = object_pose_gt_k.L_world_;

//         //if not use gt translation, find centroid of object
//         if(!FLAGS_init_object_pose_from_gt) {
//             const auto frame_k_1 = tracker_->getPreviousFrame();
//             const auto frame_k = tracker_->getCurrentFrame();

//             auto object_points = FeatureFilterIterator(
//                 const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
//                 [object_id, &frame_k](const Feature::Ptr& f) -> bool {
//                     return Feature::IsUsable(f) && f->instance_label_ == object_id &&
//                     frame_k->exists(f->tracklet_id_) && frame_k->isFeatureUsable(f->tracklet_id_);
//                 });

//             gtsam::Point3 centroid(0, 0, 0);
//             size_t count = 0;
//             for(const auto& feature : object_points) {
//                 gtsam::Point3 lmk = frame_k_1->backProjectToWorld(feature->tracklet_id_);
//                 centroid += lmk;
//                 count++;
//             }

//             centroid /= count;
//             //update starting pose
//             starting_pose = gtsam::Pose3(object_pose_gt_k.L_world_.rotation(), centroid);


//         }
//         //new object
//         object_poses_.insert2(object_id , gtsam::FastMap<FrameId, gtsam::Pose3>{});
//         object_poses_.at(object_id).insert2(frame_id_k_1, starting_pose);
//     }

//     auto& pose_map = object_poses_.at(object_id);
//     if(!pose_map.exists(frame_id_k_1)) {
//         //this can happen if the object misses a frame (e.g becomes too small - this is problematic while we do not properly propogate out own tracking labels!!)
//         LOG(ERROR)<< "Previous frame " << frame_id_k_1 << " does not exist for object " << object_id << ". They should be in order?";
//         return;
//     }

//     const gtsam::Pose3& object_pose_k_1 = pose_map.at(frame_id_k_1);
//     gtsam::Pose3 object_pose_k = prev_H_world_curr * object_pose_k_1;
//     pose_map.insert2(frame_id, object_pose_k);
// }

void RGBDInstanceFrontendModule::propogateObjectPoses(const MotionEstimateMap& motion_estimates, FrameId frame_id) {

    gtsam::Point3Vector object_centroids_k_1, object_centroids_k;

    for(const auto& [object_id, motion_estimate] : motion_estimates) {
        const auto frame_k_1 = tracker_->getPreviousFrame();
        const auto frame_k = tracker_->getCurrentFrame();

        auto object_points = FeatureFilterIterator(
            const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
            [object_id, &frame_k](const Feature::Ptr& f) -> bool {
                return Feature::IsUsable(f) && f->instance_label_ == object_id &&
                    frame_k->exists(f->tracklet_id_) && frame_k->isFeatureUsable(f->tracklet_id_);
            });

        gtsam::Point3 centroid_k_1(0, 0, 0);
        gtsam::Point3 centroid_k(0, 0, 0);
        size_t count = 0;
        for(const auto& feature : object_points) {
            gtsam::Point3 lmk_k_1 = frame_k_1->backProjectToCamera(feature->tracklet_id_);
            centroid_k_1 += lmk_k_1;

            gtsam::Point3 lmk_k = frame_k->backProjectToCamera(feature->tracklet_id_);
            centroid_k += lmk_k;

            count++;
        }

        centroid_k_1 /= count;
        centroid_k /= count;

        centroid_k_1 = frame_k_1->getPose() * centroid_k_1;
        centroid_k = frame_k->getPose() * centroid_k;


        object_centroids_k_1.push_back(centroid_k_1);
        object_centroids_k.push_back(centroid_k);
    }

    if(FLAGS_init_object_pose_from_gt) {
        dyno::propogateObjectPoses(
            object_poses_,
            motion_estimates,
            object_centroids_k_1,
            object_centroids_k,
            frame_id,
            gt_packet_map_);
    }
    else {
        CHECK(false);
        dyno::propogateObjectPoses(
            object_poses_,
            motion_estimates,
            object_centroids_k_1,
            object_centroids_k,
            frame_id);
    }

}


} //dyno
