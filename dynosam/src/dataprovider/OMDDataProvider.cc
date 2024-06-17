/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#include "dynosam/dataprovider/OMDDataProvider.hpp"

#include "dynosam/utils/CsvParser.hpp"
#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/visualizer/ColourMap.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include "dynosam/frontend/vision/VisionTools.hpp" //for getObjectLabels
#include "dynosam/common/Algorithms.hpp"

#include "dynosam/pipeline/ThreadSafeTemporalBuffer.hpp"

#include <glog/logging.h>
#include <filesystem>

namespace dyno {

class OMDAllLoader {

public:
    DYNO_POINTER_TYPEDEFS(OMDAllLoader)

    OMDAllLoader(const std::string& file_path)
    :   rgbd_folder_path_(file_path + "/rgbd_undistorted"),
        instance_masks_folder_(file_path + "/instance_masks"),
        optical_flow_folder_path_(file_path + "/optical_flow"),
        vicon_file_path_(file_path + "/vicon.csv"),
        kalibr_file_path_(file_path + "/kalibr.yaml"),
        vicon_calibration_file_path_(file_path + "/vicon.yaml")
    {
        throwExceptionIfPathInvalid(rgbd_folder_path_);
        throwExceptionIfPathInvalid(instance_masks_folder_);
        throwExceptionIfPathInvalid(optical_flow_folder_path_);
        throwExceptionIfPathInvalid(vicon_file_path_);
        throwExceptionIfPathInvalid(kalibr_file_path_);
        throwExceptionIfPathInvalid(vicon_calibration_file_path_);

        //first load images and size
        //the size will be used as a refernce for all other loaders
        //size is number of images
        //we use flow to set the size as for this dataset, there will be one less
        //optical flow as the flow ids index t to t+1 (which is gross, I know!!)
        //TODO: code and comments replicated in ClusterSlamDataProvider
        loadFlowImagesAndSize(optical_flow_image_paths_, dataset_size_);

        //load rgb and aligned depth image file paths into rgb_image_paths_ and aligned_depth_image_paths_
        //does some sanity checks
        //as well as timestamps
        loadRGBDImagesAndTime();

        //load instance masks
        loadPathsInDirectory(instance_masks_image_paths_, instance_masks_folder_);
        //remove the ones up to the dataset size
        instance_masks_image_paths_.resize(dataset_size_);

        //want one less rgb image than optical flow
        CHECK_EQ(optical_flow_image_paths_.size() - 1, rgb_image_paths_.size());
        CHECK_EQ(instance_masks_image_paths_.size(), optical_flow_image_paths_.size());

        setGroundTruthPacketFromVicon(ground_truth_packets_);

        setIntrisicsAndTransforms();
    }

    size_t size() const {
        return dataset_size_;
    }

    cv::Mat getRGB(size_t idx) const {
        CHECK_LT(idx, rgb_image_paths_.size());

        cv::Mat rgb;
        loadRGB(rgb_image_paths_.at(idx), rgb);
        CHECK(!rgb.empty());

        //this set of images are loaded as 8UC4
        // CHECK_EQ(rgb.type(), CV_8UC4) << "Somehow the image type has changed...";
        // rgb.convertTo(rgb, CV_8UC3);

        //debug check -> draw keypoints on image!
        // const LandmarksMap& kps_map = left_landmarks_map_.at(idx);

        // for(const auto&[landmark_id, kp] : kps_map) {
        //     const auto cluster_id = landmark_mapping_.at(landmark_id);

        //     cv::Point2f pt(utils::gtsamPointToCv(kp));
        //     utils::drawCircleInPlace(rgb, pt, ColourMap::getObjectColour(cluster_id));
        // }

        return rgb;
    }

    cv::Mat getOpticalFlow(size_t idx) const {
        CHECK_LT(idx,optical_flow_image_paths_.size());

        cv::Mat flow;
        loadFlow(optical_flow_image_paths_.at(idx), flow);
        CHECK(!flow.empty());
        return flow;
    }

    cv::Mat getInstanceMask(size_t idx) const {
        CHECK_LT(idx, rgb_image_paths_.size());
        CHECK_LT(idx, dataset_size_);

        cv::Mat mask, relabelled_mask;
        loadMask(instance_masks_image_paths_.at(idx), mask);
        CHECK(!mask.empty());

        associateGTWithObject(mask, getDepthImage(idx), idx, relabelled_mask);

        // cv::Mat mask_viz = ImageType::SemanticMask::toRGB(relabelled_mask);

        // cv::imshow("Relabelled mask", mask_viz);
        // cv::waitKey(1);

        return relabelled_mask;
    }

    cv::Mat getDepthImage(size_t idx) const {
        CHECK_LT(idx, aligned_depth_image_paths_.size());

        cv::Mat depth;
        loadDepth(aligned_depth_image_paths_.at(idx), depth);
        // loadRGB(aligned_depth_image_paths_.at(idx), depth);
        CHECK(!depth.empty());
        // CHECK(depth.type() == CV_16UC1);
        // depth.convertTo(depth, CV_64F);

        //The D435 publishes depth in "16-bit unsigned integers in millimeter resolution."
        //TODO: paramterise and check why this is different for OMD
        //why is dyno-sam so different that it needs the baseline scale factor as well as the scaling term
        //is this just depth to dispartiy?
        //imD.at<float>(i,j) = mbf/(imD.at<float>(i,j)/mDepthMapFactor);
        depth /= 1000.0;

        // cv::Mat depth;
        // const_disparity.copyTo(depth);
        // for (int i = 0; i < depth.rows; i++)
        // {
        //     for (int j = 0; j < depth.cols; j++)
        //     {
        //         if (depth.at<double>(i, j) < 0) {
        //             depth.at<double>(i, j) = 0;
        //         }
        //         else {
        //             depth.at<double>(i, j) = 387.5744 / (depth.at<double>(i, j) / 1000.0);
        //         }
        //     }
        // }

        return depth;
    }

    const GroundTruthInputPacket& getGtPacket(size_t idx) const {
        return ground_truth_packets_.at(idx);
    }

    const CameraParams& getLeftCameraParams() const {
        return rgbd_camera_params_;
    }

    double getTimestamp(size_t idx) {
        return static_cast<double>(times_.at(idx));
    }


private:
    void loadFlowImagesAndSize(std::vector<std::string>& images_paths, size_t& dataset_size) {
        std::vector<std::filesystem::path> files_in_directory = getAllFilesInDir(optical_flow_folder_path_);
        dataset_size = files_in_directory.size();
        CHECK_GT(dataset_size, 0);

        for (const std::string file_path : files_in_directory) {
            throwExceptionIfPathInvalid(file_path);
            images_paths.push_back(file_path);
        }
   }

   inline Timestamp toTimestamp(double time_sec, double time_nsec) const {
    return (Timestamp)time_sec + (Timestamp)time_nsec / 1e+9;
   }

   void loadRGBDImagesAndTime() {
        // from dataset contains *_aligned_depth.png and *_color.png
        // need to construct both
        auto image_files = getAllFilesInDir(rgbd_folder_path_);

        //maps the image label prefix number with the two image paths - rgb (.first) and depth (.second)
        //this enables us to ensure that each image number (which use the same prefix) has both
        //rgb and depth images available in the dataset
        gtsam::FastMap<int, std::pair<std::string, std::string>> image_label_to_path_pair;

        const auto colour_substring = "color.png";
        const auto depth_substring = "aligned_depth.png";

        namespace fs = std::filesystem;

        for(const std::string file_path : image_files) {
            //extract image id
            //this should be in the form (XXXX_aligned_depth or XXXX_color)
            //get the filename from the file path (as the full file path contains "_" in other places
            //e.g root/data/omm/swinging_4_unconstrained/rgbd/010707_color.png
            //we just want the 010707_color.png componenent of the file path
            auto split_lines = trimAndSplit(std::string(fs::path(file_path).filename()), "_");
            int image_number = -1;

            try {
                image_number = std::stoi(split_lines.at(0));
            }
            catch(const std::invalid_argument& ex) {
                //this fails on the csv file
            }

            //if image number extracted successfully
            if(image_number != -1) {
                 if(!image_label_to_path_pair.exists(image_number)) {
                    image_label_to_path_pair.insert2(image_number, std::make_pair("", ""));
                }

                auto& file_path_pair = image_label_to_path_pair.at(image_number);

                if(file_path.find(colour_substring) != std::string::npos) {
                    //is rgb image
                    file_path_pair.first = file_path;
                }
                else if (file_path.find(depth_substring) != std::string::npos) {
                    //is depth image
                    file_path_pair.second = file_path;
                }
            }
            //special case - rgbd.csv file is inside this folder
            //and gives timestamp per frame number
            else if(file_path.find("rgbd.csv") != std::string::npos) {
                loadTimestampsFromRGBDCsvFile(file_path);
            }
            else {
                LOG(FATAL) << "Could not load rgbd image at path " << file_path;
            }
        }

        //go through the map and add all image paths that have both image paths
        //assume everything is ordered by FastMap
        for(const auto& [image_num, file_path_pair] : image_label_to_path_pair) {
            (void)image_num;
            if(file_path_pair.first.empty() || file_path_pair.second.empty()) {
                continue;
            }

            rgb_image_paths_.push_back(file_path_pair.first);
            aligned_depth_image_paths_.push_back(file_path_pair.second);

            //want one less than the number of flow images
            if(rgb_image_paths_.size() + 1 >= dataset_size_) {
                break;
            }
        }

        CHECK_EQ(rgb_image_paths_.size(), aligned_depth_image_paths_.size());
        CHECK_EQ(times_.size(), rgb_image_paths_.size());

        LOG(INFO) << "Loaded " << rgb_image_paths_.size() << " rgbd images";
   }

   void loadTimestampsFromRGBDCsvFile(const std::string& file_path) {
        //exepect a csv file with the header [frame_num, time_sec, time_nsec]
        //inside the rgbd folder
        //TODO: for now dont use a csv reader!!
        std::ifstream file(file_path);
        CsvReader csv_reader(file);
        auto it = csv_reader.begin();
        //skip header
        ++it;

        for(; it != csv_reader.end(); it++) {
            const auto row = *it;
            int frame_num = row.at<int>(0);
            double time_sec = row.at<double>(1);
            double time_nsec = row.at<double>(2);

            Timestamp time = toTimestamp(time_sec, time_nsec);
            times_.push_back(time);
            timestamp_frame_map_.insert({time, frame_num});

            //want one less than the number of flow images
            if(times_.size() + 1 >= dataset_size_) {
                break;
            }
        }
        LOG(INFO) << "Loaded " << times_.size() << " timestamps from RGBD csv file: " << file_path;

   }

   struct ViconPoseData {
        Timestamp timestamp;
        gtsam::Pose3 T_world_object; //or T_OA using the omd dataset notation, the inverse of what is given by the dataset (T_AO)
        ObjectId object_id;
   };


    void setGroundTruthPacketFromVicon(GroundTruthPacketMap& ground_truth_packets) {
        LOG(INFO) << "Loading object/camera pose gt from vicon: " << vicon_file_path_;
        std::ifstream file(vicon_file_path_);
        CsvReader csv_reader(file);
        auto it = csv_reader.begin();
        //skip header
        ++it;

        using ObjectIdPosePair = std::pair<ObjectId, gtsam::Pose3>;
        std::vector<ViconPoseData> vicon_pose_data;
        gtsam::FastMap<FrameId, std::vector<ObjectPoseGT>> temp_object_poses;

        for(; it != csv_reader.end(); it++) {
            const auto row = *it;
            //Though the stereo camera, RGB-D camera, and IMU were recorded on the same machine,
            //they are not hardware synchronized. The Vicon was recorded on a separate system with an unknown temporal offset
            //and clock drift.
            double time_sec = row.at<double>(0);
            double time_nsec = row.at<double>(1);
            Timestamp vicon_time = toTimestamp(time_sec, time_nsec);
            //object is in the form boxX or sensor payload (the camera)
            std::string object = row.at<std::string>(2);
            ObjectId object_id;
            if(object == "sensor_payload") {
                //camera
                object_id = 0;
            }
            else if(object.find("box") != std::string::npos) {
                //should be in boxX, so extract the last character as a number
                ObjectId box_id = object.back() - '0';
                CHECK_NE(box_id, 0); ///canot be 0, this is reserved for the camera
                object_id = box_id;
            }
            else {
                LOG(FATAL) << "Unknown object type: " << object;
            }

            double qx = row.at<double>(3);
            double qy = row.at<double>(4);
            double qz = row.at<double>(5);
            double qw = row.at<double>(6);

            double tx = row.at<double>(7);
            double ty = row.at<double>(8);
            double tz = row.at<double>(9);

            //transformation from object to world frame where world is the vicon frame
            //given as T_AO in dataset paper
            gtsam::Pose3 T_object_world(gtsam::Rot3(qw, qx, qy, qz), gtsam::Point3(tx, ty, tz));
            //world is the vicon frame
            // gtsam::Pose3 T_world_object = T_object_world.inverse();
            gtsam::Pose3 T_world_object = T_object_world;


            ViconPoseData vicon_data;
            vicon_data.timestamp = vicon_time;
            vicon_data.object_id = object_id;
            vicon_data.T_world_object = T_world_object;
            vicon_pose_data.push_back(vicon_data);
            // vicon_timestamp_buffer.addValue(vicon_time, std::make_pair(object_id, pose));

        }

        //convert camera times to map for lookup
        ThreadsafeTemporalBuffer<FrameId> camera_timestamp_buffer;
        for(Timestamp camera_times : times_) {
            CHECK(timestamp_frame_map_.exists(camera_times));
            camera_timestamp_buffer.addValue(camera_times, timestamp_frame_map_.at(camera_times));
        }
        //dataset size is number of optical flow images - we want one less camera image!!
        CHECK_EQ(camera_timestamp_buffer.size(), dataset_size_ - 1);
        const auto earliest_camera_timestamp = camera_timestamp_buffer.getOldestTimestamp();
        const auto latest_camera_timestamp = camera_timestamp_buffer.getNewestTimestamp();

        //this is very slow!!!
        for(const auto& vicon_data : vicon_pose_data) {
            const ObjectId object_id = vicon_data.object_id;
            const gtsam::Pose3& T_world_object = vicon_data.T_world_object;
            const Timestamp& vicon_timestamp = vicon_data.timestamp;

            // LOG(INFO) << "Found object id " << object_id;

            //dont include if less than first or greater than last camera time
            if (vicon_timestamp < earliest_camera_timestamp || vicon_timestamp > latest_camera_timestamp) {
                continue;
            }



            //get closest camera timestamp to the vicon timestamp - this is the value we want to interpolate
            //to as as this will be the timestamp of the frame we actually use
            //the min delta should really be the delta between camera frames. According to the associated RA-L
            //paper the rate of the RGBD sensor is 30Hz = 0.033
            //the rate of the vicon system is 200Hz
            constexpr static auto approx_rgbd_frame_rate = 0.033;
            FrameId frame_id;
            Timestamp camera_timestamp;
            if(!camera_timestamp_buffer.getNearestValueToTime(vicon_timestamp, approx_rgbd_frame_rate, &frame_id, &camera_timestamp)) {
                continue;
                //TODO: throw warning?
            }

            if (fpEqual(camera_timestamp, vicon_timestamp)) {
                //TODO: found exact match!!!!
                // DLOG(INFO) << "Exact mattch found for frame " << frame_id << " " << camera_timestamp << " " << vicon_timestamp << " object id: " << object_id;

                if(!ground_truth_packets.exists(frame_id)) {
                    GroundTruthInputPacket gt_packet;
                    gt_packet.frame_id_ = frame_id;
                    gt_packet.timestamp_ = camera_timestamp;
                    ground_truth_packets.insert2(frame_id, gt_packet);
                }

                if(!temp_object_poses.exists(frame_id)) {
                    temp_object_poses.insert2(frame_id, std::vector<ObjectPoseGT>{});
                }

                GroundTruthInputPacket& gt_packet = ground_truth_packets.at(frame_id);

                std::vector<ObjectPoseGT>& tmp_object_vector = temp_object_poses.at(frame_id);

                const static gtsam::Pose3 camera_to_world(gtsam::Rot3::RzRyRx(1, 0, 1), gtsam::traits<gtsam::Point3>::Identity());

                if(object_id == 0) {
                    //if object is sensor payload we need to apply extra transforms to get T_world_depthcam from T_world_object
                    //in this case T_world_object = T_world_apparatus (T_OA) as the object is the apparatus frame measured from the vicon
                    //T_world_depthcam =  T_world_apparatus * T_apparatus_leftstereo * T_leftstereo_depthcam
                    //where T_apparatus_leftstereo is given from the vicon.yaml config
                    //and T_leftstereo_depthcam is constructed from the kalibr config
                    gt_packet.X_world_ = T_world_object * T_apparatus_depthcam_;
                     //now in camera convention!!!
                    // gt_packet.X_world_ = camera_to_world.inverse() * gt_packet.X_world_;
                }
                else {
                    ObjectPoseGT object_pose_gt;
                    object_pose_gt.frame_id_ = frame_id;
                    object_pose_gt.object_id_ = object_id;
                    //if this is object, the transform is the object as seen in the vicon frame
                    //we want it in the depth sensor frame
                    //T_world_object -> T_vicon_object
                    object_pose_gt.L_world_ = T_world_object;
                    // object_pose_gt.L_world_ = camera_to_world.inverse() * object_pose_gt.L_world_;
                    //cannot set L_camera yet as we might not have the camera pose

                    //if object is already in the vector dont add
                    //this is 'wrong' but currently we dont interpolate the gt data
                    auto it_this = std::find_if(tmp_object_vector.begin(), tmp_object_vector.end(),
                        [=](const ObjectPoseGT& gt_object) { return gt_object.object_id_ == object_id; });
                    if(it_this == tmp_object_vector.end()) {
                        tmp_object_vector.push_back(object_pose_gt);
                    }
                }
            }
            else {
                LOG(FATAL) << "Currently expecting vicon and camera tiemstamp data to have at least one synched timestamp per frame!!";
            }

            //it seems that everything is very fast so (maybe a hack?) we'll just take the cloestst timestamp


            // //get closest timestamp before and after
            // //there will be an edge case on the first and last valid vicon timestamp
            // //ie. the first one that is > earliest_camera_timestamp and the last one that is
            // //smaller than latest_camera_timestamp as there will bot be a camera timestamp
            // //either side of this one!
            // Timestamp camera_timestamp_before, camera_timestamp_after; //before and after the queried vicon value
            // FrameId frame_id_before, frame_id_after;
            // bool has_camera_value_before = camera_timestamp_buffer.getValueAtOrBeforeTime(
            //     vicon_timestamp,
            //     &camera_timestamp_before,
            //     &frame_id_before
            // );

            // bool has_camera_value_after = camera_timestamp_buffer.getValueAtOrBeforeTime(
            //     vicon_timestamp,
            //     &camera_timestamp_after,
            //     &frame_id_after
            // );

            // if(has_camera_value_before) {
            //     CHECK_LT(camera_timestamp_before, vicon_timestamp);
            //     CHECK_LT(camera_timestamp_before, camera_timestamp);
            // }

            // if(has_camera_value_after) {
            //     CHECK_GT(camera_timestamp_after, vicon_timestamp);
            //     CHECK_GT(camera_timestamp_after, camera_timestamp);
            // }




        }

        //assumes we get one for every frame!!!!?
        FrameId previous_frame = 0;
        for(auto& [frame_id, gt_packet] : ground_truth_packets) {
            //get object vector for the tmp list
            auto& unprocessed_objects = temp_object_poses.at(gt_packet.frame_id_);
            for(auto& object_pose_gt : unprocessed_objects) {
                //TODO: I think this is wrong!!
                object_pose_gt.L_camera_ = gt_packet.X_world_.inverse() * object_pose_gt.L_world_;
                CHECK_EQ(frame_id, object_pose_gt.frame_id_);
            }

            gt_packet.object_poses_ = unprocessed_objects;

            if(frame_id > 0) {
                const GroundTruthInputPacket& previous_gt_packet = ground_truth_packets.at(frame_id - 1);
                gt_packet.calculateAndSetMotions(previous_gt_packet);
                CHECK_EQ(frame_id - 1, previous_frame) << "Frames are not in ascending order!!!";
            }

            previous_frame = frame_id;
        }

        CHECK_EQ(ground_truth_packets.size(), times_.size());
    }

    void setIntrisicsAndTransforms() {
        //kalirb file gives the rigid body transforms from the sensors to the apparatus frame
        YamlParser yaml_parser(kalibr_file_path_);

        std::vector<double> v_cam1_cam0;
        yaml_parser.getNestedYamlParam("cam1", "T_cn_cnm1", &v_cam1_cam0);
        //transform from cam1 into camera 0 frame
        gtsam::Pose3 T_cam0_cam1 = utils::poseVectorToGtsamPose3(v_cam1_cam0).inverse();

        std::vector<double> v_cam2_cam1;
        yaml_parser.getNestedYamlParam("cam2_undistort", "T_cn_cnm1", &v_cam2_cam1);
        //transform from cam2 into camera 1 frame
        gtsam::Pose3 T_cam1_cam2 = utils::poseVectorToGtsamPose3(v_cam2_cam1).inverse();

        //transformation from cam2 INTO cam0
        //in this case camera2 is the RGBD camera and cam0 is the stereo left camera
        gtsam::Pose3 T_cam0_cam2 = T_cam0_cam1 * T_cam1_cam2;
        T_leftstereo_depthcam_ = T_cam0_cam2;

        //now load apparaturs to left camera (cam0) transform to put cam0 into the sensor frame (or A)
        YamlParser vicon_yaml_parser(vicon_calibration_file_path_);
        std::vector<double> v_apparatus_left;
        vicon_yaml_parser.getYamlParam("T_apparatus_left", &v_apparatus_left);
        //transformation from cam0 (left) into the apparatus (A) frame
        gtsam::Pose3 T_apparatus_cam0 = utils::poseVectorToGtsamPose3(v_apparatus_left);
        T_apparatus_leftstereo_ = T_apparatus_cam0;
        //transform of cam2 (depth camera) into the apparatus (A) frame
        //we can use the premultiply this with the measurement (inverse) from vicon.csv
        //to put the vicon sensor pyload measurement into the depth frame which will match with our visual odometry
        gtsam::Pose3 T_apparatus_cam2 = T_apparatus_cam0 * T_cam0_cam2;
        T_apparatus_depthcam_ = T_apparatus_cam2;



        //NOTE: expect the rgbd camera to always be cam2
        CameraParams::IntrinsicsCoeffs intrinsics;
        std::vector<double> intrinsics_v;
        yaml_parser.getNestedYamlParam("cam2_undistort", "intrinsics", &intrinsics_v);
        CHECK_EQ(intrinsics_v.size(), 4u);
        intrinsics.resize(4u);
        // Move elements from one to the other.
        std::copy_n(std::make_move_iterator(intrinsics_v.begin()),
                    intrinsics.size(),
                    intrinsics.begin());

        CameraParams::DistortionCoeffs distortion;
        yaml_parser.getNestedYamlParam("cam2_undistort", "distortion_coeffs", &distortion);

        std::vector<int> resolution;
        yaml_parser.getNestedYamlParam("cam2_undistort", "resolution", &resolution);
        CHECK_EQ(resolution.size(), 2);
        cv::Size image_size(resolution[0], resolution[1]);

        std::string distortion_model, camera_model;
        yaml_parser.getNestedYamlParam("cam2_undistort", "distortion_model", &distortion_model);
        yaml_parser.getNestedYamlParam("cam2_undistort", "camera_model", &camera_model);
        auto model = CameraParams::stringToDistortion(distortion_model, camera_model);

        rgbd_camera_params_ = CameraParams(
            intrinsics,
            distortion,
            image_size,
            model
        );

        LOG(INFO) << rgbd_camera_params_.toString();

    }

    void associateGTWithObject(const cv::Mat& instance_mask, const cv::Mat& depth, FrameId frame_id, cv::Mat& relabelled_mask) const {
        const GroundTruthInputPacket& gt_packet = getGtPacket(frame_id);
        const ObjectIds gt_object_ids = gt_packet.getObjectIds();

        ObjectIds object_ids = vision_tools::getObjectLabels(instance_mask);
        const size_t n = object_ids.size();

        instance_mask.copyTo(relabelled_mask);

        if(n == 0) {
            return;
        }

        //instance id (track id -> all keypoints for that cluster)
        std::set<ObjectId> instance_id_set;
        gtsam::FastMap<ObjectId, Keypoints> instance_kps;
        gtsam::FastMap<ObjectId, Landmarks> instance_lmks;

        Camera camera(rgbd_camera_params_);

        //sample sparsely across image and fill instance_kps
        int step = 3;
        for (int i = 0; i < instance_mask.rows - step; i = i + step)
        {
            for (int j = 0; j < instance_mask.cols - step; j = j + step)
            {
                ObjectId instance_label = instance_mask.at<ObjectId>(i, j);
                if(instance_label == background_label) {
                    continue;
                }
                CHECK_NE(instance_label, background_label);

                instance_id_set.insert(instance_label);

                Keypoint keypoint(j, i);

                if(!instance_kps.exists(instance_label)) {
                    instance_kps.insert2(instance_label, Keypoints{});
                }
                instance_kps.at(instance_label).push_back(keypoint);

                //lmk in camera frame
                const Depth d = functional_keypoint::at<Depth>(keypoint, depth);
                Landmark lmk;
                camera.backProject(keypoint, d, &lmk);

                if(!instance_lmks.exists(instance_label)) {
                    instance_lmks.insert2(instance_label, Landmarks{});
                }
                instance_lmks.at(instance_label).push_back(lmk);
            }
        }

        std::vector<int> instance_ids(instance_id_set.begin(), instance_id_set.end());
        CHECK_EQ(instance_ids.size(), n);

        //want to assign instance_ids to gt labels
        const size_t m = gt_object_ids.size();

        Eigen::MatrixXd cost;
        cost.resize(n, m);

        for(size_t rows = 0; rows < n; rows++) {
            int instance_id = instance_ids.at(rows);
            // all lmks in the camera frame for the detected mask
            auto sampled_lmks = instance_lmks.at(instance_id);
            Landmark avg_lmk_camera = gtsam::mean(sampled_lmks);
            Landmark avg_lmk_world = gt_packet.X_world_ * avg_lmk_camera;

            for(size_t cols = 0; cols < m; cols++) {
                // LOG(INFO) << cols;
                ObjectId gt_object_id = gt_object_ids.at(cols);
                // LOG(INFO) << gt_object_id;

                ObjectPoseGT gt_object_pose;
                CHECK(gt_packet.getObject(gt_object_id, gt_object_pose));

                gtsam::Point3 gt_translation = gt_object_pose.L_world_.translation();

                //distance between the gt object pose (translation) and the cluster of detected points
                cost(rows, cols) = gtsam::distance3(gt_translation, avg_lmk_world);

            }
        }

        Eigen::VectorXi assignment;
        internal::HungarianAlgorithm().solve(cost, assignment);

        LOG(INFO) << " With " << n << " original ids and " << m << " gt ids";

        ObjectIds old_labels = instance_ids;
        ObjectIds new_labels;
        for(size_t i = 0; i < assignment.size(); i++) {
            int j = assignment[i];

            // //the i-th object is assignment to the j-th cluster
            // ObjectId instance_id = instance_ids.at(i);
            int assigned_object_id = gt_object_ids.at(j);
            new_labels.push_back(assigned_object_id);

            // LOG(INFO) << "Relabelled " << instance_id << " to " << assigned_object_id;

            // const cv::Mat object_mask = relabelled_mask == instance_id;
            // relabelled_mask.setTo(cv::Scalar(assigned_object_id), object_mask);
        }

        vision_tools::relabelMasks(instance_mask, relabelled_mask, old_labels, new_labels);

    }



private:
    const std::string rgbd_folder_path_;
    const std::string instance_masks_folder_;
    const std::string optical_flow_folder_path_;
    const std::string vicon_file_path_; //vicon measurement (.csv) file
    const std::string kalibr_file_path_;
    const std::string vicon_calibration_file_path_; //vicon calibration (.yaml) file giving transform between left stereo and apparatus

    std::vector<std::string> rgb_image_paths_; //index from 0 to match the naming convention of the dataset
    std::vector<std::string> aligned_depth_image_paths_;
    std::vector<Timestamp> times_; //loaded from rgbd.csv file and associated with rgbd images. Should be the same length as rgb/aligned depth files
    //timestamps between vicon (gt) and camera (from rgbd) are not synchronized!!
    gtsam::FastMap<Timestamp, FrameId> timestamp_frame_map_;

    std::vector<std::string> optical_flow_image_paths_;
    std::vector<std::string> instance_masks_image_paths_;

    GroundTruthPacketMap ground_truth_packets_;
    size_t dataset_size_; //set in setGroundTruthPacket. Reflects the number of files in the /optical_flow folder which is one per frame

    //below are set in the setIntrisicsAndTransforms
    CameraParams rgbd_camera_params_;
    gtsam::Pose3 T_apparatus_leftstereo_; //T_AL or the transform from the left stereo to the apparaturs frame, as given in vicon.yaml
    gtsam::Pose3 T_leftstereo_depthcam_; //T_LD or the transfrom from the depthcam to the left stereo. Given in the kalibr.yaml
    gtsam::Pose3 T_apparatus_depthcam_; //T_AL * T_LD

};

struct OMMTimestampLoader : public TimestampBaseLoader {

    OMDAllLoader::Ptr loader_;

    OMMTimestampLoader(OMDAllLoader::Ptr loader) : loader_(CHECK_NOTNULL(loader)) {}
    std::string getFolderName() const override { return ""; }

    size_t size() const override {
        return loader_->size();
    }

    double getItem(size_t idx) override {
        return loader_->getTimestamp(idx);
    }
};


OMDDataLoader::OMDDataLoader(const fs::path& dataset_path) : OMDDatasetProvider(dataset_path) {
    LOG(INFO) << "Starting OMDDataLoader with path" << dataset_path;

    //this would go out of scope but we capture it in the functional loaders
    auto loader = std::make_shared<OMDAllLoader>(dataset_path);
    auto timestamp_loader = std::make_shared<OMMTimestampLoader>(loader);

    left_camera_params_ = loader->getLeftCameraParams();

    CHECK(getCameraParams());

    auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
        [loader](size_t idx) {
            return loader->getRGB(idx);
        }
    );

    auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
        [loader](size_t idx) {
            return loader->getOpticalFlow(idx);
        }
    );

    auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
        [loader](size_t idx) {
            return loader->getDepthImage(idx);
        }
    );

    auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
        [loader](size_t idx) {
            return loader->getInstanceMask(idx);
        }
    );


    auto gt_loader = std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
        [loader](size_t idx) {
            return loader->getGtPacket(idx);
        }
    );

    this->setLoaders(
        timestamp_loader,
        rgb_loader,
        optical_flow_loader,
        depth_loader,
        instance_mask_loader,
        gt_loader
    );

    auto callback = [&](size_t frame_id,
        Timestamp timestamp,
        cv::Mat rgb,
        cv::Mat optical_flow,
        cv::Mat depth,
        cv::Mat instance_mask,
        GroundTruthInputPacket gt_object_pose_gt) -> bool
    {
        CHECK_EQ(timestamp, gt_object_pose_gt.timestamp_);

        CHECK(ground_truth_packet_callback_);
        if(ground_truth_packet_callback_) ground_truth_packet_callback_(gt_object_pose_gt);

        ImageContainer::Ptr image_container = nullptr;
        image_container = ImageContainer::Create(
                timestamp,
                frame_id,
                ImageWrapper<ImageType::RGBMono>(rgb),
                ImageWrapper<ImageType::Depth>(depth),
                ImageWrapper<ImageType::OpticalFlow>(optical_flow),
                ImageWrapper<ImageType::MotionMask>(instance_mask));
        CHECK(image_container);
        CHECK(image_container_callback_);
        if(image_container_callback_) image_container_callback_(image_container);
        return true;
    };

    this->setCallback(callback);

    setStartingFrame(1u); ///have to start at at least 1 so we can index backwards with optical flow
}

} //dyno
