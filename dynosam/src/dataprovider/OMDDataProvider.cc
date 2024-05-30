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

#include "dynosam/pipeline/ThreadSafeTemporalBuffer.hpp"

#include <glog/logging.h>
#include <filesystem>

namespace dyno {

class OMDAllLoader {

public:
    DYNO_POINTER_TYPEDEFS(OMDAllLoader)

    OMDAllLoader(const std::string& file_path)
    :   rgbd_folder_path_(file_path + "/rgbd"),
        instance_masks_folder_(file_path + "/instance_masks"),
        optical_flow_folder_path_(file_path + "/optical_flow"),
        vicon_file_path_(file_path + "/vicon.csv"),
        kalibr_file_path_(file_path + "/kalibr.yaml")
    {
        throwExceptionIfPathInvalid(rgbd_folder_path_);
        throwExceptionIfPathInvalid(instance_masks_folder_);
        throwExceptionIfPathInvalid(optical_flow_folder_path_);
        throwExceptionIfPathInvalid(vicon_file_path_);
        throwExceptionIfPathInvalid(kalibr_file_path_);

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

        setIntrisics();
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

        // associateDetectedBBWithObject(mask, idx, relabelled_mask);

        return mask;
    }

    cv::Mat getDepthImage(size_t idx) const {
        CHECK_LT(idx, aligned_depth_image_paths_.size());

        cv::Mat depth;
        loadDepth(aligned_depth_image_paths_.at(idx), depth);
        CHECK(!depth.empty());

        return depth;
    }

    const GroundTruthInputPacket& getGtPacket(size_t idx) const {
        return ground_truth_packets_.at(idx);
    }

    const CameraParams& getLeftCameraParams() const {
        return rgbd_camera_params;
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

        for(it; it != csv_reader.end(); it++) {
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


    void setGroundTruthPacketFromVicon(GroundTruthPacketMap& ground_truth_packets) {
        LOG(INFO) << "Loading object/camera pose gt from vicon: " << vicon_file_path_;
        std::ifstream file(vicon_file_path_);
        CsvReader csv_reader(file);
        auto it = csv_reader.begin();
        //skip header
        ++it;

        using ObjectIdPosePair = std::pair<ObjectId, gtsam::Pose3>;
        //map vicon timestamps to their closest camera id timestamp
        ThreadsafeTemporalBuffer<ObjectIdPosePair> vicon_timestamp_buffer;
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

            gtsam::Pose3 pose(gtsam::Rot3(qw, qx, qy, qz), gtsam::Point3(tx, ty, tz));
            vicon_timestamp_buffer.addValue(vicon_time, std::make_pair(object_id, pose));

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
        for(const auto& [vicon_timestamp, object_id_pose_pair] : vicon_timestamp_buffer.buffered_values()) {
            const ObjectId object_id = object_id_pose_pair.first;
            const gtsam::Pose3& pose = object_id_pose_pair.second;

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
                DLOG(INFO) << "Exact mattch found for frame " << frame_id << " " << camera_timestamp << " " << vicon_timestamp << " object id: " << object_id;

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

                if(object_id == 0) {
                    gt_packet.X_world_ = pose;
                }
                else {
                    ObjectPoseGT object_pose_gt;
                    object_pose_gt.frame_id_ = frame_id;
                    object_pose_gt.object_id_ = object_id;
                    object_pose_gt.L_world_ = pose;
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

    void setIntrisics() {
        YamlParser yaml_parser(kalibr_file_path_);

        //NOTE: expect the rgbd camera to always be cam2
        CameraParams::IntrinsicsCoeffs intrinsics;
        std::vector<double> intrinsics_v;
        yaml_parser.getNestedYamlParam("cam2", "intrinsics", &intrinsics_v);
        CHECK_EQ(intrinsics_v.size(), 4u);
        intrinsics.resize(4u);
        // Move elements from one to the other.
        std::copy_n(std::make_move_iterator(intrinsics_v.begin()),
                    intrinsics.size(),
                    intrinsics.begin());

        std::vector<int> resolution;
        yaml_parser.getNestedYamlParam("cam2", "resolution", &resolution);
        CHECK_EQ(resolution.size(), 2);
        cv::Size image_size(resolution[0], resolution[1]);

        std::string distortion_model, camera_model;
        yaml_parser.getNestedYamlParam("cam2", "distortion_model", &distortion_model);
        yaml_parser.getNestedYamlParam("cam2", "camera_model", &camera_model);
        auto model = CameraParams::stringToDistortion(distortion_model, camera_model);

        rgbd_camera_params = CameraParams(
            intrinsics,
            CameraParams::DistortionCoeffs{0, 0, 0,0},
            image_size,
            model
        );

    }

    // void associateDetectedBBWithObject(const cv::Mat& instance_mask, FrameId frame_id, cv::Mat& relabelled_mask) const {
    //     const GroundTruthInputPacket& gt_packet = getGtPacket(frame_id);
    //     ObjectIds object_ids = vision_tools::getObjectLabels(instance_mask);
    //     const size_t n = object_ids.size();

    //     instance_mask.copyTo(relabelled_mask);

    //     if(n == 0) {
    //         return;
    //     }

    //     //get
    // }



private:
    const std::string rgbd_folder_path_;
    const std::string instance_masks_folder_;
    const std::string optical_flow_folder_path_;
    const std::string vicon_file_path_;
    const std::string kalibr_file_path_;

    std::vector<std::string> rgb_image_paths_; //index from 0 to match the naming convention of the dataset
    std::vector<std::string> aligned_depth_image_paths_;
    std::vector<Timestamp> times_; //loaded from rgbd.csv file and associated with rgbd images. Should be the same length as rgb/aligned depth files
    //timestamps between vicon (gt) and camera (from rgbd) are not synchronized!!
    gtsam::FastMap<Timestamp, FrameId> timestamp_frame_map_;

    std::vector<std::string> optical_flow_image_paths_;
    std::vector<std::string> instance_masks_image_paths_;

    GroundTruthPacketMap ground_truth_packets_;
    size_t dataset_size_; //set in setGroundTruthPacket. Reflects the number of files in the /optical_flow folder which is one per frame
    CameraParams rgbd_camera_params;

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
}

} //dyno
