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
        vicon_file_path_(file_path + "/vicon.csv")
    {
        throwExceptionIfPathInvalid(rgbd_folder_path_);
        throwExceptionIfPathInvalid(instance_masks_folder_);
        throwExceptionIfPathInvalid(optical_flow_folder_path_);
        throwExceptionIfPathInvalid(vicon_file_path_);

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
            // LOG(INFO) << frame_num;

            Timestamp time = (Timestamp)time_sec + (Timestamp)time_nsec / 1e+9;
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

        //firstly map the vicon timestamp system to frame ids

        gtsam::FastMap<FrameId, std::vector<ObjectPoseGT>> temp_object_poses;

        FrameId frame_id = 0;
        for(it; it != csv_reader.end(); it++) {
            const auto row = *it;
            //Though the stereo camera, RGB-D camera, and IMU were recorded on the same machine,
            //they are not hardware synchronized. The Vicon was recorded on a separate system with an unknown temporal offset
            //and clock drift.
            // double time_sec = row.at<double>(0);
            // double time_nsec = row.at<double>(1);

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

            // Timestamp time = (Timestamp)time_sec + (Timestamp)time_nsec / 1e+9;

            if(!ground_truth_packets.exists(frame_id)) {
                GroundTruthInputPacket gt_packet;
                gt_packet.frame_id_ = frame_id;
                // gt_packet.timestamp_ = time;
                ground_truth_packets.insert2(frame_id, gt_packet);
            }

        //     if(!temp_object_poses.exists(frame_id)) {
        //         temp_object_poses.insert2(frame_id, std::vector<ObjectPoseGT>{});
        //     }

        //     std::vector<ObjectPoseGT>& tmp_object_vector = temp_object_poses.at(frame_id);

        //     GroundTruthInputPacket& gt_packet = ground_truth_packets.at(frame_id);

        //     if(frame_id > 0) {
        //         //check we have the previous frame!! (ie.e we process in order!)
        //         CHECK(ground_truth_packets.exists(frame_id-1));
        //     }

        //     //values should already be set when we add the gt_packet to the map earlier
        //     CHECK_EQ(gt_packet.frame_id_,frame_id);
        //     CHECK_EQ(gt_packet.timestamp_,time);

        //     if(object_id == 0) {
        //         gt_packet.X_world_ = pose;
        //     }
        //     else {
        //         ObjectPoseGT object_pose_gt;
        //         object_pose_gt.frame_id_ = frame_id;
        //         object_pose_gt.object_id_ = object_id;
        //         object_pose_gt.L_world_ = pose;
        //         //cannot set L_camera yet as we might not have the camera pose
        //         tmp_object_vector.push_back(object_pose_gt);
        //     }

        //     //want one less than the number of flow images
        //     //assume that we get all the ground truth timestamps in order!!!!!
        //     //need to exit here as we only load timestamps (timestamp_frame_map_)
        //     //up to the size of the dataset!!!
            if(ground_truth_packets.size() + 1 >= dataset_size_) {
                break;
            }

            frame_id++;

        }

        // for(auto& [frame_id, gt_packet] : ground_truth_packets) {
        //     //get object vector for the tmp list
        //     auto& unprocessed_objects = temp_object_poses.at(gt_packet.frame_id_);
        //     for(auto& object_pose_gt : unprocessed_objects) {
        //         object_pose_gt.L_camera_ = gt_packet.X_world_.inverse() * object_pose_gt.L_world_;
        //         CHECK_EQ(frame_id, object_pose_gt.frame_id_);
        //     }

        //     gt_packet.object_poses_ = unprocessed_objects;

        //     if(frame_id > 0) {
        //         const GroundTruthInputPacket& previous_gt_packet = ground_truth_packets.at(frame_id - 1);
        //         gt_packet.calculateAndSetMotions(previous_gt_packet);
        //     }
        // }

    }



private:
    const std::string rgbd_folder_path_;
    const std::string instance_masks_folder_;
    const std::string optical_flow_folder_path_;
    const std::string vicon_file_path_;

    std::vector<std::string> rgb_image_paths_; //index from 0 to match the naming convention of the dataset
    std::vector<std::string> aligned_depth_image_paths_;
    std::vector<Timestamp> times_; //loaded from rgbd.csv file and associated with rgbd images. Should be the same length as rgb/aligned depth files
    //timestamps between vicon (gt) and camera (from rgbd) are not synchronized!! Which to use!!!?
    std::map<Timestamp, FrameId> timestamp_frame_map_;

    std::vector<std::string> optical_flow_image_paths_;
    std::vector<std::string> instance_masks_image_paths_;

    GroundTruthPacketMap ground_truth_packets_;
    size_t dataset_size_; //set in setGroundTruthPacket. Reflects the number of files in the /optical_flow folder which is one per frame

};


OMDDataLoader::OMDDataLoader(const fs::path& dataset_path) : OMDDatasetProvider(dataset_path) {
    LOG(INFO) << "Starting OMDDataLoader with path" << dataset_path;

    //this would go out of scope but we capture it in the functional loaders
    auto loader = std::make_shared<OMDAllLoader>(dataset_path);
}

} //dyno
