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

#pragma once

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"

#include "dynosam/frontend/FrontendInputPacket.hpp"

namespace dyno {

class KittiCameraPoseFolder : public dyno::DataFolder<gtsam::Pose3> {

public:
    DYNO_POINTER_TYPEDEFS(KittiCameraPoseFolder)

    std::string getFolderName() const override { return "pose_gt.txt"; }
    gtsam::Pose3 getItem(size_t idx) override {
        return gt_camera_poses_.at(idx);
    }

private:
    void onPathInit() override {
        std::ifstream object_pose_stream((std::string)getAbsolutePath(), std::ios::in);

        while (!object_pose_stream.eof())
        {
            std::string s;
            getline(object_pose_stream, s);
            if (!s.empty())
            {
                std::stringstream ss;
                ss << s;
                int t;
                ss >> t;
                cv::Mat cv_pose = cv::Mat::eye(4, 4, CV_64F);

                ss >> cv_pose.at<double>(0, 0) >> cv_pose.at<double>(0, 1) >> cv_pose.at<double>(0, 2) >>
                    cv_pose.at<double>(0, 3) >> cv_pose.at<double>(1, 0) >> cv_pose.at<double>(1, 1) >>
                    cv_pose.at<double>(1, 2) >> cv_pose.at<double>(1, 3) >> cv_pose.at<double>(2, 0) >>
                    cv_pose.at<double>(2, 1) >> cv_pose.at<double>(2, 2) >> cv_pose.at<double>(2, 3) >>
                    cv_pose.at<double>(3, 0) >> cv_pose.at<double>(3, 1) >> cv_pose.at<double>(3, 2) >>
                    cv_pose.at<double>(3, 3);

                gtsam::Pose3 pose = utils::cvMatToGtsamPose3(cv_pose);
                gt_camera_poses_.push_back(pose);
            }
        }
        object_pose_stream.close();
    }

    std::vector<gtsam::Pose3> gt_camera_poses_;

};

class KittiObjectPoseFolder : public dyno::DataFolder<GroundTruthInputPacket> {

public:
    DYNO_POINTER_TYPEDEFS(KittiObjectPoseFolder)
    KittiObjectPoseFolder(TimestampFile::Ptr timestamp_loader, KittiCameraPoseFolder::Ptr camera_pose_loader)
    : timestamp_loader_(timestamp_loader), camera_pose_loader_(camera_pose_loader) {}

    /**
     * @brief "object_pose.txt" as file name
     *
     * @return std::string
     */
    std::string getFolderName() const override { return "object_pose.txt"; }
    GroundTruthInputPacket getItem(size_t idx) override {
        //no guarantee (by ordering) that camera_pose_loader has been inited correctly before the get item so we add the data here and not in the onPathInit
        CHECK(camera_pose_loader_);
        CHECK(camera_pose_loader_->isAbsolutePathSet());

        const gtsam::Pose3 camera_pose = camera_pose_loader_->getItem(idx);
        const Timestamp timestamp = timestamp_loader_->getItem(idx);

        //  frame_id < nTimes - 1??
        GroundTruthInputPacket gt_packet;
        gt_packet.timestamp_ = timestamp;
        gt_packet.frame_id_ = idx;
        // add ground truths for this fid
        for (size_t i = 0; i < object_ids_vector_[idx].size(); i++)
        {
            gt_packet.object_poses_.push_back(object_poses_[object_ids_vector_[idx][i]]);
        // sanity check
        CHECK_EQ(gt_packet.object_poses_[i].frame_id_, idx);
        }
        gt_packet.X_world_ = camera_pose;

        //TODO: recalcualte every time?
        return gt_packet;
    }


private:
    /**
     * @brief Setup ifstream and read everything in the data vector
     *
     */
    void onPathInit() override {
        std::ifstream object_pose_stream((std::string)getAbsolutePath(), std::ios::in);

        //hmm we dont know the number of expected frames here...
        while (!object_pose_stream.eof())
        {
            std::string s;
            getline(object_pose_stream, s);
            if (!s.empty())
            {
                std::stringstream ss;
                ss << s;

                std::vector<double> object_pose_vec(10, 0);
                ss >> object_pose_vec[0] >> object_pose_vec[1] >> object_pose_vec[2] >> object_pose_vec[3] >> object_pose_vec[4] >> object_pose_vec[5] >>
                    object_pose_vec[6] >> object_pose_vec[7] >> object_pose_vec[8] >> object_pose_vec[9];

                ObjectPoseGT object_pose = constructObjectPoseGT(object_pose_vec);
                object_poses_.push_back(object_pose);
            }
        }
        object_pose_stream.close();

        //no guarantee that the timestamp loader onPathInit() has been called first (although in reality it will becuase) it is part of the default
        //folder structure
        //check that it has been initalised and then use the timestamps already loaded there to construct the gt packets
        CHECK(timestamp_loader_->isAbsolutePathSet()) << "Timestamp laoder should be already initalised so we can get the timestamps from it since they are preloaded!!";
        const size_t n_times = timestamp_loader_->size();
        CHECK(n_times > 0u);
        object_ids_vector_.resize(n_times); //only load up to the number of timestamps given
        for(size_t i = 0; i < object_poses_.size(); i++) {
            const ObjectPoseGT& obect_pose_gt = object_poses_[i];
            size_t frame_id = obect_pose_gt.frame_id_;
            if(frame_id >= object_ids_vector_.size()) {
                break;
            }

            object_ids_vector_[frame_id].push_back(i);
        }
    }

    ObjectPoseGT constructObjectPoseGT(const std::vector<double>& obj_pose_gt) const {
        // FrameID ObjectID B1 B2 B3 B4 t1 t2 t3 r1
        // Where ti are the coefficients of 3D object location **t** in camera coordinates, and r1 is the Rotation around
        // Y-axis in camera coordinates.
        // B1-4 is 2D bounding box of object in the image, used for visualization.
        CHECK(obj_pose_gt.size() == 10);
        ObjectPoseGT object_pose;
        object_pose.frame_id_ = obj_pose_gt[0];
        object_pose.object_id_ = obj_pose_gt[1];


        double b1 = obj_pose_gt[2];
        double b2 = obj_pose_gt[3];
        double b3 = obj_pose_gt[4];
        double b4 = obj_pose_gt[5];

        //convert to cv::Rect where image starts top right
        cv::Point tl(b1, b2);
        cv::Point br(b3, b4);
        object_pose.bounding_box_ = cv::Rect(tl, br);

        cv::Mat t(3, 1, CV_64FC1);
        t.at<double>(0) = obj_pose_gt[6];
        t.at<double>(1) = obj_pose_gt[7];
        t.at<double>(2) = obj_pose_gt[8];

        // from Euler to Rotation Matrix
        cv::Mat R(3, 3, CV_64FC1);

        // assign r vector
        double y = obj_pose_gt[9] + (3.1415926 / 2);  // +(3.1415926/2)
        double x = 0.0;
        double z = 0.0;

        // the angles are in radians.
        double cy = cos(y);
        double sy = sin(y);
        double cx = cos(x);
        double sx = sin(x);
        double cz = cos(z);
        double sz = sin(z);

        double m00, m01, m02, m10, m11, m12, m20, m21, m22;

        m00 = cy * cz + sy * sx * sz;
        m01 = -cy * sz + sy * sx * cz;
        m02 = sy * cx;
        m10 = cx * sz;
        m11 = cx * cz;
        m12 = -sx;
        m20 = -sy * cz + cy * sx * sz;
        m21 = sy * sz + cy * sx * cz;
        m22 = cy * cx;

        R.at<double>(0, 0) = m00;
        R.at<double>(0, 1) = m01;
        R.at<double>(0, 2) = m02;
        R.at<double>(1, 0) = m10;
        R.at<double>(1, 1) = m11;
        R.at<double>(1, 2) = m12;
        R.at<double>(2, 0) = m20;
        R.at<double>(2, 1) = m21;
        R.at<double>(2, 2) = m22;

        // construct 4x4 transformation matrix
        cv::Mat Pose = cv::Mat::eye(4, 4, CV_64F);
        Pose.at<double>(0, 0) = R.at<double>(0, 0);
        Pose.at<double>(0, 1) = R.at<double>(0, 1);
        Pose.at<double>(0, 2) = R.at<double>(0, 2);
        Pose.at<double>(0, 3) = t.at<double>(0);
        Pose.at<double>(1, 0) = R.at<double>(1, 0);
        Pose.at<double>(1, 1) = R.at<double>(1, 1);
        Pose.at<double>(1, 2) = R.at<double>(1, 2);
        Pose.at<double>(1, 3) = t.at<double>(1);
        Pose.at<double>(2, 0) = R.at<double>(2, 0);
        Pose.at<double>(2, 1) = R.at<double>(2, 1);
        Pose.at<double>(2, 2) = R.at<double>(2, 2);
        Pose.at<double>(2, 3) = t.at<double>(2);

        object_pose.L_camera_ = utils::cvMatToGtsamPose3(Pose);
        return object_pose;
    }

private:
    TimestampFile::Ptr timestamp_loader_;
    KittiCameraPoseFolder::Ptr camera_pose_loader_;
    std::vector<ObjectIds> object_ids_vector_;
    std::vector<ObjectPoseGT> object_poses_;
    std::vector<GroundTruthInputPacket> gt_packets_; //unused?
};


//additional loaders are depth, semantic mask, camera pose gt and gt input packet
class KittiDataLoader : public DynoDatasetProvider<cv::Mat, cv::Mat, gtsam::Pose3, GroundTruthInputPacket> {

public:

    struct Params {
        MaskType mask_type;

        //needed for input dispartity to depth conversion
        double base_line =  387.5744;
        double depth_scale_factor = 256.0;

        static Params fromYaml(const std::string& params_folder) {
            YamlParser yaml_parser(params_folder + "DatasetParams.yaml");

            Params params;

            std::string mask_type;
            yaml_parser.getYamlParam("mask_type", &mask_type);
            params.mask_type = maskTypeFromString(mask_type);

            yaml_parser.getYamlParam("base_line", &params.base_line);
            yaml_parser.getYamlParam("depth_scale_factor", &params.depth_scale_factor);
            return params;

        }
    };



    /**
     * @brief Construct a new Kitti Data Loader object
     *
     * Path should be the full path to the set of folders
     *  /depth
     *  /flow
     *  /image_0
     *  /motion
     *  /semantic
     *  /object_pose.txt
     *  /pose_gt.txt
     *  /times.txt
     *
     * @param dataset_path
     * @param params
     */
    KittiDataLoader(const fs::path& dataset_path, const Params& params) : DynoDatasetProvider<cv::Mat, cv::Mat, gtsam::Pose3, GroundTruthInputPacket>(
        dataset_path), params_(params)
    {
        TimestampFile::Ptr timestamp_file =  std::make_shared<TimestampFile>();
        RGBDataFolder::Ptr rgb_folder = std::make_shared<RGBDataFolder>();

        CHECK(timestamp_file);
        CHECK(rgb_folder);

        SegMaskFolder::Ptr mask_folder = nullptr;
        if(params.mask_type == MaskType::MOTION) {
            mask_folder = std::make_shared<MotionSegMaskFolder>(rgb_folder);
            LOG(INFO) << "Using MaskType::MOTION for loading mask";
        }
        else if(params.mask_type == MaskType::SEMANTIC_INSTANCE) {
            mask_folder = std::make_shared<InstantanceSegMaskFolder>(rgb_folder);
            LOG(INFO) << "Using MaskType::SEMANTIC_INSTANCE for loading mask";
        }
        else {
            LOG(FATAL) << "Unknown MaskType for KittiDataLoader";
        }
        CHECK_NOTNULL(mask_folder);

        KittiCameraPoseFolder::Ptr camera_pose_folder = std::make_shared<KittiCameraPoseFolder>();
        KittiObjectPoseFolder::Ptr object_pose_gt_folder = std::make_shared<KittiObjectPoseFolder>(
            timestamp_file, camera_pose_folder
        );



        this->setLoaders(
            timestamp_file,
            rgb_folder,
            //should really be called kitti optical flow and kitti depth data folder
            std::make_shared<OpticalFlowDataFolder>(),
            std::make_shared<DepthDataFolder>(),
            mask_folder,
            camera_pose_folder,
            object_pose_gt_folder
        );

        auto callback = [&](size_t frame_id,
            Timestamp timestamp,
            cv::Mat rgb,
            cv::Mat optical_flow,
            cv::Mat depth,
            cv::Mat instance_mask,
            gtsam::Pose3 camera_pose_gt,
            GroundTruthInputPacket gt_object_pose_gt) -> bool
        {
            CHECK(camera_pose_gt.equals(gt_object_pose_gt.X_world_));
            CHECK(timestamp == gt_object_pose_gt.timestamp_);

            CHECK(ground_truth_packet_callback_);
            if(ground_truth_packet_callback_) ground_truth_packet_callback_(gt_object_pose_gt);

            auto image_container = ImageContainer::Create(
                timestamp,
                frame_id,
                ImageWrapper<ImageType::RGBMono>(rgb),
                ImageWrapper<ImageType::Depth>(depth),
                ImageWrapper<ImageType::OpticalFlow>(optical_flow),
                ImageWrapper<ImageType::SemanticMask>(instance_mask)
            );


            CHECK(image_container_callback_);
            if(image_container_callback_) image_container_callback_(image_container);
            return true;
        };

        this->setCallback(callback);
    }

    ImageContainer::Ptr imageContainerPreprocessor(ImageContainer::Ptr image_container) override {
        if(!image_container->hasDepth()) {
            throw std::runtime_error("Cannot preprocess ImageContainer in Kitti dataset as no depth!");
        }

        const auto& base_line = params_.base_line;
        const auto& depth_scale_factor = params_.depth_scale_factor;

        const cv::Mat& const_disparity = image_container->getDepth();
        cv::Mat depth;
        const_disparity.copyTo(depth);
        for (int i = 0; i < const_disparity.rows; i++)
        {
            for (int j = 0; j < const_disparity.cols; j++)
            {
                if (const_disparity.at<double>(i, j) < 0) {
                    depth.at<double>(i, j) = 0;
                }
                else {
                    depth.at<double>(i, j) = base_line / (const_disparity.at<double>(i, j) / depth_scale_factor);
                }
            }
        }

        cv::Mat& disparity = image_container->get<ImageType::Depth>();
        depth.copyTo(disparity);
        depth.convertTo(depth, CV_64F);

        return image_container;
    }

private:
    const Params params_;


};

} //dyno
