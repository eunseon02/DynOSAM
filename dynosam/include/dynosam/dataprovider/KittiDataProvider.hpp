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

namespace dyno {

class KittiObjectPoseFolder : public dyno::DataFolder<GroundTruthInputPacket> {

public:
    KittiObjectPoseFolder() {}

    /**
     * @brief "object_pose.txt" as file name
     *
     * @return std::string
     */
    std::string getFolderName() const override { return "object_pose.txt"; }
    GroundTruthInputPacket getItem(size_t idx) override {
        return gt_packets.at(idx);
    }


private:
    /**
     * @brief Setup ifstream and read everything in the data vector
     *
     */
    void onPathInit() override {
        std::ifstream object_pose_stream((std::string)absolute_folder_path_, std::ios::in);

        std::vector<ObjectPoseGT> object_poses;

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
                object_poses.push_back(object_pose);
            }
        }
        object_pose_stream.close();
    }

    ObjectPoseGT constructObjectPoseGT(const std::vector<double>& obj_pose_gt) const {
        // FrameID ObjectID B1 B2 B3 B4 t1 t2 t3 r1
        // Where ti are the coefficients of 3D object location **t** in camera coordinates, and r1 is the Rotation around
        // Y-axis in camera coordinates.
        // B1-4 is 2D bounding box of object in the image, used for visualization.
        CHECK(obj_pose_gt.size() == 10);
        ObjectPoseGT object_pose;
        object_pose.frame_id = obj_pose_gt[0];
        object_pose.object_id = obj_pose_gt[1];


        double b1 = obj_pose_gt[2];
        double b2 = obj_pose_gt[3];
        double b3 = obj_pose_gt[4];
        double b4 = obj_pose_gt[5];

        //convert to cv::Rect where image starts top right
        cv::Point tl(b1, b2);
        cv::Point br(b3, b4);
        object_pose.bounding_box = cv::Rect(tl, br);

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

        // object_pose.pose = utils::cvMatToGtsamPose3(Pose);
        return object_pose;
    }

private:
    std::vector<GroundTruthInputPacket> gt_packets;

};

class KittiDataLoader : public DynoDatasetProvider<cv::Mat> {

public:
    KittiDataLoader(const fs::path& dataset_path) : DynoDatasetProvider<cv::Mat>(
        dataset_path,
        std::make_shared<DepthDataFolder>()
    ) {}


    bool constructFrame(size_t frame_id, cv::Mat rgb, Timestamp timestamp, cv::Mat depth) override {
        LOG(ERROR) << rgb.size();
        LOG(ERROR) << depth.size();
        LOG(ERROR) << frame_id << " " << timestamp;
        return true;
    }
};

} //dyno
