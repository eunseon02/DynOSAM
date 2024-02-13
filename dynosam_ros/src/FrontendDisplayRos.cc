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

#include <dynosam/visualizer/ColourMap.hpp>
#include <dynosam/frontend/RGBDInstance-Definitions.hpp>
#include <dynosam/utils/SafeCast.hpp>

#include "dynosam_ros/FrontendDisplayRos.hpp"
#include "dynosam_ros/RosUtils.hpp"

#include <glog/logging.h>
#include "rclcpp/qos.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>

#include "geometry_msgs/msg/transform_stamped.hpp"

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/memory.h>

#include <string>

namespace dyno {

FrontendDisplayRos::FrontendDisplayRos(const DisplayParams params, rclcpp::Node::SharedPtr node) : DisplayRos(params), node_(CHECK_NOTNULL(node)) {

    const rclcpp::QoS& sensor_data_qos = rclcpp::SensorDataQoS();
    tracking_image_pub_ = image_transport::create_publisher(node.get(), "~/tracking_image");
    static_tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("~/static", 1);
    dynamic_tracked_points_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("~/dynamic", 1);
    odometry_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("~/odom", 1);
    object_pose_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("~/composed_object_poses", 1);
    object_pose_path_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("~/composed_object_paths", 1);
    odometry_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("~/odom_path", 2);
    object_motion_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("~/object_motions", 1);
    object_bbx_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("~/object_bbx", 1);

    gt_odometry_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("~/ground_truth/odom", 1);
    gt_object_pose_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("~/ground_truth/object_poses", 1);
    gt_object_path_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("~/ground_truth/object_paths", 1);
    gt_odom_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("~/ground_truth/odom_path", 1);
    gt_bounding_box_pub_= image_transport::create_publisher(node.get(), "~/ground_truth/bounding_boxes");

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);


}

void FrontendDisplayRos::spinOnce(const FrontendOutputPacketBase::ConstPtr& frontend_output) {
    //TODO: does frontend or backend publish tf transform?
    publishOdometry(frontend_output->T_world_camera_, frontend_output->getTimestamp());
    publishDebugImage(frontend_output->debug_image_);

    if(frontend_output->gt_packet_) {
        const auto& rgb_image = frontend_output->frame_.tracking_images_.get<ImageType::RGBMono>();
        publishGroundTruthInfo(frontend_output->getTimestamp(), frontend_output->gt_packet_.value(), rgb_image);
    }

    RGBDInstanceOutputPacket::ConstPtr rgbd_output = safeCast<FrontendOutputPacketBase, RGBDInstanceOutputPacket>(frontend_output);
    if(rgbd_output) {
        processRGBDOutputpacket(rgbd_output);
        //TODO:currently only with RGBDInstanceOutputPacket becuase camera poses is in RGBDInstanceOutputPacket but should be in base (FrontendOutputPacketBase)
        publishOdometryPath(odometry_path_pub_,rgbd_output->camera_poses_, frontend_output->getTimestamp());
    }

}



void FrontendDisplayRos::processRGBDOutputpacket(const RGBDInstanceOutputPacket::ConstPtr& rgbd_frontend_output) {
    CHECK(rgbd_frontend_output);
    // publishPointCloud(static_tracked_points_pub_, rgbd_frontend_output->static_landmarks_);
    // publishPointCloud(dynamic_tracked_points_pub_, rgbd_frontend_output->dynamic_landmarks_);

    {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        for(const auto& status_estimate : rgbd_frontend_output->static_landmarks_) {
            const LandmarkStatus& status =  status_estimate.first;
            const LandmarkEstimate& estimate = status_estimate.second;
            const Landmark& lmk = rgbd_frontend_output->T_world_camera_ * estimate.second;

            pcl::PointXYZRGB pt;
            if(status.label_ == background_label) {
                // publish static lmk's as white
                pt = pcl::PointXYZRGB(lmk(0), lmk(1), lmk(2), 0, 0, 0);
            }
            else {
                const cv::Scalar colour = ColourMap::getObjectColour(status.label_);
                pt = pcl::PointXYZRGB(lmk(0), lmk(1), lmk(2), colour(0), colour(1), colour(2));
            }
            cloud.points.push_back(pt);
        }


        sensor_msgs::msg::PointCloud2 pc2_msg;
        pcl::toROSMsg(cloud, pc2_msg);
        pc2_msg.header.frame_id = params_.world_frame_id_;
        static_tracked_points_pub_->publish(pc2_msg);
    }

    std::map<ObjectId, pcl::PointCloud<pcl::PointXYZ> > clouds_per_obj;
    {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        for(const auto& status_estimate :  rgbd_frontend_output->dynamic_landmarks_) {
            const LandmarkStatus& status =  status_estimate.first;
            const LandmarkEstimate& estimate = status_estimate.second;
            const Landmark& lmk = rgbd_frontend_output->T_world_camera_ * estimate.second;

            pcl::PointXYZRGB pt;
            pcl::PointXYZ pt_xyz;
            if(status.label_ == background_label) {
                // publish static lmk's as white
                pt = pcl::PointXYZRGB(lmk(0), lmk(1), lmk(2), 0, 0, 0);
            }
            else {
                const cv::Scalar colour = ColourMap::getObjectColour(status.label_);
                pt = pcl::PointXYZRGB(lmk(0), lmk(1), lmk(2), colour(0), colour(1), colour(2));
                pt_xyz = pcl::PointXYZ(lmk(0), lmk(1), lmk(2));
            }
            cloud.points.push_back(pt);
            clouds_per_obj[status.label_].push_back(pt_xyz);
        }


        sensor_msgs::msg::PointCloud2 pc2_msg;
        pcl::toROSMsg(cloud, pc2_msg);
        pc2_msg.header.frame_id = params_.world_frame_id_;
        dynamic_tracked_points_pub_->publish(pc2_msg);
    }

    {
        visualization_msgs::msg::MarkerArray object_bbx_marker_array;
        static visualization_msgs::msg::Marker delete_marker;
        delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        object_bbx_marker_array.markers.push_back(delete_marker);

        for (const auto& [object_id, obj_cloud] : clouds_per_obj){
            pcl::PointXYZ centroid;
            pcl::computeCentroid(obj_cloud, centroid);

            const cv::Scalar colour = ColourMap::getObjectColour(object_id);

            visualization_msgs::msg::Marker txt_marker;
            txt_marker.header.frame_id = "world";
            txt_marker.ns = "object_id";
            txt_marker.id = object_id;
            txt_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            txt_marker.action = visualization_msgs::msg::Marker::ADD;
            txt_marker.header.stamp = node_->now();
            txt_marker.scale.z = 2.0;
            txt_marker.color.r = colour(0)/255.0;
            txt_marker.color.g = colour(1)/255.0;
            txt_marker.color.b = colour(2)/255.0;
            txt_marker.color.a = 1;
            txt_marker.text = "obj "+std::to_string(object_id);
            txt_marker.pose.position.x = centroid.x;
            txt_marker.pose.position.y = centroid.y-2.0;
            txt_marker.pose.position.z = centroid.z-1.0;
            object_bbx_marker_array.markers.push_back(txt_marker);

            pcl::PointCloud<pcl::PointXYZ>::Ptr obj_cloud_ptr = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ> >(obj_cloud);
            pcl::PointCloud<pcl::PointXYZ> filtered_cloud;

            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outlier_remover;
            outlier_remover.setInputCloud(obj_cloud_ptr);
            outlier_remover.setMeanK(100);
            outlier_remover.setStddevMulThresh(1.0);
            outlier_remover.filter(filtered_cloud);

            pcl::MomentOfInertiaEstimation<pcl::PointXYZ> bbx_extractor;
            bbx_extractor.setInputCloud(pcl::make_shared<pcl::PointCloud<pcl::PointXYZ> >(filtered_cloud));
            bbx_extractor.compute();

            pcl::PointXYZ min_point_AABB;
            pcl::PointXYZ max_point_AABB;
            bbx_extractor.getAABB(min_point_AABB, max_point_AABB);

            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world";
            marker.ns = "object_bbx";
            marker.id = object_id;
            marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.header.stamp = node_->now();
            marker.scale.x = 0.1;

            marker.pose.orientation.x = 0;
            marker.pose.orientation.y = 0;
            marker.pose.orientation.z = 0;
            marker.pose.orientation.w = 1;

            marker.color.r = colour(0)/255.0;
            marker.color.g = colour(1)/255.0;
            marker.color.b = colour(2)/255.0;
            marker.color.a = 1;

            for (pcl::PointXYZ this_line_list_point : findLineListPointsFromAABBMinMax(min_point_AABB, max_point_AABB)){
                geometry_msgs::msg::Point p;
                p.x = this_line_list_point.x;
                p.y = this_line_list_point.y;
                p.z = this_line_list_point.z;
                marker.points.push_back(p);
            }

            // // bottom
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = min_point_AABB.x;
            //     p0.y = min_point_AABB.y;
            //     p0.z = min_point_AABB.z;
            //     p1.x = max_point_AABB.x;
            //     p1.y = min_point_AABB.y;
            //     p1.z = min_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = min_point_AABB.x;
            //     p0.y = min_point_AABB.y;
            //     p0.z = min_point_AABB.z;
            //     p1.x = min_point_AABB.x;
            //     p1.y = max_point_AABB.y;
            //     p1.z = min_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = max_point_AABB.x;
            //     p0.y = min_point_AABB.y;
            //     p0.z = min_point_AABB.z;
            //     p1.x = max_point_AABB.x;
            //     p1.y = max_point_AABB.y;
            //     p1.z = min_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = min_point_AABB.x;
            //     p0.y = max_point_AABB.y;
            //     p0.z = min_point_AABB.z;
            //     p1.x = max_point_AABB.x;
            //     p1.y = max_point_AABB.y;
            //     p1.z = min_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }

            // // top
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = min_point_AABB.x;
            //     p0.y = min_point_AABB.y;
            //     p0.z = max_point_AABB.z;
            //     p1.x = max_point_AABB.x;
            //     p1.y = min_point_AABB.y;
            //     p1.z = max_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = min_point_AABB.x;
            //     p0.y = min_point_AABB.y;
            //     p0.z = max_point_AABB.z;
            //     p1.x = min_point_AABB.x;
            //     p1.y = max_point_AABB.y;
            //     p1.z = max_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = max_point_AABB.x;
            //     p0.y = min_point_AABB.y;
            //     p0.z = max_point_AABB.z;
            //     p1.x = max_point_AABB.x;
            //     p1.y = max_point_AABB.y;
            //     p1.z = max_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = min_point_AABB.x;
            //     p0.y = max_point_AABB.y;
            //     p0.z = max_point_AABB.z;
            //     p1.x = max_point_AABB.x;
            //     p1.y = max_point_AABB.y;
            //     p1.z = max_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }

            // // vertical lines
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = min_point_AABB.x;
            //     p0.y = min_point_AABB.y;
            //     p0.z = min_point_AABB.z;
            //     p1.x = min_point_AABB.x;
            //     p1.y = min_point_AABB.y;
            //     p1.z = max_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = max_point_AABB.x;
            //     p0.y = min_point_AABB.y;
            //     p0.z = min_point_AABB.z;
            //     p1.x = max_point_AABB.x;
            //     p1.y = min_point_AABB.y;
            //     p1.z = max_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = min_point_AABB.x;
            //     p0.y = max_point_AABB.y;
            //     p0.z = min_point_AABB.z;
            //     p1.x = min_point_AABB.x;
            //     p1.y = max_point_AABB.y;
            //     p1.z = max_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }
            // {
            //     geometry_msgs::msg::Point p0;
            //     geometry_msgs::msg::Point p1;
            //     p0.x = max_point_AABB.x;
            //     p0.y = max_point_AABB.y;
            //     p0.z = min_point_AABB.z;
            //     p1.x = max_point_AABB.x;
            //     p1.y = max_point_AABB.y;
            //     p1.z = max_point_AABB.z;
            //     marker.points.push_back(p0);
            //     marker.points.push_back(p1);
            // }

            object_bbx_marker_array.markers.push_back(marker);
        }
        object_bbx_pub_->publish(object_bbx_marker_array);
    }


    // publishStaticCloud(rgbd_frontend_output->static_landmarks_);
    // publishObjectCloud(rgbd_frontend_output->dynamic_keypoint_measurements_, rgbd_frontend_output->dynamic_landmarks_);
    publishObjectPositions(
        object_pose_pub_,
        rgbd_frontend_output->propogated_object_poses_,
        rgbd_frontend_output->getFrameId(),
        rgbd_frontend_output->getTimestamp(),
        "frontend");

    publishObjectPaths(
        object_pose_path_pub_,
        rgbd_frontend_output->propogated_object_poses_,
        rgbd_frontend_output->getFrameId(),
        rgbd_frontend_output->getTimestamp(),
        "frontend",
        60
    );


}

// void FrontendDisplayRos::publishStaticCloud(const Landmarks& static_landmarks) {
//     pcl::PointCloud<pcl::PointXYZRGB> cloud;

//     for(const Landmark& lmk : static_landmarks) {
//         // publish static lmk's as white
//         pcl::PointXYZRGB pt(lmk(0), lmk(1), lmk(2), 0, 0, 0);
//         cloud.points.push_back(pt);
//     }


//     sensor_msgs::msg::PointCloud2 pc2_msg;
//     pcl::toROSMsg(cloud, pc2_msg);
//     pc2_msg.header.frame_id = "world";
//     static_tracked_points_pub_->publish(pc2_msg);
// }

// void FrontendDisplayRos::publishObjectCloud(const StatusKeypointMeasurements& dynamic_measurements, const Landmarks& dynamic_landmarks) {
//     pcl::PointCloud<pcl::PointXYZRGB> cloud;
//     CHECK_EQ(dynamic_measurements.size(), dynamic_landmarks.size());

//     const size_t num_measurements = dynamic_measurements.size();

//     for(size_t i = 0; i < num_measurements; i++) {
//         const StatusKeypointMeasurement& kpm = dynamic_measurements.at(i);
//         const KeypointStatus& status = kpm.first;
//         const Landmark& lmk = dynamic_landmarks.at(i);

//         const cv::Scalar colour = ColourMap::getObjectColour(status.label_);
//         pcl::PointXYZRGB pt(lmk(0), lmk(1), lmk(2), colour(0), colour(1), colour(2));
//         cloud.points.push_back(pt);
//     }

//     sensor_msgs::msg::PointCloud2 pc2_msg;
//     pcl::toROSMsg(cloud, pc2_msg);
//     pc2_msg.header.frame_id = "world";
//     dynamic_tracked_points_pub_->publish(pc2_msg);
// }


// void FrontendDisplayRos::publishObjectPositions(const std::map<ObjectId, gtsam::Pose3>& propogated_object_poses, FrameId frame_id) {
//     visualization_msgs::msg::MarkerArray object_pose_marker_array;
//     visualization_msgs::msg::MarkerArray object_path_marker_array;

//     static visualization_msgs::msg::Marker delete_marker;
//     delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

//     object_pose_marker_array.markers.push_back(delete_marker);

//     for(const auto&[object_id, pose] : propogated_object_poses) {

//             //object centroid per frame
//             visualization_msgs::msg::Marker marker;
//             marker.header.frame_id = "world";
//             marker.ns = "frontend_composed_object_positions";
//             marker.id = object_id;
//             marker.type = visualization_msgs::msg::Marker::SPHERE;
//             marker.action = visualization_msgs::msg::Marker::ADD;
//             marker.header.stamp = node_->now();
//             marker.pose.position.x = pose.x();
//             marker.pose.position.y = pose.y();
//             marker.pose.position.z = pose.z();
//             marker.pose.orientation.x = pose.rotation().toQuaternion().x();
//             marker.pose.orientation.y = pose.rotation().toQuaternion().y();
//             marker.pose.orientation.z = pose.rotation().toQuaternion().z();
//             marker.pose.orientation.w = pose.rotation().toQuaternion().w();
//             marker.scale.x = 1;
//             marker.scale.y = 1;
//             marker.scale.z = 1;
//             marker.color.a = 1.0; // Don't forget to set the alpha!

//             const cv::Scalar colour = ColourMap::getObjectColour(object_id);
//             marker.color.r = colour(0)/255.0;
//             marker.color.g = colour(1)/255.0;
//             marker.color.b = colour(2)/255.0;

//             object_pose_marker_array.markers.push_back(marker);

//             //update past trajectotries of objects
//             auto it = object_trajectories_.find(object_id);
//             if(it == object_trajectories_.end()) {
//                 object_trajectories_[object_id] = gtsam::Pose3Vector();
//             }

//             object_trajectories_update_[object_id] = frame_id;
//             object_trajectories_[object_id].push_back(pose);
//     }

//     //iterate over object trajectories and display the ones with enough poses and the ones weve seen recently
//     for(const auto& [object_id, poses] : object_trajectories_) {
//         const FrameId last_seen_frame = object_trajectories_update_.at(object_id);

//         //if weve seen the object in the last 30 frames and the length is at least 2
//         if(poses.size() < 2u) {
//             continue;
//         }

//         //draw a line list for viz
//         visualization_msgs::msg::Marker line_list_marker;
//         line_list_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
//         line_list_marker.header.frame_id = "world";
//         line_list_marker.ns = "frontend_composed_object_path";
//         line_list_marker.id = object_id;
//         line_list_marker.header.stamp = node_->now();
//         line_list_marker.scale.x = 0.3;

//         line_list_marker.pose.orientation.x = 0;
//         line_list_marker.pose.orientation.y = 0;
//         line_list_marker.pose.orientation.z = 0;
//         line_list_marker.pose.orientation.w = 1;

//         const cv::Scalar colour = ColourMap::getObjectColour(object_id);
//         line_list_marker.color.r = colour(0)/255.0;
//         line_list_marker.color.g = colour(1)/255.0;
//         line_list_marker.color.b = colour(2)/255.0;
//         line_list_marker.color.a = 1;

//         //only draw the last 60 poses
//         const size_t traj_size = std::min(60, static_cast<int>(poses.size()));
//         // const size_t traj_size = poses.size();
//         //have to duplicate the first in each drawn pair so that we construct a complete line
//         for(size_t i = poses.size() - traj_size + 1; i < poses.size(); i++) {
//             const gtsam::Pose3& prev_pose = poses.at(i-1);
//             const gtsam::Pose3& curr_pose = poses.at(i);

//             {
//                 geometry_msgs::msg::Point p;
//                 p.x = prev_pose.x();
//                 p.y = prev_pose.y();
//                 p.z = prev_pose.z();

//                 line_list_marker.points.push_back(p);
//             }

//             {
//                 geometry_msgs::msg::Point p;
//                 p.x = curr_pose.x();
//                 p.y = curr_pose.y();
//                 p.z = curr_pose.z();

//                 line_list_marker.points.push_back(p);
//             }
//         }

//          object_path_marker_array.markers.push_back(line_list_marker);
//     }

//     // Publish centroids of composed object poses
//     object_pose_pub_->publish(object_pose_marker_array);

//     // Publish composed object path
//     object_pose_path_pub_->publish(object_path_marker_array);

// }


// void FrontendDisplayRos::publishObjectMotions(const MotionEstimateMap& motion_estimates, const std::map<ObjectId, gtsam::Pose3>& propogated_object_poses) {
//     CHECK_EQ(motion_estimates.size(), propogated_object_poses.size());

//     //should have a 1 to 1 between the motion map and the propogated poses (same object ids in both)
// }



void FrontendDisplayRos::publishOdometry(const gtsam::Pose3& T_world_camera, Timestamp timestamp) {
    DisplayRos::publishOdometry(odometry_pub_, T_world_camera, timestamp);
    geometry_msgs::msg::TransformStamped t;
    utils::convertWithHeader(T_world_camera, t, timestamp, params_.world_frame_id_, params_.camera_frame_id_);
    // Send the transformation
    tf_broadcaster_->sendTransform(t);

}

// void FrontendDisplayRos::publishOdometryPath(const gtsam::Pose3& T_world_camera, Timestamp timestamp) {
//     geometry_msgs::msg::PoseStamped pose_stamped;
//     utils::convertWithHeader(T_world_camera, pose_stamped, timestamp, "world");

//     static std_msgs::msg::Header header;
//     header.stamp = utils::toRosTime(timestamp);
//     header.frame_id = "world";
//     odom_path_msg_.header = header;

//     odom_path_msg_.poses.push_back(pose_stamped);
//     odometry_path_pub_->publish(odom_path_msg_);

// }


void FrontendDisplayRos::publishDebugImage(const cv::Mat& debug_image) {
    if(debug_image.empty()) return;

    // cv::Mat resized_image;
    // cv::resize(debug_image, resized_image, cv::Size(640, 480));

    std_msgs::msg::Header hdr;
    sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(hdr, "bgr8", debug_image).toImageMsg();
    tracking_image_pub_.publish(msg);
}

//TODO: lots of repeated code with this funyction and publishObjectPositions - need to functionalise
void FrontendDisplayRos::publishGroundTruthInfo(Timestamp timestamp, const GroundTruthInputPacket& gt_packet, const cv::Mat& rgb) {
    //odometry gt
    const gtsam::Pose3& T_world_camera = gt_packet.X_world_;
    nav_msgs::msg::Odometry odom_msg;
    utils::convertWithHeader(T_world_camera, odom_msg, timestamp, "world", "camera");
    gt_odometry_pub_->publish(odom_msg);

    const auto frame_id = gt_packet.frame_id_;


    //odom path gt
    geometry_msgs::msg::PoseStamped pose_stamped;
    utils::convertWithHeader(T_world_camera, pose_stamped, timestamp, "world");
    static std_msgs::msg::Header header;
    header.stamp = utils::toRosTime(timestamp);
    header.frame_id = "world";
    gt_odom_path_msg_.header = header;
    gt_odom_path_msg_.poses.push_back(pose_stamped);

    gt_odom_path_pub_->publish(gt_odom_path_msg_);

    //prepare display image
    cv::Mat disp_image;
    rgb.copyTo(disp_image);

    static std::map<ObjectId, gtsam::Pose3Vector> gt_object_trajectories; //Used for gt path updating
    static std::map<ObjectId, FrameId> gt_object_trajectories_update; // The last frame id that the object was seen in

    std::set<ObjectId> seen_objects;

    //prepare gt object pose markers
    visualization_msgs::msg::MarkerArray object_pose_marker_array;
    visualization_msgs::msg::MarkerArray object_path_marker_array;
    static visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    object_pose_marker_array.markers.push_back(delete_marker);
    object_path_marker_array.markers.push_back(delete_marker);

    for(const auto& object_pose_gt : gt_packet.object_poses_) {
        const gtsam::Pose3 L_world = T_world_camera * object_pose_gt.L_camera_;
        const ObjectId object_id = object_pose_gt.object_id_;

        seen_objects.insert(object_id);

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.ns = "ground_truth_object_poses";
        marker.id = object_id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.header.stamp = node_->now();
        marker.pose.position.x = L_world.x();
        marker.pose.position.y = L_world.y();
        marker.pose.position.z = L_world.z();
        marker.pose.orientation.x = L_world.rotation().toQuaternion().x();
        marker.pose.orientation.y = L_world.rotation().toQuaternion().y();
        marker.pose.orientation.z = L_world.rotation().toQuaternion().z();
        marker.pose.orientation.w = L_world.rotation().toQuaternion().w();
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.a = 1.0; // Don't forget to set the alpha!

        const cv::Scalar colour = ColourMap::getObjectColour(object_id);
        marker.color.r = colour(0)/255.0;
        marker.color.g = colour(1)/255.0;
        marker.color.b = colour(2)/255.0;

        visualization_msgs::msg::Marker text_marker = marker;
        text_marker.ns = "ground_truth_object_labels";
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.text = std::to_string(object_id);
        text_marker.pose.position.z += 1.0; //make it higher than the pose marker
        text_marker.scale.z = 0.7;

        object_pose_marker_array.markers.push_back(marker);
        object_pose_marker_array.markers.push_back(text_marker);

        //draw on bbox
        object_pose_gt.drawBoundingBox(disp_image);

        //update past trajectotries of objects
        auto it = gt_object_trajectories.find(object_id);
        if(it == gt_object_trajectories.end()) {
            gt_object_trajectories[object_id] = gtsam::Pose3Vector();
        }

        gt_object_trajectories_update[object_id] = frame_id;
        gt_object_trajectories[object_id].push_back(L_world);

    }

    //repeated code from publishObjectPositions function
    //iterate over object trajectories and display the ones with enough poses and the ones weve seen recently
    for(const auto& [object_id, poses] : gt_object_trajectories) {
        const FrameId last_seen_frame = gt_object_trajectories_update.at(object_id);

        //if weve seen the object in the last 30 frames and the length is at least 2
        if(poses.size() < 2u) {
            continue;
        }

        //draw a line list for viz
        visualization_msgs::msg::Marker line_list_marker;
        line_list_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        line_list_marker.header.frame_id = "world";
        line_list_marker.ns = "gt_frontend_composed_object_path";
        line_list_marker.id = object_id;
        line_list_marker.header.stamp = node_->now();
        line_list_marker.scale.x = 0.3;

        line_list_marker.pose.orientation.x = 0;
        line_list_marker.pose.orientation.y = 0;
        line_list_marker.pose.orientation.z = 0;
        line_list_marker.pose.orientation.w = 1;

        const cv::Scalar colour = ColourMap::getObjectColour(object_id);
        line_list_marker.color.r = colour(0)/255.0;
        line_list_marker.color.g = colour(1)/255.0;
        line_list_marker.color.b = colour(2)/255.0;
        line_list_marker.color.a = 1;

        //only draw the last 60 poses
        const size_t traj_size = std::min(60, static_cast<int>(poses.size()));
        // const size_t traj_size = poses.size();
        //have to duplicate the first in each drawn pair so that we construct a complete line
        for(size_t i = poses.size() - traj_size + 1; i < poses.size(); i++) {
            const gtsam::Pose3& prev_pose = poses.at(i-1);
            const gtsam::Pose3& curr_pose = poses.at(i);

            {
                geometry_msgs::msg::Point p;
                p.x = prev_pose.x();
                p.y = prev_pose.y();
                p.z = prev_pose.z();

                line_list_marker.points.push_back(p);
            }

            {
                geometry_msgs::msg::Point p;
                p.x = curr_pose.x();
                p.y = curr_pose.y();
                p.z = curr_pose.z();

                line_list_marker.points.push_back(p);
            }
        }

        object_path_marker_array.markers.push_back(line_list_marker);
    }

    // Publish centroids of composed object poses
    gt_object_pose_pub_->publish(object_pose_marker_array);

    // Publish composed object path
    gt_object_path_pub_->publish(object_path_marker_array);

    cv::Mat resized_image;
    cv::resize(disp_image, resized_image, cv::Size(640, 480));

    std_msgs::msg::Header hdr;
    sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(hdr, "bgr8", resized_image).toImageMsg();
    gt_bounding_box_pub_.publish(msg);
}

}
