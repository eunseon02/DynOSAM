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

#include "dynosam_ros/DisplayRos.hpp"
#include "dynosam_ros/RosUtils.hpp"

#include <dynosam/visualizer/ColourMap.hpp>

namespace dyno {

void DisplayRos::publishPointCloud(PointCloud2Pub::SharedPtr pub, const StatusLandmarkEstimates& landmarks) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;

    for(const auto& status_estimate : landmarks) {
        const LandmarkStatus& status =  status_estimate.first;
        const LandmarkEstimate& estimate = status_estimate.second;
        const Landmark& lmk = estimate.second;

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
    pub->publish(pc2_msg);
}

void DisplayRos::publishOdometry(OdometryPub::SharedPtr pub, const gtsam::Pose3& T_world_camera, Timestamp timestamp) {
    nav_msgs::msg::Odometry odom_msg;
    utils::convertWithHeader(T_world_camera, odom_msg, timestamp, params_.world_frame_id_, params_.camera_frame_id_);
    pub->publish(odom_msg);
}

void DisplayRos::publishOdometryPath(PathPub::SharedPtr pub, const gtsam::Pose3Vector& poses, Timestamp latest_timestamp) {
    nav_msgs::msg::Path path;
    for(const gtsam::Pose3& odom : poses) {
        geometry_msgs::msg::PoseStamped pose_stamped;
        utils::convertWithHeader(odom, pose_stamped, latest_timestamp, params_.world_frame_id_);
        path.poses.push_back(pose_stamped);

    }

    path.header.stamp = utils::toRosTime(latest_timestamp);
    path.header.frame_id = params_.world_frame_id_;
    pub->publish(path);

}

void DisplayRos::publishObjectPositions(
        MarkerArrayPub::SharedPtr pub,
        const ObjectPoseMap& object_positions,
        FrameId frame_id,
        Timestamp latest_timestamp,
        const std::string& prefix_marker_namespace,
        bool draw_labels,
        double scale)
{
    visualization_msgs::msg::MarkerArray object_pose_marker_array;
    static visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    auto ros_time = utils::toRosTime(latest_timestamp);

    object_pose_marker_array.markers.push_back(delete_marker);

    for(const auto&[object_id, poses_map] : object_positions) {

        //do not draw if in current frame
        if(!poses_map.exists(frame_id)) {
            continue;
        }

        const gtsam::Pose3& pose = poses_map.at(frame_id);

        //assume
        //object centroid per frame
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = params_.world_frame_id_;
        marker.ns = prefix_marker_namespace + "_object_positions";
        marker.id = object_id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.header.stamp = ros_time;
        marker.pose.position.x = pose.x();
        marker.pose.position.y = pose.y();
        marker.pose.position.z = pose.z();
        marker.scale.x = scale;
        marker.scale.y = scale;
        marker.scale.z = scale;
        marker.color.a = 1.0; // Don't forget to set the alpha!

        const cv::Scalar colour = ColourMap::getObjectColour(object_id);
        marker.color.r = colour(0)/255.0;
        marker.color.g = colour(1)/255.0;
        marker.color.b = colour(2)/255.0;

        object_pose_marker_array.markers.push_back(marker);

        if(draw_labels) {
            visualization_msgs::msg::Marker text_marker = marker;
            text_marker.ns = prefix_marker_namespace + "_object_labels";
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.text = std::to_string(object_id);
            text_marker.pose.position.z += 1.0; //make it higher than the pose marker
            text_marker.scale.z = 0.7;
            object_pose_marker_array.markers.push_back(text_marker);
        }
    }

    pub->publish(object_pose_marker_array);

}

void DisplayRos::publishObjectPaths(
        MarkerArrayPub::SharedPtr pub,
        const ObjectPoseMap& object_positions,
        FrameId frame_id,
        Timestamp latest_timestamp,
        const std::string& prefix_marker_namespace,
        const int min_poses
    )
{
    visualization_msgs::msg::MarkerArray object_path_marker_array;
    static visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    auto ros_time = utils::toRosTime(latest_timestamp);
    object_path_marker_array.markers.push_back(delete_marker);

    for(const auto&[object_id, poses_map] : object_positions) {
        //keys for FastMap are sorted by std::less so largest key should be in end()
        // const FrameId last_seen_frame = poses_map.end()->first;

        // LOG(INFO) << last_seen_frame << " " << frame_id;

        // //dont draw objects more than 10 frames ago
        // if(frame_id - last_seen_frame > 10u) {
        //     continue;
        // }

        if(poses_map.size() < 2u) {
            continue;
        }

        //draw a line list for viz
        visualization_msgs::msg::Marker line_list_marker;
        line_list_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        line_list_marker.header.frame_id = "world";
        line_list_marker.ns =  prefix_marker_namespace + "_object_path";
        line_list_marker.id = object_id;
        line_list_marker.header.stamp = ros_time;
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

        size_t traj_size;
        //draw all the poses
        if(min_poses == -1) {
            traj_size = poses_map.size();
        }
        else {
            traj_size = std::min(min_poses, static_cast<int>(poses_map.size()));
        }

        //totally assume in order
        auto map_iter = poses_map.end();
        //equivalant to map_iter-traj_size + 1
        //we want to go backwards to the starting point specified by trajectory size
        std::advance(map_iter, (-traj_size + 1));
        for(; map_iter != poses_map.end(); map_iter++) {
            auto prev_iter = std::prev(map_iter, 1);
            const gtsam::Pose3& prev_pose = prev_iter->second;;
            const gtsam::Pose3& curr_pose = map_iter->second;

            //check frames are consequative
            CHECK_EQ(prev_iter->first + 1, map_iter->first);

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

    pub->publish(object_path_marker_array);
}


pcl::PointCloud<pcl::PointXYZ> findLineListPointsFromAABBMinMax(pcl::PointXYZ min_point_AABB, pcl::PointXYZ max_point_AABB){
    pcl::PointCloud<pcl::PointXYZ> line_list_points;

    // bottom
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
    // top
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
    // vertical
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z));
    line_list_points.push_back(pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z));

    return line_list_points;
}

}
