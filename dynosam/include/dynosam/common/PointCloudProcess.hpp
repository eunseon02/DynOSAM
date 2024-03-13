
/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au), Yiduo Wang (yiduo.wang@sydney.edu.au)
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

#include "dynosam/common/Types.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/visualizer/ColourMap.hpp"

#include <pcl/point_types.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>


namespace dyno {

using CloudPerObject = gtsam::FastMap<ObjectId, pcl::PointCloud<pcl::PointXYZRGB>>;

struct ObjectBBX{
public: 
    // Orientation is included, but identity for AABB

    ObjectBBX():orientation_(Eigen::Matrix3f::Identity())
    {}

    pcl::PointXYZ min_bbx_point_;
    pcl::PointXYZ max_bbx_point_;
    pcl::PointXYZ bbx_position_;

    Eigen::Matrix3f orientation_;
};

using BbxPerObject = gtsam::FastMap<ObjectId,  ObjectBBX>;

CloudPerObject groupObjectCloud(const StatusLandmarkEstimates& landmarks, const gtsam::Pose3& T_world_camera);

// Axis Aligned Bounding Box (AABB)
// template<typename PointT>
// ObjectBBX findAABBFromCloud(const typename pcl::PointCloud<PointT>::Ptr obj_cloud_ptr){
//     pcl::PointCloud<PointT> filtered_cloud;

//     // statistical outlier removal
//     pcl::StatisticalOutlierRemoval<PointT> outlier_remover;
//     outlier_remover.setInputCloud(obj_cloud_ptr);
//     outlier_remover.setMeanK(100);
//     outlier_remover.setStddevMulThresh(1.0);
//     outlier_remover.filter(filtered_cloud);

//     // axis aligned bounding box
//     pcl::MomentOfInertiaEstimation<PointT> bbx_extractor;
//     bbx_extractor.setInputCloud(pcl::make_shared<pcl::PointCloud<PointT> >(filtered_cloud));
//     bbx_extractor.compute();

//     PointT min_point_AABB;
//     PointT max_point_AABB;
//     bbx_extractor.getAABB(min_point_AABB, max_point_AABB);

//     ObjectBBX aabb;
//     aabb.min_bbx_point_ = pcl::PointXYZ(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
//     aabb.max_bbx_point_ = pcl::PointXYZ(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);

//     return aabb;
// }

// TODO: template this
ObjectBBX findAABBFromCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_cloud_ptr);

template<typename PointT>
pcl::PointCloud<pcl::PointXYZ> findLineListPointsFromAABBMinMax(const PointT& min_point_AABB, const PointT& max_point_AABB) {
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

// Oriented Bounding Box
ObjectBBX findOBBFromCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_cloud_ptr);

} //dyno