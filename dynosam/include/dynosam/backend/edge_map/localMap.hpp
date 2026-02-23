#ifndef LOCAL_MAP_H
#define LOCAL_MAP_H

#include "dynosam/backend/edge_map/KeyFrame.hpp"
#include "dynosam/backend/edge_map/ElementEdge.hpp"
#include "dynosam/backend/edge_map/FeatureMerger.hpp"
#include "dynosam/frontend/vision/DisjointSet.hpp"
#include <pcl/common/distances.h>

#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>

//-- Association result between local map and frame
typedef std::pair<std::vector<cv::Point3d>, std::vector<dyno::elementEdge>> match3d_2d;

namespace dyno{

class localMap{
public:

    enum class State {
        NOT_INITIALIZED,  //-- Local map not initialized
        INITIALIZED,      //-- Local map initialized (two frames)
        LOST              //-- Local map cannot be associated with the latest keyframe
    };

    State msState;

    //-- Map consists of a set of edges
    std::vector<elementEdge> mvElementEdges;
    //-- Mapping from elementEdge's element_id to its index in mvElementEdges
    std::map<unsigned int, int> mmElementID2index;

    
    //-- Edge cluster data type, stored in mvEleEdgeClusters, indexed by mmClusterID2index
    std::vector<elementEdgeCluster> mvEleEdgeClusters;
    std::map<unsigned int, int> mmClusterID2index;

    //-- Keyframes that compose this local map and their poses
    std::vector<KeyFramePtr> mvKeyFrames;
    std::map<int, int> mmKFID2KFindex;

    //-- Reference coordinate frame pose for the current local map
    //Sophus::SE3d T_ref;

    //-- Initialize a local map when there are two keyframes in the local map.
    //-- This local map only stores co-visible edge features from the two frames
    void initLocalMap();

    void addFrame2LocalMap(KeyFramePtr frame_cur);

    //-- Remove the **first** keyframe from the mvKeyFrames queue
    void removeKeyFrameFront();

    //-- Fit all clusters in 3D space to obtain integrated edges for local mapping / BA
    void clustersFitting3D();

    //-- Fit all clusters using reprojection and epipolar line constraints to obtain integrated edges for local mapping / BA
    void clusterFittingProjection();

    //-- Get the association relationship between the current local map and a keyframe
    void getAssoFrameMergeEdge(int kf_id_dst, std::vector<match3d_2d>& matches, std::vector<double>& weights);

    localMap()
    {
        //-- Initialize cluster color generator
        elementEdgeCluster::initRandom();
        //-- Not initialized
        msState = State::NOT_INITIALIZED;
    }

private:
    //-- The following functions are union-find (disjoint set) operations

    //-- Find the root node of the union-find set
    unsigned int findRoot(unsigned int curr_id);

    //-- Prune the union-find set so that all nodes point directly to the root
    void pruningMap();

    void mergeElementCluster(int cluster_idx_1, int cluster_idx_2);

    //-- Decide whether to remove a cluster based on each cluster's count_not_update
    void elementClusterCulling();

    //-- Associate two keyframes based on many-to-many association strategy
    associationResult associationMulti2Multi(KeyFramePtr frame_ref, KeyFramePtr frame_cur);

    //-- Merge point clouds of a single cluster to obtain the merged point cloud
    void getMergedCluster(const std::vector<pcl::PointCloud<pcl::PointXYZ>>& clusterCloud,
                                      pcl::PointCloud<pcl::PointXYZ>& mergedCloud);

    
    pcl::PointXYZ calcCloudCentroid(const pcl::PointCloud<pcl::PointXYZ>& pointCloud, pcl::PointXYZ current);

    void assignWeights();

};

//-- Define smart pointer using using
using localMapPtr = std::shared_ptr<localMap>;

}

#endif