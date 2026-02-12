#include "dynosam/backend/edge_map/localMap.hpp"
using namespace edge_map;

//-- Initialize a local map when the number of keyframes is greater than 2
void localMap::initLocalMap()
{
    assert(mvKeyFrames.size() == 2);

    //-- Get two keyframes
    KeyFramePtr pKF_0 = mvKeyFrames[0];
    KeyFramePtr pKF_1 = mvKeyFrames[1];

    //* STEP-0 First, add all edges from pKF_0 to the map
    for(int i = 0; i < pKF_0->mvEdges.size(); ++i)
    {
        //-- All edges in the keyframe are valid edges
        int edge_idx = i;
        elementEdge ele(pKF_0->KF_ID, edge_idx);
        //-- Add current edge element to local map
        mvElementEdges.push_back(ele);
        mmElementID2index[ele.element_id] = mvElementEdges.size()-1;
        //-- Keyframe indexes the current local map
        pKF_0->mmEdgeIndex2ElementEdgeID[i] = ele.element_id;
    }

    //-- Initialize the reference coordinate frame of the entire local map
    // T_ref = pKF_0->KF_pose_g;

    //-- Associate two frames
    associationResult res = associationMulti2Multi(pKF_0, pKF_1);

    //* Summarize association results
    //-- Build index, first is edge category id, second is root_idx of pKF_0 edge corresponding to edge category
    std::unordered_map<int, int> labelRootMap;
    std::unordered_map<int, int> edgeLabelMap = std::move(res.second); //-- edge-label map
    std::unordered_map<int, int> curRefMap = std::move(res.first);     //-- current frame-reference frame map

    //-- Mark which edges in pKF_0 can be associated and which cannot
    std::vector<bool> isAssociated(pKF_0->mvEdges.size(), false);

    // * STEP-1 Perform union-find merge within pKF_0
    for(const auto &pair : edgeLabelMap)
    {
        const int label = pair.second;
        const int edge_idx = pair.first;

        //-- If traversed, it means it can be associated
        isAssociated[edge_idx] = true;

        if(labelRootMap.find(label) == labelRootMap.end())
        {
            labelRootMap[label] = edge_idx;
        }else{
            //-- Get root_idx of current index
            int root_idx = labelRootMap[label];

            //-- Map frame edge ID to map edge ID, and merge these edges using union-find
            unsigned int map_id_1 = pKF_0->mmEdgeIndex2ElementEdgeID.at(root_idx);
            unsigned int map_id_2 = pKF_0->mmEdgeIndex2ElementEdgeID.at(edge_idx);

            //-- Union-find merge
            int map_idx_2 = mmElementID2index.at(map_id_2);
            mvElementEdges[map_idx_2].union_id = findRoot(map_id_1);
        }
    }

    // * STEP-2 Traverse association list of pKF_1, add edges that can be associated to Map
    for(const auto& pair: curRefMap)
    {
        int cur_idx = pair.first;
        int ref_idx = pair.second;

        int label_id = edgeLabelMap[ref_idx];

        //-- If the associated reference frame is invalid and not found in frame2MapElement, skip
        if(pKF_0->mmEdgeIndex2ElementEdgeID.find(ref_idx)==pKF_0->mmEdgeIndex2ElementEdgeID.end()) continue;

        int size = mvElementEdges.size();
        elementEdge ele(pKF_1->KF_ID, cur_idx);
        //-- Map id of reference frame associated with this edge of current frame
        ele.union_id = pKF_0->mmEdgeIndex2ElementEdgeID[ref_idx];

        mvElementEdges.push_back(ele);
        mmElementID2index[ele.element_id] = mvElementEdges.size()-1;
        pKF_1->mmEdgeIndex2ElementEdgeID[cur_idx] = ele.element_id;
    }

    // * STEP-3 Remove edges from pKF_0 that cannot be associated with pKF_1 from local map
    for (auto it = mvElementEdges.begin(); it != mvElementEdges.end(); ) {
        if(it->kf_id == pKF_0->KF_ID && !isAssociated[it->kf_edge_idx]) 
        {
            //-- First delete keyframe to local map index
            if (pKF_0->mmEdgeIndex2ElementEdgeID.find(it->kf_edge_idx) != pKF_0->mmEdgeIndex2ElementEdgeID.end())
            {
                pKF_0->mmEdgeIndex2ElementEdgeID.erase(it->kf_edge_idx);
            }
            //-- Then delete elementEdge from local map
            it = mvElementEdges.erase(it); // erase returns next valid iterator
        }else{
            ++it; // Otherwise continue
        }
    }
    //-- If elements are deleted, need to reset id2idx map
    mmElementID2index.clear();
    for(size_t i = 0; i < mvElementEdges.size(); ++i)
    {
        elementEdge& ele = mvElementEdges[i];
        mmElementID2index[ele.element_id] = i;
    }

    //-- Pruning
    pruningMap();

    //-- Generate elementEdge Cluster
    //-- Mapping of edge map elements to category IDs,
    //-- first is label of map element, second is list of map elements, i.e., a group of map elements is a complete edge
    std::map<unsigned int, std::vector<unsigned int>> eleEdgeMap;
    eleEdgeMap.clear();
    for(int i = 0; i < mvElementEdges.size(); ++i)
    {
        unsigned int curr_id = mvElementEdges[i].element_id;
        unsigned int root_id = findRoot(curr_id);
        //std::cout<<root_idx<<std::endl;
        //-- There will be a BUG when root_idx is 0, need to check
        eleEdgeMap[root_id].push_back(mvElementEdges[i].element_id);
    }

    for (auto& pair : eleEdgeMap)
    {
        unsigned int cluster_id = pair.first;
        std::vector<unsigned int> clusters = pair.second;
        elementEdgeCluster e_cluster(cluster_id, clusters);
        mvEleEdgeClusters.push_back(std::move(e_cluster));
        mmClusterID2index[cluster_id] = mvEleEdgeClusters.size() - 1;
    }

    mmKFID2KFindex[pKF_0->KF_ID] = 0;
    mmKFID2KFindex[pKF_1->KF_ID] = 1;

}

//-- Input curr_id is the element_id of current elementEdge
unsigned int localMap::findRoot(unsigned int curr_id)
{
    //-- Root node of same class as current frame
    int curr_idx = mmElementID2index.at(curr_id);
    int union_id = mvElementEdges.at(curr_idx).union_id;
    if(union_id == -1){//-- If root node of current curr_id is -1, it means this is an independent root node
        return curr_id;
    }else{//-- Otherwise, recursively find root node
        return findRoot(union_id);
    }
}

void localMap::pruningMap(){
    for(int i = 0; i < mvElementEdges.size(); ++i){
        unsigned int ele_id = mvElementEdges[i].element_id;
        unsigned int root_id = findRoot(ele_id);
        //std::cout<<root_id << "," << mvElementEdges[i].union_id <<std::endl;
        if (root_id != ele_id)
            mvElementEdges[i].union_id = root_id;
    }
}

//-- Project frame_cur onto frame_ref to achieve many-to-many association between frame_cur and frame_ref
//-- By default, frame_ref timestamp is before frame_cur
associationResult localMap::associationMulti2Multi(KeyFramePtr frame_ref, KeyFramePtr frame_cur)
{
    //-- Create a union-find set for edges of reference frame, and update this union-find set in subsequent continuous optimization
    const int ref_edge_num = frame_ref->mvEdges.size();
    DisjointSet edgeRefSet(ref_edge_num);

    std::vector<bool> isAssociated(ref_edge_num, false); //-- Some edges of reference frame are not associated, record here

    //-- Since current frame is reprojected to reference frame to associate reference frame, current frame stores association relationship with reference frame
    std::unordered_map<int, int> currframeMap; //-- <idx1, idx2> records that idx1-th edge of current frame and idx2-th edge of reference frame are of the same class
    currframeMap.reserve(frame_cur->mvEdges.size() / 2);

    //-- Pose difference
    Sophus::SE3d T_ref_cur = frame_ref->KF_pose_g.inverse() * frame_cur->KF_pose_g;

    //-- Cache reference frame's mmIndexMap to avoid repeated lookups
    const auto& refIndexMap = frame_ref->mmIndexMap;

    //-- Use valid edges of current frame to associate edges in reference frame, and merge reference frame associations that are associated to multiple using union-find
    const int cur_edge_num = frame_cur->mvEdges.size();
    for(int i = 0; i < cur_edge_num; ++i)
    {
        Edge& query_edge = frame_cur->mvEdges[i];
        //-- Associate one edge of current frame with reference frame
        std::vector<int> associated_edges = frame_ref->edgeWiseCorrespondenceLocalMapping(query_edge,T_ref_cur);
        
        if(associated_edges.empty()) continue;
        
        //-- At this point, all edges in the returned list need to be merged
        if(associated_edges.size() >= 2)
        {
            //-- associated_edges[0] is ID of 0-th edge
            const int root_idx = refIndexMap.at(associated_edges[0]);

            for(size_t j = 1; j < associated_edges.size(); j++)
            {
                //-- associated_edges[j] is ID of j-th edge among these associated edges
                int curr_idx = refIndexMap.at(associated_edges[j]);
                //-- Merge these two idx in union-find set, i.e., mvEdges[root_idx] and mvEdges[curr_idx] are of the same class
                edgeRefSet.to_union(root_idx, curr_idx);
            }
        }

        //-- Update association relationship, if correct association is established, get association between current edge and reference frame
        currframeMap[i] = refIndexMap.at(associated_edges[0]);
        //-- Record whether this edge of reference frame is associated
        for(const auto& edge_id : associated_edges)
        {
            const int curr_idx = refIndexMap.at(edge_id);
            isAssociated[curr_idx] = true;
        }
    }
    //-- At this point, many-to-many clustering is complete
    edgeRefSet.pruningSet();

    //-- Organize classification and visualization, need to create a unique color for all classes in union-find set, and look up this color during visualization and render
    std::unordered_map<int, int> clusterMap; //-- Map corresponding classification to color, classification is idx of root node of union-find set of ref frame
                                             //-- first is idx of each edge of ref frame, second is corresponding category
    clusterMap.reserve(ref_edge_num / 2);
    
    for(int i = 0; i < ref_edge_num; ++i){
        if(isAssociated[i]==false) continue;
        //-- For associable edges, find their root node through union-find set
        clusterMap[i] = edgeRefSet.find(i);
    }
    associationResult res;
    res.first = std::move(currframeMap);
    res.second = std::move(clusterMap);
    //-- Pre-store association relationship between current frame and reference frame
    frame_cur->mmMapAssociations[frame_ref->KF_ID] = res;
    return res;
}

void localMap::addFrame2LocalMap(KeyFramePtr frame_cur)
{
    //-- First check if initialization is complete, if not, skip all subsequent execution
    if(msState == State::NOT_INITIALIZED)
    {
        if(mvKeyFrames.size() < 2)
        {
            mvKeyFrames.push_back(frame_cur);
            mmKFID2KFindex[frame_cur->KF_ID] = mvKeyFrames.size()-1;
        }
            
        if (mvKeyFrames.size() ==2)
        {
            initLocalMap();
            std::cout<<"finish init"<<std::endl;
            msState = State::INITIALIZED;
        }
        return;
    }

    //-- First initialize mbModifiedCur of all clusters to mark whether these clusters have been modified
    for (auto& cluster : mvEleEdgeClusters)
    {
        cluster.mbModifiedCur = false;
    }

    int N = frame_cur->mvEdges.size();
    //-- Association list of current frame, edges that can be associated with any previous reference frame will be set to true
    std::vector<bool> isAssociated(N, false);
    frame_cur->mmEdgeIndex2ElementEdgeID.clear();

    for(int i = mvKeyFrames.size()-1; i >= 0; --i)
    {
        KeyFramePtr frame_ref = mvKeyFrames[i];
        associationResult res;
        //-- First check if current frame has established association with reference frame
        if(frame_cur->mmMapAssociations.find(frame_ref->KF_ID) != frame_cur->mmMapAssociations.end()){
            //-- If already associated, no need to re-execute association
            res = frame_cur->mmMapAssociations[frame_ref->KF_ID];
        }else{
            //-- If not associated, need to re-associate
            res = associationMulti2Multi(frame_ref, frame_cur);
        }

        //-- Summarize association results
        //-- Build index, first is edge category id, second is idx of ref edge corresponding to edge category
        std::unordered_map<int, int> labelRootMap;
        std::unordered_map<int, int> edgeLabelMap = std::move(res.second); //-- edge-label map
        std::unordered_map<int, int> curRefMap = std::move(res.first);     //-- current frame-reference frame map

        //-- Perform union-find merge within reference frame (edges of reference frame itself are the same edge)
        for(const auto &pair : edgeLabelMap)
        {
            const int label = pair.second;
            const int edge_idx = pair.first; //-- This edge_idx must exist in local map

            if(labelRootMap.find(label) == labelRootMap.end())
            {
                labelRootMap[label] = edge_idx;
            }else{
                int root_idx = labelRootMap[label];

                //-- Map frame edge ID to map edge ID, and merge these edges using union-find
                unsigned int map_id_1 = frame_ref->mmEdgeIndex2ElementEdgeID.at(root_idx);
                unsigned int map_id_2 = frame_ref->mmEdgeIndex2ElementEdgeID.at(edge_idx);

                // * Union-find merge
                int map_idx_2 = mmElementID2index.at(map_id_2);
                int map_idx_1 = mmElementID2index.at(map_id_1);
                //-- Check if two elementEdges that need to be merged are originally of the same class
                //-- If originally same class, no additional processing needed; if different classes, need to merge
                unsigned int cluster_id_1 = findRoot(map_id_1);
                unsigned int cluster_id_2 = findRoot(map_id_2);
                if(cluster_id_1 != cluster_id_2)
                {
                    //-- Not same class, need to merge two clusters
                    int cluster_idx_1 = mmClusterID2index.at(cluster_id_1);
                    int cluster_idx_2 = mmClusterID2index.at(cluster_id_2);
                    
                    mergeElementCluster(cluster_idx_1, cluster_idx_2);

                }
            }
        }

        //-- Traverse association list of current frame, add frames that can be associated to Map
        for(const auto& pair: curRefMap){
            int cur_idx = pair.first;
            //-- If current edge has already been associated by other reference frame, skip this edge
            if(isAssociated[cur_idx]==true) continue;
            int ref_id = pair.second;
            int label_id = edgeLabelMap[ref_id];

            //-- If associated reference frame is invalid and not found in frame2MapElement, skip
            if(frame_ref->mmEdgeIndex2ElementEdgeID.find(ref_id)==frame_ref->mmEdgeIndex2ElementEdgeID.end()) continue;

            //-- For edges that have been associated, create elementEdge and union
            isAssociated[cur_idx] = true;
            
            elementEdge ele(frame_cur->KF_ID, cur_idx);
            int map_id_1 = frame_ref->mmEdgeIndex2ElementEdgeID.at(ref_id);
            
            ele.union_id = findRoot(map_id_1);
            
            mvElementEdges.push_back(ele);
            mmElementID2index[ele.element_id] = mvElementEdges.size()-1;
            frame_cur->mmEdgeIndex2ElementEdgeID[cur_idx] = ele.element_id;

            //-- Add current elementEdge to elementEdgeCluster
            unsigned int cluster_id = ele.union_id;
            if(mmClusterID2index.find(cluster_id)==mmClusterID2index.end())
            {
                std::cout<<"\033[31m [LOCAL MAPPING] \033[0m " << "can't locate root cluster" << std::endl;
            }
            int cluster_idx = mmClusterID2index[cluster_id];
            mvEleEdgeClusters[cluster_idx].mvElementEdgeIDs.push_back(ele.element_id);
            mvEleEdgeClusters[cluster_idx].mbModifiedCur = true;

        }
    }

    //-- Traverse unassociated edges of current frame, add them to map
    for(int i = 0; i < frame_cur->mvEdges.size(); ++i){
        if(isAssociated[i]== true) continue;
        //-- For unassociated edges, add to map
        int size = mvElementEdges.size();
        
        elementEdge ele(frame_cur->KF_ID, i);
        mvElementEdges.push_back(ele);
        mmElementID2index[ele.element_id] = mvElementEdges.size()-1;
        frame_cur->mmEdgeIndex2ElementEdgeID[i] = ele.element_id;

        unsigned int cluster_id = ele.element_id;
        std::vector<unsigned int> clusters;
        clusters.push_back(ele.element_id);
        elementEdgeCluster e_cluster(cluster_id, clusters);
        mvEleEdgeClusters.push_back(std::move(e_cluster));
        mmClusterID2index[cluster_id] = mvEleEdgeClusters.size() - 1;
        
    }
    //-- Add current frame to Map
    mvKeyFrames.push_back(frame_cur);
    mmKFID2KFindex[frame_cur->KF_ID] = mvKeyFrames.size()-1;

    //-- Count modified clusters, update count_not_update of all clusters
    for (auto& cluster : mvEleEdgeClusters)
    {
        if(cluster.mbModifiedCur == false)
        {
            cluster.count_not_update += 1;
        }else{
            //-- If modified, restart counting from 0
            cluster.count_not_update = 0;
        }
    }
    elementClusterCulling();
}

void localMap::mergeElementCluster(int cluster_idx_1, int cluster_idx_2)
{
    //-- Check which of the two clusters is larger
    elementEdgeCluster& cluster_1 = mvEleEdgeClusters[cluster_idx_1];
    elementEdgeCluster& cluster_2 = mvEleEdgeClusters[cluster_idx_2];
    int size_1 = cluster_1.mvElementEdgeIDs.size();
    int size_2 = cluster_2.mvElementEdgeIDs.size();

    if(size_1 > size_2)
    {
        //-- If size_1 is larger, merge cluster_2 into cluster_1
        // * STEP-1 Adjust root nodes of all edges in cluster_2
        for(auto& id : cluster_2.mvElementEdgeIDs)
        {
            int idx = mmElementID2index[id];
            mvElementEdges[idx].union_id = cluster_1.cluster_id; //-- cluster_id is the root node id
        }
        // * STEP-2 Move all edges from cluster_2 to cluster_1
        auto& vElementIDS_1 = cluster_1.mvElementEdgeIDs;
        vElementIDS_1.insert(vElementIDS_1.end(), cluster_2.mvElementEdgeIDs.begin(), cluster_2.mvElementEdgeIDs.end());
        
        // * STEP-3 Delete cluster_2
        mvEleEdgeClusters.erase(mvEleEdgeClusters.begin() + cluster_idx_2); // Delete i-th element
        mmClusterID2index.clear();
        for (int i = 0; i < mvEleEdgeClusters.size(); ++i)
        {
            unsigned int c_id = mvEleEdgeClusters[i].cluster_id;
            mmClusterID2index[c_id] = i;
        }
        // * STEP-4 Latest operation, so record that cluster_1 was modified this round
        cluster_1.mbModifiedCur = true;

    }else{
        //-- If size_2 is larger, merge cluster_1 into cluster_2
        // * STEP-1 Adjust root nodes of all edges in cluster_1
        for(auto& id : cluster_1.mvElementEdgeIDs)
        {
            int idx = mmElementID2index[id];
            mvElementEdges[idx].union_id = cluster_2.cluster_id; //-- cluster_id is the root node id
        }
        // * STEP-2 Move all edges from cluster_1 to cluster_2
        auto& vElementIDS_2 = cluster_2.mvElementEdgeIDs;
        vElementIDS_2.insert(vElementIDS_2.end(), cluster_1.mvElementEdgeIDs.begin(), cluster_1.mvElementEdgeIDs.end());
        
        // * STEP-3 Delete cluster_1
        mvEleEdgeClusters.erase(mvEleEdgeClusters.begin() + cluster_idx_1); // Delete i-th element
        mmClusterID2index.clear();
        for (int i = 0; i < mvEleEdgeClusters.size(); ++i)
        {
            unsigned int c_id = mvEleEdgeClusters[i].cluster_id;
            mmClusterID2index[c_id] = i;
        }
        // * STEP-4 Latest operation, so record that cluster_2 was modified this round
        cluster_2.mbModifiedCur = true;
    }
}

void localMap::elementClusterCulling()
{
    std::vector<int> ele_index_tobe_deleted;

    for(auto cluster_iter = mvEleEdgeClusters.begin(); cluster_iter != mvEleEdgeClusters.end(); )
    {
        if(cluster_iter->count_not_update > 3)
        {
            //-- Delete
            // * STEP-1 First delete all edges in this cluster
            int N = cluster_iter->mvElementEdgeIDs.size();
            std::vector<int> indicesToDelete;
            indicesToDelete.reserve(N);
            for(int i = 0; i < N; ++i)
            {
                //-- Delete elementEdge corresponding to ele_id
                unsigned int ele_id = cluster_iter->mvElementEdgeIDs[i];
                int ele_idx = mmElementID2index[ele_id];
                elementEdge& ele_edge = mvElementEdges[ele_idx];

                //-- substep-1 Delete keyframe index to this elementEdge
                KeyFramePtr pKF = mvKeyFrames.at(mmKFID2KFindex[ele_edge.kf_id]);
                pKF->mmEdgeIndex2ElementEdgeID.erase(ele_edge.kf_edge_idx);
                
                //-- substep-2 Record index of elementEdge to be deleted
                indicesToDelete.push_back(ele_idx);
            }
            //-- substep-3 Organize recorded elementEdges to be deleted
            ele_index_tobe_deleted.insert(ele_index_tobe_deleted.end(), indicesToDelete.begin(), indicesToDelete.end());

            // * STEP-2 Delete this cluster
            cluster_iter = mvEleEdgeClusters.erase(cluster_iter);
        }else{
            cluster_iter++;
        }
    }

    // * STEP-3 Perform unified deletion based on organized elementEdges to be deleted
    std::sort(ele_index_tobe_deleted.begin(), ele_index_tobe_deleted.end(), std::greater<int>());
    for(int idx : ele_index_tobe_deleted) {
        mvElementEdges.erase(mvElementEdges.begin() + idx);
    }
    //-- Reset index mapping of elementEdge
    mmElementID2index.clear();
    for(size_t i = 0; i < mvElementEdges.size(); ++i)
    {
        elementEdge& ele = mvElementEdges[i];
        mmElementID2index[ele.element_id] = i;
    }

    // * STEP-4 Reorganize ID-index mapping of clusters
    mmClusterID2index.clear();
    for (int i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        unsigned int c_id = mvEleEdgeClusters[i].cluster_id;
        mmClusterID2index[c_id] = i;
    }

}

void localMap::removeKeyFrameFront()
{
    assert(mvKeyFrames.size() > 0);

    KeyFramePtr pKF_del = mvKeyFrames[0];
    int kf_id = pKF_del->KF_ID;

    std::cout<<"remove frame "<<kf_id<<std::endl;

    // * STEP-1 Find all edges from this keyframe in elementEdge
    std::vector<int> indicesToDelete;
    indicesToDelete.reserve(pKF_del->mvEdges.size());

    std::set<unsigned int> clusters_tobe_culled;

    for(size_t i = 0; i < mvElementEdges.size(); ++i)
    {
        elementEdge& ele_curr = mvElementEdges[i];
        if(ele_curr.kf_id == kf_id)
        {
            indicesToDelete.push_back(i);
            //-- Find cluster of this elementEdge
            unsigned int cluster_id = findRoot(ele_curr.element_id);
            if(mmClusterID2index.find(cluster_id) != mmClusterID2index.end())
            {
                //-- Mark this cluster, it needs to be cleaned up later
                clusters_tobe_culled.insert(cluster_id);
            }else{
                std::cout<<cluster_id<<std::endl;
                std::cout<<"\033[31m [ERROR] \033[0m" << "locate a cluster_id that doesn't match root_id"<<std::endl;
            }
        }
    }

    // * STEP-2 Remove indices of these edges from clusters
    std::vector<int> clusterIdxToDelete;
    clusterIdxToDelete.reserve(mvEleEdgeClusters.size()/10);
    for(auto id = clusters_tobe_culled.begin(); id != clusters_tobe_culled.end(); ++id)
    {
        int index = mmClusterID2index[*id];
        elementEdgeCluster& cluster = mvEleEdgeClusters[index];
        std::vector<int> indicesToDelete_ele;
        int N = cluster.mvElementEdgeIDs.size();
        indicesToDelete_ele.reserve(N/2);
        
        for(int i = 0; i < N; ++i)
        {
            unsigned int ele_id = cluster.mvElementEdgeIDs[i];
            int ele_idx = mmElementID2index[ele_id];
            if(mvElementEdges[ele_idx].kf_id == kf_id)
            {
                //-- Mark this position
                indicesToDelete_ele.push_back(i);
            }
        }

        //-- Delete edge indices of this keyframe in this cluster
        std::sort(indicesToDelete_ele.begin(), indicesToDelete_ele.end(), std::greater<int>());
        for(int idx : indicesToDelete_ele) {
            cluster.mvElementEdgeIDs.erase(cluster.mvElementEdgeIDs.begin() + idx);
        }

        //-- If this cluster is empty, delete this cluster, mark here
        if(cluster.mvElementEdgeIDs.size()==0)
        {
            clusterIdxToDelete.push_back(index);
            continue;
        }

        //-- If this cluster is not empty, after deletion, cluster_id needs to be reassigned to prevent old root from being deleted
        //-- Let all elementEdges in this cluster use any edge as root
        int new_N = cluster.mvElementEdgeIDs.size();
        unsigned int ele_id_new_root = cluster.mvElementEdgeIDs[0];
        int ele_idx_new_root = mmElementID2index[ele_id_new_root];
        mvElementEdges[ele_idx_new_root].union_id = -1;
        for(size_t i = 1; i < new_N; ++i)
        {
            unsigned int ele_id = cluster.mvElementEdgeIDs[i];
            int ele_idx = mmElementID2index[ele_id];
            mvElementEdges[ele_idx].union_id = mvElementEdges[ele_idx_new_root].element_id;
        }
        //-- Reset cluster_id of this cluster
        if(mmClusterID2index.find(ele_id_new_root) == mmClusterID2index.end())
        {
            cluster.cluster_id = ele_id_new_root;
        }else{
            if(cluster.cluster_id != ele_id_new_root){
                //-- Theoretically impossible for other cluster's cluster_id to be id of elementEdge in this cluster
                std::cout<<"\033[31m [ERROR] \033[0m"<<"wrong cluster construction"<<std::endl;
            }
        }
    }

    //-- Delete clusters
    std::sort(clusterIdxToDelete.begin(), clusterIdxToDelete.end(), std::greater<int>());
    for(int idx : clusterIdxToDelete) {
        mvEleEdgeClusters.erase(mvEleEdgeClusters.begin() + idx);
    }
    //-- Reorganize id-index mapping of clusters
    mmClusterID2index.clear();
    for(int i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        unsigned int cluster_id = mvEleEdgeClusters[i].cluster_id;
        mmClusterID2index[cluster_id] = i;
    }


    // * STEP-3 Delete these edges from mvElementEdges
    //-- Delete from back to front to avoid breaking indices
    std::sort(indicesToDelete.begin(), indicesToDelete.end(), std::greater<int>());
    for(int idx : indicesToDelete) {
        mvElementEdges.erase(mvElementEdges.begin() + idx);
    }
    //-- Re-index mvElementEdges
    mmElementID2index.clear();
    for(size_t i = 0; i < mvElementEdges.size(); ++i)
    {
        elementEdge& ele = mvElementEdges[i];
        mmElementID2index[ele.element_id] = i;
    }


    // * STEP-4 Remove keyframe and clean up information related to keyframe and local map
    pKF_del->mmEdgeIndex2ElementEdgeID.clear();
    pKF_del->mmMapAssociations.clear();

    //-- Remove first frame
    mvKeyFrames.erase(mvKeyFrames.begin());
    mmKFID2KFindex.clear();
    for(int i = 0; i < mvKeyFrames.size(); ++i)
    {
        int ckf_id = mvKeyFrames[i]->KF_ID;
        mmKFID2KFindex[ckf_id] = i;
    }

    //-- Modify reference coordinate position of local_map to pose of current first frame
    // assert(mvKeyFrames.size()>0);
    // T_ref = mvKeyFrames[0]->KF_pose_g;
}

pcl::PointXYZ localMap::calcCloudCentroid(const pcl::PointCloud<pcl::PointXYZ>& pointCloud, pcl::PointXYZ current)
{
    pcl::PointXYZ NaN(0,0,0);
    pcl::PointXYZ centroid = current;
    // //-- Prevent current point itself from having bad depth, don't consider current point
    // centroid.x = 0;
    // centroid.y = 0;
    // centroid.z = 0;
    for (const auto& point : pointCloud.points) {  
        centroid.x += point.x;  
        centroid.y += point.y;  
        centroid.z += point.z;  
    }
    
    int numPoints = pointCloud.points.size() + 1;
    if(numPoints == 1){
        return NaN;
    }
    centroid.x /= numPoints;
    centroid.y /= numPoints;
    centroid.z /= numPoints;
    return centroid;  
}

void localMap::getMergedCluster(const std::vector<pcl::PointCloud<pcl::PointXYZ>>& clusterCloud,
                                  pcl::PointCloud<pcl::PointXYZ>& mergedCloud)
{
    //-- Find the longest edge
    int maxLength = 0;
    int maxIndex = -1;
    for (int i = 0; i < clusterCloud.size(); ++i){
        if (clusterCloud[i].size() > maxLength){
            maxLength = clusterCloud[i].size();
            maxIndex = i;
        }
    }

    //-- Sample the longest edge
    pcl::PointCloud<pcl::PointXYZ> sampled_longest_cloud;
    const int sampleBias = 3;
    const auto& longestCloud = clusterCloud[maxIndex]; 
    const size_t cloudSize = longestCloud.size();

    sampled_longest_cloud.reserve(cloudSize / sampleBias + 2);
    sampled_longest_cloud.push_back(longestCloud[0]);  // First point
    for(size_t i = sampleBias; i < cloudSize - 1; i += sampleBias) {
        sampled_longest_cloud.push_back(longestCloud[i]);
    }
    if(cloudSize > 1) {  // Last point
        sampled_longest_cloud.push_back(longestCloud.back());
    }


    //-- Use longest edge as reference edge, build KDtree
    pcl::PointCloud<pcl::PointXYZ>::Ptr p_ref_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    *p_ref_cloud = sampled_longest_cloud;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  
    kdtree.setInputCloud (p_ref_cloud);

    //-- Endpoints of longest edge, used to determine where newly added edges are inserted
    pcl::PointXYZ end_point_1 = p_ref_cloud->points[0];
    pcl::PointXYZ end_point_2 = p_ref_cloud->points.back();

    //-- Store information of other points associated with current sampled edge
    std::vector<pcl::PointCloud<pcl::PointXYZ>> associatedClouds;
    associatedClouds.resize(p_ref_cloud->points.size());
    
    //-- Find other associated parts related to this edge through foot point relationship
    for(int i = 0; i < clusterCloud.size(); ++i){
        std::vector<bool> isAssociated;
        if(i == maxIndex){
            continue;
        }
        //-- Traverse current edge, calculate foot point for each edge point
        for(int j = 0; j < clusterCloud[i].size(); ++j){
            //-- Nearest neighbor search to get reference point of perpendicular line
            pcl::PointXYZ query = clusterCloud[i][j];
            int K = 1;
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            kdtree.nearestKSearch(query, K, pointIdxNKNSearch, pointNKNSquaredDistance);
            int nearestIndex = pointIdxNKNSearch[0];
            float nearestDistance = pointNKNSquaredDistance[0];
            //-- Get endpoints of perpendicular line based on left-right direction
            int index_front = nearestIndex-1>=0 ? nearestIndex-1 : 0;
            int index_rear = nearestIndex+1<p_ref_cloud->points.size() ? nearestIndex+1 : p_ref_cloud->points.size()-1;
            pcl::PointXYZ point_a = p_ref_cloud->points[index_front];
            pcl::PointXYZ point_b = p_ref_cloud->points[index_rear];
            //-- Calculate foot point
            Eigen::Vector3d bias;
            bias(0) = point_a.x - point_b.x;
            bias(1) = point_a.y - point_b.y;
            bias(2) = point_a.z - point_b.z;
            Eigen::Vector3d connect;
            connect(0) = point_b.x - query.x;
            connect(1) = point_b.y - query.y;
            connect(2) = point_b.z - query.z;
            double length = - connect.dot(bias)/(bias.norm() * bias.norm());
            pcl::PointXYZ foot;
            foot.x = length*bias(0) + point_b.x;
            foot.y = length*bias(1) + point_b.y;
            foot.z = length*bias(2) + point_b.z;
            //-- Check if foot point is between two points
            if(length < 0 || length > 1){//-- Foot point not between two points, indicates point association is off
                isAssociated.push_back(false);
            }else{//-- Foot point is correct
                isAssociated.push_back(true);
                //-- Based on association relationship, update point cloud associated with reference edge, only update within certain distance threshold
                if(nearestDistance < 0.03){
                    int asso_idx = nearestIndex;
                    float nearest_foot_dist = 0;
                    pcl::PointXYZ point_n = p_ref_cloud->points[nearestIndex];
                    //-- Determine which of point_a, point_b, point_n the foot point is closest to
                    double distance_1 = pcl::euclideanDistance(foot, point_n);
                    double distance_2 = pcl::euclideanDistance(foot, point_a);
                    double distance_3 = pcl::euclideanDistance(foot, point_b);
                    if(distance_1 <= distance_2 && distance_1 <= distance_3){
                        asso_idx = nearestIndex;
                        nearest_foot_dist = distance_1;
                    }else if(distance_2 <= distance_1 && distance_2 <= distance_3){
                        asso_idx = index_front;
                        nearest_foot_dist = distance_2;
                    }else if(distance_2 <= distance_1 && distance_2 <= distance_3){
                        asso_idx = index_rear;
                        nearest_foot_dist = distance_3;
                    }

                    if(nearest_foot_dist < 0.02){
                        associatedClouds[asso_idx].points.push_back(query);
                    }
                }
            }
        }

        // //-- Find longest segment of false associations in current edge
        // int maxLength = 0, currentLength = 0;
        // int start_index = -1, end_index = -1;
        // int final_start_index;
        // for(int j = 0; j < isAssociated.size(); ++j){
        //     bool b = isAssociated[j];
        //     if(b == false){
        //         //-- If false appears for the first time or changes from true to false, record this index
        //         if(currentLength == 0) start_index = j;
        //         currentLength++;
        //     }else{
        //         if(currentLength > maxLength){
        //             maxLength = currentLength;
        //             final_start_index = start_index;
        //             end_index = j;
        //         }
        //         currentLength = 0;
        //     }
        // }
        // //-- After traversing once, if the end is all false and maximum, adjust it
        // if(currentLength > maxLength){
        //     maxLength = currentLength;
        //     final_start_index = start_index;
        //     end_index = isAssociated.size();
        // }
        // //-- Now the part between final_start_index and end_index-1 is the part that needs to be added
        // int false_length = end_index - final_start_index;
        // //-- Don't take ones that are too short
        // if(false_length < 5){
        //     continue;
        // }

        // //-- Organize the extra segment
        // pcl::PointCloud<pcl::PointXYZ> addCloud;
        // for(int j = final_start_index; j < end_index; ++j){
        //     if(j == final_start_index || j == end_index - 1 || (j-final_start_index)%3==0){
        //         addCloud.push_back(clusterCloud[i].points[j]);
        //     }
        // }

        // //-- Association relationship of the extra segment also needs to be modified
        // std::vector<pcl::PointCloud<pcl::PointXYZ>> add_associatedClouds;
        // add_associatedClouds.resize(addCloud.size());

        // //-- Determine where this segment should be inserted
        // pcl::PointXYZ end_point_3 = addCloud.points[0];
        // pcl::PointXYZ end_point_4 = addCloud.points.back();
        // double dst1 = sqrt((end_point_3.x-end_point_1.x)*(end_point_3.x-end_point_1.x)
        //                   +(end_point_3.y-end_point_1.y)*(end_point_3.y-end_point_1.y)
        //                   +(end_point_3.z-end_point_1.z)*(end_point_3.z-end_point_1.z));
        // double dst2 = sqrt((end_point_3.x-end_point_2.x)*(end_point_3.x-end_point_2.x)
        //                   +(end_point_3.y-end_point_2.y)*(end_point_3.y-end_point_2.y)
        //                   +(end_point_3.z-end_point_2.z)*(end_point_3.z-end_point_2.z));
        // double dst3 = sqrt((end_point_4.x-end_point_1.x)*(end_point_4.x-end_point_1.x)
        //                   +(end_point_4.y-end_point_1.y)*(end_point_4.y-end_point_1.y)
        //                   +(end_point_4.z-end_point_1.z)*(end_point_4.z-end_point_1.z));
        // double dst4 = sqrt((end_point_4.x-end_point_2.x)*(end_point_4.x-end_point_2.x)
        //                   +(end_point_4.y-end_point_2.y)*(end_point_4.y-end_point_2.y)
        //                   +(end_point_4.z-end_point_2.z)*(end_point_4.z-end_point_2.z));
        // pcl::PointCloud<pcl::PointXYZ> cloud_merge = *p_ref_cloud;
        // if(dst1 <= dst2 && dst1 <= dst3 && dst1 <= dst4){
        //     //-- Insertion order is 4--3|1--2
        //     std::reverse(addCloud.begin(), addCloud.end());
        //     cloud_merge = addCloud + cloud_merge;
        //     associatedClouds.insert(associatedClouds.begin(), 
        //                             add_associatedClouds.begin(), add_associatedClouds.end());
        // }else if(dst2 <= dst1 && dst2 <= dst3 && dst2 <= dst4){
        //     //-- Insertion order is 1--2|3--4
        //     cloud_merge = cloud_merge + addCloud;
        //     associatedClouds.insert(associatedClouds.end(), 
        //                             add_associatedClouds.begin(), add_associatedClouds.end());
        // }else if(dst3 <= dst1 && dst3 <= dst2 && dst3 <= dst4){
        //     //-- Insertion order is 3--4|1--2
        //     cloud_merge = addCloud + cloud_merge;
        //     associatedClouds.insert(associatedClouds.begin(), 
        //                             add_associatedClouds.begin(), add_associatedClouds.end());
        // }else if(dst4 <= dst1 && dst4 <= dst2 && dst4 <= dst3){
        //     //-- Insertion order is 1--2|4--3
        //     std::reverse(addCloud.begin(), addCloud.end());
        //     cloud_merge = cloud_merge + addCloud;
        //     associatedClouds.insert(associatedClouds.end(), 
        //                             add_associatedClouds.begin(), add_associatedClouds.end());
        // }

        // //-- Update kdtree
        // *p_ref_cloud = cloud_merge;
        // pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_new;  
        // kdtree_new.setInputCloud (p_ref_cloud);
        // kdtree = kdtree_new;
        // end_point_1 = p_ref_cloud->points[0];
        // end_point_2 = p_ref_cloud->points.back();

        // //-- Update isAssociated
        // for(int j = final_start_index; j < end_index; ++j){
        //     isAssociated[j] = true;
        // }

    } 
    // //-- Calculate centroid of associations
    pcl::PointCloud<pcl::PointXYZ> cloud_adjust;
    pcl::PointXYZ NaN(0,0,0);
    for(int cnt = 0; cnt < p_ref_cloud->points.size(); ++cnt)
    {
        pcl::PointXYZ centroid = calcCloudCentroid(associatedClouds[cnt], p_ref_cloud->points[cnt]);
        if(centroid.x == 0 && centroid.y == 0 && centroid.z == 0){
            continue; //-- Encounter NaN point without association, ignore this point
        }
        cloud_adjust.push_back(centroid);
        //cloud_adjust.push_back(p_ref_cloud->points[cnt]);
    }

    mergedCloud = cloud_adjust;
    return;
}

void localMap::clustersFitting3D()
{
    //-- Traverse all clusters
    for(size_t i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        
        elementEdgeCluster& cluster = mvEleEdgeClusters[i];
        
        const std::vector<unsigned int>& ele_ids = cluster.mvElementEdgeIDs;
        if(ele_ids.size() < mvKeyFrames.size()/2)continue;

        std::vector<int> ele_indices(ele_ids.size(), -1);
        for(size_t j = 0; j < ele_ids.size(); ++j)
        {
            ele_indices[j] = mmElementID2index[ele_ids[j]];
        }

        std::vector<pcl::PointCloud<pcl::PointXYZ>> clusterCloud;
        clusterCloud.reserve(ele_indices.size());
        //-- Construct
        int N = ele_indices.size();
        for(size_t j = 0; j < N; ++j)
        {
            elementEdge& ele_edge = mvElementEdges.at(ele_indices[j]);
            int kf_id = ele_edge.kf_id;
            int edge_idx = ele_edge.kf_edge_idx;
            int kf_idx = mmKFID2KFindex.at(kf_id);
            Edge& edge = mvKeyFrames[kf_idx]->mvEdges[edge_idx];

            Sophus::SE3d T_ref_cur = mvKeyFrames[kf_idx]->KF_pose_g;
            
            pcl::PointCloud<pcl::PointXYZ> cloud;
            for(size_t k = 0; k < edge.mvPoints.size(); ++k)
            {
                const orderedEdgePoint& pt = edge.mvPoints[k];
                //-- Calculate 3D coordinates
                Eigen::Vector3d point_3d(pt.x_3d,pt.y_3d,pt.z_3d);
                //-- Reproject to get new projection points
                point_3d = T_ref_cur * point_3d;
                pcl::PointXYZ point(point_3d.x(),point_3d.y(),point_3d.z());
                cloud.points.push_back(point);
            }
            clusterCloud.push_back(cloud);
        }
        //-- merged_cloud in T_ref coordinate frame
        pcl::PointCloud<pcl::PointXYZ> merged_cloud;
        getMergedCluster(clusterCloud, merged_cloud);

        //-- Convert pcl::PointCloud type to opencv point cloud type
        std::vector<cv::Point3d> result;
        result.reserve(merged_cloud.size());
        
        const pcl::PointXYZ* data = merged_cloud.points.data();
        const size_t size = merged_cloud.size();
        for (size_t i = 0; i < size; ++i) {
            result.emplace_back(data[i].x, data[i].y, data[i].z);
        }
        //-- Store merged result in cluster
        cluster.mvMergedCloud_ref = std::move(result);
        cluster.mbMerged = true; 

    }
}




void localMap::clusterFittingProjection()
{
    //-- Clear before reconstructing
    for(size_t i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        elementEdgeCluster& cluster = mvEleEdgeClusters[i];
        cluster.mbMerged = false;
        cluster.mvMergedCloud_ref.clear();
        cluster.dist_thres = -1.0;
    }

    //-- Traverse all clusters
    for(size_t i = 0; i < mvEleEdgeClusters.size(); ++i)
    {
        
        elementEdgeCluster& cluster = mvEleEdgeClusters[i];
        
        const std::vector<unsigned int>& ele_ids = cluster.mvElementEdgeIDs;
        if(ele_ids.size() < 5)continue;

        std::vector<int> ele_indices(ele_ids.size(), -1);
        for(size_t j = 0; j < ele_ids.size(); ++j)
        {
            ele_indices[j] = mmElementID2index[ele_ids[j]];
        }

        std::vector<elementEdge> clusterElement;
        std::vector<int> kf_indices;
        clusterElement.reserve(ele_indices.size());
        kf_indices.reserve(ele_indices.size());
        //-- Construct
        int N = ele_indices.size();
        for(size_t j = 0; j < N; ++j)
        {
            elementEdge& ele_edge = mvElementEdges.at(ele_indices[j]);
            int kf_id = ele_edge.kf_id;
            int kf_idx = mmKFID2KFindex.at(kf_id);
            clusterElement.push_back(ele_edge);
            kf_indices.push_back(kf_idx);
        }
        //-- merged_cloud in T_ref coordinate frame
        std::vector<cv::Point3d> merged_cloud;
        double dist_thres;
        std::vector<int> involved_elements;
        //-- featureMerger::getMergedClusterProjection(clusterElement, kf_indices, mvKeyFrames, merged_cloud, involved_elements);
        // featureMerger::getMergedClusterIncremental(clusterElement, kf_indices, mvKeyFrames, merged_cloud, involved_elements);
        featureMerger::getMergedClusterIterative(clusterElement, kf_indices, mvKeyFrames, merged_cloud, dist_thres, involved_elements);
        //-- Store merged result in cluster
        cluster.mvMergedCloud_ref = std::move(merged_cloud);
        cluster.mvInvolvedLocalMapElementIndices = std::move(involved_elements);
        cluster.mbMerged = true; 
        cluster.dist_thres = dist_thres;

    }

    assignWeights();
}

void localMap::assignWeights()
{
    std::vector<double> dist_thres_values;
    for (auto& cluster : mvEleEdgeClusters)
    {
        if (cluster.mbMerged == false) continue;
        dist_thres_values.push_back(cluster.dist_thres);
        cluster.weightBA = 1.0;
    }

    // 1. Check if input is empty
    if (dist_thres_values.empty()) {
        return;
    }

    // 2. Create sorted copy (from large to small)
    std::vector<double> sorted_values = dist_thres_values;
    std::sort(sorted_values.begin(), sorted_values.end(), std::greater<double>());

    // 3. Calculate mean value
    double sum = std::accumulate(sorted_values.begin(), sorted_values.end(), 0.0);
    double mean = sum / sorted_values.size();

    for (auto& cluster : mvEleEdgeClusters)
    {
        if (cluster.mbMerged == false) continue;
        double curr_dist = cluster.dist_thres;

        // 4. If dist is less than or equal to mean, weight is 1
        if (curr_dist <= mean) {
            cluster.weightBA = 1.0;
            continue;
        }

        // 5. Calculate relative position (percentile) of dist in sorted list
        auto it = std::lower_bound(sorted_values.begin(), sorted_values.end(), curr_dist, std::greater<double>());
        double rank = std::distance(sorted_values.begin(), it);
        double percentile = rank / sorted_values.size();

        // 6. Design weight function: larger dist results in smaller weight, decay speed gradually slows down
        // Use exponential decay function, but adjust decay rate
        double normalized_dist = (curr_dist - mean) / (sorted_values.front() - mean);
        double weight = std::exp(-normalized_dist * (1.0 + percentile));
        cluster.weightBA = weight;
    }
}


void localMap::getAssoFrameMergeEdge(int kf_id_dst, std::vector<match3d_2d>& matches, std::vector<double>& weights)
{
    matches.clear();
    weights.clear();

    for (const auto& cluster : mvEleEdgeClusters)
    {
        if (cluster.mbMerged == false) continue;
        if (cluster.weightBA <= 0) continue;
        
        std::vector<cv::Point3d> merged_cloud = cluster.mvMergedCloud_ref;
        std::vector<elementEdge> associated_ele_edges;

        //-- Edges that constitute each cluster
        std::vector<int> involved_ele_indices = cluster.mvInvolvedLocalMapElementIndices;
        if(involved_ele_indices.empty())
        {
            continue;
        }

        for (size_t i = 0; i < involved_ele_indices.size(); ++i)
        {
            int idx = involved_ele_indices[i];
            unsigned int ele_id = cluster.mvElementEdgeIDs[idx];

            int ele_idx = mmElementID2index.at(ele_id);
            elementEdge& ele_edge = mvElementEdges[ele_idx];
            int kf_id = ele_edge.kf_id;
            int kf_edge_index = ele_edge.kf_edge_idx;

            if(kf_id == kf_id_dst)
            {
                associated_ele_edges.push_back(ele_edge);
            }
        }

        if( !associated_ele_edges.empty())
        {
            match3d_2d match;
            match.first = merged_cloud;
            match.second = associated_ele_edges;
            
            double weight = cluster.weightBA;

            matches.push_back(match);
            weights.push_back(weight);
        }
    }
}

