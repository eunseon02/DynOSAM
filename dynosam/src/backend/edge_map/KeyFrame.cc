#include "dynosam/backend/edge_map/KeyFrame.hpp"
#include <glog/logging.h>

namespace dyno {

KeyFrame::KeyFrame(int ID, Sophus::SE3d pose, double stamp, std::vector<Edge> vEdges, const cv::Mat& matRGB, const cv::Mat& matDepth,
    const float& fx, const float& fy, const float& cx, const float& cy)
{
    // Set keyframe ID
    KF_ID = ID;

    // Set pose and timestamp
    KF_stamp = stamp;
    KF_pose_g = pose;

    // Set camera intrinsics
    mCx = cx;
    mCy = cy;
    mFx = fx;
    mFy = fy;

    // Assign original frame image
    cv::cvtColor(matRGB, mMatGray, cv::COLOR_BGR2GRAY);
    mHeight = mMatGray.rows;
    mWidth =  mMatGray.cols;

    mvEdges = std::move(vEdges);

    // Depth-related preprocessing: remove inconsistent edge features
    assignProperty3D(matDepth); // Assign depth to edges
    
    // edgeCullingDepth();         // Remove all edges with invalid depth and invalid edge points in valid edges
    edgeCullingDepthParallel();
    
    // All edge points in remaining edges now have valid depth
    edgeCullingContinuity();    // Ensure 3D point depth continuity for each ordered edge

    // Update frame_edge_ID and frame_point_index for each edge point in this frame
    assignPropertyIdx();
    
    //constructSearchPlain();
    constructSearchPlainParallel();
}

void KeyFrame::searchRadius(float x, float y, double radius, std::vector<orderedEdgePoint>& result)
{
    result.clear();

    // Define rectangular boundary of search region (integer pixel coordinates)
    int minX = static_cast<int>(std::max(0.0, x - radius));
    int maxX = static_cast<int>(std::min(mMatSearch.cols - 1.0, x + radius));
    int minY = static_cast<int>(std::max(0.0, y - radius));
    int maxY = static_cast<int>(std::min(mMatSearch.rows - 1.0, y + radius));

    // Distance from searched points to (x,y)
    std::vector<float> list_distance;

    // Traverse all pixels in search region, find edge points within radius (excluding (-1,-1) invalid points)
    for(int py = minY; py <= maxY; ++py)
    {
        for(int px = minX; px <= maxX; ++px)
        {
            const cv::Vec2i& pixel = mMatSearch.at<cv::Vec2i>(py, px);
            int edgeID = pixel[0];     //-- frame_edge_ID
            int pointIdx = pixel[1];   //-- frame_point_index

            // Skip invalid points
            if (edgeID == -1 || pointIdx == -1) continue;

            // Calculate distance (Euclidean distance)
            float dx = px - x;
            float dy = py - y;
            float distance = std::sqrt(dx * dx + dy * dy);

            // If distance is within radius, add to result
            if (distance <= radius)
            {
                // Get original point data
                const auto& edge = mvEdges[mmIndexMap.at(edgeID)];
                orderedEdgePoint point = edge.mvPoints[pointIdx];

                // Ensure frame_edge_ID and frame_point_index of original point data match search results
                assert(point.frame_edge_ID == edgeID && point.frame_point_index == pointIdx);

                result.push_back(point);
                float distance = sqrt((point.x-x)*(point.x-x) + (point.y-y)*(point.y-y));
                list_distance.push_back(distance);
            }
        }
    }

    assert(list_distance.size() == result.size());

    // Create index array
    std::vector<size_t> indices(result.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort index array by distance of neighbor points relative to (x,y)
    std::sort(indices.begin(), indices.end(), 
              [&list_distance](size_t i, size_t j) { return list_distance[i] < list_distance[j]; });

    // Rearrange result according to sorted indices
    std::vector<orderedEdgePoint> sorted_result;
    sorted_result.reserve(result.size()); 
    for (size_t i : indices) {
        sorted_result.push_back(result[i]);
    }
    result = std::move(sorted_result);
}

bool KeyFrame::isPointsAssociated(const orderedEdgePoint& pt1, const orderedEdgePoint& pt2)
{
    float res = fabs(pt1.imgGradAngle-pt2.imgGradAngle);
    if(res > 180) res = 360 - res;
    // Gradient direction consistency association
    if(res<10.0){
        return true;
    }else{
        return false;
    }
}

bool KeyFrame::isPointsAssociatedAlign(const orderedEdgePoint& warped_pt, 
                                       const orderedEdgePoint& neighbor_pt, 
                                       const float& depth_warp)
{
    bool isAngleValid = isPointsAssociated(warped_pt, neighbor_pt);
    float depth_orig = neighbor_pt.depth;
    bool isDepthValid = false;
    if(std::abs(depth_orig - depth_warp) < 0.01)
    {
        isDepthValid = true;
    }
    return (isAngleValid && isDepthValid);
}

std::vector<int> KeyFrame::edgeWiseCorrespondenceReproject(Edge& query_edge, const Sophus::SE3d& T2curr)
{

    // * STEP 1. Reproject reference frame edges to current frame coordinates
    std::vector<orderedEdgePoint>& queryList = query_edge.mvPoints;
    std::vector<cv::Point> warped_queryList;
    std::vector<float> warped_depthList;
    const size_t num_points = queryList.size();
    
    warped_queryList.reserve(num_points);
    warped_depthList.reserve(num_points);

    // Precompute inverse of camera intrinsics (reduce division operations)
    const float inv_fx = 1.0f / mFx;
    const float inv_fy = 1.0f / mFy;

    for(int i = 0; i < num_points; ++i)
    {
        // Recover 3D point from pixel and depth values
        const auto& pt = queryList[i];
        float z = pt.depth;
        float x = (static_cast<float>(pt.x) - mCx) * inv_fx * z;
        float y = (static_cast<float>(pt.y) - mCy) * inv_fy * z;
        
        // Reproject to get new projected point
        Eigen::Vector3d point = T2curr * Eigen::Vector3d(x, y, z);
        warped_queryList.emplace_back(
            mFx * point.x() / point.z() + mCx,
            mFy * point.y() / point.z() + mCy
        );
        warped_depthList.emplace_back(point.z());
    }

    // * STEP 2. Radius neighborhood search and voting to find which edge each query edge point most wants to associate with

    // first: current frame edge ID, second: number of query edge points that want to associate with this current frame edge
    std::map<int, int> edgeVoteMapTotal;
    const float radius = 6.0f;
    const int threshold_value = std::min(static_cast<int>(num_points * 0.3f), 5);

    for(int i = 0; i < num_points; ++i)
    {
        orderedEdgePoint& pt = queryList[i];
        float x = warped_queryList[i].x;
        float y = warped_queryList[i].y;
        float depth = warped_depthList[i];
        std::vector<orderedEdgePoint> neighbors_points;
        searchRadius(x, y, radius, neighbors_points);

        // Pre-store neighbor matching relationships for this point
        pt.mvAssoFrameEdgeIDs.clear();
        pt.mvAssoFramePointIndices.clear();
        pt.mvAssoFrameEdgeIDs.reserve(neighbors_points.size());
        pt.mvAssoFramePointIndices.reserve(neighbors_points.size());

        // For each point, build a vote to find which edge this point most tends to associate with
        std::unordered_map<int, int> edgeVoteMap;
        for (const auto& neighbor : neighbors_points) 
        {
            // if (isPointsAssociated(pt, neighbor)) 
            if (isPointsAssociatedAlign(pt, neighbor, depth)) 
            {
                // Direct increment, avoid find check
                edgeVoteMap[neighbor.frame_edge_ID]++;
                // After confirming association, update association cache
                pt.mvAssoFrameEdgeIDs.push_back(neighbor.frame_edge_ID);
                pt.mvAssoFramePointIndices.push_back(neighbor.frame_point_index);
            }
        }

        if (!edgeVoteMap.empty()) 
        {
            // Find edge with most votes, max_pair.first is the edge that current query point most wants to associate with
            const auto max_pair = *std::max_element(
                edgeVoteMap.begin(), edgeVoteMap.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; }
            );
            // Each point has only one most preferred edge to associate with
            edgeVoteMapTotal[max_pair.first] += 1;
        }
    }

    // * STEP 3. Organize votes to determine which current edges the query edge can associate with
    // edgeVoteMapTotal now contains voting relationships between query edge and candidate edges
    
    std::vector<int> result;
    result.reserve(edgeVoteMapTotal.size());  // Pre-allocate memory
    
    // Find current frame edges that meet threshold requirements for association
    for (const auto& [edge_id, votes] : edgeVoteMapTotal) 
    {
        if(votes > threshold_value) result.push_back(edge_id);
    }

    if (result.empty()) {
        return result;
    }

    // * STEP 4: Update association relationships (use hash table to speed up lookup)
    const std::unordered_set<int> validAssociation(result.begin(), result.end());
    for(auto& pt : query_edge.mvPoints) 
    {
        // Directly modify original data, avoid copying
        for(size_t j = 0; j < pt.mvAssoFrameEdgeIDs.size(); ++j) 
        {
            if(validAssociation.count(pt.mvAssoFrameEdgeIDs[j])) 
            {
                pt.asso_edge_ID = pt.mvAssoFrameEdgeIDs[j];
                pt.asso_point_index = pt.mvAssoFramePointIndices[j];
                pt.mbAssociated = true;
                break;
            }
        }
        // Clear memory (use swap to ensure memory release)
        std::vector<int>().swap(pt.mvAssoFrameEdgeIDs);
        std::vector<int>().swap(pt.mvAssoFramePointIndices);
    }
    
    return result;
}

std::vector<int> KeyFrame::edgeWiseCorrespondenceLocalMapping(Edge& query_edge, const Sophus::SE3d& T2curr)
{

    // * STEP 1. Reproject current frame edge to reference frame coordinates
    std::vector<orderedEdgePoint>& queryList = query_edge.mvPoints;
    std::vector<cv::Point> warped_queryList;
    std::vector<float> warped_depthList;
    const size_t num_points = queryList.size();

    warped_queryList.reserve(num_points);
    warped_depthList.reserve(num_points);

    // Precompute inverse of camera intrinsics (reduce division operations)
    const float inv_fx = 1.0f / mFx;
    const float inv_fy = 1.0f / mFy;

    for(int i = 0; i < num_points; ++i)
    {
        // Recover 3D point from pixel and depth values
        const auto& pt = queryList[i];
        float z = pt.depth;
        float x = (static_cast<float>(pt.x) - mCx) * inv_fx * z;
        float y = (static_cast<float>(pt.y) - mCy) * inv_fy * z;
        
        // Reproject to get new projected point
        Eigen::Vector3d point = T2curr * Eigen::Vector3d(x, y, z);
        warped_queryList.emplace_back(
            mFx * point.x() / point.z() + mCx,
            mFy * point.y() / point.z() + mCy
        );
        warped_depthList.emplace_back(point.z());
    }

    // * STEP 2. Radius neighborhood search and voting to find which edge each query edge point most wants to associate with

    // first: current frame edge ID, second: number of query edge points that want to associate with this current frame edge
    std::map<int, int> edgeVoteMapTotal;
    const float radius = 2.0f;
    const int threshold_value = static_cast<int>(num_points * 0.3f);

    for(int i = 0; i < num_points; ++i)
    {
        orderedEdgePoint& pt = queryList[i];
        float x = warped_queryList[i].x;
        float y = warped_queryList[i].y;
        float depth = warped_depthList[i];
        std::vector<orderedEdgePoint> neighbors_points;
        searchRadius(x, y, radius, neighbors_points);

        // For each point, build a vote to find which edge this point most tends to associate with
        std::unordered_map<int, int> edgeVoteMap;
        for (const auto& neighbor : neighbors_points) 
        {
            // if (isPointsAssociated(pt, neighbor)) 
            if (isPointsAssociatedAlign(pt, neighbor, depth))
            {
                // Direct increment, avoid find check
                edgeVoteMap[neighbor.frame_edge_ID]++;
            }
        }

        if (!edgeVoteMap.empty()) 
        {
            // Find edge with most votes, max_pair.first is the edge that current query point most wants to associate with
            const auto max_pair = *std::max_element(
                edgeVoteMap.begin(), edgeVoteMap.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; }
            );
            // Each point has only one most preferred edge to associate with
            edgeVoteMapTotal[max_pair.first] += 1;
        }
    }

    // * STEP 3. Organize votes to determine which current edges the query edge can associate with
    // edgeVoteMapTotal now contains voting relationships between query edge and candidate edges
    
    std::vector<int> result;
    result.reserve(edgeVoteMapTotal.size());  // Pre-allocate memory
    
    // Find current frame edges that meet threshold requirements for association
    for (const auto& [edge_id, votes] : edgeVoteMapTotal) 
    {
        // Check if edge corresponding to edge_id is in local map
        int edge_index = mmIndexMap[edge_id];
        if(mmEdgeIndex2ElementEdgeID.find(edge_index) == mmEdgeIndex2ElementEdgeID.end())
        {
            // Skip if not in local map
            continue;
        }else{
            // Only allow association if in local map
            if(votes > threshold_value) result.push_back(edge_id);
        }
    }
    
    // No need to update association relationships, directly return edge-level results
    return result;
}

void KeyFrame::assignPropertyIdx()
{
    // Construct ID to index mapping based on edge IDs
    for(size_t i = 0; i < mvEdges.size(); ++i)
    {
        // Update mapping relationship between edge_id and edge index in mvEdges
        const int edge_id = mvEdges[i].edge_ID;

        if(mmIndexMap.find(edge_id) != mmIndexMap.end())
        {
            std::cout<<"\033[31m"<<"[ERROR]"<<"\033[0m"<<
            " WRONG EDGE POINT INDEX "<<edge_id<<", INDICES SHOULD BE DIFFERENT!"<<std::endl;
            continue;
        }else{
            mmIndexMap[edge_id] = i;
        }

        auto& edge = mvEdges[i];

        // For each edge point in the edge, update its index to all edges in the frame
        for(int j = 0; j < edge.mvPoints.size(); ++j)
        {
            auto& point = edge.mvPoints[j];
            // Update edge ID index
            point.frame_edge_ID = edge_id;
            // Update edge point list index
            point.frame_point_index = static_cast<int>(j);
        }
    }
}

// Build search array: put all points from mvEdges into a cv::Mat
void KeyFrame::constructSearchPlain()
{
    // Create a CV_32SC2 type Mat, initial value set to (-1, -1) to indicate invalid positions
    mMatSearch = cv::Mat(mHeight, mWidth, CV_32SC2, cv::Scalar(-1, -1));

    for (size_t i = 0; i < mvEdges.size(); ++i) 
    {
        const auto& edge = mvEdges[i];
        for (size_t j = 0; j < edge.mvPoints.size(); ++j) 
        {
            const auto& point = edge.mvPoints[j];
            
            // Ensure coordinates are within image bounds
            if(point.x >= 0 && point.x < mWidth && point.y >= 0 && point.y < mHeight){
                // Access specified position and assign value
                auto& pixel = mMatSearch.at<cv::Vec2i>(point.y, point.x);
                pixel[0] = point.frame_edge_ID;      // First channel stores edge ID
                pixel[1] = point.frame_point_index;  // Second channel stores point index
            }else{
                std::cerr << "Point (" << point.x << ", " << point.y 
                          << ") out of bounds!" << std::endl;
            }
        }
    }
}

void KeyFrame::constructSearchPlainParallel()
{
    // Create and initialize matrix
    mMatSearch = cv::Mat(mHeight, mWidth, CV_32SC2, cv::Scalar(-1, -1));
    
    // Use parallel_for_each to process all edges in parallel
    tbb::parallel_for_each(mvEdges.begin(), mvEdges.end(),
        [&](const auto& edge) {
            // Traverse all points of current edge
            for (const auto& point : edge.mvPoints) {
                // Directly write to matrix
                auto& pixel = mMatSearch.at<cv::Vec2i>(point.y, point.x);
                pixel[0] = point.frame_edge_ID;      // Store edge ID
                pixel[1] = point.frame_point_index; // Store point index
            }
        });
}

cv::Mat KeyFrame::visualizeSearchPlain()
{
    std::map<int, std::vector<cv::Point>> edgeMap;

    for (int y = 0; y < mMatSearch.rows; ++y) {
        for (int x = 0; x < mMatSearch.cols; ++x) {
            const cv::Vec2i& pixel = mMatSearch.at<cv::Vec2i>(y, x);
            int edgeID = pixel[0];  // Channel 1: frame_edge_ID
            if (edgeID != -1) {     // Ignore invalid points (-1,-1)
                edgeMap[edgeID].emplace_back(x, y);
            }
        }
    }
    // Step 2: Create color image (3-channel BGR)
    cv::Mat colorMat(mMatSearch.size(), CV_8UC3, cv::Scalar(0, 0, 0)); // Default black

    // Step 3: Generate random color for each edgeID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(50, 255); // Avoid too dark colors

    for (const auto& [edgeID, points] : edgeMap) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen)); // Random BGR color

        // Draw all points for this edgeID
        for (const auto& pt : points) {
            colorMat.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(
                static_cast<uchar>(color[0]),
                static_cast<uchar>(color[1]),
                static_cast<uchar>(color[2])
            );
        }
    }

    return colorMat;
}

void KeyFrame::assignProperty3D(const cv::Mat& matDepth)
{
    // Parallelize outer loop
    tbb::parallel_for(0, (int)mvEdges.size(), [&](int i) {
        // Inner loop remains serial
        for(int j = 0; j < mvEdges[i].mvPoints.size(); ++j) {
            assignProperty3DEach(mvEdges[i].mvPoints[j], matDepth);
        }
    });

    // for(int i = 0; i < mvEdges.size(); ++i)
    // {
    //     for(int j = 0; j < mvEdges[i].mvPoints.size(); ++j)
    //     {
    //         assignProperty3DEach(mvEdges[i].mvPoints[j], matDepth);
    //     }
    // }
}

void KeyFrame::assignProperty3DEach(orderedEdgePoint& pt, const cv::Mat& matDepth)
{
    int x_idx = pt.x;
    int y_idx = pt.y;

    // Original point's true depth
    // Note: matDepth should already be converted to CV_32F in createKeyFrameFromFrame
    // So we can simply read as float, just like in keyframe.cpp
    float depth_orig = matDepth.at<float>(y_idx, x_idx);

    // Compute adjusted depth and visibility score in 5x5 patch
    std::vector<float> validDepthList; // List of depths for all points with non-zero depth
    int patch_total = 0;               // Total number of pixels in current patch
    for(int x_bias = -2; x_bias <= 2; ++x_bias){
        for(int y_bias = -2; y_bias <= 2; ++y_bias){
            int curr_x_idx = x_idx + x_bias;
            int curr_y_idx = y_idx + y_bias;
            
            // Check if this position is within image region
            if(curr_x_idx < 0 || curr_x_idx >= mWidth ||
               curr_y_idx < 0 || curr_y_idx >= mHeight) continue;

            patch_total += 1; // Accumulate total pixels
            // If in image region, check if depth value is valid
            // Note: matDepth should already be CV_32F, so read as float directly
            float depth = matDepth.at<float>(curr_y_idx, curr_x_idx);
            if(depth > 0.2) validDepthList.push_back(depth);

        }
    }
    std::sort(validDepthList.begin(), validDepthList.end());// Sort from small to large
    int size = validDepthList.size();
    float adjusted_depth = 0;     // Adjusted depth

    // Compute foreground depth
    if(size >= 8){
        // Check for jumps and return first group of continuous data
        std::vector<size_t> jump_indices;
        float rel_thres = 0.05;
        for (size_t i = 1; i < validDepthList.size(); ++i){
            float dx = validDepthList[i] - validDepthList[i-1];
            float x = validDepthList[i-1];
            float relative_change = dx/x; // All values in validDepthList are > 0, no division by zero concern

            if (std::fabs(relative_change) > rel_thres) {
                jump_indices.push_back(i); // Record jump position
                break;
            }
        }
        
        // If depth values in a patch are discontinuous, extract the smallest part
        std::vector<float> adjustDepthList;
        if(jump_indices.empty())
        {
            adjustDepthList = validDepthList;
        }else{
            size_t first_jump = jump_indices[0];
            adjustDepthList = std::vector<float>(validDepthList.begin(), validDepthList.begin() + first_jump);
        }

        int partitionSize = adjustDepthList.size();
        // Take median of depths in smallest part as depth value
        float medianValue = (partitionSize%2==0) ? 
                            (adjustDepthList[partitionSize/2-1] + adjustDepthList[partitionSize/2])/2.0 : 
                            adjustDepthList[partitionSize/2];
        if(depth_orig >= adjustDepthList.front() && depth_orig <= adjustDepthList.back()){
            // If true depth is within this interval, use true depth (true depth itself is foreground)
            adjusted_depth = depth_orig;
        }else{
            // If true depth is not in foreground interval, modify depth to foreground interval
            adjusted_depth = medianValue;
        }
    }

    pt.depth = adjusted_depth; // Assign depth to point feature

    // Compute distance score
    if(pt.depth > 0.2){
        Eigen::Vector3d pt_3d;
        pt_3d.x() = (pt.x - mCx)/mFx * pt.depth;
        pt_3d.y() = (pt.y - mCy)/mFy * pt.depth;
        pt_3d.z() = pt.depth;
        double range = pt_3d.norm();
        // Use inverse sigmoid function to compute distance score
        pt.score_depth = 1.0 / (std::exp((range - 2.5) * 1.0) + 1);
        // Update 3D point in class
        pt.x_3d = pt_3d.x();
        pt.y_3d = pt_3d.y();
        pt.z_3d = pt_3d.z();
    }else{
        pt.score_depth = 0;
    }
    
}

// Remove all edges with invalid depth and invalid edge points in valid edges
void KeyFrame::edgeCullingDepth()
{
    // Traverse all edges, remove edges with many invalid depths
    for(auto edgeIter = mvEdges.begin(); edgeIter != mvEdges.end(); )
    {
        // Get reference to current edge, avoid copying
        Edge& currentEdge = *edgeIter;

        int validPointCount = 0;
        int totalPointCount = currentEdge.mvPoints.size();
        
        // First count number of valid points
        for(const auto& point : currentEdge.mvPoints)
        {
            if(point.depth > 0.2f && point.depth < 5.0f)
            {
                validPointCount++;
            }
        }

        // Compute valid point ratio
        float validRatio = static_cast<float>(validPointCount) / totalPointCount;

        if(validRatio >= 0.3f)
        {
            // If edge is kept, remove invalid points within it
            auto newEnd = std::remove_if(currentEdge.mvPoints.begin(), 
                                        currentEdge.mvPoints.end(),
                                        [](const auto& point) {
                                            return point.depth <= 0.2f || point.depth >= 5.0f;
                                        });
            currentEdge.mvPoints.erase(newEnd, currentEdge.mvPoints.end());
            edgeIter++;  // Keep this edge, move to next
        }
        else
        {
            // If edge is invalid (more than 70% invalid points), remove entire edge
            edgeIter = mvEdges.erase(edgeIter);
        }
    }
}

void KeyFrame::edgeCullingDepthParallel()
{
    // Use char instead of atomic<bool>, use memory_order_relaxed to ensure basic thread safety
    std::vector<char> retainFlags(mvEdges.size());
    
    tbb::parallel_for(0, (int)mvEdges.size(), [&](int i) {
        Edge& currentEdge = mvEdges[i];
        int validPointCount = 0;
        const int totalPointCount = currentEdge.mvPoints.size();
        
        // Count number of valid points
        for(const auto& point : currentEdge.mvPoints) {
            if(point.depth > 0.2f && point.depth < 5.0f) {
                validPointCount++;
            }
        }

        // Compute valid ratio and decide whether to retain
        float validRatio = totalPointCount > 0 ? 
            static_cast<float>(validPointCount) / totalPointCount : 0.0f;
        retainFlags[i] = (validRatio >= 0.3f) ? 1 : 0;

        // If edge is to be retained, first filter out invalid points
        if(retainFlags[i]) {
            auto newEnd = std::remove_if(currentEdge.mvPoints.begin(), 
                                        currentEdge.mvPoints.end(),
                                        [](const auto& point) {
                                            return point.depth <= 0.2f || point.depth >= 5.0f;
                                        });
            currentEdge.mvPoints.erase(newEnd, currentEdge.mvPoints.end());
        }
    });

    // Phase 2: Serial execution of actual deletion operation
    auto newEnd = std::remove_if(mvEdges.begin(), mvEdges.end(),
        [&retainFlags, &mvEdges = this->mvEdges](const Edge& edge) {
            size_t index = &edge - &mvEdges[0];
            return retainFlags[index] == 0;
        });
    mvEdges.erase(newEnd, mvEdges.end());
}

// Ensure 3D point depth continuity for each ordered edge
void KeyFrame::edgeCullingContinuity()
{
    std::vector<bool> isEdgeValid(mvEdges.size(), true);
    // Called after CullingDepth, at this point each point in Edge is assumed to have valid depth
    tbb::parallel_for(0, (int)mvEdges.size(), [&](int cnt) {
    //for(int cnt = 0; cnt < mvEdges.size(); ++cnt)
        Edge& edge = mvEdges[cnt];
        
        //* STEP 1. Detect depth jumps in edge
        std::vector<bool> jumpFlags(edge.mvPoints.size(), false);
        float lastDepth = edge.mvPoints[0].depth;
        for (size_t i = 1; i < edge.mvPoints.size(); ++i) 
        {
            // Current point's depth
            float currentDepth = edge.mvPoints[i].depth;
            // Compare depth to determine if continuous
            jumpFlags[i] = (std::fabs(currentDepth - lastDepth) > 0.05f);
            lastDepth = currentDepth;
        }
        // Number of jumps
        int jump_num = std::count(jumpFlags.begin(), jumpFlags.end(), true);
        float jump_avg =  static_cast<float>(edge.mvPoints.size())/static_cast<float>(jump_num);
        if(jump_avg < 5){
            // If there are many jumps, this edge is at a foreground/background ambiguous position
            isEdgeValid[cnt] = false;
            // For such edges, consider directly deleting without re-splicing, so skip
            return;
        }

        //*STEP 2. For edges with few jumps, first segment according to jumpFlags
        int start_ptr = 0;
        // A segment split from an edge, first is start index, second is end index
        std::vector<std::pair<int, int>> segment;

        for(size_t i = 1; i < jumpFlags.size(); ++i)
        {
            if(jumpFlags[i])
            {
                // If position i jumps, then start -- i-1 is a continuous segment
                segment.push_back(std::make_pair(start_ptr, i-1));
                start_ptr = i;
            }
        }

        // Add last segment, at this point segment contains all edge segments (including discontinuous single-point segments)
        segment.push_back(std::make_pair(start_ptr, jumpFlags.size()-1));
        
        // * STEP 3. Use union-find to merge continuous segments
        // Represent union-find set
        DisjointSet mergeSet(segment.size());

        for(int i = 0; i < segment.size(); ++i)
        {
            float depth_end = edge.mvPoints[segment[i].second].depth; // Depth of segment end point
            // Check if subsequent segments can be spliced with segment i
            for(int j = i+2; j < segment.size(); ++j)
            {
                float depth_front = edge.mvPoints[segment[j].first].depth; // Depth of segment start point
                // If start/end point depths of two segments are continuous, they can be spliced
                if(std::fabs(depth_front - depth_end) < 0.01)
                {
                    // If they can be spliced, merge these two nodes in union-find set
                    mergeSet.to_union(i,j);
                }
            }
        }

        // * STEP 4. Select maximum continuous segment for subsequent splicing
        // Organize union-find set to get total point count for each set
        mergeSet.pruningSet();
        std::map<int, std::vector<int>> cluster; // first is root_idx, second is indices of all segments in the set
        for(int i = 0; i < segment.size(); ++i)
        {
            int root_idx = mergeSet.find(i);
            cluster[root_idx].push_back(i);
        }
        // At this point, segments in cluster.second are themselves ordered
        int max_length = -1;
        std::vector<int> max_cluster;
        for(const auto& pair : cluster)
        {
            std::vector<int> current_cluster = pair.second;
            int current_length = 0;
            
            // Compute length of current cluster
            for(int i = 0; i < current_cluster.size(); ++i)
            {
                const auto& seg = segment[current_cluster[i]];
                current_length += seg.second - seg.first + 1;
            }

            if(current_length > max_length){
                max_length = current_length;
                max_cluster = current_cluster;
            }
        }

        // * STEP 5. Splice according to max_cluster
        // Reconstruct mvPoints
        std::vector<orderedEdgePoint> new_mvPoints;
        for(int i = 0; i < max_cluster.size(); ++i)
        {
            // Start and end index of each segment
            int start_idx = segment[max_cluster[i]].first;
            int end_index = segment[max_cluster[i]].second;
            for(int j = start_idx; j <= end_index; ++j)
            {
                new_mvPoints.push_back(edge.mvPoints[j]);
            }
        }
        if(new_mvPoints.size() >= 5)
        {
            edge.mvPoints = new_mvPoints;
        }else{
            isEdgeValid[cnt] = false;
        }
    });

    // * STEP 6. Remove edges that are too short after splicing and edges with too many jumps
    int cnt_idx = 0;
    for(auto iter = mvEdges.begin(); iter != mvEdges.end(); )
    {
        if(isEdgeValid[cnt_idx] == true){
            iter ++;
        }else{
            iter = mvEdges.erase(iter);
        }
        cnt_idx += 1;
    }
}

std::vector<orderedEdgePoint> KeyFrame::getCoarseSampledPoints(int bias, int maximum_point)
{
    std::vector<orderedEdgePoint> selectedPoints;
    for(int i = 0; i < mvEdges.size(); ++i)
    {
        const Edge& edge = mvEdges[i];
        // Get sampling sequence
        for(int j = 0; j < edge.mvPoints.size(); ++j)
        {
            const orderedEdgePoint& pt = edge.mvPoints[j];
            selectedPoints.push_back(pt);
        }
    }
    // Sample according to spatial uniformity principle, prioritize points with higher scores
    // Sort points by score
    std::sort(selectedPoints.begin(),selectedPoints.end(),
              [](const orderedEdgePoint& a, const orderedEdgePoint& b){ 
                 return a.score_depth > b.score_depth; 
              });

    // Create all-black image as mask
    cv::Mat mask(mHeight, mWidth, CV_8U, cv::Scalar(0));
    // Sample and sort according to mask
    std::vector<orderedEdgePoint> sampledPoints;
    sampledPoints.reserve(std::min(maximum_point, static_cast<int>(selectedPoints.size())));

    for(const auto& pt : selectedPoints)
    {
        if(mask.at<uint8_t>(pt.y, pt.x) == 255) continue;
        // Current point can be selected
        sampledPoints.push_back(pt);
        cv::circle(mask, cv::Point(pt.x, pt.y), bias, 255, -1);
        if(sampledPoints.size() >= maximum_point) break;
    }
    // Currently all points after complete sampling
    return sampledPoints;
}

}  // namespace dyno

