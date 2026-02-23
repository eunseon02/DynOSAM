/**
* This file is part of ORB-SLAM2 / OA-SLAM (ported into dynosam edge_map).
*/

#include "dynosam/backend/edge_map/Map.hpp"
#include "dynosam/backend/edge_map/Graph.hpp"
#include "dynosam/backend/edge_map/KeyFrame.hpp"
#include "dynosam/frontend/vision/Object.hpp"
#include<mutex>

namespace dyno
{

EdgeMap::EdgeMap():mnMaxKFid(0),mnBigChangeIdx(0)
{
    graph_3d = new dyno::Graph();//nullptr;
}

void EdgeMap::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->KF_ID>mnMaxKFid)
        mnMaxKFid=pKF->KF_ID;
}
void EdgeMap::AddEdge(Edge *pEdge)
{
    unique_lock<mutex> lock(mMutexMap);
    mspEdges.insert(pEdge);
}
void EdgeMap::EraseEdge(Edge *pEdge)
{
    unique_lock<mutex> lock(mMutexMap);
    mspEdges.erase(pEdge);
}
// void Map::AddMapPoint(MapPoint *pMP)
// {
//     unique_lock<mutex> lock(mMutexMap);
//     mspMapPoints.insert(pMP);
// }

// void Map::EraseMapPoint(MapPoint *pMP)
// {
//     unique_lock<mutex> lock(mMutexMap);
//     mspMapPoints.erase(pMP);

//     // TODO: This only erase the pointer.
//     // Delete the MapPoint
// }

void EdgeMap::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

// void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
// {
//     unique_lock<mutex> lock(mMutexMap);
//     mvpReferenceMapPoints = vpMPs;
// }

void EdgeMap::SetReferenceEdges(const vector<Edge*> &vpEdges)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceEdges = vpEdges;
}

void EdgeMap::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int EdgeMap::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> EdgeMap::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

// vector<MapPoint*> Map::GetAllMapPoints()
// {
//     unique_lock<mutex> lock(mMutexMap);
//     return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
// }

// long unsigned int Map::MapPointsInMap()
// {
//     unique_lock<mutex> lock(mMutexMap);
//     return mspMapPoints.size();
// }

long unsigned int EdgeMap::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

// vector<MapPoint*> Map::GetReferenceMapPoints()
// {
//     unique_lock<mutex> lock(mMutexMap);
//     return mvpReferenceMapPoints;
// }

long unsigned int EdgeMap::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void EdgeMap::clear()
{
    // for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        // delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    // mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    // mvpReferenceMapPoints.clear();  // Commented out as mvpReferenceMapPoints is not defined
    mvpKeyFrameOrigins.clear();

    for(set<dyno::Object*>::iterator sit=mspObjects.begin(), send=mspObjects.end(); sit!=send; sit++)
        delete *sit;
    mspObjects.clear();
}

void EdgeMap::AddObject(dyno::Object *obj){
    unique_lock<mutex> lock(mMutexMap);
    mspObjects.insert(obj);
}

std::vector<dyno::Object*> EdgeMap::GetAllObjects()
{
    unique_lock<mutex> lock(mMutexMap);
    return std::vector<dyno::Object*>(mspObjects.begin(),mspObjects.end());
}

}  // namespace dyno
