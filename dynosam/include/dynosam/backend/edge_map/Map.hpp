/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MAP_H
#define MAP_H

#include "dynosam/backend/edge_map/KeyFrame.hpp"
#include "dynosam_common/Ellipsoid.hpp"
#include <set>
#include <unordered_map>

#include <mutex>

// Forward declare dyno::Object so we can store pointers without full definition here.
namespace dyno {
class Object;
}

// Forward declare classes
namespace dyno {
class Graph;
class Object;
class KeyFrame;
class MapObject;

class EdgeMap
{
public:
    EdgeMap();

    void AddKeyFrame(KeyFrame* pKF);
    void AddEdge(Edge* pEdge);
    void EraseEdge(Edge* pEdge);
    // void AddMapPoint(MapPoint* pMP);
    // void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);
    // void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void SetReferenceEdges(const std::vector<Edge*> &vpEdges);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<KeyFrame*> GetAllKeyFrames();
    // std::vector<MapPoint*> GetAllMapPoints();
    std::vector<Edge*> GetAllEdges();
    std::vector<dyno::Object*> GetAllObjects();
    // std::vector<MapPoint*> GetReferenceMapPoints();

    // long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    // const std::unordered_map<unsigned int, Eigen::Matrix<double, 3, Eigen::Dynamic>>& GetAllMapObjectsPoints() {
    //     return ellipsoids_points_;

    void AddObject(dyno::Object *obj);


    //MapObject* GetObjWithTrId(int tr_id);

    size_t GetNumberMapObjects() const {
        return mspObjects.size();
    }
    // size_t GetNumberPoints() const {
    //     return mspMapPoints.size();
    // }
    //ADDED TOBE DELETED
    std::set<KeyFrame*> getKeyFrames() const {
        return mspKeyFrames;
    }

    dyno::Graph *graph_3d;


protected:
    // std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;
    std::set<Edge*> mspEdges;
    std::set<dyno::Object*> mspObjects;

    // std::vector<MapPoint*> mvpReferenceMapPoints;
    std::vector<Edge*> mvpReferenceEdges;
    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;

    // std::unordered_map<unsigned int, Eigen::Matrix<double, 3, Eigen::Dynamic>> ellipsoids_points_;
};

}  // namespace dyno

#endif  // MAP_H
