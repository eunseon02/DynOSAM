#include "dynosam_ros/displays/dynamic_slam_displays/DynamicSlamDisplaysCommon.hpp"
#include <glog/logging.h>


namespace dyno {


ObjectOdometry DynamicSlamDisplayCommon::constructObjectOdometry(
            const MotionEstimateMap& motions, 
            const ObjectPoseMap& poses, 
            const std::string& frame_id,
            FrameId frame_id_k,
            Timestamp timestamp)
{

    //need to get poses for k-1
    //TODO: no way to ensure that the motions are for frame k
    //this is a weird data-structure to use and motions are per frame and 
    //ObjectPoseMap is over all k to K
    const FrameId frame_id_k_1 = frame_id_k - 1u;
    
    ObjectOdometry object_odom;
    for(const auto& [object_id, object_motion] : motions) {
        const gtsam::Pose3& motion = object_motion;

        if(!poses.exists(object_id, frame_id_k_1)) {
            LOG(WARNIG) << "Cannot construct ObjectOdometry for object " 
                << object_id << ", at frame " << frame_id_k_1
                << " Missing entry in ObjectPoseMap";
            continue;
        }

        const gtsam::Pose3& pose = poses.at(object_id, frame_id_k_1)
    }

}


}