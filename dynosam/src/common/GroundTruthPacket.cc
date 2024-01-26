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

#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/common/Exceptions.hpp"

#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/visualizer/ColourMap.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>


namespace dyno {

//copied from KittiSemanticToMotion.cc but not linked so might mismatch with semantic image?
bool isMoving(const gtsam::Pose3& prev_L_world, const gtsam::Pose3& curr_L_world, double tol_m = 0.4) {
    gtsam::Vector3 t_diff = curr_L_world.translation() - prev_L_world.translation();
    // L2 norm - ie. magnitude
    double t_error = t_diff.norm();
    return t_error > tol_m;
}

void ObjectPoseGT::setMotions(const ObjectPoseGT& previous_object_gt, const gtsam::Pose3& prev_X_world, const gtsam::Pose3& curr_X_world) {
    checkAndThrow(previous_object_gt.frame_id_ == frame_id_ - 1, "Previous object gt frame is not at k-1. Current frame = " + std::to_string(frame_id_) + " and previous frame =" + std::to_string(previous_object_gt.frame_id_));
    checkAndThrow(previous_object_gt.object_id_ == object_id_, "Previous object gt does not have the same object id");


    // for t-1
    const gtsam::Pose3 L_camera_prev = previous_object_gt.L_camera_;
    const gtsam::Pose3 L_world_prev = previous_object_gt.L_world_;
    CHECK(L_world_prev.equals(prev_X_world * L_camera_prev));
    // for t
    const gtsam::Pose3 L_camera_curr = L_camera_;
    const gtsam::Pose3 L_world_curr = L_world_;
    CHECK(L_world_curr.equals(curr_X_world * L_camera_curr));

    // H between t-1 and t in the object frame ground truth
    gtsam::Pose3 H_L_gt = L_world_prev.inverse() * L_world_curr;
    prev_H_current_L_ = H_L_gt;

    // H between t-1 and t in the camera frame at t-1
    gtsam::Pose3 H_X_gt = L_camera_prev * H_L_gt * L_camera_prev.inverse();
    prev_H_current_X_ = H_X_gt;

    // H between t-1 and t in the world frame
    gtsam::Pose3 H_W_gt = L_world_prev * H_L_gt * L_world_prev.inverse();
    prev_H_current_world_ = H_W_gt;

    MotionInfo motion_info;
    motion_info.is_moving_ = isMoving(L_world_prev, L_world_curr);
    motion_info.has_stopped_ = false; //default

    if(previous_object_gt.motion_info_) {
        //moving in the previous frame but not in this frame
        if(previous_object_gt.motion_info_->is_moving_ && !motion_info.is_moving_) {
            motion_info.has_stopped_ = true;
        }
    }

    motion_info_ = motion_info;

}

void ObjectPoseGT::drawBoundingBox(cv::Mat& img) const {
    const cv::Scalar colour = ColourMap::getObjectColour(object_id_, true);
    const std::string label = "Object: " + std::to_string(object_id_);
    utils::drawLabeledBoundingBox(img, label, colour, bounding_box_);
}


ObjectPoseGT::operator std::string() const {
    std::stringstream ss;
    ss << "FrameId: " << frame_id_
       << " ObjectId: " << object_id_;

    if(motion_info_) {
        ss << " Is moving " << motion_info_->is_moving_
           << " Has stopped " << motion_info_->has_stopped_;
    }
    return ss.str();
}


bool ObjectPoseGT::operator==(const ObjectPoseGT& other) const {
    return frame_id_ == other.frame_id_ &&
        object_id_ == other.object_id_ &&
        L_camera_.equals(other.L_camera_) &&
        L_world_.equals(other.L_world_);
        // bounding_box_ == other.bounding_box_;
        //TODO: not sure how to compare the std::optional as the underlying objects dont have operator==
}


std::ostream& operator<<(std::ostream &os, const ObjectPoseGT& object_pose) {
    os << (std::string)object_pose;
    return os;
}

bool GroundTruthInputPacket::getObject(ObjectId object_id, ObjectPoseGT& object_pose_gt) const {
    auto it_this = std::find_if(object_poses_.begin(), object_poses_.end(),
                [=](const ObjectPoseGT& gt_object) { return gt_object.object_id_ == object_id; });
    if(it_this == object_poses_.end()) {
        return false;
    }

    object_pose_gt = *it_this;
    return true;

}

ObjectIds GroundTruthInputPacket::getObjectIds() const {
    ObjectIds object_ids;
    for(const ObjectPoseGT& objects : object_poses_) {
        object_ids.push_back(objects.object_id_);
    }
    return object_ids;
}


bool GroundTruthInputPacket::findAssociatedObject(ObjectId label, GroundTruthInputPacket& other, ObjectPoseGT** obj, ObjectPoseGT** other_obj) {
    size_t obj_idx, other_obj_idx;

    if(findAssociatedObject(label, other, obj_idx, other_obj_idx)) {
        *obj = &object_poses_.at(obj_idx);
        *other_obj = &other.object_poses_.at(other_obj_idx);
        return true;
    }


    return false;
}

bool GroundTruthInputPacket::findAssociatedObject(ObjectId label, const GroundTruthInputPacket& other, const ObjectPoseGT** obj, const ObjectPoseGT** other_obj) const {
    size_t obj_idx, other_obj_idx;

    if(findAssociatedObject(label, other, obj_idx, other_obj_idx)) {
        *obj = &object_poses_.at(obj_idx);
        *other_obj = &other.object_poses_.at(other_obj_idx);
        return true;
    }

    return false;
}

bool GroundTruthInputPacket::findAssociatedObject(ObjectId label, const GroundTruthInputPacket& other, ObjectPoseGT** obj, const ObjectPoseGT** other_obj) {
    size_t obj_idx, other_obj_idx;

    if(findAssociatedObject(label, other, obj_idx, other_obj_idx)) {
        *obj = &object_poses_.at(obj_idx);
        *other_obj = &other.object_poses_.at(other_obj_idx);
        return true;
    }

    return false;
}

bool GroundTruthInputPacket::findAssociatedObject(ObjectId label, const GroundTruthInputPacket& other, size_t& obj_idx, size_t& other_obj_idx) const {
    auto it_this = std::find_if(object_poses_.begin(), object_poses_.end(),
                [=](const ObjectPoseGT& gt_object) { return gt_object.object_id_ == label; });
    if(it_this == object_poses_.end()) {
        return false;
    }

    auto it_other = std::find_if(other.object_poses_.cbegin(), other.object_poses_.cend(),
                [=](const ObjectPoseGT& gt_object) { return gt_object.object_id_ == label; });
    if(it_other == other.object_poses_.cend()) {
        return false;
    }

    obj_idx = std::distance(object_poses_.begin(), it_this);
    other_obj_idx = std::distance(other.object_poses_.cbegin(), it_other);
    return true;
}



size_t GroundTruthInputPacket::calculateAndSetMotions(const GroundTruthInputPacket& previous_object_packet, ObjectIds& motions_set) {
    motions_set.clear();
    if(previous_object_packet.frame_id_ != frame_id_ -1) { return false; }

    //iterate over all objects and try to find assications in previous packet
    for(ObjectPoseGT& object_pose_gt : object_poses_) {
        const ObjectId label = object_pose_gt.object_id_;

        ObjectPoseGT* current_object_pose_gt;
        const ObjectPoseGT* previous_object_pose_gt;
        if(findAssociatedObject(label, previous_object_packet, &current_object_pose_gt, &previous_object_pose_gt)) {
            CHECK_NOTNULL(current_object_pose_gt);
            CHECK_NOTNULL(previous_object_pose_gt);
            CHECK_EQ(*current_object_pose_gt, object_pose_gt);

            //set motions for current motion
            current_object_pose_gt->setMotions(*previous_object_pose_gt, previous_object_packet.X_world_, X_world_);
            motions_set.push_back(label);
        }

    }

    return motions_set.size();
}

size_t GroundTruthInputPacket::calculateAndSetMotions(const GroundTruthInputPacket& previous_object_packet) {
    ObjectIds motion_set;
    return calculateAndSetMotions(previous_object_packet, motion_set);
}

GroundTruthInputPacket::operator std::string() const {
    std::stringstream ss;
    ss << "FrameId: " << frame_id_
        << " objects: " << container_to_string(getObjectIds());
    return ss.str();
}


std::ostream& operator<<(std::ostream &os, const GroundTruthInputPacket& gt_packet) {
    os << (std::string)gt_packet;
    return os;
}

} // dyno
