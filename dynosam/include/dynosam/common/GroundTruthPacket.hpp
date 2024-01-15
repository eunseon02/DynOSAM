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

#pragma once

#include "dynosam/common/Types.hpp"
#include "dynosam/pipeline/PipelinePayload.hpp"

#include <opencv4/opencv2/opencv.hpp> //for cv::Rect
#include <gtsam/geometry/Pose3.h> //for Pose3


namespace dyno {


struct ObjectPoseGT {
    DYNO_POINTER_TYPEDEFS(ObjectPoseGT)

    FrameId frame_id_; //k
    ObjectId object_id_;
    gtsam::Pose3 L_camera_; //!object pose in camera frame
    gtsam::Pose3 L_world_; //!object pose in world frame
    cv::Rect bounding_box_; //!box of detection on image plane

    /// @brief 3D object 'dimensions' in meters. Not all datasets will contain. Used to represent a 3D bounding box.
    /// Expected order is {width, height, length}
    std::optional<gtsam::Vector3> object_dimensions_;

    /// @brief Motion in the world frame that takes us from k-1 (frame_id -1) to k (frame_id)
    std::optional<gtsam::Pose3> prev_H_current_world_;

    /// @brief Motion in the (ground truth object) frame (^WL_{k-1}) that takes us from k-1 (frame_id -1) to k (frame_id)
    std::optional<gtsam::Pose3> prev_H_current_L_;

    /// @brief Motion in the (ground truth camera) frame (^WX_{k-1}) that takes us from k-1 (frame_id -1) to k (frame_id)
    std::optional<gtsam::Pose3> prev_H_current_X_;

    /**
     * @brief Draws the object label and bounding box on the provided image
     *
     * @param img
     */
    void drawBoundingBox(cv::Mat& img) const;

    /**
     * @brief Calculates and sets prev_*_current_world_ using ground truth object in the previous frame.
     *
     * This object is epxected to be at frame k, and the previous motion should be at frame k-1. The previous_object_gt
     * should have the same object ID and an exception will be thrown if either of these checks fail.
     *
     * Both ObjectPoseGT are expected to have the L_world_ variable set correctly.
     *
     * @param previous_motion const ObjectPoseGT&
     * @param prev_X_world const gtsam::Pose3& Camera pose at the previous frame (k-1).
     * @param curr_X_world const gtsam::Pose3& Camera pose at the current frame (k).
     */
    void setMotions(const ObjectPoseGT& previous_object_gt, const gtsam::Pose3& prev_X_world, const gtsam::Pose3& curr_X_world);

    operator std::string() const;

    bool operator==(const ObjectPoseGT& other) const;
    friend std::ostream &operator<<(std::ostream &os, const ObjectPoseGT& object_pose);
};

class GroundTruthInputPacket : public PipelinePayload {
public:
    DYNO_POINTER_TYPEDEFS(GroundTruthInputPacket)

    //must have a default constructor for dataset loading and IO
    GroundTruthInputPacket() {}

    GroundTruthInputPacket(Timestamp timestamp, FrameId id, const gtsam::Pose3 X, const std::vector<ObjectPoseGT>& poses)
        : PipelinePayload(timestamp), frame_id_(id), X_world_(X), object_poses_(poses) {}

    FrameId frame_id_;
    gtsam::Pose3 X_world_; //camera pose in world frame
    std::vector<ObjectPoseGT> object_poses_;

    bool getObject(ObjectId object_id, ObjectPoseGT& object_pose_gt) const;

    ObjectIds getObjectIds() const;

    /**
     * @brief Query an ObjectPoseGT in this packet and anOTHER packet using a object label
     *
     * If the query object is in both this and the other packet, true is returned and obj and other_obj are set
     *
     * We pass in a pointer to a pointer so we can modify the value of pointer itself
     *
     * @param label
     * @param other
     * @param obj
     * @param other_obj
     * @return true
     * @return false
     */
    bool findAssociatedObject(ObjectId label, GroundTruthInputPacket& other, ObjectPoseGT** obj, ObjectPoseGT** other_obj);
    bool findAssociatedObject(ObjectId label, const GroundTruthInputPacket& other, ObjectPoseGT** obj, const ObjectPoseGT** other_obj);

    /**
     * @brief Query an ObjectPoseGT in this packet and anOTHER packet using a object label
     *
     * If the query object is in both this and the other packet, true is returned and the index location obj and other_obj are set.
     * The index is the position in the object_poses_ vector (respectively) where the ObjectPoseGT can be found
     *
     * @param label
     * @param other
     * @param obj_idx
     * @param other_obj_idx
     * @return true
     * @return false
     */
    bool findAssociatedObject(ObjectId label, const GroundTruthInputPacket& other, size_t& obj_idx, size_t& other_obj_idx) const;

    /**
     * @brief Calcualtes and sets the object motion ground truth variables of this GroundTruthInputPacket using the previous object motions.
     * This packet is considered to be time k and the previous object packet is k-1. This should only be used in some derived
     * DataProvider/Loader when constructing the GroundTruthPacket.
     *
     * If the previous obejct packet is not at the right frame (k-1), false will be returned.
     * Each object in the current packet will be queried in the previous packet, and, if exists, the motion will be calcuted at set.
     *
     * The vector of ObjectId's indicate which objects motions were calculated
     *
     * @param previous_object_packet
     * @param motions_set
     * @return size_t number of motions set
     */
    size_t calculateAndSetMotions(const GroundTruthInputPacket& previous_object_packet, ObjectIds& motions_set);
    size_t calculateAndSetMotions(const GroundTruthInputPacket& previous_object_packet);

    operator std::string() const;

    friend std::ostream &operator<<(std::ostream &os, const GroundTruthInputPacket& gt_packet);

private:


};




} //dyno
