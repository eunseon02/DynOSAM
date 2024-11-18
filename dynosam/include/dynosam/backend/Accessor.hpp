/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/common/Types.hpp"

namespace dyno {

/**
 * @brief Accessor defines the interface between the structured output of a
 * BackendModule and any formulation used to construct the Dynamic SLAM problem.
 *
 * Each derived Accessor is associated with a derived Formulation and knows how
 * to access the variables in that problem and convert them into the form
 * expected by the backend. This is equivalent to constructing \mathbf{O}_k = [
 * ^wX_k,   \othmotion{\worldf}{k-1}{\mathcal{H}}{k}, ^{\worldf}\mathcal{L}_{k}
 * , {^\worldf\mathcal{M}_k}].
 *
 * @tparam MAP
 */
template <class MAP>
class Accessor {
 public:
  using Map = MAP;
  using This = Accessor<Map>;

  DYNO_POINTER_TYPEDEFS(This)

  Accessor(const gtsam::Values* theta, typename Map::Ptr map);
  virtual ~Accessor() {}

  /**
   * @brief Get the sensor pose (^wX_k) from theta at the requested time-step.
   *
   * @param frame_id FrameId
   * @return StateQuery<gtsam::Pose3>
   */
  virtual StateQuery<gtsam::Pose3> getSensorPose(FrameId frame_id) const = 0;

  /**
   * @brief Get an absolute object motion (_{k-1}^wH_k) from theta the requested
   * time-step (k) and object (j).
   *
   * @param frame_id FrameId
   * @param object_id ObjectId
   * @return StateQuery<gtsam::Pose3>
   */
  virtual StateQuery<gtsam::Pose3> getObjectMotion(
      FrameId frame_id, ObjectId object_id) const = 0;

  /**
   * @brief Get the pose (^wL_k) of an object (j) at time-step (k).
   *
   * @param frame_id FrameId
   * @param object_id ObjectId
   * @return StateQuery<gtsam::Pose3>
   */
  virtual StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id,
                                                 ObjectId object_id) const = 0;

  /**
   * @brief Get a dynamic landmark (^wm_k) with tracklet id (i) at time-step
   * (k).
   *
   * @param frame_id FrameId
   * @param tracklet_id TrackletId
   * @return StateQuery<gtsam::Point3>
   */
  virtual StateQuery<gtsam::Point3> getDynamicLandmark(
      FrameId frame_id, TrackletId tracklet_id) const = 0;

  /**
   * @brief Get a static landmark (^wm) with tracklet id (i).
   *
   * @param tracklet_id TrackletId
   * @return StateQuery<gtsam::Point3>
   */
  virtual StateQuery<gtsam::Point3> getStaticLandmark(
      TrackletId tracklet_id) const;

  /**
   * @brief Collects all object poses at some time-step (k). This is equivalent
   * to \mathcal{L}_k. Is a (non-pure) virtual function so may be overwritten -
   * default implementation uses the pure-virtual getObjectPose function to
   * collect all object poses.
   *
   * @param frame_id FrameId frame_id) const;
   * @return EstimateMap<ObjectId, gtsam::Pose3>
   */
  virtual EstimateMap<ObjectId, gtsam::Pose3> getObjectPoses(
      FrameId frame_id) const;

  /**
   * @brief Collects all object motions from some time-step (k-1) to k. This is
   * equivalent to \mathcal{H}_k. Uses the pure-virtual getObjectMotion function
   * to collect all object motions.
   *
   *
   * @param frame_id FrameId
   * @return MotionEstimateMap
   */
  MotionEstimateMap getObjectMotions(FrameId frame_id) const;

  /**
   * @brief Virtual function that is called to indcate the associated
   * formulation has been updated. This can be used to trigger an internal
   * update of variables/cache if necessary.
   *
   * @param const BackendMetaData&
   */
  virtual void postUpdateCallback(const BackendMetaData&){};

  /**
   * @brief Gets an estimate map of objects with their motion and and PREVIOUS
   * pose. This will find motions and poses (using the internal API) and return
   * a motion from k-1 to k and the associated pose at k-1.
   *
   * @param frame_id FrameId
   * @return EstimateMap<ObjectId, std::pair<Motion3, gtsam::Pose3>>
   */
  EstimateMap<ObjectId, std::pair<Motion3, gtsam::Pose3>> getObjectPrevPosePair(
      FrameId frame_id) const;

  /**
   * @brief Gets an estimate map of objects with their motion and and CURRENT
   * pose. This will find motions and poses (using the internal API) and return
   * a motion from k-1 to k and the associated pose at k.
   *
   * @param frame_id
   * @return EstimateMap<ObjectId, std::pair<Motion3, gtsam::Pose3>>
   */
  EstimateMap<ObjectId, std::pair<Motion3, gtsam::Pose3>> getObjectPosePair(
      FrameId frame_id) const;

  /**
   * @brief Collects all object poses for frame 0 to k.
   * (Non-pure )Virtual function that may be overwritten - default
   * implementation uses the pure-virtual getObjectPoses to collect all poses.
   *
   * @return ObjectPoseMap
   */
  virtual ObjectPoseMap getObjectPoses() const;

  /**
   * @brief Get all dynamic landmarks for all objects (\mathcal{J}_k) at
   * time-step k.
   *
   * @param frame_id FrameId
   * @return StatusLandmarkEstimates
   */
  StatusLandmarkEstimates getDynamicLandmarkEstimates(FrameId frame_id) const;

  /**
   * @brief Get all dynamic landmarks for object j at time-step k.
   *
   * @param frame_id FrameId
   * @param object_id ObjectId
   * @return StatusLandmarkEstimates
   */
  virtual StatusLandmarkEstimates getDynamicLandmarkEstimates(
      FrameId frame_id, ObjectId object_id) const;

  /**
   * @brief Get all static landmarks at time-step k.
   *
   * @param frame_id FrameId
   * @return StatusLandmarkEstimates
   */
  StatusLandmarkEstimates getStaticLandmarkEstimates(FrameId frame_id) const;

  /**
   * @brief Get all static landmarks from time-step 0 to k.
   *
   * @return StatusLandmarkEstimates
   */
  StatusLandmarkEstimates getFullStaticMap() const;

  /**
   * @brief Get all landmarks (static and dynamic) at time-step k.
   *
   * @param frame_id FrameId
   * @return StatusLandmarkEstimates
   */
  StatusLandmarkEstimates getLandmarkEstimates(FrameId frame_id) const;

  /**
   * @brief Check if there exists an estimate for object motion at time-step (k)
   * for object id (j). If result is true and a motion is provided, return true
   * and set the motion.
   *
   * @param frame_id FrameId
   * @param object_id ObjectId
   * @param motion Motion3*. Default value is nullptr. If provided (ie non-null)
   * and motion exists, value is set.
   * @return true
   * @return false
   */
  bool hasObjectMotionEstimate(FrameId frame_id, ObjectId object_id,
                               Motion3* motion = nullptr) const;

  /**
   * @brief Check if there exists an estimate for object motion at time-step (k)
   * for object id (j). If result is true, motion reference value is set to the
   * motion.
   *
   * @param frame_id FrameId
   * @param object_id ObjectId
   * @param motion Motion3&
   * @return true
   * @return false
   */
  bool hasObjectMotionEstimate(FrameId frame_id, ObjectId object_id,
                               Motion3& motion) const;

  /**
   * @brief Check if there exists an estimate for object pose at time-step (k)
   * for object id (j). If result is true and a pose is provided, return true
   * and set the pose.
   *
   * @param frame_id FrameId
   * @param object_id ObjectId
   * @param pose gtsam::Pose3* Default value is nullptr. If provided (ie
   * non-null) and pose exists, value is set.
   * @return true
   * @return false
   */
  bool hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id,
                             gtsam::Pose3* pose = nullptr) const;

  /**
   * @brief Check if there exists an estimate for object pose at time-step (k)
   * for object id (j). If result is true, pose reference value is set to the
   * pose.
   *
   * @param frame_id FrameId
   * @param object_id ObjectId
   * @param pose gtsam::Pose3&
   * @return true
   * @return false
   */
  bool hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id,
                             gtsam::Pose3& pose) const;

  /**
   * @brief Computes a the centroid of each object at this frame using the
   * estimated dynamic points.
   *
   * Internally, uses the overloaded std::tuple<gtsam::Point3, bool>
   * computeObjectCentroid function and only includes centroids which are valid
   * (ie returned with computeObjectCentroid()->second == true)
   *
   * @param frame_id FrameId
   * @return gtsam::FastMap<ObjectId, gtsam::Point3>
   */
  gtsam::FastMap<ObjectId, gtsam::Point3> computeObjectCentroids(
      FrameId frame_id) const;

  /**
   * @brief Computes the centroid of the requested object using the estimated
   * dynamic points.
   *
   * Returns false (as part of the pair) if object cannot be found or has zero
   * points.
   *
   * @param frame_id FrameId
   * @param object_id ObjectId
   * @return std::tuple<gtsam::Point3, bool>
   */
  std::tuple<gtsam::Point3, bool> computeObjectCentroid(
      FrameId frame_id, ObjectId object_id) const;

  /**
   * @brief Check if the key exists in the current theta.
   *
   * @param key gtsam::Key
   * @return true
   * @return false
   */
  bool exists(gtsam::Key key) const;

  /**
   * @brief Access a key in the current theta.
   *
   * @tparam ValueType
   * @param key gtsam::Key
   * @return StateQuery<ValueType>
   */
  template <typename ValueType>
  StateQuery<ValueType> query(gtsam::Key key) const;

 protected:
  auto map() const { return map_; }

 private:
  /**
   * @brief Compute all pairs of object motions and poses at the requested
   * time-steps. As usual, motion_frame_id is treated as k, and therefore
   * motions from motion_frame_id-1 to motion_frame_id will be retrieved.
   *
   * @param motion_frame_id FrameId
   * @param pose_frame_id FrameId
   * @return EstimateMap<ObjectId, std::pair<Motion3, gtsam::Pose3>>
   */
  EstimateMap<ObjectId, std::pair<Motion3, gtsam::Pose3>>
  getRequestedObjectPosePair(FrameId motion_frame_id,
                             FrameId pose_frame_id) const;

 private:
  const gtsam::Values* theta_;  //! Pointer to the set of current values stored
                                //! in the associated formulation
  typename Map::Ptr map_;       //! Pointer to internal map structure;
};

}  // namespace dyno

#include "dynosam/backend/Accessor-impl.hpp"
