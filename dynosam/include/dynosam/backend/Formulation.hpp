/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/Accessor.hpp"

#include "dynosam/common/Map.hpp"

namespace dyno {

template<typename MAP>
struct MapTraits {
    using Map = MAP;
    using FrameNode = typename Map::FrameNodeM;
    using ObjectNode = typename Map::ObjectNodeM;
    using LandmarkNode = typename Map::LandmarkNodeM;

    using FrameNodePtr = typename FrameNode::Ptr;
    using ObjectNodePtr = typename ObjectNode::Ptr;
    using LandmarkNodePtr = typename LandmarkNode::Ptr;
};

struct UpdateObservationResult {
    gtsam::FastMap<ObjectId, std::set<FrameId>> objects_affected_per_frame; //per frame
    DebugInfo::Optional debug_info{};

    //TODO: debug info
    inline UpdateObservationResult& operator+=(const UpdateObservationResult& oth) {
        for(const auto& [key, value] : oth.objects_affected_per_frame) {
            objects_affected_per_frame[key].insert(value.begin(), value.end());
        }
        return *this;
    }

    void updateAffectedObject(FrameId frame_id, ObjectId object_id) {
        if(!objects_affected_per_frame.exists(object_id)) {
            objects_affected_per_frame.insert2(object_id,  std::set<FrameId>{});
        }
        objects_affected_per_frame[object_id].insert(frame_id);
    }
};


template<typename MAP>
struct PointUpdateContext {
    using MapTraitsType = MapTraits<MAP>;

    typename MapTraitsType::LandmarkNodePtr lmk_node;
    typename MapTraitsType::FrameNodePtr frame_node_k_1;
    typename MapTraitsType::FrameNodePtr frame_node_k;

    gtsam::Pose3 X_k_measured; //! Camera pose from measurement (or initial)
    gtsam::Pose3 X_k_1_measured; //! Camera pose from measurement (or initial)

    //! If true then this frame is the first frame where a motion is available (i.e we have a pair of valid frames)
    //! e.g k-1 is the FIRST frame for this object and now, since we are at k, we can create a motion from k-1 to k
    bool is_starting_motion_frame {false};

    inline ObjectId getObjectId() const { return lmk_node->template getObjectId(); }
    inline TrackletId getTrackletId() const { return lmk_node->template tracklet_id; }

};

template<typename MAP>
struct ObjectUpdateContext {
    using MapTraitsType = MapTraits<MAP>;

    /// @brief Frame that is part of the update context. Shared pointer to a frame node as defined by the Map type
    typename MapTraitsType::FrameNodePtr frame_node_k;
    /// @brief Object that is part of the update context. Shared pointer to a object node as defined by the Map type
    typename MapTraitsType::ObjectNodePtr object_node;

    //! Indicates that we have a valid motion pair from k-1 to k (this frame)
    //! and therefore k is at least the second frame for which this object has been consequatively tracked
    //! When this is false, it means that the frame k-1 (getFrameId() - 1u) did not track/observe this object
    //! and therfore no motion can be created between them. This happens on the first observation of this object.
    bool has_motion_pair{false};

    inline FrameId getFrameId() const { return frame_node_k->template getId(); }
    inline ObjectId getObjectId() const { return object_node->template getId(); }
};


struct UpdateObservationParams {
    //! If true, vision related updated will backtrack to the start of a new tracklet and all the measurements to the graph
    //! should make false in batch case where we want to be explicit about which frames are added!
    bool do_backtrack = false;
    bool enable_debug_info = true;
};

//forward declare
class BackendParams;

struct FormulationParams {
    size_t min_dynamic_observations = 3u;
    size_t min_static_observations = 2u;
    std::string suffix = "";
};

template<typename MAP>
class Formulation {
public:
    using Map = MAP;
    using This = Formulation<Map>;

    using PointUpdateContextType = PointUpdateContext<Map>;
    using ObjectUpdateContextType = ObjectUpdateContext<Map>;
    using AccessorType = Accessor<MAP>;
    using AccessorTypePointer = typename Accessor<MAP>::Ptr;

    DYNO_POINTER_TYPEDEFS(This)

    Formulation(const FormulationParams& params, typename Map::Ptr map, const NoiseModels& noise_models);
    virtual ~Formulation() = default;

    virtual void dynamicPointUpdateCallback(
        const PointUpdateContextType& context,
        UpdateObservationResult& result,
        gtsam::Values& new_values,
        gtsam::NonlinearFactorGraph& new_factors) = 0;

    virtual void objectUpdateContext(
        const ObjectUpdateContextType& context,
        UpdateObservationResult& result,
        gtsam::Values& new_values,
        gtsam::NonlinearFactorGraph& new_factors) = 0;

    virtual bool isDynamicTrackletInMap(const LandmarkNode3d2d::Ptr& lmk_node) const = 0;

    void setTheta(const gtsam::Values& linearization);
    void updateTheta(const gtsam::Values& linearization);

    //make logger with full logger name including possible suffix from config
    BackendLogger::UniquePtr makeFullyQualifiedLogger() const;

    void setInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values);
    void setInitialPosePrior(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::NonlinearFactorGraph& new_factors);

    void addOdometry(FrameId from_frame, FrameId to_frame, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);
    void addOdometry(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors);

    UpdateObservationResult updateStaticObservations(
            FrameId from_frame,
            FrameId to_frame,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        UpdateObservationResult updateDynamicObservations(
            FrameId from_frame,
            FrameId to_frame,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);


        UpdateObservationResult updateStaticObservations(
            FrameId frame_id_k,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        UpdateObservationResult updateDynamicObservations(
            FrameId frame_id_k,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        //log everything all frames!!
        void logBackendFromMap(const BackendMetaData& backend_info);
        typename AccessorType::Ptr accessorFromTheta() const;

        typename Map::Ptr map() const { return map_; }
        const gtsam::Values& getTheta() const { return theta_; }
        const gtsam::NonlinearFactorGraph& getGraph() const { return factors_; }
        const std::string getFullyQualifiedName() const { return fully_qualified_name_.value_or(setFullyQualifiedName()); }


protected:
    gtsam::Pose3 getInitialOrLinearizedSensorPose(FrameId frame_id) const;

    virtual std::string loggerPrefix() const = 0;
    virtual typename AccessorType::Ptr createAccessor(const gtsam::Values* values) const = 0;

private:
    std::string setFullyQualifiedName() const; //but isnt actually const ;)


protected:
    const FormulationParams params_;
    typename Map::Ptr map_;
    const NoiseModels noise_models_;


    gtsam::FastMap<gtsam::Key, bool> is_other_values_in_map; //! the set of (static related) values managed by this updater. Allows checking if values have already been added over successifve function calls

private:
    //! Current linearisation that will be associated with the current graph
    gtsam::Values theta_;
    gtsam::NonlinearFactorGraph factors_;
    mutable typename AccessorType::Ptr accessor_theta_;
    //! Full name of the formulation and accounts for the additional configuration from the FormulationParams
    mutable std::optional<std::string> fully_qualified_name_{std::nullopt};

};


} // dyno

#include "dynosam/backend/Formulation-impl.hpp"
