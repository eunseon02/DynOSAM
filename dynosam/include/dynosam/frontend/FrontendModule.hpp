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
#include "dynosam/common/ModuleBase.hpp"
#include "dynosam/common/Exceptions.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/FrontendOutputPacket.hpp"
#include "dynosam/common/ModulePacketVariants.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"
// #include "dynosam/common"

#include <type_traits>

namespace dyno {


struct InvalidImageContainerException : public DynosamException {
    InvalidImageContainerException(const ImageContainer& container, const std::string& what)
    : DynosamException("Image container with config: " + container.toString() + "\n was invalid - " + what) {}
};


/**
 * @brief Base class to actually do processing. Data passed to this module from the frontend
 *
 */

class FrontendModule : public ModuleBase<FrontendInputPacketBase, FrontendOutputPacketBase>  {

public:
    DYNO_POINTER_TYPEDEFS(FrontendModule)

    using Base = ModuleBase<FrontendInputPacketBase, FrontendOutputPacketBase>;
    using Base::SpinReturn;

    FrontendModule(const FrontendParams& params, ImageDisplayQueue* display_queue = nullptr);
    virtual ~FrontendModule() = default;


protected:
    // /**
    //  * @brief Estimates the pose of frame k given a reference frame at k-1 using 2d2d point correspondances.
    //  *
    //  * The frontend params will determine the type of pnp solver to use (1-point or 2-point ransac)
    //  *
    //  * @param frame_k_1
    //  * @param frame_k
    //  * @param R_curr_ref
    //  */
    // void outlierRejectionMono(Frame::Ptr frame_k_1,
    //                           Frame::Ptr frame_k,
    //                           std::optional<gtsam::Rot3> R_curr_ref = {}) const;


    // /**
    //  * @brief Estimates the pose of the frame k given a reference frame at k-1 using 3d3d point correspondances
    //  *
    //  * @param frame_k_1
    //  * @param frame_k
    //  * @param translation_info_matrix
    //  * @param R_curr_ref
    //  */
    // void outlierRejectionStereo(Frame::Ptr frame_k_1,
    //                             Frame::Ptr frame_k,
    //                             gtsam::Matrix3* translation_info_matrix,
    //                             std::optional<gtsam::Rot3> R_curr_ref = {}) const;

    // /**
    //  * @brief Estimates the pose of the frame k given a reference frame at k-1 using 3d2d point correspondances
    //  *
    //  * @param frame_k_1
    //  * @param frame_k
    //  */
    // void outlierRejectionPnP(Frame::Ptr frame_k_1,
    //                          Frame::Ptr frame_k) const;


    /**
     * @brief Defines the result of checking the image container which is a done polymorphically per module (as
     * each module has its own requirements)
     *
     * The requirement string indicates what should be true about the ImageContainer for the function to return true;
     *
     */
    struct ImageValidationResult {
        const bool valid_;
        const std::string requirement_; //printed if result is not valid. Set by the function call

        ImageValidationResult(bool valid, const std::string& requirement) :
            valid_(valid), requirement_(requirement) {}

    };

protected:

    void validateInput(const FrontendInputPacketBase::ConstPtr& input) const override;

    /**
     * @brief Checks that the incoming ImageContainer meeets the minimum requirement for the derived module.
     *
     * This is specifically to check that the right image types are contained with the container.
     * If the result is false, InvalidImageContainerException is thrown with the requirements specified by the
     * returned ImageValidationResult.
     *
     * @param image_container const ImageContainer::Ptr&
     * @return ImageValidationResult
     */
    virtual ImageValidationResult validateImageContainer(const ImageContainer::Ptr& image_container) const = 0;

protected:
    const FrontendParams base_params_;
    ImageDisplayQueue* display_queue_;

    GroundTruthPacketMap gt_packet_map_;
    ObjectPoseMap object_poses_; //! Keeps a track of the current object locations by propogating the motions. Really just (viz)
    gtsam::Pose3Vector camera_poses_; //! Keeps track of current camera trajectory. Really just for (viz) and drawn everytime

};







} //dyno
