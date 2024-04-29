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

#include "dynosam/backend/BackendDefinitions.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace dyno {

class LandmarkQuadricFactor : public gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Pose3, gtsam::Vector3> {

public:
    typedef boost::shared_ptr<LandmarkQuadricFactor> shared_ptr;
    typedef LandmarkQuadricFactor This;
    typedef gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Pose3, gtsam::Vector3> Base;

    LandmarkQuadricFactor() {}
    LandmarkQuadricFactor(gtsam::Key landmark_key, gtsam::Key object_pose_key, gtsam::Key quadric_key, gtsam::SharedNoiseModel model)
        :   Base(model, landmark_key, object_pose_key, quadric_key) {}

    gtsam::NonlinearFactor::shared_ptr clone() const override
    {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

     gtsam::Vector evaluateError(const gtsam::Point3& m_world, const gtsam::Pose3& L_world, const gtsam::Vector3& P,
                              boost::optional<gtsam::Matrix&> J1 = boost::none,
                              boost::optional<gtsam::Matrix&> J2 = boost::none,
                              boost::optional<gtsam::Matrix&> J3 = boost::none) const override;

    static gtsam::Matrix44 constructQ(const gtsam::Vector3& radii);
    static gtsam::Vector1 residual(const gtsam::Point3& m_world, const gtsam::Pose3& L_world, const gtsam::Vector3& P);

    gtsam::Key landmarkKey() const { return key1(); }
    gtsam::Key objectPoseKey() const { return key2(); }
    gtsam::Key quadricKey() const { return key3(); }

     /** Prints the boundingbox factor with optional string */
    void print(const std::string& s = "",
                const gtsam::KeyFormatter& keyFormatter =
                    DynoLikeKeyFormatter) const override;

    /** Returns true if equal keys, measurement, noisemodel and calibration */
    bool equals(const LandmarkQuadricFactor& other, double tol = 1e-9) const;

};

} //dyno
