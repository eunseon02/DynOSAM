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

#include "dynosam/factors/LandmarkQuadricFactor.hpp"
#include <gtsam/base/numericalDerivative.h>

#include <boost/bind/bind.hpp>

namespace dyno {

gtsam::Vector LandmarkQuadricFactor::evaluateError(const gtsam::Point3& m_world, const gtsam::Pose3& L_world, const gtsam::Vector3& P,
                              boost::optional<gtsam::Matrix&> J1,
                              boost::optional<gtsam::Matrix&> J2,
                              boost::optional<gtsam::Matrix&> J3) const {

        if(J1) {
            // error w.r.t to point in world
            Eigen::Matrix<double, 1, 3> dq_dm =
                gtsam::numericalDerivative31<gtsam::Vector1, gtsam::Point3, gtsam::Pose3, gtsam::Vector3>(
                    std::bind(&LandmarkQuadricFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3),
                m_world, L_world, P);
            *J1 = dq_dm;
        }

        if(J2) {
            //error w.r.t to object pose
            Eigen::Matrix<double, 1, 6> dq_dx =
                gtsam::numericalDerivative32<gtsam::Vector1, gtsam::Point3, gtsam::Pose3, gtsam::Vector3>(
                    std::bind(&LandmarkQuadricFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3),
                m_world, L_world, P);
            *J2 = dq_dx;
        }

        if(J3) {
            //error w.r.t to radii
            Eigen::Matrix<double, 1, 3> dq_dp =
                gtsam::numericalDerivative33<gtsam::Vector1, gtsam::Point3, gtsam::Pose3, gtsam::Vector3>(
                    std::bind(&LandmarkQuadricFactor::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3),
                m_world, L_world, P);
            *J3 = dq_dp;
        }
        return residual(m_world, L_world, P);
    }


gtsam::Vector1 LandmarkQuadricFactor::residual(const gtsam::Point3& m_world, const gtsam::Pose3& L_world, const gtsam::Vector3& P) {
    const gtsam::Matrix44 Q = constructQ(P);

    const gtsam::Matrix44 L_inverse_matrix = L_world.inverse().matrix();
    gtsam::Vector4 m_world_homogenous(m_world(0), m_world(1), m_world(2), 1);
    gtsam::Vector4 m_local_homogenous = L_inverse_matrix * m_world_homogenous;

    double error = m_local_homogenous.transpose() * Q * m_local_homogenous;
    // double error = m_world_homogenous.transpose() * (L_inverse_matrix.transpose() * Q * L_inverse_matrix) * m_world_homogenous;
    return gtsam::Vector1(error);
}


gtsam::Matrix44 LandmarkQuadricFactor::constructQ(const gtsam::Vector3& radii) {
    auto exp = [](double x) {
        return 1.0/(pow(x, 2));
    };

    // gtsam::Matrix44 Qc = (gtsam::Vector4() << (1.0 / radii).array().pow(2), -1.0)
    //                     .finished().asDiagonal();
    gtsam::Matrix44 Qc = (gtsam::Vector4() << (radii.unaryExpr(exp)).array(), -1.0)
                        .finished().asDiagonal();

    return Qc;
}


void LandmarkQuadricFactor::print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const {
    std::cout << s << "LandmarkQuadricFactor("
              << keyFormatter(key1()) << ","
              << keyFormatter(key2()) << ","
              << keyFormatter(key3()) << ")" << std::endl;
    std::cout << "    NoiseModel: ";
    noiseModel()->print();
    std::cout << std::endl;
}

bool LandmarkQuadricFactor::equals(const LandmarkQuadricFactor& other, double tol) const {
    return key1() == other.key1() && key2() == other.key2() && key3() == other.key3() &&
        noiseModel()->equals(*other.noiseModel(), tol);
}

} //dyno
