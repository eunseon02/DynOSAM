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

#include "dynosam/common/Map.hpp"
#include "dynosam/common/Types.hpp"


#include <glog/logging.h>
#include <gtest/gtest.h>
#include <exception>

using namespace dyno;

TEST(Map, basicAddOnlyStatic) {

    StatusKeypointMeasurements measurements;
    TrackletIds expected_tracklets;
    //10 measurements with unique tracklets at frame 0
    for(size_t i = 0; i < 10; i++) {
        KeypointStatus status = KeypointStatus::Static(0);
        auto estimate = std::make_pair(i, Keypoint());
        measurements.push_back(std::make_pair(status, estimate));

        expected_tracklets.push_back(i);
    }

    Map map;
    map.updateObservations(measurements);

    EXPECT_TRUE(map.frameExists(0));
    EXPECT_FALSE(map.frameExists(1));

    EXPECT_TRUE(map.trackletExists(0));
    EXPECT_TRUE(map.trackletExists(9));
    EXPECT_FALSE(map.trackletExists(10));

    EXPECT_EQ(map.getStaticTrackletsByFrame(0), expected_tracklets);

    //add another 5 points at frame 1
    for(size_t i = 0; i < 5; i++) {
        KeypointStatus status = KeypointStatus::Static(1);
        auto estimate = std::make_pair(i, Keypoint());
        measurements.push_back(std::make_pair(status, estimate));

        expected_tracklets.push_back(i);
    }

    //apply update
    map.updateObservations(measurements);

    EXPECT_EQ(map.getStaticTrackletsByFrame(0), expected_tracklets);


}
