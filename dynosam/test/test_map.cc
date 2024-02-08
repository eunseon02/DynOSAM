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
#include "internal/helpers.hpp"


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
        measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(
            i, background_label, 0
        ));

        expected_tracklets.push_back(i);
    }

    Map::Ptr map = Map::create();
    map->updateObservations(measurements);

    EXPECT_TRUE(map->frameExists(0));
    EXPECT_FALSE(map->frameExists(1));

    EXPECT_TRUE(map->landmarkExists(0));
    EXPECT_TRUE(map->landmarkExists(9));
    EXPECT_FALSE(map->landmarkExists(10));

    EXPECT_EQ(map->getStaticTrackletsByFrame(0), expected_tracklets);

    //expected tracklets in frame 0
    TrackletIds expected_tracklets_f0 = expected_tracklets;

    TrackletIds expected_tracklets_f1;
    //add another 5 points at frame 1
    for(size_t i = 0; i < 5; i++) {
        measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(
            i, background_label, 1
        ));

        expected_tracklets.push_back(i);
        expected_tracklets_f1.push_back(i);
    }

    //apply update
    map->updateObservations(measurements);

    EXPECT_EQ(map->getStaticTrackletsByFrame(0), expected_tracklets_f0);
    EXPECT_EQ(map->getStaticTrackletsByFrame(1), expected_tracklets_f1);

    //check for frames in some landmarks
    //should be seen in frames 0 and 1
    LandmarkNode::Ptr lmk1 = map->getLandmark(0);
    std::vector<FrameId> lmk_1_seen_frames = lmk1->frames_seen.collectIds<FrameId>();
    std::vector<FrameId> lmk_1_seen_frames_expected = {0, 1};
    EXPECT_EQ(lmk_1_seen_frames, lmk_1_seen_frames_expected);

    //should be seen in frames 0
    LandmarkNode::Ptr lmk6 = map->getLandmark(6);
    std::vector<FrameId> lmk_6_seen_frames = lmk6->frames_seen.collectIds<FrameId>();
    std::vector<FrameId> lmk_6_seen_frames_expected = {0};
    EXPECT_EQ(lmk_6_seen_frames, lmk_6_seen_frames_expected);

    //check that the frames here are the ones in the map
    EXPECT_EQ(map->getFrame(lmk_1_seen_frames.at(0)), map->getFrame(lmk_6_seen_frames.at(0)));

    //finally check that there are no objects
    EXPECT_EQ(map->getFrame(lmk_1_seen_frames.at(0))->objects_seen.size(),0);
    EXPECT_EQ(map->numObjectsSeen(), 0u);
}


TEST(Map, setStaticOrdering) {
    //add frames out of order
    Map::Ptr map = Map::create();

    StatusKeypointMeasurements measurements;
    //frame 0
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 0, 0));
    //frame 2
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 0, 2));
    //frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 0, 1));
    //frame 3
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 0, 3));
    map->updateObservations(measurements);

    auto lmk = map->getLandmark(1);
    EXPECT_TRUE(lmk != nullptr);

    FrameIds expected_frame_ids = {0, 1, 2, 3};
    EXPECT_EQ(lmk->frames_seen.collectIds<FrameId>(), expected_frame_ids);

    EXPECT_EQ(lmk->frames_seen.getFirstIndex(), 0u);
    EXPECT_EQ(lmk->frames_seen.getLastIndex(), 3u);

}

TEST(Map, basicObjectAdd) {
    Map::Ptr map = Map::create();
    StatusKeypointMeasurements measurements;
    //add two dynamic points on object 1 and frames 0 and 1
    //tracklet 0, object 1 frame 0
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
    //tracklet 0, object 1 frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));

    map->updateObservations(measurements);
    EXPECT_EQ(map->numObjectsSeen(), 1u);
    EXPECT_TRUE(map->objectExists(1));

    auto object1 = map->getObject(1);
    //object 1 should have 1 point seen at frames 0 and 1
    EXPECT_EQ(object1->dynamic_landmarks.collectIds<TrackletId>(), TrackletIds{0});
    EXPECT_EQ(object1->dynamic_landmarks.size(), 1u);

    //now check that the frames also have these measurements
    auto frame_0 = map->getFrame(0);
    auto frame_1 = map->getFrame(1);
    EXPECT_TRUE(frame_0 != nullptr);
    EXPECT_TRUE(frame_1 != nullptr);

    EXPECT_EQ(frame_0->dynamic_landmarks.collectIds<TrackletId>(), TrackletIds{0});
    EXPECT_EQ(frame_1->dynamic_landmarks.collectIds<TrackletId>(), TrackletIds{0});

    //sanity check that there are no static points
    EXPECT_EQ(frame_0->static_landmarks.size(), 0u);
    EXPECT_EQ(frame_1->static_landmarks.size(), 0u);


    //check object id and seen frames
    auto lmk_0 = map->getLandmark(0);
    EXPECT_EQ(lmk_0->object_id, 1); //object Id 1;
    EXPECT_EQ(lmk_0->frames_seen.collectIds<FrameId>(), FrameIds({0, 1})); //seen frames


    //finally check that the landmark referred to by the frames are the same one as getLandmark(0)
    //this also implicitly tests FastMapNodeSet::find(index)
    auto frame_0_dynamic_lmks = frame_0->dynamic_landmarks;
    auto frame_1_dynamic_lmks = frame_1->dynamic_landmarks;

    //look from the lmk with id 0
    auto lmk_itr_frame_0 = frame_0_dynamic_lmks.find(0);
    auto lmk_itr_frame_1 = frame_1_dynamic_lmks.find(0);
    //should not be at the end as we have this landmark
    EXPECT_FALSE(lmk_itr_frame_0 == frame_0_dynamic_lmks.end());
    EXPECT_FALSE(lmk_itr_frame_1 == frame_1_dynamic_lmks.end());

    //check the lmk is the one we got from the map
    EXPECT_EQ(lmk_0, *lmk_itr_frame_0);
    EXPECT_EQ(lmk_0, *lmk_itr_frame_1);
}

TEST(Map, objectSeenFrames) {
    Map::Ptr map = Map::create();
    StatusKeypointMeasurements measurements;

    //add 2 objects
    //object 1 seen at frames 0 and 1
    //tracklet 0, object 1 frame 0
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
    //tracklet 0, object 1 frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));
    //tracklet 0 has two observations

    //object 2 seen at frames 1 and 2
    //tracklet 1, object 2 frame 1
    //tracklet 1 has 1 observations
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 2, 1));
    //tracklet 2, object 2 frame 2
    //tracklet 2 has 1 observations
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(2, 2, 2));

    //object 3 seen at frames 0, 1, 2
    //tracklet 3, object 3 frame 0
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 0));
    //tracklet 3, object 3 frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 1));
    //tracklet 3, object 3 frame 2
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 2));
    //tracklet 3 has 3 observations

    map->updateObservations(measurements);
    EXPECT_EQ(map->numObjectsSeen(), 3u);

    auto object1 = map->getObject(1);
    auto object2 = map->getObject(2);
    auto object3 = map->getObject(3);

    FrameNodePtrSet expected_frame_set_object1;
    expected_frame_set_object1.insert(CHECK_NOTNULL(map->getFrame(0)));
    expected_frame_set_object1.insert(CHECK_NOTNULL(map->getFrame(1)));

    FrameNodePtrSet expected_frame_set_object2;
    expected_frame_set_object2.insert(CHECK_NOTNULL(map->getFrame(1)));
    expected_frame_set_object2.insert(CHECK_NOTNULL(map->getFrame(2)));

    FrameNodePtrSet expected_frame_set_object3;
    expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(0)));
    expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(1)));
    expected_frame_set_object3.insert(CHECK_NOTNULL(map->getFrame(2)));

    EXPECT_EQ(object1->getSeenFrames(), expected_frame_set_object1);
    EXPECT_EQ(object2->getSeenFrames(), expected_frame_set_object2);
    EXPECT_EQ(object3->getSeenFrames(), expected_frame_set_object3);


}

TEST(Map, getLandmarksSeenAtFrame) {
    Map::Ptr map = Map::create();
    StatusKeypointMeasurements measurements;

    //add 2 objects
    //object 1 seen at frames 0 and 1
    //tracklet 0, object 1 frame 0
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 0));
    //tracklet 0, object 1 frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 1, 1));

    //object 2 seen at frames 1 and 2
    //tracklet 1, object 2 frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(1, 2, 1));
    //tracklet 2, object 2 frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(2, 2, 1));

    //object 3 seen at frames 0, 1, 2
    //tracklet 3, object 3 frame 0
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 0));
    //tracklet 3, object 3 frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(3, 3, 1));
    //tracklet 3, object 3 frame 1
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(4, 3, 1));
    //tracklet 3 has 3 observations

    map->updateObservations(measurements);
    EXPECT_EQ(map->numObjectsSeen(), 3u);

    auto object1 = map->getObject(1);
    auto object2 = map->getObject(2);
    auto object3 = map->getObject(3);

    LandmarkNodePtrSet expected_lmk_set_object1_frame0;
    expected_lmk_set_object1_frame0.insert(CHECK_NOTNULL(map->getLandmark(0)));

    LandmarkNodePtrSet expected_lmk_set_object1_frame1;
    expected_lmk_set_object1_frame1.insert(CHECK_NOTNULL(map->getLandmark(0)));

    LandmarkNodePtrSet expected_lmk_set_object2_frame1;
    expected_lmk_set_object2_frame1.insert(CHECK_NOTNULL(map->getLandmark(1)));
    expected_lmk_set_object2_frame1.insert(CHECK_NOTNULL(map->getLandmark(2)));

    LandmarkNodePtrSet expected_lmk_set_object3_frame0;
    expected_lmk_set_object3_frame0.insert(CHECK_NOTNULL(map->getLandmark(3)));

    LandmarkNodePtrSet expected_lmk_set_object3_frame1;
    expected_lmk_set_object3_frame1.insert(CHECK_NOTNULL(map->getLandmark(3)));
    expected_lmk_set_object3_frame1.insert(CHECK_NOTNULL(map->getLandmark(4)));

    EXPECT_EQ(object1->getLandmarksSeenAtFrame(0), expected_lmk_set_object1_frame0);
    EXPECT_EQ(object1->getLandmarksSeenAtFrame(1), expected_lmk_set_object1_frame1);
    EXPECT_EQ(object1->getLandmarksSeenAtFrame(2), LandmarkNodePtrSet{});
    EXPECT_EQ(object2->getLandmarksSeenAtFrame(1), expected_lmk_set_object2_frame1);
    EXPECT_EQ(object3->getLandmarksSeenAtFrame(0), expected_lmk_set_object3_frame0);
    EXPECT_EQ(object3->getLandmarksSeenAtFrame(1), expected_lmk_set_object3_frame1);


}


TEST(Map, testSimpleEstimateAccess) {
    Map::Ptr map = Map::create();
    StatusKeypointMeasurements measurements;

    //frame cannot be 0? what is invalid frame then?
    EXPECT_EQ(map->lastEstimateUpdate(), 0u);

    //tracklet 0, static, frame
    measurements.push_back(dyno_testing::makeStatusKeypointMeasurement(0, 0, 0));
    map->updateObservations(measurements);

    auto frame_0 = map->getFrame(0);
    auto pose_0_query = frame_0->getPoseEstimate();
    EXPECT_FALSE(pose_0_query);
    EXPECT_FALSE(pose_0_query.isValid());

    gtsam::Values estimate;

    //some random pose
    gtsam::Rot3 R = gtsam::Rot3::Rodrigues(0.3,0.4,-0.5);
    gtsam::Point3 t(3.5,-8.2,4.2);
    gtsam::Pose3 pose_0_actual(R,t);
    gtsam::Key pose_0_key = CameraPoseSymbol(0);
    estimate.insert(pose_0_key, pose_0_actual);

    map->updateEstimates(estimate, gtsam::NonlinearFactorGraph{}, 0);
    pose_0_query = frame_0->getPoseEstimate();

    EXPECT_TRUE(pose_0_query);
    EXPECT_TRUE(pose_0_query.isValid());
    EXPECT_TRUE(gtsam::assert_equal(pose_0_query.get(), pose_0_actual));
    EXPECT_EQ(pose_0_query.key_, pose_0_key);

}
