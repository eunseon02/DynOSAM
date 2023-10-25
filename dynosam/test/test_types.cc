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

#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/vision/Feature.hpp"

using namespace dyno;


#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(ImageType, testRGBMonoValidation) {

    {
        //invalid type
        cv::Mat input(cv::Size(50, 50), CV_64F);
        EXPECT_THROW({ImageType::RGBMono::validate(input);}, InvalidImageTypeException);
    }

    {
        //okay type
        cv::Mat input(cv::Size(50, 50), CV_8UC1);
        EXPECT_NO_THROW({ImageType::RGBMono::validate(input);});
    }

    {
        //okay type
        cv::Mat input(cv::Size(50, 50), CV_8UC3);
        EXPECT_NO_THROW({ImageType::RGBMono::validate(input);});
    }


}

TEST(ImageType, testDepthValidation) {

    {
        //okay type
        cv::Mat input(cv::Size(50, 50), CV_64F);
        EXPECT_NO_THROW({ImageType::Depth::validate(input);});
    }

    {
        //invalud type
        cv::Mat input(cv::Size(50, 50), CV_8UC3);
        EXPECT_THROW({ImageType::Depth::validate(input);}, InvalidImageTypeException);
    }

}


TEST(ImageType, testOpticalFlowValidation) {

    {
        //okay type
        cv::Mat input(cv::Size(50, 50), CV_32FC2);
        EXPECT_NO_THROW({ImageType::OpticalFlow::validate(input);});
    }

    {
        //invalud type
        cv::Mat input(cv::Size(50, 50), CV_8UC3);
        EXPECT_THROW({ImageType::OpticalFlow::validate(input);}, InvalidImageTypeException);
    }

}

TEST(ImageType, testSemanticMaskValidation) {
    //TODO:

}

TEST(ImageType, testMotionMaskValidation) {
    //TODO:
}


TEST(ImageContainerSubset, testBasicSubsetContainer) {

    cv::Mat depth(cv::Size(25, 25), CV_64F);
    uchar* depth_ptr = depth.data;

    cv::Mat optical_flow(cv::Size(25, 25), CV_32FC2);
    uchar* optical_flow_ptr = optical_flow.data;
    ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow> ics{
        ImageWrapper<ImageType::Depth>(depth),
        ImageWrapper<ImageType::OpticalFlow>(optical_flow)
    };

    cv::Mat retrieved_depth = ics.get<ImageType::Depth>();
    cv::Mat retrieved_of = ics.get<ImageType::OpticalFlow>();

    EXPECT_EQ(retrieved_depth.data, depth_ptr);
    EXPECT_EQ(retrieved_of.data, optical_flow_ptr);

    EXPECT_EQ(depth.size(), retrieved_depth.size());
    EXPECT_EQ(optical_flow.size(), retrieved_of.size());

}


TEST(ImageContainerSubset, testSubsetContainerExists) {

    cv::Mat depth(cv::Size(25, 25), CV_64F);
    cv::Mat optical_flow(cv::Size(25, 25), CV_32FC2);
    ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow, ImageType::RGBMono> ics{
        ImageWrapper<ImageType::Depth>(depth),
        ImageWrapper<ImageType::OpticalFlow>(optical_flow),
        ImageWrapper<ImageType::RGBMono>()
    };

    EXPECT_TRUE(ics.exists<ImageType::Depth>());
    EXPECT_TRUE(ics.exists<ImageType::OpticalFlow>());

    EXPECT_FALSE(ics.exists<ImageType::RGBMono>());

}


TEST(ImageContainerSubset, testBasicMakeSubset) {

    cv::Mat depth(cv::Size(50, 50), CV_64F);
    cv::Mat optical_flow(cv::Size(50, 50), CV_32FC2);
    uchar* optical_flow_ptr = optical_flow.data;
    ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow> ics{
        ImageWrapper<ImageType::Depth>(depth),
        ImageWrapper<ImageType::OpticalFlow>(optical_flow)
    };

    EXPECT_EQ(ics.index<ImageType::OpticalFlow>(), 1u);

    ImageContainerSubset<ImageType::OpticalFlow> subset = ics.makeSubset<ImageType::OpticalFlow>();
    EXPECT_EQ(subset.index<ImageType::OpticalFlow>(), 0u);

    cv::Mat retrieved_of = ics.get<ImageType::OpticalFlow>();
    cv::Mat subset_retrieved_of = subset.get<ImageType::OpticalFlow>();

    EXPECT_EQ(retrieved_of.data, optical_flow_ptr);
    EXPECT_EQ(subset_retrieved_of.data, optical_flow_ptr);

}


TEST(ImageContainerSubset, testSafeGet) {

    cv::Mat depth(cv::Size(25, 25), CV_64F);
    cv::Mat optical_flow(cv::Size(25, 25), CV_32FC2);
    ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow, ImageType::RGBMono> ics{
        ImageWrapper<ImageType::Depth>(depth),
        ImageWrapper<ImageType::OpticalFlow>(optical_flow),
        ImageWrapper<ImageType::RGBMono>()
    };

    cv::Mat tmp;
    EXPECT_FALSE(ics.safeGet<ImageType::RGBMono>(tmp));

    //tmp shoudl now be the same as the depth ptr
    EXPECT_TRUE(ics.safeGet<ImageType::Depth>(tmp));
    EXPECT_EQ(depth.data, tmp.data);
    EXPECT_EQ(depth.size(), tmp.size());

    //tmp shoudl now be the same as the optical flow ptr
    EXPECT_TRUE(ics.safeGet<ImageType::OpticalFlow>(tmp));
    EXPECT_EQ(optical_flow.data, tmp.data);
    EXPECT_EQ(optical_flow.size(), tmp.size());
}


TEST(ImageContainerSubset, testSafeClone) {

    cv::Mat depth(cv::Size(50, 50), CV_64F);
    cv::Mat optical_flow(cv::Size(50, 50), CV_32FC2);
    ImageContainerSubset<ImageType::Depth, ImageType::OpticalFlow, ImageType::RGBMono> ics{
        ImageWrapper<ImageType::Depth>(depth),
        ImageWrapper<ImageType::OpticalFlow>(optical_flow),
        ImageWrapper<ImageType::RGBMono>()
    };

    cv::Mat tmp;
    EXPECT_FALSE(ics.cloneImage<ImageType::RGBMono>(tmp));
    EXPECT_TRUE(tmp.empty());

    EXPECT_TRUE(ics.cloneImage<ImageType::Depth>(tmp));
    EXPECT_NE(depth.data, tmp.data); //no longer the same data
    EXPECT_EQ(depth.size(), tmp.size());

    EXPECT_TRUE(ics.cloneImage<ImageType::OpticalFlow>(tmp));
    EXPECT_NE(optical_flow.data, tmp.data); //no longer the same data
    EXPECT_EQ(optical_flow.size(), tmp.size());
}

TEST(ImageContainer, testImageContainerIndexing) {

    EXPECT_EQ(ImageContainer::Index<ImageType::RGBMono>(), 0u);
    EXPECT_EQ(ImageContainer::Index<ImageType::Depth>(), 1u);
    EXPECT_EQ(ImageContainer::Index<ImageType::OpticalFlow>(), 2u);
    EXPECT_EQ(ImageContainer::Index<ImageType::SemanticMask>(), 3u);
    EXPECT_EQ(ImageContainer::Index<ImageType::MotionMask>(), 4u);
}


TEST(ImageContainer, CreateRGBDSemantic) {

    cv::Mat rgb(cv::Size(50, 50), CV_8UC1);
    cv::Mat depth(cv::Size(50, 50), CV_64F);
    cv::Mat optical_flow(cv::Size(50, 50), CV_32FC2);
    cv::Mat semantic_mask(cv::Size(50, 50), CV_32SC1);
    ImageContainer::Ptr rgbd_semantic = ImageContainer::Create(
        0u,
        0u,
        ImageWrapper<ImageType::RGBMono>(rgb),
        ImageWrapper<ImageType::Depth>(depth),
        ImageWrapper<ImageType::OpticalFlow>(optical_flow),
        ImageWrapper<ImageType::SemanticMask>(semantic_mask)
    );

    EXPECT_TRUE(rgbd_semantic->hasSemanticMask());
    EXPECT_TRUE(rgbd_semantic->hasDepth());
    EXPECT_FALSE(rgbd_semantic->hasMotionMask());
    EXPECT_FALSE(rgbd_semantic->isMonocular());


}

TEST(ImageContainer, CreateRGBDSemanticWithInvalidSizes) {

    cv::Mat rgb(cv::Size(25, 25), CV_8UC1);
    cv::Mat depth(cv::Size(50, 50), CV_64F);
    cv::Mat optical_flow(cv::Size(50, 50), CV_32FC2);
    cv::Mat semantic_mask(cv::Size(50, 50), CV_32SC1);
    EXPECT_THROW({ImageContainer::Create(
        0u,
        0u,
        ImageWrapper<ImageType::RGBMono>(rgb),
        ImageWrapper<ImageType::Depth>(depth),
        ImageWrapper<ImageType::OpticalFlow>(optical_flow),
        ImageWrapper<ImageType::SemanticMask>(semantic_mask)
    );}, InvalidImageContainerException);

}


TEST(Feature, checkInvalidState) {

    Feature f;
    EXPECT_TRUE(f.inlier_);
    EXPECT_FALSE(f.usable()); //inlier initally but invalid tracking label

    f.tracklet_id_ = 10;
    EXPECT_TRUE(f.usable());

    f.markInvalid();
    EXPECT_FALSE(f.usable());

    f.tracklet_id_ = 10u;
    EXPECT_TRUE(f.usable());

    f.inlier_ = false;
    EXPECT_FALSE(f.usable());

}

TEST(Feature, checkDepth) {

    Feature f;
    EXPECT_FALSE(f.hasDepth());

    f.depth_ = 12.0;
    EXPECT_TRUE(f.hasDepth());

}
