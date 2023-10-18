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

TEST(ImageContainer, testImageTypeIndexs) {

    EXPECT_EQ(ImageType::RGBMono::index, 0u);
    EXPECT_EQ(ImageType::MotionMask::index, 4u);

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
