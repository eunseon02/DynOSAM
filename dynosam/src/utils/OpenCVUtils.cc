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

#include "dynosam/utils/OpenCVUtils.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include<iostream>
#include<fstream>

namespace dyno {
namespace utils {

const float FLOW_TAG_FLOAT = 202021.25f;
const char *FLOW_TAG_STRING = "PIEH";

cv::Mat readOpticalFlow( const std::string& path ) {
    using namespace cv;

    Mat_<Point2f> flow;
    std::ifstream file(path.c_str(), std::ios_base::binary);
    if ( !file.good() )
        return CV_CXX_MOVE(flow); // no file - return empty matrix

    float tag;
    file.read((char*) &tag, sizeof(float));
    if ( tag != FLOW_TAG_FLOAT )
        return CV_CXX_MOVE(flow);

    int width, height;

    file.read((char*) &width, 4);
    file.read((char*) &height, 4);

    flow.create(height, width);

    for ( int i = 0; i < flow.rows; ++i )
    {
        for ( int j = 0; j < flow.cols; ++j )
        {
            Point2f u;
            file.read((char*) &u.x, sizeof(float));
            file.read((char*) &u.y, sizeof(float));
            if ( !file.good() )
            {
                flow.release();
                return CV_CXX_MOVE(flow);
            }

            flow(i, j) = u;
        }
    }
    file.close();
    return CV_CXX_MOVE(flow);

}


bool writeOpticalFlow( const std::string& path, const cv::Mat& flow) {
    using namespace cv;
    const int nChannels = 2;

    Mat input = flow;
    if ( input.channels() != nChannels || input.depth() != CV_32F || path.length() == 0 )
        return false;

    std::ofstream file(path.c_str(), std::ofstream::binary);
    if ( !file.good() )
        return false;

    int nRows, nCols;

    nRows = (int) input.size().height;
    nCols = (int) input.size().width;

    const int headerSize = 12;
    char header[headerSize];
    memcpy(header, FLOW_TAG_STRING, 4);
    // size of ints is known - has been asserted in the current function
    memcpy(header + 4, reinterpret_cast<const char*>(&nCols), sizeof(nCols));
    memcpy(header + 8, reinterpret_cast<const char*>(&nRows), sizeof(nRows));
    file.write(header, headerSize);
    if ( !file.good() )
        return false;

//    if ( input.isContinuous() ) //matrix is continous - treat it as a single row
//    {
//        nCols *= nRows;
//        nRows = 1;
//    }

    int row;
    char* p;
    for ( row = 0; row < nRows; row++ )
    {
        p = input.ptr<char>(row);
        file.write(p, nCols * nChannels * sizeof(float));
        if ( !file.good() )
            return false;
    }
    file.close();
    return true;
}

} //utils
} //dyno
