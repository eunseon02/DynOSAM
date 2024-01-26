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

#include "dynosam/utils/CsvWriter.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace dyno;

TEST(CsvHeader, testZeroConstruction) {
    CsvHeader header;
    EXPECT_EQ(header.size(), 0u);
}

TEST(CsvHeader, testConstruction) {


    CsvHeader header("1", "2", "3");
    EXPECT_EQ(header.size(), 3u);

    EXPECT_EQ(header.at(0), "1");
    EXPECT_EQ(header.at(1), "2");
    EXPECT_EQ(header.at(2), "3");

}

TEST(CsvHeader, testToString) {
    CsvHeader header("1", "2", "3");
    const std::string header_string = "1,2,3";
    EXPECT_EQ(header.toString(","), header_string);

}

TEST(CsvHeader, testToStringOneHeader) {
    CsvHeader header("1");
    const std::string header_string = "1";
    EXPECT_EQ(header.toString(), header_string);

}


TEST(CsvWriter, testInvalidHeaderConstruction) {
    EXPECT_THROW({CsvWriter(CsvHeader{});}, InvalidCsvHeaderException);
}


TEST(CsvWriter, testBasicWrite) {
    CsvHeader header("frame_id", "timestamp");
    CsvWriter writer(header);

    writer << 0 << 1.1;
    std::stringstream ss;
    writer.write(ss);

    const std::string expected_write = "frame_id,timestamp\n0,1.1";
    EXPECT_EQ(ss.str(), expected_write);
}


TEST(CsvWriter, testWriteMultiLines) {
    CsvHeader header("frame_id", "timestamp");
    CsvWriter writer(header);

    writer << 0 << 1.1 << 1 << 1.3;
    std::stringstream ss;
    writer.write(ss);

    const std::string expected_write = "frame_id,timestamp\n0,1.1\n1,1.3";
    EXPECT_EQ(ss.str(), expected_write);
}
