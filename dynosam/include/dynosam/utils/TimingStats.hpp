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

#include "dynosam/utils/Statistics.hpp"
#include "dynosam/utils/Timing.hpp"
#include <atomic>

namespace dyno {
namespace utils {

class TimingStatsCollector {
public:
    TimingStatsCollector(const std::string& tag);
    ~TimingStatsCollector();

    /**
     * @brief Resets the current tic_time and sets is_valid = true,
     * such that when the collector tries to toc, the tic (comparison time) will be valid
     *
     *
     */
    void reset();

    bool isValid() const;

private:
    /**
     * @brief Creates a toc time to compare against the latest tic time and logs the diff
     * as sample to the collector
     *
     * Only logs if is_valid_ == true, after which is_valid will be set to false.
     * The collector then needs to be reset to be used again
     *
     */
    void tocAndLog();

private:
    Timer::TimePoint tic_time_; //! Time on creation (the time to compare against)
    StatsCollector collector_;

    std::atomic_bool is_valid_{true}; //! Indiactes validity of the tic_time and if the collector has been reset allowing a new toc to be made

};



} //utils
} //dyno
