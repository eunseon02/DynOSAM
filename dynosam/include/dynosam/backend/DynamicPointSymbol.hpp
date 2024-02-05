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

#include <map>
#include <unordered_map>

#include <gtsam/inference/Symbol.h>

namespace dyno
{

//implementation of Contor's pairing function: http://szudzik.com/ElegantPairing.pdf
//When x and y are non−negative integers, pairing them outputs a single non−negative integer
// that is uniquely associated with that pair.
struct CantorPairingFunction {
    using Pair = std::pair<std::uint64_t, std::uint64_t>;

    static std::uint64_t pair(const Pair& input);
    static Pair depair(const std::uint64_t z);
};



class DynamicPointSymbol {
protected:
    unsigned char c_;
    std::uint64_t j_;

    TrackletId tracklet_id_;
    FrameId frame_id_;

public:

    /** Default constructor */
    DynamicPointSymbol() :
        c_(0), j_(0), tracklet_id_(-1), frame_id_(0) {
    }

    /** Copy constructor */
    DynamicPointSymbol(const DynamicPointSymbol& key);
    /** Constructor */
    DynamicPointSymbol(unsigned char c, TrackletId tracklet_id, FrameId frame_id);

    /** Constructor that decodes an integer Key */
    DynamicPointSymbol(gtsam::Key key);

    /** return Key (integer) representation */
    gtsam::Key key() const;

    /** Cast to integer */
    operator gtsam::Key() const { return key(); }

    /// Print
    void print(const std::string& s = "") const;

    /// Check equality
    bool equals(const DynamicPointSymbol& expected, double tol = 0.0) const;

    /** Retrieve key character */
    unsigned char chr() const {
        return c_;
    }

    /** Retrieve key index */
    std::uint64_t index() const {
        return j_;
    }

    TrackletId trackletId() const {
        return tracklet_id_;
    }

    FrameId frameId() const {
        return frame_id_;
    }



    /** Create a string from the key */
    operator std::string() const;

    /// Return string representation of the key
    std::string string() const { return std::string(*this); }

    /** Comparison for use in maps */
    bool operator<(const DynamicPointSymbol& comp) const {
        return c_ < comp.c_ || (comp.c_ == c_ && j_ < comp.j_);
    }

    /** Comparison for use in maps */
    bool operator==(const DynamicPointSymbol& comp) const {
        return comp.c_ == c_ && comp.j_ == j_;
    }

    /** Comparison for use in maps */
    bool operator==(gtsam::Key comp) const {
        return comp == (gtsam::Key)(*this);
    }

    /** Comparison for use in maps */
    bool operator!=(const DynamicPointSymbol& comp) const {
        return comp.c_ != c_ || comp.j_ != j_;
    }

    /** Comparison for use in maps */
    bool operator!=(gtsam::Key comp) const {
        return comp != (gtsam::Key)(*this);
    }


private:
    gtsam::Symbol asSymbol() const;

    static std::uint64_t constructIndex(TrackletId tracklet_id, FrameId frame_id);
    static void recover(std::uint64_t z, TrackletId& tracklet_id, FrameId& frame_id);

};


}
