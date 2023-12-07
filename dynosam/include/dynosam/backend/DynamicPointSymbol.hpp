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

struct FrameTrackletPair
{
  const FrameId frame_id_ = 0;
  const TrackletId tracklet_id_ = 0;

  FrameTrackletPair()
  {
  }

  FrameTrackletPair(FrameId frame_id, TrackletId tracklet_id) : frame_id_(frame_id), tracklet_id_(tracklet_id)
  {
  }

  bool operator==(const FrameTrackletPair& other) const
  {
    return (frame_id_ == other.frame_id_ && tracklet_id_ == other.tracklet_id_);
  }
};

struct FrameTrackletHash
{
  /// number was arbitrarily chosen with no good justification
  static constexpr size_t sl = 17191;
  static constexpr size_t sl2 = sl * sl;

  std::size_t operator()(const FrameTrackletPair& index) const
  {
    return static_cast<unsigned int>(index.frame_id_ * sl + index.tracklet_id_ * sl2);
  }
};

template <typename ValueType>
struct FrameTrackletHashMapType
{
  typedef std::unordered_map<FrameTrackletPair, ValueType, FrameTrackletHash> type;
};

using FrameTrackletSymbolMap = FrameTrackletHashMapType<gtsam::Symbol>::type;

class DynamicPointSymbol : public gtsam::Symbol
{
public:
  DynamicPointSymbol();
  DynamicPointSymbol(FrameId frame_id, size_t tracklet_id, unsigned char c);
  DynamicPointSymbol(gtsam::Symbol sym);
  DynamicPointSymbol(gtsam::Key key);

  FrameId frameID() const;
  TrackletId trackletID() const;

  /**
   * @brief Resets the static trackletId to 0 and clears the frame_tracklet_map_. Only really used for testing.
   *
   */
  static void clear();

private:
  gtsam::Symbol makeUniqueSymbolFromPair(const FrameTrackletPair& pair, unsigned char c);

private:
  FrameId frame_id_;
  TrackletId tracklet_id_;

  static FrameTrackletSymbolMap frame_tracklet_map_;

  using KeyTrackletPair = std::map<gtsam::Key, FrameTrackletPair>;
  static KeyTrackletPair key_tracklet_map_;  // used for reverse look up
  static size_t id_;
};


}
