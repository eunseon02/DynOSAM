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

#include "dynosam/backend/DynamicPointSymbol.hpp"
#include <glog/logging.h>

namespace dyno
{

size_t DynamicPointSymbol::id_{ 0 };
FrameTrackletSymbolMap DynamicPointSymbol::frame_tracklet_map_{};
DynamicPointSymbol::KeyTrackletPair DynamicPointSymbol::key_tracklet_map_{};

DynamicPointSymbol::DynamicPointSymbol() : gtsam::Symbol()
{
}

DynamicPointSymbol::DynamicPointSymbol(size_t frame_id, size_t tracklet_id, unsigned char c)
  : gtsam::Symbol(), frame_id_(frame_id), tracklet_id_(tracklet_id)
{
  gtsam::Symbol sym = makeUniqueSymbolFromPair(FrameTrackletPair(frame_id, tracklet_id), c);
  this->c_ = sym.chr();
  this->j_ = sym.index();
}

DynamicPointSymbol::DynamicPointSymbol(gtsam::Symbol sym) : DynamicPointSymbol(gtsam::Key(sym))
{
}

DynamicPointSymbol::DynamicPointSymbol(gtsam::Key key)
{
  FrameTrackletPair pair;
  if (key_tracklet_map_.find(key) != key_tracklet_map_.end())
  {
    pair = key_tracklet_map_.at(key);
  }
  else
  {
    throw std::runtime_error("DynamicPointSymbol is being made from Symbol " + gtsam::DefaultKeyFormatter(key) +
                             " but key has not appeared before."
                             " Dynamic Symbol must be constructed with a frame and tracklet ID and then "
                             "DynamicPointSymbol can be reconstructed via DynamicPointSymbol(gtsam::Key)");
  }
  gtsam::Symbol sym = makeUniqueSymbolFromPair(pair, gtsam::Symbol(key).chr());
  this->c_ = sym.chr();
  this->j_ = sym.index();
  this->tracklet_id_ = pair.tracklet_id_;
  this->frame_id_ = pair.frame_id_;
}

gtsam::Symbol DynamicPointSymbol::makeUniqueSymbolFromPair(const FrameTrackletPair& pair, unsigned char c)
{
  // exists, therefore return key
  gtsam::Symbol sym;
  if (frame_tracklet_map_.find(pair) != frame_tracklet_map_.end())
  {
    sym = frame_tracklet_map_.at(pair);
  }
  else
  {
    // make new symbol
    sym = gtsam::Symbol(c, DynamicPointSymbol::id_);
    DynamicPointSymbol::id_++;
    frame_tracklet_map_[pair] = sym;
    key_tracklet_map_[sym] = pair;
  }
  return sym;
}

size_t DynamicPointSymbol::frameID() const
{
  return frame_id_;
}
size_t DynamicPointSymbol::trackletID() const
{
  return tracklet_id_;
}

void DynamicPointSymbol::clear()
{
  id_ = 0;
  frame_tracklet_map_.clear();
}

};  // namespace dyno
