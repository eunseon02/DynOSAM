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


#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/DynamicPointSymbol.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/utils/Metrics.hpp"

#include <gtsam/inference/LabeledSymbol.h>
#include <gtsam/inference/Symbol.h>

namespace dyno {


bool checkIfLabeledSymbol(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  return (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0);
}

bool reconstructMotionInfo(gtsam::Key key, ObjectId& object_label, FrameId& frame_id) {
  //assume motion key is a labelled symbol
  if(!checkIfLabeledSymbol(key)) {return false; }

  const gtsam::LabeledSymbol as_labeled_symbol(key);
  if(as_labeled_symbol.chr() != kObjectMotionSymbolChar) { return false;}

  frame_id = static_cast<FrameId>(as_labeled_symbol.index());

  char label = as_labeled_symbol.label();
  object_label = label - '0';
  return true;
}

std::string DynoLikeKeyFormatter(gtsam::Key key)
{
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    //if used as motion
    if(asLabeledSymbol.chr() == kObjectMotionSymbolChar) {
      return (std::string) asLabeledSymbol;
    }
    return (std::string) asLabeledSymbol;
  }

  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    if(asLabeledSymbol.chr() == kDynamicLandmarkSymbolChar) {
      const DynamicPointSymbol asDynamicPointSymbol(key);
      return (std::string) asDynamicPointSymbol;
    }
    else {
      return (std::string) asSymbol;
    }

  }
  else {
    return std::to_string(key);
  }
}



SymbolChar DynoChrExtractor(gtsam::Key key) {
   const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    return asLabeledSymbol.chr();
  }
  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    return asSymbol.chr();
  }
  else {
    return InvalidDynoSymbol;
  }

}

std::string DynoLikeKeyFormatterVerbose(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    //if used as motion
    if(asLabeledSymbol.chr() == kObjectMotionSymbolChar) {
      ObjectId object_label;
      FrameId frame_id;
      CHECK(reconstructMotionInfo(asLabeledSymbol, object_label, frame_id));

      std::stringstream ss;
      ss << "H: label" << object_label << ", frames: " << frame_id - 1 << " -> " << frame_id;
      return ss.str();
    }
    return (std::string) asLabeledSymbol;
  }

  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    if(asLabeledSymbol.chr() == kDynamicLandmarkSymbolChar) {
      const DynamicPointSymbol asDynamicPointSymbol(key);

      FrameId frame_id = asDynamicPointSymbol.frameId();
      TrackletId tracklet_id = asDynamicPointSymbol.trackletId();
      std::stringstream ss;
      ss << kDynamicLandmarkSymbolChar << ": frame " << frame_id << ", tracklet " << tracklet_id;
      return ss.str();

    }
    else {
      return (std::string) asSymbol;
    }

  }
  else {
    return std::to_string(key);
  }
}


}
