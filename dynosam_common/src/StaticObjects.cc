#include "dynosam_common/StaticObjects.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include <unordered_set>
#include <algorithm>

namespace dyno {
namespace static_objects {

cv::Mat ObjectDetectionResult::colouredMask() const {
  // input image should never be empty
  if (num() == 0 || labelled_mask.empty()) {
    return input_image;
  }
  return utils::labelMaskToRGB(labelled_mask,
                               background_label,
                               input_image);
}

ObjectIds ObjectDetectionResult::objectIds() const {
  if (labelled_mask.empty() || labelled_mask.rows == 0 || labelled_mask.cols == 0) {
    return ObjectIds();
  }
  
  // Extract unique object IDs from the labelled mask
  // Handle different image types (CV_8UC1 vs CV_32SC1)
  const bool is_8bit = (labelled_mask.type() == CV_8UC1);
  std::unordered_set<int> unique_labels;
  
  for (int row = 0; row < labelled_mask.rows; ++row) {
    if (is_8bit) {
      const uint8_t* rowPtr = labelled_mask.ptr<uint8_t>(row);
      for (int col = 0; col < labelled_mask.cols; ++col) {
        int label = static_cast<int>(rowPtr[col]);
        if (label != background_label) {
          unique_labels.insert(label);
        }
      }
    } else {
      const int* rowPtr = labelled_mask.ptr<int>(row);
      for (int col = 0; col < labelled_mask.cols; ++col) {
        int label = rowPtr[col];
        if (label != background_label) {
          unique_labels.insert(label);
        }
      }
    }
  }
  
  ObjectIds object_ids(unique_labels.begin(), unique_labels.end());
  std::sort(object_ids.begin(), object_ids.end());
  return object_ids;
}

std::ostream& operator<<(std::ostream& os,
                         const static_objects::ObjectDetectionResult& res) {
  os << "ObjectDetectionResult:\n";
  os << "  detections (" << res.detections.size() << "):\n";
  for (const auto& det : res.detections) {
    os << "    category_id=" << det.category_id
       << ", score=" << det.score
       << ", bbox=(" << det.bbox[0] << "," << det.bbox[1] << ","
       << det.bbox[2] << "," << det.bbox[3] << ")\n";
  }
  os << "  labelled_mask.empty()=" << res.labelled_mask.empty()
     << ", input_image.empty()=" << res.input_image.empty();
  return os;
}

}  // namespace static_objects
}  // namespace dyno