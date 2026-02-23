#pragma once

#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "dynosam_common/Utils.hpp"  // For BBox2
#include "dynosam_common/Ellipse.hpp"  // For Ellipse
#include "dynosam_common/Types.hpp"  // For ObjectIds

//TODO : 
namespace dyno {
namespace static_objects {

/**
 * @brief The result of an object detection from a detection network
 *
 */
struct Detection
{
    typedef std::shared_ptr<Detection> Ptr;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Detection(unsigned int cat, double det_score, const BBox2 bb)
    : category_id(cat), score(det_score), bbox(bb), ell(Ellipse()) {}

    Detection(unsigned int cat, double det_score, const BBox2 bb, Ellipse ell_data)
    : category_id(cat), score(det_score), bbox(bb), ell(ell_data) {}


    friend std::ostream& operator <<(std::ostream& os, const Detection& det) {
        os << "Detection:  cat = " << det.category_id << "  score = "
           << det.score << "  bbox = " << det.bbox.transpose();
        return os;
    }
    unsigned int category_id;
    double score;
    BBox2 bbox;
    Ellipse ell;

private:
    Detection() = delete;
};


  /**
   * @brief Holds object detection/tracking result
   *
   */
  struct ObjectDetectionResult {
    std::vector<Detection> detections;
    cv::Mat labelled_mask;
    cv::Mat input_image;  // Should be a 3 channel RGB image. Should always be set
  
    cv::Mat colouredMask() const;
    //! number of detections
    inline size_t num() const { return detections.size(); }
    ObjectIds objectIds() const;
  
    friend std::ostream& operator<<(std::ostream& os,
                                    const static_objects::ObjectDetectionResult& res);
  };
  
}  // namespace static_objects
}  // namespace dyno