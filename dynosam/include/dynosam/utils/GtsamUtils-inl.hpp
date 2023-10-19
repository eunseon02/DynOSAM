#pragma once

#include "dynosam/utils/GtsamUtils.hpp"

namespace dyno
{
namespace utils
{

template <typename T>
inline gtsam::Point2 cvPointToGtsam(const cv::Point_<T>& point)
{
  return gtsam::Point2(static_cast<double>(point.x), static_cast<double>(point.y));
}

template <typename T>
inline std::vector<gtsam::Point2> cvPointsToGtsam(const std::vector<cv::Point_<T>>& points)
{
  std::vector<gtsam::Point2> gtsam_points;
  for (const auto& p : points)
  {
    gtsam_points.push_back(cvPointToGtsam<T>(p));
  }
  return gtsam_points;
}

}  // namespace utils
}  // namespace dyno
