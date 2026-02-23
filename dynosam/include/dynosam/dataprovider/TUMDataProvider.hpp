#pragma once

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam_cv/CameraParams.hpp"
#include <string>
#include <vector>

namespace dyno {

// depth, optical flow, motion mask
using TUMProvider = DynoDatasetProvider<cv::Mat, cv::Mat>;

/**
 * @brief Data loader for TUM RGB-D datasets
 */
class TUMDataProvider : public TUMProvider {
 public:
  DYNO_POINTER_TYPEDEFS(TUMDataProvider)

  TUMDataProvider(const std::string& tum_path, const std::string& association_file,
                  const CameraParams& camera_params);

  virtual ~TUMDataProvider() = default;

  CameraParams::Optional getCameraParams() const override { return camera_params_; }

 private:
  CameraParams camera_params_;
};

}  // namespace dyno
