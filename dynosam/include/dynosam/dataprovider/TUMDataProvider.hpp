#pragma once

#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam_cv/CameraParams.hpp"
#include <string>
#include <vector>

namespace dyno {

class TUMDataProvider : public DataProvider {
 public:
  DYNO_POINTER_TYPEDEFS(TUMDataProvider)

  TUMDataProvider(const std::string& tum_path, const std::string& association_file,
                  const CameraParams& camera_params);

  virtual ~TUMDataProvider() = default;

  int datasetSize() const override { return static_cast<int>(rgb_files_.size()); }
  
  bool spin() override;

  CameraParams::Optional getCameraParams() const override { return camera_params_; }

  size_t size() const { return rgb_files_.size(); }

 private:
  void loadAssociationFile(const std::string& association_file);

  std::string tum_path_;
  CameraParams camera_params_;
  std::vector<std::string> rgb_files_;
  std::vector<std::string> depth_files_;
  std::vector<double> timestamps_;
  size_t current_frame_ = 0;
};

}  // namespace dyno
