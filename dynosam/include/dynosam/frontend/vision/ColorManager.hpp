/**
* This file is part of OA-SLAM.
*
* Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
* (Inria, LORIA, Université de Lorraine)
* OA-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OA-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OA-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include <random>
#include <iostream>

#include <opencv2/core.hpp>

namespace dyno {

// Simple uniform random color generator used for object visualization.
struct RandomUniformColorGenerator {
    static cv::Scalar Generate() {
        cv::Scalar color;
    color[0] = static_cast<double>(distr(gen));
    color[1] = static_cast<double>(distr(gen));
    color[2] = static_cast<double>(distr(gen));
        return color;
    }

  inline static std::random_device rd{};
  inline static std::mt19937 gen{rd()};
  inline static std::uniform_int_distribution<int> distr{0, 255};
};

inline constexpr int kNumCategoryColors = 500;

// Optional manager for per-category colors.
class CategoryColorsManager {
public:
    static const CategoryColorsManager& GetInstance() {
    if (!instance_) {
      instance_ = new CategoryColorsManager();
        }
    return *instance_;
    }

    static void FreeInstance() {
    delete instance_;
    instance_ = nullptr;
    }

  const cv::Scalar& operator[](size_t idx) const { return colors_[idx]; }

private:
  CategoryColorsManager()
      : colors_(std::vector<cv::Scalar>(kNumCategoryColors)) {
    for (int i = 0; i < kNumCategoryColors; ++i) {
      colors_[i] = RandomUniformColorGenerator::Generate();
    }
  }

  inline static CategoryColorsManager* instance_ = nullptr;
    std::vector<cv::Scalar> colors_;
};

}  // namespace dyno

