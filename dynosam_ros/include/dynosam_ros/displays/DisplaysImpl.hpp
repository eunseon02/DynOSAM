/*
 *   Copyright (c) 2025
 *   All rights reserved.
 */

#pragma once

#ifdef USE_DYNAMIC_SLAM_INTERFACES

#else
#include "dynosam_ros/displays/inbuilt_displays/BackendInbuiltDisplayRos.hpp"
#include "dynosam_ros/displays/inbuilt_displays/FrontendInbuiltDisplayRos.hpp"

namespace dyno {
typedef FrontendInbuiltDisplayRos FrontendDisplayRos;
typedef BackendInbuiltDisplayRos BackendDisplayRos;
}  // namespace dyno

#endif
