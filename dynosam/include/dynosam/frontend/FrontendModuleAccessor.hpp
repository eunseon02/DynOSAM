#pragma once

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include <memory>

namespace dyno {

// Global variable to store RGBDInstanceFrontendModule pointer for edge visualization
// This is a workaround to allow VoViewer to access frontend module's local map data
// Set by PipelineManager when frontend is created
extern std::weak_ptr<RGBDInstanceFrontendModule> g_frontend_module;

}  // namespace dyno
