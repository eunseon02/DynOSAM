#pragma once

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include <memory>

// Forward declaration (voViewer is in global namespace, not in dyno namespace)
class voViewer;

namespace dyno {

// Global variable to store RGBDInstanceFrontendModule pointer for edge visualization
// This is a workaround to allow VoViewer to access frontend module's local map data
// Set by PipelineManager when frontend is created
extern std::weak_ptr<RGBDInstanceFrontendModule> g_frontend_module;

// Global variable to store VoViewer pointer for pause state checking
// Set by dyno_sam.cc when VoViewer is created
extern ::voViewer* g_vo_viewer;

}  // namespace dyno
