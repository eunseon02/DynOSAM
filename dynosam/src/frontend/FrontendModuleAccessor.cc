#include "dynosam/frontend/FrontendModuleAccessor.hpp"
#include "dynosam/visualizer/VoViewer.hpp"

namespace dyno {

// Definition of g_frontend_module for VoViewer visualization
std::weak_ptr<RGBDInstanceFrontendModule> g_frontend_module;

// Definition of g_vo_viewer for pause state checking
::voViewer* g_vo_viewer = nullptr;

}  // namespace dyno
