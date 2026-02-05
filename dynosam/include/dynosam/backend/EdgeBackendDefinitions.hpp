/*
 * Edge Backend Module Definitions
 * Defines traits and types for Edge-based Backend Module
 */

#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendFormulationFactory.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam_opt/Map.hpp"

namespace dyno {

// EdgeBackendModule은 RegularBackendModuleTraits를 재사용하거나
// 새로운 Traits를 정의할 수 있습니다
// 여기서는 기존 MapType을 재사용하는 것으로 가정
using EdgeBackendModuleTraits =
    BackendModuleTraits<VisionImuPacket, CameraMeasurement>;

/// @brief BackendFormulationFactory templated on the edge map type
/// Edge mapping은 Formulation을 사용하지 않을 수도 있으므로
/// 필요에 따라 수정
using EdgeFormulationFactory =
    BackendFormulationFactory<EdgeBackendModuleTraits::MapType>;

}  // namespace dyno
