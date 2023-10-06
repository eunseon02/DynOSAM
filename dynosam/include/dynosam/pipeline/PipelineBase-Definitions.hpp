#pragma once

#include <functional>

namespace dyno {

enum class PipelineReturnCode {
    SUCCESS,
    IS_SHUTDOWN,
    OUTPUT_PUSH_FAILURE,
    PROCESSING_FAILURE,
    GET_PACKET_FAILURE
};

using OnPipelineFailureCallback = std::function<void(PipelineReturnCode)>;

}
