#pragma once

#include "dynosam/dataprovider/DataProviderModule.hpp"
#include "dynosam/common/Types.hpp"

namespace dyno {

struct RGBFrame;
struct MonoFrame;
struct StereoFrame;
struct ImuMeasurement;
struct MotionSegmentationFrame;
struct InstanceSegmentationFrame;


class DataProviderInterface {

public:
    DataProviderInterface();

};


} //dyno
