#pragma once

#include <opencv2/core/cvdef.h>

// OpenCV CUDA support detection
// For OpenCV 3.x: Try to include cvconfig.h (may not be available in all builds)
// For OpenCV 4.x: cvconfig.h is not available externally, CMake detects CUDA and defines DYNO_CUDA_OPENCV_ENABLED
// 
// This header tries to be compatible with both:
// 1. OpenCV 3.x with cvconfig.h
// 2. OpenCV 4.x where CMake sets DYNO_CUDA_OPENCV_ENABLED

// Try cvconfig.h for OpenCV 3.x compatibility (only if not already defined by CMake)
#ifndef DYNO_CUDA_OPENCV_ENABLED
  #if defined(__OPENCV_BUILD)
    // Internal OpenCV build - cvconfig.h should be available
    #include <opencv2/cvconfig.h>
    #ifdef HAVE_CUDA
      #define DYNO_CUDA_OPENCV_ENABLED
    #endif
  #elif defined(__has_include) && __has_include(<opencv2/cvconfig.h>)
    // External build but cvconfig.h exists (OpenCV 3.x or custom build)
    #include <opencv2/cvconfig.h>
#ifdef HAVE_CUDA
#define DYNO_CUDA_OPENCV_ENABLED
#endif
  #endif
#endif
// For OpenCV 4.x external builds, DYNO_CUDA_OPENCV_ENABLED should be set by CMake

constexpr bool isOpencvCudaEnabled() {
#ifdef DYNO_CUDA_OPENCV_ENABLED
  return true;
#else
  return false;
#endif
}
