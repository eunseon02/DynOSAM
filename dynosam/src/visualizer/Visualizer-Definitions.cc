/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/visualizer/Visualizer-Definitions.hpp"

#include <glog/logging.h>

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/Flags.hpp"

DEFINE_bool(pause_image_display_each_frame, false,
            "Pause OpenCV display on every frame (waitKey(0)) for step-by-step inspection.");

namespace dyno {

OpenCVImageDisplayQueue::OpenCVImageDisplayQueue(
    ImageDisplayQueue* display_queue, bool parallel_run)
    : display_queue_(CHECK_NOTNULL(display_queue)),
      parallel_run_(parallel_run) {}

void OpenCVImageDisplayQueue::process() {
  static int obj_edge_next_bar = 0;   // set to 1 to advance one frame
  static bool obj_edge_bar_initialized = false;

  auto ensureObjEdgeControlBar = [&]() {
    if (obj_edge_bar_initialized) return;
    cv::createTrackbar(
        "Next", "Tracks Object Edge", nullptr, 1,
        [](int pos, void* userdata) {
          if (!userdata) return;
          *static_cast<int*>(userdata) = pos;
        },
        &obj_edge_next_bar);
    obj_edge_bar_initialized = true;
  };

  auto maybeStepOnObjEdge = [&]() {
    if (!obj_edge_bar_initialized) return;
    // Block until "Next" is pressed once.
    while (obj_edge_next_bar == 0) {
      cv::waitKey(30);
    }
    obj_edge_next_bar = 0;
    cv::setTrackbarPos("Next", "Tracks Object Edge", 0);
  };

  bool queue_state = false;
  if (parallel_run_) {
    ImageToDisplay image_to_display;
    queue_state = display_queue_->popBlockingWithTimeout(image_to_display, 2);

    if (queue_state) {
      // Keep only the latest "Tracks Object Edge" image to avoid stale-frame
      // visualization when stepping with Next.
      if (image_to_display.name_ == "Tracks Object Edge") {
        std::vector<ImageToDisplay> backlog;
        if (display_queue_->popAll(backlog, 0u)) {
          for (const auto& item : backlog) {
            if (item.name_ == "Tracks Object Edge") {
              image_to_display = item;
            }
          }
        }
      }

      // cv::namedWindow(image_to_display.name_);
      cv::imshow(image_to_display.name_, image_to_display.image_);
      if (image_to_display.name_ == "Tracks Object Edge") {
        ensureObjEdgeControlBar();
      }
      cv::waitKey(1);
      if (FLAGS_pause_image_display_each_frame) {
        cv::waitKey(0);
      } else if (image_to_display.name_ == "Tracks Object Edge") {
        maybeStepOnObjEdge();
      }
    }
  } else {
    std::vector<ImageToDisplay> images_to_display;
    queue_state = display_queue_->popAll(images_to_display, 2);

    if (queue_state) {
      bool obj_edge_shown = false;
      for (const auto& image_to_display : images_to_display) {
        cv::imshow(image_to_display.name_, image_to_display.image_);
        if (image_to_display.name_ == "Tracks Object Edge") {
          obj_edge_shown = true;
        }
      }
      if (obj_edge_shown) {
        ensureObjEdgeControlBar();
      }
      cv::waitKey(1);
      if (FLAGS_pause_image_display_each_frame) {
        cv::waitKey(0);
      } else if (obj_edge_shown) {
        maybeStepOnObjEdge();
      }
    }
  }
}

}  // namespace dyno
