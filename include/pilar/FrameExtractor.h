// Pilar - FrameExtractor
// Extract frames at fixed intervals using OpenCV VideoCapture.

#pragma once

#include <functional>
#include <string>
#include <ostream>
#include <opencv2/core.hpp>

#include "pilar/Config.h"

namespace pilar {

class FrameExtractor {
public:
  using FrameCallback = std::function<void(int64_t index, const cv::Mat& frame)>;

  // Reads frames from `video_path` and calls `on_frame` for sampled frames.
  // Returns number of frames emitted.
  int64_t extract(const std::string& video_path,
                  const FrameExtractionConfig& cfg,
                  const FrameCallback& on_frame,
                  std::ostream& log);
};

} // namespace pilar

