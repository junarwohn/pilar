// Pilar - SummaryGenerator
// For now, simply save every cropped frame as a JPEG file.

#pragma once

#include <cstdint>
#include <string>
#include <ostream>
#include <opencv2/core.hpp>

#include "pilar/Config.h"

namespace pilar {

class SummaryGenerator {
public:
  // Saves a single frame to `out_dir` with an incremental name.
  // Returns the written file path on success, empty string on failure.
  std::string saveFrame(const cv::Mat& frame,
                        const std::string& out_dir,
                        int64_t index,
                        const SummaryConfig& cfg,
                        std::ostream& log) const;
};

} // namespace pilar

