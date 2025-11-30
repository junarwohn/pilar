// Pilar - SubtitleCropper
// Simple bottom-crop implementation for subtitle region.

#pragma once

#include <opencv2/core.hpp>

#include "pilar/Config.h"

namespace pilar {

class SubtitleCropper {
public:
  // Crops the bottom portion of `input` based on cfg.bottom_percent.
  // If cropping disabled or invalid, returns a clone of input.
  cv::Mat cropBottom(const cv::Mat& input, const CropConfig& cfg) const;
};

} // namespace pilar

