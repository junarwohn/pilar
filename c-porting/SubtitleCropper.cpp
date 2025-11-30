#include "pilar/SubtitleCropper.h"

#include <algorithm>

namespace pilar {

cv::Mat SubtitleCropper::cropBottom(const cv::Mat& input, const CropConfig& cfg) const {
  if (!cfg.enable || input.empty()) {
    return input.clone();
  }
  const double pct = std::clamp(cfg.bottom_percent, 0.0, 1.0);
  if (pct <= 0.0) return input.clone();

  int rows = input.rows;
  int cols = input.cols;
  int start_row = static_cast<int>(rows * (1.0 - pct));
  start_row = std::clamp(start_row, 0, rows - 1);
  int height = rows - start_row;
  if (height <= 0) return input.clone();

  cv::Rect roi(0, start_row, cols, height);
  return input(roi).clone();
}

} // namespace pilar
