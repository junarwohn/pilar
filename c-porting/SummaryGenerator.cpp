#include "pilar/SummaryGenerator.h"

#include <filesystem>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace pilar {

static std::string zeroPad(int64_t value, int width = 6) {
  std::ostringstream oss;
  oss << std::setw(width) << std::setfill('0') << value;
  return oss.str();
}

std::string SummaryGenerator::saveFrame(const cv::Mat& frame,
                                        const std::string& out_dir,
                                        int64_t index,
                                        const SummaryConfig& cfg,
                                        std::ostream& log) const {
  if (frame.empty()) return {};

  try {
    fs::create_directories(out_dir);
  } catch (...) {
    log << "[Summary] Failed to create directory: " << out_dir << "\n";
    return {};
  }

  const fs::path out_path = fs::path(out_dir) / ("frame_" + zeroPad(index) + ".jpg");
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::max(1, std::min(100, cfg.jpeg_quality))};
  const bool ok = cv::imwrite(out_path.string(), frame, params);
  if (!ok) {
    log << "[Summary] Failed to write: " << out_path << "\n";
    return {};
  }
  return out_path.string();
}

} // namespace pilar
