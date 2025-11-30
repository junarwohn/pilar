// Pilar - Configuration structs
// English-only comments and modern C++17 style

#pragma once

#include <string>

namespace pilar {

struct DownloadConfig {
  // Downloader binary name (e.g., yt-dlp)
  std::string downloader = "yt-dlp";
  // Extra CLI arguments appended to the downloader command
  std::string extra_args;
  // Target filename (placed under the chosen directory)
  std::string output_filename = "video.mp4";
};

struct FrameExtractionConfig {
  // Fixed interval between frames in seconds (>= 0.01 recommended)
  double interval_seconds = 1.0;
  // Optional resize width for frames before downstream processing (0 = keep)
  int resize_width = 0;
};

struct CropConfig {
  // Crop bottom percentage of the frame (0.0 - 1.0). 0.20 means bottom 20%
  double bottom_percent = 0.20;
  bool enable = true;
};

struct SummaryConfig {
  // JPEG quality for saved frames (1-100)
  int jpeg_quality = 90;
};

struct PipelineConfig {
  DownloadConfig download;
  FrameExtractionConfig frames;
  CropConfig crop;
  SummaryConfig summary;
};

} // namespace pilar

