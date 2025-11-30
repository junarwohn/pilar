#include "pilar/FrameExtractor.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace pilar {

static inline int computeStepFrames(double fps, double interval_sec) {
  if (interval_sec <= 0.0) return 1;
  const double step = std::max(1.0, std::round(fps * interval_sec));
  return static_cast<int>(step);
}

int64_t FrameExtractor::extract(const std::string& video_path,
                                const FrameExtractionConfig& cfg,
                                const FrameCallback& on_frame,
                                std::ostream& log) {
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    log << "[Extractor] Failed to open video: " << video_path << "\n";
    return 0;
  }

  double fps = cap.get(cv::CAP_PROP_FPS);
  if (!(fps > 0.0)) {
    fps = 30.0; // Fallback FPS
    log << "[Extractor] FPS unknown; using fallback 30.0\n";
  }
  int step_frames = computeStepFrames(fps, cfg.interval_seconds);
  log << "[Extractor] FPS=" << fps << ", step=" << step_frames << " frames\n";

  int64_t emitted = 0;
  int64_t frame_index = 0;

  cv::Mat frame;
  while (true) {
    if (!cap.read(frame)) break; // EOF or read error

    if (frame_index % step_frames == 0) {
      cv::Mat processed = frame;
      if (cfg.resize_width > 0 && frame.cols > 0) {
        int new_w = cfg.resize_width;
        int new_h = static_cast<int>(std::round(
            frame.rows * (static_cast<double>(new_w) / frame.cols)));
        cv::resize(frame, processed, cv::Size(new_w, new_h));
      }
      on_frame(frame_index, processed);
      ++emitted;
    }
    ++frame_index;
  }

  log << "[Extractor] Emitted " << emitted << " frames\n";
  return emitted;
}

} // namespace pilar
