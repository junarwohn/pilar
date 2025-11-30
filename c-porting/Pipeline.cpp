#include "pilar/Pipeline.h"

#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "pilar/VideoDownloader.h"
#include "pilar/FrameExtractor.h"
#include "pilar/SubtitleCropper.h"
#include "pilar/SummaryGenerator.h"

namespace fs = std::filesystem;

namespace pilar {

int Pipeline::run(const std::string& url,
                  const std::string& output_dir,
                  const PipelineConfig& cfg,
                  std::ostream& log) {
  // 1) Download
  VideoDownloader downloader;
  const std::string video_path = downloader.download(url, output_dir, cfg.download, log);
  if (video_path.empty()) {
    log << "[Pipeline] Download step failed.\n";
    return 1;
  }

  // Prepare subdirectory for frames
  const fs::path frames_dir = fs::path(output_dir) / "frames";

  // 2) Extract frames, 3) Crop, 4) Save
  FrameExtractor extractor;
  SubtitleCropper cropper;
  SummaryGenerator generator;

  int64_t saved = 0;
  auto on_frame = [&](int64_t index, const cv::Mat& frame) {
    cv::Mat cropped = cropper.cropBottom(frame, cfg.crop);
    if (cropped.empty()) return;
    const std::string out = generator.saveFrame(cropped, frames_dir.string(), index, cfg.summary, log);
    if (!out.empty()) ++saved;
  };

  const int64_t emitted = extractor.extract(video_path, cfg.frames, on_frame, log);
  log << "[Pipeline] Emitted=" << emitted << ", Saved=" << saved << "\n";
  if (emitted <= 0 || saved <= 0) {
    log << "[Pipeline] No frames produced.\n";
  }

  return 0;
}

} // namespace pilar
