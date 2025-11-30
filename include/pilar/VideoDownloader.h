// Pilar - VideoDownloader
// Shell out to yt-dlp to download a video into a target path.

#pragma once

#include <string>
#include <ostream>

#include "pilar/Config.h"

namespace pilar {

class VideoDownloader {
public:
  // Downloads the video from `url` to `out_dir/cfg.output_filename`.
  // Returns the absolute file path on success, empty string on failure.
  std::string download(const std::string& url,
                       const std::string& out_dir,
                       const DownloadConfig& cfg,
                       std::ostream& log);

private:
  static std::string quote(const std::string& s);
};

} // namespace pilar

