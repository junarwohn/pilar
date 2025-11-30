#include "pilar/VideoDownloader.h"

#include <cstdlib>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace pilar {

std::string VideoDownloader::quote(const std::string& s) {
  std::ostringstream oss;
  oss << '"';
  for (char c : s) {
    if (c == '"' || c == '\\') oss << '\\';
    oss << c;
  }
  oss << '"';
  return oss.str();
}

std::string VideoDownloader::download(const std::string& url,
                                      const std::string& out_dir,
                                      const DownloadConfig& cfg,
                                      std::ostream& log) {
  try {
    fs::create_directories(out_dir);
  } catch (...) {
    log << "[Downloader] Failed to create directory: " << out_dir << "\n";
    return {};
  }

  const fs::path out_path = fs::path(out_dir) / cfg.output_filename;

  // Merge extra args from config and environment
  std::string extra = cfg.extra_args;
  if (const char* env = std::getenv("PILAR_YTDLP_ARGS")) {
    if (env[0] != '\0') {
      if (!extra.empty()) extra += ' ';
      extra += env;
    }
  }

  auto make_cmd = [&](const std::string& extra_args) {
    return cfg.downloader + " " + extra_args +
           " -o " + quote(out_path.string()) +
           " -f mp4 " + quote(url);
  };

  std::string cmd = make_cmd(extra);
  log << "[Downloader] Running: " << cmd << "\n";
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    log << "[Downloader] Downloader returned non-zero: " << ret << "\n";
    // Fallback: clear yt-dlp cache to mitigate nsig failures
    if (extra.find("--rm-cache-dir") == std::string::npos) {
      std::string fallback_extra = extra;
      if (!fallback_extra.empty()) fallback_extra += ' ';
      fallback_extra += "--rm-cache-dir";
      cmd = make_cmd(fallback_extra);
      log << "[Downloader] Retrying with cache clear: " << cmd << "\n";
      ret = std::system(cmd.c_str());
      if (ret != 0) {
        log << "[Downloader] Retry failed, code: " << ret << "\n";
        return {};
      }
    } else {
      return {};
    }
  }

  if (!fs::exists(out_path)) {
    log << "[Downloader] Expected file not found: " << out_path << "\n";
    return {};
  }

  log << "[Downloader] Downloaded to: " << out_path << "\n";
  return fs::absolute(out_path).string();
}

} // namespace pilar
