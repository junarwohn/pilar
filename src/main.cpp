#include <iostream>
#include <string>
#include <filesystem>

#include "pilar/Pipeline.h"

using namespace pilar;

static void print_usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " <youtube_url> <output_dir> [interval_seconds] [bottom_percent]\n";
  std::cerr << "Example: " << argv0 << " https://www.youtube.com/watch?v=dQw4w9WgXcQ out 1.0 0.20\n";
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string url = argv[1];
  const std::string output_dir = argv[2];

  PipelineConfig cfg;
  // Defaults already set in struct. Allow optional overrides
  if (argc >= 4) {
    try { cfg.frames.interval_seconds = std::stod(argv[3]); }
    catch (...) { std::cerr << "Invalid interval_seconds; using default." << std::endl; }
  }
  if (argc >= 5) {
    try { cfg.crop.bottom_percent = std::stod(argv[4]); }
    catch (...) { std::cerr << "Invalid bottom_percent; using default." << std::endl; }
  }

  // Download to output_dir/video.mp4
  cfg.download.output_filename = "video.mp4";

  std::cout << "Pilar starting...\n";
  Pipeline pipeline;
  const int rc = pipeline.run(url, output_dir, cfg, std::cout);
  if (rc != 0) {
    std::cerr << "Pilar failed with code: " << rc << "\n";
  } else {
    std::cout << "Pilar finished successfully." << std::endl;
  }
  return rc;
}

