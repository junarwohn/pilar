// Pilar - Pipeline Orchestrator
// Wires downloader, extractor, cropper, and summary generator together.

#pragma once

#include <string>
#include <ostream>

#include "pilar/Config.h"

namespace pilar {

class Pipeline {
public:
  // Runs the pipeline: download -> extract -> crop -> save.
  // Returns 0 on success, non-zero on failure.
  int run(const std::string& url,
          const std::string& output_dir,
          const PipelineConfig& cfg,
          std::ostream& log);
};

} // namespace pilar

