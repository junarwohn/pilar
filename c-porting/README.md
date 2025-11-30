This folder hosts the modules targeted for C porting.

Scope currently includes core processing logic built by C++:
- Frame extraction (`FrameExtractor.cpp`)
- Subtitle region crop (`SubtitleCropper.cpp`)
- Frame saving (`SummaryGenerator.cpp`)
- Orchestration glue (`Pipeline.cpp`)

The public C++ interfaces remain under `include/pilar/`. Build system links these sources directly so you can iterate here without touching `src/`.

