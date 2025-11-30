cmake_minimum_required(VERSION 3.12)

project(Pilar VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable more warnings by default
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Try to find OpenCV via CMake config; fall back to pkg-config (opencv4)
find_package(OpenCV QUIET)
if (NOT OpenCV_FOUND)
  find_package(PkgConfig QUIET)
  if (PkgConfig_FOUND)
    pkg_check_modules(OPENCV4 QUIET opencv4)
  endif()
  if (OPENCV4_FOUND)
    message(STATUS "Found OpenCV via pkg-config: ${OPENCV4_VERSION}")
    set(OpenCV_INCLUDE_DIRS ${OPENCV4_INCLUDE_DIRS})
    set(OpenCV_LIBS ${OPENCV4_LIBRARIES})
    # Some flags from pkg-config (e.g., -DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)
    add_compile_options(${OPENCV4_CFLAGS_OTHER})
    link_directories(${OPENCV4_LIBRARY_DIRS})
  else()
    message(FATAL_ERROR "OpenCV not found. Install OpenCV or set OpenCV_DIR (e.g., /usr/lib/cmake/opencv4)")
  endif()
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(pilar
  src/main.cpp
  src/VideoDownloader.cpp
  c-porting/FrameExtractor.cpp
  c-porting/SubtitleCropper.cpp
  c-porting/SummaryGenerator.cpp
  c-porting/Pipeline.cpp
)

target_link_libraries(pilar
  ${OpenCV_LIBS}
)

install(TARGETS pilar RUNTIME DESTINATION bin)
