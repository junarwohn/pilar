cmake_minimum_required(VERSION 3.10)

# Set the project name and specify the C++ standard
project(pilar)

# Specify C++ standard (C++17 or later for filesystem support)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set gcc optimization flag
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Define the executable and the source file
add_executable(main main.cc)

# Link the OpenCV libraries to the executable
target_link_libraries(main ${OpenCV_LIBS})

# Link the filesystem library if needed (GCC version 9 or earlier)
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(main stdc++fs)
endif()

# Link additional libraries required for Unix/Linux systems (for sys/stat.h, dirent.h)
if(UNIX)
    target_link_libraries(main pthread)
endif()

