# Project Aurora - Top-level CMake configuration
cmake_minimum_required(VERSION 3.20)
project(aurora VERSION 0.1.0 LANGUAGES CXX)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Windows/MSVC: Enable M_PI and other math constants from <cmath>
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Protobuf and its dependencies
find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)
find_package(utf8_range REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# OpenMP for parallel simulation
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel simulation enabled")
else()
    message(STATUS "OpenMP not found - simulation will be single-threaded")
endif()

# Fetch dependencies
include(FetchContent)

# GoogleTest for unit testing
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# Prevent GoogleTest from overriding our compiler/linker options
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# KissFFT for FFT computations (BSD license)
FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG 131.1.0
)
# Disable KissFFT tools and tests (they require libpng and fftw)
set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
# Use double precision to match project convention
set(KISSFFT_DATATYPE "double" CACHE STRING "" FORCE)
# Build as static library
set(KISSFFT_STATIC ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(kissfft)

# Google Benchmark for performance testing (Apache 2.0)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
# Disable benchmark tests
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Enable testing early so module tests are registered
enable_testing()
include(GoogleTest)

# Collect all proto files
file(GLOB_RECURSE PROTO_FILES "${CMAKE_SOURCE_DIR}/proto/*.proto")

# Generate C++ sources from proto files (only if proto files exist)
if(PROTO_FILES)
    # Set import path for cross-module imports
    set(PROTOBUF_IMPORT_DIRS "${CMAKE_SOURCE_DIR}/proto")
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

    # Create aurora_proto library
    add_library(aurora_proto ${PROTO_SRCS})
    target_include_directories(aurora_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(aurora_proto PUBLIC
        protobuf::libprotobuf
        absl::log_internal_check_op
        absl::log_internal_conditions
        absl::log_internal_message
        absl::log_internal_log_sink_set
        absl::status
        absl::statusor
        absl::strings
        absl::raw_hash_set
        absl::hash
        absl::cord
        utf8_range::utf8_validity
    )
else()
    # Create empty interface library if no proto files yet
    add_library(aurora_proto INTERFACE)
    target_link_libraries(aurora_proto INTERFACE protobuf::libprotobuf)
endif()

# Add module subdirectories
add_subdirectory(shared)
add_subdirectory(forward)
add_subdirectory(acquisition)
add_subdirectory(simulation)
add_subdirectory(dsp)
add_subdirectory(calibration)
add_subdirectory(detection)
add_subdirectory(reconstruction)

# Integration and physics tests
add_subdirectory(test)

# Documentation examples
add_subdirectory(docs/examples)

# CLI tools
add_subdirectory(tools)

# Doxygen documentation generation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_PRIVATE NO)
    set(DOXYGEN_PROJECT_NAME "Aurora")
    set(DOXYGEN_PROJECT_BRIEF "Coherent LiDAR Simulation Framework")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_SOURCE_DIR}/README.md)

    doxygen_add_docs(docs
        ${CMAKE_SOURCE_DIR}/detection/include
        ${CMAKE_SOURCE_DIR}/reconstruction/include
        ${CMAKE_SOURCE_DIR}/forward/include
        ${CMAKE_SOURCE_DIR}/dsp/include
        ${CMAKE_SOURCE_DIR}/simulation/include
        ${CMAKE_SOURCE_DIR}/acquisition/include
        ${CMAKE_SOURCE_DIR}/calibration/include
        ${CMAKE_SOURCE_DIR}/README.md
        COMMENT "Generate API documentation"
    )
    message(STATUS "Doxygen found: run 'make docs' to generate documentation")
else()
    message(STATUS "Doxygen not found: documentation target will not be available")
endif()
