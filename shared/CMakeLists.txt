# Shared utilities module CMake configuration
# Header-only library for common utilities

add_library(aurora_shared INTERFACE)

target_include_directories(aurora_shared
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aurora_shared
    INTERFACE
        aurora_proto
        Eigen3::Eigen
)
