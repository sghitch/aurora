# Acquisition module CMake configuration
add_library(aurora_acquisition
    src/sequence_planner.cpp
)

target_include_directories(aurora_acquisition
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aurora_acquisition
    PUBLIC
        aurora_proto
)

add_subdirectory(test)
