# Calibration module CMake configuration
add_library(aurora_calibration
    src/placeholder.cpp
)

target_include_directories(aurora_calibration
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aurora_calibration
    PUBLIC
        aurora_proto
)

add_subdirectory(test)
