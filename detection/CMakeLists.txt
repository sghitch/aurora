# Detection module CMake configuration
add_library(aurora_detection
    src/angle_estimator.cpp
    src/cfar_detector.cpp
    src/detector.cpp
    src/doppler_extractor.cpp
    src/phase_gradient_estimator.cpp
)

target_include_directories(aurora_detection
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aurora_detection
    PUBLIC
        aurora_proto
        aurora_forward
        Eigen3::Eigen
)

add_subdirectory(test)
