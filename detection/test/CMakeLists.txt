# Detection module tests CMake configuration
add_executable(detection_tests
    test_angle_estimator.cpp
    test_cfar_detector.cpp
    test_detection_protos.cpp
    test_detector.cpp
    test_doppler_extractor.cpp
    test_phase_gradient_estimator.cpp
)

target_include_directories(detection_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/test/common
)

target_link_libraries(detection_tests
    PRIVATE
        aurora_detection
        aurora_forward
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(detection_tests)
