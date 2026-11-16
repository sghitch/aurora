# Calibration module tests CMake configuration
add_executable(calibration_tests
    test_placeholder.cpp
)

target_link_libraries(calibration_tests
    PRIVATE
        aurora_calibration
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(calibration_tests)
