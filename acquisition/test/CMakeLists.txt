# Acquisition module tests CMake configuration
add_executable(acquisition_tests
    test_sequence_planner.cpp
)

target_link_libraries(acquisition_tests
    PRIVATE
        aurora_acquisition
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(acquisition_tests)
