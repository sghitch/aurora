# Integration tests CMake configuration
add_executable(integration_tests
    test_shared_protos.cpp
    test_signal_chain_integration.cpp
    test_detection_integration.cpp
)

target_include_directories(integration_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/test/common
)

target_link_libraries(integration_tests
    PRIVATE
        aurora_proto
        aurora_forward
        aurora_acquisition
        aurora_simulation
        aurora_dsp
        aurora_calibration
        aurora_detection
        aurora_reconstruction
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(integration_tests)
