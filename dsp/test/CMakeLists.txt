# DSP module tests CMake configuration
add_executable(dsp_tests
    test_iq_demodulator.cpp
    test_fft_wrapper.cpp
    test_window_functions.cpp
    test_range_doppler_processor.cpp
    test_measurement_builder.cpp
    test_peak_doppler_processor.cpp
)

target_include_directories(dsp_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/test/common
)

target_link_libraries(dsp_tests
    PRIVATE
        aurora_dsp
        GTest::gtest_main
)

include(GoogleTest)
# Use add_test instead of gtest_discover_tests to run tests as a single process.
# This allows SetUpTestSuite() to work as intended, avoiding repeated
# ForwardModel initialization (once for the entire suite instead of per test).
# Use generator expression for multi-config generators (Visual Studio).
add_test(NAME dsp_tests COMMAND $<TARGET_FILE:dsp_tests>)
