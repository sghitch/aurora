# Reconstruction module tests CMake configuration
add_executable(reconstruction_tests
    test_bandwidth_weight_calculator.cpp
    test_direct_operator.cpp
    test_fast_online_operator.cpp
    test_k_space_preprocessor.cpp
    test_nufft_operator.cpp
    test_pcg_integration.cpp
    test_pcg_operator.cpp
    test_pcg_precomputation.cpp
    test_preconditioner.cpp
    test_reconstructor.cpp
    test_regularizer.cpp
    test_warm_start_cache.cpp
)

target_include_directories(reconstruction_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/test/common
)

target_link_libraries(reconstruction_tests
    PRIVATE
        aurora_reconstruction
        aurora_forward
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(reconstruction_tests)
