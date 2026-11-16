# Aurora CLI Tools Tests

add_executable(pcg_precompute_test
    pcg_precompute_test.cpp
)

target_link_libraries(pcg_precompute_test
    aurora_forward
    aurora_reconstruction
    aurora_proto
    Eigen3::Eigen
    GTest::gtest_main
)

# Register tests with CTest
gtest_discover_tests(pcg_precompute_test)
