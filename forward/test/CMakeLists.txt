# Forward module tests CMake configuration
add_executable(forward_tests
    test_bandwidth_model.cpp
    test_proto_roundtrip.cpp
    test_forward_model.cpp
    test_k_space_geometry.cpp
    test_k_space_slice_utils.cpp
    test_measurement_operator.cpp
    test_physics_validation.cpp
)

target_include_directories(forward_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/test/common
)

target_link_libraries(forward_tests
    PRIVATE
        aurora_forward
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(forward_tests)
