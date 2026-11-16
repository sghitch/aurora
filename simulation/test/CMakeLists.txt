# Simulation module tests CMake configuration
add_executable(simulation_tests
    test_adc_model.cpp
    test_chirp_generator.cpp
    test_fmcw_simulator.cpp
    test_mesh_integration.cpp
    test_mesh_proto.cpp
    test_mesh_target_view.cpp
    test_noise_model.cpp
    test_occlusion_bounding_sphere.cpp
    test_occlusion_raycast.cpp
    test_pupil_model.cpp
    test_scene.cpp
    test_speckle_field.cpp
    test_views.cpp
    test_visibility_checker.cpp
)

target_include_directories(simulation_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/test/common
)

target_link_libraries(simulation_tests
    PRIVATE
        aurora_simulation
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(simulation_tests)
