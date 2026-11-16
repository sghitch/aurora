# Simulation module CMake configuration
add_library(aurora_simulation
    src/adc_model.cpp
    src/chirp_generator.cpp
    src/fmcw_simulator.cpp
    src/noise_model.cpp
    src/pupil_model.cpp
    src/scene.cpp
    src/speckle_field.cpp
    src/views.cpp
    src/visibility_checker.cpp
)

target_include_directories(aurora_simulation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aurora_simulation
    PUBLIC
        aurora_forward
        aurora_proto
        aurora_shared
        Eigen3::Eigen
)

# Enable OpenMP for parallel simulation if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(aurora_simulation PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(aurora_simulation PRIVATE AURORA_USE_OPENMP)
endif()

add_subdirectory(test)
