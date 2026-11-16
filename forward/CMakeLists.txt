# Forward module CMake configuration
add_library(aurora_forward
    src/bandwidth_model.cpp
    src/fft2d.cpp
    src/forward_model.cpp
    src/k_space_geometry.cpp
    src/k_space_slice_utils.cpp
    src/measurement_operator.cpp
)

target_include_directories(aurora_forward
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aurora_forward
    PUBLIC
        aurora_proto
        Eigen3::Eigen
    PRIVATE
        kissfft
)

add_subdirectory(test)

# Enable OpenMP for parallel FFT computation if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(aurora_forward PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(aurora_forward PRIVATE AURORA_USE_OPENMP)
endif()
