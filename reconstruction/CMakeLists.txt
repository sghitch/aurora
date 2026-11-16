# Reconstruction module CMake configuration
add_library(aurora_reconstruction
    src/bandwidth_weight_calculator.cpp
    src/direct_operator.cpp
    src/fast_online_operator.cpp
    src/gridding_operator.cpp
    src/k_space_preprocessor.cpp
    src/nufft_operator.cpp
    src/pcg_operator.cpp
    src/pcg_precomputation.cpp
    src/pcg_precomputation_builder.cpp
    src/preconditioner.cpp
    src/reconstruction_operator.cpp
    src/reconstructor.cpp
    src/regularizer.cpp
    src/warm_start_cache.cpp
)

target_include_directories(aurora_reconstruction
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aurora_reconstruction
    PUBLIC
        aurora_proto
        aurora_forward
        Eigen3::Eigen
    PRIVATE
        kissfft
)

# Enable OpenMP for parallel reconstruction if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(aurora_reconstruction PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(aurora_reconstruction PRIVATE AURORA_USE_OPENMP)
endif()

add_subdirectory(test)
