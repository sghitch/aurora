# Aurora CLI Tools
# Build targets for command-line utilities

# PCG Precomputation CLI Tool
add_executable(aurora-pcg-precompute
    pcg_precompute.cpp
)

target_link_libraries(aurora-pcg-precompute
    aurora_forward
    aurora_reconstruction
    aurora_proto
    Eigen3::Eigen
)

# Install target
install(TARGETS aurora-pcg-precompute
    RUNTIME DESTINATION bin
)

# Add test subdirectory
add_subdirectory(test)
