# Documentation examples CMake configuration
#
# End-to-end examples exercising the simulation -> DSP -> detection and
# simulation -> DSP -> reconstruction pipelines.

add_executable(detection_example detection_example.cpp)
target_link_libraries(detection_example
    PRIVATE
        aurora_forward
        aurora_simulation
        aurora_dsp
        aurora_detection
        aurora_acquisition
        aurora_proto
        Eigen3::Eigen
)

add_executable(reconstruction_example reconstruction_example.cpp)
target_link_libraries(reconstruction_example
    PRIVATE
        aurora_forward
        aurora_simulation
        aurora_dsp
        aurora_reconstruction
        aurora_proto
        Eigen3::Eigen
)
