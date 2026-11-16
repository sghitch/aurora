# DSP module CMake configuration
add_library(aurora_dsp
    src/iq_demodulator.cpp
    src/fft_wrapper.cpp
    src/window_functions.cpp
    src/range_doppler_processor.cpp
    src/measurement_builder.cpp
    src/peak_doppler_processor.cpp
)

target_include_directories(aurora_dsp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(aurora_dsp
    PUBLIC
        aurora_proto
        aurora_forward
    PRIVATE
        kissfft::kissfft
)

add_subdirectory(test)
