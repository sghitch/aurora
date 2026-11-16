# Benchmarks CMake configuration

# Signal chain benchmarks
add_executable(signal_chain_benchmarks
    bench_signal_chain.cpp
)

target_include_directories(signal_chain_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/test/common
)

target_link_libraries(signal_chain_benchmarks
    PRIVATE
        aurora_proto
        aurora_simulation
        aurora_dsp
        aurora_forward
        benchmark::benchmark
        benchmark::benchmark_main
)

# Register benchmarks (optional: can run directly as executable)
# Benchmarks are not added to ctest by default since they're for performance analysis
