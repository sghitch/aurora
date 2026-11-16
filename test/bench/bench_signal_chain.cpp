// Signal chain performance benchmarks
//
// Performance targets:
//
// Simulation Benchmarks:
//   - bench_chirp_generation:    < 100 us
//   - bench_target_response:     < 50 us
//   - bench_noise_injection:     < 20 us
//   - bench_adc_model:           < 10 us
//   - bench_burst_simulation:    < 500 us
//
// DSP Benchmarks:
//   - bench_iq_demodulation:     < 30 us
//   - bench_range_fft:           < 50 us
//   - bench_window_apply:        < 5 us
//   - bench_chirp_processing:    < 100 us
//   - bench_doppler_extraction:  < 20 us
//
// End-to-End Benchmarks:
//   - bench_single_burst:        < 1 ms
//   - bench_multi_chirp_burst:   < 20 ms (100 chirps)
//   - bench_burst_processing:    < 10 ms
//
// Memory Benchmarks:
//   - bench_memory_burst:        < 100 MB
//   - bench_memory_tensor:       < 50 MB
//   - bench_memory_steady_state: < 500 MB

#include <benchmark/benchmark.h>

#include <complex>
#include <memory>
#include <random>
#include <vector>

#include "dsp/fft_wrapper.h"
#include "dsp/iq_demodulator.h"
#include "dsp/measurement_builder.h"
#include "dsp/range_doppler_processor.h"
#include "dsp/window_functions.h"
#include "forward/forward_model.h"
#include "test_forward_model_builder.h"
#include "simulation/adc_model.h"
#include "simulation/chirp_generator.h"
#include "simulation/fmcw_simulator.h"
#include "simulation/noise_model.h"

#include "burst.pb.h"
#include "calibration_artifact.pb.h"
#include "detector_geometry.pb.h"
#include "dsp_config.pb.h"
#include "forward_config.pb.h"
#include "sequence_plan.pb.h"
#include "sim_config.pb.h"

using namespace aurora::simulation;
using namespace aurora::dsp;

namespace {

// =============================================================================
// Test parameters matching typical system configuration
// =============================================================================
constexpr double kSampleRateHz = 125e6;        // 125 MHz ADC
constexpr double kChirpBandwidthHz = 200e6;    // 200 MHz bandwidth
constexpr double kChirpDurationS = 12.5e-6;    // 12.5 us chirp
constexpr double kWavelength1550nm = 1550e-9;  // Primary wavelength
constexpr double kWavelength905nm = 905e-9;    // Secondary wavelength
constexpr uint32_t kNumDetectors = 8;          // 8 detector pixels
constexpr uint32_t kFftSize = 2048;            // FFT size for range processing
constexpr uint32_t kNumSamples = 1563;         // Samples per chirp at 125 MHz

// =============================================================================
// Configuration helpers
// =============================================================================

SimConfig create_benchmark_sim_config() {
    SimConfig config;

    auto* adc = config.mutable_adc();
    adc->set_sample_rate_hz(kSampleRateHz);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    auto* noise = config.mutable_noise();
    noise->set_enable_shot_noise(true);
    noise->set_enable_thermal_noise(true);
    noise->set_enable_rin_noise(false);
    noise->set_enable_quantization_noise(true);
    noise->set_enable_dark_current(false);
    noise->set_detector_responsivity_a_per_w(1.0);
    noise->set_transimpedance_ohm(50.0);
    noise->set_temperature_k(300.0);
    noise->set_noise_bandwidth_hz(150e6);

    config.set_tx_power_w(1.0);
    config.set_lo_power_w(1.0);
    config.set_system_efficiency(0.5);
    config.set_rng_seed(42);

    // Occlusion mode (must be explicitly set)
    config.set_occlusion_mode(OCCLUSION_NONE);

    // MeasurementOperator settings (must be explicitly set)
    config.mutable_bandwidth_config()->set_quadrature_order(11);  // 11×11 midpoint quadrature
    config.mutable_bandwidth_config()->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    config.set_specular_threshold(1.0 - 1e-10);   // Treat as specular if >= this

    // Add a point target for simulation
    auto* scene = config.mutable_scene();
    auto* target = scene->add_point_targets();
    target->mutable_position_m()->set_x(0.0);
    target->mutable_position_m()->set_y(0.0);
    target->mutable_position_m()->set_z(50.0);
    target->mutable_surface()->set_reflectivity_905nm(0.5);
    target->mutable_surface()->set_reflectivity_1550nm(0.5);
    // Surface statistics are validated at scene construction (correlation
    // length must be positive); use the same values as the integration tests.
    target->mutable_surface()->set_roughness_rms_m(1e-6);
    target->mutable_surface()->set_correlation_length_m(10e-6);

    return config;
}

aurora::forward::ForwardConfig create_benchmark_forward_config() {
    aurora::forward::ForwardConfig config;

    auto* detector = config.mutable_detector();
    auto* array = detector->add_arrays();
    array->set_array_id("main");
    array->set_nominal_pitch_m(0.5e-3);
    array->set_bandwidth_hz(150e6);
    array->add_wavelengths_m(kWavelength905nm);
    array->add_wavelengths_m(kWavelength1550nm);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_POINT);

    for (uint32_t i = 0; i < kNumDetectors; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((static_cast<double>(i) - 3.5) * 0.5e-3);
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.4e-3);
        pixel->set_height_m(0.4e-3);
    }

    auto* optics = config.mutable_optics();
    optics->set_fourier_lens_focal_m(0.2);
    optics->set_fourier_lens_diameter_m(0.05);
    optics->add_wavelengths_m(kWavelength905nm);
    optics->add_wavelengths_m(kWavelength1550nm);

    auto* pupil = optics->mutable_pupil();
    pupil->set_fov_half_angle_rad(0.5);
    pupil->set_rolloff_width_rad(0.05);

    return config;
}

DspConfig create_benchmark_dsp_config() {
    DspConfig config;
    config.set_range_fft_size(kFftSize);
    config.mutable_range_window()->set_type(WindowConfig::WINDOW_HANN);
    config.set_enable_dc_removal(true);

    auto* adc = config.mutable_adc_config();
    adc->set_sample_rate_hz(kSampleRateHz);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    auto* iq = config.mutable_iq_demod_config();
    iq->set_lpf_cutoff_hz(10e6);
    iq->set_lpf_order(4);

    return config;
}

aurora::shared::CalibrationArtifact create_benchmark_calibration() {
    aurora::shared::CalibrationArtifact cal;
    cal.set_calibration_id("benchmark");
    cal.set_timestamp_s(0.0);

    for (uint32_t i = 0; i < kNumDetectors; ++i) {
        auto* det_cal = cal.add_detector_calibrations();
        det_cal->set_detector_index(i);
        det_cal->mutable_complex_gain()->set_real(1.0);
        det_cal->mutable_complex_gain()->set_imag(0.0);
        det_cal->set_timing_skew_s(0.0);
    }

    for (double wl : {kWavelength905nm, kWavelength1550nm}) {
        auto* k_cal = cal.add_k_space_calibrations();
        k_cal->set_wavelength_m(wl);
        k_cal->set_wavelength_scale(1.0);
        for (uint32_t i = 0; i < kNumDetectors; ++i) {
            k_cal->add_kx_offsets(0.0);
            k_cal->add_ky_offsets(0.0);
        }
    }

    return cal;
}

// Helper to build ForwardModel from ForwardConfig
aurora::forward::ForwardModel build_benchmark_forward_model(
    const aurora::forward::ForwardConfig& config) {
    return aurora::test::TestForwardModelBuilder::build(config);
}

// Cached ForwardModel for benchmarks (avoid rebuilding for each benchmark)
aurora::forward::ForwardModel& get_benchmark_forward_model() {
    static auto forward_config = create_benchmark_forward_config();
    static auto forward_model = build_benchmark_forward_model(forward_config);
    return forward_model;
}

aurora::acquisition::ChirpPlan create_chirp_plan(
    uint32_t index,
    double wavelength_m,
    aurora::acquisition::ChirpPlan::Direction direction =
        aurora::acquisition::ChirpPlan::DIRECTION_UP) {

    aurora::acquisition::ChirpPlan plan;
    plan.set_chirp_index(index);
    plan.set_chirp_start_s(index * kChirpDurationS);
    plan.set_chirp_duration_s(kChirpDurationS);
    plan.set_chirp_start_hz(0.0);
    plan.set_chirp_bandwidth_hz(kChirpBandwidthHz);
    plan.set_direction(direction);
    plan.add_wavelength_m(wavelength_m);
    return plan;
}

aurora::acquisition::SequencePlan create_single_chirp_plan() {
    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(1);
    plan.set_start_timestamp_s(0.0);

    *plan.add_chirps() = create_chirp_plan(0, kWavelength1550nm);

    return plan;
}

aurora::acquisition::SequencePlan create_multi_chirp_plan(int num_chirps) {
    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(1);
    plan.set_start_timestamp_s(0.0);

    for (int c = 0; c < num_chirps; ++c) {
        *plan.add_chirps() = create_chirp_plan(
            static_cast<uint32_t>(c), kWavelength1550nm);
    }

    return plan;
}

// =============================================================================
// SIMULATION BENCHMARKS
// =============================================================================

// Benchmark: Chirp waveform generation (< 100 us)
static void BM_ChirpGeneration(benchmark::State& state) {
    aurora::acquisition::ChirpPlan chirp_plan;
    chirp_plan.set_chirp_duration_s(kChirpDurationS);
    chirp_plan.set_chirp_start_hz(0.0);
    chirp_plan.set_chirp_bandwidth_hz(kChirpBandwidthHz);
    chirp_plan.set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);

    for (auto _ : state) {
        auto waveform = ChirpGenerator::generate_waveform(chirp_plan, kSampleRateHz);
        benchmark::DoNotOptimize(waveform);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("samples: " + std::to_string(kNumSamples));
}
BENCHMARK(BM_ChirpGeneration)->Unit(benchmark::kMicrosecond);

// Benchmark: Phase generation (separate from waveform)
static void BM_ChirpPhaseGeneration(benchmark::State& state) {
    aurora::acquisition::ChirpPlan chirp_plan;
    chirp_plan.set_chirp_duration_s(kChirpDurationS);
    chirp_plan.set_chirp_start_hz(0.0);
    chirp_plan.set_chirp_bandwidth_hz(kChirpBandwidthHz);
    chirp_plan.set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);

    for (auto _ : state) {
        auto phase = ChirpGenerator::generate_phase(chirp_plan, kSampleRateHz);
        benchmark::DoNotOptimize(phase);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ChirpPhaseGeneration)->Unit(benchmark::kMicrosecond);

// Benchmark: Noise injection (< 20 us)
static void BM_NoiseInjection(benchmark::State& state) {
    NoiseModelConfig noise_config;
    noise_config.set_enable_shot_noise(true);
    noise_config.set_enable_thermal_noise(true);
    noise_config.set_enable_quantization_noise(true);
    noise_config.set_detector_responsivity_a_per_w(1.0);
    noise_config.set_transimpedance_ohm(50.0);
    noise_config.set_temperature_k(300.0);
    noise_config.set_noise_bandwidth_hz(150e6);

    aurora::shared::AdcModelConfig adc_config;
    adc_config.set_sample_rate_hz(kSampleRateHz);
    adc_config.set_bits(14);
    adc_config.set_input_range_v(2.0);

    NoiseModel noise_model(noise_config, adc_config);
    std::mt19937_64 rng(42);

    std::vector<double> signal(kNumSamples, 0.1);
    double optical_power = 1e-6;
    double bandwidth = 150e6;

    for (auto _ : state) {
        auto noisy = noise_model.add_noise(signal, optical_power, bandwidth, rng);
        benchmark::DoNotOptimize(noisy);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NoiseInjection)->Unit(benchmark::kMicrosecond);

// Benchmark: ADC digitization (< 10 us)
static void BM_AdcModel(benchmark::State& state) {
    aurora::shared::AdcModelConfig adc_config;
    adc_config.set_sample_rate_hz(kSampleRateHz);
    adc_config.set_bits(14);
    adc_config.set_input_range_v(2.0);

    AdcModel adc(adc_config);
    std::mt19937_64 rng(42);

    std::vector<double> signal(kNumSamples);
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] = 0.5 * std::sin(2.0 * 3.14159 * 1e6 * i / kSampleRateHz);
    }

    for (auto _ : state) {
        auto digitized = adc.sample(signal, kSampleRateHz, rng);
        benchmark::DoNotOptimize(digitized);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_AdcModel)->Unit(benchmark::kMicrosecond);

// Benchmark: Full burst simulation (< 500 us)
static void BM_BurstSimulation(benchmark::State& state) {
    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);

    auto plan = create_single_chirp_plan();

    for (auto _ : state) {
        auto burst = simulator.simulate_burst(plan);
        benchmark::DoNotOptimize(burst);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("8 detectors x 1 chirp");
}
BENCHMARK(BM_BurstSimulation)->Unit(benchmark::kMicrosecond);

// =============================================================================
// DSP BENCHMARKS
// =============================================================================

// Benchmark: IQ demodulation (< 30 us)
static void BM_IQDemodulation(benchmark::State& state) {
    IQDemodulator::Config iq_config;
    iq_config.sample_rate_hz = kSampleRateHz;
    iq_config.reference_freq_hz = 1e6;
    iq_config.lpf_cutoff_hz = 10e6;
    iq_config.lpf_order = 4;

    IQDemodulator demod(iq_config);

    // Generate test signal: beat frequency at 1 MHz
    std::vector<double> signal(kNumSamples);
    double beat_freq = 1e6;
    for (size_t i = 0; i < signal.size(); ++i) {
        double t = i / kSampleRateHz;
        signal[i] = std::cos(2.0 * 3.14159 * beat_freq * t);
    }

    for (auto _ : state) {
        auto iq = demod.demodulate(signal, beat_freq);
        benchmark::DoNotOptimize(iq);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_IQDemodulation)->Unit(benchmark::kMicrosecond);

// Benchmark: Range FFT (< 50 us for 2048 points)
static void BM_RangeFFT(benchmark::State& state) {
    const int fft_size = state.range(0);
    FftWrapper fft(fft_size);

    std::vector<std::complex<double>> input(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        input[i] = std::complex<double>(
            std::cos(2.0 * 3.14159 * 10 * i / fft_size),
            std::sin(2.0 * 3.14159 * 10 * i / fft_size));
    }

    for (auto _ : state) {
        auto output = fft.execute(input);
        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("FFT size: " + std::to_string(fft_size));
}
BENCHMARK(BM_RangeFFT)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

// Benchmark: Real-to-complex FFT (more efficient for real signals)
static void BM_RealFFT(benchmark::State& state) {
    const int fft_size = state.range(0);
    FftWrapper fft(fft_size);

    std::vector<double> input(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        input[i] = std::cos(2.0 * 3.14159 * 10 * i / fft_size);
    }

    for (auto _ : state) {
        auto output = fft.execute_real(input);
        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_RealFFT)
    ->Arg(2048)
    ->Unit(benchmark::kMicrosecond);

// Benchmark: Window function application (< 5 us)
static void BM_WindowApply(benchmark::State& state) {
    WindowConfig config;
    config.set_type(WindowConfig::WINDOW_HANN);

    std::vector<std::complex<double>> signal(kNumSamples);
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] = std::complex<double>(1.0, 0.0);
    }

    for (auto _ : state) {
        auto windowed = signal;
        WindowFunctions::apply_inplace(windowed, config);
        benchmark::DoNotOptimize(windowed);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WindowApply)->Unit(benchmark::kMicrosecond);

// Benchmark: Window generation for different types
static void BM_WindowGeneration(benchmark::State& state) {
    WindowConfig config;
    config.set_type(static_cast<WindowConfig::WindowType>(state.range(0)));

    for (auto _ : state) {
        auto window = WindowFunctions::generate(config, kFftSize);
        benchmark::DoNotOptimize(window);
    }

    const char* names[] = {"RECT", "HANN", "HAMMING", "BLACKMAN", "KAISER"};
    state.SetLabel(names[state.range(0)]);
}
BENCHMARK(BM_WindowGeneration)
    ->Arg(WindowConfig::WINDOW_RECTANGULAR)
    ->Arg(WindowConfig::WINDOW_HANN)
    ->Arg(WindowConfig::WINDOW_HAMMING)
    ->Arg(WindowConfig::WINDOW_BLACKMAN)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// END-TO-END BENCHMARKS
// =============================================================================

// Benchmark: Single burst (simulate + DSP) (< 1 ms)
static void BM_SingleBurst(benchmark::State& state) {
    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto dsp_config = create_benchmark_dsp_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);
    RangeDopplerProcessor processor(dsp_config, get_benchmark_forward_model());

    auto plan = create_single_chirp_plan();

    for (auto _ : state) {
        auto burst = simulator.simulate_burst(plan);
        auto range_cube = processor.process_range(burst);
        benchmark::DoNotOptimize(range_cube);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SingleBurst)->Unit(benchmark::kMillisecond);

// Benchmark: Multi-chirp burst simulation + range processing (< 20 ms)
static void BM_MultiChirpBurst(benchmark::State& state) {
    const int num_chirps = state.range(0);

    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto dsp_config = create_benchmark_dsp_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);
    RangeDopplerProcessor processor(dsp_config, get_benchmark_forward_model());

    auto plan = create_multi_chirp_plan(num_chirps);

    for (auto _ : state) {
        auto burst = simulator.simulate_burst(plan);
        auto range_cube = processor.process_range(burst);
        benchmark::DoNotOptimize(range_cube);
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel(std::to_string(num_chirps) + " chirps");
}
BENCHMARK(BM_MultiChirpBurst)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Unit(benchmark::kMillisecond);

// Benchmark: Burst processing to MeasurementTensor (< 10 ms)
static void BM_BurstProcessing(benchmark::State& state) {
    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto dsp_config = create_benchmark_dsp_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);
    RangeDopplerProcessor processor(dsp_config, get_benchmark_forward_model());

    auto plan = create_multi_chirp_plan(10);

    // Pre-simulate burst for processing benchmark
    auto burst = simulator.simulate_burst(plan);

    for (auto _ : state) {
        auto range_cube = processor.process_range(burst);
        auto tensor = processor.to_measurement_tensor(range_cube);
        benchmark::DoNotOptimize(tensor);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BurstProcessing)->Unit(benchmark::kMillisecond);

// Benchmark: DSP only (no simulation) for processor latency
static void BM_DspOnly(benchmark::State& state) {
    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto dsp_config = create_benchmark_dsp_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);
    RangeDopplerProcessor processor(dsp_config, get_benchmark_forward_model());

    auto plan = create_multi_chirp_plan(state.range(0));
    auto burst = simulator.simulate_burst(plan);

    for (auto _ : state) {
        auto range_cube = processor.process_range(burst);
        benchmark::DoNotOptimize(range_cube);
    }

    auto stats = processor.get_latency_stats();
    state.counters["preprocess_us"] = stats.preprocess_us;
    state.counters["iq_demod_us"] = stats.iq_demod_us;
    state.counters["window_us"] = stats.window_us;
    state.counters["fft_us"] = stats.fft_us;

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DspOnly)
    ->Arg(1)
    ->Arg(10)
    ->Arg(50)
    ->Unit(benchmark::kMillisecond);

// Benchmark: Doppler processing (up/down chirp pairing)
static void BM_DopplerProcessing(benchmark::State& state) {
    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto dsp_config = create_benchmark_dsp_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);
    RangeDopplerProcessor processor(dsp_config, get_benchmark_forward_model());

    // Create plan with up/down chirp pairs
    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(1);
    plan.set_start_timestamp_s(0.0);

    *plan.add_chirps() = create_chirp_plan(
        0, kWavelength1550nm, aurora::acquisition::ChirpPlan::DIRECTION_UP);
    *plan.add_chirps() = create_chirp_plan(
        1, kWavelength1550nm, aurora::acquisition::ChirpPlan::DIRECTION_DOWN);

    auto burst = simulator.simulate_burst(plan);

    for (auto _ : state) {
        auto range_cube = processor.process_doppler(burst);
        benchmark::DoNotOptimize(range_cube);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_DopplerProcessing)->Unit(benchmark::kMicrosecond);

// =============================================================================
// MEMORY BENCHMARKS
// =============================================================================

// Benchmark: Memory footprint for burst storage
static void BM_MemoryBurst(benchmark::State& state) {
    const int num_chirps = state.range(0);

    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);
    auto plan = create_multi_chirp_plan(num_chirps);

    for (auto _ : state) {
        auto burst = simulator.simulate_burst(plan);
        benchmark::DoNotOptimize(burst);

        // Report approximate memory usage
        state.counters["burst_bytes"] = burst.ByteSizeLong();
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MemoryBurst)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(200)
    ->Unit(benchmark::kMillisecond);

// Benchmark: Memory footprint for MeasurementTensor
static void BM_MemoryTensor(benchmark::State& state) {
    const int num_chirps = state.range(0);

    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto dsp_config = create_benchmark_dsp_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);
    RangeDopplerProcessor processor(dsp_config, get_benchmark_forward_model());

    auto plan = create_multi_chirp_plan(num_chirps);
    auto burst = simulator.simulate_burst(plan);
    auto range_cube = processor.process_range(burst);

    for (auto _ : state) {
        auto tensor = processor.to_measurement_tensor(range_cube);
        benchmark::DoNotOptimize(tensor);

        state.counters["tensor_bytes"] = tensor.ByteSizeLong();
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MemoryTensor)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Unit(benchmark::kMillisecond);

// Benchmark: Steady-state memory over consecutive bursts
static void BM_MemorySteadyState(benchmark::State& state) {
    const int num_bursts = 10;
    const int chirps_per_burst = 16;

    auto sim_config = create_benchmark_sim_config();
    auto forward_config = create_benchmark_forward_config();
    auto dsp_config = create_benchmark_dsp_config();
    auto calibration = create_benchmark_calibration();

    FmcwSimulator simulator(get_benchmark_forward_model(), sim_config);
    RangeDopplerProcessor processor(dsp_config, get_benchmark_forward_model());

    auto plan = create_multi_chirp_plan(chirps_per_burst);

    for (auto _ : state) {
        std::vector<aurora::shared::MeasurementTensor> tensors;
        tensors.reserve(num_bursts);

        for (int i = 0; i < num_bursts; ++i) {
            auto burst = simulator.simulate_burst(plan);
            auto range_cube = processor.process_range(burst);
            tensors.push_back(processor.to_measurement_tensor(range_cube));
        }

        benchmark::DoNotOptimize(tensors);

        // Report total memory for all tensors
        size_t total_bytes = 0;
        for (const auto& t : tensors) {
            total_bytes += t.ByteSizeLong();
        }
        state.counters["total_tensor_bytes"] = static_cast<double>(total_bytes);
    }

    state.SetItemsProcessed(state.iterations() * num_bursts);
    state.SetLabel(std::to_string(num_bursts) + " bursts x " +
                   std::to_string(chirps_per_burst) + " chirps");
}
BENCHMARK(BM_MemorySteadyState)->Unit(benchmark::kMillisecond);

}  // namespace

BENCHMARK_MAIN();
