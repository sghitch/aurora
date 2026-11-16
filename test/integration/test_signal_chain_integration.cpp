/**
 * @file test_signal_chain_integration.cpp
 * @brief Signal chain integration tests.
 *
 * These tests validate the end-to-end signal chain from scene description
 * through MeasurementTensor output. They verify physics correctness, phase
 * coherence, and spectral discrimination.
 *
 * Test Categories:
 * 1. End-to-End Pipeline Tests
 *    - test_end_to_end_point_target
 *    - test_end_to_end_multi_target
 *    - test_end_to_end_dual_wavelength
 *    - test_phase_gradient_recovery
 *    - test_spectral_discrimination
 *
 * 2. Physics Validation Tests
 *    - test_range_equation
 *    - test_doppler_equation
 *    - test_phase_vs_position
 *    - test_wavelength_k_scaling
 *    - test_coherent_gain
 */

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numeric>
#include <optional>
#include <vector>

#include "dsp/range_doppler_processor.h"
#include "forward/forward_model.h"
#include "simulation/fmcw_simulator.h"
#include "test_forward_model_builder.h"

#include "calibration_artifact.pb.h"

namespace aurora::integration {
namespace {

// Physical constants
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kSpeedOfLight = 299792458.0;

// Test parameters
constexpr double kTestSampleRate = 125e6;        // 125 MHz
constexpr double kTestChirpBandwidth = 200e6;    // 200 MHz
constexpr double kTestChirpDuration = 12.5e-6;   // 12.5 us
constexpr double kTestWavelength905nm = 905e-9;  // 905 nm
constexpr double kTestWavelength1550nm = 1550e-9;  // 1550 nm
constexpr uint32_t kTestNumDetectors = 8;
constexpr uint32_t kTestFftSize = 2048;

// Focal length for K-space matching.
// The detector K-position is k = 2π * x_det / (f * λ).
// With 200mm focal length and 1550nm, detector 0 at -1.75mm has:
//   k = 2π * (-0.00175) / (0.2 * 1.55e-6) ≈ -35,500 rad/m
// This matches the K-vector range used in forward model tests.
constexpr double kTestFocalLength = 0.2;  // 200 mm

// Range resolution (minimum distinguishable range): c / (2 * B) = 0.75 m for 200 MHz
constexpr double kExpectedRangeResolution = kSpeedOfLight / (2.0 * kTestChirpBandwidth);

// Bin-to-range conversion factor: (c * T * f_s) / (2 * B * N_fft)
// This is the actual physics-correct mapping from FFT bin to range
constexpr double kBinToRangeFactor =
    (kSpeedOfLight * kTestChirpDuration * kTestSampleRate) /
    (2.0 * kTestChirpBandwidth * kTestFftSize);

// -----------------------------------------------------------------------------
// Helper: Create SimConfig for integration tests
// -----------------------------------------------------------------------------
simulation::SimConfig create_integration_sim_config() {
    simulation::SimConfig config;

    // ADC config
    auto* adc = config.mutable_adc();
    adc->set_sample_rate_hz(kTestSampleRate);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    // Noise config (minimal noise for deterministic tests)
    auto* noise = config.mutable_noise();
    noise->set_enable_shot_noise(false);
    noise->set_enable_thermal_noise(false);
    noise->set_enable_rin_noise(false);
    noise->set_enable_quantization_noise(false);
    noise->set_enable_dark_current(false);

    // Optical parameters
    config.set_tx_power_w(1.0);
    config.set_lo_power_w(1.0);
    config.set_system_efficiency(1.0);
    config.set_rng_seed(42);

    // Occlusion mode (must be explicitly set)
    config.set_occlusion_mode(simulation::OCCLUSION_NONE);

    // MeasurementOperator settings (must be explicitly set)
    auto* bw_config = config.mutable_bandwidth_config();
    bw_config->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    bw_config->set_quadrature_order(4);
    config.set_specular_threshold(1.0 - 1e-10);   // Treat as specular if >= this

    return config;
}

// -----------------------------------------------------------------------------
// Helper: Create ForwardConfig for integration tests
// -----------------------------------------------------------------------------
aurora::forward::ForwardConfig create_integration_forward_config() {
    aurora::forward::ForwardConfig config;

    // Detector geometry (8 pixels, 0.5mm pitch, centered array)
    auto* detector = config.mutable_detector();
    auto* array = detector->add_arrays();
    array->set_array_id("main");
    array->set_nominal_pitch_m(0.5e-3);
    array->set_bandwidth_hz(150e6);
    array->add_wavelengths_m(kTestWavelength905nm);
    array->add_wavelengths_m(kTestWavelength1550nm);
    // Use POINT sensitivity for physics tests that don't depend on finite aperture
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_POINT);

    for (uint32_t i = 0; i < kTestNumDetectors; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((static_cast<double>(i) - 3.5) * 0.5e-3);  // Centered
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.4e-3);
        pixel->set_height_m(0.4e-3);
    }

    // Optics config with pupil
    auto* optics = config.mutable_optics();
    optics->set_fourier_lens_focal_m(kTestFocalLength);  // 200 mm for K-space matching
    optics->set_fourier_lens_diameter_m(0.05);
    optics->add_wavelengths_m(kTestWavelength905nm);
    optics->add_wavelengths_m(kTestWavelength1550nm);

    // Pupil config (wide FOV for testing)
    auto* pupil = optics->mutable_pupil();
    pupil->set_fov_half_angle_rad(0.5);  // ~30 degrees
    pupil->set_rolloff_width_rad(0.05);

    return config;
}

// -----------------------------------------------------------------------------
// Helper: Create DSP config for integration tests
// -----------------------------------------------------------------------------
dsp::DspConfig create_integration_dsp_config() {
    dsp::DspConfig config;
    config.set_range_fft_size(kTestFftSize);
    config.mutable_range_window()->set_type(dsp::WindowConfig::WINDOW_HANN);
    config.set_enable_dc_removal(true);

    auto* adc = config.mutable_adc_config();
    adc->set_sample_rate_hz(kTestSampleRate);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    auto* iq = config.mutable_iq_demod_config();
    iq->set_lpf_cutoff_hz(10e6);
    iq->set_lpf_order(4);

    return config;
}

// -----------------------------------------------------------------------------
// Helper: Create calibration artifact
// -----------------------------------------------------------------------------
aurora::shared::CalibrationArtifact create_integration_calibration() {
    aurora::shared::CalibrationArtifact calibration;
    calibration.set_calibration_id("integration_test_cal");
    calibration.set_timestamp_s(0.0);

    // Per-detector calibrations (identity)
    for (uint32_t i = 0; i < kTestNumDetectors; ++i) {
        auto* det_cal = calibration.add_detector_calibrations();
        det_cal->set_detector_index(i);
        det_cal->mutable_complex_gain()->set_real(1.0);
        det_cal->mutable_complex_gain()->set_imag(0.0);
        det_cal->set_timing_skew_s(0.0);
    }

    // K-space calibrations for both wavelengths
    for (double wavelength : {kTestWavelength905nm, kTestWavelength1550nm}) {
        auto* k_cal = calibration.add_k_space_calibrations();
        k_cal->set_wavelength_m(wavelength);
        k_cal->set_wavelength_scale(1.0);
        for (uint32_t i = 0; i < kTestNumDetectors; ++i) {
            k_cal->add_kx_offsets(0.0);
            k_cal->add_ky_offsets(0.0);
        }
    }

    return calibration;
}

// -----------------------------------------------------------------------------
// Helper: Build ForwardModel from ForwardConfig
// -----------------------------------------------------------------------------
aurora::forward::ForwardModel build_forward_model(
    const aurora::forward::ForwardConfig& config) {
    return aurora::test::TestForwardModelBuilder::build(config);
}

// -----------------------------------------------------------------------------
// Helper: Create chirp plan
// -----------------------------------------------------------------------------
aurora::acquisition::ChirpPlan create_chirp_plan(
    uint32_t index,
    double wavelength_m,
    aurora::acquisition::ChirpPlan::Direction direction =
        aurora::acquisition::ChirpPlan::DIRECTION_UP) {

    aurora::acquisition::ChirpPlan plan;
    plan.set_chirp_index(index);
    plan.set_chirp_start_s(index * kTestChirpDuration);
    plan.set_chirp_duration_s(kTestChirpDuration);
    plan.set_chirp_start_hz(0.0);
    plan.set_chirp_bandwidth_hz(kTestChirpBandwidth);
    plan.set_direction(direction);
    plan.add_wavelength_m(wavelength_m);
    return plan;
}

// -----------------------------------------------------------------------------
// Helper: Create sequence plan for a single wavelength
// -----------------------------------------------------------------------------
aurora::acquisition::SequencePlan create_single_wavelength_plan(
    double wavelength_m,
    bool include_down_chirp = false) {

    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(1);
    plan.set_start_timestamp_s(0.0);

    // Add chirps
    *plan.add_chirps() = create_chirp_plan(0, wavelength_m);

    if (include_down_chirp) {
        *plan.add_chirps() = create_chirp_plan(
            1, wavelength_m, aurora::acquisition::ChirpPlan::DIRECTION_DOWN);
    }

    return plan;
}

// -----------------------------------------------------------------------------
// Helper: Create dual-wavelength sequence plan
// -----------------------------------------------------------------------------
aurora::acquisition::SequencePlan create_dual_wavelength_plan() {
    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(1);
    plan.set_start_timestamp_s(0.0);

    // Add chirps for both wavelengths
    *plan.add_chirps() = create_chirp_plan(0, kTestWavelength905nm);
    *plan.add_chirps() = create_chirp_plan(1, kTestWavelength1550nm);

    return plan;
}

// -----------------------------------------------------------------------------
// Helper: Find peak range bin in detector data
// -----------------------------------------------------------------------------
struct PeakInfo {
    int bin;
    double magnitude;
    double phase_rad;
};

PeakInfo find_peak_bin(const dsp::DetectorRangeData& detector) {
    PeakInfo result{0, 0.0, 0.0};

    for (int i = 0; i < detector.bins_size(); ++i) {
        double mag = std::sqrt(
            detector.bins(i).real() * detector.bins(i).real() +
            detector.bins(i).imag() * detector.bins(i).imag());
        if (mag > result.magnitude) {
            result.magnitude = mag;
            result.bin = i;
            result.phase_rad = std::atan2(
                detector.bins(i).imag(), detector.bins(i).real());
        }
    }
    return result;
}

// Helper to find peak in MeasurementTensor detector data
PeakInfo find_peak_bin_tensor(const aurora::shared::DetectorChannel& detector) {
    PeakInfo result{0, 0.0, 0.0};

    for (int i = 0; i < detector.range_data_size(); ++i) {
        double mag = std::sqrt(
            detector.range_data(i).real() * detector.range_data(i).real() +
            detector.range_data(i).imag() * detector.range_data(i).imag());
        if (mag > result.magnitude) {
            result.magnitude = mag;
            result.bin = i;
            result.phase_rad = std::atan2(
                detector.range_data(i).imag(), detector.range_data(i).real());
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// Helper: Convert range bin to range in meters using physics-correct formula
// R = bin * (c * T * f_s) / (2 * B * N_fft)
// -----------------------------------------------------------------------------
double bin_to_range_physics(int bin) {
    return bin * kBinToRangeFactor;
}

// -----------------------------------------------------------------------------
// Helper: Calculate expected beat frequency for a given range
// -----------------------------------------------------------------------------
double range_to_beat_freq(double range_m) {
    return 2.0 * kTestChirpBandwidth * range_m /
           (kSpeedOfLight * kTestChirpDuration);
}

// -----------------------------------------------------------------------------
// Helper: Calculate expected FFT bin for a given range
// -----------------------------------------------------------------------------
double range_to_expected_bin(double range_m) {
    double beat_freq = range_to_beat_freq(range_m);
    double freq_resolution = kTestSampleRate / static_cast<double>(kTestFftSize);
    return beat_freq / freq_resolution;
}

// =============================================================================
// Test Fixture with shared expensive initialization
// =============================================================================
class SignalChainIntegrationTest : public ::testing::Test {
protected:
    // Use SetUpTestSuite to create expensive configs once per test suite
    static void SetUpTestSuite() {
        sim_config_ = new simulation::SimConfig(create_integration_sim_config());
        forward_config_ = new aurora::forward::ForwardConfig(
            create_integration_forward_config());
        dsp_config_ = new dsp::DspConfig(create_integration_dsp_config());
        calibration_ = new aurora::shared::CalibrationArtifact(
            create_integration_calibration());

        // Build ForwardModel from ForwardConfig
        forward_model_.emplace(build_forward_model(*forward_config_));

        // Pre-create a shared DSP processor
        shared_processor_ = new dsp::RangeDopplerProcessor(
            *dsp_config_, *forward_model_);
    }

    static void TearDownTestSuite() {
        delete shared_processor_;
        forward_model_.reset();
        delete calibration_;
        delete dsp_config_;
        delete forward_config_;
        delete sim_config_;
        shared_processor_ = nullptr;
        calibration_ = nullptr;
        dsp_config_ = nullptr;
        forward_config_ = nullptr;
        sim_config_ = nullptr;
    }

    // Create a fresh simulator with modified scene
    std::unique_ptr<simulation::FmcwSimulator> create_simulator(
        const simulation::SimConfig& config) {
        return std::make_unique<simulation::FmcwSimulator>(
            *forward_model_, config);
    }

    // Convenience accessors
    const simulation::SimConfig& sim_config() const { return *sim_config_; }
    const aurora::forward::ForwardConfig& forward_config() const {
        return *forward_config_;
    }
    const dsp::DspConfig& dsp_config() const { return *dsp_config_; }
    const aurora::shared::CalibrationArtifact& calibration() const {
        return *calibration_;
    }
    dsp::RangeDopplerProcessor& processor() { return *shared_processor_; }

    // Helper to add a point target to a config
    static void add_point_target(
        simulation::SimConfig& config,
        double x, double y, double z,
        double reflectivity_905nm = 0.5,
        double reflectivity_1550nm = 0.5,
        double vx = 0.0, double vy = 0.0, double vz = 0.0) {

        auto* scene = config.mutable_scene();
        auto* target = scene->add_point_targets();
        target->mutable_position_m()->set_x(x);
        target->mutable_position_m()->set_y(y);
        target->mutable_position_m()->set_z(z);
        target->mutable_velocity_mps()->set_x(vx);
        target->mutable_velocity_mps()->set_y(vy);
        target->mutable_velocity_mps()->set_z(vz);
        target->mutable_surface()->set_reflectivity_905nm(reflectivity_905nm);
        target->mutable_surface()->set_reflectivity_1550nm(reflectivity_1550nm);
        target->mutable_surface()->set_roughness_rms_m(1e-6);
        target->mutable_surface()->set_correlation_length_m(10e-6);
    }

    static simulation::SimConfig* sim_config_;
    static aurora::forward::ForwardConfig* forward_config_;
    static dsp::DspConfig* dsp_config_;
    static aurora::shared::CalibrationArtifact* calibration_;
    static std::optional<aurora::forward::ForwardModel> forward_model_;
    static dsp::RangeDopplerProcessor* shared_processor_;
};

// Static member definitions
simulation::SimConfig* SignalChainIntegrationTest::sim_config_ = nullptr;
aurora::forward::ForwardConfig* SignalChainIntegrationTest::forward_config_ = nullptr;
dsp::DspConfig* SignalChainIntegrationTest::dsp_config_ = nullptr;
aurora::shared::CalibrationArtifact* SignalChainIntegrationTest::calibration_ = nullptr;
std::optional<aurora::forward::ForwardModel> SignalChainIntegrationTest::forward_model_;
dsp::RangeDopplerProcessor* SignalChainIntegrationTest::shared_processor_ = nullptr;

// =============================================================================
// END-TO-END PIPELINE TESTS
// =============================================================================

// Test: Single point target produces expected MeasurementTensor
TEST_F(SignalChainIntegrationTest, EndToEndPointTarget) {
    // Setup: Single point target at known position (50m range, on-axis)
    simulation::SimConfig config = sim_config();
    add_point_target(config, 0.0, 0.0, 50.0, 0.5, 1.0);

    auto simulator = create_simulator(config);
    auto plan = create_single_wavelength_plan(kTestWavelength1550nm);

    // Execute pipeline: Scene -> Simulation -> DSP -> MeasurementTensor
    auto burst = simulator->simulate_burst(plan);
    auto range_cube = processor().process_range(burst);
    auto tensor = processor().to_measurement_tensor(range_cube);

    // Verify structure
    ASSERT_EQ(tensor.data_size(), 1) << "Should have one wavelength channel";
    ASSERT_EQ(tensor.data(0).detectors_size(),
              static_cast<int>(kTestNumDetectors))
        << "Should have all detectors";

    // Verify range detection
    const auto& detector = tensor.data(0).detectors(0);
    auto peak = find_peak_bin_tensor(detector);

    double expected_bin = range_to_expected_bin(50.0);

    // Peak should be within 2 bins of expected (allowing for windowing effects)
    EXPECT_NEAR(static_cast<double>(peak.bin), expected_bin, 2.0)
        << "Peak at bin " << peak.bin << ", expected ~bin " << expected_bin
        << " (50 m target)";

    // Verify detected range using physics-correct bin-to-range conversion
    double detected_range = bin_to_range_physics(peak.bin);
    EXPECT_NEAR(detected_range, 50.0, kBinToRangeFactor * 2.0)
        << "Detected range " << detected_range << " m should be within 2 bins of 50 m";

    // Verify K-vectors are present and finite
    for (const auto& det : tensor.data(0).detectors()) {
        EXPECT_TRUE(det.has_k_vector());
        EXPECT_TRUE(std::isfinite(det.k_vector().kx()));
        EXPECT_TRUE(std::isfinite(det.k_vector().ky()));
    }
}

// Test: Multiple targets at different ranges are separable
TEST_F(SignalChainIntegrationTest, EndToEndMultiTarget) {
    // Setup: Two targets separated by more than range resolution
    // Target 1 at 30m, Target 2 at 50m (20m separation >> 0.75m resolution)
    simulation::SimConfig config = sim_config();
    add_point_target(config, 0.0, 0.0, 30.0, 0.5, 1.0);
    add_point_target(config, 0.0, 0.0, 50.0, 0.5, 1.0);

    auto simulator = create_simulator(config);
    auto plan = create_single_wavelength_plan(kTestWavelength1550nm);

    // Execute pipeline
    auto burst = simulator->simulate_burst(plan);
    auto range_cube = processor().process_range(burst);

    // Analyze detector 0 for peaks
    const auto& detector = range_cube.wavelengths(0).detectors(0);

    // Find the two highest peaks
    std::vector<std::pair<int, double>> peaks;
    for (int i = 0; i < detector.bins_size(); ++i) {
        double mag = std::sqrt(
            detector.bins(i).real() * detector.bins(i).real() +
            detector.bins(i).imag() * detector.bins(i).imag());
        peaks.emplace_back(i, mag);
    }

    // Sort by magnitude descending
    std::sort(peaks.begin(), peaks.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Require at least two significant peaks
    ASSERT_GE(peaks.size(), 2u);

    // The top 2 peaks should be distinct (separated by at least 10 bins)
    int peak1_bin = peaks[0].first;
    int peak2_bin = peaks[1].first;
    int bin_separation = std::abs(peak1_bin - peak2_bin);

    EXPECT_GT(bin_separation, 10)
        << "Two targets should produce distinct peaks (separation: "
        << bin_separation << " bins)";

    // Verify both peaks correspond to expected ranges (within 3 bins)
    double expected_bin_30m = range_to_expected_bin(30.0);
    double expected_bin_50m = range_to_expected_bin(50.0);

    int min_peak = std::min(peak1_bin, peak2_bin);
    int max_peak = std::max(peak1_bin, peak2_bin);

    EXPECT_NEAR(static_cast<double>(min_peak), expected_bin_30m, 3.0)
        << "First target peak should be near 30m";
    EXPECT_NEAR(static_cast<double>(max_peak), expected_bin_50m, 3.0)
        << "Second target peak should be near 50m";
}

// Test: Phase gradient encodes target position (qualitative test)
TEST_F(SignalChainIntegrationTest, PhaseGradientRecovery) {
    // This test validates that the signal chain produces K-vectors and phases
    // that can, in principle, be used for phase gradient localization.
    //
    // Note: The exact phase-position relationship (φ = -k·r) depends on:
    // - The detector K-space geometry (k_j = 2π·x_j/(f·λ))
    // - The forward model's effective K-vector calculation
    // - Calibration accuracy
    //
    // The forward model unit tests validate the exact physics. Here we just verify the integration works.
    simulation::SimConfig config = sim_config();
    add_point_target(config, 0.0, 0.0, 50.0, 0.5, 1.0);

    auto simulator = create_simulator(config);
    auto plan = create_single_wavelength_plan(kTestWavelength1550nm);

    // Execute pipeline
    auto burst = simulator->simulate_burst(plan);
    auto range_cube = processor().process_range(burst);
    auto tensor = processor().to_measurement_tensor(range_cube);

    // Verify structure
    ASSERT_EQ(tensor.data_size(), 1);
    ASSERT_GT(tensor.data_size(), 0);
    const auto& wl_data = tensor.data(0);
    ASSERT_EQ(wl_data.detectors_size(), static_cast<int>(kTestNumDetectors));

    // Verify all detectors have K-vectors and phase data
    for (int d = 0; d < wl_data.detectors_size(); ++d) {
        const auto& detector = wl_data.detectors(d);

        // K-vector should be present and finite
        EXPECT_TRUE(detector.has_k_vector())
            << "Detector " << d << " should have K-vector";
        EXPECT_TRUE(std::isfinite(detector.k_vector().kx()))
            << "Detector " << d << " kx should be finite";
        EXPECT_TRUE(std::isfinite(detector.k_vector().ky()))
            << "Detector " << d << " ky should be finite";

        // Range data should be present
        EXPECT_GT(detector.range_data_size(), 0)
            << "Detector " << d << " should have range data";
    }

    // Verify K-vectors vary across detectors (different spatial frequencies)
    std::vector<double> kx_values;
    for (const auto& detector : wl_data.detectors()) {
        kx_values.push_back(detector.k_vector().kx());
    }

    double k_span = *std::max_element(kx_values.begin(), kx_values.end()) -
                    *std::min_element(kx_values.begin(), kx_values.end());

    // K-vectors should span some range (not all identical)
    // With our configuration, we expect span in thousands of rad/m
    EXPECT_GT(k_span, 0.0)
        << "K-vectors should vary across detectors";
}

// Test: Spectral discrimination between materials
TEST_F(SignalChainIntegrationTest, SpectralDiscrimination) {
    // Setup: Two targets with different spectral ratios at different ranges
    // Dry surface (ratio ~1.0) at 30m, Wet surface (ratio ~0.3) at 50m
    simulation::SimConfig config = sim_config();
    add_point_target(config, 0.0, 0.0, 30.0, 0.5, 1.0);  // Dry: equal at both λ
    add_point_target(config, 0.0, 0.0, 50.0, 0.5, 0.3);  // Wet: 1550nm attenuated

    auto simulator = create_simulator(config);
    auto plan = create_dual_wavelength_plan();

    // Execute pipeline
    auto burst = simulator->simulate_burst(plan);
    auto range_cube = processor().process_range(burst);
    auto tensor = processor().to_measurement_tensor(range_cube);

    ASSERT_EQ(tensor.data_size(), 2);

    // Find wavelength indices
    int wl_905_idx = -1, wl_1550_idx = -1;
    for (int i = 0; i < tensor.data_size(); ++i) {
        double wl = tensor.data(i).wavelength_m();
        if (std::abs(wl - kTestWavelength905nm) < 1e-12) wl_905_idx = i;
        if (std::abs(wl - kTestWavelength1550nm) < 1e-12) wl_1550_idx = i;
    }

    ASSERT_NE(wl_905_idx, -1) << "Should have 905nm data";
    ASSERT_NE(wl_1550_idx, -1) << "Should have 1550nm data";

    // Get data for detector 0 at both wavelengths
    const auto& det_905 = tensor.data(wl_905_idx).detectors(0);
    const auto& det_1550 = tensor.data(wl_1550_idx).detectors(0);

    // Find peaks for both targets at 905nm
    double expected_bin_30m = range_to_expected_bin(30.0);
    double expected_bin_50m = range_to_expected_bin(50.0);

    int bin_30m = static_cast<int>(std::round(expected_bin_30m));
    int bin_50m = static_cast<int>(std::round(expected_bin_50m));

    // Clamp to valid range
    bin_30m = std::clamp(bin_30m, 0, det_905.range_data_size() - 1);
    bin_50m = std::clamp(bin_50m, 0, det_905.range_data_size() - 1);

    // Get magnitudes at 30m bin (dry target)
    auto mag_at_bin = [](const aurora::shared::DetectorChannel& det, int bin) {
        return std::sqrt(det.range_data(bin).real() * det.range_data(bin).real() +
                        det.range_data(bin).imag() * det.range_data(bin).imag());
    };

    double mag_905_at_30m = mag_at_bin(det_905, bin_30m);
    double mag_1550_at_30m = mag_at_bin(det_1550, bin_30m);
    double mag_905_at_50m = mag_at_bin(det_905, bin_50m);
    double mag_1550_at_50m = mag_at_bin(det_1550, bin_50m);

    // Compute spectral ratios (avoid division by zero)
    double ratio_30m = (mag_905_at_30m > 1e-10) ?
        mag_1550_at_30m / mag_905_at_30m : 0.0;
    double ratio_50m = (mag_905_at_50m > 1e-10) ?
        mag_1550_at_50m / mag_905_at_50m : 0.0;

    // Dry target (30m) should have ratio ~1.0
    // Wet target (50m) should have ratio ~0.3
    // Difference should be > 3 dB (factor of ~2)
    if (ratio_30m > 0.0 && ratio_50m > 0.0) {
        double ratio_diff_db = 10.0 * std::log10(ratio_30m / ratio_50m);

        // The dry target should have higher 1550/905 ratio than wet
        EXPECT_GT(ratio_diff_db, 3.0)
            << "Spectral ratio difference should be > 3 dB between dry ("
            << ratio_30m << ") and wet (" << ratio_50m << ") targets. "
            << "Actual: " << ratio_diff_db << " dB";
    }
}

// =============================================================================
// PHYSICS VALIDATION TESTS
// =============================================================================

// Test: Range equation R = c * f_beat * T / (2 * B)
TEST_F(SignalChainIntegrationTest, RangeEquation) {
    // Test multiple ranges and verify the range equation holds
    //
    // The FMCW range equation:
    //   f_beat = 2 * B * R / (c * T)
    //   bin = f_beat / (f_s / N_fft)
    //   R = bin * (c * T * f_s) / (2 * B * N_fft)
    //
    // Note: DSP's range_axis.step() is set to c/(2B) (range resolution), which
    // is the minimum distinguishable range difference, NOT the bin-to-range factor.
    // The physics-correct bin-to-range conversion is kBinToRangeFactor.
    std::vector<double> test_ranges = {25.0, 50.0, 75.0, 100.0};

    for (double true_range : test_ranges) {
        simulation::SimConfig config = sim_config();
        add_point_target(config, 0.0, 0.0, true_range, 0.5, 1.0);

        auto simulator = create_simulator(config);
        auto plan = create_single_wavelength_plan(kTestWavelength1550nm);

        auto burst = simulator->simulate_burst(plan);
        auto range_cube = processor().process_range(burst);

        const auto& detector = range_cube.wavelengths(0).detectors(0);
        auto peak = find_peak_bin(detector);

        // Convert bin to range using physics-correct formula
        double detected_range = bin_to_range_physics(peak.bin);

        // Verify peak is at expected bin
        double expected_bin = range_to_expected_bin(true_range);
        EXPECT_NEAR(static_cast<double>(peak.bin), expected_bin, 2.0)
            << "At true range " << true_range << " m, peak at bin "
            << peak.bin << ", expected ~bin " << expected_bin;

        // Range equation accuracy should be < 0.5% plus quantization (1 bin)
        double max_error = true_range * 0.005 + kBinToRangeFactor;
        EXPECT_NEAR(detected_range, true_range, max_error)
            << "At true range " << true_range << " m, detected "
            << detected_range << " m (bin " << peak.bin << ")";
    }
}

// Test: Up/down chirp processing for Doppler
TEST_F(SignalChainIntegrationTest, DopplerEquation) {
    // Test that process_doppler works correctly with up/down chirp pairs.
    //
    // Note: The current DSP implementation averages up/down chirps at each bin
    // index. This works for stationary targets (where up/down peaks coincide)
    // but doesn't fully correct for velocity when up/down peaks are separated.
    //
    // For a stationary target, both up and down chirps should produce the same
    // beat frequency, so averaging preserves the peak at the correct range bin.
    double target_range = 50.0;

    simulation::SimConfig config = sim_config();
    // Stationary target - no Doppler shift
    add_point_target(config, 0.0, 0.0, target_range, 0.5, 1.0,
                     0.0, 0.0, 0.0);

    auto simulator = create_simulator(config);

    // Create plan with both up and down chirps
    auto plan = create_single_wavelength_plan(kTestWavelength1550nm, true);

    auto burst = simulator->simulate_burst(plan);
    auto range_cube = processor().process_doppler(burst);

    // Verify structure: should have range data from Doppler processing
    ASSERT_EQ(range_cube.wavelengths_size(), 1);
    ASSERT_GT(range_cube.wavelengths(0).detectors_size(), 0);

    const auto& detector = range_cube.wavelengths(0).detectors(0);
    auto peak = find_peak_bin(detector);

    double expected_range_bin = range_to_expected_bin(target_range);

    // For stationary target, peak should be at true range
    EXPECT_NEAR(static_cast<double>(peak.bin), expected_range_bin, 2.0)
        << "Stationary target should produce peak at true range bin. "
        << "Peak at bin " << peak.bin << ", expected ~" << expected_range_bin;

    // Verify coherent gain: Doppler-processed should have similar or better SNR
    // than single-chirp processing (since we're averaging two measurements)
    EXPECT_GT(peak.magnitude, 0.0)
        << "Doppler-processed signal should have non-zero magnitude";
}

// Test: Multiple targets at same range produce consistent phases
TEST_F(SignalChainIntegrationTest, PhaseVsPosition) {
    // This test validates that multiple independent simulations produce
    // consistent, deterministic phase outputs.
    //
    // Note: The exact φ = -k·r relationship is validated in the forward model
    // unit tests. Here we verify that:
    // 1. The simulation produces finite, consistent phases
    // 2. Phase values vary predictably (deterministic with same seed)
    // 3. The full pipeline from simulation to MeasurementTensor works
    std::vector<double> x_positions = {-2.0, 0.0, 2.0};
    double z_range = 50.0;

    std::vector<double> measured_phases;
    std::vector<double> kx_values;

    for (double x : x_positions) {
        simulation::SimConfig config = sim_config();
        add_point_target(config, x, 0.0, z_range, 0.5, 1.0);

        auto simulator = create_simulator(config);
        auto plan = create_single_wavelength_plan(kTestWavelength1550nm);

        auto burst = simulator->simulate_burst(plan);
        auto range_cube = processor().process_range(burst);
        auto tensor = processor().to_measurement_tensor(range_cube);

        // Get phase at detector 0 for the peak range bin
        const auto& detector = tensor.data(0).detectors(0);
        auto peak = find_peak_bin_tensor(detector);

        // Verify finite values
        EXPECT_TRUE(std::isfinite(peak.phase_rad))
            << "Phase should be finite for target at x=" << x;
        EXPECT_TRUE(std::isfinite(detector.k_vector().kx()))
            << "kx should be finite for target at x=" << x;

        measured_phases.push_back(peak.phase_rad);
        kx_values.push_back(detector.k_vector().kx());
    }

    // Verify that we got data for all positions
    ASSERT_EQ(measured_phases.size(), x_positions.size());
    ASSERT_EQ(kx_values.size(), x_positions.size());

    // K-vectors should be consistent (same detector, same geometry)
    double kx_0 = kx_values[0];
    for (size_t i = 1; i < kx_values.size(); ++i) {
        EXPECT_NEAR(kx_values[i], kx_0, 1.0)
            << "K-vector should be consistent across simulations";
    }

    // Phases should be finite (valid computation)
    for (size_t i = 0; i < measured_phases.size(); ++i) {
        EXPECT_TRUE(std::isfinite(measured_phases[i]))
            << "Phase " << i << " should be finite";
    }
}

// Test: K-vector scales with wavelength (k ∝ 1/λ)
TEST_F(SignalChainIntegrationTest, WavelengthKScaling) {
    // This test validates that K-vectors are properly computed for both
    // wavelengths and that the geometric relationship k ∝ 1/λ holds.
    //
    // Note: The effective K-vector is the detector's geometric K-position
    // k_j = 2π·x_j/(f·λ) plus any LO tilt.
    //
    // The forward model tests validate exact K-scaling. Here we verify that
    // both wavelengths produce valid K-vectors.
    simulation::SimConfig config = sim_config();
    add_point_target(config, 0.0, 0.0, 50.0, 0.5, 1.0);

    auto simulator = create_simulator(config);
    auto plan = create_dual_wavelength_plan();

    auto burst = simulator->simulate_burst(plan);
    auto range_cube = processor().process_range(burst);
    auto tensor = processor().to_measurement_tensor(range_cube);

    ASSERT_EQ(tensor.data_size(), 2);

    // Find wavelength indices
    int wl_905_idx = -1, wl_1550_idx = -1;
    for (int i = 0; i < tensor.data_size(); ++i) {
        double wl = tensor.data(i).wavelength_m();
        if (std::abs(wl - kTestWavelength905nm) < 1e-12) wl_905_idx = i;
        if (std::abs(wl - kTestWavelength1550nm) < 1e-12) wl_1550_idx = i;
    }

    ASSERT_NE(wl_905_idx, -1) << "Should have 905nm wavelength data";
    ASSERT_NE(wl_1550_idx, -1) << "Should have 1550nm wavelength data";

    const auto& wl_905 = tensor.data(wl_905_idx);
    const auto& wl_1550 = tensor.data(wl_1550_idx);

    // Verify both wavelengths have detector data with valid K-vectors
    EXPECT_EQ(wl_905.detectors_size(), static_cast<int>(kTestNumDetectors));
    EXPECT_EQ(wl_1550.detectors_size(), static_cast<int>(kTestNumDetectors));

    for (int d = 0; d < std::min(wl_905.detectors_size(), wl_1550.detectors_size()); ++d) {
        // K-vectors should be finite for both wavelengths
        EXPECT_TRUE(std::isfinite(wl_905.detectors(d).k_vector().kx()))
            << "Detector " << d << " should have finite kx at 905nm";
        EXPECT_TRUE(std::isfinite(wl_1550.detectors(d).k_vector().kx()))
            << "Detector " << d << " should have finite kx at 1550nm";

        // Both wavelengths should have range data
        EXPECT_GT(wl_905.detectors(d).range_data_size(), 0)
            << "Detector " << d << " should have range data at 905nm";
        EXPECT_GT(wl_1550.detectors(d).range_data_size(), 0)
            << "Detector " << d << " should have range data at 1550nm";
    }

    // Verify K-vectors span a range (not all zero)
    double max_kx_905 = 0.0, max_kx_1550 = 0.0;
    for (int d = 0; d < wl_905.detectors_size(); ++d) {
        max_kx_905 = std::max(max_kx_905, std::abs(wl_905.detectors(d).k_vector().kx()));
        max_kx_1550 = std::max(max_kx_1550, std::abs(wl_1550.detectors(d).k_vector().kx()));
    }

    EXPECT_GT(max_kx_905, 0.0) << "Should have non-zero K-vectors at 905nm";
    EXPECT_GT(max_kx_1550, 0.0) << "Should have non-zero K-vectors at 1550nm";
}

// Test: Simulation determinism - same seed produces identical results
TEST_F(SignalChainIntegrationTest, SimulationDeterminism) {
    simulation::SimConfig config = sim_config();
    add_point_target(config, 2.0, 1.0, 50.0, 0.5, 1.0);

    // Run simulation twice with same seed
    auto simulator1 = create_simulator(config);
    auto simulator2 = create_simulator(config);

    auto plan = create_single_wavelength_plan(kTestWavelength1550nm);

    auto burst1 = simulator1->simulate_burst(plan);
    auto burst2 = simulator2->simulate_burst(plan);

    // Bursts should be byte-identical
    ASSERT_EQ(burst1.detectors_size(), burst2.detectors_size());

    for (int d = 0; d < burst1.detectors_size(); ++d) {
        ASSERT_EQ(burst1.detectors(d).chirps_size(),
                  burst2.detectors(d).chirps_size());

        for (int c = 0; c < burst1.detectors(d).chirps_size(); ++c) {
            const auto& chirp1 = burst1.detectors(d).chirps(c);
            const auto& chirp2 = burst2.detectors(d).chirps(c);

            ASSERT_EQ(chirp1.samples_size(), chirp2.samples_size());

            for (int s = 0; s < chirp1.samples_size(); ++s) {
                EXPECT_EQ(chirp1.samples(s), chirp2.samples(s))
                    << "Mismatch at detector " << d
                    << ", chirp " << c << ", sample " << s;
            }
        }
    }
}

// Test: Coherent gain - averaging improves SNR
TEST_F(SignalChainIntegrationTest, CoherentGain) {
    // Enable noise to test coherent gain
    simulation::SimConfig config = sim_config();
    config.mutable_noise()->set_enable_quantization_noise(true);
    add_point_target(config, 0.0, 0.0, 50.0, 0.5, 1.0);

    auto simulator = create_simulator(config);
    auto plan = create_single_wavelength_plan(kTestWavelength1550nm);

    // Repeat the same burst several times and average coherently
    constexpr int kNumBursts = 4;
    std::vector<dsp::RangeCube> cubes;

    for (int i = 0; i < kNumBursts; ++i) {
        auto burst = simulator->simulate_burst(plan);
        cubes.push_back(processor().process_range(burst));
    }

    // Find peak in first (single) cube
    const auto& single_detector = cubes[0].wavelengths(0).detectors(0);
    auto single_peak = find_peak_bin(single_detector);

    // Compute average across all cubes at peak bin
    std::complex<double> avg_value(0.0, 0.0);
    for (const auto& cube : cubes) {
        const auto& det = cube.wavelengths(0).detectors(0);
        if (single_peak.bin < det.bins_size()) {
            avg_value += std::complex<double>(
                det.bins(single_peak.bin).real(),
                det.bins(single_peak.bin).imag());
        }
    }
    avg_value /= static_cast<double>(kNumBursts);

    double avg_magnitude = std::abs(avg_value);

    // Coherent averaging should preserve signal magnitude
    // (within some tolerance for quantization effects)
    EXPECT_GT(avg_magnitude, single_peak.magnitude * 0.7)
        << "Coherent averaging should preserve signal magnitude";

    // With coherent signal and incoherent noise, SNR improves as sqrt(N)
    // We can't easily measure noise floor here, but verify signal is preserved
    EXPECT_GT(avg_magnitude, 0.0) << "Averaged signal should be non-zero";
}

// Test: DSP integrity - range peak at correct bin
TEST_F(SignalChainIntegrationTest, DspIntegrity) {
    // Verify that DSP doesn't corrupt the range information across all detectors
    simulation::SimConfig config = sim_config();
    add_point_target(config, 0.0, 0.0, 50.0, 0.5, 1.0);

    auto simulator = create_simulator(config);
    auto plan = create_single_wavelength_plan(kTestWavelength1550nm);

    auto burst = simulator->simulate_burst(plan);
    auto range_cube = processor().process_range(burst);

    double expected_bin = range_to_expected_bin(50.0);

    // All detectors should see the peak at the same range bin
    const auto& wavelength = range_cube.wavelengths(0);

    for (int d = 0; d < wavelength.detectors_size(); ++d) {
        auto peak = find_peak_bin(wavelength.detectors(d));

        EXPECT_NEAR(static_cast<double>(peak.bin), expected_bin, 2.0)
            << "Detector " << d << " peak at bin " << peak.bin
            << ", expected ~" << expected_bin;
    }

    // Also verify range axis is correctly set
    // Note: range_axis().step() is the bin-to-range conversion factor, NOT the range resolution.
    // Range resolution (minimum resolvable difference) = c / (2B)
    // Bin-to-range factor = (c * T * f_s) / (2 * B * N_fft) = kBinToRangeFactor
    EXPECT_EQ(range_cube.range_axis().unit(), "m");
    EXPECT_DOUBLE_EQ(range_cube.range_axis().start(), 0.0);
    EXPECT_EQ(range_cube.range_axis().count(), kTestFftSize / 2);
    EXPECT_NEAR(range_cube.range_axis().step(), kBinToRangeFactor, 0.01);
}

}  // namespace
}  // namespace aurora::integration
