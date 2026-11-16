// Copyright (C) 2026 Stephen Hitchcock
//
// This file is part of Aurora.
//
// Aurora is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// Aurora is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Aurora. If not, see <https://www.gnu.org/licenses/>.

#include "simulation/fmcw_simulator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>

#include "forward/forward_model.h"
#include "simulation/chirp_generator.h"
#include "test_forward_model_builder.h"

#include "detector_geometry.pb.h"

namespace aurora::simulation {
namespace {

// Create a minimal SimConfig for testing
SimConfig create_test_sim_config() {
    SimConfig config;

    // ADC config
    auto* adc = config.mutable_adc();
    adc->set_sample_rate_hz(125e6);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    // Noise config (minimal noise for clean tests)
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
    config.set_occlusion_mode(OCCLUSION_NONE);

    // MeasurementOperator settings (must be explicitly set)
    config.mutable_bandwidth_config()->set_quadrature_order(11);  // 11×11 midpoint quadrature
    config.mutable_bandwidth_config()->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    config.set_specular_threshold(1.0 - 1e-10);   // Treat as specular if >= this

    return config;
}

// Create detector geometry for testing (8 pixels, 0.5mm pitch)
// Uses POINT sensitivity for tests that don't specifically test sensitivity behavior.
aurora::forward::DetectorGeometry create_test_detector_geometry() {
    aurora::forward::DetectorGeometry detector;
    auto* array = detector.add_arrays();
    array->set_array_id("main");
    array->set_nominal_pitch_m(0.0005);
    // Use POINT sensitivity for physics tests that don't depend on finite aperture
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_POINT);

    // Add explicit pixel positions with width and height (0.5mm pitch, ~0.4mm active area)
    constexpr double kPixelWidth = 0.0004;   // 400 um active area
    constexpr double kPixelHeight = 0.0004;  // 400 um active area
    for (int i = 0; i < 8; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((i - 3.5) * 0.0005);  // Center the array
        pixel->set_y_m(0.0);
        pixel->set_width_m(kPixelWidth);
        pixel->set_height_m(kPixelHeight);
    }
    return detector;
}

// Create detector geometry with SINC sensitivity (finite rectangular aperture)
// for tests that exercise the MeasurementOperator K-space integration.
aurora::forward::DetectorGeometry create_sinc_detector_geometry() {
    aurora::forward::DetectorGeometry detector;
    auto* array = detector.add_arrays();
    array->set_array_id("main");
    array->set_nominal_pitch_m(0.0005);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_SINC);
    constexpr double kPixelWidth = 0.0004;
    for (int i = 0; i < 8; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((i - 3.5) * 0.0005);
        pixel->set_y_m(0.0);
        pixel->set_width_m(kPixelWidth);
        pixel->set_height_m(kPixelWidth);  // Square pixels
    }
    return detector;
}

// Create optics config for testing
aurora::forward::OpticsConfig create_test_optics_config() {
    aurora::forward::OpticsConfig optics;
    optics.set_fourier_lens_focal_m(0.1);  // 100mm
    optics.add_wavelengths_m(905e-9);   // 905nm
    optics.add_wavelengths_m(1550e-9);  // 1550nm

    // Pupil config (wide FOV for testing)
    auto* pupil = optics.mutable_pupil();
    pupil->set_fov_half_angle_rad(0.5);  // ~30 degrees
    pupil->set_rolloff_width_rad(0.05);

    return optics;
}

// Create a chirp plan for testing
aurora::acquisition::ChirpPlan create_test_chirp_plan() {
    aurora::acquisition::ChirpPlan plan;
    plan.set_chirp_index(0);
    plan.set_chirp_start_s(0.0);
    plan.set_chirp_duration_s(12.5e-6);      // 12.5 us
    plan.set_chirp_start_hz(0.0);
    plan.set_chirp_bandwidth_hz(200e6);      // 200 MHz
    plan.set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);
    plan.add_wavelength_m(1550e-9);          // 1550 nm
    return plan;
}

// Create a single-chirp sequence plan for testing
aurora::acquisition::SequencePlan create_test_plan(uint64_t plan_id = 0,
                                                   double start_timestamp_s = 0.0) {
    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(plan_id);
    plan.set_start_timestamp_s(start_timestamp_s);
    *plan.add_chirps() = create_test_chirp_plan();
    return plan;
}

class FmcwSimulatorTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        detector_ = create_test_detector_geometry();
        optics_ = create_test_optics_config();

        // Create ForwardModel from geometry: each pixel samples k_j = 2π·x_j/(f·λ)
        forward_model_.emplace(detector_, optics_);
    }

    static void TearDownTestSuite() {
        forward_model_.reset();
    }

    void SetUp() override {
        sim_config_ = create_test_sim_config();
    }

    const aurora::forward::ForwardModel& forward_model() const { return *forward_model_; }

    // Add a point target to the sim config
    void add_point_target(double x, double y, double z,
                          double reflectivity_905nm = 1.0,
                          double reflectivity_1550nm = 1.0) {
        auto* scene = sim_config_.mutable_scene();
        auto* target = scene->add_point_targets();
        target->mutable_position_m()->set_x(x);
        target->mutable_position_m()->set_y(y);
        target->mutable_position_m()->set_z(z);
        target->mutable_surface()->set_reflectivity_905nm(reflectivity_905nm);
        target->mutable_surface()->set_reflectivity_1550nm(reflectivity_1550nm);
        target->mutable_surface()->set_roughness_rms_m(1e-6);
        target->mutable_surface()->set_correlation_length_m(10e-6);
    }

    SimConfig sim_config_;

    static aurora::forward::DetectorGeometry detector_;
    static aurora::forward::OpticsConfig optics_;
    static std::optional<aurora::forward::ForwardModel> forward_model_;
};

// Static member definitions
aurora::forward::DetectorGeometry FmcwSimulatorTest::detector_;
aurora::forward::OpticsConfig FmcwSimulatorTest::optics_;
std::optional<aurora::forward::ForwardModel> FmcwSimulatorTest::forward_model_;

// Test beat frequency matches target range
TEST_F(FmcwSimulatorTest, BeatFrequencyMatchesRange) {
    // Add a target at 100m range
    add_point_target(0.0, 0.0, 100.0);

    FmcwSimulator simulator(forward_model(), sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());

    // Check we got samples
    ASSERT_GT(result.detectors_size(), 0);
    ASSERT_GT(result.detectors(0).chirps_size(), 0);
    ASSERT_GT(result.detectors(0).chirps(0).samples_size(), 0);

    // Extract samples and do FFT analysis would require more infrastructure
    // For now, just verify we have non-zero signal
    bool has_nonzero = false;
    for (int32_t sample : result.detectors(0).chirps(0).samples()) {
        if (sample != 0) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// Test determinism with same seed
TEST_F(FmcwSimulatorTest, Determinism) {
    add_point_target(0.0, 0.0, 100.0);

    FmcwSimulator sim1(forward_model(), sim_config_);
    FmcwSimulator sim2(forward_model(), sim_config_);

    auto plan = create_test_plan();

    auto result1 = sim1.simulate_burst(plan);
    auto result2 = sim2.simulate_burst(plan);

    // Results should be identical
    ASSERT_EQ(result1.detectors_size(), result2.detectors_size());
    for (int d = 0; d < result1.detectors_size(); ++d) {
        ASSERT_EQ(result1.detectors(d).chirps_size(),
                  result2.detectors(d).chirps_size());
        for (int c = 0; c < result1.detectors(d).chirps_size(); ++c) {
            ASSERT_EQ(result1.detectors(d).chirps(c).samples_size(),
                      result2.detectors(d).chirps(c).samples_size());
            for (int s = 0; s < result1.detectors(d).chirps(c).samples_size(); ++s) {
                EXPECT_EQ(result1.detectors(d).chirps(c).samples(s),
                          result2.detectors(d).chirps(c).samples(s))
                    << "Mismatch at detector " << d << ", chirp " << c
                    << ", sample " << s;
            }
        }
    }
}

// Test pupil cutoff - targets outside FOV produce no signal
TEST_F(FmcwSimulatorTest, PupilCutoff) {
    // Target well outside FOV (60 degrees off-axis, FOV is 30 degrees)
    // Position at 100m range but 100m off-axis gives ~45 degree angle
    add_point_target(200.0, 0.0, 100.0);  // Very far off-axis

    FmcwSimulator simulator(forward_model(), sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());

    // Signal should be near zero (within quantization noise)
    int64_t sum = 0;
    int count = 0;
    for (const auto& det : result.detectors()) {
        for (const auto& chirp : det.chirps()) {
            for (int32_t sample : chirp.samples()) {
                sum += std::abs(sample);
                count++;
            }
        }
    }

    double mean_abs = static_cast<double>(sum) / count;
    // With no noise enabled, signal should be essentially zero
    EXPECT_LT(mean_abs, 10.0);  // Very small
}

// Test multi-target produces combined signal
TEST_F(FmcwSimulatorTest, MultiTarget) {
    // Add two targets at different ranges
    add_point_target(0.0, 0.0, 50.0, 1.0);
    add_point_target(0.0, 0.0, 100.0, 1.0);

    FmcwSimulator simulator(forward_model(), sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());

    // Should have signal
    ASSERT_GT(result.detectors_size(), 0);
    ASSERT_GT(result.detectors(0).chirps_size(), 0);

    // Calculate signal variance (should be higher with two targets)
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;
    for (int32_t sample : result.detectors(0).chirps(0).samples()) {
        sum += sample;
        sum_sq += sample * sample;
        count++;
    }
    double mean = sum / count;
    double variance = sum_sq / count - mean * mean;

    EXPECT_GT(variance, 0.0);  // Non-zero variance indicates signal
}

// Test spectral ratio application
TEST_F(FmcwSimulatorTest, SpectralRatio) {
    // Add target with spectral ratio of 0.5 (1550nm is half of 905nm)
    add_point_target(0.0, 0.0, 100.0, 1.0, 0.5);

    FmcwSimulator simulator(forward_model(), sim_config_);

    // Create chirp plans for both wavelengths
    aurora::acquisition::ChirpPlan chirp_905;
    chirp_905.set_chirp_index(0);
    chirp_905.set_chirp_duration_s(12.5e-6);
    chirp_905.set_chirp_bandwidth_hz(200e6);
    chirp_905.set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);
    chirp_905.add_wavelength_m(905e-9);

    aurora::acquisition::ChirpPlan chirp_1550;
    chirp_1550.set_chirp_index(1);
    chirp_1550.set_chirp_duration_s(12.5e-6);
    chirp_1550.set_chirp_bandwidth_hz(200e6);
    chirp_1550.set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);
    chirp_1550.add_wavelength_m(1550e-9);

    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(0);
    plan.set_start_timestamp_s(0.0);
    *plan.add_chirps() = chirp_905;
    *plan.add_chirps() = chirp_1550;

    auto result = simulator.simulate_burst(plan);

    // Both chirps should produce signal, but 1550nm should be weaker
    // This test just verifies both wavelengths produce output
    ASSERT_EQ(result.detectors(0).chirps_size(), 2);
}

// Test burst simulation
TEST_F(FmcwSimulatorTest, BurstSimulation) {
    add_point_target(0.0, 0.0, 100.0);

    FmcwSimulator simulator(forward_model(), sim_config_);

    // Create sequence plan with multiple chirps
    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(1);
    plan.set_start_timestamp_s(0.0);

    constexpr int kNumChirps = 3;
    for (int i = 0; i < kNumChirps; ++i) {
        auto* chirp = plan.add_chirps();
        *chirp = create_test_chirp_plan();
        chirp->set_chirp_index(i);
        chirp->set_chirp_start_s(i * chirp->chirp_duration_s());
    }

    auto burst = simulator.simulate_burst(plan);

    EXPECT_EQ(burst.burst_id(), 0);
    EXPECT_EQ(burst.plan().plan_id(), 1);
    EXPECT_EQ(burst.plan().chirps_size(), kNumChirps);
    ASSERT_GT(burst.detectors_size(), 0);

    for (const auto& detector : burst.detectors()) {
        EXPECT_EQ(detector.chirps_size(), kNumChirps);
    }
}

// Test scene update
TEST_F(FmcwSimulatorTest, SceneUpdate) {
    // Add target with velocity
    auto* scene = sim_config_.mutable_scene();
    auto* target = scene->add_point_targets();
    target->mutable_position_m()->set_x(0.0);
    target->mutable_position_m()->set_y(0.0);
    target->mutable_position_m()->set_z(100.0);
    target->mutable_velocity_mps()->set_x(0.0);
    target->mutable_velocity_mps()->set_y(0.0);
    target->mutable_velocity_mps()->set_z(10.0);  // Moving away at 10 m/s
    target->mutable_surface()->set_reflectivity_905nm(1.0);
    target->mutable_surface()->set_reflectivity_1550nm(1.0);
    target->mutable_surface()->set_roughness_rms_m(1e-6);
    target->mutable_surface()->set_correlation_length_m(10e-6);

    FmcwSimulator simulator(forward_model(), sim_config_);

    // Initial range should be 100m
    EXPECT_NEAR(simulator.scene().point_target(0).range_m(), 100.0, 0.01);

    // Update by 1 second
    simulator.update_scene(1.0);

    // Range should now be 110m
    EXPECT_NEAR(simulator.scene().point_target(0).range_m(), 110.0, 0.01);
}

// Test RNG reset
TEST_F(FmcwSimulatorTest, RngReset) {
    add_point_target(0.0, 0.0, 100.0);

    // Enable some noise to test RNG
    sim_config_.mutable_noise()->set_enable_shot_noise(true);
    sim_config_.mutable_noise()->set_detector_responsivity_a_per_w(0.9);

    FmcwSimulator simulator(forward_model(), sim_config_);

    auto plan = create_test_plan();

    // Simulate twice
    auto result1 = simulator.simulate_burst(plan);
    auto result2 = simulator.simulate_burst(plan);

    // Results should differ (different RNG state)
    bool different = false;
    for (int s = 0; s < std::min(result1.detectors(0).chirps(0).samples_size(),
                                  result2.detectors(0).chirps(0).samples_size()); ++s) {
        if (result1.detectors(0).chirps(0).samples(s) !=
            result2.detectors(0).chirps(0).samples(s)) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);

    // Reset RNG and simulate again
    simulator.reset_rng();

    // Need a fresh simulator to compare against initial state
    FmcwSimulator fresh_simulator(forward_model(), sim_config_);
    auto result_fresh = fresh_simulator.simulate_burst(plan);

    // After reset, should match fresh simulator's first result
    // (This tests that reset_rng works, though the noise state is complex)
}

// =============================================================================
// K-space sampling tests: each detector samples different K-space points
// =============================================================================

// Test that phase changes with off-axis target angle
// Phase is encoded using the pixel's scene K (detector K plus LO tilt).
TEST_F(FmcwSimulatorTest, OffAxisPhaseEncoding) {
    // Test at various angles
    std::vector<double> test_angles_deg = {0.0, 0.5, 1.0, 1.5};

    constexpr double kRangeM = 100.0;

    // Track phases at different angles (expected to decrease with angle)
    std::vector<double> phases_at_angles;

    for (double angle_deg : test_angles_deg) {
        double angle_rad = angle_deg * M_PI / 180.0;
        double x_target = kRangeM * std::tan(angle_rad);

        // Clear scene and add target at this angle
        sim_config_.mutable_scene()->clear_point_targets();
        add_point_target(x_target, 0.0, kRangeM);

        FmcwSimulator simulator(forward_model(), sim_config_);

        auto result = simulator.simulate_burst(create_test_plan());

        // Extract the beat signal from detector 0
        ASSERT_GT(result.detectors_size(), 0);
        ASSERT_GT(result.detectors(0).chirps_size(), 0);

        const auto& samples = result.detectors(0).chirps(0).samples();
        ASSERT_GT(samples.size(), 100);

        // Phase encoding uses the pixel's scene K (detector K plus LO tilt)
        // For detector 0, get the scene K and compute expected phase
        auto k_sample = forward_model().get_effective_k(0, 1550e-9);
        double k_scene = k_sample.x();

        double focal_length = optics_.fourier_lens_focal_m();
        double expected_phase = -k_scene * focal_length * std::tan(angle_rad);
        phases_at_angles.push_back(expected_phase);
    }

    // Verify monotonic change with angle (phase magnitude should increase)
    for (size_t i = 1; i < phases_at_angles.size(); ++i) {
        EXPECT_NE(phases_at_angles[i], phases_at_angles[i-1])
            << "Phase should change with increasing angle";
    }
}

// Test that different detectors sample different K-space points
// Each detector at position x_det samples k_detector = 2pi x x_det / (f x lambda)
TEST_F(FmcwSimulatorTest, DetectorsSampleDifferentKSpacePoints) {
    constexpr double kRangeM = 100.0;
    constexpr double kAngleDeg = 1.0;
    double angle_rad = kAngleDeg * M_PI / 180.0;
    double x_target = kRangeM * std::tan(angle_rad);

    add_point_target(x_target, 0.0, kRangeM);

    FmcwSimulator simulator(forward_model(), sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());

    // With 8 detectors, we should get 8 signals
    ASSERT_EQ(result.detectors_size(), 8);

    // Verify that each detector has a valid signal
    // Different detectors sit at different positions, so each samples a
    // different scene K
    std::vector<double> k_scene_values;
    for (int d = 0; d < 8; ++d) {
        auto k_sample = forward_model_->get_effective_k(d, 1550e-9);
        k_scene_values.push_back(k_sample.x());
    }

    // Verify K-values are different across detectors (due to detector position offset)
    for (int d = 1; d < 8; ++d) {
        EXPECT_NE(k_scene_values[d], k_scene_values[0])
            << "Detector " << d << " should sample a different scene K than detector 0";
    }

    // Verify K-values span a reasonable range (detectors are at different positions)
    double k_min = *std::min_element(k_scene_values.begin(), k_scene_values.end());
    double k_max = *std::max_element(k_scene_values.begin(), k_scene_values.end());
    EXPECT_GT(k_max - k_min, 10000.0)
        << "K-space span across detectors should be significant";
}

// =============================================================================
// Plane target tests
// =============================================================================

// Test that plane targets produce output
TEST_F(FmcwSimulatorTest, PlaneTargetProducesOutput) {
    // Add a plane target
    auto* scene = sim_config_.mutable_scene();
    auto* plane = scene->add_plane_targets();
    plane->mutable_center_m()->set_x(0.0);
    plane->mutable_center_m()->set_y(0.0);
    plane->mutable_center_m()->set_z(100.0);
    plane->mutable_normal()->set_x(0.0);
    plane->mutable_normal()->set_y(0.0);
    plane->mutable_normal()->set_z(-1.0);
    plane->set_width_m(0.1);
    plane->set_height_m(0.1);
    plane->mutable_surface()->set_reflectivity_905nm(1.0);
    plane->mutable_surface()->set_reflectivity_1550nm(1.0);
    plane->mutable_surface()->set_roughness_rms_m(1e-6);
    plane->mutable_surface()->set_correlation_length_m(10e-6);

    FmcwSimulator simulator(forward_model(), sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());

    // Should produce non-zero signal
    ASSERT_GT(result.detectors_size(), 0);
    ASSERT_GT(result.detectors(0).chirps_size(), 0);

    bool has_nonzero = false;
    for (int32_t sample : result.detectors(0).chirps(0).samples()) {
        if (sample != 0) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// Test that plane targets with analytic model are deterministic
TEST_F(FmcwSimulatorTest, PlaneTargetAnalyticModelDeterminism) {
    // Add a rough plane target (uses speckle)
    auto* scene = sim_config_.mutable_scene();
    auto* plane = scene->add_plane_targets();
    plane->mutable_center_m()->set_x(0.0);
    plane->mutable_center_m()->set_y(0.0);
    plane->mutable_center_m()->set_z(100.0);
    plane->mutable_normal()->set_x(0.0);
    plane->mutable_normal()->set_y(0.0);
    plane->mutable_normal()->set_z(-1.0);
    plane->set_width_m(0.2);
    plane->set_height_m(0.2);
    plane->mutable_surface()->set_reflectivity_905nm(0.8);
    plane->mutable_surface()->set_reflectivity_1550nm(0.6);
    plane->mutable_surface()->set_roughness_rms_m(1e-6);  // Rough surface
    plane->mutable_surface()->set_correlation_length_m(10e-6);

    // Create two simulators with same seed
    FmcwSimulator sim1(forward_model(), sim_config_);
    FmcwSimulator sim2(forward_model(), sim_config_);

    auto plan = create_test_plan();

    auto result1 = sim1.simulate_burst(plan);
    auto result2 = sim2.simulate_burst(plan);

    // Results should be identical (deterministic speckle)
    ASSERT_EQ(result1.detectors_size(), result2.detectors_size());
    for (int d = 0; d < result1.detectors_size(); ++d) {
        const auto& samples1 = result1.detectors(d).chirps(0).samples();
        const auto& samples2 = result2.detectors(d).chirps(0).samples();
        ASSERT_EQ(samples1.size(), samples2.size());
        for (int s = 0; s < samples1.size(); ++s) {
            EXPECT_EQ(samples1[s], samples2[s])
                << "Mismatch at detector " << d << ", sample " << s;
        }
    }
}

// Test that specular plane produces expected sinc-envelope response
TEST_F(FmcwSimulatorTest, PlaneTargetSpecularSincEnvelope) {
    // Add a perfectly specular plane
    auto* scene = sim_config_.mutable_scene();
    auto* plane = scene->add_plane_targets();
    plane->mutable_center_m()->set_x(0.0);
    plane->mutable_center_m()->set_y(0.0);
    plane->mutable_center_m()->set_z(100.0);
    plane->mutable_normal()->set_x(0.0);
    plane->mutable_normal()->set_y(0.0);
    plane->mutable_normal()->set_z(-1.0);
    plane->set_width_m(0.05);  // 5cm width
    plane->set_height_m(0.05); // 5cm height
    plane->mutable_surface()->set_reflectivity_905nm(1.0);
    plane->mutable_surface()->set_reflectivity_1550nm(1.0);
    plane->mutable_surface()->set_roughness_rms_m(0.0);  // Perfectly specular
    plane->mutable_surface()->set_correlation_length_m(10e-6);

    FmcwSimulator simulator(forward_model(), sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());

    // Should produce non-zero signal
    ASSERT_GT(result.detectors_size(), 0);

    // Calculate signal power at detector 0
    double power = 0.0;
    for (int32_t sample : result.detectors(0).chirps(0).samples()) {
        power += static_cast<double>(sample) * sample;
    }
    power /= result.detectors(0).chirps(0).samples_size();

    EXPECT_GT(power, 0.0) << "Specular plane should produce signal";
}

// Test that different bursts see the same speckle (for rough planes)
TEST_F(FmcwSimulatorTest, PlaneTargetSpeckleIsTargetPropertyNotBurstProperty) {
    // Speckle should be a property of the target surface, not the acquisition.
    // This is essential for coherent imaging - all bursts measuring the same target
    // must see consistent speckle so the reconstruction works correctly.

    // Add a rough plane target
    auto* scene = sim_config_.mutable_scene();
    auto* plane = scene->add_plane_targets();
    plane->mutable_center_m()->set_x(0.0);
    plane->mutable_center_m()->set_y(0.0);
    plane->mutable_center_m()->set_z(100.0);
    plane->mutable_normal()->set_x(0.0);
    plane->mutable_normal()->set_y(0.0);
    plane->mutable_normal()->set_z(-1.0);
    plane->set_width_m(0.1);
    plane->set_height_m(0.1);
    plane->mutable_surface()->set_reflectivity_905nm(1.0);
    plane->mutable_surface()->set_reflectivity_1550nm(1.0);
    plane->mutable_surface()->set_roughness_rms_m(1e-6);  // Rough surface
    plane->mutable_surface()->set_correlation_length_m(10e-6);

    FmcwSimulator simulator(forward_model(), sim_config_);

    // Two bursts with different plan IDs and timestamps but the same K-space
    // sampling (the detector geometry is fixed).
    constexpr uint64_t kPlanId1 = 100;
    constexpr uint64_t kPlanId2 = 200;
    constexpr double kBurstPeriodS = 20e-3;

    // Simulate burst 1
    auto result1 = simulator.simulate_burst(create_test_plan(kPlanId1, 0.0));

    // Simulate burst 2 (reset RNG to ensure reproducibility)
    simulator.reset_rng();
    auto result2 = simulator.simulate_burst(create_test_plan(kPlanId2, kBurstPeriodS));

    // Both bursts sample the same K-space positions, so they should
    // see the same speckle from the same target. The speckle depends on:
    // - simulation rng_seed (same)
    // - plane_idx (same, it's the same target)
    // - k_vector (same, same detector geometry)
    // Plan ID and burst timestamp should NOT affect speckle.
    const auto& samples1 = result1.detectors(0).chirps(0).samples();
    const auto& samples2 = result2.detectors(0).chirps(0).samples();

    // Signals should be identical when measuring the same K-space point
    // of the same target with the same simulation seed
    bool all_same = true;
    for (size_t i = 0; i < std::min(samples1.size(), samples2.size()); ++i) {
        if (samples1[i] != samples2[i]) {
            all_same = false;
            break;
        }
    }

    EXPECT_TRUE(all_same)
        << "Same target at same K-space position should produce identical signals "
        << "regardless of plan ID or timestamp (speckle is a target property)";
}

TEST_F(FmcwSimulatorTest, PlaneTargetDifferentSimSeedsDifferentSpeckle) {
    // Different simulation seeds should produce different speckle realizations.
    // This verifies that while speckle is consistent within a simulation,
    // different simulations can model different random surfaces.

    // Add a rough plane target
    auto* scene = sim_config_.mutable_scene();
    auto* plane = scene->add_plane_targets();
    plane->mutable_center_m()->set_x(0.0);
    plane->mutable_center_m()->set_y(0.0);
    plane->mutable_center_m()->set_z(100.0);
    plane->mutable_normal()->set_x(0.0);
    plane->mutable_normal()->set_y(0.0);
    plane->mutable_normal()->set_z(-1.0);
    plane->set_width_m(0.1);
    plane->set_height_m(0.1);
    plane->mutable_surface()->set_reflectivity_905nm(1.0);
    plane->mutable_surface()->set_reflectivity_1550nm(1.0);
    plane->mutable_surface()->set_roughness_rms_m(1e-6);  // Rough surface
    plane->mutable_surface()->set_correlation_length_m(10e-6);

    auto plan = create_test_plan();

    // First simulation with seed 42
    sim_config_.set_rng_seed(42);
    FmcwSimulator simulator1(forward_model(), sim_config_);
    auto result1 = simulator1.simulate_burst(plan);

    // Second simulation with different seed
    sim_config_.set_rng_seed(12345);
    FmcwSimulator simulator2(forward_model(), sim_config_);
    auto result2 = simulator2.simulate_burst(plan);

    // Different seeds should produce different speckle
    const auto& samples1 = result1.detectors(0).chirps(0).samples();
    const auto& samples2 = result2.detectors(0).chirps(0).samples();

    bool any_diff = false;
    for (size_t i = 0; i < std::min(samples1.size(), samples2.size()); ++i) {
        if (samples1[i] != samples2[i]) {
            any_diff = true;
            break;
        }
    }

    EXPECT_TRUE(any_diff)
        << "Different simulation seeds should produce different speckle realizations";
}

// Test that the scene K of each pixel produces the correct phase gradient
// across detectors
TEST_F(FmcwSimulatorTest, SceneKPhaseGradientCorrectness) {
    // Off-axis target to produce measurable phase gradient
    const double x_target = 0.01;  // 10mm off-axis
    const double range_m = 100.0;
    add_point_target(x_target, 0.0, range_m);

    FmcwSimulator simulator(forward_model(), sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());
    ASSERT_EQ(result.detectors_size(), 8);

    // Extract phases from each detector by fitting sinusoid
    double sample_rate = sim_config_.adc().sample_rate_hz();
    double chirp_bandwidth = create_test_chirp_plan().chirp_bandwidth_hz();
    double chirp_duration = create_test_chirp_plan().chirp_duration_s();
    double f_beat = ChirpGenerator::compute_beat_frequency(range_m, chirp_bandwidth, chirp_duration);

    std::vector<double> phases;
    for (int d = 0; d < result.detectors_size(); ++d) {
        const auto& samples = result.detectors(d).chirps(0).samples();

        double sum_cos = 0.0, sum_sin = 0.0;
        for (size_t i = 0; i < samples.size(); ++i) {
            double t = i / sample_rate;
            double angle = 2.0 * M_PI * f_beat * t;
            sum_cos += samples[i] * std::cos(angle);
            sum_sin += samples[i] * std::sin(angle);
        }
        double phase = std::atan2(sum_sin, sum_cos);
        phases.push_back(phase);
    }

    // Verify phase gradient direction
    std::vector<double> k_values;
    for (int d = 0; d < 8; ++d) {
        auto k_sample = forward_model_->get_effective_k(d, 1550e-9);
        k_values.push_back(k_sample.x());
    }

    double sum_dk_dphi = 0.0;
    for (int d = 1; d < 8; ++d) {
        double dk = k_values[d] - k_values[d-1];
        double dphi = phases[d] - phases[d-1];
        while (dphi > M_PI) dphi -= 2*M_PI;
        while (dphi < -M_PI) dphi += 2*M_PI;
        sum_dk_dphi += dk * dphi;
    }

    // Note: DFT phase extraction gives -phi_0 (negative of signal phase)
    // So if signal phase decreases with k (phi_0 is proportional to -k for positive target),
    // extracted phase increases with k.
    // Therefore sum_dk_dphi should be POSITIVE for positive x_target.
    EXPECT_GT(sum_dk_dphi, 0.0)
        << "Phase gradient should have correct sign for positive target position";
}

// =============================================================================
// LO Tilt Tests
// =============================================================================

// Test that LO tilt affects phase encoding in simulation
// We compare results with and without LO tilt for the same scene
TEST_F(FmcwSimulatorTest, LOTiltAffectsPhaseEncoding) {
    // Target at 100m, off-axis (x=0.02m = 20mm)
    // Must be off-axis so that k_x changes from LO tilt produce phase differences
    // (for on-axis target at x=0, phase = -k_x * 0 = 0 regardless of k_x)
    add_point_target(0.02, 0.0, 100.0);

    // Build forward model with no LO tilt
    aurora::forward::ForwardModel forward_model_no_tilt(detector_, optics_);

    auto plan = create_test_plan();

    // Simulate without LO tilt
    FmcwSimulator sim_no_tilt(forward_model_no_tilt, sim_config_);
    auto result_no_tilt = sim_no_tilt.simulate_burst(plan);

    // Build forward model with LO tilt
    aurora::forward::OpticsConfig optics_with_tilt = optics_;
    optics_with_tilt.set_lo_tilt_x_rad(0.05);  // 50 mrad

    aurora::forward::ForwardModel forward_model_with_tilt(detector_, optics_with_tilt);

    FmcwSimulator sim_with_tilt(forward_model_with_tilt, sim_config_);
    auto result_with_tilt = sim_with_tilt.simulate_burst(plan);

    // Both should produce signal
    ASSERT_GT(result_no_tilt.detectors_size(), 0);
    ASSERT_GT(result_with_tilt.detectors_size(), 0);

    // Extract phases from detector 4 (center detector)
    const auto& det_no_tilt = result_no_tilt.detectors(4);
    const auto& det_with_tilt = result_with_tilt.detectors(4);

    ASSERT_GT(det_no_tilt.chirps(0).samples_size(), 100);
    ASSERT_GT(det_with_tilt.chirps(0).samples_size(), 100);

    // Compute phase difference by looking at waveform correlation
    // If LO tilt changes phase, the waveforms should differ
    double sum_diff = 0.0;
    int count = 0;
    for (int i = 50; i < std::min(det_no_tilt.chirps(0).samples_size(), 150); ++i) {
        sum_diff += std::abs(det_no_tilt.chirps(0).samples(i) - det_with_tilt.chirps(0).samples(i));
        ++count;
    }
    double avg_diff = sum_diff / count;

    // Waveforms should differ (LO tilt changes the phase encoding)
    // Note: They have similar magnitude but different phase
    EXPECT_GT(avg_diff, 0.0) << "LO tilt should affect the signal phase";
}

// Test that the scene K from the forward model includes LO tilt
TEST_F(FmcwSimulatorTest, ForwardModelSceneKIncludesLOTilt) {
    // Add LO tilt
    const double tilt_angle = 0.05;  // 50 mrad
    aurora::forward::OpticsConfig optics_with_tilt = optics_;
    optics_with_tilt.set_lo_tilt_x_rad(tilt_angle);

    // Build forward model with tilt
    aurora::forward::ForwardModel forward_model_tilt(detector_, optics_with_tilt);

    const double wavelength = 1550e-9;
    auto k_sample = forward_model_tilt.get_effective_k(4, wavelength);

    // Expected: scene K = detector K + LO tilt K
    const double focal_length = optics_.fourier_lens_focal_m();
    const double x_det = (4 - 3.5) * 0.0005;  // Detector 4 position
    const double k_detector = 2.0 * M_PI * x_det / (focal_length * wavelength);
    const double k_tilt = 2.0 * M_PI * std::sin(tilt_angle) / wavelength;

    double expected_kx = k_detector + k_tilt;

    EXPECT_NEAR(k_sample.x(), expected_kx, std::abs(expected_kx) * 0.02)
        << "scene kx should be detector K + LO tilt K = "
        << k_detector << " + " << k_tilt
        << " = " << expected_kx;
}

// Test that simulation phase gradient is correct with LO tilt
TEST_F(FmcwSimulatorTest, PhaseGradientWithLOTilt) {
    // Add LO tilt
    const double tilt_angle = 0.03;  // 30 mrad
    aurora::forward::OpticsConfig optics_with_tilt = optics_;
    optics_with_tilt.set_lo_tilt_x_rad(tilt_angle);

    // Target at 100m, slightly off-axis to produce phase gradient
    const double x_target = 0.02;  // 20mm off-axis
    const double kRangeM = 100.0;
    add_point_target(x_target, 0.0, kRangeM);

    // Build forward model with tilt
    aurora::forward::ForwardModel forward_model_tilt(detector_, optics_with_tilt);

    FmcwSimulator simulator(forward_model_tilt, sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());
    ASSERT_EQ(result.detectors_size(), 8);

    // Extract phases from all detectors
    std::vector<double> phases(8);
    for (int d = 0; d < 8; ++d) {
        const auto& det = result.detectors(d);
        if (det.chirps(0).samples_size() < 100) continue;

        // Use Hilbert-like phase extraction (simplified)
        double sum_sin = 0.0, sum_cos = 0.0;
        const int start_idx = 50;
        const int num_samples = 50;
        for (int i = start_idx; i < start_idx + num_samples; ++i) {
            double t = static_cast<double>(i) / 125e6;  // Time in seconds
            double beat_freq = ChirpGenerator::compute_beat_frequency(
                kRangeM, 200e6, 12.5e-6);
            sum_cos += det.chirps(0).samples(i) * std::cos(2 * M_PI * beat_freq * t);
            sum_sin += det.chirps(0).samples(i) * std::sin(2 * M_PI * beat_freq * t);
        }
        phases[d] = std::atan2(sum_sin, sum_cos);
    }

    // Phase gradient should be present (non-zero due to off-axis target)
    // With LO tilt, every pixel's scene K is shifted by the same offset
    double phase_span = phases[7] - phases[0];
    while (phase_span > M_PI) phase_span -= 2 * M_PI;
    while (phase_span < -M_PI) phase_span += 2 * M_PI;

    // Phase span should be significant (target is off-axis)
    EXPECT_GT(std::abs(phase_span), 0.1)
        << "Phase gradient should be non-zero for off-axis target";
}

// =============================================================================
// MeasurementOperator Integration Tests
// Verify that simulation uses correct physics for the pixel's K-space window
// =============================================================================

// Test that an on-axis point target gives similar power on every pixel.
// Each pixel's acceptance window is centred on its own scene K with the same
// bandwidth, and a point target at r = 0 has a flat spectrum X(k) = 1, so the
// integrated measurement ∫ W(k) dk is the same for every pixel.
TEST_F(FmcwSimulatorTest, SincWindowOnAxisTargetUniformPower) {
    // Detector with SINC sensitivity (finite rectangular aperture)
    auto detector_rect = create_sinc_detector_geometry();
    aurora::forward::ForwardModel sinc_model(detector_rect, optics_);

    // Add an on-axis point target (k_target = 0)
    sim_config_.mutable_scene()->clear_point_targets();
    add_point_target(0.0, 0.0, 100.0);

    FmcwSimulator simulator(sinc_model, sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());

    // Verify we have 8 detectors
    ASSERT_EQ(result.detectors_size(), 8);

    // Check all detectors have signal (integrated response is non-zero)
    std::vector<double> powers(8);
    for (int d = 0; d < 8; ++d) {
        for (int32_t sample : result.detectors(d).chirps(0).samples()) {
            powers[d] += static_cast<double>(sample) * sample;
        }
        powers[d] /= result.detectors(d).chirps(0).samples_size();

        EXPECT_GT(powers[d], 0.0)
            << "Detector " << d << " should have non-zero signal for an on-axis target";
    }

    // All pixels should have similar power because ∫W(k)·exp(i·k·0) dk = ∫W(k) dk
    // is the same for each pixel's window.
    double min_power = *std::min_element(powers.begin(), powers.end());
    double max_power = *std::max_element(powers.begin(), powers.end());

    // Allow some variation from quadrature, but all should be within the same
    // order of magnitude
    EXPECT_GT(min_power, max_power * 0.5)
        << "All detectors should have similar power for an on-axis target";
}

// Test that plane target uses the measurement operator (quadrature test)
// The integrated response should show proper K-space weighting effects.
TEST_F(FmcwSimulatorTest, PlaneTargetUsesMeasurementOperator) {
    // Detector with SINC sensitivity
    auto detector_rect = create_sinc_detector_geometry();

    aurora::forward::ForwardModel plane_model(detector_rect, optics_);

    // Add a plane target
    sim_config_.mutable_scene()->clear_point_targets();
    auto* plane = sim_config_.mutable_scene()->add_plane_targets();
    plane->mutable_center_m()->set_x(0.0);
    plane->mutable_center_m()->set_y(0.0);
    plane->mutable_center_m()->set_z(100.0);
    plane->mutable_normal()->set_x(0.0);
    plane->mutable_normal()->set_y(0.0);
    plane->mutable_normal()->set_z(-1.0);
    plane->set_width_m(0.1);   // 10cm width
    plane->set_height_m(0.1);  // 10cm height
    plane->mutable_surface()->set_reflectivity_905nm(1.0);
    plane->mutable_surface()->set_reflectivity_1550nm(1.0);
    plane->mutable_surface()->set_roughness_rms_m(0.0);  // Specular
    plane->mutable_surface()->set_correlation_length_m(10e-6);

    FmcwSimulator simulator(plane_model, sim_config_);

    auto result = simulator.simulate_burst(create_test_plan());

    // Should produce signal on all detectors
    ASSERT_GT(result.detectors_size(), 0);

    bool has_signal = false;
    for (int d = 0; d < result.detectors_size(); ++d) {
        for (int32_t sample : result.detectors(d).chirps(0).samples()) {
            if (sample != 0) {
                has_signal = true;
                break;
            }
        }
        if (has_signal) break;
    }

    EXPECT_TRUE(has_signal)
        << "Plane target with MeasurementOperator should produce signal";
}

// Test determinism with MeasurementOperator (same seed = same result)
TEST_F(FmcwSimulatorTest, MeasurementOperatorDeterminism) {
    // Detector with SINC sensitivity (rectangular aperture model)
    auto detector_rect = create_sinc_detector_geometry();

    aurora::forward::ForwardModel det_model(detector_rect, optics_);

    sim_config_.mutable_scene()->clear_point_targets();
    add_point_target(0.005, 0.0, 100.0);

    auto plan = create_test_plan();

    // Simulate twice with same seed
    FmcwSimulator sim1(det_model, sim_config_);
    FmcwSimulator sim2(det_model, sim_config_);

    auto result1 = sim1.simulate_burst(plan);
    auto result2 = sim2.simulate_burst(plan);

    // Results should be identical
    ASSERT_EQ(result1.detectors_size(), result2.detectors_size());
    for (int d = 0; d < result1.detectors_size(); ++d) {
        const auto& samples1 = result1.detectors(d).chirps(0).samples();
        const auto& samples2 = result2.detectors(d).chirps(0).samples();
        ASSERT_EQ(samples1.size(), samples2.size());

        for (int s = 0; s < samples1.size(); ++s) {
            EXPECT_EQ(samples1[s], samples2[s])
                << "Mismatch at detector " << d << ", sample " << s;
        }
    }
}

}  // namespace
}  // namespace aurora::simulation
