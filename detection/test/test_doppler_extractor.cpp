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

/**
 * @file test_doppler_extractor.cpp
 * @brief Unit tests for DopplerExtractor velocity extraction.
 *
 * Tests verify:
 * - Velocity extraction accuracy at high SNR
 * - Correct handling of approaching/receding targets
 * - Velocity ambiguity detection
 * - Confidence computation
 * - Error handling for invalid inputs
 */

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <vector>

#include "detection/detection_error.h"
#include "detection/doppler_extractor.h"

namespace aurora::detection {
namespace {

// Test parameters
constexpr double kWavelength1550nm = 1550e-9;  // meters
constexpr double kWavelength905nm = 905e-9;    // meters
constexpr double kChirpDuration = 12.5e-6;     // seconds
constexpr int kNumDetectors = 8;

// Create test configuration
DetectionConfig create_test_config() {
    DetectionConfig config;
    config.set_enable_doppler(true);
    config.set_min_velocity_mps(0.1);
    return config;
}

// Generate complex samples with given phase
std::complex<double> sample_with_phase(double phase, double magnitude = 1.0) {
    return magnitude * std::exp(std::complex<double>(0.0, phase));
}

// Generate up/down samples for a given velocity
// Up-chirp phase: φ_up = φ_base + 2πvT/λ (Doppler adds)
// Down-chirp phase: φ_down = φ_base - 2πvT/λ (Doppler subtracts)
// Phase difference: Δφ = 4πvT/λ
void generate_samples_for_velocity(
    double velocity_mps,
    double wavelength_m,
    int num_samples,
    std::vector<std::complex<double>>& up_samples,
    std::vector<std::complex<double>>& down_samples,
    std::mt19937& rng,
    double phase_noise_sigma = 0.0,
    double chirp_duration_s = kChirpDuration) {

    up_samples.clear();
    down_samples.clear();

    // Phase difference for this velocity: Δφ = 4πvT/λ
    double phase_diff = 4.0 * M_PI * velocity_mps * chirp_duration_s / wavelength_m;

    // Only create distribution when sigma > 0 (MSVC validates sigma at construction)
    std::unique_ptr<std::normal_distribution<double>> noise;
    if (phase_noise_sigma > 0.0) {
        noise = std::make_unique<std::normal_distribution<double>>(0.0, phase_noise_sigma);
    }

    for (int i = 0; i < num_samples; ++i) {
        // Base phase varies per detector (simulates different K-positions)
        double base_phase = 0.1 * i;

        // Add half the phase diff to up, subtract half from down
        // So difference is phase_diff
        double up_phase = base_phase + phase_diff / 2.0;
        double down_phase = base_phase - phase_diff / 2.0;

        // Add noise if specified
        if (noise) {
            up_phase += (*noise)(rng);
            down_phase += (*noise)(rng);
        }

        up_samples.push_back(sample_with_phase(up_phase));
        down_samples.push_back(sample_with_phase(down_phase));
    }
}

class DopplerExtractorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_ = create_test_config();
        rng_.seed(42);
    }

    DetectionConfig config_;
    std::mt19937 rng_;
};

// Test: Basic construction
TEST_F(DopplerExtractorTest, Construction) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    EXPECT_TRUE(analyzer.is_enabled());
    EXPECT_DOUBLE_EQ(analyzer.wavelength_m(), kWavelength1550nm);
    EXPECT_DOUBLE_EQ(analyzer.chirp_duration_s(), kChirpDuration);
    EXPECT_GT(analyzer.max_unambiguous_velocity(), 0.0);
    EXPECT_GT(analyzer.velocity_resolution(), 0.0);
}

// Test: Max unambiguous velocity computation
TEST_F(DopplerExtractorTest, MaxUnambiguousVelocity) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // v_max = λ / (4 × T_chirp)
    double expected = kWavelength1550nm / (4.0 * kChirpDuration);
    EXPECT_DOUBLE_EQ(analyzer.max_unambiguous_velocity(), expected);

    // For 1550nm, 12.5μs: v_max ≈ 31 mm/s = 0.031 m/s
    EXPECT_NEAR(analyzer.max_unambiguous_velocity(), 0.031, 0.001);
}

// Test: Zero velocity extraction
TEST_F(DopplerExtractorTest, ZeroVelocity) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    std::vector<std::complex<double>> up_samples, down_samples;
    generate_samples_for_velocity(0.0, kWavelength1550nm, kNumDetectors,
                                  up_samples, down_samples, rng_);

    std::vector<double> weights(kNumDetectors, 1.0);
    auto result = analyzer.extract_velocity(up_samples, down_samples, weights);

    EXPECT_NEAR(result.velocity_mps(), 0.0, 0.1);
    EXPECT_GT(result.confidence(), 0.5);
    EXPECT_FALSE(result.ambiguous());
}

// Test: Approaching target (positive velocity)
TEST_F(DopplerExtractorTest, ApproachingTarget) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // Use velocity within unambiguous range (v_max ≈ 0.031 m/s)
    double target_velocity = 0.01;  // 10 mm/s approaching

    std::vector<std::complex<double>> up_samples, down_samples;
    generate_samples_for_velocity(target_velocity, kWavelength1550nm,
                                  kNumDetectors, up_samples, down_samples, rng_,
                                  0.0, kChirpDuration);

    auto result = analyzer.extract_velocity(up_samples, down_samples);

    EXPECT_NEAR(result.velocity_mps(), target_velocity, 0.001);
    EXPECT_GT(result.confidence(), 0.5);
    EXPECT_FALSE(result.ambiguous());
}

// Test: Receding target (negative velocity)
TEST_F(DopplerExtractorTest, RecedingTarget) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // Use velocity within unambiguous range (v_max ≈ 0.031 m/s)
    double target_velocity = -0.015;  // -15 mm/s receding

    std::vector<std::complex<double>> up_samples, down_samples;
    generate_samples_for_velocity(target_velocity, kWavelength1550nm,
                                  kNumDetectors, up_samples, down_samples, rng_,
                                  0.0, kChirpDuration);

    auto result = analyzer.extract_velocity(up_samples, down_samples);

    EXPECT_NEAR(result.velocity_mps(), target_velocity, 0.001);
    EXPECT_GT(result.confidence(), 0.5);
}

// Test: Velocity near ambiguity threshold is flagged
TEST_F(DopplerExtractorTest, VelocityNearAmbiguityThreshold) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // Use velocity at 95% of max unambiguous
    double target_velocity = 0.95 * analyzer.max_unambiguous_velocity();

    std::vector<std::complex<double>> up_samples, down_samples;
    generate_samples_for_velocity(target_velocity, kWavelength1550nm,
                                  kNumDetectors, up_samples, down_samples, rng_,
                                  0.0, kChirpDuration);

    auto result = analyzer.extract_velocity(up_samples, down_samples);

    EXPECT_TRUE(result.ambiguous());
}

// Test: Different wavelengths have different max velocities
TEST_F(DopplerExtractorTest, WavelengthAffectsMaxVelocity) {
    DopplerExtractor analyzer_1550(config_, kWavelength1550nm, kChirpDuration);
    DopplerExtractor analyzer_905(config_, kWavelength905nm, kChirpDuration);

    // Shorter wavelength = lower max unambiguous velocity
    EXPECT_LT(analyzer_905.max_unambiguous_velocity(),
              analyzer_1550.max_unambiguous_velocity());

    // Ratio should match wavelength ratio
    double ratio = analyzer_1550.max_unambiguous_velocity() /
                   analyzer_905.max_unambiguous_velocity();
    EXPECT_NEAR(ratio, kWavelength1550nm / kWavelength905nm, 0.001);
}

// Test: Phase to velocity conversion
TEST_F(DopplerExtractorTest, PhaseToVelocityConversion) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // Phase diff of π should give v = λ / (4T) = v_max
    // This is the max unambiguous velocity
    double phase_pi = M_PI;
    double expected_v = kWavelength1550nm / (4.0 * kChirpDuration);
    EXPECT_NEAR(analyzer.phase_to_velocity(phase_pi), expected_v, 1e-10);
    EXPECT_DOUBLE_EQ(analyzer.phase_to_velocity(phase_pi),
                     analyzer.max_unambiguous_velocity());

    // Verify round-trip with a velocity within unambiguous range
    double velocity = 0.01;  // 10 mm/s
    double phase = analyzer.velocity_to_phase(velocity);
    double recovered = analyzer.phase_to_velocity(phase);
    EXPECT_NEAR(recovered, velocity, 1e-10);
}

// Test: Compute phase difference directly
TEST_F(DopplerExtractorTest, ComputePhaseDifference) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // Up sample at phase 0.5, down at phase 0.3
    auto up = sample_with_phase(0.5);
    auto down = sample_with_phase(0.3);

    double diff = analyzer.compute_phase_difference(up, down);
    EXPECT_NEAR(diff, 0.2, 1e-10);

    // Opposite order should give negative
    diff = analyzer.compute_phase_difference(down, up);
    EXPECT_NEAR(diff, -0.2, 1e-10);
}

// Test: Weighted extraction
TEST_F(DopplerExtractorTest, WeightedExtraction) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // Use velocity within unambiguous range
    double target_velocity = 0.005;  // 5 mm/s

    // Generate samples with one outlier
    std::vector<std::complex<double>> up_samples, down_samples;
    generate_samples_for_velocity(target_velocity, kWavelength1550nm,
                                  kNumDetectors, up_samples, down_samples, rng_,
                                  0.0, kChirpDuration);

    // Corrupt one sample
    up_samples[3] = sample_with_phase(2.5);  // Outlier

    // Uniform weights
    std::vector<double> uniform_weights(kNumDetectors, 1.0);
    auto result_uniform = analyzer.extract_velocity(up_samples, down_samples,
                                                     uniform_weights);

    // Down-weight the outlier
    std::vector<double> adjusted_weights(kNumDetectors, 1.0);
    adjusted_weights[3] = 0.01;
    auto result_weighted = analyzer.extract_velocity(up_samples, down_samples,
                                                      adjusted_weights);

    // Weighted result should be closer to true velocity
    double error_uniform = std::abs(result_uniform.velocity_mps() - target_velocity);
    double error_weighted = std::abs(result_weighted.velocity_mps() - target_velocity);
    EXPECT_LT(error_weighted, error_uniform);
}

// Test: Noisy measurements reduce confidence
TEST_F(DopplerExtractorTest, NoiseReducesConfidence) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // Use velocity within unambiguous range
    double target_velocity = 0.008;  // 8 mm/s

    // Clean samples
    std::vector<std::complex<double>> up_clean, down_clean;
    generate_samples_for_velocity(target_velocity, kWavelength1550nm,
                                  kNumDetectors, up_clean, down_clean, rng_,
                                  0.0, kChirpDuration);

    // Noisy samples (0.3 rad noise)
    std::vector<std::complex<double>> up_noisy, down_noisy;
    generate_samples_for_velocity(target_velocity, kWavelength1550nm,
                                  kNumDetectors, up_noisy, down_noisy, rng_,
                                  0.3, kChirpDuration);

    auto result_clean = analyzer.extract_velocity(up_clean, down_clean);
    auto result_noisy = analyzer.extract_velocity(up_noisy, down_noisy);

    // Clean should have higher confidence
    EXPECT_GT(result_clean.confidence(), result_noisy.confidence());
}

// Test: Empty input throws
TEST_F(DopplerExtractorTest, EmptyInputThrows) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    std::vector<std::complex<double>> empty;
    std::vector<std::complex<double>> non_empty(8);
    std::vector<double> weights(8, 1.0);

    EXPECT_THROW({
        analyzer.extract_velocity(empty, non_empty, weights);
    }, DetectionError);
}

// Test: Mismatched sizes throw
TEST_F(DopplerExtractorTest, MismatchedSizesThrows) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    std::vector<std::complex<double>> up(8);
    std::vector<std::complex<double>> down(6);  // Different size
    std::vector<double> weights(8, 1.0);

    EXPECT_THROW({
        analyzer.extract_velocity(up, down, weights);
    }, DetectionError);
}

// Test: Zero weights throw
TEST_F(DopplerExtractorTest, ZeroWeightsThrows) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    std::vector<std::complex<double>> up(8);
    std::vector<std::complex<double>> down(8);
    std::vector<double> weights(8, 0.0);  // All zero

    EXPECT_THROW({
        analyzer.extract_velocity(up, down, weights);
    }, DetectionError);
}

// Test: Negative weights throw
TEST_F(DopplerExtractorTest, NegativeWeightsThrows) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    std::vector<std::complex<double>> up(8);
    std::vector<std::complex<double>> down(8);
    std::vector<double> weights = {1.0, 1.0, 1.0, -0.5, 1.0, 1.0, 1.0, 1.0};

    EXPECT_THROW({
        analyzer.extract_velocity(up, down, weights);
    }, DetectionError);
}

// Test: Invalid wavelength throws
TEST_F(DopplerExtractorTest, InvalidWavelengthThrows) {
    EXPECT_THROW({
        DopplerExtractor analyzer(config_, 0.0, kChirpDuration);
    }, DetectionError);

    EXPECT_THROW({
        DopplerExtractor analyzer(config_, -1550e-9, kChirpDuration);
    }, DetectionError);
}

// Test: Invalid chirp duration throws
TEST_F(DopplerExtractorTest, InvalidChirpDurationThrows) {
    EXPECT_THROW({
        DopplerExtractor analyzer(config_, kWavelength1550nm, 0.0);
    }, DetectionError);

    EXPECT_THROW({
        DopplerExtractor analyzer(config_, kWavelength1550nm, -1e-6);
    }, DetectionError);
}

// Test: Result contains max unambiguous velocity
TEST_F(DopplerExtractorTest, ResultContainsMaxVelocity) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    std::vector<std::complex<double>> up_samples, down_samples;
    generate_samples_for_velocity(0.005, kWavelength1550nm, kNumDetectors,
                                  up_samples, down_samples, rng_,
                                  0.0, kChirpDuration);

    auto result = analyzer.extract_velocity(up_samples, down_samples);

    EXPECT_DOUBLE_EQ(result.max_unambiguous_velocity_mps(),
                     analyzer.max_unambiguous_velocity());
}

// Test: Uniform weights convenience method
TEST_F(DopplerExtractorTest, UniformWeightsConvenience) {
    DopplerExtractor analyzer(config_, kWavelength1550nm, kChirpDuration);

    // Use velocity within unambiguous range
    double target_velocity = 0.007;  // 7 mm/s

    std::vector<std::complex<double>> up_samples, down_samples;
    generate_samples_for_velocity(target_velocity, kWavelength1550nm,
                                  kNumDetectors, up_samples, down_samples, rng_,
                                  0.0, kChirpDuration);

    // Compare convenience method with explicit uniform weights
    auto result1 = analyzer.extract_velocity(up_samples, down_samples);

    std::vector<double> uniform_weights(kNumDetectors, 1.0);
    auto result2 = analyzer.extract_velocity(up_samples, down_samples,
                                              uniform_weights);

    EXPECT_DOUBLE_EQ(result1.velocity_mps(), result2.velocity_mps());
    EXPECT_DOUBLE_EQ(result1.confidence(), result2.confidence());
    EXPECT_EQ(result1.ambiguous(), result2.ambiguous());
}

// Test: Disabled Doppler in config
TEST_F(DopplerExtractorTest, DisabledDopplerConfig) {
    DetectionConfig disabled_config;
    disabled_config.set_enable_doppler(false);

    DopplerExtractor analyzer(disabled_config, kWavelength1550nm, kChirpDuration);

    EXPECT_FALSE(analyzer.is_enabled());

    // Should still work (caller decides whether to call)
    // Use velocity within unambiguous range
    std::vector<std::complex<double>> up_samples, down_samples;
    generate_samples_for_velocity(0.005, kWavelength1550nm, kNumDetectors,
                                  up_samples, down_samples, rng_,
                                  0.0, kChirpDuration);

    auto result = analyzer.extract_velocity(up_samples, down_samples);
    EXPECT_NEAR(result.velocity_mps(), 0.005, 0.0001);
}

}  // namespace
}  // namespace aurora::detection
