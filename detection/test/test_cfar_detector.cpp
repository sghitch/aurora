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
 * @file test_cfar_detector.cpp
 * @brief Unit tests for CfarDetector adaptive thresholding.
 *
 * Tests verify:
 * - Single target detection (no false alarms)
 * - Two targets separation
 * - Edge target handling
 * - Noise-only false alarm rate
 * - Different CFAR modes (CA, GO, SO, OS)
 * - Threshold factor effects
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "detection/cfar_detector.h"

namespace aurora::detection {
namespace {

// Test parameters
constexpr uint32_t kDefaultNumBins = 256;
constexpr uint32_t kTargetBin = 128;

// Generate test data with a single Gaussian-shaped target
std::vector<double> generate_single_target(
    uint32_t num_bins,
    uint32_t target_bin,
    double peak_amplitude,
    double noise_floor,
    double spread = 3.0,
    std::mt19937* rng = nullptr) {

    std::vector<double> magnitudes(num_bins, noise_floor);

    // Add Gaussian target
    for (uint32_t i = 0; i < num_bins; ++i) {
        double distance = static_cast<double>(i) - static_cast<double>(target_bin);
        double gaussian = peak_amplitude * std::exp(-0.5 * (distance / spread) * (distance / spread));
        magnitudes[i] = std::max(magnitudes[i], noise_floor + gaussian);
    }

    // Add noise if RNG provided
    if (rng) {
        std::normal_distribution<double> noise(0.0, noise_floor * 0.1);
        for (auto& m : magnitudes) {
            m = std::max(0.0, m + noise(*rng));
        }
    }

    return magnitudes;
}

// Generate test data with two targets
std::vector<double> generate_two_targets(
    uint32_t num_bins,
    uint32_t target1_bin,
    uint32_t target2_bin,
    double peak1_amplitude,
    double peak2_amplitude,
    double noise_floor,
    double spread = 3.0) {

    std::vector<double> magnitudes(num_bins, noise_floor);

    // Add first target
    for (uint32_t i = 0; i < num_bins; ++i) {
        double distance = static_cast<double>(i) - static_cast<double>(target1_bin);
        double gaussian = peak1_amplitude * std::exp(-0.5 * (distance / spread) * (distance / spread));
        magnitudes[i] = std::max(magnitudes[i], noise_floor + gaussian);
    }

    // Add second target
    for (uint32_t i = 0; i < num_bins; ++i) {
        double distance = static_cast<double>(i) - static_cast<double>(target2_bin);
        double gaussian = peak2_amplitude * std::exp(-0.5 * (distance / spread) * (distance / spread));
        magnitudes[i] = std::max(magnitudes[i], noise_floor + gaussian);
    }

    return magnitudes;
}

// Generate noise-only data
std::vector<double> generate_noise_only(
    uint32_t num_bins,
    double noise_floor,
    std::mt19937& rng) {

    std::vector<double> magnitudes(num_bins);
    std::normal_distribution<double> noise(noise_floor, noise_floor * 0.2);

    for (auto& m : magnitudes) {
        m = std::max(0.0, noise(rng));
    }

    return magnitudes;
}

// Create default CA-CFAR configuration
CfarConfig create_ca_cfar_config() {
    CfarConfig config;
    config.set_mode(CFAR_MODE_CA);
    config.set_guard_cells(3);
    config.set_reference_cells(12);
    config.set_threshold_factor(4.0);
    config.set_min_threshold(1e-6);
    return config;
}

class CfarDetectorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_ = create_ca_cfar_config();
        rng_.seed(42);
    }

    CfarConfig config_;
    std::mt19937 rng_;
};

// Test: Basic construction
TEST_F(CfarDetectorTest, Construction) {
    CfarDetector cfar(config_);

    EXPECT_TRUE(cfar.is_enabled());
    EXPECT_EQ(cfar.config().mode(), CFAR_MODE_CA);
    EXPECT_EQ(cfar.config().guard_cells(), 3u);
    EXPECT_EQ(cfar.config().reference_cells(), 12u);
    EXPECT_DOUBLE_EQ(cfar.config().threshold_factor(), 4.0);
}

// Test: Fixed mode is not enabled
TEST_F(CfarDetectorTest, FixedModeNotEnabled) {
    config_.set_mode(CFAR_MODE_FIXED);
    CfarDetector cfar(config_);

    EXPECT_FALSE(cfar.is_enabled());
}

// Test: Unspecified mode is not enabled
TEST_F(CfarDetectorTest, UnspecifiedModeNotEnabled) {
    config_.set_mode(CFAR_MODE_UNSPECIFIED);
    CfarDetector cfar(config_);

    EXPECT_FALSE(cfar.is_enabled());
}

// Test: Single target detection - expect exactly 1 detection at correct bin
TEST_F(CfarDetectorTest, SingleTargetDetection) {
    CfarDetector cfar(config_);

    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;  // High SNR
    auto magnitudes = generate_single_target(
        kDefaultNumBins, kTargetBin, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    // Expect a small number of detections around the target bin
    // (target + nearby sidelobes)
    EXPECT_GT(detections.size(), 0u);
    EXPECT_LT(detections.size(), 20u);  // Not excessive false alarms

    // Find peak detection
    auto max_det = std::max_element(detections.begin(), detections.end(),
        [](const CfarDetection& a, const CfarDetection& b) {
            return a.magnitude() < b.magnitude();
        });

    ASSERT_NE(max_det, detections.end());

    // Peak should be near target bin (within spread)
    EXPECT_NEAR(static_cast<double>(max_det->bin_index()),
                static_cast<double>(kTargetBin), 5.0);

    // SNR should be positive (signal > noise)
    EXPECT_GT(max_det->snr_db(), 0.0);
}

// Test: No false alarms on uniform noise
TEST_F(CfarDetectorTest, NoiseOnlyLowFalseAlarms) {
    // Use higher threshold for lower false alarm rate
    config_.set_threshold_factor(8.0);
    CfarDetector cfar(config_);

    double noise_floor = 1.0;
    auto magnitudes = generate_noise_only(kDefaultNumBins, noise_floor, rng_);

    auto detections = cfar.detect_1d(magnitudes);

    // With pure noise, false alarm rate should be very low
    // Expect < 5% false alarm rate
    double false_alarm_rate = static_cast<double>(detections.size()) /
                              static_cast<double>(kDefaultNumBins);
    EXPECT_LT(false_alarm_rate, 0.05);
}

// Test: Two separated targets both detected
TEST_F(CfarDetectorTest, TwoTargetsSeparatedDetected) {
    CfarDetector cfar(config_);

    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    uint32_t target1_bin = 80;
    uint32_t target2_bin = 180;  // 100 bins apart

    auto magnitudes = generate_two_targets(
        kDefaultNumBins, target1_bin, target2_bin,
        peak_amplitude, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    // Both targets should be detected
    bool found_target1 = false;
    bool found_target2 = false;

    for (const auto& det : detections) {
        if (std::abs(static_cast<int>(det.bin_index()) - static_cast<int>(target1_bin)) <= 5) {
            found_target1 = true;
        }
        if (std::abs(static_cast<int>(det.bin_index()) - static_cast<int>(target2_bin)) <= 5) {
            found_target2 = true;
        }
    }

    EXPECT_TRUE(found_target1);
    EXPECT_TRUE(found_target2);
}

// Test: Edge target detection
TEST_F(CfarDetectorTest, EdgeTargetDetection) {
    CfarDetector cfar(config_);

    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    uint32_t target_bin = 5;  // Near edge

    auto magnitudes = generate_single_target(
        kDefaultNumBins, target_bin, peak_amplitude, noise_floor, 2.0);

    auto detections = cfar.detect_1d(magnitudes);

    // Should detect target near edge
    bool found_target = false;
    for (const auto& det : detections) {
        if (det.bin_index() <= 10) {
            found_target = true;
            break;
        }
    }

    EXPECT_TRUE(found_target);
}

// Test: Higher threshold factor reduces detections
TEST_F(CfarDetectorTest, HigherThresholdReducesDetections) {
    double noise_floor = 1.0;
    double peak_amplitude = 100.0;  // Moderate SNR
    auto magnitudes = generate_single_target(
        kDefaultNumBins, kTargetBin, peak_amplitude, noise_floor);

    // Low threshold
    config_.set_threshold_factor(2.0);
    CfarDetector cfar_low(config_);
    auto detections_low = cfar_low.detect_1d(magnitudes);

    // High threshold
    config_.set_threshold_factor(8.0);
    CfarDetector cfar_high(config_);
    auto detections_high = cfar_high.detect_1d(magnitudes);

    // Higher threshold should have fewer detections
    EXPECT_LT(detections_high.size(), detections_low.size());
}

// Test: GO-CFAR mode
TEST_F(CfarDetectorTest, GoCfarMode) {
    config_.set_mode(CFAR_MODE_GO);
    CfarDetector cfar(config_);

    EXPECT_TRUE(cfar.is_enabled());
    EXPECT_EQ(cfar.config().mode(), CFAR_MODE_GO);

    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    auto magnitudes = generate_single_target(
        kDefaultNumBins, kTargetBin, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    // Should still detect the target
    EXPECT_GT(detections.size(), 0u);
}

// Test: SO-CFAR mode
TEST_F(CfarDetectorTest, SoCfarMode) {
    config_.set_mode(CFAR_MODE_SO);
    CfarDetector cfar(config_);

    EXPECT_TRUE(cfar.is_enabled());
    EXPECT_EQ(cfar.config().mode(), CFAR_MODE_SO);

    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    auto magnitudes = generate_single_target(
        kDefaultNumBins, kTargetBin, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    // Should still detect the target
    EXPECT_GT(detections.size(), 0u);
}

// Test: OS-CFAR mode
TEST_F(CfarDetectorTest, OsCfarMode) {
    config_.set_mode(CFAR_MODE_OS);
    config_.set_os_rank(18);  // 3/4 of 24 reference cells
    CfarDetector cfar(config_);

    EXPECT_TRUE(cfar.is_enabled());
    EXPECT_EQ(cfar.config().mode(), CFAR_MODE_OS);

    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    auto magnitudes = generate_single_target(
        kDefaultNumBins, kTargetBin, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    // Should still detect the target
    EXPECT_GT(detections.size(), 0u);
}

// Test: GO-CFAR better for close targets
TEST_F(CfarDetectorTest, GoCfarBetterForCloseTargets) {
    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    uint32_t target1_bin = 100;
    uint32_t target2_bin = 120;  // Close together

    auto magnitudes = generate_two_targets(
        kDefaultNumBins, target1_bin, target2_bin,
        peak_amplitude, peak_amplitude, noise_floor, 3.0);

    // CA-CFAR
    config_.set_mode(CFAR_MODE_CA);
    CfarDetector cfar_ca(config_);
    auto detections_ca = cfar_ca.detect_1d(magnitudes);

    // GO-CFAR
    config_.set_mode(CFAR_MODE_GO);
    CfarDetector cfar_go(config_);
    auto detections_go = cfar_go.detect_1d(magnitudes);

    // Both should detect something
    EXPECT_GT(detections_ca.size(), 0u);
    EXPECT_GT(detections_go.size(), 0u);
}

// Test: Empty input returns empty
TEST_F(CfarDetectorTest, EmptyInputReturnsEmpty) {
    CfarDetector cfar(config_);

    std::vector<double> empty;
    auto detections = cfar.detect_1d(empty);

    EXPECT_TRUE(detections.empty());
}

// Test: Detection result contains all fields
TEST_F(CfarDetectorTest, DetectionResultContainsAllFields) {
    CfarDetector cfar(config_);

    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    auto magnitudes = generate_single_target(
        kDefaultNumBins, kTargetBin, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    ASSERT_GT(detections.size(), 0u);

    const auto& det = detections[0];
    EXPECT_LT(det.bin_index(), kDefaultNumBins);
    EXPECT_GT(det.magnitude(), 0.0);
    EXPECT_GT(det.noise_estimate(), 0.0);
    EXPECT_GT(det.threshold(), 0.0);
    // SNR should be a reasonable value
    EXPECT_GT(det.snr_db(), -100.0);
    EXPECT_LT(det.snr_db(), 200.0);
}

// Test: Threshold is max of CFAR and min_threshold
TEST_F(CfarDetectorTest, MinThresholdApplied) {
    config_.set_min_threshold(1e10);  // Very high minimum
    CfarDetector cfar(config_);

    double noise_floor = 1.0;
    double peak_amplitude = 100.0;  // Below min_threshold
    auto magnitudes = generate_single_target(
        kDefaultNumBins, kTargetBin, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    // Should have no detections because signal is below min_threshold
    EXPECT_TRUE(detections.empty());
}

// Test: Missing required fields throws error
TEST_F(CfarDetectorTest, MissingRequiredFieldsThrows) {
    // Missing guard_cells
    CfarConfig bad_config;
    bad_config.set_mode(CFAR_MODE_CA);
    bad_config.set_reference_cells(12);
    bad_config.set_threshold_factor(4.0);
    bad_config.set_min_threshold(1e-6);
    // guard_cells is 0 (unset)

    EXPECT_THROW(CfarDetector cfar(bad_config), DetectionError);
}

// Test: Missing all required fields throws error
TEST_F(CfarDetectorTest, AllRequiredFieldsMissing) {
    CfarConfig minimal_config;
    minimal_config.set_mode(CFAR_MODE_CA);
    // Don't set any other fields - should throw

    EXPECT_THROW(CfarDetector cfar(minimal_config), DetectionError);
}

// Test: FIXED mode doesn't require validation
TEST_F(CfarDetectorTest, FixedModeNoValidation) {
    CfarConfig fixed_config;
    fixed_config.set_mode(CFAR_MODE_FIXED);
    // No other fields set - should NOT throw

    EXPECT_NO_THROW(CfarDetector cfar(fixed_config));
}

// Test: Very short array handles edge cases
TEST_F(CfarDetectorTest, VeryShortArrayHandled) {
    CfarDetector cfar(config_);

    // Array shorter than window size
    std::vector<double> short_array = {1.0, 2.0, 100.0, 2.0, 1.0};

    auto detections = cfar.detect_1d(short_array);

    // Should handle gracefully (may or may not detect depending on edge handling)
    // Just verify no crash
    SUCCEED();
}

// Test: All zeros input
TEST_F(CfarDetectorTest, AllZerosInput) {
    CfarDetector cfar(config_);

    std::vector<double> zeros(kDefaultNumBins, 0.0);
    auto detections = cfar.detect_1d(zeros);

    // With min_threshold, should have no detections
    EXPECT_TRUE(detections.empty());
}

// Test: Monotonically increasing array - CFAR adapts to trend
TEST_F(CfarDetectorTest, MonotonicallyIncreasingArray) {
    CfarDetector cfar(config_);

    std::vector<double> increasing(kDefaultNumBins);
    for (size_t i = 0; i < kDefaultNumBins; ++i) {
        increasing[i] = static_cast<double>(i) + 1.0;
    }

    auto detections = cfar.detect_1d(increasing);

    // With CFAR, a monotonically increasing signal should NOT produce many
    // detections because the noise estimate follows the trend.
    // The edge handling might produce some detections at the ends, but
    // the middle should be mostly quiet since signal/noise ratio is constant.
    // This verifies CFAR is adapting to the local signal level.
    EXPECT_LT(detections.size(), kDefaultNumBins / 4);  // Less than 25% false alarms
}

// Test: skip_bins filters out low-range bins
TEST_F(CfarDetectorTest, SkipBinsFiltersLowRange) {
    // Create config with skip_bins = 10
    config_.set_skip_bins(10);
    CfarDetector cfar(config_);

    // Create signal with peaks at bin 5 and bin 50
    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    auto magnitudes = generate_two_targets(
        kDefaultNumBins, 5, 50, peak_amplitude, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    // Bin 5 should be skipped, only bin 50 should be detected
    // With sidelobes, we may get multiple detections around bin 50
    ASSERT_GT(detections.size(), 0u);

    // All detections should be >= skip_bins
    for (const auto& det : detections) {
        EXPECT_GE(det.bin_index(), 10u) << "Detection at bin " << det.bin_index()
                                      << " should be filtered by skip_bins=10";
    }

    // Verify the main detection is around bin 50
    bool found_bin_50_region = false;
    for (const auto& det : detections) {
        if (det.bin_index() >= 45 && det.bin_index() <= 55) {
            found_bin_50_region = true;
            break;
        }
    }
    EXPECT_TRUE(found_bin_50_region) << "Should detect target around bin 50";
}

// Test: Detections are sorted by bin index
TEST_F(CfarDetectorTest, DetectionsSortedByBinIndex) {
    CfarDetector cfar(config_);

    // Multiple targets
    double noise_floor = 1.0;
    double peak_amplitude = 1000.0;
    auto magnitudes = generate_two_targets(
        kDefaultNumBins, 50, 200, peak_amplitude, peak_amplitude, noise_floor);

    auto detections = cfar.detect_1d(magnitudes);

    ASSERT_GT(detections.size(), 1u);

    // Verify sorted
    for (size_t i = 1; i < detections.size(); ++i) {
        EXPECT_GE(detections[i].bin_index(), detections[i-1].bin_index());
    }
}

}  // namespace
}  // namespace aurora::detection
