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
 * @file test_angle_estimator.cpp
 * @brief Unit tests for AngleEstimator (2D phase-gradient angle estimation).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "detection/angle_estimator.h"

using aurora::detection::AngleEstimator;
using aurora::detection::AngleFitResult;

namespace {

constexpr double kFocalLength = 0.2;        // meters
constexpr double kWavelength = 1550e-9;     // meters
constexpr double kPitch = 0.5e-3;           // meters
constexpr double kFovX = 0.2;               // radians (full)
constexpr double kFovY = 0.2;               // radians (full)
constexpr double kNoiseFloor = 1e-12;       // linear power
constexpr uint32_t kMaxBins = 4096;
constexpr double kTwoPi = 2.0 * M_PI;

/// K-spacing between adjacent pixels: Δk = 2π·pitch/(f·λ).
double detector_k_step() {
    return kTwoPi * kPitch / (kFocalLength * kWavelength);
}

/// K-vector sampled by a detector pixel at Fourier-plane position (x, y):
/// k = 2π·(x, y)/(f·λ).
Eigen::Vector2d pixel_k(double x_m, double y_m) {
    const double scale = kTwoPi / (kFocalLength * kWavelength);
    return Eigen::Vector2d(scale * x_m, scale * y_m);
}

/// Build samples for a point target at (theta_x, theta_y) with the given K-vectors.
std::vector<AngleEstimator::Sample> synthesize(
    const std::vector<Eigen::Vector2d>& k_vectors,
    double theta_x,
    double theta_y,
    double amplitude = 1.0) {
    std::vector<AngleEstimator::Sample> samples;
    Eigen::Vector2d r(kFocalLength * theta_x, kFocalLength * theta_y);
    for (const auto& k : k_vectors) {
        double phase = -k.dot(r);
        samples.push_back({k, std::polar(amplitude, phase)});
    }
    return samples;
}

/// K-vectors of the 8-pixel linear detector array (pixels along x at kPitch).
std::vector<Eigen::Vector2d> linear_array_k() {
    std::vector<Eigen::Vector2d> k;
    for (int d = 0; d < 8; ++d) {
        k.push_back(pixel_k((d - 3.5) * kPitch, 0.0));
    }
    return k;
}

/// K-vectors of an 8-pixel two-dimensional detector array: a 4×2 grid of
/// pixels at kPitch in both x and y.
std::vector<Eigen::Vector2d> grid_array_k() {
    std::vector<Eigen::Vector2d> k;
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 4; ++col) {
            k.push_back(pixel_k((col - 1.5) * kPitch, (row - 0.5) * kPitch));
        }
    }
    return k;
}

}  // namespace

TEST(AngleEstimatorTest, RejectsInvalidConfiguration) {
    EXPECT_THROW(AngleEstimator(0.0, kNoiseFloor, kFovX, kFovY, kMaxBins),
                 std::invalid_argument);
    EXPECT_THROW(AngleEstimator(kFocalLength, kNoiseFloor, 0.0, kFovY, kMaxBins),
                 std::invalid_argument);
    EXPECT_THROW(AngleEstimator(kFocalLength, kNoiseFloor, kFovX, kFovY, 1),
                 std::invalid_argument);
}

TEST(AngleEstimatorTest, TooFewSamplesIsInvalid) {
    AngleEstimator estimator(kFocalLength, kNoiseFloor, kFovX, kFovY, kMaxBins);
    auto result = estimator.estimate({});
    EXPECT_FALSE(result.valid());

    auto one = synthesize({Eigen::Vector2d(1000.0, 0.0)}, 0.001, 0.0);
    EXPECT_FALSE(estimator.estimate(one).valid());
}

TEST(AngleEstimatorTest, BoresightTarget) {
    AngleEstimator estimator(kFocalLength, kNoiseFloor, kFovX, kFovY, kMaxBins);
    auto samples = synthesize(linear_array_k(), 0.0, 0.0);
    auto result = estimator.estimate(samples);

    ASSERT_TRUE(result.valid());
    EXPECT_NEAR(result.azimuth_rad(), 0.0, 1e-6);
    EXPECT_FALSE(result.elevation_observable());
    EXPECT_TRUE(std::isinf(result.elevation_uncertainty_rad()));
    EXPECT_LT(result.residual_rad(), 1e-6);
    EXPECT_EQ(result.num_samples(), 8u);
}

TEST(AngleEstimatorTest, UniformArrayWithinUnambiguousRange) {
    // Uniform 8-pixel array: unambiguous azimuth period = 2π/(f·Δk).
    double period = kTwoPi / (kFocalLength * detector_k_step());
    double theta = 0.3 * period;

    AngleEstimator estimator(kFocalLength, kNoiseFloor, kFovX, kFovY, kMaxBins);
    auto result = estimator.estimate(synthesize(linear_array_k(), theta, 0.0));

    ASSERT_TRUE(result.valid());
    EXPECT_NEAR(result.azimuth_rad(), theta, 1e-6);
    EXPECT_LT(result.residual_rad(), 1e-6);
}

TEST(AngleEstimatorTest, NegativeAzimuth) {
    double period = kTwoPi / (kFocalLength * detector_k_step());
    double theta = -0.25 * period;

    AngleEstimator estimator(kFocalLength, kNoiseFloor, kFovX, kFovY, kMaxBins);
    auto result = estimator.estimate(synthesize(linear_array_k(), theta, 0.0));

    ASSERT_TRUE(result.valid());
    EXPECT_NEAR(result.azimuth_rad(), theta, 1e-6);
}

TEST(AngleEstimatorTest, ElevationFromTwoDimensionalArray) {
    // A 4×2 pixel grid spans ky as well as kx, so both angles are observable.
    // Pixel pitch is the same on both axes, so the unambiguous period is too.
    double period = kTwoPi / (kFocalLength * detector_k_step());
    double theta_x = 0.2 * period;
    double theta_y = -0.15 * period;

    AngleEstimator estimator(kFocalLength, kNoiseFloor, kFovX, kFovY, kMaxBins);
    auto result = estimator.estimate(synthesize(grid_array_k(), theta_x, theta_y));

    ASSERT_TRUE(result.valid());
    EXPECT_TRUE(result.elevation_observable());
    EXPECT_NEAR(result.azimuth_rad(), theta_x, 1e-6);
    EXPECT_NEAR(result.elevation_rad(), theta_y, 1e-6);
    EXPECT_TRUE(std::isfinite(result.elevation_uncertainty_rad()));
}

TEST(AngleEstimatorTest, UncertaintyShrinksWithSnr) {
    AngleEstimator estimator(kFocalLength, kNoiseFloor, kFovX, kFovY, kMaxBins);
    auto weak = estimator.estimate(synthesize(linear_array_k(), 0.001, 0.0, 1e-5));
    auto strong = estimator.estimate(synthesize(linear_array_k(), 0.001, 0.0, 1e-2));

    ASSERT_TRUE(weak.valid());
    ASSERT_TRUE(strong.valid());
    EXPECT_LT(strong.azimuth_uncertainty_rad(), weak.azimuth_uncertainty_rad());
}

TEST(AngleEstimatorTest, DegenerateKSpaceIsInvalid) {
    // All samples at the same K: phase carries no position information.
    std::vector<Eigen::Vector2d> k(8, Eigen::Vector2d(5000.0, 0.0));
    AngleEstimator estimator(kFocalLength, kNoiseFloor, kFovX, kFovY, kMaxBins);
    auto result = estimator.estimate(synthesize(k, 0.001, 0.0));
    EXPECT_FALSE(result.valid());
}

TEST(AngleEstimatorTest, RobustToModerateNoise) {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> noise(0.0, 0.05);

    double period = kTwoPi / (kFocalLength * detector_k_step());
    double theta = 0.2 * period;
    auto samples = synthesize(linear_array_k(), theta, 0.0);
    for (auto& s : samples) {
        s.value += std::complex<double>(noise(rng), noise(rng));
    }

    AngleEstimator estimator(kFocalLength, kNoiseFloor, kFovX, kFovY, kMaxBins);
    auto result = estimator.estimate(samples);

    ASSERT_TRUE(result.valid());
    EXPECT_NEAR(result.azimuth_rad(), theta, 0.05 * period);
    EXPECT_GT(result.residual_rad(), 0.0);
}
