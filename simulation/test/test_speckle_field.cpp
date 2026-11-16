#include "simulation/speckle_field.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

namespace aurora::simulation {
namespace {

class SpeckleFieldTest : public ::testing::Test {
  protected:
    static constexpr double kTolerance = 1e-12;
    static constexpr double kStatisticalTolerance = 0.05;  // 5% for statistical tests
    static constexpr int kNumSamples = 10000;
};

// Test: Same inputs always produce same result (determinism)
TEST_F(SpeckleFieldTest, Determinism) {
    Eigen::Vector2d k_vector(1000.0, 500.0);
    double correlation_length = 1e-4;
    uint64_t seed = 42;

    auto result1 = SpeckleField::sample(k_vector, correlation_length, seed);
    auto result2 = SpeckleField::sample(k_vector, correlation_length, seed);

    EXPECT_EQ(result1.real(), result2.real());
    EXPECT_EQ(result1.imag(), result2.imag());
}

// Test: Reproducible across separate function calls
TEST_F(SpeckleFieldTest, ReproducibleAcrossCalls) {
    Eigen::Vector2d k_vector(500.0, -300.0);
    double correlation_length = 2e-4;
    uint64_t seed = 12345;

    // First batch of calls
    std::vector<std::complex<double>> batch1;
    for (int i = 0; i < 10; ++i) {
        Eigen::Vector2d k(k_vector.x() + i * 100, k_vector.y());
        batch1.push_back(SpeckleField::sample(k, correlation_length, seed));
    }

    // Second batch of calls (same inputs)
    std::vector<std::complex<double>> batch2;
    for (int i = 0; i < 10; ++i) {
        Eigen::Vector2d k(k_vector.x() + i * 100, k_vector.y());
        batch2.push_back(SpeckleField::sample(k, correlation_length, seed));
    }

    // Should be identical
    for (size_t i = 0; i < batch1.size(); ++i) {
        EXPECT_EQ(batch1[i], batch2[i]);
    }
}

// Test: Different seeds produce different results
TEST_F(SpeckleFieldTest, DifferentSeedsDifferentResults) {
    Eigen::Vector2d k_vector(1000.0, 500.0);
    double correlation_length = 1e-4;

    auto result1 = SpeckleField::sample(k_vector, correlation_length, 42);
    auto result2 = SpeckleField::sample(k_vector, correlation_length, 43);

    // Very unlikely to be equal with different seeds
    EXPECT_NE(result1, result2);
}

// Test: Complex Gaussian statistics - mean of real part
TEST_F(SpeckleFieldTest, ComplexGaussianMeanReal) {
    const double correlation_length = 1e-4;
    const double k_grid = 2.0 * M_PI / correlation_length;

    double sum_real = 0.0;
    for (int i = 0; i < kNumSamples; ++i) {
        // Sample at different K-vectors spread across many grid cells
        Eigen::Vector2d k(i * k_grid * 1.5, 0);
        auto sample = SpeckleField::sample(k, correlation_length, 42);
        sum_real += sample.real();
    }

    double mean_real = sum_real / kNumSamples;
    // Mean should be close to 0 (within statistical tolerance)
    EXPECT_NEAR(mean_real, 0.0, kStatisticalTolerance);
}

// Test: Complex Gaussian statistics - mean of imaginary part
TEST_F(SpeckleFieldTest, ComplexGaussianMeanImag) {
    const double correlation_length = 1e-4;
    const double k_grid = 2.0 * M_PI / correlation_length;

    double sum_imag = 0.0;
    for (int i = 0; i < kNumSamples; ++i) {
        Eigen::Vector2d k(0, i * k_grid * 1.5);
        auto sample = SpeckleField::sample(k, correlation_length, 42);
        sum_imag += sample.imag();
    }

    double mean_imag = sum_imag / kNumSamples;
    EXPECT_NEAR(mean_imag, 0.0, kStatisticalTolerance);
}

// Test: Complex Gaussian statistics - unit variance of |xi|^2
TEST_F(SpeckleFieldTest, ComplexGaussianUnitVariance) {
    const double correlation_length = 1e-4;
    const double k_grid = 2.0 * M_PI / correlation_length;

    double sum_mag_sq = 0.0;
    for (int i = 0; i < kNumSamples; ++i) {
        // Sample at different grid cells
        Eigen::Vector2d k(i * k_grid * 1.5, i * k_grid * 0.7);
        auto sample = SpeckleField::sample(k, correlation_length, 42);
        sum_mag_sq += std::norm(sample);  // |xi|^2
    }

    double mean_mag_sq = sum_mag_sq / kNumSamples;
    // E[|xi|^2] should be 1 for unit variance complex Gaussian
    EXPECT_NEAR(mean_mag_sq, 1.0, kStatisticalTolerance);
}

// Test: Real and imaginary parts are uncorrelated
TEST_F(SpeckleFieldTest, RealImagUncorrelated) {
    const double correlation_length = 1e-4;
    const double k_grid = 2.0 * M_PI / correlation_length;

    double sum_re = 0.0, sum_im = 0.0;
    double sum_re_im = 0.0;
    double sum_re_sq = 0.0, sum_im_sq = 0.0;

    for (int i = 0; i < kNumSamples; ++i) {
        Eigen::Vector2d k(i * k_grid * 1.5, i * k_grid * 0.3);
        auto sample = SpeckleField::sample(k, correlation_length, 42);

        double re = sample.real();
        double im = sample.imag();

        sum_re += re;
        sum_im += im;
        sum_re_im += re * im;
        sum_re_sq += re * re;
        sum_im_sq += im * im;
    }

    double n = static_cast<double>(kNumSamples);
    double mean_re = sum_re / n;
    double mean_im = sum_im / n;

    // Compute correlation coefficient
    double var_re = sum_re_sq / n - mean_re * mean_re;
    double var_im = sum_im_sq / n - mean_im * mean_im;
    double cov_re_im = sum_re_im / n - mean_re * mean_im;

    double correlation = cov_re_im / std::sqrt(var_re * var_im);

    // Correlation should be close to 0
    EXPECT_NEAR(correlation, 0.0, 0.1);  // Allow 10% tolerance for correlation
}

// Test: K-vectors close together return highly correlated values
TEST_F(SpeckleFieldTest, CloseKVectorsCorrelated) {
    const double correlation_length = 1e-4;
    const double k_grid = 2.0 * M_PI / correlation_length;

    // Base K-vector
    Eigen::Vector2d k1(100.0, 200.0);

    // K-vector very close (offset by small fraction of grid)
    Eigen::Vector2d k2 = k1 + Eigen::Vector2d(k_grid * 0.01, k_grid * 0.01);

    auto sample1 = SpeckleField::sample(k1, correlation_length, 42);
    auto sample2 = SpeckleField::sample(k2, correlation_length, 42);

    // With smooth interpolation, very close K-vectors should be highly similar
    // The difference should be small relative to the magnitude
    double diff = std::abs(sample1 - sample2);
    double avg_mag = (std::abs(sample1) + std::abs(sample2)) / 2.0;

    // Very close K-vectors should differ by less than 5% of magnitude
    EXPECT_LT(diff, avg_mag * 0.05 + 0.01);  // Add small epsilon for near-zero cases
}

// Test: K-vectors far apart return uncorrelated values
TEST_F(SpeckleFieldTest, FarKVectorsUncorrelated) {
    const double correlation_length = 1e-4;
    const double k_grid = 2.0 * M_PI / correlation_length;

    // Collect pairs of samples at various far-apart K-vectors
    // and verify they are uncorrelated (correlation coefficient near 0)
    std::vector<std::complex<double>> samples1, samples2;

    for (int i = 0; i < 100; ++i) {
        // Different base positions with different seeds for variety
        Eigen::Vector2d k1(i * k_grid * 0.1, 0);
        Eigen::Vector2d k2 = k1 + Eigen::Vector2d(k_grid * 3.0, 0);  // 3 grid cells apart

        samples1.push_back(SpeckleField::sample(k1, correlation_length, i));
        samples2.push_back(SpeckleField::sample(k2, correlation_length, i));
    }

    // Compute correlation of real parts
    double sum1 = 0, sum2 = 0, sum12 = 0, sum1sq = 0, sum2sq = 0;
    for (size_t i = 0; i < samples1.size(); ++i) {
        double r1 = samples1[i].real();
        double r2 = samples2[i].real();
        sum1 += r1;
        sum2 += r2;
        sum12 += r1 * r2;
        sum1sq += r1 * r1;
        sum2sq += r2 * r2;
    }

    double n = static_cast<double>(samples1.size());
    double var1 = sum1sq / n - (sum1 / n) * (sum1 / n);
    double var2 = sum2sq / n - (sum2 / n) * (sum2 / n);
    double cov = sum12 / n - (sum1 / n) * (sum2 / n);
    double correlation = cov / std::sqrt(var1 * var2);

    // Correlation should be low (uncorrelated)
    EXPECT_LT(std::abs(correlation), 0.3);
}

// Test: Correlation scale depends on 2*pi/l_c
TEST_F(SpeckleFieldTest, CorrelationScaleScaling) {
    // Two different correlation lengths
    const double ell_c_small = 1e-4;  // Smaller l_c -> larger k_grid -> more K-space correlation
    const double ell_c_large = 2e-4;  // Larger l_c -> smaller k_grid -> less K-space correlation

    const double k_grid_small = 2.0 * M_PI / ell_c_small;  // Larger grid spacing
    const double k_grid_large = 2.0 * M_PI / ell_c_large;  // Smaller grid spacing

    // Use a K-offset that's significant relative to k_grid_large but small relative to k_grid_small
    double k_offset = k_grid_large * 0.8;  // 80% of the larger correlation grid

    Eigen::Vector2d k1(0.0, 0.0);
    Eigen::Vector2d k2(k_offset, 0.0);

    // With small l_c (large k_grid): offset is small fraction of grid -> high correlation
    auto sample_small_1 = SpeckleField::sample(k1, ell_c_small, 42);
    auto sample_small_2 = SpeckleField::sample(k2, ell_c_small, 42);
    double diff_small = std::abs(sample_small_1 - sample_small_2);

    // With large l_c (small k_grid): offset is large fraction of grid -> lower correlation
    auto sample_large_1 = SpeckleField::sample(k1, ell_c_large, 42);
    auto sample_large_2 = SpeckleField::sample(k2, ell_c_large, 42);
    double diff_large = std::abs(sample_large_1 - sample_large_2);

    // The difference should be larger for the larger l_c (smaller correlation scale)
    EXPECT_GT(diff_large, diff_small);
}

// Test: Smaller correlation length gives more K-space correlation
TEST_F(SpeckleFieldTest, SmallerCorrelationLengthMoreKSpaceCorrelation) {
    // Physics: smaller surface features (l_c) -> wider speckle in K-space
    // This means smaller l_c -> larger k_grid -> MORE correlation between K samples
    const double ell_c_large = 1e-3;  // 1 mm surface features
    const double ell_c_small = 1e-5;  // 10 um surface features

    const double k_grid_large = 2.0 * M_PI / ell_c_large;  // ~6,283 rad/m
    const double k_grid_small = 2.0 * M_PI / ell_c_small;  // ~628,318 rad/m

    // k_grid_small >> k_grid_large (smaller surface features -> wider K-space correlation)
    EXPECT_GT(k_grid_small, k_grid_large);

    // Use an offset that's between the two grid sizes
    // This offset is large relative to k_grid_large (low correlation)
    // but small relative to k_grid_small (high correlation)
    double k_offset = k_grid_large * 2.0;  // 2x the smaller grid size

    Eigen::Vector2d k1(0.0, 0.0);
    Eigen::Vector2d k2(k_offset, 0.0);

    // With large l_c (small k_grid): offset spans multiple correlation lengths
    auto sample_large_1 = SpeckleField::sample(k1, ell_c_large, 42);
    auto sample_large_2 = SpeckleField::sample(k2, ell_c_large, 42);
    double diff_large = std::abs(sample_large_1 - sample_large_2);

    // With small l_c (large k_grid): offset is small fraction of correlation length
    auto sample_small_1 = SpeckleField::sample(k1, ell_c_small, 42);
    auto sample_small_2 = SpeckleField::sample(k2, ell_c_small, 42);
    double diff_small = std::abs(sample_small_1 - sample_small_2);

    // Smaller l_c should give higher correlation (smaller difference)
    EXPECT_LT(diff_small, diff_large);
}

// Test: k_vector = (0, 0) works correctly
TEST_F(SpeckleFieldTest, ZeroKVector) {
    Eigen::Vector2d k_zero(0.0, 0.0);
    double correlation_length = 1e-4;

    auto sample = SpeckleField::sample(k_zero, correlation_length, 42);

    // Should return a valid complex number (not NaN or Inf)
    EXPECT_TRUE(std::isfinite(sample.real()));
    EXPECT_TRUE(std::isfinite(sample.imag()));
}

// Test: Negative k_vector values work correctly
TEST_F(SpeckleFieldTest, NegativeKVector) {
    Eigen::Vector2d k_negative(-1000.0, -500.0);
    double correlation_length = 1e-4;

    auto sample = SpeckleField::sample(k_negative, correlation_length, 42);

    // Should return a valid complex number
    EXPECT_TRUE(std::isfinite(sample.real()));
    EXPECT_TRUE(std::isfinite(sample.imag()));

    // Verify determinism for negative values
    auto sample2 = SpeckleField::sample(k_negative, correlation_length, 42);
    EXPECT_EQ(sample, sample2);
}

// Test: Very small correlation length doesn't cause numerical issues
TEST_F(SpeckleFieldTest, VerySmallCorrelationLength) {
    Eigen::Vector2d k_vector(1000.0, 500.0);
    double correlation_length = 1e-9;  // 1 nanometer

    auto sample = SpeckleField::sample(k_vector, correlation_length, 42);

    // Should return a valid complex number
    EXPECT_TRUE(std::isfinite(sample.real()));
    EXPECT_TRUE(std::isfinite(sample.imag()));
}

// Test: Very large correlation length works correctly
TEST_F(SpeckleFieldTest, VeryLargeCorrelationLength) {
    Eigen::Vector2d k_vector1(0.0, 0.0);
    Eigen::Vector2d k_vector2(1e6, 1e6);  // Very large K offset
    double correlation_length = 1.0;  // 1 meter

    auto sample1 = SpeckleField::sample(k_vector1, correlation_length, 42);
    auto sample2 = SpeckleField::sample(k_vector2, correlation_length, 42);

    // Both should be valid
    EXPECT_TRUE(std::isfinite(sample1.real()));
    EXPECT_TRUE(std::isfinite(sample1.imag()));
    EXPECT_TRUE(std::isfinite(sample2.real()));
    EXPECT_TRUE(std::isfinite(sample2.imag()));

    // With 1m correlation length, grid size is 2*pi rad/m
    // K vectors 1e6 rad/m apart should be in different cells
    EXPECT_NE(sample1, sample2);
}

// Test: Smooth continuity across grid cell boundaries
TEST_F(SpeckleFieldTest, SmoothContinuityAcrossBoundaries) {
    const double correlation_length = 1e-4;
    const double k_grid = 2.0 * M_PI / correlation_length;

    // Sample along a line that crosses a cell boundary
    // The boundary is at multiples of k_grid
    double k_boundary = k_grid;  // First boundary at x=k_grid

    std::vector<std::complex<double>> samples;
    std::vector<double> k_values;

    // Sample from 0.9*k_grid to 1.1*k_grid (crossing the boundary)
    for (int i = -10; i <= 10; ++i) {
        double kx = k_boundary + i * (k_grid * 0.01);
        Eigen::Vector2d k(kx, 0.0);
        samples.push_back(SpeckleField::sample(k, correlation_length, 42));
        k_values.push_back(kx);
    }

    // Check that there are no large jumps between adjacent samples
    // (which would indicate a discontinuity)
    double max_jump = 0.0;
    for (size_t i = 1; i < samples.size(); ++i) {
        double jump = std::abs(samples[i] - samples[i-1]);
        max_jump = std::max(max_jump, jump);
    }

    // With smooth interpolation, the maximum jump should be small
    // (no discontinuity at the boundary)
    // The smoothstep function ensures C1 continuity
    EXPECT_LT(max_jump, 0.2);  // Should be much smaller than typical sample magnitude
}

// Test: hash_combine produces different results for different inputs
TEST_F(SpeckleFieldTest, HashCombineDifferentInputs) {
    uint64_t seed = 42;

    uint64_t hash1 = SpeckleField::hash_combine(seed, 0ULL);
    uint64_t hash2 = SpeckleField::hash_combine(seed, 1ULL);
    uint64_t hash3 = SpeckleField::hash_combine(seed, 1000ULL);

    EXPECT_NE(hash1, hash2);
    EXPECT_NE(hash2, hash3);
    EXPECT_NE(hash1, hash3);
}

// Test: hash_combine is deterministic
TEST_F(SpeckleFieldTest, HashCombineDeterministic) {
    uint64_t seed = 42;

    uint64_t hash1 = SpeckleField::hash_combine(seed, 12345ULL);
    uint64_t hash2 = SpeckleField::hash_combine(seed, 12345ULL);

    EXPECT_EQ(hash1, hash2);
}

// Test: hash_combine with grid indices
TEST_F(SpeckleFieldTest, HashCombineGridIndices) {
    uint64_t seed = 42;

    uint64_t hash1 = SpeckleField::hash_combine(seed, 0LL, 0LL);
    uint64_t hash2 = SpeckleField::hash_combine(seed, 0LL, 1LL);
    uint64_t hash3 = SpeckleField::hash_combine(seed, 1LL, 0LL);
    uint64_t hash4 = SpeckleField::hash_combine(seed, -1LL, 0LL);

    // All should be different
    EXPECT_NE(hash1, hash2);
    EXPECT_NE(hash1, hash3);
    EXPECT_NE(hash1, hash4);
    EXPECT_NE(hash2, hash3);
    EXPECT_NE(hash2, hash4);
    EXPECT_NE(hash3, hash4);
}

// Test: Variance of real and imaginary parts are each 0.5
TEST_F(SpeckleFieldTest, RealImagVarianceHalf) {
    const double correlation_length = 1e-4;
    const double k_grid = 2.0 * M_PI / correlation_length;

    double sum_re_sq = 0.0;
    double sum_im_sq = 0.0;
    double sum_re = 0.0;
    double sum_im = 0.0;

    for (int i = 0; i < kNumSamples; ++i) {
        Eigen::Vector2d k(i * k_grid * 1.5, i * k_grid * 0.7);
        auto sample = SpeckleField::sample(k, correlation_length, 42);

        sum_re += sample.real();
        sum_im += sample.imag();
        sum_re_sq += sample.real() * sample.real();
        sum_im_sq += sample.imag() * sample.imag();
    }

    double n = static_cast<double>(kNumSamples);
    double mean_re = sum_re / n;
    double mean_im = sum_im / n;
    double var_re = sum_re_sq / n - mean_re * mean_re;
    double var_im = sum_im_sq / n - mean_im * mean_im;

    // Each should have variance 0.5 for unit variance complex Gaussian
    EXPECT_NEAR(var_re, 0.5, kStatisticalTolerance);
    EXPECT_NEAR(var_im, 0.5, kStatisticalTolerance);
}

}  // namespace
}  // namespace aurora::simulation
