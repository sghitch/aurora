/**
 * @file test_phase_gradient_estimator.cpp
 * @brief Unit tests for the PhaseGradientEstimator base class.
 *
 * Tests verify:
 * - Phase gradient fitting produces correct angles
 * - Phase unwrapping handles various scenarios
 * - Uncertainty computation follows Cramer-Rao bound
 * - Extent estimation for point vs extended targets
 * - Phase wrapping utility function
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "detection/phase_gradient_estimator.h"

namespace aurora::detection {
namespace {

// Test configuration
constexpr double kFocalLength = 0.1;       // meters (100mm)
constexpr double kNoiseFloor = 1e-6;       // linear noise floor

/**
 * @brief Concrete test implementation of PhaseGradientEstimator.
 *
 * Since PhaseGradientEstimator is abstract (protected constructor and methods),
 * we create a concrete derived class for testing.
 */
class TestablePhaseGradientEstimator : public PhaseGradientEstimator {
  public:
    TestablePhaseGradientEstimator(double focal_length_m, double noise_floor_linear)
        : PhaseGradientEstimator(focal_length_m, noise_floor_linear) {}

    // Expose protected methods for testing
    using PhaseGradientEstimator::fit_phase_gradient;
    using PhaseGradientEstimator::unwrap_phases;
    using PhaseGradientEstimator::compute_uncertainty;
    using PhaseGradientEstimator::estimate_extent_from_residuals;
    using PhaseGradientEstimator::wrap_phase;
    using PhaseGradientEstimator::focal_length_m;
    using PhaseGradientEstimator::noise_floor_linear;
};

class PhaseGradientEstimatorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        estimator_ = std::make_unique<TestablePhaseGradientEstimator>(
            kFocalLength, kNoiseFloor);
    }

    std::unique_ptr<TestablePhaseGradientEstimator> estimator_;
};

// =============================================================================
// Constructor and Accessor Tests
// =============================================================================

TEST_F(PhaseGradientEstimatorTest, AccessorsReturnCorrectValues) {
    EXPECT_DOUBLE_EQ(estimator_->focal_length_m(), kFocalLength);
    EXPECT_DOUBLE_EQ(estimator_->noise_floor_linear(), kNoiseFloor);
}

// =============================================================================
// Phase Gradient Fitting Tests
// =============================================================================

TEST_F(PhaseGradientEstimatorTest, FitPhaseGradientOnAxis) {
    // Simulate an on-axis target (x = 0)
    // Phase should be constant across all K values
    std::vector<double> k_values = {1000.0, 2000.0, 3000.0, 4000.0};
    std::vector<double> phases = {0.0, 0.0, 0.0, 0.0};  // No phase gradient
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0};

    auto result = estimator_->fit_phase_gradient(phases, k_values, weights);

    EXPECT_TRUE(result.valid());
    EXPECT_NEAR(result.angle_rad(), 0.0, 0.001);
    // When phases are constant (no variance), r_squared is 0/0 which gives 0
    // This is correct behavior - the fit is perfect but SST is 0
    EXPECT_GE(result.r_squared(), 0.0);
    EXPECT_LE(result.r_squared(), 1.0);
}

TEST_F(PhaseGradientEstimatorTest, FitPhaseGradientOffAxis) {
    // Simulate an off-axis target
    // Phase: φ = -k * x, where x = f * tan(θ)
    double target_angle_rad = 0.01;  // ~0.57 degrees
    double x_target = kFocalLength * std::tan(target_angle_rad);

    std::vector<double> k_values = {1000.0, 2000.0, 3000.0, 4000.0};
    std::vector<double> phases;
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0};

    for (double k : k_values) {
        phases.push_back(-k * x_target);
    }

    auto result = estimator_->fit_phase_gradient(phases, k_values, weights);

    EXPECT_TRUE(result.valid());
    EXPECT_NEAR(result.angle_rad(), target_angle_rad, 0.0001);
    EXPECT_GT(result.r_squared(), 0.999);
}

TEST_F(PhaseGradientEstimatorTest, FitPhaseGradientWithNoise) {
    // Simulate a target with noisy phases
    double target_angle_rad = 0.02;
    double x_target = kFocalLength * std::tan(target_angle_rad);

    std::vector<double> k_values = {1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0};
    std::vector<double> phases;
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    // Add small noise to phases
    std::vector<double> noise = {0.01, -0.01, 0.02, -0.02, 0.01, -0.01};
    for (size_t i = 0; i < k_values.size(); ++i) {
        phases.push_back(-k_values[i] * x_target + noise[i]);
    }

    auto result = estimator_->fit_phase_gradient(phases, k_values, weights);

    EXPECT_TRUE(result.valid());
    EXPECT_NEAR(result.angle_rad(), target_angle_rad, 0.005);  // Tolerance for noise
    EXPECT_GT(result.r_squared(), 0.99);
}

TEST_F(PhaseGradientEstimatorTest, FitPhaseGradientInsufficientSamples) {
    // Less than 2 samples should fail
    std::vector<double> k_values = {1000.0};
    std::vector<double> phases = {0.0};
    std::vector<double> weights = {1.0};

    auto result = estimator_->fit_phase_gradient(phases, k_values, weights);

    EXPECT_FALSE(result.valid());
}

TEST_F(PhaseGradientEstimatorTest, FitPhaseGradientMismatchedSizes) {
    std::vector<double> k_values = {1000.0, 2000.0, 3000.0};
    std::vector<double> phases = {0.0, 0.0};  // Different size
    std::vector<double> weights = {1.0, 1.0, 1.0};

    auto result = estimator_->fit_phase_gradient(phases, k_values, weights);

    EXPECT_FALSE(result.valid());
}

TEST_F(PhaseGradientEstimatorTest, FitPhaseGradientWithWeights) {
    // Higher weights should pull the fit toward those samples
    double target_angle_rad = 0.01;
    double x_target = kFocalLength * std::tan(target_angle_rad);

    std::vector<double> k_values = {1000.0, 2000.0, 3000.0, 4000.0};
    std::vector<double> phases;
    std::vector<double> high_weights = {10.0, 10.0, 10.0, 10.0};
    std::vector<double> low_weights = {0.1, 0.1, 0.1, 0.1};

    for (double k : k_values) {
        phases.push_back(-k * x_target);
    }

    auto result_high = estimator_->fit_phase_gradient(phases, k_values, high_weights);
    auto result_low = estimator_->fit_phase_gradient(phases, k_values, low_weights);

    // Both should recover the correct angle
    EXPECT_NEAR(result_high.angle_rad(), target_angle_rad, 0.0001);
    EXPECT_NEAR(result_low.angle_rad(), target_angle_rad, 0.0001);

    // Higher weights should give lower uncertainty
    EXPECT_LT(result_high.uncertainty_rad(), result_low.uncertainty_rad());
}

// =============================================================================
// Phase Unwrapping Tests
// =============================================================================

TEST_F(PhaseGradientEstimatorTest, UnwrapPhasesNoWrapping) {
    // Phases that don't need unwrapping
    std::vector<double> wrapped = {0.0, 0.5, 1.0, 1.5};
    std::vector<double> k_values = {1000.0, 2000.0, 3000.0, 4000.0};
    double prior = 0.0;

    auto unwrapped = estimator_->unwrap_phases(wrapped, k_values, prior);

    ASSERT_EQ(unwrapped.size(), wrapped.size());
    for (size_t i = 0; i < wrapped.size(); ++i) {
        EXPECT_NEAR(unwrapped[i], wrapped[i], 0.001);
    }
}

TEST_F(PhaseGradientEstimatorTest, UnwrapPhasesWithWrapping) {
    // Phases that wrap around π
    std::vector<double> wrapped = {2.5, -2.8, 2.9, -2.5};  // Near ±π
    std::vector<double> k_values = {1000.0, 1100.0, 1200.0, 1300.0};
    double prior = 0.0;

    auto unwrapped = estimator_->unwrap_phases(wrapped, k_values, prior);

    ASSERT_EQ(unwrapped.size(), wrapped.size());

    // Check that the unwrapped phases are continuous (differences < π)
    for (size_t i = 1; i < unwrapped.size(); ++i) {
        double diff = std::abs(unwrapped[i] - unwrapped[i-1]);
        EXPECT_LT(diff, M_PI) << "Phase jump at index " << i;
    }
}

TEST_F(PhaseGradientEstimatorTest, UnwrapPhasesEmpty) {
    std::vector<double> wrapped;
    std::vector<double> k_values;
    double prior = 0.0;

    auto unwrapped = estimator_->unwrap_phases(wrapped, k_values, prior);

    EXPECT_TRUE(unwrapped.empty());
}

TEST_F(PhaseGradientEstimatorTest, UnwrapPhasesSingleSample) {
    std::vector<double> wrapped = {1.5};
    std::vector<double> k_values = {1000.0};
    double prior = 0.0;

    auto unwrapped = estimator_->unwrap_phases(wrapped, k_values, prior);

    ASSERT_EQ(unwrapped.size(), 1u);
    EXPECT_NEAR(unwrapped[0], 1.5, 0.001);
}

// =============================================================================
// Uncertainty Computation Tests
// =============================================================================

TEST_F(PhaseGradientEstimatorTest, ComputeUncertaintyBasic) {
    double k_span = 1000.0;
    double snr_linear = 100.0;
    uint32_t num_samples = 4;

    double uncertainty = estimator_->compute_uncertainty(k_span, snr_linear, num_samples);

    EXPECT_GT(uncertainty, 0.0);
    EXPECT_LT(uncertainty, M_PI);
}

TEST_F(PhaseGradientEstimatorTest, ComputeUncertaintyHigherSnrLowerUncertainty) {
    double k_span = 1000.0;
    uint32_t num_samples = 4;

    double low_snr_uncertainty = estimator_->compute_uncertainty(k_span, 10.0, num_samples);
    double high_snr_uncertainty = estimator_->compute_uncertainty(k_span, 100.0, num_samples);

    EXPECT_LT(high_snr_uncertainty, low_snr_uncertainty);
}

TEST_F(PhaseGradientEstimatorTest, ComputeUncertaintyLargerKSpanLowerUncertainty) {
    double snr_linear = 100.0;
    uint32_t num_samples = 4;

    double small_span_uncertainty = estimator_->compute_uncertainty(1000.0, snr_linear, num_samples);
    double large_span_uncertainty = estimator_->compute_uncertainty(10000.0, snr_linear, num_samples);

    EXPECT_LT(large_span_uncertainty, small_span_uncertainty);
}

TEST_F(PhaseGradientEstimatorTest, ComputeUncertaintyMoreSamplesLowerUncertainty) {
    double k_span = 1000.0;
    double snr_linear = 100.0;

    double few_samples = estimator_->compute_uncertainty(k_span, snr_linear, 4);
    double many_samples = estimator_->compute_uncertainty(k_span, snr_linear, 16);

    EXPECT_LT(many_samples, few_samples);
}

TEST_F(PhaseGradientEstimatorTest, ComputeUncertaintyInvalidInputs) {
    // Zero or negative inputs should return maximum uncertainty
    EXPECT_EQ(estimator_->compute_uncertainty(0.0, 100.0, 4), M_PI);
    EXPECT_EQ(estimator_->compute_uncertainty(1000.0, 0.0, 4), M_PI);
    EXPECT_EQ(estimator_->compute_uncertainty(1000.0, 100.0, 0), M_PI);
}

// =============================================================================
// Extent Estimation Tests
// =============================================================================

TEST_F(PhaseGradientEstimatorTest, EstimateExtentPointTarget) {
    // Point target: uniform magnitudes (no roll-off)
    std::vector<double> residuals = {0.01, -0.01, 0.01, -0.01};
    std::vector<double> k_values = {1000.0, 2000.0, 3000.0, 4000.0};
    std::vector<double> magnitudes = {1.0, 1.0, 1.0, 1.0};

    double extent = estimator_->estimate_extent_from_residuals(residuals, k_values, magnitudes);

    // Point target should have zero or very small extent
    EXPECT_GE(extent, 0.0);
    EXPECT_LT(extent, 0.01);
}

TEST_F(PhaseGradientEstimatorTest, EstimateExtentExtendedTarget) {
    // Extended target: Gaussian roll-off in magnitudes
    std::vector<double> residuals = {0.1, 0.05, 0.0, 0.05, 0.1};
    std::vector<double> k_values = {1000.0, 2000.0, 3000.0, 4000.0, 5000.0};
    // Gaussian roll-off centered at k=3000
    std::vector<double> magnitudes = {0.5, 0.8, 1.0, 0.8, 0.5};

    double extent = estimator_->estimate_extent_from_residuals(residuals, k_values, magnitudes);

    EXPECT_GE(extent, 0.0);
}

TEST_F(PhaseGradientEstimatorTest, EstimateExtentEmptyInputs) {
    std::vector<double> empty;

    EXPECT_EQ(estimator_->estimate_extent_from_residuals(empty, empty, empty), 0.0);
}

TEST_F(PhaseGradientEstimatorTest, EstimateExtentMismatchedSizes) {
    std::vector<double> residuals = {0.0, 0.0, 0.0};
    std::vector<double> k_values = {1000.0, 2000.0};  // Different size
    std::vector<double> magnitudes = {1.0, 1.0, 1.0};

    EXPECT_EQ(estimator_->estimate_extent_from_residuals(residuals, k_values, magnitudes), 0.0);
}

// =============================================================================
// Wrap Phase Tests
// =============================================================================

TEST_F(PhaseGradientEstimatorTest, WrapPhaseInRange) {
    // Phases in range should be unchanged
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(0.0), 0.0, 0.001);
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(1.0), 1.0, 0.001);
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(-1.0), -1.0, 0.001);
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(M_PI - 0.01), M_PI - 0.01, 0.001);
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(-M_PI + 0.01), -M_PI + 0.01, 0.001);
}

TEST_F(PhaseGradientEstimatorTest, WrapPhasePositive) {
    // Phases > π should wrap
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(M_PI + 1.0), -M_PI + 1.0, 0.001);
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(2.0 * M_PI), 0.0, 0.001);
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(3.0 * M_PI), M_PI, 0.001);
}

TEST_F(PhaseGradientEstimatorTest, WrapPhaseNegative) {
    // Phases < -π should wrap
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(-M_PI - 1.0), M_PI - 1.0, 0.001);
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(-2.0 * M_PI), 0.0, 0.001);
    EXPECT_NEAR(TestablePhaseGradientEstimator::wrap_phase(-3.0 * M_PI), -M_PI, 0.001);
}

TEST_F(PhaseGradientEstimatorTest, WrapPhaseLargeValues) {
    // Large positive and negative values
    double large_pos = 10.0 * M_PI + 0.5;
    double large_neg = -10.0 * M_PI - 0.5;

    double wrapped_pos = TestablePhaseGradientEstimator::wrap_phase(large_pos);
    double wrapped_neg = TestablePhaseGradientEstimator::wrap_phase(large_neg);

    EXPECT_GE(wrapped_pos, -M_PI);
    EXPECT_LE(wrapped_pos, M_PI);
    EXPECT_GE(wrapped_neg, -M_PI);
    EXPECT_LE(wrapped_neg, M_PI);
}

}  // namespace
}  // namespace aurora::detection
