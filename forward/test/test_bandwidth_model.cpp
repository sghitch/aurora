#include <gtest/gtest.h>
#include <cmath>
#include "forward/bandwidth_model.h"

using aurora::forward::BandwidthModel;
using aurora::forward::BandwidthWindowType;
using aurora::forward::BANDWIDTH_WINDOW_SINC;
using aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN;
using aurora::forward::BANDWIDTH_WINDOW_KAISER;
using aurora::forward::BANDWIDTH_WINDOW_UNSPECIFIED;

// ============================================================================
// Sinc Window Tests
// ============================================================================

TEST(BandwidthModel, SincAtZero) {
    // sinc(0) should be 1.0
    EXPECT_DOUBLE_EQ(BandwidthModel::sinc_window(0.0, 1000.0), 1.0);
}

TEST(BandwidthModel, SincAtNull) {
    // sinc(1) = 0, so sinc_window at delta_k = bandwidth should be ~0
    double bw = 1000.0;
    EXPECT_NEAR(BandwidthModel::sinc_window(bw, bw), 0.0, 1e-10);
}

TEST(BandwidthModel, SincWindowZeroBandwidth) {
    // Zero bandwidth should return 1.0 (no filtering)
    EXPECT_DOUBLE_EQ(BandwidthModel::sinc_window(100.0, 0.0), 1.0);
}

TEST(BandwidthModel, SincWindowSymmetry) {
    double bw = 1000.0;
    EXPECT_DOUBLE_EQ(BandwidthModel::sinc_window(500.0, bw),
                     BandwidthModel::sinc_window(-500.0, bw));
}

TEST(BandwidthModel, SincWindowNegativeLobes) {
    // Sinc has negative lobes between nulls
    double bw = 1000.0;
    // At delta_k = 1.5 * bandwidth, sinc is negative
    double val = BandwidthModel::sinc_window(1.5 * bw, bw);
    EXPECT_LT(val, 0.0);
}

// ============================================================================
// Gaussian Window Tests
// ============================================================================

TEST(BandwidthModel, GaussianAtZero) {
    EXPECT_DOUBLE_EQ(BandwidthModel::gaussian_window(0.0, 1000.0), 1.0);
}

TEST(BandwidthModel, GaussianSymmetry) {
    double bw = 1000.0;
    EXPECT_DOUBLE_EQ(BandwidthModel::gaussian_window(500.0, bw),
                     BandwidthModel::gaussian_window(-500.0, bw));
}

TEST(BandwidthModel, GaussianDecreases) {
    double bw = 1000.0;
    double g0 = BandwidthModel::gaussian_window(0.0, bw);
    double g1 = BandwidthModel::gaussian_window(bw / 2, bw);
    double g2 = BandwidthModel::gaussian_window(bw, bw);

    EXPECT_GT(g0, g1);
    EXPECT_GT(g1, g2);
}

TEST(BandwidthModel, GaussianAlwaysPositive) {
    double bw = 1000.0;
    // Unlike sinc, Gaussian is always positive
    for (double x = 0; x <= 3 * bw; x += bw / 4) {
        EXPECT_GE(BandwidthModel::gaussian_window(x, bw), 0.0);
    }
}

TEST(BandwidthModel, GaussianAtOneSigma) {
    // At delta_k = bandwidth, Gaussian should be exp(-0.5) ≈ 0.6065
    double bw = 1000.0;
    double expected = std::exp(-0.5);
    EXPECT_NEAR(BandwidthModel::gaussian_window(bw, bw), expected, 1e-10);
}

TEST(BandwidthModel, GaussianZeroBandwidth) {
    EXPECT_DOUBLE_EQ(BandwidthModel::gaussian_window(100.0, 0.0), 1.0);
}

// ============================================================================
// Kaiser Window Tests
// ============================================================================

TEST(BandwidthModel, KaiserAtZero) {
    EXPECT_DOUBLE_EQ(BandwidthModel::kaiser_window(0.0, 1000.0), 1.0);
}

TEST(BandwidthModel, KaiserSymmetry) {
    double bw = 1000.0;
    EXPECT_DOUBLE_EQ(BandwidthModel::kaiser_window(500.0, bw),
                     BandwidthModel::kaiser_window(-500.0, bw));
}

TEST(BandwidthModel, KaiserCompactSupport) {
    double bw = 1000.0;
    // Kaiser window is zero outside |delta_k| > bandwidth
    EXPECT_DOUBLE_EQ(BandwidthModel::kaiser_window(bw * 1.1, bw), 0.0);
    EXPECT_DOUBLE_EQ(BandwidthModel::kaiser_window(-bw * 1.1, bw), 0.0);
}

TEST(BandwidthModel, KaiserInSupport) {
    double bw = 1000.0;
    // Inside support, Kaiser should be positive and < 1
    double val = BandwidthModel::kaiser_window(bw / 2, bw);
    EXPECT_GT(val, 0.0);
    EXPECT_LT(val, 1.0);
}

TEST(BandwidthModel, KaiserZeroBandwidth) {
    EXPECT_DOUBLE_EQ(BandwidthModel::kaiser_window(100.0, 0.0), 1.0);
}

// ============================================================================
// Window Function Dispatch Tests
// ============================================================================

TEST(BandwidthModel, WindowFunctionDispatchSinc) {
    double delta_k = 500.0;
    double bw = 1000.0;
    EXPECT_DOUBLE_EQ(
        BandwidthModel::window_function(delta_k, bw, BANDWIDTH_WINDOW_SINC),
        BandwidthModel::sinc_window(delta_k, bw));
}

TEST(BandwidthModel, WindowFunctionDispatchGaussian) {
    double delta_k = 500.0;
    double bw = 1000.0;
    EXPECT_DOUBLE_EQ(
        BandwidthModel::window_function(delta_k, bw, BANDWIDTH_WINDOW_GAUSSIAN),
        BandwidthModel::gaussian_window(delta_k, bw));
}

TEST(BandwidthModel, WindowFunctionDispatchKaiser) {
    double delta_k = 500.0;
    double bw = 1000.0;
    EXPECT_DOUBLE_EQ(
        BandwidthModel::window_function(delta_k, bw, BANDWIDTH_WINDOW_KAISER),
        BandwidthModel::kaiser_window(delta_k, bw));
}

TEST(BandwidthModel, WindowFunctionUnspecifiedThrows) {
    double delta_k = 500.0;
    double bw = 1000.0;
    EXPECT_THROW(
        BandwidthModel::window_function(delta_k, bw, BANDWIDTH_WINDOW_UNSPECIFIED),
        std::invalid_argument);
}

TEST(BandwidthModel, KSpaceWeightUnspecifiedThrows) {
    EXPECT_THROW(
        BandwidthModel::k_space_weight({0, 0}, 1000, 1000, BANDWIDTH_WINDOW_UNSPECIFIED),
        std::invalid_argument);
}

// ============================================================================
// K-Space Weight Tests
// ============================================================================

TEST(BandwidthModel, KSpaceWeightAtZeroSinc) {
    EXPECT_DOUBLE_EQ(
        BandwidthModel::k_space_weight({0, 0}, 1000, 1000, BANDWIDTH_WINDOW_SINC),
        1.0);
}

TEST(BandwidthModel, KSpaceWeightAtZeroGaussian) {
    EXPECT_DOUBLE_EQ(
        BandwidthModel::k_space_weight({0, 0}, 1000, 1000, BANDWIDTH_WINDOW_GAUSSIAN),
        1.0);
}

TEST(BandwidthModel, KSpaceWeightAtZeroKaiser) {
    EXPECT_DOUBLE_EQ(
        BandwidthModel::k_space_weight({0, 0}, 1000, 1000, BANDWIDTH_WINDOW_KAISER),
        1.0);
}

TEST(BandwidthModel, KSpaceWeightRequiresWindowType) {
    // Verify that k_space_weight with explicit SINC matches sinc_window product
    Eigen::Vector2d delta_k(500.0, 200.0);
    double bw = 1000.0;
    double expected = BandwidthModel::sinc_window(delta_k.x(), bw) *
                      BandwidthModel::sinc_window(delta_k.y(), bw);
    EXPECT_DOUBLE_EQ(
        BandwidthModel::k_space_weight(delta_k, bw, bw, BANDWIDTH_WINDOW_SINC),
        expected);
}

TEST(BandwidthModel, KSpaceWeightSymmetry) {
    double bw = 1000.0;
    double w1 = BandwidthModel::k_space_weight({500.0, 200.0}, bw, bw, BANDWIDTH_WINDOW_SINC);
    double w2 = BandwidthModel::k_space_weight({-500.0, -200.0}, bw, bw, BANDWIDTH_WINDOW_SINC);
    EXPECT_DOUBLE_EQ(w1, w2);
}

TEST(BandwidthModel, KSpaceWeightSeparable) {
    double bw = 1000.0;
    Eigen::Vector2d delta_k(300.0, 400.0);

    double wx = BandwidthModel::sinc_window(delta_k.x(), bw);
    double wy = BandwidthModel::sinc_window(delta_k.y(), bw);
    double w2d = BandwidthModel::k_space_weight(delta_k, bw, bw, BANDWIDTH_WINDOW_SINC);

    EXPECT_DOUBLE_EQ(w2d, wx * wy);
}

TEST(BandwidthModel, KSpaceWeightAtNullSinc) {
    double bw = 1000.0;
    // At delta_k = bandwidth, sinc is 0, so product is 0
    EXPECT_NEAR(
        BandwidthModel::k_space_weight({bw, 0}, bw, bw, BANDWIDTH_WINDOW_SINC),
        0.0, 1e-10);
}

// ============================================================================
// Rect Convolution Tests
// ============================================================================

TEST(BandwidthModel, RectConvolutionSymmetry) {
    EXPECT_DOUBLE_EQ(
        BandwidthModel::rect_convolution(0.5, 1.0, 2.0),
        BandwidthModel::rect_convolution(-0.5, 1.0, 2.0));
}

TEST(BandwidthModel, RectConvolutionPeak) {
    EXPECT_DOUBLE_EQ(BandwidthModel::rect_convolution(0.0, 1.0, 2.0), 1.0);
}

TEST(BandwidthModel, RectConvolutionOutsideSupport) {
    // Support is (w1 + w2) / 2 = 1.5
    EXPECT_DOUBLE_EQ(BandwidthModel::rect_convolution(2.0, 1.0, 2.0), 0.0);
}

TEST(BandwidthModel, RectConvolutionAtEdge) {
    // At exactly the edge, should be 0
    double w1 = 1.0, w2 = 2.0;
    double edge = (w1 + w2) / 2.0;  // 1.5
    EXPECT_DOUBLE_EQ(BandwidthModel::rect_convolution(edge, w1, w2), 0.0);
}

TEST(BandwidthModel, RectConvolutionFlatRegionEdge) {
    // Flat region extends to (w_max - w_min) / 2 = 0.5
    double w1 = 1.0, w2 = 2.0;
    double flat_edge = (w2 - w1) / 2.0;  // 0.5
    EXPECT_DOUBLE_EQ(BandwidthModel::rect_convolution(flat_edge, w1, w2), 1.0);
}

TEST(BandwidthModel, RectConvolutionSlopeRegion) {
    // In the sloped region, should be between 0 and 1
    double w1 = 1.0, w2 = 2.0;
    double x = 1.0;  // Between flat_edge (0.5) and edge (1.5)
    double result = BandwidthModel::rect_convolution(x, w1, w2);
    EXPECT_GT(result, 0.0);
    EXPECT_LT(result, 1.0);
    // Linear interpolation: (1.5 - 1.0) / (1.5 - 0.5) = 0.5
    EXPECT_DOUBLE_EQ(result, 0.5);
}

TEST(BandwidthModel, RectConvolutionEqualWidths) {
    // Two equal rect functions convolve to a triangle
    double w = 1.0;
    EXPECT_DOUBLE_EQ(BandwidthModel::rect_convolution(0.0, w, w), 1.0);
    EXPECT_DOUBLE_EQ(BandwidthModel::rect_convolution(w, w, w), 0.0);
    EXPECT_DOUBLE_EQ(BandwidthModel::rect_convolution(w / 2, w, w), 0.5);
}

// ============================================================================
// Resolution Limit Tests
// ============================================================================

TEST(BandwidthModel, ResolutionLimit) {
    double bw = 1000.0;  // rad/m
    auto limit = BandwidthModel::resolution_limit(bw, bw);
    double expected = 2.0 * M_PI / bw;
    EXPECT_NEAR(limit.x(), expected, 1e-10);
    EXPECT_NEAR(limit.y(), expected, 1e-10);
}

TEST(BandwidthModel, ResolutionLimitZeroBandwidth) {
    auto limit = BandwidthModel::resolution_limit(0.0, 1000.0);
    EXPECT_DOUBLE_EQ(limit.x(), 0.0);  // No resolution limit if no bandwidth
    EXPECT_GT(limit.y(), 0.0);
}

TEST(BandwidthModel, ResolutionLimitAsymmetric) {
    double bw_x = 1000.0;
    double bw_y = 2000.0;
    auto limit = BandwidthModel::resolution_limit(bw_x, bw_y);
    EXPECT_NEAR(limit.x(), 2.0 * M_PI / bw_x, 1e-10);
    EXPECT_NEAR(limit.y(), 2.0 * M_PI / bw_y, 1e-10);
    EXPECT_GT(limit.x(), limit.y());  // Larger bandwidth = smaller resolution
}

// ============================================================================
// Conversion Tests
// ============================================================================

TEST(BandwidthModel, KBandwidthToSpatialWidthRoundTrip) {
    double bw = 500.0;
    double spatial = BandwidthModel::k_bandwidth_to_spatial_width(bw);
    double bw_back = BandwidthModel::spatial_width_to_k_bandwidth(spatial);
    EXPECT_NEAR(bw_back, bw, 1e-10);
}

TEST(BandwidthModel, KBandwidthToSpatialWidthZero) {
    EXPECT_DOUBLE_EQ(BandwidthModel::k_bandwidth_to_spatial_width(0.0), 0.0);
    EXPECT_DOUBLE_EQ(BandwidthModel::spatial_width_to_k_bandwidth(0.0), 0.0);
}

TEST(BandwidthModel, KBandwidthToFourierWidth) {
    double bw = 1000.0;           // rad/m
    double wavelength = 1550e-9;  // meters
    double focal = 0.05;          // meters

    double result = BandwidthModel::k_bandwidth_to_fourier_width(bw, wavelength, focal);
    // Expected: bw * f * λ / (2π) = 1000 * 0.05 * 1550e-9 / (2π)
    double expected = bw * focal * wavelength / (2.0 * M_PI);
    EXPECT_NEAR(result, expected, 1e-15);
}

// ============================================================================
// Spacing Validation Tests
// ============================================================================

TEST(BandwidthModel, IsSpacingValid) {
    double bw = 1000.0;  // limit = 2π/1000 ≈ 0.00628 m
    EXPECT_TRUE(BandwidthModel::is_spacing_valid(0.01, bw));
    EXPECT_FALSE(BandwidthModel::is_spacing_valid(0.001, bw));
}

TEST(BandwidthModel, IsSpacingValidAtLimit) {
    double bw = 1000.0;
    double limit = 2.0 * M_PI / bw;

    // Exactly at limit * tolerance_factor (0.9) should pass (>= comparison)
    EXPECT_TRUE(BandwidthModel::is_spacing_valid(limit * 0.9, bw));

    // Just below the tolerance threshold should fail
    EXPECT_FALSE(BandwidthModel::is_spacing_valid(limit * 0.89, bw));

    // Well above the tolerance threshold should pass
    EXPECT_TRUE(BandwidthModel::is_spacing_valid(limit * 0.91, bw));
}

TEST(BandwidthModel, IsSpacingValidZeroBandwidth) {
    // Zero bandwidth means no limit, any spacing is valid
    EXPECT_TRUE(BandwidthModel::is_spacing_valid(0.001, 0.0));
    EXPECT_TRUE(BandwidthModel::is_spacing_valid(1e-10, 0.0));
}

TEST(BandwidthModel, IsSpacingValidCustomTolerance) {
    double bw = 1000.0;
    double limit = 2.0 * M_PI / bw;

    // With tolerance factor 0.5, spacing at 0.6 * limit should pass
    EXPECT_TRUE(BandwidthModel::is_spacing_valid(limit * 0.6, bw, 0.5));
    EXPECT_FALSE(BandwidthModel::is_spacing_valid(limit * 0.4, bw, 0.5));
}

// ============================================================================
// Fourier Duality Tests
// ============================================================================

TEST(BandwidthModel, SpatialKSpaceEquivalence) {
    // Verify Fourier duality: spatial rect conv ↔ k-space sinc product
    // At the center both give 1.0
    double det_w = 0.5e-3;
    double det_h = 0.5e-3;

    double spatial_result = BandwidthModel::rect_convolution(0.0, det_w, det_w) *
                            BandwidthModel::rect_convolution(0.0, det_h, det_h);

    double k_result = BandwidthModel::k_space_weight({0, 0}, 1000, 1000, BANDWIDTH_WINDOW_SINC);

    EXPECT_DOUBLE_EQ(spatial_result, 1.0);
    EXPECT_DOUBLE_EQ(k_result, 1.0);
}

TEST(BandwidthModel, ResolutionBandwidthInverse) {
    // Resolution limit is 2π/bandwidth
    // spatial_width = k_bandwidth_to_spatial_width(bw) = 2π/bw
    // So resolution_limit should equal k_bandwidth_to_spatial_width

    double bw_x = 1234.5;
    double bw_y = 6789.0;

    auto res_limit = BandwidthModel::resolution_limit(bw_x, bw_y);
    double spatial_x = BandwidthModel::k_bandwidth_to_spatial_width(bw_x);
    double spatial_y = BandwidthModel::k_bandwidth_to_spatial_width(bw_y);

    EXPECT_NEAR(res_limit.x(), spatial_x, 1e-10);
    EXPECT_NEAR(res_limit.y(), spatial_y, 1e-10);
}

// ============================================================================
// Window Type Comparison Tests
// ============================================================================

TEST(BandwidthModel, WindowTypesAllOneAtZero) {
    double bw = 1000.0;
    EXPECT_DOUBLE_EQ(BandwidthModel::k_space_weight({0, 0}, bw, bw, BANDWIDTH_WINDOW_SINC), 1.0);
    EXPECT_DOUBLE_EQ(BandwidthModel::k_space_weight({0, 0}, bw, bw, BANDWIDTH_WINDOW_GAUSSIAN), 1.0);
    EXPECT_DOUBLE_EQ(BandwidthModel::k_space_weight({0, 0}, bw, bw, BANDWIDTH_WINDOW_KAISER), 1.0);
}

TEST(BandwidthModel, WindowTypesDecayDifferently) {
    double bw = 1000.0;
    Eigen::Vector2d delta_k(bw / 2, 0);

    double sinc_val = BandwidthModel::k_space_weight(delta_k, bw, bw, BANDWIDTH_WINDOW_SINC);
    double gauss_val = BandwidthModel::k_space_weight(delta_k, bw, bw, BANDWIDTH_WINDOW_GAUSSIAN);
    double kaiser_val = BandwidthModel::k_space_weight(delta_k, bw, bw, BANDWIDTH_WINDOW_KAISER);

    // All should be less than 1 away from center
    EXPECT_LT(sinc_val, 1.0);
    EXPECT_LT(gauss_val, 1.0);
    EXPECT_LT(kaiser_val, 1.0);

    // All should still be positive at half-bandwidth
    EXPECT_GT(sinc_val, 0.0);
    EXPECT_GT(gauss_val, 0.0);
    EXPECT_GT(kaiser_val, 0.0);
}

TEST(BandwidthModel, KaiserHasCompactSupport2D) {
    double bw = 1000.0;
    // Kaiser is zero outside support
    Eigen::Vector2d outside(bw * 1.5, 0);
    EXPECT_DOUBLE_EQ(
        BandwidthModel::k_space_weight(outside, bw, bw, BANDWIDTH_WINDOW_KAISER),
        0.0);
}

TEST(BandwidthModel, GaussianNeverZero) {
    double bw = 1000.0;
    // Gaussian asymptotically approaches zero but never reaches it
    Eigen::Vector2d far_away(bw * 5, bw * 5);
    double val = BandwidthModel::k_space_weight(far_away, bw, bw, BANDWIDTH_WINDOW_GAUSSIAN);
    EXPECT_GT(val, 0.0);
    EXPECT_LT(val, 1e-10);  // But very small
}
