#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <stdexcept>

#include "forward/bandwidth_model.h"
#include "forward/forward_model.h"
#include "forward/measurement_operator.h"
#include "test_forward_model_builder.h"

namespace aurora::forward {
namespace {

using std::complex;

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

/// Window centered at the origin with equal bandwidths in kx and ky.
MeasurementOperator::WindowParams centered_window(double bandwidth) {
    return MeasurementOperator::WindowParams{
        .k_center = Eigen::Vector2d{0.0, 0.0},
        .bandwidth_x = bandwidth,
        .bandwidth_y = bandwidth
    };
}

// =============================================================================
// Weight at the window center
// =============================================================================

TEST(MeasurementOperator, WeightIsUnityAtCenter) {
    MeasurementOperator::WindowParams params{
        .k_center = Eigen::Vector2d{1000.0, 500.0},
        .bandwidth_x = 200.0,
        .bandwidth_y = 300.0
    };

    double w = MeasurementOperator::weight(
        Eigen::Vector2d{1000.0, 500.0}, params, BANDWIDTH_WINDOW_SINC);
    EXPECT_DOUBLE_EQ(w, 1.0);
}

TEST(MeasurementOperator, AllWindowTypesAreUnityAtCenter) {
    MeasurementOperator::WindowParams params{
        .k_center = Eigen::Vector2d{1000.0, 0.0},
        .bandwidth_x = 500.0,
        .bandwidth_y = 500.0
    };

    for (auto type : {BANDWIDTH_WINDOW_SINC, BANDWIDTH_WINDOW_GAUSSIAN,
                      BANDWIDTH_WINDOW_KAISER}) {
        EXPECT_DOUBLE_EQ(
            MeasurementOperator::weight(params.k_center, params, type), 1.0)
            << "window type " << type;
    }
}

TEST(MeasurementOperator, WeightMatchesWindowFunction) {
    // W(k) = H(kx - kcx) · H(ky - kcy) with H the 1D window function
    MeasurementOperator::WindowParams params{
        .k_center = Eigen::Vector2d{1000.0, -200.0},
        .bandwidth_x = 500.0,
        .bandwidth_y = 400.0
    };

    Eigen::Vector2d k_query{1250.0, -100.0};
    double expected = BandwidthModel::sinc_window(250.0, 500.0) *
                      BandwidthModel::sinc_window(100.0, 400.0);

    double actual = MeasurementOperator::weight(
        k_query, params, BANDWIDTH_WINDOW_SINC);

    EXPECT_NEAR(actual, expected, 1e-12);
}

// =============================================================================
// Separability
// =============================================================================

TEST(MeasurementOperator, WeightIs2DSeparable) {
    MeasurementOperator::WindowParams params{
        .k_center = Eigen::Vector2d{1000.0, 500.0},
        .bandwidth_x = 300.0,
        .bandwidth_y = 400.0
    };

    // Query point offset from the center in both axes
    Eigen::Vector2d k_query{1100.0, 600.0};

    // 1D weights: offset along one axis only, the other at the center
    double w_x = MeasurementOperator::weight(
        Eigen::Vector2d{1100.0, 500.0}, params, BANDWIDTH_WINDOW_SINC);
    double w_y = MeasurementOperator::weight(
        Eigen::Vector2d{1000.0, 600.0}, params, BANDWIDTH_WINDOW_SINC);

    double w_2d = MeasurementOperator::weight(
        k_query, params, BANDWIDTH_WINDOW_SINC);

    // 2D weight equals the product of the 1D weights
    EXPECT_NEAR(w_2d, w_x * w_y, 1e-12);
}

// =============================================================================
// Sinc nulls
// =============================================================================

TEST(MeasurementOperator, SincWindowHasNullsAtBandwidthMultiples) {
    const double bw_x = 300.0;
    const double bw_y = 400.0;
    MeasurementOperator::WindowParams params{
        .k_center = Eigen::Vector2d{1000.0, 500.0},
        .bandwidth_x = bw_x,
        .bandwidth_y = bw_y
    };

    // sinc(Δk/bw) = 0 at Δk = n·bw for n ≠ 0
    for (int n : {1, 2, 3, -1, -2}) {
        Eigen::Vector2d k_null_x = params.k_center + Eigen::Vector2d{n * bw_x, 0.0};
        EXPECT_NEAR(MeasurementOperator::weight(k_null_x, params, BANDWIDTH_WINDOW_SINC),
                    0.0, 1e-12) << "kx null n=" << n;

        Eigen::Vector2d k_null_y = params.k_center + Eigen::Vector2d{0.0, n * bw_y};
        EXPECT_NEAR(MeasurementOperator::weight(k_null_y, params, BANDWIDTH_WINDOW_SINC),
                    0.0, 1e-12) << "ky null n=" << n;
    }

    // Between the first and second null the sinc sidelobe is negative
    Eigen::Vector2d k_sidelobe = params.k_center + Eigen::Vector2d{1.5 * bw_x, 0.0};
    EXPECT_LT(MeasurementOperator::weight(k_sidelobe, params, BANDWIDTH_WINDOW_SINC), 0.0);
}

TEST(MeasurementOperator, WeightVanishesFarFromCenter) {
    auto params = centered_window(100.0);

    // 50 bandwidths away: sinc sidelobe amplitude ~ 1/(50π)
    double w = MeasurementOperator::weight(
        Eigen::Vector2d{5000.0, 0.0}, params, BANDWIDTH_WINDOW_SINC);
    EXPECT_LT(std::abs(w), 1e-2);

    double w_gauss = MeasurementOperator::weight(
        Eigen::Vector2d{5000.0, 0.0}, params, BANDWIDTH_WINDOW_GAUSSIAN);
    EXPECT_LT(std::abs(w_gauss), 1e-12);
}

// =============================================================================
// Point target response
// =============================================================================

TEST(MeasurementOperator, PointTargetWeightAtOriginEqualsWindowIntegral) {
    // A point target at r = 0 has X(k) = 1, so the response is ∫W(k) dk.
    // Compare with integrate_response of the constant 1 at the same quadrature.
    auto params = centered_window(1000.0);
    const int quadrature = 11;

    auto one = [](const Eigen::Vector2d&) -> complex<double> { return {1.0, 0.0}; };

    for (auto type : {BANDWIDTH_WINDOW_SINC, BANDWIDTH_WINDOW_GAUSSIAN,
                      BANDWIDTH_WINDOW_KAISER}) {
        double w_origin = MeasurementOperator::point_target_weight(
            Eigen::Vector2d{0.0, 0.0}, params, type, quadrature);
        auto integral = MeasurementOperator::integrate_response(
            one, params, type, quadrature);

        EXPECT_GT(w_origin, 0.0) << "window type " << type;
        EXPECT_NEAR(w_origin, std::abs(integral), 1e-9 * std::abs(integral))
            << "window type " << type;
        EXPECT_NEAR(integral.imag(), 0.0, 1e-12) << "window type " << type;
    }
}

TEST(MeasurementOperator, PointTargetWeightDecreasesWithOffset) {
    // exp(i·k·r) varies across the window for r ≠ 0, so partial cancellation
    // reduces the integrated magnitude relative to the target at the origin.
    auto params = centered_window(1000.0);

    double w_origin = MeasurementOperator::point_target_weight(
        Eigen::Vector2d{0.0, 0.0}, params, BANDWIDTH_WINDOW_GAUSSIAN, 11);

    // With bandwidth = 1000 rad/m, a 10 mm offset spans many phase cycles
    double w_offset = MeasurementOperator::point_target_weight(
        Eigen::Vector2d{0.01, 0.01}, params, BANDWIDTH_WINDOW_GAUSSIAN, 11);

    EXPECT_LT(w_offset, w_origin);
}

// =============================================================================
// Integration of extended spectra
// =============================================================================

TEST(MeasurementOperator, IntegrateConstantEqualsWindowArea) {
    // ∫ c · W(k) dk = c · ∫ W(k) dk. For the Gaussian window the area within
    // the support box is analytic: bw_x · bw_y · 2π · erf(s/√2)², where s is
    // the number of bandwidths to the support edge.
    MeasurementOperator::WindowParams params{
        .k_center = Eigen::Vector2d{2000.0, -300.0},
        .bandwidth_x = 1000.0,
        .bandwidth_y = 700.0
    };
    const int quadrature = 101;

    auto bounds = MeasurementOperator::compute_support(
        params, 0.001, BANDWIDTH_WINDOW_GAUSSIAN);
    double s_x = (bounds.k_max_x - params.k_center.x()) / params.bandwidth_x;
    double s_y = (bounds.k_max_y - params.k_center.y()) / params.bandwidth_y;
    double expected_area = params.bandwidth_x * params.bandwidth_y * kTwoPi *
                           std::erf(s_x / std::sqrt(2.0)) *
                           std::erf(s_y / std::sqrt(2.0));

    auto one = [](const Eigen::Vector2d&) -> complex<double> { return {1.0, 0.0}; };
    auto area = MeasurementOperator::integrate_response(
        one, params, BANDWIDTH_WINDOW_GAUSSIAN, quadrature);

    EXPECT_NEAR(area.real(), expected_area, 1e-3 * expected_area);
    EXPECT_NEAR(area.imag(), 0.0, 1e-12);

    // Linearity in the constant
    const complex<double> c{2.5, -1.5};
    auto constant = [c](const Eigen::Vector2d&) -> complex<double> { return c; };
    auto scaled = MeasurementOperator::integrate_response(
        constant, params, BANDWIDTH_WINDOW_GAUSSIAN, quadrature);

    EXPECT_NEAR(scaled.real(), (c * area).real(), 1e-9 * expected_area);
    EXPECT_NEAR(scaled.imag(), (c * area).imag(), 1e-9 * expected_area);
}

TEST(MeasurementOperator, IntegrateResponseIsNotNormalized) {
    // A wider window integrates over a larger area and gives a larger result;
    // a normalized operator would give equal results.
    auto narrow = centered_window(100.0);
    auto wide = centered_window(500.0);

    auto one = [](const Eigen::Vector2d&) -> complex<double> { return {1.0, 0.0}; };

    auto result_narrow = MeasurementOperator::integrate_response(
        one, narrow, BANDWIDTH_WINDOW_GAUSSIAN, 9);
    auto result_wide = MeasurementOperator::integrate_response(
        one, wide, BANDWIDTH_WINDOW_GAUSSIAN, 9);

    EXPECT_GT(std::abs(result_wide), std::abs(result_narrow));
}

TEST(MeasurementOperator, IntegrateResponsePreservesPhase) {
    // A spectrum with a linear phase ramp integrates to a complex value with a
    // non-zero imaginary part; integrating |X| would discard it.
    auto params = centered_window(1000.0);

    auto X_k = [](const Eigen::Vector2d& k) -> complex<double> {
        return std::exp(complex<double>(0, k.x() * 0.001 + 0.3));
    };

    auto result = MeasurementOperator::integrate_response(
        X_k, params, BANDWIDTH_WINDOW_GAUSSIAN, 9);

    EXPECT_NE(result.imag(), 0.0);
}

// =============================================================================
// Support bounds
// =============================================================================

TEST(MeasurementOperator, SupportBoundsAreSymmetricAndContainCenter) {
    MeasurementOperator::WindowParams params{
        .k_center = Eigen::Vector2d{1000.0, -250.0},
        .bandwidth_x = 500.0,
        .bandwidth_y = 200.0
    };

    for (auto type : {BANDWIDTH_WINDOW_SINC, BANDWIDTH_WINDOW_GAUSSIAN,
                      BANDWIDTH_WINDOW_KAISER}) {
        auto bounds = MeasurementOperator::compute_support(params, 0.01, type);

        EXPECT_TRUE(bounds.is_valid()) << "window type " << type;
        EXPECT_TRUE(bounds.contains(params.k_center)) << "window type " << type;

        // Symmetric about the center in each axis
        EXPECT_NEAR(params.k_center.x() - bounds.k_min_x,
                    bounds.k_max_x - params.k_center.x(), 1e-9) << "window type " << type;
        EXPECT_NEAR(params.k_center.y() - bounds.k_min_y,
                    bounds.k_max_y - params.k_center.y(), 1e-9) << "window type " << type;

        // Half-width scales with the per-axis bandwidth
        double half_x = bounds.k_max_x - params.k_center.x();
        double half_y = bounds.k_max_y - params.k_center.y();
        EXPECT_NEAR(half_x / params.bandwidth_x, half_y / params.bandwidth_y, 1e-9)
            << "window type " << type;
    }
}

TEST(MeasurementOperator, SupportBoundsShrinkWithLooserThreshold) {
    auto params = centered_window(100.0);

    auto bounds_tight = MeasurementOperator::compute_support(
        params, 0.1, BANDWIDTH_WINDOW_GAUSSIAN);
    auto bounds_loose = MeasurementOperator::compute_support(
        params, 0.001, BANDWIDTH_WINDOW_GAUSSIAN);

    double width_tight = bounds_tight.k_max_x - bounds_tight.k_min_x;
    double width_loose = bounds_loose.k_max_x - bounds_loose.k_min_x;

    EXPECT_LT(width_tight, width_loose);
}

TEST(MeasurementOperator, SupportBoundsContainsMethod) {
    MeasurementOperator::SupportBounds bounds{
        .k_min_x = 100.0,
        .k_max_x = 200.0,
        .k_min_y = -50.0,
        .k_max_y = 50.0
    };

    // Interior point
    EXPECT_TRUE(bounds.contains(Eigen::Vector2d{150.0, 0.0}));

    // Boundary points are included
    EXPECT_TRUE(bounds.contains(Eigen::Vector2d{100.0, 0.0}));
    EXPECT_TRUE(bounds.contains(Eigen::Vector2d{200.0, 0.0}));

    // Exterior points
    EXPECT_FALSE(bounds.contains(Eigen::Vector2d{99.0, 0.0}));
    EXPECT_FALSE(bounds.contains(Eigen::Vector2d{201.0, 0.0}));
    EXPECT_FALSE(bounds.contains(Eigen::Vector2d{150.0, 51.0}));
}

// =============================================================================
// Argument validation
// =============================================================================

TEST(MeasurementOperator, ThrowsForUnspecifiedWindowType) {
    auto params = centered_window(100.0);

    EXPECT_THROW(
        MeasurementOperator::weight(
            Eigen::Vector2d{0.0, 0.0}, params, BANDWIDTH_WINDOW_UNSPECIFIED),
        std::invalid_argument);

    EXPECT_THROW(
        MeasurementOperator::compute_support(
            params, 0.01, BANDWIDTH_WINDOW_UNSPECIFIED),
        std::invalid_argument);

    EXPECT_THROW(
        MeasurementOperator::point_target_weight(
            Eigen::Vector2d{0.0, 0.0}, params, BANDWIDTH_WINDOW_UNSPECIFIED, 9),
        std::invalid_argument);
}

TEST(MeasurementOperator, ThrowsForNonPositiveQuadraturePoints) {
    auto params = centered_window(100.0);
    auto one = [](const Eigen::Vector2d&) -> complex<double> { return {1.0, 0.0}; };

    EXPECT_THROW(
        MeasurementOperator::integrate_response(one, params, BANDWIDTH_WINDOW_GAUSSIAN, 0),
        std::invalid_argument);
    EXPECT_THROW(
        MeasurementOperator::integrate_response(one, params, BANDWIDTH_WINDOW_GAUSSIAN, -3),
        std::invalid_argument);
    EXPECT_THROW(
        MeasurementOperator::point_target_weight(
            Eigen::Vector2d{0.0, 0.0}, params, BANDWIDTH_WINDOW_GAUSSIAN, 0),
        std::invalid_argument);

    // A single quadrature point is allowed
    EXPECT_NO_THROW(
        MeasurementOperator::integrate_response(one, params, BANDWIDTH_WINDOW_GAUSSIAN, 1));
}

// =============================================================================
// build_window_params from the ForwardModel
// =============================================================================

TEST(BuildWindowParams, UsesSceneKAndPixelBandwidths) {
    // 8-pixel linear array, 50 mm Fourier lens, small LO tilt in x
    DetectorGeometry detector;
    *detector.add_arrays() = aurora::test::create_linear_array(
        "window_params_test", 8, 0.5e-3, {1550e-9}, 150e6);

    OpticsConfig optics;
    optics.set_fourier_lens_focal_m(0.05);
    optics.add_wavelengths_m(1550e-9);
    optics.set_lo_tilt_x_rad(0.01);
    optics.set_lo_tilt_y_rad(0.0);

    ForwardModel forward_model =
        aurora::test::TestForwardModelBuilder::build(detector, optics);

    const double wavelength = 1550e-9;
    for (uint32_t det = 0; det < forward_model.num_detectors(); ++det) {
        auto params = build_window_params(forward_model, det, wavelength);

        // Center is the pixel's scene K (detector K plus LO tilt)
        Eigen::Vector2d expected_center = forward_model.get_effective_k(det, wavelength);
        EXPECT_DOUBLE_EQ(params.k_center.x(), expected_center.x()) << "detector " << det;
        EXPECT_DOUBLE_EQ(params.k_center.y(), expected_center.y()) << "detector " << det;

        // Bandwidths are the pixel's K-space acceptance bandwidths
        auto k_sample = forward_model.get_k_sample(det, wavelength);
        EXPECT_DOUBLE_EQ(params.bandwidth_x, k_sample.kx_bandwidth()) << "detector " << det;
        EXPECT_DOUBLE_EQ(params.bandwidth_y, k_sample.ky_bandwidth()) << "detector " << det;
        EXPECT_GT(params.bandwidth_x, 0.0);
        EXPECT_GT(params.bandwidth_y, 0.0);
    }
}

TEST(BuildWindowParams, BandwidthFollowsPixelSize) {
    // Δk = 2π·w/(f·λ) for a pixel of width w
    const double pitch = 0.5e-3;
    const double focal = 0.05;
    const double wavelength = 1550e-9;

    DetectorGeometry detector;
    *detector.add_arrays() = aurora::test::create_linear_array(
        "window_params_bw", 4, pitch, {wavelength}, 150e6);
    const auto& pixel = detector.arrays(0).pixels(1);

    OpticsConfig optics;
    optics.set_fourier_lens_focal_m(focal);
    optics.add_wavelengths_m(wavelength);

    ForwardModel forward_model =
        aurora::test::TestForwardModelBuilder::build(detector, optics);
    auto params = build_window_params(forward_model, 1, wavelength);

    EXPECT_NEAR(params.bandwidth_x, kTwoPi * pixel.width_m() / (focal * wavelength), 1e-6);
    EXPECT_NEAR(params.bandwidth_y, kTwoPi * pixel.height_m() / (focal * wavelength), 1e-6);

    // Without LO tilt the center is the geometric detector K
    EXPECT_NEAR(params.k_center.x(), kTwoPi * pixel.x_m() / (focal * wavelength), 1e-6);
    EXPECT_NEAR(params.k_center.y(), 0.0, 1e-9);
}

TEST(BuildWindowParams, ThrowsForInvalidDetector) {
    DetectorGeometry detector;
    *detector.add_arrays() = aurora::test::create_linear_array(
        "window_params_throw", 8, 0.5e-3, {1550e-9}, 150e6);

    OpticsConfig optics;
    optics.set_fourier_lens_focal_m(0.05);
    optics.add_wavelengths_m(1550e-9);

    ForwardModel forward_model =
        aurora::test::TestForwardModelBuilder::build(detector, optics);

    EXPECT_THROW(build_window_params(forward_model, 8, 1550e-9), std::out_of_range);
}

}  // namespace
}  // namespace aurora::forward
