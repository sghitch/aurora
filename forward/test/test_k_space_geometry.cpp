#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "forward/bandwidth_model.h"
#include "forward/k_space_geometry.h"
#include "test_forward_model_builder.h"

namespace aurora::forward {

using aurora::test::create_linear_array;
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

/// Optics K-limit used by the fixture (rad/m).
constexpr double kTestKMax = 2.0e5;

class KSpaceGeometryTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up the test optics configuration
        optics_.set_fourier_lens_focal_m(0.05);  // 50mm focal length
        optics_.set_k_max_rad_per_m(kTestKMax);
        optics_.set_lo_tilt_x_rad(0.02);  // ~1.15 degrees
        optics_.set_lo_tilt_y_rad(0.0);

        // Set up 8-pixel linear detector array (0.5mm pitch)
        detector_ = create_linear_array("test_8px", 8, 0.5e-3,
                                        {905e-9, 1550e-9}, 150e6);
    }

    OpticsConfig optics_;
    DetectorArray detector_;
};

// =============================================================================
// Constructor and Accessor Tests
// =============================================================================

TEST_F(KSpaceGeometryTest, ConstructorStoresConfigs) {
    KSpaceGeometry geom(optics_, detector_);

    EXPECT_EQ(geom.num_detectors(), 8u);
    EXPECT_NEAR(geom.focal_length_m(), 0.05, 1e-12);

    auto tilt = geom.lo_tilt_rad();
    EXPECT_NEAR(tilt.x(), 0.02, 1e-12);
    EXPECT_NEAR(tilt.y(), 0.0, 1e-12);
}

// =============================================================================
// Detector K-Position Tests
// =============================================================================

TEST_F(KSpaceGeometryTest, DetectorKPositionFormula) {
    KSpaceGeometry geom(optics_, detector_);

    double wavelength = 1550e-9;
    double f = optics_.fourier_lens_focal_m();

    // For the center-ish pixel (index 4), check the formula
    const auto& pixel = detector_.pixels(4);
    double x_det = pixel.x_m();

    double expected_kx = kTwoPi * x_det / (f * wavelength);
    double actual_kx = geom.detector_k_x(4, wavelength);

    EXPECT_NEAR(actual_kx, expected_kx, 1e-9);
}

TEST_F(KSpaceGeometryTest, DetectorKPositionVector) {
    KSpaceGeometry geom(optics_, detector_);

    double wavelength = 1550e-9;

    auto k = geom.detector_k(3, wavelength);

    EXPECT_NEAR(k.x(), geom.detector_k_x(3, wavelength), 1e-12);
    EXPECT_NEAR(k.y(), geom.detector_k_y(3, wavelength), 1e-12);
}

TEST_F(KSpaceGeometryTest, DetectorToKFormula) {
    KSpaceGeometry geom(optics_, detector_);

    double wavelength = 1550e-9;
    double f = optics_.fourier_lens_focal_m();
    double x_m = 1.25e-3;

    // k = 2π · x / (f · λ)
    EXPECT_NEAR(geom.detector_to_k(x_m, wavelength),
                kTwoPi * x_m / (f * wavelength), 1e-9);
    EXPECT_NEAR(geom.detector_to_k(0.0, wavelength), 0.0, 1e-12);
    EXPECT_NEAR(geom.detector_to_k(-x_m, wavelength),
                -geom.detector_to_k(x_m, wavelength), 1e-9);
}

TEST_F(KSpaceGeometryTest, DetectorToKMatchesPixelKPositions) {
    KSpaceGeometry geom(optics_, detector_);

    double wavelength = 905e-9;
    for (uint32_t i = 0; i < 8; ++i) {
        double x_m = detector_.pixels(static_cast<int>(i)).x_m();
        EXPECT_NEAR(geom.detector_k_x(i, wavelength),
                    geom.detector_to_k(x_m, wavelength), 1e-9)
            << "Pixel " << i;
    }
}

TEST_F(KSpaceGeometryTest, DetectorToKThrowsOnInvalidWavelength) {
    KSpaceGeometry geom(optics_, detector_);

    EXPECT_THROW(geom.detector_to_k(1e-3, 0.0), std::invalid_argument);
    EXPECT_THROW(geom.detector_to_k(1e-3, -1e-9), std::invalid_argument);
}

TEST_F(KSpaceGeometryTest, AllDetectorKReturnsCorrectCount) {
    KSpaceGeometry geom(optics_, detector_);

    auto all_k = geom.all_detector_k(1550e-9);

    EXPECT_EQ(all_k.size(), 8u);

    // Verify each element matches individual call
    for (uint32_t i = 0; i < 8; ++i) {
        auto k = geom.detector_k(i, 1550e-9);
        EXPECT_NEAR(all_k[i].x(), k.x(), 1e-12);
        EXPECT_NEAR(all_k[i].y(), k.y(), 1e-12);
    }
}

TEST_F(KSpaceGeometryTest, WavelengthScalingInverseProportional) {
    // Key physics test: k ∝ 1/λ
    KSpaceGeometry geom(optics_, detector_);

    double lambda1 = 905e-9;
    double lambda2 = 1550e-9;

    double k1 = geom.detector_k_x(4, lambda1);
    double k2 = geom.detector_k_x(4, lambda2);

    // k1 / k2 should equal λ2 / λ1
    double expected_ratio = lambda2 / lambda1;
    double actual_ratio = k1 / k2;

    EXPECT_NEAR(actual_ratio, expected_ratio, 1e-9);
}

TEST_F(KSpaceGeometryTest, LinearArrayHasKyZero) {
    // For a linear array along x, all y positions are 0
    KSpaceGeometry geom(optics_, detector_);

    for (uint32_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(geom.detector_k_y(i, 1550e-9), 0.0, 1e-12)
            << "Pixel " << i << " should have ky = 0";
    }
}

TEST_F(KSpaceGeometryTest, CenteredArrayHasSymmetricK) {
    // Array is centered, so K-values should be symmetric around 0
    KSpaceGeometry geom(optics_, detector_);

    double wavelength = 1550e-9;

    // First and last pixels should have opposite K
    double k_first = geom.detector_k_x(0, wavelength);
    double k_last = geom.detector_k_x(7, wavelength);

    EXPECT_NEAR(k_first, -k_last, 1e-9);
}

TEST_F(KSpaceGeometryTest, DetectorKThrowsOnInvalidIndex) {
    KSpaceGeometry geom(optics_, detector_);

    EXPECT_THROW(geom.detector_k_x(8, 1550e-9), std::out_of_range);
    EXPECT_THROW(geom.detector_k_y(100, 1550e-9), std::out_of_range);
    EXPECT_THROW(geom.detector_k(8, 1550e-9), std::out_of_range);
}

TEST_F(KSpaceGeometryTest, DetectorKThrowsOnInvalidWavelength) {
    KSpaceGeometry geom(optics_, detector_);

    EXPECT_THROW(geom.detector_k_x(0, 0.0), std::invalid_argument);
    EXPECT_THROW(geom.detector_k_x(0, -1e-9), std::invalid_argument);
    EXPECT_THROW(geom.all_detector_k(0.0), std::invalid_argument);
}

// =============================================================================
// Detector K-Space Window Tests
// =============================================================================

TEST_F(KSpaceGeometryTest, DetectorKBandwidthFormula) {
    KSpaceGeometry geom(optics_, detector_);

    double wavelength = 1550e-9;
    double f = optics_.fourier_lens_focal_m();

    // Pixel 0 has width from create_linear_array: pitch * 0.8 = 0.5e-3 * 0.8
    const auto& pixel = detector_.pixels(0);
    double w_det = pixel.width_m();
    double expected = kTwoPi * w_det / (f * wavelength);

    double actual = geom.detector_k_bandwidth_x(0, wavelength);

    EXPECT_NEAR(actual, expected, 1e-6);
    EXPECT_GT(actual, 0.0);  // Sanity check: bandwidth should be positive
}

TEST_F(KSpaceGeometryTest, DetectorKBandwidthScalesInverselyWithWavelength) {
    // Key physics test: Δk ∝ 1/λ
    KSpaceGeometry geom(optics_, detector_);

    double lambda1 = 905e-9;
    double lambda2 = 1550e-9;

    double bw1 = geom.detector_k_bandwidth_x(0, lambda1);
    double bw2 = geom.detector_k_bandwidth_x(0, lambda2);

    // bw1 / bw2 should equal λ2 / λ1
    double expected_ratio = lambda2 / lambda1;
    double actual_ratio = bw1 / bw2;

    EXPECT_NEAR(actual_ratio, expected_ratio, 1e-9);
}

TEST_F(KSpaceGeometryTest, DetectorKBandwidthYFormula) {
    KSpaceGeometry geom(optics_, detector_);

    double wavelength = 1550e-9;
    double f = optics_.fourier_lens_focal_m();

    const auto& pixel = detector_.pixels(0);
    double h_det = pixel.height_m();
    double expected = kTwoPi * h_det / (f * wavelength);

    double actual = geom.detector_k_bandwidth_y(0, wavelength);

    EXPECT_NEAR(actual, expected, 1e-6);
}

TEST_F(KSpaceGeometryTest, DetectorKBandwidthThrowsOnInvalidIndex) {
    KSpaceGeometry geom(optics_, detector_);

    EXPECT_THROW(geom.detector_k_bandwidth_x(8, 1550e-9), std::out_of_range);
    EXPECT_THROW(geom.detector_k_bandwidth_y(100, 1550e-9), std::out_of_range);
}

TEST_F(KSpaceGeometryTest, DetectorKBandwidthThrowsOnInvalidWavelength) {
    KSpaceGeometry geom(optics_, detector_);

    EXPECT_THROW(geom.detector_k_bandwidth_x(0, 0.0), std::invalid_argument);
    EXPECT_THROW(geom.detector_k_bandwidth_y(0, -1e-9), std::invalid_argument);
}

// =============================================================================
// LO Tilt Tests
// =============================================================================

TEST_F(KSpaceGeometryTest, LOTiltKFormula) {
    KSpaceGeometry geom(optics_, detector_);

    double wavelength = 1550e-9;

    // k_LO = 2π × sin(θ) / λ
    double expected_kx = kTwoPi * std::sin(0.02) / wavelength;
    double expected_ky = 0.0;  // lo_tilt_y_rad = 0

    auto lo_k = geom.lo_tilt_k(wavelength);

    EXPECT_NEAR(lo_k.x(), expected_kx, 1e-3);
    EXPECT_NEAR(lo_k.y(), expected_ky, 1e-12);
}

TEST_F(KSpaceGeometryTest, LOTiltKScalesWithWavelength) {
    // LO tilt K is proportional to 1/λ
    KSpaceGeometry geom(optics_, detector_);

    double lambda1 = 905e-9;
    double lambda2 = 1550e-9;

    auto lo_k1 = geom.lo_tilt_k(lambda1);
    auto lo_k2 = geom.lo_tilt_k(lambda2);

    // k1 / k2 should equal λ2 / λ1
    double expected_ratio = lambda2 / lambda1;
    double actual_ratio = lo_k1.x() / lo_k2.x();

    EXPECT_NEAR(actual_ratio, expected_ratio, 1e-9);
}

TEST_F(KSpaceGeometryTest, LOTiltKThrowsOnInvalidWavelength) {
    KSpaceGeometry geom(optics_, detector_);

    EXPECT_THROW(geom.lo_tilt_k(0.0), std::invalid_argument);
    EXPECT_THROW(geom.lo_tilt_k(-1e-9), std::invalid_argument);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(KSpaceGeometryTest, ZeroLOTiltCentersAtOrigin) {
    optics_.set_lo_tilt_x_rad(0.0);
    optics_.set_lo_tilt_y_rad(0.0);

    KSpaceGeometry geom(optics_, detector_);

    auto lo_k = geom.lo_tilt_k(1550e-9);
    EXPECT_NEAR(lo_k.x(), 0.0, 1e-12);
    EXPECT_NEAR(lo_k.y(), 0.0, 1e-12);
}

TEST_F(KSpaceGeometryTest, DiagonalLOTilt) {
    optics_.set_lo_tilt_x_rad(0.01);
    optics_.set_lo_tilt_y_rad(0.01);

    KSpaceGeometry geom(optics_, detector_);

    auto lo_k = geom.lo_tilt_k(1550e-9);

    // Both x and y should have equal non-zero values
    EXPECT_NEAR(lo_k.x(), lo_k.y(), 1e-12);
    EXPECT_GT(lo_k.x(), 0.0);
}

// =============================================================================
// Detector Sensitivity Tests
// =============================================================================

TEST_F(KSpaceGeometryTest, DetectorSensitivityDelegatesToBandwidthModel) {
    // Verify K-space sinc sensitivity uses BandwidthModel::sinc_window
    KSpaceGeometry geometry(optics_, detector_);

    double wavelength = 1550e-9;
    double bw_kx = geometry.detector_k_bandwidth_x(0, wavelength);
    double bw_ky = geometry.detector_k_bandwidth_y(0, wavelength);

    // Test at delta_k = 0 (center) - should be 1.0
    Eigen::Vector2d delta_k_zero(0.0, 0.0);
    double sens_center = geometry.detector_sensitivity(0, wavelength, delta_k_zero);
    EXPECT_NEAR(sens_center, 1.0, 1e-10);

    // Test at delta_k = bandwidth (first null of sinc)
    Eigen::Vector2d delta_k_null(bw_kx, 0.0);
    double sens_null = geometry.detector_sensitivity(0, wavelength, delta_k_null);
    // For normalized sinc, sinc(1) = 0 exactly; we use a loose tolerance for numerical precision
    EXPECT_LT(std::abs(sens_null), 0.1);

    // Verify matches BandwidthModel::sinc_window directly
    Eigen::Vector2d test_delta_k(bw_kx / 2.0, bw_ky / 3.0);
    double expected = BandwidthModel::sinc_window(test_delta_k.x(), bw_kx) *
                      BandwidthModel::sinc_window(test_delta_k.y(), bw_ky);
    double actual = geometry.detector_sensitivity(0, wavelength, test_delta_k);
    EXPECT_DOUBLE_EQ(actual, expected);
}

}  // namespace
}  // namespace aurora::forward
