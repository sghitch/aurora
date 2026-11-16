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

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <optional>
#include <random>

#include "forward/forward_model.h"
#include "forward/k_space_geometry.h"
#include "test_forward_model_builder.h"

namespace aurora::forward {

namespace {

constexpr double kTwoPi = 2.0 * M_PI;

// Physics Validation Test Suite
//
// These tests verify fundamental optical physics invariants that must hold
// for the Forward module to correctly implement coherent sensing.
//
// Physical model summary:
// - Each detector pixel samples a fixed spatial frequency: k = 2*pi * x_det / (f * lambda)
// - Fourier lens maps K-space to detector positions: x_det = f * lambda * k / (2*pi)
// - Phase encodes target position: phi = -k . r
// - K-vectors scale inversely with wavelength: k is proportional to 1/lambda

class PhysicsValidationTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        // Set up detector geometry (8 pixels, 0.5mm pitch)
        auto* array = detector_.add_arrays();
        array->set_array_id("physics_test_detector");
        array->set_nominal_pitch_m(0.5e-3);
        array->add_wavelengths_m(905e-9);
        array->add_wavelengths_m(1550e-9);
        array->set_sensitivity_type(DETECTOR_SENSITIVITY_SINC);

        const int num_pixels = 8;
        const double pitch = 0.5e-3;
        const double pixel_width = 0.4e-3;

        for (int i = 0; i < num_pixels; ++i) {
            auto* pixel = array->add_pixels();
            pixel->set_index(i);
            pixel->set_x_m((i - (num_pixels - 1) / 2.0) * pitch);
            pixel->set_y_m(0.0);
            pixel->set_width_m(pixel_width);
            pixel->set_height_m(pixel_width);
        }

        // Set up optics configuration (50mm Fourier lens)
        optics_.set_fourier_lens_focal_m(0.05);  // 50mm focal length
        optics_.add_wavelengths_m(905e-9);
        optics_.add_wavelengths_m(1550e-9);

        // Build the forward model using test helper
        forward_model_.emplace(aurora::test::TestForwardModelBuilder::build(
            detector_, optics_));

        // Create K-space geometry helper
        k_geometry_.emplace(optics_, detector_.arrays(0));
    }

    static void TearDownTestSuite() {
        forward_model_.reset();
        k_geometry_.reset();
    }

    static DetectorGeometry detector_;
    static OpticsConfig optics_;
    static std::optional<ForwardModel> forward_model_;
    static std::optional<KSpaceGeometry> k_geometry_;
};

// Static member definitions
DetectorGeometry PhysicsValidationTest::detector_;
OpticsConfig PhysicsValidationTest::optics_;
std::optional<ForwardModel> PhysicsValidationTest::forward_model_;
std::optional<KSpaceGeometry> PhysicsValidationTest::k_geometry_;

// ============================================================================
// Test 1: Diffraction Physics
// ============================================================================
//
// Verify that spatial frequency maps to propagation angle as
//   sin(theta) = lambda * k / (2*pi)
//
// This tests that the optical model correctly relates spatial frequency
// (K-vector) to angle in the scene.
//
// Acceptance: Within 0.1% of the analytic relation
//
TEST_F(PhysicsValidationTest, DiffractionPhysics) {
    // Test parameters
    const double lambda = 905e-9;     // wavelength in meters
    const double f = 0.05;            // focal length in meters

    ASSERT_TRUE(k_geometry_.has_value());

    // Test a range of Fourier-plane positions
    for (double x_detector : {0.1e-3, 0.5e-3, 1.0e-3, 1.75e-3}) {
        // Propagation angle of the plane wave focused at x_detector:
        //   tan(theta) = x / f
        double theta = std::atan(x_detector / f);

        // Spatial frequency of that plane wave from the diffraction relation
        //   k = 2*pi * sin(theta) / lambda
        double k_expected = kTwoPi * std::sin(theta) / lambda;

        // KSpaceGeometry uses the paraxial form k = 2*pi * x / (f * lambda),
        // which equals the exact relation with sin(theta) replaced by tan(theta).
        // For x = 1.75 mm at f = 50 mm, theta is about 0.035 rad, so the
        // difference is well below 0.1%.
        double k_actual = k_geometry_->detector_to_k(x_detector, lambda);

        double relative_error = std::abs(k_actual - k_expected) / k_expected;

        EXPECT_LT(relative_error, 0.001)
            << "x_detector = " << x_detector * 1e3 << " mm: "
            << "expected k = " << k_expected << " rad/m, "
            << "got k = " << k_actual << " rad/m, "
            << "relative error = " << relative_error * 100 << "%";
    }
}

// ============================================================================
// Test 2: Fourier Relationship
// ============================================================================
//
// Verify the detector-to-K relationship of the Fourier lens:
//   k = 2*pi * x_detector / (f * lambda)
// so that a plane wave with spatial frequency k focuses at
//   x_detector = f * lambda * k / (2*pi)
//
// This tests the fundamental optical Fourier transform performed by the lens.
//
// Acceptance: Matches analytical FT within 1%
//
TEST_F(PhysicsValidationTest, FourierRelationship) {
    ASSERT_TRUE(k_geometry_.has_value());

    const double f = optics_.fourier_lens_focal_m();

    // Test for both wavelengths
    for (double lambda : {905e-9, 1550e-9}) {
        // Test a range of K-vectors
        for (double k_scene : {10000.0, 50000.0, 100000.0}) {
            // Fourier-plane position where this spatial frequency focuses
            // x = f * lambda * k / (2*pi)
            double x_focus = f * lambda * k_scene / kTwoPi;

            // A detector at that position samples k_scene
            double k_recovered = k_geometry_->detector_to_k(x_focus, lambda);

            // Verify within 1%
            double k_relative_error = std::abs(k_recovered - k_scene) / k_scene;
            EXPECT_LT(k_relative_error, 0.01)
                << "lambda = " << lambda * 1e9 << " nm, k = " << k_scene << " rad/m: "
                << "detector at x = " << x_focus * 1e6 << " um "
                << "recovered k = " << k_recovered << " rad/m";
        }
    }
}

// ============================================================================
// Test 3: Wavelength K-Scaling
// ============================================================================
//
// Verify K-vectors scale inversely with wavelength:
//   k is proportional to 1/lambda  ->  k(lambda1)/k(lambda2) = lambda2/lambda1
//
// This is fundamental to the Fourier optics: the same detector position
// corresponds to different K-vectors at different wavelengths.
//
// Physics: k = 2*pi * x_det / (f * lambda)
// For the same detector position x_det:
//   k1/k2 = lambda2/lambda1
//
// Acceptance: k proportional to 1/lambda within 0.1%
//
TEST_F(PhysicsValidationTest, WavelengthKScaling) {
    ASSERT_TRUE(k_geometry_.has_value());

    const double lambda1 = 905e-9;
    const double lambda2 = 1550e-9;
    const double expected_ratio = lambda2 / lambda1;  // approximately 1.713

    // Test with various detector positions
    // Same detector position, different wavelengths -> K scales as 1/lambda
    for (double x_detector : {0.5e-3, 1.0e-3, 1.5e-3, -1.0e-3}) {
        double k_at_905 = k_geometry_->detector_to_k(x_detector, lambda1);
        double k_at_1550 = k_geometry_->detector_to_k(x_detector, lambda2);

        double k_ratio = k_at_905 / k_at_1550;

        EXPECT_NEAR(k_ratio, expected_ratio, expected_ratio * 0.001)
            << "x_detector = " << x_detector * 1e3 << " mm: "
            << "expected k ratio " << expected_ratio
            << ", got " << k_ratio
            << " (k_905 = " << k_at_905 << ", k_1550 = " << k_at_1550 << ")";
    }

    // Verify the formula directly: k = 2*pi * x / (f * lambda)
    const double f = optics_.fourier_lens_focal_m();
    for (double x : {0.5e-3, 1.0e-3}) {
        for (double lambda : {905e-9, 1550e-9}) {
            double k_expected = kTwoPi * x / (f * lambda);
            double k_actual = k_geometry_->detector_to_k(x, lambda);

            EXPECT_NEAR(k_actual, k_expected, k_expected * 0.001)
                << "Formula verification failed: x = " << x * 1e3 << " mm, lambda = " << lambda * 1e9 << " nm";
        }
    }
}

// ============================================================================
// Test 4: Phase Position Encoding
// ============================================================================
//
// Verify phase correctly encodes target position:
//   phi = -k . r = -kx * x - ky * y
//
// This is the fundamental relationship that enables position estimation
// from K-space phase measurements.
//
// Acceptance: phi = -k.r within 10^-10 rad
//
TEST_F(PhysicsValidationTest, PhasePositionEncoding) {
    // Test with various K-vectors
    for (double kx : {10000.0, 50000.0, 100000.0}) {
        for (double ky : {0.0, 20000.0, 40000.0}) {
            Eigen::Vector2d k_vec(kx, ky);

            // Test with various target positions
            for (double x : {0.0, 1e-3, 2e-3, -1e-3}) {
                for (double y : {0.0, 0.5e-3, -0.5e-3}) {
                    Eigen::Vector2d target_pos(x, y);

                    // Expected phase: phi = -k . r
                    double expected_phase = -(kx * x + ky * y);

                    // Compute using ForwardModel static method
                    double computed_phase = ForwardModel::compute_phase(k_vec, target_pos);

                    // compute_phase should match formula exactly
                    EXPECT_NEAR(computed_phase, expected_phase, 1e-10)
                        << "compute_phase mismatch at k=(" << kx << ", " << ky << "), "
                        << "r=(" << x * 1e3 << ", " << y * 1e3 << ") mm";

                    // Also verify phasor
                    auto phasor = ForwardModel::compute_phasor(k_vec, target_pos);
                    double phasor_phase = std::arg(phasor);
                    double phasor_mag = std::abs(phasor);

                    // Phasor should have unit magnitude
                    EXPECT_NEAR(phasor_mag, 1.0, 1e-10)
                        << "Phasor should have unit magnitude";

                    // Normalize phases to [-pi, pi] for comparison
                    auto normalize_phase = [](double p) {
                        while (p > M_PI) p -= kTwoPi;
                        while (p < -M_PI) p += kTwoPi;
                        return p;
                    };

                    double expected_normalized = normalize_phase(expected_phase);
                    double phasor_normalized = phasor_phase;  // std::arg already in [-pi, pi]

                    // Compute phase difference handling wraparound
                    double phase_diff = std::abs(expected_normalized - phasor_normalized);
                    if (phase_diff > M_PI) phase_diff = kTwoPi - phase_diff;

                    EXPECT_LT(phase_diff, 1e-10)
                        << "Phasor phase mismatch at k=(" << kx << ", " << ky << "), "
                        << "r=(" << x * 1e3 << ", " << y * 1e3 << ") mm: "
                        << "expected " << expected_normalized
                        << ", got " << phasor_normalized;
                }
            }
        }
    }
}

// Verify that target at origin produces zero phase for all K-vectors
TEST_F(PhysicsValidationTest, OriginProducesZeroPhase) {
    Eigen::Vector2d origin(0.0, 0.0);

    for (double kx : {0.0, 50000.0, 100000.0, -80000.0}) {
        for (double ky : {0.0, 30000.0, -40000.0}) {
            Eigen::Vector2d k_vec(kx, ky);

            double phase = ForwardModel::compute_phase(k_vec, origin);
            EXPECT_NEAR(phase, 0.0, 1e-10)
                << "Phase at origin should be 0, got " << phase
                << " for k=(" << kx << ", " << ky << ")";

            auto phasor = ForwardModel::compute_phasor(k_vec, origin);
            EXPECT_NEAR(std::real(phasor), 1.0, 1e-10)
                << "Phasor at origin should be 1+0i";
            EXPECT_NEAR(std::imag(phasor), 0.0, 1e-10)
                << "Phasor at origin should be 1+0i";
        }
    }
}

// Verify phase gradient is linear in position
TEST_F(PhysicsValidationTest, PhaseGradientLinear) {
    const double kx = 50000.0;
    const double ky = 20000.0;
    Eigen::Vector2d k_vec(kx, ky);

    // Three positions along x-axis
    Eigen::Vector2d pos0(0.0, 0.0);
    Eigen::Vector2d pos1(1e-4, 0.0);
    Eigen::Vector2d pos2(2e-4, 0.0);

    double phi0 = ForwardModel::compute_phase(k_vec, pos0);
    double phi1 = ForwardModel::compute_phase(k_vec, pos1);
    double phi2 = ForwardModel::compute_phase(k_vec, pos2);

    // Phase gradient: delta_phi/delta_x = -kx
    double delta_phi_01 = phi1 - phi0;
    double delta_phi_12 = phi2 - phi1;

    // Both increments should be equal (linear)
    EXPECT_NEAR(delta_phi_01, delta_phi_12, 1e-10)
        << "Phase gradient not linear";

    // And should equal -kx * delta_x
    double expected_delta = -kx * 1e-4;
    EXPECT_NEAR(delta_phi_01, expected_delta, 1e-10)
        << "Phase gradient doesn't match -kx * delta_x";

    // Same test along y-axis
    Eigen::Vector2d pos_y1(0.0, 1e-4);
    Eigen::Vector2d pos_y2(0.0, 2e-4);

    double phi_y0 = ForwardModel::compute_phase(k_vec, pos0);
    double phi_y1 = ForwardModel::compute_phase(k_vec, pos_y1);
    double phi_y2 = ForwardModel::compute_phase(k_vec, pos_y2);

    double delta_phi_y01 = phi_y1 - phi_y0;
    double delta_phi_y12 = phi_y2 - phi_y1;

    EXPECT_NEAR(delta_phi_y01, delta_phi_y12, 1e-10)
        << "Phase gradient not linear in y";

    double expected_delta_y = -ky * 1e-4;
    EXPECT_NEAR(delta_phi_y01, expected_delta_y, 1e-10)
        << "Phase gradient doesn't match -ky * delta_y";
}

// ============================================================================
// Test 5: ForwardModel K-space Sampling
// ============================================================================
//
// Verify ForwardModel provides K-samples for all detectors and that the
// K-space span matches expected values from geometry.
//
TEST_F(PhysicsValidationTest, ForwardModelKSpaceSampling) {
    ASSERT_TRUE(forward_model_.has_value());

    const double wavelength = 1550e-9;

    // Get the K-samples of every detector pixel
    auto k_samples = forward_model_->get_all_k_samples(wavelength);

    EXPECT_EQ(k_samples.size(), 8u) << "Expected 8 K-samples (one per detector)";

    // Verify K-samples span expected range
    double k_min = std::numeric_limits<double>::max();
    double k_max = std::numeric_limits<double>::lowest();

    for (const auto& sample : k_samples) {
        k_min = std::min(k_min, sample.kx_eff());
        k_max = std::max(k_max, sample.kx_eff());
    }

    // Detector span is approximately 3.5mm (8 pixels at 0.5mm pitch)
    // k_range = 2*pi * x_span / (f * lambda)
    const double f = optics_.fourier_lens_focal_m();
    const double x_span = 7 * 0.5e-3;  // 7 gaps between 8 pixels
    const double expected_k_range = kTwoPi * x_span / (f * wavelength);

    double actual_k_range = k_max - k_min;
    EXPECT_NEAR(actual_k_range, expected_k_range, expected_k_range * 0.05)
        << "K-space span should match detector geometry. "
        << "Expected " << expected_k_range << " rad/m, got " << actual_k_range << " rad/m";
}

// ============================================================================
// Test 6: K-Sample Formula
// ============================================================================
//
// For every detector pixel, k = 2*pi * x_det / (f * lambda)
//
TEST_F(PhysicsValidationTest, KSampleFormula) {
    const double f = optics_.fourier_lens_focal_m();

    // Get detector positions from geometry
    const auto& array = detector_.arrays(0);

    for (double wavelength : {905e-9, 1550e-9}) {
        // Get K-samples from forward model
        auto k_samples = forward_model_->get_all_k_samples(wavelength);
        ASSERT_EQ(k_samples.size(), 8u);

        for (int d = 0; d < 8; ++d) {
            const auto& pixel = array.pixels(d);
            double x_det = pixel.x_m();

            // Expected k = 2*pi * x_det / (f * lambda)
            double expected_kx = kTwoPi * x_det / (f * wavelength);
            double expected_ky = kTwoPi * pixel.y_m() / (f * wavelength);

            // Get actual from forward model
            const auto& sample = k_samples[d];
            EXPECT_EQ(sample.detector_index(), static_cast<uint32_t>(d));
            EXPECT_NEAR(sample.wavelength_m(), wavelength, 1e-15);

            EXPECT_NEAR(sample.kx_eff(), expected_kx, 1e-6)
                << "Detector " << d << " at " << wavelength * 1e9
                << " nm: kx should be 2*pi * x_det / (f * lambda). "
                << "Expected " << expected_kx << ", got " << sample.kx_eff();
            EXPECT_NEAR(sample.ky_eff(), expected_ky, 1e-6)
                << "Detector " << d << " at " << wavelength * 1e9
                << " nm: ky should be 2*pi * y_det / (f * lambda). "
                << "Expected " << expected_ky << ", got " << sample.ky_eff();
        }
    }
}

// ============================================================================
// Test 7: Detector Sensitivity - SINC at Center
// ============================================================================
//
// Verify SINC sensitivity = 1.0 at the center (delta_k = 0).
// The sinc function has its maximum of 1.0 at zero argument.
//
TEST_F(PhysicsValidationTest, SincSensitivityAtCenter) {
    ASSERT_TRUE(k_geometry_.has_value());

    const double wavelength = 1550e-9;
    const Eigen::Vector2d zero_offset(0.0, 0.0);

    // Test detector sensitivity at center for all detectors
    for (uint32_t det_idx = 0; det_idx < k_geometry_->num_detectors(); ++det_idx) {
        double sensitivity = k_geometry_->detector_sensitivity(det_idx, wavelength, zero_offset);

        EXPECT_DOUBLE_EQ(sensitivity, 1.0)
            << "Detector " << det_idx << ": sensitivity at center should be 1.0, got "
            << sensitivity;
    }
}

// ============================================================================
// Test 8: Detector Sensitivity - SINC First Null
// ============================================================================
//
// Verify SINC sensitivity ~0 at the first null where sinc(1) = 0.
// First null occurs when argument = 1, i.e., delta_k = 2*pi / width
// More precisely: sinc(x) = sin(pi*x)/(pi*x), null at x = 1
// So delta_k * (width/2) = pi  =>  delta_k = 2*pi/width
//
TEST_F(PhysicsValidationTest, SincSensitivityAtFirstNull) {
    ASSERT_TRUE(k_geometry_.has_value());

    const double wavelength = 1550e-9;
    const uint32_t det_idx = 0;

    // Get detector K-bandwidth to find the null location
    double bw_x = k_geometry_->detector_k_bandwidth_x(det_idx, wavelength);
    double bw_y = k_geometry_->detector_k_bandwidth_y(det_idx, wavelength);

    // First null of sinc at delta_k = bandwidth (for sinc normalized as sin(pi*x)/(pi*x))
    // The detector sensitivity formula uses sinc(dk * w/2) where w is width in real space
    // But k_bandwidth = 2*pi*w/(f*lambda), so dk at null = bandwidth
    Eigen::Vector2d null_x(bw_x, 0.0);
    Eigen::Vector2d null_y(0.0, bw_y);

    double sens_at_null_x = k_geometry_->detector_sensitivity(det_idx, wavelength, null_x);
    double sens_at_null_y = k_geometry_->detector_sensitivity(det_idx, wavelength, null_y);

    // Should be very close to zero (within numerical tolerance)
    EXPECT_NEAR(sens_at_null_x, 0.0, 1e-10)
        << "Sensitivity at first x-null should be ~0, got " << sens_at_null_x;
    EXPECT_NEAR(sens_at_null_y, 0.0, 1e-10)
        << "Sensitivity at first y-null should be ~0, got " << sens_at_null_y;
}

// ============================================================================
// Test 9: POINT Mode - Sensitivity Always 1.0
// ============================================================================
//
// For POINT sensitivity mode, detector_sensitivity returns 1.0
// regardless of offset or position.
//
class PointSensitivityTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Set up detector with POINT sensitivity
        auto* array = detector_.add_arrays();
        array->set_array_id("point_test_detector");
        array->set_nominal_pitch_m(0.5e-3);
        array->add_wavelengths_m(1550e-9);
        array->set_sensitivity_type(DETECTOR_SENSITIVITY_POINT);  // POINT mode

        const int num_pixels = 8;
        const double pitch = 0.5e-3;
        for (int i = 0; i < num_pixels; ++i) {
            auto* pixel = array->add_pixels();
            pixel->set_index(i);
            pixel->set_x_m((i - 3.5) * pitch);
            pixel->set_y_m(0.0);
            pixel->set_width_m(0.4e-3);
            pixel->set_height_m(0.4e-3);
        }

        // Set up optics
        optics_.set_fourier_lens_focal_m(0.05);
        optics_.add_wavelengths_m(1550e-9);

        k_geometry_.emplace(optics_, detector_.arrays(0));
    }

    DetectorGeometry detector_;
    OpticsConfig optics_;
    std::optional<KSpaceGeometry> k_geometry_;
};

TEST_F(PointSensitivityTest, DetectorSensitivityAlwaysOne) {
    const double wavelength = 1550e-9;

    // Test various offsets - all should return 1.0
    std::vector<Eigen::Vector2d> offsets = {
        {0.0, 0.0},
        {10000.0, 0.0},
        {0.0, 10000.0},
        {50000.0, 50000.0},
        {-100000.0, -100000.0}
    };

    for (uint32_t det_idx = 0; det_idx < k_geometry_->num_detectors(); ++det_idx) {
        for (const auto& offset : offsets) {
            double sensitivity = k_geometry_->detector_sensitivity(det_idx, wavelength, offset);
            EXPECT_DOUBLE_EQ(sensitivity, 1.0)
                << "POINT mode: sensitivity should be 1.0 at any offset";
        }
    }
}

// ============================================================================
// Test 10: UNSPECIFIED Sensitivity Type Throws
// ============================================================================
//
// Verify that using UNSPECIFIED sensitivity type throws an exception.
// This ensures all consumers explicitly specify sensitivity behavior.
// The validation happens at KSpaceGeometry construction time (fail fast).
//
TEST(UnspecifiedSensitivityTest, ConstructorThrowsOnUnspecifiedSensitivity) {
    // Set up detector WITHOUT specifying sensitivity_type
    // (defaults to UNSPECIFIED = 0)
    DetectorArray array;
    array.set_array_id("unspecified_test_detector");
    array.set_nominal_pitch_m(0.5e-3);
    array.add_wavelengths_m(1550e-9);
    // Intentionally NOT setting sensitivity_type

    auto* pixel = array.add_pixels();
    pixel->set_index(0);
    pixel->set_x_m(0.0);
    pixel->set_y_m(0.0);
    pixel->set_width_m(0.4e-3);
    pixel->set_height_m(0.4e-3);

    // Set up optics
    OpticsConfig optics;
    optics.set_fourier_lens_focal_m(0.05);
    optics.add_wavelengths_m(1550e-9);

    // Constructor should throw because sensitivity_type is UNSPECIFIED
    EXPECT_THROW(
        KSpaceGeometry(optics, array),
        std::invalid_argument
    ) << "UNSPECIFIED sensitivity_type should throw std::invalid_argument at construction";
}

// ============================================================================
// Test 11: Sensitivity Monotonic Decrease
// ============================================================================
//
// Verify that SINC sensitivity decreases monotonically as we move away
// from the center (within the first lobe).
//
TEST_F(PhysicsValidationTest, SensitivityMonotonicDecrease) {
    ASSERT_TRUE(k_geometry_.has_value());

    const double wavelength = 1550e-9;
    const uint32_t det_idx = 0;

    // Get bandwidth to define test range (stay within first lobe)
    double bw_x = k_geometry_->detector_k_bandwidth_x(det_idx, wavelength);

    // Test monotonic decrease along x direction
    double prev_sensitivity = 1.0;
    for (double frac = 0.1; frac < 1.0; frac += 0.1) {
        Eigen::Vector2d offset(frac * bw_x, 0.0);
        double sensitivity = k_geometry_->detector_sensitivity(det_idx, wavelength, offset);

        EXPECT_LT(sensitivity, prev_sensitivity)
            << "Sensitivity should decrease as offset increases: at " << frac << " bandwidth, "
            << "got " << sensitivity << " which is not less than " << prev_sensitivity;

        prev_sensitivity = sensitivity;
    }
}

}  // namespace
}  // namespace aurora::forward
