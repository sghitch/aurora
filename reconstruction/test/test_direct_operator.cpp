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

#include "reconstruction/direct_operator.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "forward/forward_model.h"
#include "test_helpers.h"

#include "bandwidth_config.pb.h"
#include "detector_geometry.pb.h"
#include "k_space_slice.pb.h"
#include "optics_config.pb.h"

namespace aurora::reconstruction {

namespace {

constexpr uint32_t kNumDetectors = 8;   // point-detector geometry size
constexpr double kPitch = 0.5e-3;  // m

// Helper function that provides a default BandwidthConfig for tests
aurora::forward::BandwidthConfig make_test_bandwidth_config() {
    aurora::forward::BandwidthConfig config;
    config.set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    config.set_quadrature_order(4);
    return config;
}

// Helper to create a KSpaceSlice from K-vectors and sample weights.
// detector_index is left at 0: with the point-detector geometry this maps
// every sample to a zero-bandwidth window, i.e. a point sample at (kx, ky).
aurora::shared::KSpaceSlice create_slice(
    const std::vector<Eigen::Vector2d>& k_vectors,
    const std::vector<double>& weights) {
    aurora::shared::KSpaceSlice slice;
    slice.set_wavelength_m(905e-9);
    for (size_t i = 0; i < k_vectors.size(); ++i) {
        auto* sample = slice.add_samples();
        sample->set_kx(k_vectors[i].x());
        sample->set_ky(k_vectors[i].y());
        sample->set_weight(weights[i]);
    }
    return slice;
}

}  // namespace

class DirectOperatorTest : public ::testing::Test {
  protected:
    static constexpr double kWavelength = 905e-9;

    void SetUp() override {
        // Create simple K-space samples for testing
        // Regular grid in K-space for easy verification
        k_vectors_.clear();
        weights_.clear();
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                k_vectors_.emplace_back(
                    i * 100.0,  // kx in rad/m
                    j * 100.0   // ky in rad/m
                );
                weights_.push_back(1.0);  // Unit weight
            }
        }

        // Default grid parameters
        grid_size_ = Eigen::Vector2i(8, 8);
        grid_spacing_ = Eigen::Vector2d(0.001, 0.001);  // 1mm pixels
        grid_origin_ = Eigen::Vector2d(-0.004, -0.004); // Centered

        // Point-detector geometry (zero pixel size): no K-space bandwidth, so
        // the operator treats the hand-built K-grid in each slice as point
        // samples rather than integrating over a K-window. This makes the
        // fixture a generic operator test on a dense K-grid.
        detector_geom_ = aurora::reconstruction::test::create_point_detector_geometry(
            "test_detector", kNumDetectors, kPitch, kWavelength);

        // Set up optics configuration
        optics_.set_fourier_lens_focal_m(0.1);  // 100mm focal length
        optics_.add_wavelengths_m(kWavelength);

        // Create ForwardModel
        forward_model_ = std::make_unique<aurora::forward::ForwardModel>(
            detector_geom_, optics_);
    }

    aurora::forward::DetectorGeometry detector_geom_;
    aurora::forward::OpticsConfig optics_;
    std::unique_ptr<aurora::forward::ForwardModel> forward_model_;

    std::vector<Eigen::Vector2d> k_vectors_;
    std::vector<double> weights_;
    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    Eigen::Vector2d grid_origin_;
};

// Test construction from KSpaceSlice
TEST_F(DirectOperatorTest, ConstructFromKSpaceSlice) {
    auto slice = create_slice(k_vectors_, weights_);

    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      grid_size_, grid_spacing_, grid_origin_);

    EXPECT_EQ(op.num_samples(), k_vectors_.size());

    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        EXPECT_DOUBLE_EQ(op.k_vectors()[i].x(), k_vectors_[i].x());
        EXPECT_DOUBLE_EQ(op.k_vectors()[i].y(), k_vectors_[i].y());
        EXPECT_DOUBLE_EQ(op.weights()[i], weights_[i]);
    }
}

// Test forward of single point source
TEST_F(DirectOperatorTest, ForwardSinglePoint) {
    // Single K-vector for easy phase check
    std::vector<Eigen::Vector2d> single_k = {{1000.0, 2000.0}};  // rad/m
    std::vector<double> single_eff = {1.0};

    auto slice = create_slice(single_k, single_eff);
    Eigen::Vector2i size(16, 16);
    Eigen::Vector2d spacing(0.001, 0.001);
    Eigen::Vector2d origin(-0.008, -0.008);

    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      size, spacing, origin);

    // Point source at origin (x=0, y=0)
    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Zero(16, 16);
    scene(8, 8) = std::complex<double>(1.0, 0.0);  // Center pixel

    auto k_data = op.forward(scene, origin, spacing);

    // Phase should be: -k . r = -(1000*0 + 2000*0) = 0
    EXPECT_NEAR(std::abs(k_data(0)), 1.0, 1e-10);
    EXPECT_NEAR(std::arg(k_data(0)), 0.0, 1e-10);
}

// Test forward of off-center point source
TEST_F(DirectOperatorTest, ForwardOffCenterPoint) {
    std::vector<Eigen::Vector2d> single_k = {{1000.0, 0.0}};  // kx only
    std::vector<double> single_eff = {1.0};

    auto slice = create_slice(single_k, single_eff);
    Eigen::Vector2i size(16, 16);
    Eigen::Vector2d spacing(0.001, 0.001);
    Eigen::Vector2d origin(-0.008, -0.008);

    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      size, spacing, origin);

    // Point source at x = 0.002 m, y = 0
    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Zero(16, 16);
    int ix = 10;  // x = -0.008 + 10*0.001 = 0.002 m
    int iy = 8;   // y = 0
    scene(iy, ix) = std::complex<double>(1.0, 0.0);

    auto k_data = op.forward(scene, origin, spacing);

    // Expected phase: -kx * x = -1000 * 0.002 = -2 rad
    double expected_phase = -1000.0 * 0.002;
    EXPECT_NEAR(std::arg(k_data(0)), expected_phase, 1e-6);
}

// Test forward linearity: A(x + y) = Ax + Ay
TEST_F(DirectOperatorTest, ForwardLinearity) {
    auto slice = create_slice(k_vectors_, weights_);
    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      grid_size_, grid_spacing_, grid_origin_);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::MatrixXcd x(8, 8), y(8, 8);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
            y(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    auto Ax = op.forward(x, grid_origin_, grid_spacing_);
    auto Ay = op.forward(y, grid_origin_, grid_spacing_);
    auto Axy = op.forward(x + y, grid_origin_, grid_spacing_);

    for (Eigen::Index i = 0; i < Ax.size(); ++i) {
        EXPECT_NEAR((Axy(i) - Ax(i) - Ay(i)).real(), 0.0, 1e-10);
        EXPECT_NEAR((Axy(i) - Ax(i) - Ay(i)).imag(), 0.0, 1e-10);
    }
}

// Test forward scaling: A(cx) = c * Ax
TEST_F(DirectOperatorTest, ForwardScaling) {
    auto slice = create_slice(k_vectors_, weights_);
    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      grid_size_, grid_spacing_, grid_origin_);

    std::mt19937 rng(43);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::MatrixXcd x(8, 8);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    std::complex<double> c(2.5, -1.3);

    auto Ax = op.forward(x, grid_origin_, grid_spacing_);
    auto Acx = op.forward(c * x, grid_origin_, grid_spacing_);

    for (Eigen::Index i = 0; i < Ax.size(); ++i) {
        EXPECT_NEAR((Acx(i) - c * Ax(i)).real(), 0.0, 1e-10);
        EXPECT_NEAR((Acx(i) - c * Ax(i)).imag(), 0.0, 1e-10);
    }
}

// Test adjoint of single K produces sinusoid
TEST_F(DirectOperatorTest, AdjointSingleK) {
    std::vector<Eigen::Vector2d> single_k = {{500.0, 0.0}};  // kx only
    std::vector<double> single_eff = {1.0};

    auto slice = create_slice(single_k, single_eff);
    Eigen::Vector2i size(16, 16);
    Eigen::Vector2d origin(-0.008, -0.008);
    Eigen::Vector2d spacing(0.001, 0.001);

    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      size, spacing, origin);

    // Single K-space sample
    Eigen::VectorXcd k_samples(1);
    k_samples(0) = std::complex<double>(1.0, 0.0);

    auto image = op.adjoint(k_samples, origin, spacing, size);

    // Should produce sinusoid in x direction with frequency k=500
    // Phase at each pixel should be exp(+i * 500 * x)
    // All pixels should have unit magnitude
    for (int ix = 0; ix < 16; ++ix) {
        double x = origin.x() + ix * spacing.x();
        double expected_phase = 500.0 * x;

        // Create expected phasor and compare
        auto expected = std::polar(1.0, expected_phase);
        auto actual = image(8, ix);

        // Magnitudes should be equal (unit weight)
        EXPECT_NEAR(std::abs(actual), std::abs(expected), 1e-10);

        // Phases should match - compare via phasor multiplication
        // If phases match, actual * conj(expected) should be real and positive
        auto phase_diff = actual * std::conj(expected);
        EXPECT_NEAR(phase_diff.imag(), 0.0, 1e-10);
        EXPECT_GT(phase_diff.real(), 0.0);
    }
}

// Test adjoint consistency: <Ax, y> = <x, A*y>
TEST_F(DirectOperatorTest, AdjointConsistency) {
    auto slice = create_slice(k_vectors_, weights_);
    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      grid_size_, grid_spacing_, grid_origin_);

    std::mt19937 rng(44);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Random scene
    Eigen::MatrixXcd x(8, 8);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    // Random K-space data
    Eigen::VectorXcd y(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        y(i) = std::complex<double>(dist(rng), dist(rng));
    }

    double error = op.check_adjoint_consistency(x, y, grid_origin_, grid_spacing_);
    EXPECT_LT(error, 1e-10);
}

// Test adjoint consistency with high precision
TEST_F(DirectOperatorTest, AdjointConsistencyHighPrecision) {
    auto slice = create_slice(k_vectors_, weights_);
    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      grid_size_, grid_spacing_, grid_origin_);

    std::srand(45);
    Eigen::MatrixXcd x(8, 8);
    Eigen::VectorXcd y(k_vectors_.size());

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            x(i, j) = std::complex<double>(
                std::rand() / static_cast<double>(RAND_MAX) - 0.5,
                std::rand() / static_cast<double>(RAND_MAX) - 0.5);
        }
    }

    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        y(i) = std::complex<double>(
            std::rand() / static_cast<double>(RAND_MAX) - 0.5,
            std::rand() / static_cast<double>(RAND_MAX) - 0.5);
    }

    // Compute <Ax, y>
    auto Ax = op.forward(x, grid_origin_, grid_spacing_);
    std::complex<double> inner1(0, 0);
    for (Eigen::Index i = 0; i < Ax.size(); ++i) {
        inner1 += Ax(i) * std::conj(y(i));
    }

    // Compute <x, A*y>
    auto Aty = op.adjoint(y, grid_origin_, grid_spacing_, grid_size_);
    std::complex<double> inner2(0, 0);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            inner2 += x(i, j) * std::conj(Aty(i, j));
        }
    }

    // Check to 1e-10 precision as specified in ticket
    EXPECT_NEAR(inner1.real(), inner2.real(), 1e-10);
    EXPECT_NEAR(inner1.imag(), inner2.imag(), 1e-10);
}

// Test apply_normal matches separate calls
TEST_F(DirectOperatorTest, ApplyNormalConsistency) {
    auto slice = create_slice(k_vectors_, weights_);
    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      grid_size_, grid_spacing_, grid_origin_);

    std::mt19937 rng(46);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::MatrixXcd x(8, 8);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    // A*Ax via apply_normal
    auto AtAx = op.apply_normal(x, grid_origin_, grid_spacing_);

    // A*Ax via forward then adjoint
    auto Ax = op.forward(x, grid_origin_, grid_spacing_);
    auto AtAx2 = op.adjoint(Ax, grid_origin_, grid_spacing_, grid_size_);

    // Should be the same
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            EXPECT_NEAR((AtAx(i, j) - AtAx2(i, j)).real(), 0.0, 1e-10);
            EXPECT_NEAR((AtAx(i, j) - AtAx2(i, j)).imag(), 0.0, 1e-10);
        }
    }
}

// Test sample weighting
TEST_F(DirectOperatorTest, SampleWeighting) {
    std::vector<Eigen::Vector2d> k = {{100.0, 0.0}, {200.0, 0.0}};
    std::vector<double> eff_unity = {1.0, 1.0};
    std::vector<double> eff_half = {0.5, 0.5};

    auto slice_unity = create_slice(k, eff_unity);
    auto slice_half = create_slice(k, eff_half);

    Eigen::Vector2i size(4, 4);
    Eigen::Vector2d origin(-0.002, -0.002);
    Eigen::Vector2d spacing(0.001, 0.001);

    DirectOperator op_unity(slice_unity, *forward_model_,
                            make_test_bandwidth_config(),
                            size, spacing, origin);
    DirectOperator op_half(slice_half, *forward_model_,
                           make_test_bandwidth_config(),
                           size, spacing, origin);

    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(4, 4);

    auto result_unity = op_unity.forward(scene, origin, spacing);
    auto result_half = op_half.forward(scene, origin, spacing);

    // Half weight should give half amplitude
    for (Eigen::Index i = 0; i < result_unity.size(); ++i) {
        EXPECT_NEAR(std::abs(result_half(i)), 0.5 * std::abs(result_unity(i)), 1e-10);
    }
}

// Test phase encoding: point at (x,y) -> phase = -k.r
TEST_F(DirectOperatorTest, PhaseEncoding) {
    // Create K-space samples at various positions
    std::vector<Eigen::Vector2d> k = {
        {1000.0, 0.0},
        {0.0, 1000.0},
        {500.0, 500.0}
    };
    std::vector<double> eff = {1.0, 1.0, 1.0};

    auto slice = create_slice(k, eff);
    Eigen::Vector2i size(16, 16);
    Eigen::Vector2d origin(-0.008, -0.008);
    Eigen::Vector2d spacing(0.001, 0.001);

    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      size, spacing, origin);

    // Point at specific location
    double target_x = 0.003;
    double target_y = 0.002;

    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Zero(16, 16);

    // Find nearest grid point
    int ix = static_cast<int>((target_x - origin.x()) / spacing.x());
    int iy = static_cast<int>((target_y - origin.y()) / spacing.y());
    double actual_x = origin.x() + ix * spacing.x();
    double actual_y = origin.y() + iy * spacing.y();
    scene(iy, ix) = std::complex<double>(1.0, 0.0);

    auto k_data = op.forward(scene, origin, spacing);

    // Verify phase encoding for each K
    for (size_t i = 0; i < k.size(); ++i) {
        double expected_phase = -(k[i].x() * actual_x + k[i].y() * actual_y);
        double actual_phase = std::arg(k_data(i));
        // Wrap to [-pi, pi]
        double diff = std::fmod(actual_phase - expected_phase + M_PI, 2 * M_PI) - M_PI;
        EXPECT_NEAR(diff, 0.0, 1e-10);
    }
}

// Test adjoint peak position localization
TEST_F(DirectOperatorTest, AdjointPeakPosition) {
    // Dense K-space for good localization
    std::vector<Eigen::Vector2d> k;
    std::vector<double> eff;
    for (int i = -10; i <= 10; ++i) {
        for (int j = -10; j <= 10; ++j) {
            k.emplace_back(i * 200.0, j * 200.0);
            eff.push_back(1.0);
        }
    }

    auto slice = create_slice(k, eff);
    Eigen::Vector2i size(32, 32);
    Eigen::Vector2d origin(-0.008, -0.008);
    Eigen::Vector2d spacing(0.0005, 0.0005);

    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      size, spacing, origin);

    // Simulate point target at specific position
    double target_x = 0.002;
    double target_y = 0.001;

    // Generate K-space data for point target
    Eigen::VectorXcd k_data(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        double phase = -(k[i].x() * target_x + k[i].y() * target_y);
        k_data(i) = std::polar(1.0, phase);
    }

    auto image = op.adjoint(k_data, origin, spacing, size);

    // Find peak position
    int peak_ix = 0, peak_iy = 0;
    double max_mag = 0.0;
    for (int iy = 0; iy < size.y(); ++iy) {
        for (int ix = 0; ix < size.x(); ++ix) {
            double mag = std::abs(image(iy, ix));
            if (mag > max_mag) {
                max_mag = mag;
                peak_ix = ix;
                peak_iy = iy;
            }
        }
    }

    double peak_x = origin.x() + peak_ix * spacing.x();
    double peak_y = origin.y() + peak_iy * spacing.y();

    // Peak should be within 1 pixel of target
    EXPECT_NEAR(peak_x, target_x, spacing.x());
    EXPECT_NEAR(peak_y, target_y, spacing.y());
}

// Test wavelength independence (K-vectors scale correctly)
TEST_F(DirectOperatorTest, WavelengthIndependence) {
    // Same geometry, different wavelengths should give scaled K-vectors
    // This test verifies the operator works correctly with any K-vectors
    double f = 0.1;  // 100mm focal length
    double det_x = 0.002;  // 2mm detector offset

    double lambda1 = 905e-9;
    double lambda2 = 1550e-9;

    // K = 2*pi * x_det / (f * lambda)
    double k1 = 2 * M_PI * det_x / (f * lambda1);
    double k2 = 2 * M_PI * det_x / (f * lambda2);

    // K should scale as 1/lambda
    EXPECT_NEAR(k1 / k2, lambda2 / lambda1, 1e-10);

    // Create operators with scaled K-vectors
    std::vector<Eigen::Vector2d> kv1 = {{k1, 0.0}};
    std::vector<Eigen::Vector2d> kv2 = {{k2, 0.0}};
    std::vector<double> eff = {1.0};

    auto slice1 = create_slice(kv1, eff);
    auto slice2 = create_slice(kv2, eff);

    Eigen::Vector2i size(8, 8);
    Eigen::Vector2d origin(-0.004, -0.004);
    Eigen::Vector2d spacing(0.001, 0.001);

    DirectOperator op1(slice1, *forward_model_,
                       make_test_bandwidth_config(),
                       size, spacing, origin);
    DirectOperator op2(slice2, *forward_model_,
                       make_test_bandwidth_config(),
                       size, spacing, origin);

    // Point source at known position
    // Use small displacement to avoid phase wrapping
    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Zero(8, 8);
    scene(4, 5) = 1.0;  // x = 0.001

    auto y1 = op1.forward(scene, origin, spacing);
    auto y2 = op2.forward(scene, origin, spacing);

    // Expected phases: phase = -k * x
    double x = origin.x() + 5 * spacing.x();  // 0.001
    double expected_phase1 = -k1 * x;
    double expected_phase2 = -k2 * x;

    // Verify each phase is correct (via phasor comparison)
    auto expected1 = std::polar(1.0, expected_phase1);
    auto expected2 = std::polar(1.0, expected_phase2);

    auto diff1 = y1(0) * std::conj(expected1);
    auto diff2 = y2(0) * std::conj(expected2);

    EXPECT_NEAR(diff1.imag(), 0.0, 1e-10);
    EXPECT_GT(diff1.real(), 0.0);
    EXPECT_NEAR(diff2.imag(), 0.0, 1e-10);
    EXPECT_GT(diff2.real(), 0.0);
}

// Test empty operator
TEST_F(DirectOperatorTest, EmptyOperator) {
    std::vector<Eigen::Vector2d> empty_k;
    std::vector<double> empty_eff;

    auto slice = create_slice(empty_k, empty_eff);
    Eigen::Vector2i size(4, 4);
    Eigen::Vector2d origin(-0.002, -0.002);
    Eigen::Vector2d spacing(0.001, 0.001);

    DirectOperator op(slice, *forward_model_,
                      make_test_bandwidth_config(),
                      size, spacing, origin);

    EXPECT_EQ(op.num_samples(), 0u);

    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(4, 4);
    auto k_data = op.forward(scene, grid_origin_, grid_spacing_);
    EXPECT_EQ(k_data.size(), 0);

    Eigen::VectorXcd empty_samples(0);
    auto image = op.adjoint(empty_samples, grid_origin_, grid_spacing_, size);
    EXPECT_EQ(image.rows(), 4);
    EXPECT_EQ(image.cols(), 4);
    // Should be all zeros
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(image(i, j).real(), 0.0);
            EXPECT_DOUBLE_EQ(image(i, j).imag(), 0.0);
        }
    }
}

}  // namespace aurora::reconstruction
