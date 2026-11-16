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

#include "reconstruction/nufft_operator.h"
#include "reconstruction/direct_operator.h"

#include <chrono>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "forward/forward_model.h"
#include "test_helpers.h"

#include "bandwidth_config.pb.h"
#include "detector_geometry.pb.h"
#include "k_space_slice.pb.h"
#include "optics_config.pb.h"
#include "reconstruction_config.pb.h"

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

// Helper to create a valid NufftConfig protobuf with standard test parameters
NufftConfig make_test_config(double oversampling = 2.0, int kernel_width = 4,
                              NufftKernel kernel = KERNEL_KAISER_BESSEL) {
    NufftConfig config;
    config.set_oversampling_factor(oversampling);
    config.set_kernel_width(kernel_width);
    config.set_kernel(kernel);
    return config;
}

class NufftOperatorTest : public ::testing::Test {
  protected:
    static constexpr double kWavelength = 905e-9;

    void SetUp() override {
        // Create simple K-space samples for testing
        // Regular grid in K-space for easy verification
        k_vectors_.clear();
        weights_.clear();
        for (int i = -4; i <= 4; ++i) {
            for (int j = -4; j <= 4; ++j) {
                k_vectors_.emplace_back(
                    i * 100.0,  // kx in rad/m
                    j * 100.0   // ky in rad/m
                );
                weights_.push_back(1.0);  // Unit weight
            }
        }

        // Default grid parameters
        grid_size_ = Eigen::Vector2i(16, 16);
        grid_spacing_ = Eigen::Vector2d(0.001, 0.001);  // 1mm pixels
        grid_origin_ = Eigen::Vector2d(-0.008, -0.008); // Centered

        // Default config (protobuf)
        config_ = make_test_config();

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
    NufftConfig config_;
};

// Test construction from KSpaceSlice
TEST_F(NufftOperatorTest, ConstructFromKSpaceSlice) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    EXPECT_EQ(op.num_samples(), k_vectors_.size());

    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        EXPECT_DOUBLE_EQ(op.k_vectors()[i].x(), k_vectors_[i].x());
        EXPECT_DOUBLE_EQ(op.k_vectors()[i].y(), k_vectors_[i].y());
        EXPECT_DOUBLE_EQ(op.weights()[i], weights_[i]);
    }
}

// Test invalid grid size throws
TEST_F(NufftOperatorTest, InvalidGridSizeThrows) {
    auto slice = create_slice(k_vectors_, weights_);
    Eigen::Vector2i bad_size(0, 16);
    EXPECT_THROW(
        NufftOperator(slice, *forward_model_, bad_size, grid_spacing_, config_),
        std::invalid_argument);
}

// Test invalid oversampling throws
TEST_F(NufftOperatorTest, InvalidOversamplingThrows) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftConfig bad_config = make_test_config(0.5, 4);  // OS=0.5 is invalid
    EXPECT_THROW(
        NufftOperator(slice, *forward_model_, grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

// Test missing oversampling_factor throws
TEST_F(NufftOperatorTest, MissingOversamplingThrows) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftConfig bad_config;
    bad_config.set_kernel(KERNEL_KAISER_BESSEL);
    bad_config.set_kernel_width(4);
    // Not setting oversampling_factor
    EXPECT_THROW(
        NufftOperator(slice, *forward_model_, grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

// Test missing kernel_width throws
TEST_F(NufftOperatorTest, MissingKernelWidthThrows) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftConfig bad_config;
    bad_config.set_kernel(KERNEL_KAISER_BESSEL);
    bad_config.set_oversampling_factor(2.0);
    // Not setting kernel_width
    EXPECT_THROW(
        NufftOperator(slice, *forward_model_, grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

// Test unspecified kernel throws
TEST_F(NufftOperatorTest, UnspecifiedKernelThrows) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftConfig bad_config;
    bad_config.set_oversampling_factor(2.0);
    bad_config.set_kernel_width(4);
    // Not setting kernel (defaults to KERNEL_UNSPECIFIED = 0)
    EXPECT_THROW(
        NufftOperator(slice, *forward_model_, grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

// Test invalid kernel_width throws
TEST_F(NufftOperatorTest, InvalidKernelWidthThrows) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftConfig bad_config = make_test_config(2.0, 0);  // kernel_width=0 is invalid
    EXPECT_THROW(
        NufftOperator(slice, *forward_model_, grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

// Test oversampled size computation
TEST_F(NufftOperatorTest, OversampledSizeComputation) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    auto os_size = op.oversampled_size();
    EXPECT_EQ(os_size.x(), 32);
    EXPECT_EQ(os_size.y(), 32);
}

// Test adjoint consistency: <Ax, y> = <x, A*y>
TEST_F(NufftOperatorTest, AdjointConsistency) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Random scene
    Eigen::MatrixXcd x(grid_size_.y(), grid_size_.x());
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    // Random K-space data
    Eigen::VectorXcd y(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        y(i) = std::complex<double>(dist(rng), dist(rng));
    }

    double error = op.check_adjoint_consistency(x, y, grid_origin_, grid_spacing_);

    // NUFFT has some approximation error, so tolerance is relaxed
    EXPECT_LT(error, 1e-5);
}

// Test adjoint matches DirectOperator
TEST_F(NufftOperatorTest, AdjointMatchesDirect) {
    // NUFFT is an approximation. Use higher accuracy parameters for this test.
    NufftConfig high_accuracy_config = make_test_config(4.0, 8, KERNEL_KAISER_BESSEL);

    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator nufft_op(slice, *forward_model_, grid_size_, grid_spacing_, high_accuracy_config);
    DirectOperator direct_op(slice, *forward_model_,
                             make_test_bandwidth_config(),
                             grid_size_, grid_spacing_, grid_origin_);

    std::mt19937 rng(43);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Random K-space samples
    Eigen::VectorXcd k_samples(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        k_samples(i) = std::complex<double>(dist(rng), dist(rng));
    }

    auto nufft_result = nufft_op.adjoint(k_samples, grid_origin_, grid_spacing_, grid_size_);
    auto direct_result = direct_op.adjoint(k_samples, grid_origin_, grid_spacing_, grid_size_);

    // Compute relative error
    double max_diff = 0.0;
    double max_mag = 0.0;
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            double diff = std::abs(nufft_result(i, j) - direct_result(i, j));
            double mag = std::abs(direct_result(i, j));
            max_diff = std::max(max_diff, diff);
            max_mag = std::max(max_mag, mag);
        }
    }

    double relative_error = max_diff / (max_mag + 1e-15);
    // With OS=4, KW=8: achieves ~1.6e-5 relative error
    EXPECT_LT(relative_error, 1e-4);
}

// Test forward matches DirectOperator
TEST_F(NufftOperatorTest, ForwardMatchesDirect) {
    // NUFFT is an approximation. Use higher accuracy parameters for this test.
    NufftConfig high_accuracy_config = make_test_config(4.0, 8, KERNEL_KAISER_BESSEL);

    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator nufft_op(slice, *forward_model_, grid_size_, grid_spacing_, high_accuracy_config);
    DirectOperator direct_op(slice, *forward_model_,
                             make_test_bandwidth_config(),
                             grid_size_, grid_spacing_, grid_origin_);

    std::mt19937 rng(44);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Random scene
    Eigen::MatrixXcd scene(grid_size_.y(), grid_size_.x());
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            scene(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    auto nufft_result = nufft_op.forward(scene, grid_origin_, grid_spacing_);
    auto direct_result = direct_op.forward(scene, grid_origin_, grid_spacing_);

    // Compute relative error
    double max_diff = 0.0;
    double max_mag = 0.0;
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        double diff = std::abs(nufft_result(i) - direct_result(i));
        double mag = std::abs(direct_result(i));
        max_diff = std::max(max_diff, diff);
        max_mag = std::max(max_mag, mag);
    }

    double relative_error = max_diff / (max_mag + 1e-15);
    // With OS=4, KW=8: achieves ~4.4e-6 relative error
    EXPECT_LT(relative_error, 1e-4);
}

// Test point target localization via adjoint
TEST_F(NufftOperatorTest, PointTargetLocalization) {
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
    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    // Simulate point target at specific position
    double target_x = 0.002;
    double target_y = 0.001;

    // Generate K-space data for point target
    Eigen::VectorXcd k_data(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        double phase = -(k[i].x() * target_x + k[i].y() * target_y);
        k_data(i) = std::polar(1.0, phase);
    }

    Eigen::Vector2i size(32, 32);
    Eigen::Vector2d origin(-0.008, -0.008);
    Eigen::Vector2d spacing(0.0005, 0.0005);

    // Need to recreate operator for new grid size
    NufftOperator op2(slice, *forward_model_, size, spacing, config_);
    auto image = op2.adjoint(k_data, origin, spacing, size);

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

    // Peak should be within 2 pixels of target (NUFFT has some spreading)
    EXPECT_NEAR(peak_x, target_x, 2 * spacing.x());
    EXPECT_NEAR(peak_y, target_y, 2 * spacing.y());
}

// Test sample weighting
TEST_F(NufftOperatorTest, SampleWeighting) {
    std::vector<Eigen::Vector2d> k = {{100.0, 0.0}, {200.0, 0.0}};
    std::vector<double> eff_unity = {1.0, 1.0};
    std::vector<double> eff_half = {0.5, 0.5};

    auto slice_unity = create_slice(k, eff_unity);
    auto slice_half = create_slice(k, eff_half);

    Eigen::Vector2i size(8, 8);
    Eigen::Vector2d spacing(0.001, 0.001);
    Eigen::Vector2d origin(-0.004, -0.004);

    NufftOperator op_unity(slice_unity, *forward_model_, size, spacing, config_);
    NufftOperator op_half(slice_half, *forward_model_, size, spacing, config_);

    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(8, 8);

    auto result_unity = op_unity.forward(scene, origin, spacing);
    auto result_half = op_half.forward(scene, origin, spacing);

    // Half weight should give half amplitude
    for (Eigen::Index i = 0; i < result_unity.size(); ++i) {
        EXPECT_NEAR(std::abs(result_half(i)), 0.5 * std::abs(result_unity(i)), 1e-6);
    }
}

// Test empty operator
TEST_F(NufftOperatorTest, EmptyOperator) {
    std::vector<Eigen::Vector2d> empty_k;
    std::vector<double> empty_eff;

    auto slice = create_slice(empty_k, empty_eff);
    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    EXPECT_EQ(op.num_samples(), 0u);

    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(grid_size_.y(), grid_size_.x());
    auto k_data = op.forward(scene, grid_origin_, grid_spacing_);
    EXPECT_EQ(k_data.size(), 0);

    Eigen::VectorXcd empty_samples(0);
    auto image = op.adjoint(empty_samples, grid_origin_, grid_spacing_, grid_size_);
    EXPECT_EQ(image.rows(), grid_size_.y());
    EXPECT_EQ(image.cols(), grid_size_.x());

    // Should be all zeros
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            EXPECT_DOUBLE_EQ(image(i, j).real(), 0.0);
            EXPECT_DOUBLE_EQ(image(i, j).imag(), 0.0);
        }
    }
}

// Test forward linearity: A(x + y) = Ax + Ay
TEST_F(NufftOperatorTest, ForwardLinearity) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    std::mt19937 rng(45);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::MatrixXcd x(grid_size_.y(), grid_size_.x());
    Eigen::MatrixXcd y(grid_size_.y(), grid_size_.x());
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
            y(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    auto Ax = op.forward(x, grid_origin_, grid_spacing_);
    auto Ay = op.forward(y, grid_origin_, grid_spacing_);
    auto Axy = op.forward(x + y, grid_origin_, grid_spacing_);

    for (Eigen::Index i = 0; i < Ax.size(); ++i) {
        EXPECT_NEAR((Axy(i) - Ax(i) - Ay(i)).real(), 0.0, 1e-8);
        EXPECT_NEAR((Axy(i) - Ax(i) - Ay(i)).imag(), 0.0, 1e-8);
    }
}

// Test Gaussian kernel option
TEST_F(NufftOperatorTest, GaussianKernel) {
    // Note: Gaussian kernel has lower accuracy than Kaiser-Bessel because
    // Gaussian doesn't have compact support - it extends infinitely but we
    // truncate at the kernel width. This causes incomplete deapodization.
    // Kaiser-Bessel is the recommended kernel for accurate results.
    //
    // This test verifies Gaussian kernel produces reasonable results and
    // that adjoint consistency holds (which is more important than absolute accuracy).
    NufftConfig gaussian_config = make_test_config(2.0, 4, KERNEL_GAUSSIAN);

    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator nufft_op(slice, *forward_model_, grid_size_, grid_spacing_, gaussian_config);

    std::mt19937 rng(46);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Random scene and K-space data for adjoint consistency test
    Eigen::MatrixXcd x(grid_size_.y(), grid_size_.x());
    Eigen::VectorXcd y(k_vectors_.size());
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        y(i) = std::complex<double>(dist(rng), dist(rng));
    }

    // Test adjoint consistency: <Ax, y> ≈ <x, A*y>
    double error = nufft_op.check_adjoint_consistency(x, y, grid_origin_, grid_spacing_);

    // Gaussian should still maintain adjoint consistency even if absolute accuracy is lower
    EXPECT_LT(error, 1e-5);

    // Also verify forward/adjoint produce non-trivial results
    auto fwd = nufft_op.forward(x, grid_origin_, grid_spacing_);
    auto adj = nufft_op.adjoint(y, grid_origin_, grid_spacing_, grid_size_);

    EXPECT_GT(fwd.norm(), 0.0);
    EXPECT_GT(adj.norm(), 0.0);
}

// Test different oversampling factors
TEST_F(NufftOperatorTest, HigherOversamplingImproves) {
    auto slice = create_slice(k_vectors_, weights_);
    DirectOperator direct_op(slice, *forward_model_,
                             make_test_bandwidth_config(),
                             grid_size_, grid_spacing_, grid_origin_);

    std::mt19937 rng(47);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::VectorXcd k_samples(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        k_samples(i) = std::complex<double>(dist(rng), dist(rng));
    }

    auto direct_result = direct_op.adjoint(k_samples, grid_origin_, grid_spacing_, grid_size_);

    // Test with oversampling = 1.5
    NufftConfig config15 = make_test_config(1.5, 4);
    NufftOperator op15(slice, *forward_model_, grid_size_, grid_spacing_, config15);
    auto result15 = op15.adjoint(k_samples, grid_origin_, grid_spacing_, grid_size_);

    // Test with oversampling = 2.0
    NufftConfig config20 = make_test_config(2.0, 4);
    NufftOperator op20(slice, *forward_model_, grid_size_, grid_spacing_, config20);
    auto result20 = op20.adjoint(k_samples, grid_origin_, grid_spacing_, grid_size_);

    // Compute errors
    double error15 = 0.0, error20 = 0.0;
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            error15 += std::abs(result15(i, j) - direct_result(i, j));
            error20 += std::abs(result20(i, j) - direct_result(i, j));
        }
    }

    // Higher oversampling should give lower or equal error
    EXPECT_LE(error20, error15 * 1.1);  // Allow small tolerance
}

// Test apply_normal matches separate forward/adjoint
TEST_F(NufftOperatorTest, ApplyNormalConsistency) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    std::mt19937 rng(48);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::MatrixXcd x(grid_size_.y(), grid_size_.x());
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    // A*Ax via apply_normal
    auto AtAx = op.apply_normal(x, grid_origin_, grid_spacing_);

    // A*Ax via forward then adjoint
    auto Ax = op.forward(x, grid_origin_, grid_spacing_);
    auto AtAx2 = op.adjoint(Ax, grid_origin_, grid_spacing_, grid_size_);

    // Should be the same
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            EXPECT_NEAR((AtAx(i, j) - AtAx2(i, j)).real(), 0.0, 1e-8);
            EXPECT_NEAR((AtAx(i, j) - AtAx2(i, j)).imag(), 0.0, 1e-8);
        }
    }
}

// Performance benchmark test (not strict, just informational)
TEST_F(NufftOperatorTest, PerformanceBenchmark) {
    // Create larger problem for timing
    std::vector<Eigen::Vector2d> k;
    std::vector<double> eff;
    for (int i = -20; i <= 20; ++i) {
        for (int j = -20; j <= 20; ++j) {
            k.emplace_back(i * 50.0, j * 50.0);
            eff.push_back(1.0);
        }
    }
    // ~1681 K-samples

    Eigen::Vector2i size(64, 64);
    Eigen::Vector2d spacing(0.001, 0.001);
    Eigen::Vector2d origin(-0.032, -0.032);

    auto slice = create_slice(k, eff);
    NufftOperator nufft_op(slice, *forward_model_, size, spacing, config_);
    DirectOperator direct_op(slice, *forward_model_,
                             make_test_bandwidth_config(),
                             size, spacing, origin);

    std::mt19937 rng(49);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::VectorXcd k_samples(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        k_samples(i) = std::complex<double>(dist(rng), dist(rng));
    }

    // Time NUFFT
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        auto result = nufft_op.adjoint(k_samples, origin, spacing, size);
        (void)result;  // Prevent optimization
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double nufft_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / 10.0;

    // Time Direct
    t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        auto result = direct_op.adjoint(k_samples, origin, spacing, size);
        (void)result;
    }
    t2 = std::chrono::high_resolution_clock::now();
    double direct_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / 10.0;

    double speedup = direct_ms / nufft_ms;

    // Print results (visible in test output)
    std::cout << "Performance Benchmark (64x64, ~1681 K-samples):\n";
    std::cout << "  NUFFT adjoint: " << nufft_ms << " ms\n";
    std::cout << "  Direct adjoint: " << direct_ms << " ms\n";
    std::cout << "  Speedup: " << speedup << "x\n";

    // NUFFT should be faster (at least 2x for this problem size)
    // Note: On debug builds, speedup may be lower
    EXPECT_GT(speedup, 1.0);
}

// Test move semantics
TEST_F(NufftOperatorTest, MoveSemantics) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator op1(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    // Move construct
    NufftOperator op2(std::move(op1));
    EXPECT_EQ(op2.num_samples(), k_vectors_.size());

    // Move assign
    NufftOperator op3(slice, *forward_model_, grid_size_, grid_spacing_, config_);
    op3 = std::move(op2);
    EXPECT_EQ(op3.num_samples(), k_vectors_.size());

    // Verify it still works
    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(grid_size_.y(), grid_size_.x());
    auto result = op3.forward(scene, grid_origin_, grid_spacing_);
    EXPECT_EQ(result.size(), static_cast<Eigen::Index>(k_vectors_.size()));
}

// Test config accessor
TEST_F(NufftOperatorTest, ConfigAccessor) {
    auto config = make_test_config(1.75, 6, KERNEL_KAISER_BESSEL);
    auto slice = create_slice(k_vectors_, weights_);

    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config);

    EXPECT_EQ(op.config().kernel_width(), 6u);
    EXPECT_DOUBLE_EQ(op.config().oversampling_factor(), 1.75);
    EXPECT_EQ(op.config().kernel(), KERNEL_KAISER_BESSEL);
}

// Test grid_size accessor
TEST_F(NufftOperatorTest, GridSizeAccessor) {
    auto slice = create_slice(k_vectors_, weights_);
    NufftOperator op(slice, *forward_model_, grid_size_, grid_spacing_, config_);

    EXPECT_EQ(op.grid_size().x(), grid_size_.x());
    EXPECT_EQ(op.grid_size().y(), grid_size_.y());
}

}  // namespace aurora::reconstruction
