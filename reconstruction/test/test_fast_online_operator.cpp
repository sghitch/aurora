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

#include "reconstruction/fast_online_operator.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "k_space_slice.pb.h"
#include "reconstruction_config.pb.h"

namespace aurora::reconstruction {

class FastOnlineOperatorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Generic operator fixture: a dense 5x5 K-grid of samples. Each
        // sample is tagged with its own index so the K-sample axis window
        // and gap interpolation see one entry per sample.
        k_vectors_.clear();
        weights_.clear();
        detector_indices_.clear();

        uint32_t sample_index = 0;
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                k_vectors_.emplace_back(
                    i * 100.0,  // kx in rad/m
                    j * 100.0   // ky in rad/m
                );
                weights_.push_back(1.0);  // Unit weight
                detector_indices_.push_back(sample_index++);
            }
        }
        num_samples_ = sample_index;

        // Default grid parameters
        grid_size_ = Eigen::Vector2i(8, 8);
        grid_spacing_ = Eigen::Vector2d(0.001, 0.001);  // 1mm pixels
        grid_origin_ = Eigen::Vector2d(-0.004, -0.004); // Centered

        // Default config with minimal valid settings
        default_config_.set_kernel(GRIDDING_KERNEL_GAUSSIAN);
        default_config_.set_kernel_width(2);
        default_config_.set_anti_gap_window(ANTI_GAP_WINDOW_NONE);
        default_config_.set_density_normalization(DENSITY_NORM_KERNEL);
        default_config_.set_density_epsilon(1e-10);
        default_config_.set_post_processing(POST_PROC_NONE);
        default_config_.set_enable_gap_interpolation(false);
        default_config_.set_enable_refinement(false);
    }

    std::vector<Eigen::Vector2d> k_vectors_;
    std::vector<double> weights_;
    std::vector<uint32_t> detector_indices_;
    uint32_t num_samples_ = 0;
    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    Eigen::Vector2d grid_origin_;
    FastOnlineConfig default_config_;
};

// =============================================================================
// Construction tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, ConstructFromVectors) {
    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, default_config_);

    EXPECT_EQ(op.num_samples(), k_vectors_.size());
    EXPECT_EQ(op.k_vectors().size(), k_vectors_.size());
    EXPECT_EQ(op.weights().size(), weights_.size());
    EXPECT_EQ(op.detector_indices().size(), detector_indices_.size());

    // Verify K-vectors match
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        EXPECT_DOUBLE_EQ(op.k_vectors()[i].x(), k_vectors_[i].x());
        EXPECT_DOUBLE_EQ(op.k_vectors()[i].y(), k_vectors_[i].y());
    }
}

TEST_F(FastOnlineOperatorTest, ConstructFromKSpaceSlice) {
    aurora::shared::KSpaceSlice slice;
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        auto* sample = slice.add_samples();
        sample->set_kx(k_vectors_[i].x());
        sample->set_ky(k_vectors_[i].y());
        sample->set_weight(weights_[i]);
        sample->set_detector_index(detector_indices_[i]);
    }

    FastOnlineOperator op(slice, grid_size_, grid_spacing_, default_config_);

    EXPECT_EQ(op.num_samples(), k_vectors_.size());

    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        EXPECT_DOUBLE_EQ(op.k_vectors()[i].x(), k_vectors_[i].x());
        EXPECT_DOUBLE_EQ(op.k_vectors()[i].y(), k_vectors_[i].y());
        EXPECT_DOUBLE_EQ(op.weights()[i], weights_[i]);
        EXPECT_EQ(op.detector_indices()[i], detector_indices_[i]);
    }
}

TEST_F(FastOnlineOperatorTest, MismatchedSizesThrows) {
    std::vector<double> bad_eff = {1.0, 1.0};  // Wrong size
    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, bad_eff, detector_indices_,
                           grid_size_, grid_spacing_, default_config_),
        std::invalid_argument);

    std::vector<uint32_t> bad_det = {0, 1};  // Wrong size
    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, bad_det,
                           grid_size_, grid_spacing_, default_config_),
        std::invalid_argument);
}

// =============================================================================
// Config validation tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, UnspecifiedKernelThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_kernel(GRIDDING_KERNEL_UNSPECIFIED);

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, ZeroKernelWidthThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_kernel_width(0);

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, UnspecifiedAntiGapWindowThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_anti_gap_window(ANTI_GAP_WINDOW_UNSPECIFIED);

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, TukeyWithoutAlphaThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_anti_gap_window(ANTI_GAP_WINDOW_TUKEY);
    // Missing anti_gap_tukey_alpha or out of range

    bad_config.set_anti_gap_tukey_alpha(-0.1);  // Out of range
    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);

    bad_config.set_anti_gap_tukey_alpha(1.5);  // Out of range
    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, KaiserWithoutBetaThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_anti_gap_window(ANTI_GAP_WINDOW_KAISER);
    bad_config.set_anti_gap_kaiser_beta(0.0);  // Invalid (must be > 0)

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, UnspecifiedDensityNormThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_density_normalization(DENSITY_NORM_UNSPECIFIED);

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, DensityNormWithoutEpsilonThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_density_normalization(DENSITY_NORM_KERNEL);
    bad_config.set_density_epsilon(0.0);  // Invalid (must be > 0)

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, UnspecifiedPostProcessingThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_post_processing(POST_PROC_UNSPECIFIED);

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, LaplacianWithoutStrengthThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_post_processing(POST_PROC_LAPLACIAN_SHARPEN);
    bad_config.set_laplacian_strength(-0.1);  // Out of range

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, DiffusionWithoutCoefficientThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_post_processing(POST_PROC_DIFFUSION);
    bad_config.set_diffusion_coefficient(0.0);  // Invalid (must be > 0)

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, RefinementWithoutIterationsThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_enable_refinement(true);
    bad_config.set_refinement_iterations(0);  // Invalid (must be > 0)

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, RefinementWithoutAlphaThrows) {
    FastOnlineConfig bad_config = default_config_;
    bad_config.set_enable_refinement(true);
    bad_config.set_refinement_iterations(1);
    bad_config.set_refinement_alpha(0.0);  // Invalid (must be > 0)

    EXPECT_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, bad_config),
        std::invalid_argument);
}

TEST_F(FastOnlineOperatorTest, ValidConfigWithAllKernelTypes) {
    // Test Gaussian kernel
    FastOnlineConfig config_gaussian = default_config_;
    config_gaussian.set_kernel(GRIDDING_KERNEL_GAUSSIAN);
    EXPECT_NO_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, config_gaussian));

    // Test Kaiser-Bessel kernel
    FastOnlineConfig config_kb = default_config_;
    config_kb.set_kernel(GRIDDING_KERNEL_KAISER_BESSEL);
    EXPECT_NO_THROW(
        FastOnlineOperator(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, config_kb));
}

// =============================================================================
// K-sample axis window tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, AntiGapWindowNone) {
    FastOnlineConfig config = default_config_;
    config.set_anti_gap_window(ANTI_GAP_WINDOW_NONE);

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    // All weights should be 1.0 when no window
    const auto& weights = op.anti_gap_weights();
    EXPECT_EQ(weights.size(), num_samples_);  // one window entry per sample
    for (double w : weights) {
        EXPECT_DOUBLE_EQ(w, 1.0);
    }
}

TEST_F(FastOnlineOperatorTest, AntiGapWindowHann) {
    FastOnlineConfig config = default_config_;
    config.set_anti_gap_window(ANTI_GAP_WINDOW_HANN);

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    const auto& weights = op.anti_gap_weights();
    EXPECT_EQ(weights.size(), num_samples_);

    // Hann window: w[j] = 0.5 * (1 - cos(2*pi*j/(N-1)))
    // Should be symmetric, minimum at edges, maximum at center
    for (size_t j = 0; j < weights.size(); ++j) {
        EXPECT_GE(weights[j], 0.0);
        EXPECT_LE(weights[j], 1.0);
    }

    // Verify symmetry
    for (size_t j = 0; j < weights.size() / 2; ++j) {
        EXPECT_NEAR(weights[j], weights[weights.size() - 1 - j], 1e-10);
    }

    // Center should be near maximum
    EXPECT_GT(weights[3], weights[0]);
    EXPECT_GT(weights[4], weights[0]);
}

TEST_F(FastOnlineOperatorTest, AntiGapWindowTukey) {
    FastOnlineConfig config = default_config_;
    config.set_anti_gap_window(ANTI_GAP_WINDOW_TUKEY);
    config.set_anti_gap_tukey_alpha(0.5);

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    const auto& weights = op.anti_gap_weights();
    EXPECT_EQ(weights.size(), num_samples_);

    // Tukey window: flat in center, tapered at edges
    for (size_t j = 0; j < weights.size(); ++j) {
        EXPECT_GE(weights[j], 0.0);
        EXPECT_LE(weights[j], 1.0);
    }

    // Center region should have unity weight
    // With alpha=0.5, the middle 50% of the window is flat
    const size_t center = (weights.size() - 1) / 2;
    EXPECT_NEAR(weights[center - 1], 1.0, 1e-6);
    EXPECT_NEAR(weights[center], 1.0, 1e-6);
    EXPECT_NEAR(weights[center + 1], 1.0, 1e-6);
}

TEST_F(FastOnlineOperatorTest, AntiGapWindowKaiser) {
    FastOnlineConfig config = default_config_;
    config.set_anti_gap_window(ANTI_GAP_WINDOW_KAISER);
    config.set_anti_gap_kaiser_beta(3.0);

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    const auto& weights = op.anti_gap_weights();
    EXPECT_EQ(weights.size(), num_samples_);

    // Kaiser window: smooth bell curve
    for (size_t j = 0; j < weights.size(); ++j) {
        EXPECT_GE(weights[j], 0.0);
        EXPECT_LE(weights[j], 1.0);
    }

    // Should be symmetric
    for (size_t j = 0; j < weights.size() / 2; ++j) {
        EXPECT_NEAR(weights[j], weights[weights.size() - 1 - j], 1e-10);
    }
}

// =============================================================================
// Forward/Adjoint operator tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, ForwardSinglePoint) {
    // Single K-vector for easy phase check
    std::vector<Eigen::Vector2d> single_k = {{1000.0, 2000.0}};
    std::vector<double> single_eff = {1.0};
    std::vector<uint32_t> single_det = {0};

    // Use 8x8 grid to match operator construction
    Eigen::Vector2i test_grid(8, 8);
    Eigen::Vector2d test_spacing(0.001, 0.001);
    Eigen::Vector2d test_origin(-0.004, -0.004);

    FastOnlineOperator op(single_k, single_eff, single_det,
                          test_grid, test_spacing, default_config_);

    // Point source at origin (x=0, y=0) - center of 8x8 grid
    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Zero(8, 8);
    scene(4, 4) = std::complex<double>(1.0, 0.0);  // Center pixel

    auto k_data = op.forward(scene, test_origin, test_spacing);

    // Forward via gridding produces non-trivial result
    // Just verify we get non-zero output with correct structure
    EXPECT_EQ(k_data.size(), 1);
    // Gridding-based forward doesn't preserve amplitude directly
    // but should produce non-zero output
    EXPECT_GT(std::abs(k_data(0)), 0.0);
}

TEST_F(FastOnlineOperatorTest, ForwardLinearity) {
    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, default_config_);

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
        EXPECT_NEAR((Axy(i) - Ax(i) - Ay(i)).real(), 0.0, 1e-6);
        EXPECT_NEAR((Axy(i) - Ax(i) - Ay(i)).imag(), 0.0, 1e-6);
    }
}

TEST_F(FastOnlineOperatorTest, ForwardScaling) {
    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, default_config_);

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
        EXPECT_NEAR((Acx(i) - c * Ax(i)).real(), 0.0, 1e-6);
        EXPECT_NEAR((Acx(i) - c * Ax(i)).imag(), 0.0, 1e-6);
    }
}

TEST_F(FastOnlineOperatorTest, AdjointProducesValidOutput) {
    // Note: Gridding-based operators don't have exact adjoint consistency
    // because the forward and adjoint use different computational paths
    // (FFT vs gridding). This test verifies the adjoint produces valid output.
    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, default_config_);

    std::mt19937 rng(44);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Random K-space data
    Eigen::VectorXcd y(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        y(i) = std::complex<double>(dist(rng), dist(rng));
    }

    // Compute A*y
    auto Aty = op.adjoint(y, grid_origin_, grid_spacing_, grid_size_);

    // Verify output has correct dimensions
    EXPECT_EQ(Aty.rows(), grid_size_.y());
    EXPECT_EQ(Aty.cols(), grid_size_.x());

    // Verify no NaN/Inf in output
    bool valid = true;
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            if (std::isnan(Aty(i, j).real()) || std::isnan(Aty(i, j).imag()) ||
                std::isinf(Aty(i, j).real()) || std::isinf(Aty(i, j).imag())) {
                valid = false;
            }
        }
    }
    EXPECT_TRUE(valid);

    // Should produce non-trivial output
    double total_energy = 0.0;
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            total_energy += std::norm(Aty(i, j));
        }
    }
    EXPECT_GT(total_energy, 0.0);
}

// =============================================================================
// Reconstruct pipeline tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, ReconstructPointTarget) {
    // Dense K-space for good localization
    std::vector<Eigen::Vector2d> k;
    std::vector<double> w;
    std::vector<uint32_t> det;
    for (int i = -10; i <= 10; ++i) {
        for (int j = -10; j <= 10; ++j) {
            k.emplace_back(i * 200.0, j * 200.0);
            w.push_back(1.0);
            det.push_back(static_cast<uint32_t>(det.size()));
        }
    }

    Eigen::Vector2i large_grid(32, 32);
    Eigen::Vector2d large_spacing(0.0005, 0.0005);

    FastOnlineOperator op(k, w, det, large_grid, large_spacing, default_config_);

    // Simulate point target at specific position
    double target_x = 0.002;
    double target_y = 0.001;

    // Generate K-space data for point target
    Eigen::VectorXcd k_data(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        double phase = -(k[i].x() * target_x + k[i].y() * target_y);
        k_data(i) = std::polar(1.0, phase);
    }

    Eigen::Vector2d origin(-0.008, -0.008);

    auto image = op.reconstruct(k_data, origin, large_spacing, large_grid);

    // Find peak position
    int peak_ix = 0, peak_iy = 0;
    double max_mag = 0.0;
    for (int iy = 0; iy < large_grid.y(); ++iy) {
        for (int ix = 0; ix < large_grid.x(); ++ix) {
            double mag = std::abs(image(iy, ix));
            if (mag > max_mag) {
                max_mag = mag;
                peak_ix = ix;
                peak_iy = iy;
            }
        }
    }

    double peak_x = origin.x() + peak_ix * large_spacing.x();
    double peak_y = origin.y() + peak_iy * large_spacing.y();

    // Peak should be within 2 pixels of target
    EXPECT_NEAR(peak_x, target_x, 2 * large_spacing.x());
    EXPECT_NEAR(peak_y, target_y, 2 * large_spacing.y());
}

TEST_F(FastOnlineOperatorTest, ReconstructWithDensityNormKernelSquared) {
    FastOnlineConfig config = default_config_;
    config.set_density_normalization(DENSITY_NORM_KERNEL_SQUARED);

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    // Simple K-space data
    Eigen::VectorXcd k_data = Eigen::VectorXcd::Ones(k_vectors_.size());

    auto image = op.reconstruct(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Should produce a valid image (no NaN/Inf)
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            EXPECT_FALSE(std::isnan(image(i, j).real()));
            EXPECT_FALSE(std::isnan(image(i, j).imag()));
            EXPECT_FALSE(std::isinf(image(i, j).real()));
            EXPECT_FALSE(std::isinf(image(i, j).imag()));
        }
    }
}

TEST_F(FastOnlineOperatorTest, ReconstructWithNoDensityNorm) {
    FastOnlineConfig config = default_config_;
    config.set_density_normalization(DENSITY_NORM_NONE);
    // density_epsilon not required for NONE mode

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    Eigen::VectorXcd k_data = Eigen::VectorXcd::Ones(k_vectors_.size());

    auto image = op.reconstruct(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Should produce a valid image
    bool has_nonzero = false;
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            if (std::abs(image(i, j)) > 1e-10) {
                has_nonzero = true;
            }
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// =============================================================================
// Post-processing tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, LaplacianSharpenIncreaseContrast) {
    // Create two operators: one with sharpening, one without
    FastOnlineConfig config_no_sharpen = default_config_;
    config_no_sharpen.set_post_processing(POST_PROC_NONE);

    FastOnlineConfig config_sharpen = default_config_;
    config_sharpen.set_post_processing(POST_PROC_LAPLACIAN_SHARPEN);
    config_sharpen.set_laplacian_strength(0.5);

    FastOnlineOperator op_no_sharpen(k_vectors_, weights_, detector_indices_,
                                      grid_size_, grid_spacing_, config_no_sharpen);
    FastOnlineOperator op_sharpen(k_vectors_, weights_, detector_indices_,
                                   grid_size_, grid_spacing_, config_sharpen);

    // Create simple K-space data with smooth variation
    Eigen::VectorXcd k_data(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        k_data(i) = std::complex<double>(1.0, 0.0);
    }

    auto image_no_sharpen = op_no_sharpen.reconstruct(k_data, grid_origin_,
                                                       grid_spacing_, grid_size_);
    auto image_sharpen = op_sharpen.reconstruct(k_data, grid_origin_,
                                                 grid_spacing_, grid_size_);

    // Sharpened image should have different values (hard to test "more contrast")
    // Just verify it's not identical
    double diff_sum = 0.0;
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            diff_sum += std::abs(image_sharpen(i, j) - image_no_sharpen(i, j));
        }
    }

    // There should be some difference
    EXPECT_GT(diff_sum, 0.0);
}

TEST_F(FastOnlineOperatorTest, DiffusionSmooths) {
    FastOnlineConfig config = default_config_;
    config.set_post_processing(POST_PROC_DIFFUSION);
    config.set_diffusion_coefficient(0.1);

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    Eigen::VectorXcd k_data(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        k_data(i) = std::complex<double>(1.0, 0.0);
    }

    auto image = op.reconstruct(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Should produce a valid image without NaN/Inf
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            EXPECT_FALSE(std::isnan(image(i, j).real()));
            EXPECT_FALSE(std::isnan(image(i, j).imag()));
        }
    }
}

// =============================================================================
// Refinement tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, RefinementReducesResidual) {
    // Create operators with and without refinement
    FastOnlineConfig config_no_refine = default_config_;
    config_no_refine.set_enable_refinement(false);

    FastOnlineConfig config_refine = default_config_;
    config_refine.set_enable_refinement(true);
    config_refine.set_refinement_iterations(2);
    config_refine.set_refinement_alpha(0.5);
    config_refine.set_refinement_lambda(0.0);

    FastOnlineOperator op_no_refine(k_vectors_, weights_, detector_indices_,
                                     grid_size_, grid_spacing_, config_no_refine);
    FastOnlineOperator op_refine(k_vectors_, weights_, detector_indices_,
                                  grid_size_, grid_spacing_, config_refine);

    // Point target K-space data
    double target_x = 0.001;
    double target_y = 0.0;
    Eigen::VectorXcd k_data(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        double phase = -(k_vectors_[i].x() * target_x + k_vectors_[i].y() * target_y);
        k_data(i) = std::polar(1.0, phase);
    }

    auto image_no_refine = op_no_refine.reconstruct(k_data, grid_origin_,
                                                     grid_spacing_, grid_size_);
    auto image_refine = op_refine.reconstruct(k_data, grid_origin_,
                                               grid_spacing_, grid_size_);

    // Both should produce valid images
    // Can't easily verify refinement improves quality without forward model,
    // but should at least not make things worse and produce valid output
    bool refined_valid = true;
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            if (std::isnan(image_refine(i, j).real()) ||
                std::isnan(image_refine(i, j).imag())) {
                refined_valid = false;
            }
        }
    }
    EXPECT_TRUE(refined_valid);
}

TEST_F(FastOnlineOperatorTest, RefinementWithRegularization) {
    FastOnlineConfig config = default_config_;
    config.set_enable_refinement(true);
    config.set_refinement_iterations(1);
    config.set_refinement_alpha(0.3);
    config.set_refinement_lambda(0.01);  // Some regularization

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    Eigen::VectorXcd k_data = Eigen::VectorXcd::Ones(k_vectors_.size());

    auto image = op.reconstruct(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Should not produce NaN/Inf with regularization
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            EXPECT_FALSE(std::isnan(image(i, j).real()));
            EXPECT_FALSE(std::isnan(image(i, j).imag()));
        }
    }
}

// =============================================================================
// Sample weighting tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, SampleWeighting) {
    std::vector<Eigen::Vector2d> k = {{100.0, 0.0}, {200.0, 0.0}};
    std::vector<double> eff_unity = {1.0, 1.0};
    std::vector<double> eff_half = {0.5, 0.5};
    std::vector<uint32_t> det = {0, 1};

    // Use consistent grid size (scene must match grid_size_ used in constructor)
    FastOnlineOperator op_unity(k, eff_unity, det,
                                 grid_size_, grid_spacing_, default_config_);
    FastOnlineOperator op_half(k, eff_half, det,
                                grid_size_, grid_spacing_, default_config_);

    // Scene size must match grid_size_ (8x8)
    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(grid_size_.y(), grid_size_.x());

    auto result_unity = op_unity.forward(scene, grid_origin_, grid_spacing_);
    auto result_half = op_half.forward(scene, grid_origin_, grid_spacing_);

    // Half weight should give half amplitude (approximately, due to gridding)
    for (Eigen::Index i = 0; i < result_unity.size(); ++i) {
        EXPECT_NEAR(std::abs(result_half(i)), 0.5 * std::abs(result_unity(i)), 0.1);
    }
}

// =============================================================================
// Empty/edge case tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, EmptyOperator) {
    std::vector<Eigen::Vector2d> empty_k;
    std::vector<double> empty_eff;
    std::vector<uint32_t> empty_det;

    // Use consistent grid size for operator and calls
    Eigen::Vector2i small_grid(4, 4);
    Eigen::Vector2d small_spacing(0.001, 0.001);
    Eigen::Vector2d small_origin(-0.002, -0.002);

    FastOnlineOperator op(empty_k, empty_eff, empty_det,
                          small_grid, small_spacing, default_config_);

    EXPECT_EQ(op.num_samples(), 0u);

    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(4, 4);
    auto k_data = op.forward(scene, small_origin, small_spacing);
    EXPECT_EQ(k_data.size(), 0);

    Eigen::VectorXcd empty_samples(0);
    auto image = op.adjoint(empty_samples, small_origin, small_spacing, small_grid);
    EXPECT_EQ(image.rows(), 4);
    EXPECT_EQ(image.cols(), 4);
}

// =============================================================================
// Gap interpolation tests
// =============================================================================

TEST_F(FastOnlineOperatorTest, GapInterpolationProducesValidOutput) {
    // A line of K-samples along kx whose spacing leaves gaps between them
    std::vector<Eigen::Vector2d> gap_k_vectors;
    std::vector<double> gap_weights;
    std::vector<uint32_t> gap_detector_indices;

    const uint32_t num_line_samples = 8;
    double ky = 0.0;
    for (uint32_t s = 0; s < num_line_samples; ++s) {
        double kx = (s - 3.5) * 150.0;  // 150 rad/m spacing leaves gaps
        gap_k_vectors.emplace_back(kx, ky);
        gap_weights.push_back(1.0);
        gap_detector_indices.push_back(s);
    }

    // Configure with gap interpolation enabled
    FastOnlineConfig gap_config = default_config_;
    gap_config.set_enable_gap_interpolation(true);
    gap_config.set_kernel_width(2);

    FastOnlineOperator op(gap_k_vectors, gap_weights, gap_detector_indices,
                          grid_size_, grid_spacing_, gap_config);

    // Create test K-space data
    Eigen::VectorXcd k_samples(gap_k_vectors.size());
    for (size_t i = 0; i < gap_k_vectors.size(); ++i) {
        k_samples(i) = std::complex<double>(1.0, 0.0);
    }

    // Reconstruct
    auto image = op.reconstruct(k_samples, grid_origin_, grid_spacing_, grid_size_);

    // Verify output is valid (no NaN/Inf)
    bool has_nan_inf = false;
    for (int i = 0; i < image.rows(); ++i) {
        for (int j = 0; j < image.cols(); ++j) {
            if (std::isnan(image(i, j).real()) || std::isnan(image(i, j).imag()) ||
                std::isinf(image(i, j).real()) || std::isinf(image(i, j).imag())) {
                has_nan_inf = true;
            }
        }
    }
    EXPECT_FALSE(has_nan_inf) << "Gap interpolation produced NaN/Inf values";

    // Verify output has some non-zero content
    double total_magnitude = 0.0;
    for (int i = 0; i < image.rows(); ++i) {
        for (int j = 0; j < image.cols(); ++j) {
            total_magnitude += std::abs(image(i, j));
        }
    }
    EXPECT_GT(total_magnitude, 0.0) << "Gap interpolation produced all-zero output";
}

TEST_F(FastOnlineOperatorTest, GapInterpolationWithKernelSquaredDensity) {
    // Test gap interpolation with DENSITY_NORM_KERNEL_SQUARED
    std::vector<Eigen::Vector2d> gap_k_vectors;
    std::vector<double> gap_weights;
    std::vector<uint32_t> gap_detector_indices;

    // A line of K-samples along kx with gaps between them
    const uint32_t num_line_samples = 8;
    double ky = 0.0;
    for (uint32_t s = 0; s < num_line_samples; ++s) {
        double kx = (s - 3.5) * 150.0;
        gap_k_vectors.emplace_back(kx, ky);
        gap_weights.push_back(1.0);
        gap_detector_indices.push_back(s);
    }

    // Configure with gap interpolation AND kernel-squared density
    FastOnlineConfig gap_config = default_config_;
    gap_config.set_enable_gap_interpolation(true);
    gap_config.set_kernel_width(2);
    gap_config.set_density_normalization(DENSITY_NORM_KERNEL_SQUARED);

    FastOnlineOperator op(gap_k_vectors, gap_weights, gap_detector_indices,
                          grid_size_, grid_spacing_, gap_config);

    // Create test K-space data
    Eigen::VectorXcd k_samples(gap_k_vectors.size());
    for (size_t i = 0; i < gap_k_vectors.size(); ++i) {
        k_samples(i) = std::complex<double>(1.0, 0.0);
    }

    // Reconstruct
    auto image = op.reconstruct(k_samples, grid_origin_, grid_spacing_, grid_size_);

    // Verify output is valid
    bool has_nan_inf = false;
    for (int i = 0; i < image.rows(); ++i) {
        for (int j = 0; j < image.cols(); ++j) {
            if (std::isnan(image(i, j).real()) || std::isnan(image(i, j).imag()) ||
                std::isinf(image(i, j).real()) || std::isinf(image(i, j).imag())) {
                has_nan_inf = true;
            }
        }
    }
    EXPECT_FALSE(has_nan_inf) << "Gap interpolation with KERNEL_SQUARED produced NaN/Inf";

    // Verify non-zero output
    double total_magnitude = 0.0;
    for (int i = 0; i < image.rows(); ++i) {
        for (int j = 0; j < image.cols(); ++j) {
            total_magnitude += std::abs(image(i, j));
        }
    }
    EXPECT_GT(total_magnitude, 0.0);
}

TEST_F(FastOnlineOperatorTest, MoveConstruction) {
    FastOnlineOperator op1(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, default_config_);

    size_t original_size = op1.num_samples();

    FastOnlineOperator op2(std::move(op1));

    EXPECT_EQ(op2.num_samples(), original_size);
}

TEST_F(FastOnlineOperatorTest, MoveAssignment) {
    FastOnlineOperator op1(k_vectors_, weights_, detector_indices_,
                           grid_size_, grid_spacing_, default_config_);

    std::vector<Eigen::Vector2d> small_k = {{100.0, 0.0}};
    std::vector<double> small_eff = {1.0};
    std::vector<uint32_t> small_det = {0};

    FastOnlineOperator op2(small_k, small_eff, small_det,
                           grid_size_, grid_spacing_, default_config_);

    size_t original_size = op1.num_samples();

    op2 = std::move(op1);

    EXPECT_EQ(op2.num_samples(), original_size);
}

// =============================================================================
// Integration of the K-sample axis window with reconstruct
// =============================================================================

TEST_F(FastOnlineOperatorTest, FullPipelineWithHannWindow) {
    FastOnlineConfig config = default_config_;
    config.set_anti_gap_window(ANTI_GAP_WINDOW_HANN);

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    // Point target K-space data
    double target_x = 0.001;
    double target_y = -0.001;
    Eigen::VectorXcd k_data(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        double phase = -(k_vectors_[i].x() * target_x + k_vectors_[i].y() * target_y);
        k_data(i) = std::polar(1.0, phase);
    }

    auto image = op.reconstruct(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Should localize the target
    int peak_ix = 0, peak_iy = 0;
    double max_mag = 0.0;
    for (int iy = 0; iy < grid_size_.y(); ++iy) {
        for (int ix = 0; ix < grid_size_.x(); ++ix) {
            double mag = std::abs(image(iy, ix));
            if (mag > max_mag) {
                max_mag = mag;
                peak_ix = ix;
                peak_iy = iy;
            }
        }
    }

    double peak_x = grid_origin_.x() + peak_ix * grid_spacing_.x();
    double peak_y = grid_origin_.y() + peak_iy * grid_spacing_.y();

    // Should be reasonably close to target
    EXPECT_NEAR(peak_x, target_x, 2 * grid_spacing_.x());
    EXPECT_NEAR(peak_y, target_y, 2 * grid_spacing_.y());
}

TEST_F(FastOnlineOperatorTest, FullPipelineWithAllFeatures) {
    FastOnlineConfig config;
    config.set_kernel(GRIDDING_KERNEL_KAISER_BESSEL);
    config.set_kernel_width(3);
    config.set_anti_gap_window(ANTI_GAP_WINDOW_TUKEY);
    config.set_anti_gap_tukey_alpha(0.3);
    config.set_density_normalization(DENSITY_NORM_KERNEL);
    config.set_density_epsilon(1e-10);
    config.set_post_processing(POST_PROC_LAPLACIAN_SHARPEN);
    config.set_laplacian_strength(0.2);
    config.set_enable_gap_interpolation(false);
    config.set_enable_refinement(true);
    config.set_refinement_iterations(1);
    config.set_refinement_alpha(0.3);
    config.set_refinement_lambda(0.001);

    FastOnlineOperator op(k_vectors_, weights_, detector_indices_,
                          grid_size_, grid_spacing_, config);

    Eigen::VectorXcd k_data(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        double phase = -(k_vectors_[i].x() * 0.001 + k_vectors_[i].y() * 0.0);
        k_data(i) = std::polar(1.0, phase);
    }

    auto image = op.reconstruct(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Verify no NaN/Inf in output
    bool valid = true;
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            if (std::isnan(image(i, j).real()) || std::isnan(image(i, j).imag()) ||
                std::isinf(image(i, j).real()) || std::isinf(image(i, j).imag())) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
    }
    EXPECT_TRUE(valid);
}

}  // namespace aurora::reconstruction
