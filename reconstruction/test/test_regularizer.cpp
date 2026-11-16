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

#include "reconstruction/regularizer.h"

#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aurora::reconstruction {

class RegularizerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Default grid parameters
        size_ = Eigen::Vector2i(16, 16);
        spacing_ = Eigen::Vector2d(1.0, 1.0);  // Unit spacing for formula tests
    }

    // Helper to generate random complex matrix
    Eigen::MatrixXcd random_matrix(int rows, int cols, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXcd result(rows, cols);
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                result(j, i) = std::complex<double>(dist(rng), dist(rng));
            }
        }
        return result;
    }

    // Compute inner product <a, b> = sum(a * conj(b))
    std::complex<double> inner_product(const Eigen::MatrixXcd& a,
                                       const Eigen::MatrixXcd& b) {
        std::complex<double> sum{0.0, 0.0};
        for (int j = 0; j < a.rows(); ++j) {
            for (int i = 0; i < a.cols(); ++i) {
                sum += a(j, i) * std::conj(b(j, i));
            }
        }
        return sum;
    }

    Eigen::Vector2i size_;
    Eigen::Vector2d spacing_;
};

// =============================================================================
// Tikhonov Regularizer Tests
// =============================================================================

TEST_F(RegularizerTest, TikhonovRegularizer_ApplyLtL_ReturnsInput) {
    TikhonovRegularizer reg;
    auto x = random_matrix(8, 8, 42);

    auto result = reg.apply_LtL(x);

    // Output should equal input (identity)
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            EXPECT_DOUBLE_EQ(result(j, i).real(), x(j, i).real());
            EXPECT_DOUBLE_EQ(result(j, i).imag(), x(j, i).imag());
        }
    }
}

TEST_F(RegularizerTest, TikhonovRegularizer_Spectrum_AllOnes) {
    TikhonovRegularizer reg;

    auto spectrum = reg.frequency_response(size_, spacing_);

    for (int j = 0; j < size_.y(); ++j) {
        for (int i = 0; i < size_.x(); ++i) {
            EXPECT_DOUBLE_EQ(spectrum(j, i), 1.0);
        }
    }
}

// =============================================================================
// Gradient Regularizer Tests
// =============================================================================

TEST_F(RegularizerTest, GradientRegularizer_ApplyLtL_ZeroForConstant) {
    GradientRegularizer reg;

    // Constant matrix - gradient is zero everywhere
    Eigen::MatrixXcd constant = Eigen::MatrixXcd::Constant(16, 16, {3.5, -1.2});

    auto result = reg.apply_LtL(constant);

    // With periodic boundaries and constant input, output should be zero
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            EXPECT_NEAR(result(j, i).real(), 0.0, 1e-10);
            EXPECT_NEAR(result(j, i).imag(), 0.0, 1e-10);
        }
    }
}

TEST_F(RegularizerTest, GradientRegularizer_ApplyLtL_Symmetric) {
    GradientRegularizer reg;
    auto x = random_matrix(16, 16, 100);
    auto y = random_matrix(16, 16, 101);

    // Compute <L†Lx, y>
    auto LtLx = reg.apply_LtL(x);
    auto inner1 = inner_product(LtLx, y);

    // Compute <x, L†Ly>
    auto LtLy = reg.apply_LtL(y);
    auto inner2 = inner_product(x, LtLy);

    // Self-adjoint: <L†Lx, y> = <x, L†Ly>
    EXPECT_NEAR(inner1.real(), inner2.real(), 1e-10);
    EXPECT_NEAR(inner1.imag(), inner2.imag(), 1e-10);
}

TEST_F(RegularizerTest, GradientRegularizer_Spectrum_ZeroAtDC) {
    GradientRegularizer reg;

    auto spectrum = reg.frequency_response(size_, spacing_);

    // DC component (0, 0) should be zero
    EXPECT_NEAR(spectrum(0, 0), 0.0, 1e-10);
}

TEST_F(RegularizerTest, GradientRegularizer_Spectrum_MatchesFormula) {
    GradientRegularizer reg;
    Eigen::Vector2i size(64, 64);
    Eigen::Vector2d spacing(1.0, 1.0);

    auto spectrum = reg.frequency_response(size, spacing);

    // Test against formula: Φ = 4 - 2cos(kx) - 2cos(ky) (5-point stencil eigenvalue)
    for (int j = 0; j < size.y(); ++j) {
        for (int i = 0; i < size.x(); ++i) {
            double kx = 2.0 * M_PI * i / size.x();
            double ky = 2.0 * M_PI * j / size.y();

            // Wrap to [-π, π]
            if (kx > M_PI) kx -= 2.0 * M_PI;
            if (ky > M_PI) ky -= 2.0 * M_PI;

            double expected = 4.0 - 2.0 * std::cos(kx) - 2.0 * std::cos(ky);
            EXPECT_NEAR(spectrum(j, i), expected, 1e-10)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(RegularizerTest, GradientRegularizer_Spectrum_MaxAtNyquist) {
    GradientRegularizer reg;
    Eigen::Vector2i size(16, 16);
    Eigen::Vector2d spacing(1.0, 1.0);

    auto spectrum = reg.frequency_response(size, spacing);

    // Find maximum value
    double max_val = spectrum.maxCoeff();

    // Maximum should be at Nyquist frequency (π, π)
    // For even-sized grid, this is at (N/2, N/2)
    int ni = size.x() / 2;
    int nj = size.y() / 2;

    // At (π, π): Φ = 4 - 2cos(π) - 2cos(π) = 4 - 2(-1) - 2(-1) = 8
    EXPECT_NEAR(spectrum(nj, ni), 8.0, 1e-10);
    EXPECT_NEAR(max_val, 8.0, 1e-10);
}

// =============================================================================
// Laplacian Regularizer Tests
// =============================================================================

TEST_F(RegularizerTest, LaplacianRegularizer_ApplyLtL_ZeroForConstant) {
    LaplacianRegularizer reg;

    // Constant matrix - Laplacian is zero
    Eigen::MatrixXcd constant = Eigen::MatrixXcd::Constant(16, 16, {2.0, 1.5});

    auto result = reg.apply_LtL(constant);

    // L†L of constant should be zero
    for (int j = 0; j < 16; ++j) {
        for (int i = 0; i < 16; ++i) {
            EXPECT_NEAR(result(j, i).real(), 0.0, 1e-10);
            EXPECT_NEAR(result(j, i).imag(), 0.0, 1e-10);
        }
    }
}

TEST_F(RegularizerTest, LaplacianRegularizer_ApplyLtL_Symmetric) {
    LaplacianRegularizer reg;
    auto x = random_matrix(16, 16, 300);
    auto y = random_matrix(16, 16, 301);

    // Compute <L†Lx, y>
    auto LtLx = reg.apply_LtL(x);
    auto inner1 = inner_product(LtLx, y);

    // Compute <x, L†Ly>
    auto LtLy = reg.apply_LtL(y);
    auto inner2 = inner_product(x, LtLy);

    // Self-adjoint: <L†Lx, y> = <x, L†Ly>
    EXPECT_NEAR(inner1.real(), inner2.real(), 1e-9);
    EXPECT_NEAR(inner1.imag(), inner2.imag(), 1e-9);
}

TEST_F(RegularizerTest, LaplacianRegularizer_Spectrum_ZeroAtDC) {
    LaplacianRegularizer reg;

    auto spectrum = reg.frequency_response(size_, spacing_);

    // DC component (0, 0) should be zero
    EXPECT_NEAR(spectrum(0, 0), 0.0, 1e-10);
}

TEST_F(RegularizerTest, LaplacianRegularizer_Spectrum_MatchesFormula) {
    LaplacianRegularizer reg;
    Eigen::Vector2i size(64, 64);
    Eigen::Vector2d spacing(1.0, 1.0);

    auto spectrum = reg.frequency_response(size, spacing);

    // Test against formula: Φ = (4 - 2cos(kx) - 2cos(ky))² (biharmonic eigenvalue)
    for (int j = 0; j < size.y(); ++j) {
        for (int i = 0; i < size.x(); ++i) {
            double kx = 2.0 * M_PI * i / size.x();
            double ky = 2.0 * M_PI * j / size.y();

            // Wrap to [-π, π]
            if (kx > M_PI) kx -= 2.0 * M_PI;
            if (ky > M_PI) ky -= 2.0 * M_PI;

            double lambda = 4.0 - 2.0 * std::cos(kx) - 2.0 * std::cos(ky);
            double expected = lambda * lambda;  // Biharmonic = Laplacian squared
            EXPECT_NEAR(spectrum(j, i), expected, 1e-10)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(RegularizerTest, LaplacianRegularizer_Spectrum_MaxAtNyquist) {
    LaplacianRegularizer reg;
    Eigen::Vector2i size(16, 16);
    Eigen::Vector2d spacing(1.0, 1.0);

    auto spectrum = reg.frequency_response(size, spacing);

    // Maximum should be at Nyquist (π, π)
    int ni = size.x() / 2;
    int nj = size.y() / 2;

    // At (π, π): Φ = (4 - 2cos(π) - 2cos(π))² = 8² = 64
    EXPECT_NEAR(spectrum(nj, ni), 64.0, 1e-10);
}

// =============================================================================
// Factory Tests
// =============================================================================

TEST_F(RegularizerTest, CreateRegularizer_Tikhonov) {
    auto variant = create_regularizer(REGULARIZER_TIKHONOV);
    const Regularizer& reg = get_regularizer(variant);

    // Verify it's a TikhonovRegularizer by behavior
    auto x = random_matrix(4, 4, 500);
    auto result = reg.apply_LtL(x);

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            EXPECT_DOUBLE_EQ(result(j, i).real(), x(j, i).real());
            EXPECT_DOUBLE_EQ(result(j, i).imag(), x(j, i).imag());
        }
    }
}

TEST_F(RegularizerTest, CreateRegularizer_Gradient) {
    auto variant = create_regularizer(REGULARIZER_GRADIENT);
    const Regularizer& reg = get_regularizer(variant);

    // Verify it's a GradientRegularizer by checking DC response
    auto spectrum = reg.frequency_response(Eigen::Vector2i(8, 8),
                                            Eigen::Vector2d(1.0, 1.0));
    EXPECT_NEAR(spectrum(0, 0), 0.0, 1e-10);
}

TEST_F(RegularizerTest, CreateRegularizer_Laplacian) {
    auto variant = create_regularizer(REGULARIZER_LAPLACIAN);
    const Regularizer& reg = get_regularizer(variant);

    // Verify it's a LaplacianRegularizer by checking Nyquist value
    auto spectrum = reg.frequency_response(Eigen::Vector2i(8, 8),
                                            Eigen::Vector2d(1.0, 1.0));
    // At Nyquist (4, 4): should be 64 (8²)
    EXPECT_NEAR(spectrum(4, 4), 64.0, 1e-10);
}

TEST_F(RegularizerTest, CreateRegularizer_Unknown_Throws) {
    EXPECT_THROW(create_regularizer(REGULARIZER_UNKNOWN), std::invalid_argument);
}

// =============================================================================
// Base Class Interface Tests
// =============================================================================

TEST_F(RegularizerTest, BaseClassInterface_WorksWithAllTypes) {
    std::vector<RegularizerVariant> variants = {
        create_regularizer(REGULARIZER_TIKHONOV),
        create_regularizer(REGULARIZER_GRADIENT),
        create_regularizer(REGULARIZER_LAPLACIAN)
    };

    auto x = random_matrix(8, 8, 600);

    for (auto& v : variants) {
        const Regularizer& reg = get_regularizer(v);

        // Should not throw, and return same size
        auto result = reg.apply_LtL(x);
        EXPECT_EQ(result.rows(), x.rows());
        EXPECT_EQ(result.cols(), x.cols());

        auto spectrum = reg.frequency_response(size_, spacing_);
        EXPECT_EQ(spectrum.rows(), size_.y());
        EXPECT_EQ(spectrum.cols(), size_.x());
    }
}

// =============================================================================
// Additional Validation Tests
// =============================================================================

TEST_F(RegularizerTest, AllRegularizers_NonNegativeSpectrum) {
    // All regularizer spectra should be non-negative (they're L†L eigenvalues)
    std::vector<RegularizerVariant> variants = {
        TikhonovRegularizer{},
        GradientRegularizer{},
        LaplacianRegularizer{}
    };

    Eigen::Vector2i size(32, 32);
    Eigen::Vector2d spacing(1.0, 1.0);

    for (const auto& v : variants) {
        const Regularizer& reg = get_regularizer(v);
        auto spectrum = reg.frequency_response(size, spacing);
        for (int j = 0; j < size.y(); ++j) {
            for (int i = 0; i < size.x(); ++i) {
                EXPECT_GE(spectrum(j, i), 0.0)
                    << "Negative spectrum at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST_F(RegularizerTest, GradientAndLaplacian_SpectrumRelationship) {
    // Laplacian spectrum should be square of gradient spectrum
    // (both are based on same 5-point stencil)
    GradientRegularizer grad_reg;
    LaplacianRegularizer lap_reg;

    Eigen::Vector2i size(16, 16);
    Eigen::Vector2d spacing(1.0, 1.0);

    auto grad_spectrum = grad_reg.frequency_response(size, spacing);
    auto lap_spectrum = lap_reg.frequency_response(size, spacing);

    for (int j = 0; j < size.y(); ++j) {
        for (int i = 0; i < size.x(); ++i) {
            double grad_val = grad_spectrum(j, i);
            double expected_lap = grad_val * grad_val;
            EXPECT_NEAR(lap_spectrum(j, i), expected_lap, 1e-10)
                << "Spectrum relationship mismatch at (" << i << ", " << j << ")";
        }
    }
}

}  // namespace aurora::reconstruction
