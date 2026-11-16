#include "reconstruction/preconditioner.h"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aurora::reconstruction {

class PreconditionerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Default grid parameters
        rows_ = 16;
        cols_ = 16;
        n_pixels_ = rows_ * cols_;
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

    // Generate uniform diagonal for testing
    std::vector<double> uniform_diagonal(double value) {
        return std::vector<double>(n_pixels_, value);
    }

    // Generate gradient-style regularizer spectrum
    // Φ(kx, ky) = 4 - 2cos(kx) - 2cos(ky)
    std::vector<double> gradient_spectrum() {
        std::vector<double> spectrum(n_pixels_);
        for (int j = 0; j < rows_; ++j) {
            for (int i = 0; i < cols_; ++i) {
                double kx = 2.0 * M_PI * i / cols_;
                double ky = 2.0 * M_PI * j / rows_;
                if (kx > M_PI) kx -= 2.0 * M_PI;
                if (ky > M_PI) ky -= 2.0 * M_PI;
                spectrum[j * cols_ + i] = 4.0 - 2.0 * std::cos(kx) - 2.0 * std::cos(ky);
            }
        }
        return spectrum;
    }

    // Generate simple transfer function (identity-ish PSF)
    // For testing, use something close to identity with small spread
    std::vector<double> simple_transfer_function() {
        std::vector<double> tf(2 * n_pixels_);
        for (int j = 0; j < rows_; ++j) {
            for (int i = 0; i < cols_; ++i) {
                int idx = j * cols_ + i;
                // Simple: real part = 1.0, imaginary part = 0.0
                tf[2 * idx] = 1.0;
                tf[2 * idx + 1] = 0.0;
            }
        }
        return tf;
    }

    // Compute relative error ||a - b|| / ||b||
    double relative_error(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
        double diff_norm = 0.0;
        double b_norm = 0.0;
        for (int j = 0; j < a.rows(); ++j) {
            for (int i = 0; i < a.cols(); ++i) {
                diff_norm += std::norm(a(j, i) - b(j, i));
                b_norm += std::norm(b(j, i));
            }
        }
        return std::sqrt(diff_norm) / std::sqrt(b_norm);
    }

    // Check if matrix contains any NaN or Inf values
    bool has_nan_or_inf(const Eigen::MatrixXcd& m) {
        for (int j = 0; j < m.rows(); ++j) {
            for (int i = 0; i < m.cols(); ++i) {
                if (!std::isfinite(m(j, i).real()) || !std::isfinite(m(j, i).imag())) {
                    return true;
                }
            }
        }
        return false;
    }

    int rows_;
    int cols_;
    int n_pixels_;
};

// =============================================================================
// Identity Preconditioner Tests
// =============================================================================

TEST_F(PreconditionerTest, IdentityPreconditioner_ReturnsInput) {
    IdentityPreconditioner precond;
    auto r = random_matrix(rows_, cols_, 42);

    auto result = precond.apply(r);

    // Output should equal input
    for (int j = 0; j < rows_; ++j) {
        for (int i = 0; i < cols_; ++i) {
            EXPECT_DOUBLE_EQ(result(j, i).real(), r(j, i).real());
            EXPECT_DOUBLE_EQ(result(j, i).imag(), r(j, i).imag());
        }
    }
}

// =============================================================================
// Jacobi Preconditioner Tests
// =============================================================================

TEST_F(PreconditionerTest, JacobiPreconditioner_MatchesDiagonal) {
    // diagonal = [1, 2, 3, 4, ...] repeated
    std::vector<double> diagonal(n_pixels_);
    for (int i = 0; i < n_pixels_; ++i) {
        diagonal[i] = 1.0 + i % 4;  // 1, 2, 3, 4, 1, 2, 3, 4, ...
    }

    JacobiPreconditioner precond(diagonal, rows_, cols_);

    // Test with unit basis vector
    Eigen::MatrixXcd r = Eigen::MatrixXcd::Zero(rows_, cols_);
    r(0, 0) = 1.0;

    auto result = precond.apply(r);

    // P⁻¹ · e_0 = e_0 / d_0 = 1/1 = 1
    EXPECT_NEAR(result(0, 0).real(), 1.0 / diagonal[0], 1e-10);
    EXPECT_NEAR(result(0, 0).imag(), 0.0, 1e-10);

    // Test another position
    r = Eigen::MatrixXcd::Zero(rows_, cols_);
    r(0, 1) = 1.0;
    result = precond.apply(r);
    EXPECT_NEAR(result(0, 1).real(), 1.0 / diagonal[1], 1e-10);
}

TEST_F(PreconditionerTest, JacobiPreconditioner_ClampsSmallValues) {
    // Diagonal with some zeros
    std::vector<double> diagonal(n_pixels_, 1.0);
    diagonal[0] = 0.0;
    diagonal[1] = 1e-15;  // Very small
    diagonal[2] = -1e-15; // Small negative (shouldn't happen but test robustness)

    JacobiPreconditioner precond(diagonal, rows_, cols_);
    auto r = random_matrix(rows_, cols_, 100);

    auto result = precond.apply(r);

    // Should not produce NaN or Inf
    EXPECT_FALSE(has_nan_or_inf(result));
}

TEST_F(PreconditionerTest, JacobiPreconditioner_IdentityForOnes) {
    // All diagonal entries = 1
    std::vector<double> diagonal(n_pixels_, 1.0);

    JacobiPreconditioner precond(diagonal, rows_, cols_);
    auto r = random_matrix(rows_, cols_, 200);

    auto result = precond.apply(r);

    // P⁻¹ = I when diagonal is all ones
    for (int j = 0; j < rows_; ++j) {
        for (int i = 0; i < cols_; ++i) {
            EXPECT_NEAR(result(j, i).real(), r(j, i).real(), 1e-10);
            EXPECT_NEAR(result(j, i).imag(), r(j, i).imag(), 1e-10);
        }
    }
}

TEST_F(PreconditionerTest, JacobiPreconditioner_CombinesOperatorAndRegularizer) {
    // Test the constructor that combines operator diagonal with regularizer
    auto op_diag = uniform_diagonal(2.0);
    auto reg_spec = uniform_diagonal(1.0);
    double lambda = 0.5;

    // Combined diagonal should be 2.0 + 0.5*1.0 = 2.5
    JacobiPreconditioner precond(op_diag, reg_spec, lambda, rows_, cols_);

    Eigen::MatrixXcd r = Eigen::MatrixXcd::Constant(rows_, cols_, {1.0, 0.0});
    auto result = precond.apply(r);

    // Expected: 1.0 / 2.5 = 0.4
    EXPECT_NEAR(result(0, 0).real(), 0.4, 1e-10);
}

// =============================================================================
// Circulant Preconditioner Tests
// =============================================================================

TEST_F(PreconditionerTest, CirculantPreconditioner_NoNaNWithValidInputs) {
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);
    double lambda = 0.1;
    double epsilon = 1e-6;

    CirculantPreconditioner precond(tf, reg_spec, lambda, epsilon, rows_, cols_);
    auto r = random_matrix(rows_, cols_, 300);

    auto result = precond.apply(r);

    EXPECT_FALSE(has_nan_or_inf(result));
}

TEST_F(PreconditionerTest, CirculantPreconditioner_HandlesNullsWithEpsilon) {
    // Transfer function with zeros
    std::vector<double> tf(2 * n_pixels_, 0.0);
    auto reg_spec = uniform_diagonal(0.0);  // Zero regularizer too
    double lambda = 0.0;
    double epsilon = 1e-6;

    CirculantPreconditioner precond(tf, reg_spec, lambda, epsilon, rows_, cols_);
    auto r = random_matrix(rows_, cols_, 400);

    auto result = precond.apply(r);

    // Should not produce NaN/Inf due to epsilon clamping
    EXPECT_FALSE(has_nan_or_inf(result));
}

TEST_F(PreconditionerTest, CirculantPreconditioner_InvertsConvolution) {
    // If transfer function is uniform (no convolution), P⁻¹ ≈ scaled identity
    std::vector<double> tf(2 * n_pixels_);
    for (int i = 0; i < n_pixels_; ++i) {
        tf[2 * i] = 2.0;      // Real part = 2
        tf[2 * i + 1] = 0.0;  // Imaginary = 0
    }
    auto reg_spec = uniform_diagonal(1.0);
    double lambda = 1.0;
    double epsilon = 1e-10;

    // Filter = 2 + 1*1 + eps = 3 + eps
    // Inverse filter ≈ 1/3
    CirculantPreconditioner precond(tf, reg_spec, lambda, epsilon, rows_, cols_);

    Eigen::MatrixXcd r = Eigen::MatrixXcd::Constant(rows_, cols_, {1.0, 0.0});
    auto result = precond.apply(r);

    // Result should be approximately r / 3
    double expected = 1.0 / 3.0;
    for (int j = 0; j < rows_; ++j) {
        for (int i = 0; i < cols_; ++i) {
            EXPECT_NEAR(result(j, i).real(), expected, 0.01);
            EXPECT_NEAR(result(j, i).imag(), 0.0, 0.01);
        }
    }
}

TEST_F(PreconditionerTest, CirculantPreconditioner_InvalidEpsilonThrows) {
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);

    EXPECT_THROW(
        CirculantPreconditioner(tf, reg_spec, 0.1, 0.0, rows_, cols_),
        std::invalid_argument);

    EXPECT_THROW(
        CirculantPreconditioner(tf, reg_spec, 0.1, -1e-6, rows_, cols_),
        std::invalid_argument);
}

// =============================================================================
// Hybrid Preconditioner Tests
// =============================================================================

TEST_F(PreconditionerTest, HybridPreconditioner_NoNaNWithValidInputs) {
    auto op_diag = uniform_diagonal(1.0);
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);
    double lambda = 0.1;
    double epsilon = 1e-6;

    HybridPreconditioner precond(op_diag, tf, reg_spec, lambda, epsilon, rows_, cols_);
    auto r = random_matrix(rows_, cols_, 500);

    auto result = precond.apply(r);

    EXPECT_FALSE(has_nan_or_inf(result));
}

TEST_F(PreconditionerTest, HybridPreconditioner_CombinesDiagonalAndCirculant) {
    auto op_diag = uniform_diagonal(4.0);
    std::vector<double> tf(2 * n_pixels_);
    for (int i = 0; i < n_pixels_; ++i) {
        tf[2 * i] = 1.0;
        tf[2 * i + 1] = 0.0;
    }
    auto reg_spec = uniform_diagonal(0.0);
    double lambda = 0.0;
    double epsilon = 1e-10;

    HybridPreconditioner precond(op_diag, tf, reg_spec, lambda, epsilon, rows_, cols_);

    // For uniform diagonal D = 4I and identity-like circulant:
    // P⁻¹ = D^{-1/2} C⁻¹ D^{-1/2} = (1/2) * 1 * (1/2) = 1/4
    Eigen::MatrixXcd r = Eigen::MatrixXcd::Constant(rows_, cols_, {1.0, 0.0});
    auto result = precond.apply(r);

    double expected = 0.25;  // 1/4
    EXPECT_NEAR(result(0, 0).real(), expected, 0.05);
}

TEST_F(PreconditionerTest, HybridPreconditioner_DiffersFromPureCirculant) {
    // With non-uniform diagonal, hybrid should differ from circulant
    std::vector<double> op_diag(n_pixels_);
    for (int i = 0; i < n_pixels_; ++i) {
        op_diag[i] = 1.0 + 0.5 * (i % 4);  // Varying diagonal
    }
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(0.1);
    double lambda = 0.1;
    double epsilon = 1e-6;

    CirculantPreconditioner circ(tf, reg_spec, lambda, epsilon, rows_, cols_);
    HybridPreconditioner hybrid(op_diag, tf, reg_spec, lambda, epsilon, rows_, cols_);

    auto r = random_matrix(rows_, cols_, 600);
    auto circ_result = circ.apply(r);
    auto hybrid_result = hybrid.apply(r);

    // Results should be different
    double diff = relative_error(circ_result, hybrid_result);
    EXPECT_GT(diff, 0.01);  // At least 1% different
}

// =============================================================================
// Factory Tests
// =============================================================================

TEST_F(PreconditionerTest, CreatePreconditioner_None) {
    auto op_diag = uniform_diagonal(1.0);
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);

    auto variant = create_preconditioner(
        PRECONDITIONER_NONE, op_diag, tf, reg_spec, 0.1, 1e-6, rows_, cols_);

    const Preconditioner& precond = get_preconditioner(variant);

    // Verify it's identity by behavior
    auto r = random_matrix(rows_, cols_, 700);
    auto result = precond.apply(r);

    for (int j = 0; j < rows_; ++j) {
        for (int i = 0; i < cols_; ++i) {
            EXPECT_DOUBLE_EQ(result(j, i).real(), r(j, i).real());
            EXPECT_DOUBLE_EQ(result(j, i).imag(), r(j, i).imag());
        }
    }
}

TEST_F(PreconditionerTest, CreatePreconditioner_Jacobi) {
    auto op_diag = uniform_diagonal(2.0);
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);
    double lambda = 1.0;

    auto variant = create_preconditioner(
        PRECONDITIONER_JACOBI, op_diag, tf, reg_spec, lambda, 1e-6, rows_, cols_);

    const Preconditioner& precond = get_preconditioner(variant);

    // Verify it's Jacobi: constant input should scale by 1/(2+1) = 1/3
    Eigen::MatrixXcd r = Eigen::MatrixXcd::Constant(rows_, cols_, {1.0, 0.0});
    auto result = precond.apply(r);
    EXPECT_NEAR(result(0, 0).real(), 1.0 / 3.0, 1e-10);
}

TEST_F(PreconditionerTest, CreatePreconditioner_Circulant) {
    auto op_diag = uniform_diagonal(1.0);
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);

    auto variant = create_preconditioner(
        PRECONDITIONER_CIRCULANT, op_diag, tf, reg_spec, 0.1, 1e-6, rows_, cols_);

    const Preconditioner& precond = get_preconditioner(variant);

    auto r = random_matrix(rows_, cols_, 800);
    auto result = precond.apply(r);
    EXPECT_FALSE(has_nan_or_inf(result));
}

TEST_F(PreconditionerTest, CreatePreconditioner_Hybrid) {
    auto op_diag = uniform_diagonal(1.0);
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);

    auto variant = create_preconditioner(
        PRECONDITIONER_HYBRID, op_diag, tf, reg_spec, 0.1, 1e-6, rows_, cols_);

    const Preconditioner& precond = get_preconditioner(variant);

    auto r = random_matrix(rows_, cols_, 900);
    auto result = precond.apply(r);
    EXPECT_FALSE(has_nan_or_inf(result));
}

TEST_F(PreconditionerTest, CreatePreconditioner_Unknown_Throws) {
    auto op_diag = uniform_diagonal(1.0);
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);

    EXPECT_THROW(
        create_preconditioner(PRECONDITIONER_UNKNOWN, op_diag, tf, reg_spec,
                              0.1, 1e-6, rows_, cols_),
        std::invalid_argument);
}

// =============================================================================
// Base Class Interface Tests
// =============================================================================

TEST_F(PreconditionerTest, BaseClassInterface_WorksWithAllTypes) {
    auto op_diag = uniform_diagonal(1.0);
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);

    std::vector<PreconditionerVariant> variants;
    variants.push_back(create_preconditioner(
        PRECONDITIONER_NONE, op_diag, tf, reg_spec, 0.1, 1e-6, rows_, cols_));
    variants.push_back(create_preconditioner(
        PRECONDITIONER_JACOBI, op_diag, tf, reg_spec, 0.1, 1e-6, rows_, cols_));
    variants.push_back(create_preconditioner(
        PRECONDITIONER_CIRCULANT, op_diag, tf, reg_spec, 0.1, 1e-6, rows_, cols_));
    variants.push_back(create_preconditioner(
        PRECONDITIONER_HYBRID, op_diag, tf, reg_spec, 0.1, 1e-6, rows_, cols_));

    auto r = random_matrix(rows_, cols_, 1000);

    for (auto& v : variants) {
        const Preconditioner& precond = get_preconditioner(v);

        auto result = precond.apply(r);
        EXPECT_EQ(result.rows(), r.rows());
        EXPECT_EQ(result.cols(), r.cols());
        EXPECT_FALSE(has_nan_or_inf(result));
    }
}

// =============================================================================
// Additional Validation Tests
// =============================================================================

TEST_F(PreconditionerTest, AllPreconditioners_OutputSameSize) {
    auto op_diag = uniform_diagonal(1.0);
    auto tf = simple_transfer_function();
    auto reg_spec = uniform_diagonal(1.0);

    // Test with non-square grid
    int rows = 8;
    int cols = 16;
    int n = rows * cols;

    std::vector<double> op_diag_rect(n, 1.0);
    std::vector<double> tf_rect(2 * n, 1.0);
    std::vector<double> reg_rect(n, 1.0);

    IdentityPreconditioner identity;
    JacobiPreconditioner jacobi(op_diag_rect, rows, cols);
    CirculantPreconditioner circulant(tf_rect, reg_rect, 0.1, 1e-6, rows, cols);
    HybridPreconditioner hybrid(op_diag_rect, tf_rect, reg_rect, 0.1, 1e-6, rows, cols);

    auto r = random_matrix(rows, cols, 1100);

    EXPECT_EQ(identity.apply(r).rows(), rows);
    EXPECT_EQ(identity.apply(r).cols(), cols);
    EXPECT_EQ(jacobi.apply(r).rows(), rows);
    EXPECT_EQ(jacobi.apply(r).cols(), cols);
    EXPECT_EQ(circulant.apply(r).rows(), rows);
    EXPECT_EQ(circulant.apply(r).cols(), cols);
    EXPECT_EQ(hybrid.apply(r).rows(), rows);
    EXPECT_EQ(hybrid.apply(r).cols(), cols);
}

TEST_F(PreconditionerTest, JacobiPreconditioner_SizeMismatchThrows) {
    std::vector<double> wrong_size(10);  // Wrong size
    EXPECT_THROW(
        JacobiPreconditioner(wrong_size, rows_, cols_),
        std::invalid_argument);
}

TEST_F(PreconditionerTest, CirculantPreconditioner_SizeMismatchThrows) {
    std::vector<double> tf(10);  // Wrong size
    auto reg_spec = uniform_diagonal(1.0);

    EXPECT_THROW(
        CirculantPreconditioner(tf, reg_spec, 0.1, 1e-6, rows_, cols_),
        std::invalid_argument);
}

}  // namespace aurora::reconstruction
