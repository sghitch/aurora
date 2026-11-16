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

#ifndef AURORA_RECONSTRUCTION_PRECONDITIONER_H_
#define AURORA_RECONSTRUCTION_PRECONDITIONER_H_

/**
 * @file preconditioner.h
 * @brief Preconditioners for PCG reconstruction solver.
 *
 * Implements preconditioners that accelerate PCG convergence by approximating
 * the inverse of the normal equations matrix:
 *   H = A†WA + λL†L
 *
 * At each PCG iteration, the preconditioner applies:
 *   z = P⁻¹r
 *
 * where r is the residual. A good preconditioner P ≈ H makes PCG converge
 * in 3-6 iterations instead of 50+.
 *
 * Available preconditioners:
 * - Identity: No preconditioning (P = I)
 * - Jacobi: P = diag(H), simple diagonal scaling
 * - Circulant: P = FFT-based Toeplitz approximation
 * - Hybrid: P = D^{1/2} · C · D^{1/2}, combines both
 *
 * @see pcg_config.proto for PreconditionerType enum
 * @see pcg_precomputed_data.proto for precomputed data layout
 */

#include <complex>
#include <memory>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "pcg_config.pb.h"

namespace aurora::reconstruction {

/**
 * @brief Abstract base class for PCG preconditioners.
 *
 * Preconditioners accelerate conjugate gradient convergence by approximating
 * the inverse of the normal equations matrix H = A†WA + λL†L.
 */
class Preconditioner {
  public:
    virtual ~Preconditioner() = default;

    /**
     * @brief Apply preconditioner inverse P⁻¹ to residual.
     *
     * Computes z = P⁻¹r where P approximates the normal equations matrix.
     *
     * @param r Residual vector as complex image (Ny x Nx)
     * @return Preconditioned residual (same size as input)
     */
    virtual Eigen::MatrixXcd apply(const Eigen::MatrixXcd& r) const = 0;
};

/**
 * @brief Identity preconditioner: P = I (no preconditioning).
 *
 * P⁻¹r = r
 *
 * Equivalent to unpreconditioned CG. Useful as a baseline.
 */
class IdentityPreconditioner final : public Preconditioner {
  public:
    Eigen::MatrixXcd apply(const Eigen::MatrixXcd& r) const override {
        return r;  // Identity
    }
};

/**
 * @brief Jacobi (diagonal) preconditioner: P = diag(H).
 *
 * P⁻¹r = r ./ diag(A†WA + λL†L)
 *
 * Simple diagonal scaling. Cheap to compute and apply (O(N)), but provides
 * limited speedup (typically 2-3x fewer iterations).
 *
 * The diagonal is clamped to avoid division by near-zero values:
 *   inv_diagonal[i] = 1 / max(diagonal[i], max_diagonal * 1e-10)
 */
class JacobiPreconditioner final : public Preconditioner {
  public:
    /**
     * @brief Construct Jacobi preconditioner from precomputed diagonal.
     *
     * @param diagonal Diagonal of H = A†WA + λL†L, stored row-major as Ny*Nx vector
     * @param rows Number of rows (Ny) in the image grid
     * @param cols Number of columns (Nx) in the image grid
     */
    JacobiPreconditioner(const std::vector<double>& diagonal, int rows, int cols);

    /**
     * @brief Construct from operator diagonal and regularizer spectrum.
     *
     * Computes diagonal = operator_diagonal + lambda * regularizer_diagonal
     *
     * @param operator_diagonal Diagonal of A†WA
     * @param regularizer_spectrum Diagonal of L†L (frequency response at DC for each pixel)
     * @param lambda Regularization strength
     * @param rows Number of rows (Ny)
     * @param cols Number of columns (Nx)
     */
    JacobiPreconditioner(const std::vector<double>& operator_diagonal,
                         const std::vector<double>& regularizer_spectrum,
                         double lambda,
                         int rows,
                         int cols);

    Eigen::MatrixXcd apply(const Eigen::MatrixXcd& r) const override;

  private:
    Eigen::MatrixXd inv_diagonal_;  ///< 1/diagonal, clamped for stability
};

/**
 * @brief Circulant (FFT-based) preconditioner: P ≈ H as circulant matrix.
 *
 * Approximates H as a circulant matrix (convolution operator) and inverts via FFT:
 *   P⁻¹r = F⁻¹(Fr ./ (Ψ + λΦ + ε))
 *
 * where:
 * - Ψ = transfer function (FFT of PSF of A†WA)
 * - Φ = regularizer spectrum (FFT of L†L kernel)
 * - ε = stabilizer for division near zeros
 *
 * Cost: O(N log N) per application. Provides 3-5x speedup for uniform K-space.
 */
class CirculantPreconditioner final : public Preconditioner {
  public:
    /**
     * @brief Construct circulant preconditioner from frequency-domain data.
     *
     * @param transfer_function Ψ = FFT{PSF}, complex interleaved [re,im,re,im,...], Ny*Nx*2
     * @param regularizer_spectrum Φ = frequency response of L†L, real Ny*Nx
     * @param lambda Regularization strength
     * @param epsilon Stabilizer for division (prevents blowup near zeros)
     * @param rows Number of rows (Ny)
     * @param cols Number of columns (Nx)
     */
    CirculantPreconditioner(const std::vector<double>& transfer_function,
                            const std::vector<double>& regularizer_spectrum,
                            double lambda,
                            double epsilon,
                            int rows,
                            int cols);

    ~CirculantPreconditioner();

    CirculantPreconditioner(CirculantPreconditioner&&) noexcept;
    CirculantPreconditioner& operator=(CirculantPreconditioner&&) noexcept;

    // Non-copyable due to FFT state
    CirculantPreconditioner(const CirculantPreconditioner&) = delete;
    CirculantPreconditioner& operator=(const CirculantPreconditioner&) = delete;

    Eigen::MatrixXcd apply(const Eigen::MatrixXcd& r) const override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Eigen::MatrixXcd inv_filter_;  ///< 1 / (Ψ + λΦ + ε)
    int rows_;
    int cols_;
};

/**
 * @brief Hybrid preconditioner: P = D^{1/2} · C · D^{1/2}.
 *
 * Combines diagonal (Jacobi) and circulant preconditioning for best convergence:
 *   P⁻¹r = D^{-1/2} · C⁻¹ · D^{-1/2} · r
 *
 * The diagonal component handles spatially-varying operator weights, while
 * the circulant component captures the convolution structure.
 *
 * Cost: O(N log N) per application. Provides 5-10x speedup, recommended for production.
 */
class HybridPreconditioner final : public Preconditioner {
  public:
    /**
     * @brief Construct hybrid preconditioner.
     *
     * @param operator_diagonal Diagonal of A†WA, real Ny*Nx
     * @param transfer_function Ψ = FFT{PSF}, complex interleaved, Ny*Nx*2
     * @param regularizer_spectrum Φ = frequency response of L†L, real Ny*Nx
     * @param lambda Regularization strength
     * @param epsilon Stabilizer for division
     * @param rows Number of rows (Ny)
     * @param cols Number of columns (Nx)
     */
    HybridPreconditioner(const std::vector<double>& operator_diagonal,
                         const std::vector<double>& transfer_function,
                         const std::vector<double>& regularizer_spectrum,
                         double lambda,
                         double epsilon,
                         int rows,
                         int cols);

    ~HybridPreconditioner();

    HybridPreconditioner(HybridPreconditioner&&) noexcept;
    HybridPreconditioner& operator=(HybridPreconditioner&&) noexcept;

    // Non-copyable due to FFT state in circulant component
    HybridPreconditioner(const HybridPreconditioner&) = delete;
    HybridPreconditioner& operator=(const HybridPreconditioner&) = delete;

    Eigen::MatrixXcd apply(const Eigen::MatrixXcd& r) const override;

  private:
    Eigen::MatrixXd inv_sqrt_diagonal_;  ///< D^{-1/2}
    std::unique_ptr<CirculantPreconditioner> circulant_;
};

/**
 * @brief Variant type holding any preconditioner with value semantics.
 *
 * Note: CirculantPreconditioner and HybridPreconditioner are move-only due to
 * FFT state, so this variant is also move-only.
 */
using PreconditionerVariant = std::variant<
    IdentityPreconditioner,
    JacobiPreconditioner,
    CirculantPreconditioner,
    HybridPreconditioner>;

/**
 * @brief Get reference to Preconditioner base class from variant.
 *
 * @param v Preconditioner variant
 * @return Reference to the Preconditioner interface
 */
inline const Preconditioner& get_preconditioner(const PreconditionerVariant& v) {
    return std::visit([](const auto& p) -> const Preconditioner& { return p; }, v);
}

/**
 * @brief Get mutable reference to Preconditioner base class from variant.
 *
 * @param v Preconditioner variant
 * @return Reference to the Preconditioner interface
 */
inline Preconditioner& get_preconditioner(PreconditionerVariant& v) {
    return std::visit([](auto& p) -> Preconditioner& { return p; }, v);
}

/**
 * @brief Factory function to create preconditioner from type and precomputed data.
 *
 * Creates the appropriate preconditioner based on the PreconditionerType enum.
 * For PRECONDITIONER_NONE, returns IdentityPreconditioner.
 *
 * @param type Preconditioner type from protobuf enum
 * @param operator_diagonal Diagonal of A†WA (from PcgPrecomputedData)
 * @param transfer_function FFT of PSF (from PcgPrecomputedData), complex interleaved
 * @param regularizer_spectrum Frequency response of L†L (from PcgPrecomputedData)
 * @param lambda Regularization strength
 * @param epsilon Division stabilizer (only used for CIRCULANT and HYBRID)
 * @param rows Grid rows (Ny)
 * @param cols Grid columns (Nx)
 * @return Preconditioner variant
 * @throws std::invalid_argument if type is PRECONDITIONER_UNKNOWN
 */
PreconditionerVariant create_preconditioner(
    PreconditionerType type,
    const std::vector<double>& operator_diagonal,
    const std::vector<double>& transfer_function,
    const std::vector<double>& regularizer_spectrum,
    double lambda,
    double epsilon,
    int rows,
    int cols);

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_PRECONDITIONER_H_
