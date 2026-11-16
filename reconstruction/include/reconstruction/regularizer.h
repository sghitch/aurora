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

#ifndef AURORA_RECONSTRUCTION_REGULARIZER_H_
#define AURORA_RECONSTRUCTION_REGULARIZER_H_

/**
 * @file regularizer.h
 * @brief Regularization operators for PCG reconstruction.
 *
 * Implements regularizers used in the PCG objective function:
 *   ||W^{1/2}(Ax - y)||₂² + λ||Lx||₂²
 *
 * The regularizer L†L appears in the normal equations:
 *   (A†WA + λL†L)x = A†Wy
 *
 * Each regularizer provides:
 * 1. Spatial domain operator: apply_LtL(x) computes L†Lx
 * 2. Frequency response: Φ(k) for FFT-based preconditioner (eigenvalue of L†L)
 *
 * @see pcg_config.proto for RegularizerType enum
 */

#include <cmath>
#include <variant>

#include <Eigen/Core>

#include "pcg_config.pb.h"

namespace aurora::reconstruction {

/**
 * @brief Abstract base class for regularization operators.
 *
 * Regularizers penalize different properties of the solution:
 * - Tikhonov: penalizes solution magnitude (L = I)
 * - Gradient: penalizes gradients, preserves edges (L = ∇)
 * - Laplacian: penalizes curvature, produces smooth solutions (L = ∇²)
 */
class Regularizer {
  public:
    virtual ~Regularizer() = default;

    /**
     * @brief Apply L†L operator in spatial domain.
     *
     * Computes the action of the regularizer's normal operator on the input.
     * This is used directly in the PCG iteration.
     *
     * @param x Input complex image (Ny x Nx)
     * @return L†Lx result (same size as input)
     */
    virtual Eigen::MatrixXcd apply_LtL(const Eigen::MatrixXcd& x) const = 0;

    /**
     * @brief Compute frequency response Φ for FFT-based preconditioner.
     *
     * Returns the spectral representation of L†L, which is the eigenvalue
     * of the circulant approximation at each frequency.
     *
     * @param size Grid dimensions (Nx, Ny)
     * @param spacing Grid spacing (dx, dy) in meters
     * @return Real-valued spectrum (Ny x Nx), standard FFT ordering (DC at [0,0])
     */
    virtual Eigen::MatrixXd frequency_response(
        const Eigen::Vector2i& size,
        const Eigen::Vector2d& spacing) const = 0;
};

/**
 * @brief Tikhonov regularizer: L = I (identity).
 *
 * L†Lx = x
 * Φ(k) = 1
 *
 * Penalizes solution magnitude uniformly across all frequencies.
 * Good general-purpose regularizer.
 */
class TikhonovRegularizer final : public Regularizer {
  public:
    Eigen::MatrixXcd apply_LtL(const Eigen::MatrixXcd& x) const override {
        return x;  // Identity
    }

    Eigen::MatrixXd frequency_response(
        const Eigen::Vector2i& size,
        const Eigen::Vector2d& /*spacing*/) const override {
        return Eigen::MatrixXd::Ones(size.y(), size.x());
    }
};

/**
 * @brief Gradient regularizer: L = ∇ (first derivative).
 *
 * Uses forward finite differences with periodic boundary conditions.
 * L†L corresponds to the negative discrete Laplacian (5-point stencil):
 *   (L†Lx)_{i,j} = 4x_{i,j} - x_{i+1,j} - x_{i-1,j} - x_{i,j+1} - x_{i,j-1}
 *
 * Frequency response (eigenvalue of L†L):
 *   Φ(kx, ky) = 4 - 2cos(kx) - 2cos(ky)  (unit spacing)
 *
 * Penalizes gradients, which helps preserve edges (TV-like behavior).
 * Zero response at DC means constant solutions are not penalized.
 */
class GradientRegularizer final : public Regularizer {
  public:
    Eigen::MatrixXcd apply_LtL(const Eigen::MatrixXcd& x) const override;

    Eigen::MatrixXd frequency_response(
        const Eigen::Vector2i& size,
        const Eigen::Vector2d& spacing) const override;
};

/**
 * @brief Laplacian regularizer: L = ∇² (second derivative).
 *
 * Uses the standard 5-point discrete Laplacian stencil with periodic boundaries.
 * Since L = ∇², we have L†L = (∇²)†(∇²) = ∇⁴ (biharmonic operator).
 * apply_LtL() applies the discrete Laplacian twice.
 *
 * Frequency response (eigenvalue of L†L = ∇⁴):
 *   Φ_L(kx, ky) = 4 - 2cos(kx) - 2cos(ky)           (Laplacian eigenvalue)
 *   Φ_{L†L}(kx, ky) = (4 - 2cos(kx) - 2cos(ky))²    (biharmonic eigenvalue)
 *
 * frequency_response() returns Φ_{L†L}, consistent with apply_LtL().
 *
 * Penalizes curvature, producing smooth solutions.
 * Zero response at DC means constant solutions are not penalized.
 */
class LaplacianRegularizer final : public Regularizer {
  public:
    Eigen::MatrixXcd apply_LtL(const Eigen::MatrixXcd& x) const override;

    Eigen::MatrixXd frequency_response(
        const Eigen::Vector2i& size,
        const Eigen::Vector2d& spacing) const override;
};

/**
 * @brief Variant type holding any regularizer with value semantics.
 *
 * Provides value semantics without heap allocation. Use std::visit
 * or get_regularizer() to access the underlying Regularizer interface.
 */
using RegularizerVariant = std::variant<
    TikhonovRegularizer,
    GradientRegularizer,
    LaplacianRegularizer>;

/**
 * @brief Get reference to Regularizer base class from variant.
 *
 * @param v Regularizer variant
 * @return Reference to the Regularizer interface
 */
inline const Regularizer& get_regularizer(const RegularizerVariant& v) {
    return std::visit([](const auto& r) -> const Regularizer& { return r; }, v);
}

/**
 * @brief Get mutable reference to Regularizer base class from variant.
 *
 * @param v Regularizer variant
 * @return Reference to the Regularizer interface
 */
inline Regularizer& get_regularizer(RegularizerVariant& v) {
    return std::visit([](auto& r) -> Regularizer& { return r; }, v);
}

/**
 * @brief Factory function to create regularizer from enum type.
 *
 * @param type Regularizer type from protobuf enum
 * @return Regularizer variant
 * @throws std::invalid_argument if type is REGULARIZER_UNKNOWN
 */
RegularizerVariant create_regularizer(RegularizerType type);

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_REGULARIZER_H_
