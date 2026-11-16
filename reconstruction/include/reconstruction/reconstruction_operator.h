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

#ifndef AURORA_RECONSTRUCTION_RECONSTRUCTION_OPERATOR_H_
#define AURORA_RECONSTRUCTION_RECONSTRUCTION_OPERATOR_H_

/**
 * @file reconstruction_operator.h
 * @brief Abstract interface for forward/adjoint reconstruction operators.
 *
 * Defines the ReconstructionOperator abstract base class that provides a
 * common interface for different reconstruction operator implementations.
 * This abstraction allows plugging in different algorithms (DirectOperator,
 * NufftOperator, GPU operators, learned operators) without changing the
 * reconstruction pipeline.
 *
 * ### Mathematical Model
 *
 * Forward operator (scene -> K-space):
 * @f[
 *   y_m = w_m \sum_{i} x_i \cdot e^{-i \mathbf{k}_m \cdot \mathbf{r}_i}
 * @f]
 *
 * Adjoint operator (K-space -> scene):
 * @f[
 *   x_i = \sum_{m} w_m \cdot y_m \cdot e^{+i \mathbf{k}_m \cdot \mathbf{r}_i}
 * @f]
 *
 * Where:
 * - @f$ y_m @f$ is the m-th K-space measurement
 * - @f$ x_i @f$ is the i-th scene pixel
 * - @f$ w_m @f$ is the sample weight for the m-th K-sample
 * - @f$ \mathbf{k}_m @f$ is the K-vector for the m-th sample
 * - @f$ \mathbf{r}_i @f$ is the position of the i-th pixel
 *
 * ### Design Rationale
 *
 * Forward/adjoint operators are separated from ForwardModel because:
 * 1. **Single Responsibility**: ForwardModel handles K-space mapping;
 *    Reconstruction handles algorithms
 * 2. **Algorithm Evolution**: New operators (NUFFT, GPU, learned) can be
 *    added without touching forward module
 * 3. **Configuration Isolation**: Each operator has its own tuning parameters
 * 4. **Dependency Management**: Algorithm-specific libraries don't pollute
 *    the forward module
 *
 * @see DirectOperator for O(N*M) direct summation implementation
 * @see NufftOperator for FFT-accelerated implementation
 */

#include <complex>
#include <vector>

#include <Eigen/Core>

namespace aurora::reconstruction {

/**
 * @brief Abstract interface for forward/adjoint reconstruction operators.
 *
 * Provides the mathematical operators needed for iterative reconstruction
 * algorithms like ADMM and conjugate gradient. Implementations can vary
 * in computational complexity and numerical precision.
 *
 * ### Usage
 * @code
 * // Create operator (concrete implementation)
 * auto op = std::make_unique<DirectOperator>(slice);
 *
 * // Verify adjoint consistency (for debugging)
 * double error = op->check_adjoint_consistency(x, y, origin, spacing);
 * assert(error < 1e-10);
 *
 * // Simple adjoint reconstruction (dirty image)
 * auto dirty = op->adjoint(samples, origin, spacing, size);
 *
 * // Or use with an iterative solver such as PcgOperator
 * @endcode
 */
class ReconstructionOperator {
  public:
    virtual ~ReconstructionOperator() = default;

    // =========================================================================
    // PURE VIRTUAL INTERFACE
    // =========================================================================

    /**
     * @brief Forward operator: 2D scene -> K-space measurements.
     *
     * Maps a reflectivity scene to predicted K-space measurements.
     * Used in iterative reconstruction to compute residuals.
     *
     * @param scene Complex reflectivity grid (Ny x Nx)
     * @param grid_origin Scene grid origin (x, y) in meters
     * @param grid_spacing Scene grid spacing (dx, dy) in meters
     * @return Complex K-space measurements (one per K-vector)
     */
    virtual Eigen::VectorXcd forward(
        const Eigen::MatrixXcd& scene,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing) const = 0;

    /**
     * @brief Adjoint operator: K-space measurements -> scene estimate.
     *
     * Backprojects K-space samples to image domain. This is the
     * mathematical adjoint of the forward operator.
     *
     * @param k_samples Complex K-space measurements
     * @param grid_origin Scene grid origin (x, y) in meters
     * @param grid_spacing Scene grid spacing (dx, dy) in meters
     * @param grid_size Scene grid dimensions (Nx, Ny)
     * @return Complex scene estimate (Ny x Nx)
     */
    virtual Eigen::MatrixXcd adjoint(
        const Eigen::VectorXcd& k_samples,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing,
        const Eigen::Vector2i& grid_size) const = 0;

    /**
     * @brief Get number of K-space samples this operator works with.
     * @return Number of K-samples
     */
    virtual size_t num_samples() const = 0;

    /**
     * @brief Get K-vectors (for diagnostics/visualization).
     * @return Reference to K-vector storage
     */
    virtual const std::vector<Eigen::Vector2d>& k_vectors() const = 0;

    /**
     * @brief Get per-sample weights (for diagnostics).
     * @return Reference to sample weight storage
     */
    virtual const std::vector<double>& weights() const = 0;

    // =========================================================================
    // NON-VIRTUAL UTILITIES (use forward/adjoint internally)
    // =========================================================================

    /**
     * @brief Check forward-adjoint consistency: <Ax, y> ≈ <x, A*y>.
     *
     * Verifies that the forward and adjoint operators are mathematical
     * adjoints of each other. This is a critical correctness check.
     *
     * @param x Test scene (Ny x Nx)
     * @param y Test K-space samples
     * @param grid_origin Scene grid origin (x, y) in meters
     * @param grid_spacing Scene grid spacing (dx, dy) in meters
     * @return Relative error (should be < 1e-10 for correct implementation)
     */
    double check_adjoint_consistency(
        const Eigen::MatrixXcd& x,
        const Eigen::VectorXcd& y,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing) const;

    /**
     * @brief Compute A* A x (normal equations operator).
     *
     * Used by conjugate gradient and other solvers that need the
     * normal equations operator.
     *
     * @param x Input scene (Ny x Nx)
     * @param grid_origin Scene grid origin (x, y) in meters
     * @param grid_spacing Scene grid spacing (dx, dy) in meters
     * @return Output scene after applying A* A
     */
    Eigen::MatrixXcd apply_normal(
        const Eigen::MatrixXcd& x,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing) const;
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_RECONSTRUCTION_OPERATOR_H_
