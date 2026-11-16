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

#ifndef AURORA_RECONSTRUCTION_DIRECT_OPERATOR_H_
#define AURORA_RECONSTRUCTION_DIRECT_OPERATOR_H_

/**
 * @file direct_operator.h
 * @brief Direct summation forward/adjoint operator for reconstruction.
 *
 * Implements the ReconstructionOperator interface using direct O(N*M)
 * summation. This is the reference implementation - simple and always
 * correct. Use for small problems or as ground truth for testing other
 * operators.
 *
 * ### Bandwidth-Aware Mode
 *
 * The operator uses MeasurementOperator physics for bandwidth-aware weight
 * computation, ensuring reconstruction uses the same K-space physics as
 * simulation.
 *
 * The true measurement model is windowed in K-space:
 *   y_m = ∫ X(k) W_m(k) dk
 *
 * where W_m(k) = H(k - k_m) is the K-space window of the detector pixel
 * that acquired sample m, centered on the sample's K-position k_m. The
 * adjoint uses the same quadrature:
 *   x̂(r) = Σ_m Σ_q W_m(k_q) · y_m · exp(+i k_q · r) · Δk
 *
 * where k_q are quadrature points over the support of W_m.
 *
 * ### Complexity
 *
 * - Forward: O(N * M * Q²) where N = grid pixels, M = K-samples, Q = quadrature_order
 * - Adjoint: O(N * M * Q²)
 * - Memory: O(M) for K-vectors and weights
 *
 * For larger problems, use NufftOperator which has O(N log N + M*W^2)
 * complexity.
 *
 * @see ReconstructionOperator for interface documentation
 * @see NufftOperator for FFT-accelerated implementation
 */

#include "reconstruction/reconstruction_operator.h"
#include "reconstruction/bandwidth_weight_calculator.h"

#include "bandwidth_config.pb.h"
#include "k_space_slice.pb.h"

namespace aurora::forward {
class ForwardModel;
}  // namespace aurora::forward

namespace aurora::reconstruction {

/**
 * @brief Direct summation forward/adjoint operator.
 *
 * Implements forward and adjoint operators via direct DFT summation.
 * This provides a reference implementation that is simple to verify
 * and always correct.
 *
 * ### Example Usage
 * @code
 * // Construct with bandwidth support using MeasurementOperator physics
 * DirectOperator op(slice, forward_model, bandwidth_config,
 *                   grid_size, grid_spacing, grid_origin);
 *
 * // Verify adjoint consistency
 * assert(op.check_adjoint_consistency(x, y, origin, spacing) < 1e-10);
 *
 * // Simple adjoint reconstruction
 * auto dirty = op.adjoint(samples, origin, spacing, size);
 * @endcode
 */
class DirectOperator : public ReconstructionOperator {
  public:
    /**
     * @brief Construct with MeasurementOperator physics.
     *
     * Uses MeasurementOperator for weight computation, ensuring reconstruction
     * uses the same K-space physics as simulation. The bandwidth_config
     * controls the window function and quadrature resolution.
     *
     * @param slice KSpaceSlice containing K-space samples
     * @param forward_model ForwardModel providing the per-pixel K-space bandwidth
     * @param bandwidth_config Bandwidth configuration (window type, quadrature order)
     * @param grid_size Reconstruction grid size (Nx, Ny)
     * @param grid_spacing Grid pixel spacing (dx, dy) in meters
     * @param grid_origin Grid lower-left corner (x0, y0) in meters
     *
     * @throws std::invalid_argument if bandwidth_config is incomplete or the
     *         grid spacing is too coarse for the bandwidth-limited resolution
     */
    DirectOperator(const aurora::shared::KSpaceSlice& slice,
                   const aurora::forward::ForwardModel& forward_model,
                   const aurora::forward::BandwidthConfig& bandwidth_config,
                   const Eigen::Vector2i& grid_size,
                   const Eigen::Vector2d& grid_spacing,
                   const Eigen::Vector2d& grid_origin);

    // =========================================================================
    // ReconstructionOperator interface
    // =========================================================================

    /**
     * @brief Forward operator: scene -> K-space.
     *
     * Computes:
     * @f[
     *   y_m = w_m \sum_{i,j} x_{i,j} \cdot
     *         e^{-i (k_{x,m} \cdot r_{x,j} + k_{y,m} \cdot r_{y,i})}
     * @f]
     *
     * @param scene Complex reflectivity grid (Ny x Nx)
     * @param grid_origin Scene grid origin (x, y) in meters
     * @param grid_spacing Scene grid spacing (dx, dy) in meters
     * @return Complex K-space measurements (M x 1)
     */
    Eigen::VectorXcd forward(
        const Eigen::MatrixXcd& scene,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing) const override;

    /**
     * @brief Adjoint operator: K-space -> scene.
     *
     * Computes:
     * @f[
     *   x_{i,j} = \sum_{m} w_m \cdot y_m \cdot
     *             e^{+i (k_{x,m} \cdot r_{x,j} + k_{y,m} \cdot r_{y,i})}
     * @f]
     *
     * @param k_samples Complex K-space measurements
     * @param grid_origin Scene grid origin (x, y) in meters
     * @param grid_spacing Scene grid spacing (dx, dy) in meters
     * @param grid_size Scene grid dimensions (Nx, Ny)
     * @return Complex scene estimate (Ny x Nx)
     */
    Eigen::MatrixXcd adjoint(
        const Eigen::VectorXcd& k_samples,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing,
        const Eigen::Vector2i& grid_size) const override;

    /**
     * @brief Get number of K-space samples.
     * @return Number of K-samples
     */
    size_t num_samples() const override { return k_vectors_.size(); }

    /**
     * @brief Get K-vectors.
     * @return Reference to K-vector storage
     */
    const std::vector<Eigen::Vector2d>& k_vectors() const override {
        return k_vectors_;
    }

    /**
     * @brief Get per-sample weights.
     * @return Reference to sample weight storage
     */
    const std::vector<double>& weights() const override {
        return weights_;
    }

    // =========================================================================
    // Bandwidth-specific accessors
    // =========================================================================

    /**
     * @brief Check if bandwidth information is available.
     * @return true if at least one sample has a non-zero K-space bandwidth
     */
    bool has_bandwidth() const { return bandwidth_calc_.has_bandwidth(); }

    /**
     * @brief Get quadrature order being used.
     * @return Points per dimension for K-space quadrature
     */
    int quadrature_order() const { return quadrature_order_; }

    /**
     * @brief Get resolution limit from bandwidth.
     * @return (dx, dy) minimum resolvable separation
     */
    Eigen::Vector2d resolution_limit() const {
        return bandwidth_calc_.resolution_limit();
    }

  private:
    std::vector<Eigen::Vector2d> k_vectors_;
    std::vector<double> weights_;

    // Bandwidth support via composition
    BandwidthWeightCalculator bandwidth_calc_;
    aurora::forward::BandwidthWindowType window_type_ =
        aurora::forward::BANDWIDTH_WINDOW_UNSPECIFIED;

    // Quadrature order (points per dimension)
    int quadrature_order_ = 1;

    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    Eigen::Vector2d grid_origin_;
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_DIRECT_OPERATOR_H_
