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

#ifndef AURORA_RECONSTRUCTION_PCG_OPERATOR_H_
#define AURORA_RECONSTRUCTION_PCG_OPERATOR_H_

/**
 * @file pcg_operator.h
 * @brief PCG-based iterative reconstruction operator.
 *
 * Implements the Preconditioned Conjugate Gradient (PCG) algorithm for solving
 * the regularized least-squares problem:
 *   min_x ||W^{1/2}(Ax - y)||₂² + λ||Lx||₂²
 *
 * This is equivalent to solving the normal equations:
 *   (A†WA + λL†L)x = A†Wy
 *
 * PCG wraps an inner operator (NufftOperator or DirectOperator) and uses it
 * for forward/adjoint operations within the iterative solve.
 *
 * ### Algorithm
 *
 * Input: H (normal operator), b (RHS), P (preconditioner), x₀ (initial guess)
 * Output: x (approximate solution)
 *
 * 1.  r₀ = b - H·x₀
 * 2.  z₀ = P⁻¹·r₀
 * 3.  p₀ = z₀
 * 4.  for k = 0, 1, ..., max_iter-1:
 * 5.      q_k = H·p_k
 * 6.      α_k = ⟨r_k, z_k⟩ / ⟨p_k, q_k⟩
 * 7.      x_{k+1} = x_k + α_k·p_k
 * 8.      r_{k+1} = r_k - α_k·q_k
 * 9.      if ||r_{k+1}|| / ||r₀|| < tolerance: break
 * 10.     z_{k+1} = P⁻¹·r_{k+1}
 * 11.     β_k = ⟨r_{k+1}, z_{k+1}⟩ / ⟨r_k, z_k⟩
 * 12.     p_{k+1} = z_{k+1} + β_k·p_k
 * 13. return x_{k+1}
 *
 * @see ReconstructionOperator for the base interface
 * @see NufftOperator typically used as the inner operator
 * @see DirectOperator for reference inner operator
 */

#include "reconstruction/reconstruction_operator.h"
#include "reconstruction/direct_operator.h"
#include "reconstruction/nufft_operator.h"
#include "reconstruction/preconditioner.h"
#include "reconstruction/regularizer.h"
#include "bandwidth_config.pb.h"
#include "pcg_config.pb.h"
#include "k_space_slice.pb.h"

namespace aurora::forward {
class ForwardModel;
}  // namespace aurora::forward

#include <variant>

namespace aurora::reconstruction {

/**
 * @brief Validate PcgConfig fields.
 *
 * Checks all required fields and value ranges.
 *
 * @param config PcgConfig to validate
 * @throws std::invalid_argument if any field is invalid
 */
void validate_pcg_config(const PcgConfig& config);

/**
 * @brief Variant type for inner reconstruction operator.
 *
 * Provides value semantics for polymorphic operator storage.
 */
using InnerOperatorVariant = std::variant<DirectOperator, NufftOperator>;

/**
 * @brief Get reference to ReconstructionOperator base class from variant.
 */
inline const ReconstructionOperator& get_inner_operator(const InnerOperatorVariant& v) {
    return std::visit([](const auto& op) -> const ReconstructionOperator& { return op; }, v);
}

/**
 * @brief PCG-based iterative reconstruction operator.
 *
 * Inherits from ReconstructionOperator, like DirectOperator, NufftOperator,
 * GriddingOperator, and FastOnlineOperator.
 *
 * The forward() method delegates to the inner operator.
 * The adjoint() method performs PCG iteration using the inner operator's
 * forward/adjoint for the normal equations.
 *
 * ### Example Usage
 * @code
 * PcgConfig config;
 * config.set_max_iterations(50);
 * config.set_tolerance(1e-6);
 * config.set_regularizer_type(REGULARIZER_TIKHONOV);
 * config.set_regularizer_lambda(1e-4);
 * config.set_preconditioner_type(PRECONDITIONER_JACOBI);
 *
 * PcgOperator op(slice, forward_model, bandwidth_config,
 *                grid_size, grid_spacing, grid_origin, config);
 *
 * auto result = op.adjoint(k_samples, origin, spacing, size);
 * std::cout << "Converged in " << op.last_iterations() << " iterations\n";
 * @endcode
 */
class PcgOperator : public ReconstructionOperator {
  public:
    /**
     * @brief Construct with MeasurementOperator physics.
     *
     * Creates an inner operator (NufftOperator or DirectOperator based on config)
     * and initializes the regularizer. Note: Currently uses identity preconditioning
     * regardless of preconditioner_type; full preconditioner support requires
     * precomputed data from PcgPrecomputation.
     *
     * @param slice KSpaceSlice containing K-space samples (may have bandwidth)
     * @param forward_model ForwardModel providing the per-pixel K-space bandwidth
     * @param bandwidth_config Bandwidth configuration (window type, quadrature order)
     * @param grid_size Target reconstruction grid size (Nx, Ny)
     * @param grid_spacing Grid spacing (dx, dy) in meters
     * @param grid_origin Grid origin (x, y) in meters
     * @param config PCG configuration (all required fields must be set)
     *
     * @throws std::invalid_argument if config invalid or required fields missing
     */
    PcgOperator(const aurora::shared::KSpaceSlice& slice,
                const aurora::forward::ForwardModel& forward_model,
                const aurora::forward::BandwidthConfig& bandwidth_config,
                const Eigen::Vector2i& grid_size,
                const Eigen::Vector2d& grid_spacing,
                const Eigen::Vector2d& grid_origin,
                const PcgConfig& config);

    ~PcgOperator() override;

    // Non-copyable, movable
    PcgOperator(const PcgOperator&) = delete;
    PcgOperator& operator=(const PcgOperator&) = delete;
    PcgOperator(PcgOperator&&) noexcept;
    PcgOperator& operator=(PcgOperator&&) noexcept;

    // =========================================================================
    // ReconstructionOperator interface
    // =========================================================================

    /**
     * @brief Forward operator: scene -> K-space.
     *
     * Delegates to inner operator (NufftOperator or DirectOperator).
     */
    Eigen::VectorXcd forward(
        const Eigen::MatrixXcd& scene,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing) const override;

    /**
     * @brief Adjoint operator: K-space -> scene via PCG iteration.
     *
     * Instead of simple backprojection, performs PCG iteration to solve:
     *   (A†WA + λL†L)x = A†Wy
     *
     * Uses inner operator's forward() and adjoint() within the iteration loop.
     */
    Eigen::MatrixXcd adjoint(
        const Eigen::VectorXcd& k_samples,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing,
        const Eigen::Vector2i& grid_size) const override;

    size_t num_samples() const override;
    const std::vector<Eigen::Vector2d>& k_vectors() const override;
    const std::vector<double>& weights() const override;

    // =========================================================================
    // PCG-specific accessors
    // =========================================================================

    /** @brief Get last solve iteration count. */
    int last_iterations() const { return last_iterations_; }

    /** @brief Get last solve residual norm. */
    double last_residual() const { return last_residual_; }

    /** @brief Get configuration. */
    const PcgConfig& config() const { return config_; }

    /** @brief Access inner operator (for testing). */
    const ReconstructionOperator& inner_operator() const { return get_inner_operator(inner_operator_); }

  private:
    PcgConfig config_;

    // Inner operator for forward/adjoint (NufftOperator or DirectOperator)
    InnerOperatorVariant inner_operator_;

    // PCG components
    RegularizerVariant regularizer_;
    double lambda_ = 0.0;

    // Grid parameters (stored for creating preconditioner on-demand)
    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    Eigen::Vector2d grid_origin_;

    // Convergence tracking (mutable for const adjoint())
    mutable int last_iterations_ = 0;
    mutable double last_residual_ = 0.0;

    /**
     * @brief Apply normal operator H = A†WA + λL†L.
     *
     * @param x Input image
     * @param weights Measurement weights
     * @param origin Grid origin
     * @param spacing Grid spacing
     * @return H·x
     */
    Eigen::MatrixXcd apply_H(
        const Eigen::MatrixXcd& x,
        const Eigen::VectorXd& weights,
        const Eigen::Vector2d& origin,
        const Eigen::Vector2d& spacing) const;

    void initialize();
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_PCG_OPERATOR_H_
