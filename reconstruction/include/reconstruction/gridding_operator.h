#ifndef AURORA_RECONSTRUCTION_GRIDDING_OPERATOR_H_
#define AURORA_RECONSTRUCTION_GRIDDING_OPERATOR_H_

/**
 * @file gridding_operator.h
 * @brief Simple nearest-neighbor gridding operator for reconstruction.
 *
 * Implements the ReconstructionOperator interface using nearest-neighbor
 * gridding followed by FFT. This is faster than DirectOperator for large
 * grids but less accurate than NufftOperator due to the simple interpolation.
 *
 * ### Algorithm Overview
 *
 * **Adjoint (K-space -> Image):**
 * 1. Grid K-samples to nearest grid point (accumulate with weights)
 * 2. Normalize by weight count
 * 3. Apply inverse FFT
 *
 * **Forward (Image -> K-space):**
 * 1. Apply FFT
 * 2. Interpolate at K-positions (nearest neighbor)
 *
 * ### Complexity
 *
 * - Forward: O(N log N + M) where N = grid pixels, M = K-samples
 * - Adjoint: O(N log N + M)
 * - Memory: O(N) for FFT grid
 *
 * @see DirectOperator for O(N*M) reference implementation
 * @see NufftOperator for higher-accuracy FFT-accelerated implementation
 */

#include "reconstruction/reconstruction_operator.h"

#include <memory>

#include "k_space_slice.pb.h"
#include "reconstruction_config.pb.h"

namespace aurora::reconstruction {

/**
 * @brief Nearest-neighbor gridding forward/adjoint operator.
 *
 * Provides O(N log N) reconstruction via simple gridding and FFT.
 * Less accurate than NufftOperator but simpler and faster for cases
 * where K-samples are already close to grid points.
 */
class GriddingOperator : public ReconstructionOperator {
  public:
    /**
     * @brief Construct from K-vectors with kernel configuration.
     *
     * @param k_vectors K-space positions for each measurement (rad/m)
     * @param weights Per-sample weight for each measurement (1.0 = unweighted)
     * @param grid_size Target reconstruction grid size (Nx, Ny)
     * @param grid_spacing Grid spacing (dx, dy) in meters
     * @param config Gridding configuration (kernel type, width) - required
     *
     * @throws std::invalid_argument if sizes don't match or config invalid
     */
    GriddingOperator(std::vector<Eigen::Vector2d> k_vectors,
                     std::vector<double> weights,
                     const Eigen::Vector2i& grid_size,
                     const Eigen::Vector2d& grid_spacing,
                     const GriddingConfig& config);

    /**
     * @brief Construct from KSpaceSlice with kernel configuration.
     *
     * @param slice KSpaceSlice containing K-space samples
     * @param grid_size Target reconstruction grid size (Nx, Ny)
     * @param grid_spacing Grid spacing (dx, dy) in meters
     * @param config Gridding configuration (kernel type, width) - required
     */
    GriddingOperator(const aurora::shared::KSpaceSlice& slice,
                     const Eigen::Vector2i& grid_size,
                     const Eigen::Vector2d& grid_spacing,
                     const GriddingConfig& config);

    ~GriddingOperator() override;

    // Non-copyable
    GriddingOperator(const GriddingOperator&) = delete;
    GriddingOperator& operator=(const GriddingOperator&) = delete;

    // Movable
    GriddingOperator(GriddingOperator&&) noexcept;
    GriddingOperator& operator=(GriddingOperator&&) noexcept;

    // =========================================================================
    // ReconstructionOperator interface
    // =========================================================================

    /**
     * @brief Forward operator: scene -> K-space.
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

    /**
     * @brief Get gridding configuration.
     * @return Current gridding configuration
     */
    const GriddingConfig& config() const { return config_; }

  private:
    std::vector<Eigen::Vector2d> k_vectors_;
    std::vector<double> weights_;
    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    GriddingConfig config_;

    // Kernel parameters (computed from config)
    int kernel_width_;           ///< Kernel half-width (radius) in grid cells
    double gaussian_sigma_;      ///< Gaussian sigma in grid cells
    double kaiser_beta_;         ///< Kaiser-Bessel beta parameter

    // Pimpl for FFT plans
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void initialize();
    void setup_kernel_params();

    // Kernel evaluation
    double evaluate_kernel(double distance_sq) const;
    static double bessel_i0(double x);
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_GRIDDING_OPERATOR_H_
