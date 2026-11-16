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

#ifndef AURORA_RECONSTRUCTION_FAST_ONLINE_OPERATOR_H_
#define AURORA_RECONSTRUCTION_FAST_ONLINE_OPERATOR_H_

/**
 * @file fast_online_operator.h
 * @brief Fast Online Reconstruction Operator (FORO).
 *
 * Provides a streamlined reconstruction pipeline optimized for real-time
 * operation with simultaneous gridding and density accumulation.
 *
 * ### Algorithm Overview
 *
 * 1. **K-sample axis windowing**: Taper measurements across the array's
 *    K-sample axis to suppress ringing from the gaps between K-samples
 * 2. **Gridding + Density accumulation**: Single-pass scatter to K-grid
 *    while simultaneously building density map D[k]
 * 3. **K-space normalization**: Y_tilde[k] = Y[k] / (D[k] + epsilon)
 *    approximating (A^H * W * A)^-1
 * 4. **IFFT to image**: Standard 2D inverse FFT
 * 5. **Optional post-processing**: Laplacian sharpen or diffusion
 * 6. **Optional refinement**: 1-2 fixed-point iterations
 * 7. **Optional gap interpolation**: Phase-aware synthesis of K-space values
 *    midway between adjacent K-samples
 *
 * ### Complexity
 *
 * - Single-pass reconstruction: O(M * W^2 + N log N)
 * - With refinement: O(iterations * (M * W^2 + N log N))
 *
 * Where M = K-samples, W = kernel width, N = grid pixels.
 *
 * @see GriddingOperator for simpler gridding without density normalization
 * @see NufftOperator for higher accuracy NUFFT implementation
 */

#include "reconstruction/reconstruction_operator.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "k_space_slice.pb.h"
#include "reconstruction_config.pb.h"

namespace aurora::reconstruction {

/**
 * @brief Fast Online Reconstruction Operator (FORO).
 *
 * Provides a streamlined reconstruction pipeline optimized for real-time
 * operation. The key innovation is simultaneous gridding and density
 * accumulation in a single pass, with K-space normalization providing
 * online PSF correction.
 *
 * ### Usage
 * @code
 * // Create operator from K-space slice
 * FastOnlineConfig config;
 * config.set_kernel(GRIDDING_KERNEL_KAISER_BESSEL);
 * config.set_kernel_width(3);
 * config.set_anti_gap_window(ANTI_GAP_WINDOW_HANN);
 * config.set_density_normalization(DENSITY_NORM_KERNEL);
 * config.set_density_epsilon(1e-10);
 * config.set_post_processing(POST_PROC_NONE);
 * config.set_enable_gap_interpolation(false);
 * config.set_enable_refinement(false);
 *
 * FastOnlineOperator op(slice, grid_size, grid_spacing, config);
 *
 * // Reconstruct
 * auto image = op.reconstruct(samples, grid_origin, grid_spacing, grid_size);
 * @endcode
 */
class FastOnlineOperator : public ReconstructionOperator {
  public:
    /**
     * @brief Construct from K-vectors with detector indices.
     *
     * @param k_vectors K-space positions for each measurement (rad/m)
     * @param weights Per-sample weight for each measurement (1.0 = unweighted)
     * @param detector_indices Detector pixel index for each measurement
     * @param grid_size Target reconstruction grid size (Nx, Ny)
     * @param grid_spacing Grid spacing (dx, dy) in meters
     * @param config Fast online configuration (required)
     *
     * @throws std::invalid_argument if sizes don't match or config invalid
     */
    FastOnlineOperator(
        std::vector<Eigen::Vector2d> k_vectors,
        std::vector<double> weights,
        std::vector<uint32_t> detector_indices,
        const Eigen::Vector2i& grid_size,
        const Eigen::Vector2d& grid_spacing,
        const FastOnlineConfig& config);

    /**
     * @brief Construct from KSpaceSlice.
     *
     * Extracts K-vectors, sample weights, and detector indices from the slice.
     *
     * @param slice KSpaceSlice containing K-space samples
     * @param grid_size Target reconstruction grid size (Nx, Ny)
     * @param grid_spacing Grid spacing (dx, dy) in meters
     * @param config Fast online configuration (required)
     */
    FastOnlineOperator(
        const aurora::shared::KSpaceSlice& slice,
        const Eigen::Vector2i& grid_size,
        const Eigen::Vector2d& grid_spacing,
        const FastOnlineConfig& config);

    ~FastOnlineOperator() override;

    // Non-copyable
    FastOnlineOperator(const FastOnlineOperator&) = delete;
    FastOnlineOperator& operator=(const FastOnlineOperator&) = delete;

    // Movable
    FastOnlineOperator(FastOnlineOperator&&) noexcept;
    FastOnlineOperator& operator=(FastOnlineOperator&&) noexcept;

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
     * Uses standard gridding + IFFT without FORO-specific features.
     * For full FORO pipeline, use reconstruct() instead.
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
    // FORO-specific interface
    // =========================================================================

    /**
     * @brief Full FORO reconstruction pipeline.
     *
     * Performs the complete reconstruction with all configured features:
     * K-sample axis windowing, density normalization, post-processing, and
     * optional refinement.
     *
     * @param k_samples Complex K-space measurements
     * @param grid_origin Scene grid origin (x, y) in meters
     * @param grid_spacing Scene grid spacing (dx, dy) in meters
     * @param grid_size Scene grid dimensions (Nx, Ny)
     * @return Complex scene estimate (Ny x Nx)
     */
    Eigen::MatrixXcd reconstruct(
        const Eigen::VectorXcd& k_samples,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing,
        const Eigen::Vector2i& grid_size) const;

    /**
     * @brief Get configuration.
     * @return Current fast online configuration
     */
    const FastOnlineConfig& config() const { return config_; }

    /**
     * @brief Get detector indices for each sample.
     * @return Reference to detector index storage
     */
    const std::vector<uint32_t>& detector_indices() const {
        return detector_indices_;
    }

    /**
     * @brief Get the precomputed K-sample axis window.
     * @return Window weight per detector pixel index
     */
    const std::vector<double>& anti_gap_weights() const {
        return anti_gap_weights_;
    }

  private:
    std::vector<Eigen::Vector2d> k_vectors_;
    std::vector<double> weights_;
    std::vector<uint32_t> detector_indices_;
    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    FastOnlineConfig config_;

    // Precomputed K-sample axis window, one weight per detector pixel index
    std::vector<double> anti_gap_weights_;

    // Number of detector pixel indices (max index + 1)
    uint32_t num_detectors_;

    // Kernel parameters (computed from config)
    int kernel_width_;           ///< Kernel half-width (radius) in grid cells
    double gaussian_sigma_;      ///< Gaussian sigma in grid cells
    double kaiser_beta_;         ///< Kaiser-Bessel beta parameter

    // Pimpl for FFT plans
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // =========================================================================
    // Initialization
    // =========================================================================

    void initialize();
    void validate_config() const;
    void setup_kernel_params();
    void compute_anti_gap_weights();

    // =========================================================================
    // Core operations
    // =========================================================================

    /**
     * @brief Evaluate gridding kernel at given squared distance.
     *
     * @param distance_sq Squared distance from sample to grid cell (grid units)
     * @return Kernel weight
     */
    double evaluate_kernel(double distance_sq) const;

    /**
     * @brief Modified Bessel function of first kind, I0(x).
     * @param x Argument
     * @return I0(x)
     */
    static double bessel_i0(double x);

    /**
     * @brief Grid samples AND accumulate density map in single pass.
     *
     * This is the core FORO innovation: simultaneous gridding and density
     * accumulation eliminates a second pass over the data.
     *
     * @param k_samples Complex K-space measurements (after the K-sample axis window)
     * @param k_grid Output: accumulated complex K-grid
     * @param density_map Output: accumulated density map D[k]
     * @param grid_origin Grid origin for coordinate mapping
     */
    void grid_and_accumulate(
        const Eigen::VectorXcd& k_samples,
        Eigen::MatrixXcd& k_grid,
        Eigen::MatrixXd& density_map,
        const Eigen::Vector2d& grid_origin) const;

    /**
     * @brief Apply the K-sample axis window to the samples.
     *
     * Multiplies each sample by the precomputed weight for its detector index.
     *
     * @param k_samples Input K-space samples
     * @return Windowed K-space samples
     */
    Eigen::VectorXcd apply_anti_gap_window(
        const Eigen::VectorXcd& k_samples) const;

    /**
     * @brief Normalize K-grid by density map.
     *
     * Applies: Y_tilde[k] = Y[k] / (D[k] + epsilon)
     *
     * @param k_grid K-grid to normalize (modified in place)
     * @param density_map Density map D[k]
     */
    void normalize_by_density(
        Eigen::MatrixXcd& k_grid,
        const Eigen::MatrixXd& density_map) const;

    // =========================================================================
    // Post-processing
    // =========================================================================

    /**
     * @brief Apply configured post-processing to image.
     *
     * @param image Image to process (modified in place)
     */
    void apply_post_processing(Eigen::MatrixXcd& image) const;

    /**
     * @brief Apply Laplacian unsharp filtering for sharpening.
     *
     * @param image Image to sharpen (modified in place)
     * @param strength Sharpening strength [0, 1]
     */
    void apply_laplacian_sharpen(Eigen::MatrixXcd& image, double strength) const;

    /**
     * @brief Apply diffusion smoothing.
     *
     * @param image Image to smooth (modified in place)
     * @param coefficient Diffusion coefficient
     */
    void apply_diffusion(Eigen::MatrixXcd& image, double coefficient) const;

    // =========================================================================
    // Optional features
    // =========================================================================

    /**
     * @brief Extended K-space data with interpolated samples.
     */
    struct ExtendedKSpace {
        std::vector<Eigen::Vector2d> k_vectors;
        Eigen::VectorXcd samples;
        std::vector<double> weights;
    };

    /**
     * @brief Synthesize K-space values midway between adjacent K-samples using phase-aware interpolation.
     *
     * @param k_samples Input K-space samples
     * @return Extended K-space data including the interpolated values
     */
    ExtendedKSpace interpolate_gaps(
        const Eigen::VectorXcd& k_samples) const;

    /**
     * @brief Grid extended samples (with explicit k_vectors).
     *
     * @param extended Extended K-space data from interpolate_gaps
     * @param k_grid Output: accumulated complex K-grid
     * @param density_map Output: accumulated density map D[k]
     */
    void grid_extended(
        const ExtendedKSpace& extended,
        Eigen::MatrixXcd& k_grid,
        Eigen::MatrixXd& density_map) const;

    /**
     * @brief Perform one refinement iteration.
     *
     * Implements: x_{t+1} = x_t + alpha * (A^H * W * (y - A*x_t) - lambda * L^H*L*x_t)
     *
     * @param x0 Current image estimate
     * @param k_samples Original K-space measurements
     * @param grid_origin Grid origin
     * @param grid_spacing Grid spacing
     * @return Updated image estimate
     */
    Eigen::MatrixXcd refine(
        const Eigen::MatrixXcd& x0,
        const Eigen::VectorXcd& k_samples,
        const Eigen::Vector2d& grid_origin,
        const Eigen::Vector2d& grid_spacing) const;

    /**
     * @brief Compute Laplacian of image.
     *
     * @param image Input image
     * @return Laplacian (L^H * L applied to image)
     */
    Eigen::MatrixXcd compute_laplacian(const Eigen::MatrixXcd& image) const;
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_FAST_ONLINE_OPERATOR_H_
