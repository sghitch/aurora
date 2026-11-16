#ifndef AURORA_RECONSTRUCTION_NUFFT_OPERATOR_H_
#define AURORA_RECONSTRUCTION_NUFFT_OPERATOR_H_

/**
 * @file nufft_operator.h
 * @brief FFT-accelerated forward/adjoint operator using NUFFT algorithms.
 *
 * Implements the ReconstructionOperator interface using Non-uniform Fast
 * Fourier Transform (NUFFT) techniques. This provides O(N log N + M * W^2)
 * complexity compared to DirectOperator's O(N * M), enabling efficient
 * reconstruction for large grids.
 *
 * ### Algorithm Overview
 *
 * **Adjoint (Type 1 NUFFT: Non-uniform to uniform):**
 * K-samples -> Gridding -> Oversampled Grid -> IFFT -> Crop -> Deapodize -> Image
 *
 * **Forward (Type 2 NUFFT: Uniform to non-uniform):**
 * Image -> Apodize -> Zero-pad -> FFT -> Interpolate at K-positions -> K-samples
 *
 * ### Physics-Aware Mode
 *
 * When the samples carry a K-space bandwidth (finite detector pixels), the
 * physical measurement window W(k) = H(k - k_d) of each sample is applied
 * inside the gridding/degridding weights. This ensures NufftOperator and
 * DirectOperator produce identical physics (PSF, resolution limits).
 *
 * ### Complexity
 *
 * - Forward: O(N log N + M * W^2) where N = grid pixels, M = K-samples, W = kernel width
 * - Adjoint: O(N log N + M * W^2)
 * - Memory: O(N * oversampling_factor^2) for oversampled grid, plus O(M * W^2) for kernel samples
 *
 * @see DirectOperator for O(N*M) reference implementation
 * @see ReconstructionOperator for interface documentation
 */

#include "reconstruction/reconstruction_operator.h"

#include <memory>

#include "forward/measurement_operator.h"
#include "bandwidth_config.pb.h"
#include "k_space_slice.pb.h"
#include "reconstruction_config.pb.h"

namespace aurora::forward {
class ForwardModel;
}  // namespace aurora::forward

namespace aurora::reconstruction {

/**
 * @brief FFT-accelerated forward/adjoint operator using NUFFT.
 *
 * Implements forward and adjoint operators via NUFFT algorithms using
 * gridding/degridding with convolution kernels. This provides significant
 * speedup over DirectOperator for large problem sizes.
 *
 * ### Example Usage
 * @code
 * // Construct with bandwidth support using MeasurementOperator physics
 * NufftConfig config;
 * config.set_oversampling_factor(2.0);
 * config.set_kernel_width(4);
 * config.set_kernel(KERNEL_KAISER_BESSEL);
 *
 * NufftOperator op(slice, forward_model, grid_size, grid_spacing, config);
 *
 * // Verify adjoint consistency
 * assert(op.check_adjoint_consistency(x, y, origin, spacing) < 1e-6);
 * @endcode
 */
class NufftOperator : public ReconstructionOperator {
  public:
    /**
     * @brief Construct with MeasurementOperator physics.
     *
     * @param slice KSpaceSlice containing K-space samples (may have bandwidth)
     * @param forward_model ForwardModel providing the per-pixel K-space bandwidth
     * @param grid_size Target reconstruction grid size (Nx, Ny)
     * @param grid_spacing Grid spacing (dx, dy) in meters
     * @param config NUFFT configuration (must include window_type for physics weighting)
     *
     * @throws std::invalid_argument if config invalid, required fields missing,
     *         or window_type unspecified when bandwidth info is available
     */
    NufftOperator(const aurora::shared::KSpaceSlice& slice,
                  const aurora::forward::ForwardModel& forward_model,
                  const Eigen::Vector2i& grid_size,
                  const Eigen::Vector2d& grid_spacing,
                  const NufftConfig& config);

    /**
     * @brief Destructor.
     */
    ~NufftOperator() override;

    // Non-copyable
    NufftOperator(const NufftOperator&) = delete;
    NufftOperator& operator=(const NufftOperator&) = delete;

    // Movable
    NufftOperator(NufftOperator&&) noexcept;
    NufftOperator& operator=(NufftOperator&&) noexcept;

    // =========================================================================
    // ReconstructionOperator interface
    // =========================================================================

    /**
     * @brief Forward operator: scene -> K-space (Type 2 NUFFT).
     *
     * Algorithm:
     * 1. Apply inverse phase shift for grid origin
     * 2. Apodize (pre-correct for interpolation kernel)
     * 3. Zero-pad to oversampled grid
     * 4. Apply 2D FFT
     * 5. Interpolate at non-uniform K-positions using convolution kernel
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
     * @brief Adjoint operator: K-space -> scene (Type 1 NUFFT).
     *
     * Algorithm:
     * 1. Grid: spread K-samples onto oversampled regular grid
     * 2. Apply 2D inverse FFT
     * 3. Crop to target grid size
     * 4. Deapodize: correct for gridding kernel
     * 5. Apply phase shift for grid origin
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
    // NUFFT-specific accessors
    // =========================================================================

    /**
     * @brief Get oversampled grid size.
     * @return Oversampled grid dimensions (Nx_os, Ny_os)
     */
    Eigen::Vector2i oversampled_size() const { return oversampled_size_; }

    /**
     * @brief Get NUFFT configuration.
     * @return NUFFT configuration protobuf
     */
    const NufftConfig& config() const { return config_; }

    /**
     * @brief Get the normalized grid size.
     * @return Target grid size
     */
    Eigen::Vector2i grid_size() const { return grid_size_; }

  private:
    std::vector<Eigen::Vector2d> k_vectors_;
    std::vector<double> weights_;
    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    NufftConfig config_;

    // Physics weighting: K-space window of each sample
    std::vector<forward::MeasurementOperator::WindowParams> window_params_;
    forward::BandwidthWindowType window_type_ = forward::BANDWIDTH_WINDOW_UNSPECIFIED;
    bool use_physics_weighting_ = false;

    // Precomputed data
    Eigen::Vector2i oversampled_size_;
    double effective_beta_;             ///< Computed KB beta (may differ from config if auto-computed)
    Eigen::MatrixXd deapodization_;     ///< Deapodization correction matrix
    std::vector<Eigen::Vector2d> k_normalized_;  ///< K in grid units

    // Kernel samples for each K-sample (precomputed weights and offsets)
    struct KernelSample {
        int grid_offset_x;   ///< Wrapped grid index in x (absolute position after wrapping)
        int grid_offset_y;   ///< Wrapped grid index in y (absolute position after wrapping)
        double weight;       ///< Kernel weight
    };
    std::vector<std::vector<KernelSample>> kernel_samples_;

    // Pimpl for FFT plans
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Initialization helpers
    static void validate_config(const NufftConfig& config);
    void initialize();
    void precompute();
    void compute_deapodization();
    void compute_kernel_samples();

    // Kernel evaluation
    double evaluate_kernel(double x) const;
    double kaiser_bessel(double x, double beta, int width) const;
    double gaussian_kernel(double x, double sigma) const;

    // Bessel function (I0)
    static double bessel_i0(double x);
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_NUFFT_OPERATOR_H_
