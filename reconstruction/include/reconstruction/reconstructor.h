#ifndef AURORA_RECONSTRUCTION_RECONSTRUCTOR_H_
#define AURORA_RECONSTRUCTION_RECONSTRUCTOR_H_

/**
 * @file reconstructor.h
 * @brief Scene reconstruction facade over ForwardModel and the operators.
 *
 * The Reconstructor is a thin facade that orchestrates the reconstruction
 * pipeline for one burst: it extracts a K-space slice per range bin and
 * wavelength via ForwardModel and reconstructs an image from each slice
 * with the configured ReconstructionOperator implementation.
 *
 * ### Architecture
 *
 * ```
 * MeasurementTensor (one burst)
 *       |
 *       v
 * +---------------------+
 * |    ForwardModel     |  <-- extract_kspace_slice(range_bin, wavelength)
 * +---------------------+
 *       |
 *       v
 *    KSpaceSlice (protobuf)
 *       |
 *       v
 * +---------------------+
 * | Reconstruction      |  <-- DirectOperator, GriddingOperator, NufftOperator,
 * | Operator            |      FastOnlineOperator or PcgOperator
 * +---------------------+
 *       |
 *       v
 *   Reconstructed Image
 * ```
 *
 * ### Algorithms
 *
 * - **DIRECT_ADJOINT**: Direct summation via DirectOperator.adjoint()
 * - **GRIDDING_IFFT**: Gridding + IFFT via GriddingOperator
 * - **NUFFT_ADJOINT**: FFT-accelerated NUFFT via NufftOperator
 * - **FAST_ONLINE**: Single-pass gridding with density normalization (FORO)
 * - **PCG_REGULARIZED**: Iterative regularized least squares via PcgOperator
 *
 * ### Usage
 *
 * @code
 * // Create ForwardModel from the receiver geometry
 * ForwardModel forward_model(detector, optics);
 *
 * // Configure reconstruction
 * ReconstructionConfig config;
 * config.set_algorithm(ALGORITHM_DIRECT_ADJOINT);
 * config.set_grid_size_x(64);
 * config.set_grid_size_y(64);
 * config.set_pixel_size_m(0.001);
 * config.mutable_bandwidth_config()->set_window_type(BANDWIDTH_WINDOW_SINC);
 * config.mutable_bandwidth_config()->set_quadrature_order(11);
 * config.mutable_direct_adjoint_config()->set_apodization(
 *     DirectAdjointConfig::APOD_NONE);
 *
 * // Create reconstructor
 * Reconstructor reconstructor(config, forward_model);
 *
 * // Reconstruct the scene from one burst
 * auto scene = reconstructor.reconstruct(measurement_tensor);
 * @endcode
 *
 * @see ForwardModel for the K-sample service and slice extraction
 * @see ReconstructionOperator for the operator interface
 */

#include <chrono>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "forward/forward_model.h"
#include "k_space_slice.pb.h"
#include "measurement_tensor.pb.h"
#include "reconstructed_scene.pb.h"
#include "reconstruction/direct_operator.h"
#include "reconstruction/fast_online_operator.h"
#include "reconstruction/gridding_operator.h"
#include "reconstruction/nufft_operator.h"
#include "reconstruction/pcg_operator.h"
#include "reconstruction_config.pb.h"

namespace aurora::reconstruction {

/**
 * @brief Scene reconstruction facade.
 *
 * Orchestrates reconstruction from one MeasurementTensor to a
 * ReconstructedScene, using ForwardModel for K-space slice extraction and
 * the configured operator for image reconstruction.
 */
class Reconstructor {
  public:
    /**
     * @brief Construct reconstructor with ForwardModel reference.
     *
     * @param config Reconstruction configuration (algorithm, grid, etc.)
     * @param forward_model ForwardModel instance (must outlive Reconstructor)
     *
     * @throws ReconstructionError (CONFIGURATION_ERROR) if the algorithm is
     *         not supported or its required configuration is missing
     */
    Reconstructor(const ReconstructionConfig& config,
                  const aurora::forward::ForwardModel& forward_model);

    /**
     * @brief Reconstruct the scene from one burst.
     *
     * Processes every range bin and wavelength in the measurement tensor.
     *
     * @param measurements MeasurementTensor of one burst
     * @return ReconstructedScene with per-wavelength, per-range-bin images
     *
     * @throws ReconstructionError (INVALID_INPUT) if the tensor has no
     *         wavelength data or no range bins
     */
    ReconstructedScene reconstruct(
        const aurora::shared::MeasurementTensor& measurements);

    /**
     * @brief Reconstruct a single range bin.
     *
     * @param measurements MeasurementTensor of one burst
     * @param range_bin Range bin index
     * @param wavelength_m Target wavelength in meters
     * @return Complex image for the specified range bin (zero if the tensor
     *         holds no samples for this range bin and wavelength)
     */
    Eigen::MatrixXcd reconstruct_range_bin(
        const aurora::shared::MeasurementTensor& measurements,
        uint32_t range_bin,
        double wavelength_m);

    /**
     * @brief Get reconstruction grid origin in meters.
     * @return Grid origin (x0, y0)
     */
    Eigen::Vector2d grid_origin_m() const { return grid_origin_; }

    /**
     * @brief Get reconstruction grid spacing in meters.
     * @return Grid spacing (dx, dy)
     */
    Eigen::Vector2d grid_spacing_m() const { return grid_spacing_; }

    /**
     * @brief Get reconstruction grid size.
     * @return Grid size (nx, ny)
     */
    Eigen::Vector2i grid_size() const { return grid_size_; }

    /**
     * @brief Get current algorithm.
     * @return Configured reconstruction algorithm
     */
    ReconstructionAlgorithm algorithm() const { return algorithm_; }

  private:
    ReconstructionConfig config_;
    const aurora::forward::ForwardModel& forward_model_;

    // Reconstruction parameters
    ReconstructionAlgorithm algorithm_;
    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    Eigen::Vector2d grid_origin_;

    /**
     * @brief Validate the configuration of the selected algorithm.
     *
     * @throws ReconstructionError (CONFIGURATION_ERROR) for an unsupported
     *         algorithm or missing algorithm-specific configuration
     */
    void validate_algorithm_config() const;

    /**
     * @brief Reconstruct one K-space slice with the configured algorithm.
     *
     * @param slice K-space slice with samples (non-empty)
     * @return Reconstructed image
     */
    Eigen::MatrixXcd reconstruct_slice(
        const aurora::shared::KSpaceSlice& slice) const;

    /**
     * @brief Reconstruct using DirectOperator adjoint.
     *
     * O(N*M) direct summation reconstruction.
     *
     * @param slice K-space slice with samples
     * @return Reconstructed image
     */
    Eigen::MatrixXcd adjoint_reconstruct(
        const aurora::shared::KSpaceSlice& slice) const;

    /**
     * @brief Reconstruct using GriddingOperator.
     *
     * Fast approximate reconstruction by gridding K-samples onto a
     * uniform grid and applying an inverse FFT.
     *
     * @param slice K-space slice with samples
     * @return Reconstructed image
     */
    Eigen::MatrixXcd gridding_reconstruct(
        const aurora::shared::KSpaceSlice& slice) const;

    /**
     * @brief Reconstruct using NufftOperator.
     *
     * FFT-accelerated reconstruction with Kaiser-Bessel interpolation.
     *
     * @param slice K-space slice with samples
     * @return Reconstructed image
     */
    Eigen::MatrixXcd nufft_reconstruct(
        const aurora::shared::KSpaceSlice& slice) const;

    /**
     * @brief Reconstruct using FastOnlineOperator.
     *
     * Single-pass gridding with density accumulation and K-space
     * normalization, preceded by the K-sample axis window.
     *
     * @param slice K-space slice with samples
     * @return Reconstructed image
     */
    Eigen::MatrixXcd fast_online_reconstruct(
        const aurora::shared::KSpaceSlice& slice) const;

    /**
     * @brief Reconstruct using PcgOperator.
     *
     * Preconditioned Conjugate Gradient reconstruction with regularization.
     * Solves (A†WA + λL†L)x = A†Wy iteratively.
     *
     * @param slice K-space slice with samples
     * @return Reconstructed image
     */
    Eigen::MatrixXcd pcg_reconstruct(
        const aurora::shared::KSpaceSlice& slice) const;

    /**
     * @brief Build ReconstructedScene protobuf from reconstruction results.
     *
     * @param wavelength_images Per-wavelength, per-range-bin images
     * @param wavelengths Wavelength values in meters
     * @param range_axis Range axis from the input tensor
     * @param start_time Processing start time
     * @return Populated ReconstructedScene protobuf
     */
    ReconstructedScene build_scene(
        const std::vector<std::vector<Eigen::MatrixXcd>>& wavelength_images,
        const std::vector<double>& wavelengths,
        const aurora::shared::Axis& range_axis,
        std::chrono::steady_clock::time_point start_time) const;

    /**
     * @brief Compute image quality metrics.
     *
     * @param image Reconstructed image
     * @param peak_value Output: maximum magnitude
     * @param total_energy Output: sum of squared magnitudes
     * @param sparsity Output: fraction of significant pixels
     */
    void compute_metrics(
        const Eigen::MatrixXcd& image,
        double& peak_value,
        double& total_energy,
        double& sparsity) const;

    /**
     * @brief Extract unique wavelengths from the measurement tensor.
     *
     * @param measurements Input measurement tensor
     * @return Vector of unique wavelengths in meters
     */
    std::vector<double> extract_wavelengths(
        const aurora::shared::MeasurementTensor& measurements) const;

    /**
     * @brief Get number of range bins from the measurement tensor.
     *
     * @param measurements Input measurement tensor
     * @return Number of range bins
     */
    uint32_t get_num_range_bins(
        const aurora::shared::MeasurementTensor& measurements) const;
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_RECONSTRUCTOR_H_
