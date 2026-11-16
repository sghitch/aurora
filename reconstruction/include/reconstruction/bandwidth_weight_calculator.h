#ifndef AURORA_RECONSTRUCTION_BANDWIDTH_WEIGHT_CALCULATOR_H_
#define AURORA_RECONSTRUCTION_BANDWIDTH_WEIGHT_CALCULATOR_H_

/**
 * @file bandwidth_weight_calculator.h
 * @brief Composition-based utility for bandwidth-aware reconstruction.
 *
 * This class computes sparse weight structures for efficient forward/adjoint
 * operations. Operators can optionally use this for bandwidth-aware reconstruction
 * by pre-computing which pixels contribute to each K-space sample and vice versa.
 *
 * ### Physical Model
 *
 * Each K-space sample has a finite bandwidth (spatial extent in K-space) set
 * by the size of the detector pixel that acquired it. This bandwidth maps to
 * a spatial footprint in the Fourier plane via:
 *
 *   fourier_width = k_bandwidth * f * lambda / (2*pi)
 *
 * The pixel's K-space window W(k) = H(k - k_d), centered on the sample's
 * K-position k_d, defines a sensitivity region. Only reconstruction pixels
 * within this region contribute meaningfully.
 *
 * ### Usage
 *
 * ```cpp
 * BandwidthWeightCalculator calc(slice, forward_model);
 * if (calc.has_bandwidth()) {
 *     auto weights = calc.compute_weights(grid_size, grid_spacing, grid_origin,
 *                                          BANDWIDTH_WINDOW_SINC);
 *     // Use weights.sample_to_pixels for forward operator
 *     // Use weights.pixel_to_samples for adjoint operator
 * }
 * ```
 */

#include <vector>
#include <Eigen/Core>
#include "bandwidth_config.pb.h"
#include "k_space_slice.pb.h"
#include "forward/measurement_operator.h"

namespace aurora::forward {
class ForwardModel;
}  // namespace aurora::forward

namespace aurora::reconstruction {

/**
 * @brief Weight contribution from a K-space sample to a reconstruction pixel.
 */
struct PixelWeight {
    int ix;       ///< Pixel x index in reconstruction grid
    int iy;       ///< Pixel y index in reconstruction grid
    double weight;  ///< Weight value (sensitivity contribution)
};

/**
 * @brief Weight contribution from a reconstruction pixel to a K-space sample.
 */
struct SampleContribution {
    size_t sample_idx;  ///< Index into KSpaceSlice samples
    double weight;      ///< Weight value (sensitivity contribution)
};

/**
 * @brief Sparse bidirectional mapping between K-space samples and pixels.
 *
 * Provides both forward (sample -> pixels) and adjoint (pixel -> samples)
 * mappings for efficient sparse matrix operations.
 */
struct SparseWeights {
    /// For each K-space sample, list of contributing pixels and their weights
    std::vector<std::vector<PixelWeight>> sample_to_pixels;

    /// For each pixel (row-major: y * Nx + x), list of contributing samples
    std::vector<std::vector<SampleContribution>> pixel_to_samples;

    /**
     * @brief Compute average number of pixels per sample.
     * @return Average sparsity (entries per row in sample_to_pixels)
     */
    double average_sparsity() const;

    /**
     * @brief Check if weights structure has no non-zero entries.
     * @return true if no non-zero weight mappings are stored
     */
    bool empty() const {
        if (sample_to_pixels.empty() && pixel_to_samples.empty()) {
            return true;
        }
        for (const auto& per_sample : sample_to_pixels) {
            if (!per_sample.empty()) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Calculator for bandwidth-aware spatial weights.
 *
 * Builds one MeasurementOperator window per KSpaceSlice sample, centered on
 * the sample's K-position with the K-space bandwidth of the detector pixel
 * that acquired it (from ForwardModel), and computes sparse weight
 * structures for reconstruction using the same MeasurementOperator physics
 * as simulation.
 */
class BandwidthWeightCalculator {
  public:
    /**
     * @brief Construct calculator using MeasurementOperator physics.
     *
     * Each sample's window is centered on its (kx, ky); the window
     * bandwidths come from the pixel size of the sample's detector pixel.
     *
     * @param slice K-space slice with samples (must have detector_index fields)
     * @param forward_model ForwardModel providing the per-pixel K-space bandwidth
     *
     * @throws std::out_of_range if a sample's detector_index is not in the geometry
     */
    BandwidthWeightCalculator(const aurora::shared::KSpaceSlice& slice,
                               const aurora::forward::ForwardModel& forward_model);

    /**
     * @brief Check if bandwidth information is available.
     * @return true if at least one sample has valid bandwidth
     */
    bool has_bandwidth() const { return has_bandwidth_; }

    /**
     * @brief Get number of K-space samples.
     * @return Number of samples in the slice
     */
    size_t num_samples() const { return kx_bandwidths_.size(); }

    /**
     * @brief Get K-space bandwidth in x for a sample.
     * @param sample_idx Sample index
     * @return Bandwidth in rad/m (0 if not available)
     */
    double kx_bandwidth(size_t sample_idx) const { return kx_bandwidths_[sample_idx]; }

    /**
     * @brief Get K-space bandwidth in y for a sample.
     * @param sample_idx Sample index
     * @return Bandwidth in rad/m (0 if not available)
     */
    double ky_bandwidth(size_t sample_idx) const { return ky_bandwidths_[sample_idx]; }

    /**
     * @brief Compute average bandwidth across all valid samples.
     * @return Average (kx_bandwidth, ky_bandwidth) or (0, 0) if no bandwidth
     */
    Eigen::Vector2d average_bandwidth() const;

    /**
     * @brief Compute resolution limit from average bandwidth.
     *
     * Resolution limit = 2*pi / bandwidth
     *
     * @return (dx, dy) minimum resolvable separation in meters
     */
    Eigen::Vector2d resolution_limit() const;

    /**
     * @brief Check if grid spacing can resolve bandwidth-limited features.
     *
     * Grid spacing must be <= resolution_limit * tolerance_factor to
     * adequately sample features at the bandwidth-limited resolution.
     *
     * @param grid_spacing Grid spacing (dx, dy) in meters
     * @param tolerance_factor Safety factor (default 1.1, allows 10% coarser)
     * @return true if spacing is fine enough, false if too coarse (undersampled)
     */
    bool is_grid_valid(const Eigen::Vector2d& grid_spacing,
                       double tolerance_factor = 1.1) const;

    /**
     * @brief Compute sparse weight structure using MeasurementOperator.
     *
     * Uses MeasurementOperator::weight() to compute weights, ensuring
     * reconstruction uses the same physics as simulation.
     *
     * @param grid_size Number of pixels (Nx, Ny)
     * @param grid_spacing Pixel spacing (dx, dy) in meters
     * @param grid_origin Lower-left corner position in meters
     * @param window_type Window function type (must be specified explicitly)
     * @param weight_threshold Minimum weight to include (default 1e-6)
     * @return Sparse bidirectional weight mapping
     */
    SparseWeights compute_weights(const Eigen::Vector2i& grid_size,
                                   const Eigen::Vector2d& grid_spacing,
                                   const Eigen::Vector2d& grid_origin,
                                   aurora::forward::BandwidthWindowType window_type,
                                   double weight_threshold = 1e-6) const;

    /**
     * @brief Get detector index for a K-space sample.
     * @param sample_idx Sample index
     * @return Detector pixel index
     */
    uint32_t detector_index(size_t sample_idx) const { return detector_indices_[sample_idx]; }

    /**
     * @brief Get MeasurementOperator WindowParams for a sample.
     *
     * Returns the physics parameters needed for quadrature-based
     * forward/adjoint operators. These parameters define the K-space
     * window function W_m(k) = H(k - k_m), centered on the sample's
     * K-position with the bandwidth of its detector pixel.
     *
     * @param sample_idx Sample index
     * @return WindowParams with k_center and bandwidths
     */
    aurora::forward::MeasurementOperator::WindowParams
    get_window_params(size_t sample_idx) const {
        return aurora::forward::MeasurementOperator::WindowParams{
            .k_center = k_centers_[sample_idx],
            .bandwidth_x = kx_bandwidths_[sample_idx],
            .bandwidth_y = ky_bandwidths_[sample_idx]
        };
    }

  private:
    bool has_bandwidth_ = false;
    double wavelength_m_ = 0.0;
    double focal_length_m_ = 0.0;

    // Per-sample window: K-position (window center) and K-space bandwidth
    std::vector<Eigen::Vector2d> k_centers_;
    std::vector<double> kx_bandwidths_;
    std::vector<double> ky_bandwidths_;
    std::vector<uint32_t> detector_indices_;
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_BANDWIDTH_WEIGHT_CALCULATOR_H_
