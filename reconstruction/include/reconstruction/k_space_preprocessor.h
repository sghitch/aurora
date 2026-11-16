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

#ifndef AURORA_RECONSTRUCTION_K_SPACE_PREPROCESSOR_H_
#define AURORA_RECONSTRUCTION_K_SPACE_PREPROCESSOR_H_

/**
 * @file k_space_preprocessor.h
 * @brief K-space preprocessing for artifact reduction.
 *
 * Provides windowing and density compensation operations on K-space samples
 * before reconstruction. Separates preprocessing concerns from the
 * reconstruction algorithm itself.
 */

#include "k_space_slice.pb.h"
#include "reconstruction_config.pb.h"

namespace aurora::reconstruction {

/**
 * @brief Preprocesses K-space samples for improved reconstruction quality.
 *
 * Applies optional transformations to K-space samples in order:
 * 1. Density taper: Downweights high-|k| samples to reduce grid artifacts
 * 2. Apodization: Tapers samples near K-space edges to reduce Gibbs ringing
 * 3. Density compensation: Weights samples by inverse local density
 * 4. Normalization: Scales by sample count for consistent brightness
 *
 * ### Example Usage
 * @code
 * KSpacePreprocessor preprocessor(config.direct_adjoint_config());
 * auto processed_slice = preprocessor.apply(raw_slice, k_max_x, k_max_y);
 * // Use processed_slice for reconstruction
 * @endcode
 */
class KSpacePreprocessor {
  public:
    /**
     * @brief Construct with configuration.
     * @param config DirectAdjointConfig specifying preprocessing options
     */
    explicit KSpacePreprocessor(const DirectAdjointConfig& config);

    /**
     * @brief Construct with default settings (Hann apodization, no DCF).
     */
    KSpacePreprocessor();

    /**
     * @brief Apply preprocessing to a K-space slice.
     *
     * Modifies the sample values in-place based on configured options:
     * - Apodization window is applied based on |k|/k_max
     * - Density compensation weights (if enabled) are multiplied
     *
     * @param slice Input K-space slice (will be copied and modified)
     * @param k_max_x Maximum kx extent for normalization (rad/m)
     * @param k_max_y Maximum ky extent for normalization (rad/m)
     * @return Preprocessed K-space slice with modified sample values
     */
    aurora::shared::KSpaceSlice apply(
        const aurora::shared::KSpaceSlice& slice,
        double k_max_x,
        double k_max_y) const;

    /**
     * @brief Compute apodization weight for a K-space sample.
     *
     * @param kx Sample kx coordinate (rad/m)
     * @param ky Sample ky coordinate (rad/m)
     * @param k_max_x Maximum kx extent (rad/m)
     * @param k_max_y Maximum ky extent (rad/m)
     * @return Apodization weight in [0, 1]
     */
    double compute_apodization(double kx, double ky,
                               double k_max_x, double k_max_y) const;

    /**
     * @brief Compute density taper weight for a K-space sample.
     *
     * Combines isotropic (radial) and per-axis (kx, ky) taper weights.
     * Final weight = isotropic_weight * kx_weight * ky_weight.
     *
     * @param kx Sample kx coordinate (rad/m)
     * @param ky Sample ky coordinate (rad/m)
     * @return Combined taper weight in [0, 1]
     */
    double compute_density_taper(double kx, double ky) const;

  private:
    /**
     * @brief Compute taper weight for a single axis.
     *
     * @param k_abs Absolute k value along this axis (rad/m)
     * @param config Axis taper configuration
     * @return Taper weight in [retain_fraction, 1]
     */
    static double compute_axis_taper(double k_abs, const AxisTaperConfig& config);

    DirectAdjointConfig::ApodizationWindow apodization_;
    double apod_param_;
    bool enable_dcf_;
    uint32_t dcf_iterations_;
    bool normalize_output_;
    bool has_density_taper_;
    KSpaceDensityTaperConfig density_taper_;
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_K_SPACE_PREPROCESSOR_H_
