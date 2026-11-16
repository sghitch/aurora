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

#ifndef AURORA_DETECTION_PHASE_GRADIENT_ESTIMATOR_H_
#define AURORA_DETECTION_PHASE_GRADIENT_ESTIMATOR_H_

/**
 * @file phase_gradient_estimator.h
 * @brief Base class for phase gradient-based angle estimation.
 *
 * Provides shared algorithms for fitting phase gradients to recover
 * target angles from K-space measurements. Azimuth and elevation follow
 * the same linear phase model along their respective K axes:
 *
 *   φᵢ = -kᵢ · x₀ + c    (azimuth: kx, horizontal position)
 *   φᵢ = -kᵢ · y₀ + c    (elevation: ky, vertical position)
 *
 * @see AngleEstimator for the two-dimensional estimator built on this base
 */

#include <cstdint>
#include <vector>

#include "phase_gradient_result.pb.h"

namespace aurora::detection {

/**
 * @brief Base class for phase gradient-based angle estimation.
 *
 * Provides core algorithms shared between azimuth and elevation estimation:
 * - Weighted least squares phase gradient fitting
 * - Phase unwrapping with prior estimate
 * - Uncertainty computation from Cramer-Rao bound
 * - Angular extent estimation from fit residuals
 *
 * Derived classes supply the search over the field of view that provides
 * the prior for unwrapping.
 */
class PhaseGradientEstimator {
  public:
    virtual ~PhaseGradientEstimator() = default;

  protected:
    /**
     * @brief Construct base estimator with optics configuration.
     *
     * @param focal_length_m Fourier lens focal length in meters
     * @param noise_floor_linear Noise floor for SNR computation (linear scale)
     */
    PhaseGradientEstimator(double focal_length_m, double noise_floor_linear);

    /**
     * @brief Fit phase gradient to recover angle.
     *
     * Uses weighted least squares to fit:
     *   φᵢ = -kᵢ · x + c
     *
     * where x is the target position and c is a common phase offset.
     * The angle is then recovered as θ = atan(x / f).
     *
     * @param phases Measured phases in radians
     * @param k_values K-values in rad/m (kx for azimuth, ky for elevation)
     * @param weights Magnitude weights for each sample
     * @return GradientFitResult with angle, uncertainty, and fit quality metrics
     */
    GradientFitResult fit_phase_gradient(
        const std::vector<double>& phases,
        const std::vector<double>& k_values,
        const std::vector<double>& weights) const;

    /**
     * @brief Unwrap phases using prior angle estimate.
     *
     * For small angles, where adjacent samples differ by less than π,
     * uses sequential local unwrapping. For larger angles the prior
     * estimate selects the 2π branch of each sample.
     *
     * @param wrapped_phases Wrapped phases in [-π, π]
     * @param k_values K-values corresponding to each phase
     * @param prior_angle_rad Prior angle estimate for disambiguation
     * @return Unwrapped phases
     */
    std::vector<double> unwrap_phases(
        const std::vector<double>& wrapped_phases,
        const std::vector<double>& k_values,
        double prior_angle_rad) const;

    /**
     * @brief Compute angle uncertainty from Cramer-Rao bound.
     *
     * The Cramer-Rao lower bound on angle estimation uncertainty is:
     *   σ_θ ≈ 1 / (SNR · k_span · f)
     *
     * where SNR is signal-to-noise ratio, k_span is the span of K-values,
     * and f is the focal length.
     *
     * @param k_span Span of K-values (max - min) in rad/m
     * @param snr_linear Signal-to-noise ratio (linear scale)
     * @param num_samples Number of measurement samples
     * @return Uncertainty in radians
     */
    double compute_uncertainty(
        double k_span,
        double snr_linear,
        uint32_t num_samples) const;

    /**
     * @brief Estimate angular extent from phase gradient fit residuals.
     *
     * Extended targets cause systematic phase residuals due to the
     * sinc-like magnitude roll-off in K-space. This method fits a
     * Gaussian envelope to the magnitude roll-off to estimate extent.
     *
     * @param residuals Per-sample fit residuals
     * @param k_values K-values for each sample
     * @param magnitudes Measurement magnitudes for weighting
     * @return Estimated angular extent in radians
     */
    double estimate_extent_from_residuals(
        const std::vector<double>& residuals,
        const std::vector<double>& k_values,
        const std::vector<double>& magnitudes) const;

    /**
     * @brief Wrap phase to [-π, π].
     *
     * @param phase Phase to wrap
     * @return Phase wrapped to [-π, π]
     */
    static double wrap_phase(double phase);

    /**
     * @brief Get the focal length.
     * @return Focal length in meters
     */
    double focal_length_m() const { return focal_length_m_; }

    /**
     * @brief Get the noise floor.
     * @return Noise floor (linear scale)
     */
    double noise_floor_linear() const { return noise_floor_linear_; }

  private:
    double focal_length_m_;
    double noise_floor_linear_;
};

}  // namespace aurora::detection

#endif  // AURORA_DETECTION_PHASE_GRADIENT_ESTIMATOR_H_
