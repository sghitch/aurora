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

#ifndef AURORA_DETECTION_ANGLE_ESTIMATOR_H_
#define AURORA_DETECTION_ANGLE_ESTIMATOR_H_

/**
 * @file angle_estimator.h
 * @brief Two-dimensional phase-gradient angle estimation from K-space samples.
 *
 * For a point target at angular position (θx, θy) the K-space phase of each
 * sample is
 * @f[
 *   \phi_i = -\mathbf{k}_i \cdot \mathbf{r} + \phi_0,
 *   \quad \mathbf{r} = f \,(\theta_x, \theta_y)
 * @f]
 * where f is the Fourier lens focal length. Given the complex samples and
 * their K-vectors at one range bin, the AngleEstimator recovers r by
 *
 * 1. a coarse matched-filter (beamforming) search over the field of view,
 *    which is robust to 2π phase wrapping (when the K-samples are uniformly
 *    spaced the response is periodic in angle; the alias nearest boresight
 *    is chosen, so the unambiguous field of view is set by the K-spacing),
 *    followed by
 * 2. a weighted least-squares refinement of the linear phase model with the
 *    phases unwrapped against the coarse solution.
 *
 * Elevation is only observable when the detector pixels span ky (a
 * two-dimensional array); a linear detector array yields azimuth only.
 */

#include <complex>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "detection/phase_gradient_estimator.h"
#include "phase_gradient_result.pb.h"

namespace aurora::detection {

/**
 * @brief Estimates target angles from the phase gradient of K-space samples.
 */
class AngleEstimator : public PhaseGradientEstimator {
  public:
    /// One K-space measurement at the range bin under test.
    struct Sample {
        Eigen::Vector2d k;            ///< Scene K-vector (rad/m)
        std::complex<double> value;   ///< Complex measurement
    };

    /**
     * @brief Construct the estimator.
     *
     * @param focal_length_m Fourier lens focal length in meters
     * @param noise_floor_linear Noise floor (linear power) used for SNR
     * @param fov_x_rad Full horizontal field of view searched (radians)
     * @param fov_y_rad Full vertical field of view searched (radians)
     * @param max_coarse_bins Upper bound on coarse search bins (azimuth-only search).
     *        The actual bin count is derived from the K-span so the matched
     *        filter peak is always sampled; this caps the cost. A joint
     *        azimuth/elevation search (two-dimensional array) uses a fixed
     *        per-axis cap.
     * @throws std::invalid_argument if focal length, FOV, or bins are invalid
     */
    AngleEstimator(double focal_length_m,
                   double noise_floor_linear,
                   double fov_x_rad,
                   double fov_y_rad,
                   uint32_t max_coarse_bins);

    /**
     * @brief Estimate azimuth/elevation from samples at one range bin.
     *
     * @param samples K-space samples (at least two with distinct kx)
     * @return AngleFitResult; valid() is false if the fit is not possible
     */
    AngleFitResult estimate(const std::vector<Sample>& samples) const;

    /// Minimum ky span (rad/m) for elevation to be considered observable.
    static constexpr double kMinKySpanRadM = 1.0;

  private:
    double fov_x_rad_;
    double fov_y_rad_;
    uint32_t max_coarse_bins_;

    /// Two-level matched-filter search; returns Fourier-plane position (x, y).
    Eigen::Vector2d coarse_search(const std::vector<Sample>& samples,
                                  bool search_elevation) const;

    /// Evaluate the matched filter on a grid of angles around center_rad.
    /// With prefer_boresight, bins within the alias tolerance of the best
    /// score are considered ties and the one nearest boresight is returned.
    Eigen::Vector2d search_grid(const std::vector<Sample>& samples,
                                const Eigen::Vector2d& center_rad,
                                const Eigen::Vector2d& half_extent_rad,
                                uint32_t nx,
                                uint32_t ny,
                                bool prefer_boresight) const;

    /// |Σ y_i exp(+i k_i · r)|² for Fourier-plane position r.
    static double matched_filter_score(const std::vector<Sample>& samples,
                                       const Eigen::Vector2d& r);

    /// Weighted least-squares refinement around a prior position.
    AngleFitResult refine(const std::vector<Sample>& samples,
                          const Eigen::Vector2d& prior_xy,
                          bool fit_elevation) const;
};

}  // namespace aurora::detection

#endif  // AURORA_DETECTION_ANGLE_ESTIMATOR_H_
