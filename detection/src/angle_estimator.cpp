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

#include "detection/angle_estimator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

namespace aurora::detection {

namespace {

constexpr double kTwoPi = 2.0 * M_PI;

/// Relative threshold on normal-equation determinant for a usable fit.
constexpr double kMinRelativeDeterminant = 1e-12;

/// Level-1 grid samples per matched-filter main lobe (whole field of view).
/// Two samples per lobe keep the sampled peak above ~80% of the true peak.
constexpr double kLevel1SamplesPerLobe = 2.0;

/// Level-2 grid samples per main lobe (local refinement around the pick).
constexpr double kLevel2SamplesPerLobe = 16.0;

/// Bins scoring within this fraction of the best score count as aliases of it.
constexpr double kAliasScoreTolerance = 0.3;

/// Per-axis cap on coarse bins when a two-dimensional detector array (K-samples
/// spanning both kx and ky) allows azimuth and elevation to be searched jointly.
constexpr uint32_t kMaxCoarseBins2d = 1024;

}  // namespace

AngleEstimator::AngleEstimator(double focal_length_m,
                               double noise_floor_linear,
                               double fov_x_rad,
                               double fov_y_rad,
                               uint32_t max_coarse_bins)
    : PhaseGradientEstimator(focal_length_m, noise_floor_linear),
      fov_x_rad_(fov_x_rad),
      fov_y_rad_(fov_y_rad),
      max_coarse_bins_(max_coarse_bins) {
    if (focal_length_m <= 0.0) {
        throw std::invalid_argument("AngleEstimator: focal_length_m must be positive");
    }
    if (max_coarse_bins < 2) {
        throw std::invalid_argument("AngleEstimator: max_coarse_bins must be >= 2");
    }
    if (fov_x_rad <= 0.0) {
        throw std::invalid_argument("AngleEstimator: fov_x_rad must be positive");
    }
}

AngleFitResult AngleEstimator::estimate(const std::vector<Sample>& samples) const {
    AngleFitResult result;
    result.set_valid(false);
    result.set_num_samples(static_cast<uint32_t>(samples.size()));
    result.set_azimuth_uncertainty_rad(M_PI);
    result.set_elevation_uncertainty_rad(std::numeric_limits<double>::infinity());

    if (samples.size() < 2) {
        return result;
    }

    double ky_min = samples[0].k.y();
    double ky_max = samples[0].k.y();
    for (const auto& s : samples) {
        ky_min = std::min(ky_min, s.k.y());
        ky_max = std::max(ky_max, s.k.y());
    }
    const bool elevation_observable =
        (ky_max - ky_min) > kMinKySpanRadM && fov_y_rad_ > 0.0;
    result.set_elevation_observable(elevation_observable);

    Eigen::Vector2d prior = coarse_search(samples, elevation_observable);
    return refine(samples, prior, elevation_observable);
}

namespace {

/// Angular extent of the matched-filter main lobe for a given K-span.
double lobe_width_rad(double focal_length_m, double k_span) {
    return kTwoPi / (focal_length_m * k_span);
}

}  // namespace

double AngleEstimator::matched_filter_score(const std::vector<Sample>& samples,
                                            const Eigen::Vector2d& r) {
    std::complex<double> acc(0.0, 0.0);
    for (const auto& s : samples) {
        acc += s.value * std::polar(1.0, s.k.dot(r));
    }
    return std::norm(acc);
}

Eigen::Vector2d AngleEstimator::search_grid(const std::vector<Sample>& samples,
                                            const Eigen::Vector2d& center_rad,
                                            const Eigen::Vector2d& half_extent_rad,
                                            uint32_t nx,
                                            uint32_t ny,
                                            bool prefer_boresight) const {
    const double f = focal_length_m();

    struct Node { Eigen::Vector2d r; double score; };
    std::vector<Node> grid;
    grid.reserve(static_cast<size_t>(nx) * ny);
    double best_score = 0.0;

    for (uint32_t iy = 0; iy < ny; ++iy) {
        double theta_y = (ny > 1)
            ? center_rad.y() - half_extent_rad.y() + 2.0 * half_extent_rad.y() * iy / (ny - 1)
            : center_rad.y();
        for (uint32_t ix = 0; ix < nx; ++ix) {
            double theta_x = (nx > 1)
                ? center_rad.x() - half_extent_rad.x() + 2.0 * half_extent_rad.x() * ix / (nx - 1)
                : center_rad.x();
            Eigen::Vector2d r(f * theta_x, f * theta_y);
            double score = matched_filter_score(samples, r);
            grid.push_back({r, score});
            best_score = std::max(best_score, score);
        }
    }

    // Uniformly spaced K-samples make the response periodic in angle, so
    // several bins can tie for the maximum (aliases). Among all bins
    // within tolerance of the best score, choose the one nearest boresight:
    // the unambiguous field of view is then set by the K-sample spacing.
    const double threshold = prefer_boresight
        ? best_score * (1.0 - kAliasScoreTolerance) : best_score;
    Eigen::Vector2d best_r(0.0, 0.0);
    double best_dist = std::numeric_limits<double>::infinity();
    for (const auto& node : grid) {
        if (node.score < threshold) continue;
        double dist = node.r.norm();
        if (dist < best_dist) {
            best_dist = dist;
            best_r = node.r;
        }
    }
    return best_r;
}

Eigen::Vector2d AngleEstimator::coarse_search(const std::vector<Sample>& samples,
                                              bool search_elevation) const {
    const double f = focal_length_m();

    double kx_min = samples[0].k.x(), kx_max = samples[0].k.x();
    double ky_min = samples[0].k.y(), ky_max = samples[0].k.y();
    for (const auto& s : samples) {
        kx_min = std::min(kx_min, s.k.x());
        kx_max = std::max(kx_max, s.k.x());
        ky_min = std::min(ky_min, s.k.y());
        ky_max = std::max(ky_max, s.k.y());
    }

    const double lobe_x = lobe_width_rad(f, kx_max - kx_min);
    const double lobe_y = search_elevation ? lobe_width_rad(f, ky_max - ky_min) : 0.0;

    auto bins_for = [](double fov, double spacing, uint32_t max_bins) -> uint32_t {
        if (spacing <= 0.0) return 1u;
        double bins = std::ceil(fov / spacing) + 1.0;
        return static_cast<uint32_t>(std::clamp(bins, 2.0, static_cast<double>(max_bins)));
    };

    // Level 1: whole field of view, sampled coarsely enough to be cheap but
    // finely enough that every lobe is seen within kAliasScoreTolerance.
    const uint32_t max_bins_x = search_elevation ? kMaxCoarseBins2d : max_coarse_bins_;
    const double spacing1_x = lobe_x / kLevel1SamplesPerLobe;
    const double spacing1_y = lobe_y / kLevel1SamplesPerLobe;
    const uint32_t nx1 = bins_for(fov_x_rad_, spacing1_x, max_bins_x);
    const uint32_t ny1 = search_elevation ? bins_for(fov_y_rad_, spacing1_y, kMaxCoarseBins2d) : 1u;

    Eigen::Vector2d r1 = search_grid(
        samples, Eigen::Vector2d(0.0, 0.0),
        Eigen::Vector2d(fov_x_rad_ / 2.0, search_elevation ? fov_y_rad_ / 2.0 : 0.0),
        nx1, ny1, /*prefer_boresight=*/true);

    // Level 2: refine locally around the selected lobe so the prior handed to
    // the least-squares stage is well inside the correct 2π branch.
    const double step1_x = (nx1 > 1) ? fov_x_rad_ / (nx1 - 1) : 0.0;
    const double step1_y = (ny1 > 1) ? fov_y_rad_ / (ny1 - 1) : 0.0;
    const uint32_t nx2 = static_cast<uint32_t>(
        2.0 * std::ceil(step1_x / (lobe_x / kLevel2SamplesPerLobe)) + 1.0);
    const uint32_t ny2 = search_elevation
        ? static_cast<uint32_t>(2.0 * std::ceil(step1_y / (lobe_y / kLevel2SamplesPerLobe)) + 1.0)
        : 1u;

    return search_grid(
        samples, r1 / f, Eigen::Vector2d(step1_x, step1_y),
        nx2, ny2, /*prefer_boresight=*/false);
}

AngleFitResult AngleEstimator::refine(const std::vector<Sample>& samples,
                                      const Eigen::Vector2d& prior_xy,
                                      bool fit_elevation) const {
    AngleFitResult result;
    result.set_valid(false);
    result.set_num_samples(static_cast<uint32_t>(samples.size()));
    result.set_elevation_observable(fit_elevation);
    result.set_azimuth_uncertainty_rad(M_PI);
    result.set_elevation_uncertainty_rad(std::numeric_limits<double>::infinity());

    const double f = focal_length_m();
    const int n_params = fit_elevation ? 3 : 2;  // (x, [y], common phase)
    const size_t n = samples.size();

    // Unwrap phases against the prior and assemble weighted normal equations.
    Eigen::MatrixXd A(n, n_params);
    Eigen::VectorXd b(n);
    Eigen::VectorXd w(n);
    double total_power = 0.0;
    double kx_min = samples[0].k.x();
    double kx_max = samples[0].k.x();

    for (size_t i = 0; i < n; ++i) {
        const auto& s = samples[i];
        double expected = -s.k.dot(prior_xy);
        double phi = std::arg(s.value);
        double wraps = std::round((phi - expected) / kTwoPi);
        double unwrapped = phi - wraps * kTwoPi;

        double magnitude = std::abs(s.value);
        w(i) = magnitude;
        total_power += magnitude * magnitude;

        A(i, 0) = -s.k.x();
        if (fit_elevation) {
            A(i, 1) = -s.k.y();
        }
        A(i, n_params - 1) = 1.0;
        b(i) = unwrapped;

        kx_min = std::min(kx_min, s.k.x());
        kx_max = std::max(kx_max, s.k.x());
    }

    // Weighted least squares: (Aᵀ W A) p = Aᵀ W b
    Eigen::MatrixXd Aw = w.asDiagonal() * A;
    Eigen::MatrixXd normal = A.transpose() * Aw;
    Eigen::VectorXd rhs = A.transpose() * (w.asDiagonal() * b);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(normal);
    double sigma_max = svd.singularValues()(0);
    double sigma_min = svd.singularValues()(svd.singularValues().size() - 1);
    if (sigma_max <= 0.0 || sigma_min <= kMinRelativeDeterminant * sigma_max) {
        return result;
    }
    result.set_condition_number(sigma_max / sigma_min);

    Eigen::VectorXd p = normal.ldlt().solve(rhs);
    double x = p(0);
    double y = fit_elevation ? p(1) : prior_xy.y();

    // Residuals
    Eigen::VectorXd residuals = b - A * p;
    double sum_w = w.sum();
    double ssr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ssr += w(i) * residuals(i) * residuals(i);
    }
    double residual_rms = (sum_w > 0.0) ? std::sqrt(ssr / sum_w) : M_PI;

    // Uncertainty: Cramer-Rao style bound from SNR and K-span.
    double snr_linear = (noise_floor_linear() > 0.0)
        ? total_power / noise_floor_linear() : 1.0;
    double azimuth_uncertainty = compute_uncertainty(
        kx_max - kx_min, snr_linear, static_cast<uint32_t>(n));

    result.set_azimuth_rad(x / f);
    result.set_elevation_rad(fit_elevation ? y / f : 0.0);
    result.set_azimuth_uncertainty_rad(azimuth_uncertainty);
    if (fit_elevation) {
        double ky_min = samples[0].k.y();
        double ky_max = samples[0].k.y();
        for (const auto& s : samples) {
            ky_min = std::min(ky_min, s.k.y());
            ky_max = std::max(ky_max, s.k.y());
        }
        result.set_elevation_uncertainty_rad(compute_uncertainty(
            ky_max - ky_min, snr_linear, static_cast<uint32_t>(n)));
    }
    result.set_residual_rad(residual_rms);
    result.set_valid(true);
    return result;
}

}  // namespace aurora::detection
