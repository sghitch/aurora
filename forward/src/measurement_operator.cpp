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

#include "forward/measurement_operator.h"

#include <cmath>
#include <stdexcept>

#include "forward/bandwidth_model.h"
#include "forward/forward_model.h"

namespace aurora::forward {

// Default threshold for integrate_response() support region computation.
// Points where the window function falls below this threshold are excluded
// from the integration domain.
constexpr double kDefaultIntegrationThreshold = 0.001;

// =============================================================================
// SupportBounds Implementation
// =============================================================================

bool MeasurementOperator::SupportBounds::contains(const Eigen::Vector2d& k) const {
    return k.x() >= k_min_x && k.x() <= k_max_x &&
           k.y() >= k_min_y && k.y() <= k_max_y;
}

bool MeasurementOperator::SupportBounds::is_valid() const {
    return k_min_x < k_max_x && k_min_y < k_max_y;
}

// =============================================================================
// Weight Computation
// =============================================================================

double MeasurementOperator::weight(const Eigen::Vector2d& k,
                                   const WindowParams& params,
                                   BandwidthWindowType window_type) {
    Eigen::Vector2d delta = k - params.k_center;
    double H_x = BandwidthModel::window_function(
        delta.x(), params.bandwidth_x, window_type);
    double H_y = BandwidthModel::window_function(
        delta.y(), params.bandwidth_y, window_type);
    return H_x * H_y;
}

double MeasurementOperator::point_target_weight(const Eigen::Vector2d& r_target,
                                                const WindowParams& params,
                                                BandwidthWindowType window_type,
                                                int quadrature_order) {
    // A point target at spatial position r_target has K-space representation
    // X(k) = exp(i·k·r_target) for all k. The measurement integrates:
    //   y = ∫ W(k) · exp(i·k·r_target) dk
    // This is the Fourier transform of W evaluated at r_target.

    auto X_k = [&r_target](const Eigen::Vector2d& k) -> std::complex<double> {
        double phase = k.x() * r_target.x() + k.y() * r_target.y();
        return std::polar(1.0, phase);
    };

    std::complex<double> integrated = integrate_response(
        X_k, params, window_type, quadrature_order);

    return std::abs(integrated);
}

// =============================================================================
// Integration for Extended Targets
// =============================================================================

std::complex<double> MeasurementOperator::integrate_response(
    const std::function<std::complex<double>(const Eigen::Vector2d&)>& X_k,
    const WindowParams& params,
    BandwidthWindowType window_type,
    int quadrature_points) {

    if (quadrature_points < 1) {
        throw std::invalid_argument(
            "quadrature_points must be at least 1");
    }

    auto bounds = compute_support(params, kDefaultIntegrationThreshold, window_type);
    if (!bounds.is_valid()) {
        return {0.0, 0.0};
    }

    // Quadrature over support region using midpoint rule.
    // CRITICAL: Must integrate COMPLEX field, not magnitude!
    //
    // Midpoint rule: divide domain into N×N cells, sample at cell centers.
    // This correctly integrates constants to the region area.
    std::complex<double> integral{0.0, 0.0};

    double dx = (bounds.k_max_x - bounds.k_min_x) / static_cast<double>(quadrature_points);
    double dy = (bounds.k_max_y - bounds.k_min_y) / static_cast<double>(quadrature_points);

    for (int i = 0; i < quadrature_points; ++i) {
        double kx = bounds.k_min_x + (i + 0.5) * dx;
        for (int j = 0; j < quadrature_points; ++j) {
            double ky = bounds.k_min_y + (j + 0.5) * dy;
            Eigen::Vector2d k{kx, ky};

            double w = weight(k, params, window_type);
            std::complex<double> X = X_k(k);  // COMPLEX value, not |X|!

            integral += w * X;
        }
    }

    // Scale by quadrature cell area (NO normalization by total weight!)
    return integral * dx * dy;
}

// =============================================================================
// Support Region Computation
// =============================================================================

double MeasurementOperator::compute_support_factor(BandwidthWindowType window_type,
                                                   double threshold) {
    // Support factor determines how many bandwidth half-widths to include
    // based on the window type and threshold.
    //
    // Since W = H_x * H_y (product of 2 separable factors), we use
    // sqrt(threshold) per axis to ensure the product exceeds the caller's
    // threshold when both factors are at their limit.
    //
    // For sinc: sidelobes decay slowly, need larger support
    // For gaussian: exp(-0.5 * x^2), falls off quickly
    // For kaiser: finite support within bandwidth

    double per_axis_threshold = std::sqrt(threshold);

    switch (window_type) {
        case BANDWIDTH_WINDOW_SINC:
            if (per_axis_threshold >= 0.5) {
                return 2.0;
            } else if (per_axis_threshold >= 0.3) {
                return 3.0;
            } else if (per_axis_threshold >= 0.15) {
                return 4.0;
            } else {
                return 5.0;
            }

        case BANDWIDTH_WINDOW_GAUSSIAN:
            // Gaussian: exp(-0.5 * (x/bw)^2) = per_axis_threshold
            // x/bw = sqrt(-2 * ln(per_axis_threshold))
            if (per_axis_threshold > 0.0 && per_axis_threshold < 1.0) {
                return std::sqrt(-2.0 * std::log(per_axis_threshold));
            }
            return 3.0;  // Default for edge cases

        case BANDWIDTH_WINDOW_KAISER:
            // Kaiser has finite support within bandwidth
            return 1.0;

        case BANDWIDTH_WINDOW_UNSPECIFIED:
        default:
            throw std::invalid_argument(
                "window_type must be explicitly specified (not UNSPECIFIED)");
    }
}

MeasurementOperator::SupportBounds MeasurementOperator::compute_support(
    const WindowParams& params,
    double threshold,
    BandwidthWindowType window_type) {

    double support_factor = compute_support_factor(window_type, threshold);

    double half_x = support_factor * params.bandwidth_x;
    double half_y = support_factor * params.bandwidth_y;

    return SupportBounds{
        .k_min_x = params.k_center.x() - half_x,
        .k_max_x = params.k_center.x() + half_x,
        .k_min_y = params.k_center.y() - half_y,
        .k_max_y = params.k_center.y() + half_y
    };
}

// =============================================================================
// Helper Function: Build WindowParams
// =============================================================================

MeasurementOperator::WindowParams build_window_params(
    const ForwardModel& forward_model,
    uint32_t detector_idx,
    double wavelength_m) {

    // Scene K-space coordinates (includes LO tilt). KSpaceSlice uses scene K,
    // so the window must be centered in the same coordinates.
    auto k_sample = forward_model.get_k_sample(detector_idx, wavelength_m);

    return MeasurementOperator::WindowParams{
        .k_center = forward_model.get_effective_k(detector_idx, wavelength_m),
        .bandwidth_x = k_sample.kx_bandwidth(),
        .bandwidth_y = k_sample.ky_bandwidth()
    };
}

}  // namespace aurora::forward
