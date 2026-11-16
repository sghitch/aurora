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

#include "forward/bandwidth_model.h"
#include <cmath>
#include <stdexcept>

namespace aurora::forward {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kKaiserBeta = kPi;  // Standard NUFFT gridding value
}  // namespace

double BandwidthModel::sinc(double x) {
    if (std::abs(x) < 1e-10) {
        return 1.0;
    }
    double pi_x = kPi * x;
    return std::sin(pi_x) / pi_x;
}

double BandwidthModel::bessel_i0(double x) {
    // Polynomial approximation of modified Bessel function I₀(x)
    // Accurate to ~1e-7 for |x| < 15
    double ax = std::abs(x);
    if (ax < 3.75) {
        double t = x / 3.75;
        double t2 = t * t;
        return 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492
               + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
    } else {
        double t = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + t * (0.01328592
               + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281
               + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633
               + t * 0.00392377))))))));
    }
}

double BandwidthModel::sinc_window(double delta_k, double bandwidth) {
    if (bandwidth <= 0) {
        return 1.0;
    }
    return sinc(delta_k / bandwidth);
}

double BandwidthModel::gaussian_window(double delta_k, double bandwidth) {
    if (bandwidth <= 0) {
        return 1.0;
    }
    double normalized = delta_k / bandwidth;
    return std::exp(-0.5 * normalized * normalized);
}

double BandwidthModel::kaiser_window(double delta_k, double bandwidth) {
    if (bandwidth <= 0) {
        return 1.0;
    }
    double normalized = delta_k / bandwidth;
    if (std::abs(normalized) > 1.0) {
        return 0.0;
    }
    double arg = kKaiserBeta * std::sqrt(1.0 - normalized * normalized);
    return bessel_i0(arg) / bessel_i0(kKaiserBeta);
}

double BandwidthModel::window_function(double delta_k,
                                        double bandwidth,
                                        BandwidthWindowType window_type) {
    switch (window_type) {
        case BANDWIDTH_WINDOW_SINC:
            return sinc_window(delta_k, bandwidth);
        case BANDWIDTH_WINDOW_GAUSSIAN:
            return gaussian_window(delta_k, bandwidth);
        case BANDWIDTH_WINDOW_KAISER:
            return kaiser_window(delta_k, bandwidth);
        case BANDWIDTH_WINDOW_UNSPECIFIED:
            throw std::invalid_argument(
                "BandwidthWindowType must be specified (SINC, GAUSSIAN, or KAISER)");
        default:
            throw std::invalid_argument(
                "Unknown BandwidthWindowType: " + std::to_string(window_type));
    }
}

double BandwidthModel::k_space_weight(const Eigen::Vector2d& delta_k,
                                       double k_bandwidth_x,
                                       double k_bandwidth_y,
                                       BandwidthWindowType window_type) {
    double wx = window_function(delta_k.x(), k_bandwidth_x, window_type);
    double wy = window_function(delta_k.y(), k_bandwidth_y, window_type);
    return wx * wy;
}

double BandwidthModel::rect_convolution(double x, double width1, double width2) {
    double w_max = std::max(width1, width2);
    double w_min = std::min(width1, width2);
    double abs_x = std::abs(x);

    // Outside support
    if (abs_x >= (w_max + w_min) / 2.0) {
        return 0.0;
    }

    // Inside flat region
    if (abs_x <= (w_max - w_min) / 2.0) {
        return 1.0;
    }

    // Sloped region (linear interpolation)
    double edge = (w_max + w_min) / 2.0;
    double flat_edge = (w_max - w_min) / 2.0;
    return (edge - abs_x) / (edge - flat_edge);
}

Eigen::Vector2d BandwidthModel::resolution_limit(double k_bandwidth_x,
                                                  double k_bandwidth_y) {
    double dx = (k_bandwidth_x > 0) ? kTwoPi / k_bandwidth_x : 0.0;
    double dy = (k_bandwidth_y > 0) ? kTwoPi / k_bandwidth_y : 0.0;
    return {dx, dy};
}

double BandwidthModel::k_bandwidth_to_fourier_width(double k_bandwidth,
                                                     double wavelength_m,
                                                     double focal_length_m) {
    return k_bandwidth * focal_length_m * wavelength_m / kTwoPi;
}

double BandwidthModel::k_bandwidth_to_spatial_width(double k_bandwidth) {
    return (k_bandwidth > 0) ? kTwoPi / k_bandwidth : 0.0;
}

double BandwidthModel::spatial_width_to_k_bandwidth(double spatial_width) {
    return (spatial_width > 0) ? kTwoPi / spatial_width : 0.0;
}

bool BandwidthModel::is_spacing_valid(double grid_spacing,
                                       double k_bandwidth,
                                       double tolerance_factor) {
    if (k_bandwidth <= 0) return true;
    double limit = kTwoPi / k_bandwidth;
    return grid_spacing >= limit * tolerance_factor;
}

}  // namespace aurora::forward
