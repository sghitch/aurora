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

#include "forward/k_space_geometry.h"

#include <cmath>
#include <stdexcept>

namespace aurora::forward {

namespace {
// Mathematical constant
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
}  // namespace

KSpaceGeometry::KSpaceGeometry(const OpticsConfig& optics,
                               const DetectorArray& detector)
    : optics_(optics), detector_(detector) {
    // Validate sensitivity type at construction time
    validate_sensitivity_type();
}

// =============================================================================
// Detector K-positions
// =============================================================================

double KSpaceGeometry::detector_k_x(uint32_t detector_idx,
                                    double wavelength_m) const {
    validate_detector_index(detector_idx);
    validate_wavelength(wavelength_m);

    const auto& pixel = detector_.pixels(static_cast<int>(detector_idx));
    double x_det = pixel.x_m();
    double f = optics_.fourier_lens_focal_m();

    // k_x = 2π × x_det / (f × λ)
    return kTwoPi * x_det / (f * wavelength_m);
}

double KSpaceGeometry::detector_k_y(uint32_t detector_idx,
                                    double wavelength_m) const {
    validate_detector_index(detector_idx);
    validate_wavelength(wavelength_m);

    const auto& pixel = detector_.pixels(static_cast<int>(detector_idx));
    double y_det = pixel.y_m();
    double f = optics_.fourier_lens_focal_m();

    // k_y = 2π × y_det / (f × λ)
    return kTwoPi * y_det / (f * wavelength_m);
}

Eigen::Vector2d KSpaceGeometry::detector_k(uint32_t detector_idx,
                                           double wavelength_m) const {
    return Eigen::Vector2d(detector_k_x(detector_idx, wavelength_m),
                           detector_k_y(detector_idx, wavelength_m));
}

std::vector<Eigen::Vector2d> KSpaceGeometry::all_detector_k(
    double wavelength_m) const {
    validate_wavelength(wavelength_m);

    std::vector<Eigen::Vector2d> result;
    result.reserve(static_cast<size_t>(detector_.pixels_size()));

    for (int i = 0; i < detector_.pixels_size(); ++i) {
        result.push_back(detector_k(static_cast<uint32_t>(i), wavelength_m));
    }

    return result;
}

// =============================================================================
// Detector K-space windows
// =============================================================================

double KSpaceGeometry::detector_k_bandwidth_x(uint32_t detector_idx,
                                              double wavelength_m) const {
    validate_detector_index(detector_idx);
    validate_wavelength(wavelength_m);

    const auto& pixel = detector_.pixels(static_cast<int>(detector_idx));
    double w_det = pixel.width_m();
    double f = optics_.fourier_lens_focal_m();

    // Δk_x = 2π × w_det / (f × λ)
    return kTwoPi * w_det / (f * wavelength_m);
}

double KSpaceGeometry::detector_k_bandwidth_y(uint32_t detector_idx,
                                              double wavelength_m) const {
    validate_detector_index(detector_idx);
    validate_wavelength(wavelength_m);

    const auto& pixel = detector_.pixels(static_cast<int>(detector_idx));
    double h_det = pixel.height_m();
    double f = optics_.fourier_lens_focal_m();

    // Δk_y = 2π × h_det / (f × λ)
    return kTwoPi * h_det / (f * wavelength_m);
}

// =============================================================================
// K-space coordinate conversions
// =============================================================================

double KSpaceGeometry::detector_to_k(double x_m, double wavelength_m) const {
    validate_wavelength(wavelength_m);
    double f = optics_.fourier_lens_focal_m();
    // k = 2π × x / (f × λ)
    return kTwoPi * x_m / (f * wavelength_m);
}

// =============================================================================
// LO tilt
// =============================================================================

Eigen::Vector2d KSpaceGeometry::lo_tilt_k(double wavelength_m) const {
    validate_wavelength(wavelength_m);

    // k_LO = 2π × sin(θ) / λ
    double kx = kTwoPi * std::sin(optics_.lo_tilt_x_rad()) / wavelength_m;
    double ky = kTwoPi * std::sin(optics_.lo_tilt_y_rad()) / wavelength_m;

    return Eigen::Vector2d(kx, ky);
}

// =============================================================================
// Accessors
// =============================================================================

uint32_t KSpaceGeometry::num_detectors() const {
    return static_cast<uint32_t>(detector_.pixels_size());
}

double KSpaceGeometry::focal_length_m() const {
    return optics_.fourier_lens_focal_m();
}

Eigen::Vector2d KSpaceGeometry::lo_tilt_rad() const {
    return Eigen::Vector2d(optics_.lo_tilt_x_rad(), optics_.lo_tilt_y_rad());
}

// =============================================================================
// Validation
// =============================================================================

void KSpaceGeometry::validate_wavelength(double wavelength_m) const {
    if (wavelength_m <= 0.0) {
        throw std::invalid_argument(
            "wavelength_m must be positive, got: " +
            std::to_string(wavelength_m));
    }
}

void KSpaceGeometry::validate_detector_index(uint32_t detector_idx) const {
    if (detector_idx >= static_cast<uint32_t>(detector_.pixels_size())) {
        throw std::out_of_range(
            "detector_idx " + std::to_string(detector_idx) +
            " out of range [0, " + std::to_string(detector_.pixels_size()) +
            ")");
    }
}

void KSpaceGeometry::validate_sensitivity_type() const {
    if (detector_.sensitivity_type() == DETECTOR_SENSITIVITY_UNSPECIFIED) {
        throw std::invalid_argument(
            "DetectorArray.sensitivity_type must be explicitly specified "
            "(DETECTOR_SENSITIVITY_POINT or DETECTOR_SENSITIVITY_SINC). "
            "DETECTOR_SENSITIVITY_UNSPECIFIED is not allowed.");
    }
}

// =============================================================================
// Sensitivity Functions
// =============================================================================

double KSpaceGeometry::detector_sensitivity(uint32_t detector_idx,
                                            double wavelength_m,
                                            const Eigen::Vector2d& delta_k) const {
    validate_sensitivity_type();
    validate_detector_index(detector_idx);
    validate_wavelength(wavelength_m);

    switch (detector_.sensitivity_type()) {
        case DETECTOR_SENSITIVITY_POINT:
            return 1.0;

        case DETECTOR_SENSITIVITY_SINC: {
            // Get detector K-bandwidth (this is the "width" of the detector in K-space)
            double bw_kx = detector_k_bandwidth_x(detector_idx, wavelength_m);
            double bw_ky = detector_k_bandwidth_y(detector_idx, wavelength_m);

            // Sinc sensitivity: sinc(delta_k / bandwidth)
            // The sinc function has its first nulls at delta_k = ±bandwidth
            // Delegate to BandwidthModel for unified physics
            double sx = BandwidthModel::sinc_window(delta_k.x(), bw_kx);
            double sy = BandwidthModel::sinc_window(delta_k.y(), bw_ky);

            return sx * sy;
        }

        default:
            throw std::invalid_argument(
                "Unknown DetectorSensitivityType: " +
                std::to_string(static_cast<int>(detector_.sensitivity_type())));
    }
}

}  // namespace aurora::forward
