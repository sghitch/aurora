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

#include "simulation/pupil_model.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace aurora::simulation {

PupilModel::PupilModel(const aurora::forward::PupilConfig& config)
    : fov_half_angle_rad_(config.fov_half_angle_rad()),
      rolloff_width_rad_(config.rolloff_width_rad()),
      shape_(config.shape()),
      aspect_ratio_(config.has_aspect_ratio() ? config.aspect_ratio() : 1.0) {}

double PupilModel::compute_transmission(
    const Eigen::Vector3d& target_position_m) const {
    auto [azimuth, elevation] = position_to_angles(target_position_m);
    return compute_transmission(azimuth, elevation);
}

double PupilModel::compute_transmission(double azimuth_rad,
                                         double elevation_rad) const {
    double theta;

    if (shape_ == aurora::forward::PupilConfig::PUPIL_CIRCULAR) {
        // Angular distance from optical axis
        theta = std::sqrt(azimuth_rad * azimuth_rad +
                          elevation_rad * elevation_rad);
    } else {
        // Rectangular: use max of normalized coordinates
        // For rectangular FOV, we use the aspect ratio to scale the angles
        double half_width = fov_half_angle_rad_;
        double half_height = fov_half_angle_rad_ / aspect_ratio_;

        double norm_az = std::abs(azimuth_rad) / half_width;
        double norm_el = std::abs(elevation_rad) / half_height;

        // Use max of normalized coordinates, then scale back to angle
        theta = std::max(norm_az, norm_el) * fov_half_angle_rad_;
    }

    // Compute transmission with smooth rolloff
    double edge_angle = fov_half_angle_rad_ - rolloff_width_rad_;

    if (theta < edge_angle) {
        // Inside FOV core: full transmission
        return 1.0;
    } else if (theta < fov_half_angle_rad_) {
        // In rolloff region: smooth cosine transition
        double t = (theta - edge_angle) / rolloff_width_rad_;
        return 0.5 * (1.0 + std::cos(std::numbers::pi * t));
    } else {
        // Outside FOV: zero transmission
        return 0.0;
    }
}

std::pair<double, double> PupilModel::position_to_angles(
    const Eigen::Vector3d& position_m) const {
    double range = position_m.z();  // z is the range axis

    if (range <= 0.0) {
        // Behind sensor: return angles outside FOV
        return {std::numbers::pi, std::numbers::pi};
    }

    double azimuth = std::atan2(position_m.x(), range);
    double elevation = std::atan2(position_m.y(), range);

    return {azimuth, elevation};
}

bool PupilModel::is_in_fov(const Eigen::Vector3d& target_position_m) const {
    return compute_transmission(target_position_m) > 0.0;
}

std::vector<std::pair<double, double>> PupilModel::get_fov_boundary(
    int num_points) const {
    std::vector<std::pair<double, double>> boundary;
    boundary.reserve(static_cast<size_t>(num_points));

    if (shape_ == aurora::forward::PupilConfig::PUPIL_CIRCULAR) {
        // Circular boundary
        for (int i = 0; i < num_points; ++i) {
            double angle =
                2.0 * std::numbers::pi * static_cast<double>(i) / num_points;
            double azimuth = fov_half_angle_rad_ * std::cos(angle);
            double elevation = fov_half_angle_rad_ * std::sin(angle);
            boundary.emplace_back(azimuth, elevation);
        }
    } else {
        // Rectangular boundary
        double half_width = fov_half_angle_rad_;
        double half_height = fov_half_angle_rad_ / aspect_ratio_;

        int points_per_side = num_points / 4;

        // Top edge
        for (int i = 0; i < points_per_side; ++i) {
            double t = static_cast<double>(i) / points_per_side;
            double azimuth = half_width * (2.0 * t - 1.0);
            boundary.emplace_back(azimuth, half_height);
        }
        // Right edge
        for (int i = 0; i < points_per_side; ++i) {
            double t = static_cast<double>(i) / points_per_side;
            double elevation = half_height * (1.0 - 2.0 * t);
            boundary.emplace_back(half_width, elevation);
        }
        // Bottom edge
        for (int i = 0; i < points_per_side; ++i) {
            double t = static_cast<double>(i) / points_per_side;
            double azimuth = half_width * (1.0 - 2.0 * t);
            boundary.emplace_back(azimuth, -half_height);
        }
        // Left edge
        for (int i = 0; i < points_per_side; ++i) {
            double t = static_cast<double>(i) / points_per_side;
            double elevation = half_height * (2.0 * t - 1.0);
            boundary.emplace_back(-half_width, elevation);
        }
    }

    return boundary;
}

}  // namespace aurora::simulation
