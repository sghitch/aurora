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

#include "simulation/scene.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aurora::simulation {

Scene::Scene(const SceneDefinition& definition) {
    // Load and validate point targets
    point_targets_.reserve(definition.point_targets_size());
    for (int i = 0; i < definition.point_targets_size(); ++i) {
        const auto& pt_def = definition.point_targets(i);
        if (!pt_def.has_surface()) {
            throw std::invalid_argument(
                "PointTargetDefinition at index " + std::to_string(i) +
                " must have surface properties");
        }
        // Validate via View (will throw if invalid)
        PointTargetView::validate(pt_def);
        point_targets_.push_back(pt_def);
    }

    // Load and validate plane targets
    plane_targets_.reserve(definition.plane_targets_size());
    for (int i = 0; i < definition.plane_targets_size(); ++i) {
        const auto& plane_def = definition.plane_targets(i);
        if (!plane_def.has_surface()) {
            throw std::invalid_argument(
                "PlaneTargetDefinition at index " + std::to_string(i) +
                " must have surface properties");
        }
        // Validate via View (will throw if invalid)
        PlaneTargetView::validate(plane_def);
        plane_targets_.push_back(plane_def);
    }

    // Load and validate mesh targets
    mesh_targets_.reserve(definition.mesh_targets_size());
    for (int i = 0; i < definition.mesh_targets_size(); ++i) {
        const auto& mesh_def = definition.mesh_targets(i);
        if (!mesh_def.has_surface()) {
            throw std::invalid_argument(
                "MeshTargetDefinition at index " + std::to_string(i) +
                " must have surface properties");
        }
        // Validate via View (will throw if invalid)
        MeshTargetView::validate(mesh_def);
        mesh_targets_.push_back(mesh_def);
    }

    // Load environment parameters directly from config (no defaults assumed)
    ambient_temperature_k_ = definition.ambient_temperature_k();
    atmospheric_extinction_per_m_ = definition.atmospheric_extinction_per_m();
    background_radiance_w_per_m2_sr_ = definition.background_radiance_w_per_m2_sr();
}

void Scene::update(double delta_time_s) {
    // Update point target positions
    for (auto& target : point_targets_) {
        Eigen::Vector3d pos = aurora::shared::to_eigen(target.position_m());
        Eigen::Vector3d vel = aurora::shared::to_eigen(target.velocity_mps());
        pos += vel * delta_time_s;
        target.mutable_position_m()->set_x(pos.x());
        target.mutable_position_m()->set_y(pos.y());
        target.mutable_position_m()->set_z(pos.z());
    }

    // Update plane target positions
    for (auto& target : plane_targets_) {
        Eigen::Vector3d pos = aurora::shared::to_eigen(target.center_m());
        Eigen::Vector3d vel = aurora::shared::to_eigen(target.velocity_mps());
        pos += vel * delta_time_s;
        target.mutable_center_m()->set_x(pos.x());
        target.mutable_center_m()->set_y(pos.y());
        target.mutable_center_m()->set_z(pos.z());
    }

    // Update mesh target positions
    for (auto& target : mesh_targets_) {
        Eigen::Vector3d pos = aurora::shared::to_eigen(target.position_m());
        Eigen::Vector3d vel = aurora::shared::to_eigen(target.velocity_mps());
        pos += vel * delta_time_s;
        target.mutable_position_m()->set_x(pos.x());
        target.mutable_position_m()->set_y(pos.y());
        target.mutable_position_m()->set_z(pos.z());
    }
}

double Scene::get_reflectivity(
    const PointTargetView& target,
    double wavelength_m) {
    // Check which wavelength band we're in
    // Use tolerance for floating point comparison
    constexpr double wavelength_tolerance = 1e-10;

    double refl_905 = target.surface().proto().reflectivity_905nm();
    double refl_1550 = target.surface().proto().reflectivity_1550nm();

    if (std::abs(wavelength_m - kReferenceWavelength905nm) < wavelength_tolerance) {
        // 905nm: use 905nm reflectivity directly
        return refl_905;
    } else if (std::abs(wavelength_m - kSecondaryWavelength1550nm) < wavelength_tolerance) {
        // 1550nm: use 1550nm reflectivity directly
        return refl_1550;
    } else {
        // Interpolate or extrapolate for other wavelengths
        // Linear interpolation between 905nm and 1550nm
        double t = (wavelength_m - kReferenceWavelength905nm) /
                   (kSecondaryWavelength1550nm - kReferenceWavelength905nm);
        t = std::clamp(t, 0.0, 1.0);  // Clamp to valid range

        return refl_905 + t * (refl_1550 - refl_905);
    }
}

double Scene::get_reflectivity(
    const PlaneTargetView& target,
    double wavelength_m) {
    // Check which wavelength band we're in
    // Use tolerance for floating point comparison
    constexpr double wavelength_tolerance = 1e-10;

    double refl_905 = target.surface().proto().reflectivity_905nm();
    double refl_1550 = target.surface().proto().reflectivity_1550nm();

    if (std::abs(wavelength_m - kReferenceWavelength905nm) < wavelength_tolerance) {
        // 905nm: use 905nm reflectivity directly
        return refl_905;
    } else if (std::abs(wavelength_m - kSecondaryWavelength1550nm) < wavelength_tolerance) {
        // 1550nm: use 1550nm reflectivity directly
        return refl_1550;
    } else {
        // Interpolate or extrapolate for other wavelengths
        // Linear interpolation between 905nm and 1550nm
        double t = (wavelength_m - kReferenceWavelength905nm) /
                   (kSecondaryWavelength1550nm - kReferenceWavelength905nm);
        t = std::clamp(t, 0.0, 1.0);  // Clamp to valid range

        return refl_905 + t * (refl_1550 - refl_905);
    }
}

double Scene::get_reflectivity(
    const MeshTargetView& target,
    double wavelength_m) {
    // Check which wavelength band we're in
    // Use tolerance for floating point comparison
    constexpr double wavelength_tolerance = 1e-10;

    double refl_905 = target.surface().proto().reflectivity_905nm();
    double refl_1550 = target.surface().proto().reflectivity_1550nm();

    if (std::abs(wavelength_m - kReferenceWavelength905nm) < wavelength_tolerance) {
        // 905nm: use 905nm reflectivity directly
        return refl_905;
    } else if (std::abs(wavelength_m - kSecondaryWavelength1550nm) < wavelength_tolerance) {
        // 1550nm: use 1550nm reflectivity directly
        return refl_1550;
    } else {
        // Interpolate or extrapolate for other wavelengths
        // Linear interpolation between 905nm and 1550nm
        double t = (wavelength_m - kReferenceWavelength905nm) /
                   (kSecondaryWavelength1550nm - kReferenceWavelength905nm);
        t = std::clamp(t, 0.0, 1.0);  // Clamp to valid range

        return refl_905 + t * (refl_1550 - refl_905);
    }
}

}  // namespace aurora::simulation
