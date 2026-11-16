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

#ifndef AURORA_SIMULATION_SCENE_H_
#define AURORA_SIMULATION_SCENE_H_

/**
 * @file scene.h
 * @brief Scene representation for FMCW simulation.
 *
 * The Scene class models point targets, planar surfaces, and their spectral
 * properties for FMCW signal simulation.
 *
 * ### Spectral Properties
 *
 * Different materials have different reflectivities at 905nm and 1550nm:
 * - Dry asphalt: ~1.0 ratio (similar at both wavelengths)
 * - Wet surfaces: <0.5 ratio (water absorbs 1550nm)
 * - Vegetation: ~0.5-0.7 ratio (chlorophyll effects)
 *
 * @see SceneDefinition for protobuf definition
 * @see FmcwSimulator for signal generation from scenes
 */

#include <Eigen/Dense>
#include <vector>

#include "scene.pb.h"
#include "simulation/views.h"
#include "shared/eigen_utils.h"

namespace aurora::simulation {

/**
 * @brief Scene representation for FMCW simulation.
 *
 * The Scene class manages targets and provides access to their properties.
 * K-space responses are computed at the individual target level via
 * PointTargetView and PlaneTargetView.
 *
 * ### Example Usage
 * @code
 * SceneDefinition definition;
 * // ... populate definition ...
 *
 * Scene scene(definition);
 *
 * // Update positions after 0.1 seconds
 * scene.update(0.1);
 *
 * // Access individual targets
 * auto target = scene.point_target(0);
 * double range = target.range_m();
 * @endcode
 *
 * @see FmcwSimulator for signal generation from scenes
 */
class Scene {
  public:
    /// Reference wavelength for reflectivity (905nm)
    static constexpr double kReferenceWavelength905nm = 905e-9;

    /// Secondary wavelength (1550nm)
    static constexpr double kSecondaryWavelength1550nm = 1550e-9;

    /**
     * @brief Construct a scene from a protobuf definition.
     * @param definition Scene definition containing targets and environment
     */
    explicit Scene(const SceneDefinition& definition);

    /**
     * @brief Update target positions based on velocity.
     *
     * Applies velocity integration to all targets:
     *   position_new = position_old + velocity * delta_time
     *
     * @param delta_time_s Time step in seconds
     */
    void update(double delta_time_s);

    /**
     * @brief Get all point target definitions.
     * @return Vector of point target protos
     */
    const std::vector<PointTargetDefinition>& get_point_targets() const {
        return point_targets_;
    }

    /**
     * @brief Get all plane target definitions.
     * @return Vector of plane target protos
     */
    const std::vector<PlaneTargetDefinition>& get_plane_targets() const {
        return plane_targets_;
    }

    /**
     * @brief Get all mesh target definitions.
     * @return Vector of mesh target protos
     */
    const std::vector<MeshTargetDefinition>& get_mesh_targets() const {
        return mesh_targets_;
    }

    /**
     * @brief Get the number of mesh targets.
     * @return Number of mesh targets
     */
    size_t num_mesh_targets() const { return mesh_targets_.size(); }

    /**
     * @brief Get a mesh target view by index.
     * @param index Target index
     * @return MeshTargetView wrapping the target proto
     */
    MeshTargetView mesh_target(size_t index) const {
        return MeshTargetView(mesh_targets_.at(index));
    }

    /**
     * @brief Get a point target view by index.
     * @param index Target index
     * @return PointTargetView wrapping the target proto
     */
    PointTargetView point_target(size_t index) const {
        return PointTargetView(point_targets_.at(index));
    }

    /**
     * @brief Get a plane target view by index.
     * @param index Target index
     * @return PlaneTargetView wrapping the target proto
     */
    PlaneTargetView plane_target(size_t index) const {
        return PlaneTargetView(plane_targets_.at(index));
    }

    /**
     * @brief Get reflectivity for a target at a specific wavelength.
     *
     * Uses the per-wavelength reflectivity from surface properties.
     * Interpolates for wavelengths between 905nm and 1550nm.
     *
     * @param target Point target view to query
     * @param wavelength_m Wavelength in meters
     * @return Reflectivity at the specified wavelength
     */
    static double get_reflectivity(const PointTargetView& target,
                                   double wavelength_m);

    /**
     * @brief Get reflectivity for a plane target at a specific wavelength.
     *
     * Uses the per-wavelength reflectivity from surface properties.
     * Interpolates for wavelengths between 905nm and 1550nm.
     *
     * @param target Plane target view to query
     * @param wavelength_m Wavelength in meters
     * @return Reflectivity at the specified wavelength
     */
    static double get_reflectivity(const PlaneTargetView& target,
                                   double wavelength_m);

    /**
     * @brief Get reflectivity for a mesh target at a specific wavelength.
     *
     * Uses the per-wavelength reflectivity from default surface properties.
     * Interpolates for wavelengths between 905nm and 1550nm.
     *
     * @param target Mesh target view to query
     * @param wavelength_m Wavelength in meters
     * @return Reflectivity at the specified wavelength
     */
    static double get_reflectivity(const MeshTargetView& target,
                                   double wavelength_m);

    /**
     * @brief Get the number of point targets.
     * @return Number of point targets
     */
    size_t point_target_count() const { return point_targets_.size(); }

    /**
     * @brief Get the number of plane targets.
     * @return Number of plane targets
     */
    size_t plane_target_count() const { return plane_targets_.size(); }

    /**
     * @brief Get ambient temperature.
     * @return Temperature in Kelvin
     */
    double ambient_temperature_k() const { return ambient_temperature_k_; }

    /**
     * @brief Get atmospheric extinction coefficient.
     * @return Extinction coefficient in 1/m
     */
    double atmospheric_extinction_per_m() const {
        return atmospheric_extinction_per_m_;
    }

    /**
     * @brief Get background radiance.
     * @return Background radiance in W/(m²·sr)
     */
    double background_radiance_w_per_m2_sr() const {
        return background_radiance_w_per_m2_sr_;
    }

  private:
    std::vector<PointTargetDefinition> point_targets_;
    std::vector<PlaneTargetDefinition> plane_targets_;
    std::vector<MeshTargetDefinition> mesh_targets_;

    // Environment parameters (read from config, no defaults assumed)
    double ambient_temperature_k_ = 0.0;
    double atmospheric_extinction_per_m_ = 0.0;
    double background_radiance_w_per_m2_sr_ = 0.0;
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_SCENE_H_
