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

#ifndef AURORA_SIMULATION_VIEWS_H_
#define AURORA_SIMULATION_VIEWS_H_

/**
 * @file views.h
 * @brief Thin View wrapper classes around target definition protos.
 *
 * These classes provide convenient accessors for target definitions without
 * duplicating any proto fields. Each View holds a const reference to a proto
 * message and provides computed properties and validation.
 *
 * @see SurfaceProperties for material properties proto
 * @see PlaneTargetDefinition for plane target proto
 * @see PointTargetDefinition for point target proto
 */

#include <Eigen/Dense>
#include <complex>
#include <stdexcept>
#include <vector>

#include "scene.pb.h"
#include "shared/eigen_utils.h"

namespace aurora::simulation {

/**
 * @brief View wrapper for SurfaceProperties proto.
 *
 * Provides validation and read-only access to surface material properties
 * including per-wavelength reflectivity, roughness, and correlation length.
 */
class SurfacePropertiesView {
  public:
    /**
     * @brief Construct a view wrapping a SurfaceProperties proto.
     * @param proto The proto to wrap (must outlive this view)
     */
    explicit SurfacePropertiesView(const SurfaceProperties& proto);

    /**
     * @brief Get the underlying proto.
     * @return Const reference to the wrapped proto
     */
    const SurfaceProperties& proto() const { return proto_; }

    /**
     * @brief Compute intrinsic reflectivity at a given wavelength.
     *
     * Linearly interpolates between the 905nm and 1550nm reference values.
     * Values outside this range are clamped (no extrapolation).
     *
     * @param wavelength_m Wavelength in meters
     * @return Reflectivity in range (0, 1]
     */
    double intrinsic_reflectivity(double wavelength_m) const;

    /**
     * @brief Compute specular fraction using Rayleigh criterion.
     *
     * Uses the Rayleigh criterion: α² = exp(-(4πσ/λ)²) where σ is the
     * RMS roughness and λ is the wavelength. If specular_fraction_override
     * is set in the proto, that value is returned instead.
     *
     * @param wavelength_m Wavelength in meters
     * @return Specular fraction in range [0, 1]
     */
    double specular_fraction(double wavelength_m) const;

    /**
     * @brief Validate a SurfaceProperties proto.
     * @param proto The proto to validate
     * @throws std::invalid_argument if any field is invalid
     */
    static void validate(const SurfaceProperties& proto);

  private:
    const SurfaceProperties& proto_;
};

/**
 * @brief View wrapper for PlaneTargetDefinition proto.
 *
 * Provides convenient accessors for plane target properties including
 * center position, normal vector, dimensions, velocity, and surface properties.
 * Also provides computed properties like area and range.
 */
class PlaneTargetView {
  public:
    /**
     * @brief Construct a view wrapping a PlaneTargetDefinition proto.
     * @param proto The proto to wrap (must outlive this view)
     */
    explicit PlaneTargetView(const PlaneTargetDefinition& proto);

    /**
     * @brief Get the underlying proto.
     * @return Const reference to the wrapped proto
     */
    const PlaneTargetDefinition& proto() const { return proto_; }

    /**
     * @brief Compute the surface area of the plane.
     * @return Area in square meters
     */
    double area_m2() const;

    /**
     * @brief Compute range from origin to plane center.
     * @return Range in meters
     */
    double range_m() const;

    /**
     * @brief Get the center position as Eigen vector.
     * @return Center position in meters
     */
    Eigen::Vector3d center() const;

    /**
     * @brief Get the surface normal as Eigen vector.
     * @return Surface normal (unit vector)
     */
    Eigen::Vector3d normal() const;

    /**
     * @brief Get the velocity as Eigen vector.
     * @return Velocity in meters per second
     */
    Eigen::Vector3d velocity() const;

    /**
     * @brief Get surface properties as a view.
     * @return SurfacePropertiesView for the plane's surface
     */
    SurfacePropertiesView surface() const;

    /**
     * @brief Get width in meters.
     * @return Width in meters
     */
    double width_m() const { return proto_.width_m(); }

    /**
     * @brief Get height in meters.
     * @return Height in meters
     */
    double height_m() const { return proto_.height_m(); }

    /**
     * @brief Validate a PlaneTargetDefinition proto.
     * @param proto The proto to validate
     * @throws std::invalid_argument if any field is invalid
     */
    static void validate(const PlaneTargetDefinition& proto);

  private:
    const PlaneTargetDefinition& proto_;
};

/**
 * @brief View wrapper for PointTargetDefinition proto.
 *
 * Provides convenient accessors for point target properties including
 * position, velocity, and surface properties. Also provides computed
 * properties like range and radial velocity.
 */
class PointTargetView {
  public:
    /**
     * @brief Construct a view wrapping a PointTargetDefinition proto.
     * @param proto The proto to wrap (must outlive this view)
     */
    explicit PointTargetView(const PointTargetDefinition& proto);

    /**
     * @brief Get the underlying proto.
     * @return Const reference to the wrapped proto
     */
    const PointTargetDefinition& proto() const { return proto_; }

    /**
     * @brief Compute range from origin to target.
     * @return Range in meters
     */
    double range_m() const;

    /**
     * @brief Compute radial velocity (velocity component toward origin).
     *
     * Positive values indicate target moving away from origin.
     *
     * @return Radial velocity in meters per second
     */
    double radial_velocity_mps() const;

    /**
     * @brief Get the position as Eigen vector.
     * @return Position in meters
     */
    Eigen::Vector3d position() const;

    /**
     * @brief Get the velocity as Eigen vector.
     * @return Velocity in meters per second
     */
    Eigen::Vector3d velocity() const;

    /**
     * @brief Get surface properties as a view.
     * @return SurfacePropertiesView for the target's surface
     */
    SurfacePropertiesView surface() const;

    /**
     * @brief Validate a PointTargetDefinition proto.
     * @param proto The proto to validate
     * @throws std::invalid_argument if any field is invalid
     */
    static void validate(const PointTargetDefinition& proto);

  private:
    const PointTargetDefinition& proto_;
};

/**
 * @brief View wrapper for MeshTargetDefinition proto.
 *
 * Provides convenient accessors for mesh target geometry including
 * vertices, faces, transforms, and surface properties.
 * Transforms (position/rotation) are applied to convert from local to world space.
 */
class MeshTargetView {
  public:
    /**
     * @brief Construct a view wrapping a MeshTargetDefinition proto.
     * @param proto The proto to wrap (must outlive this view)
     */
    explicit MeshTargetView(const MeshTargetDefinition& proto);

    /**
     * @brief Get the underlying proto.
     */
    const MeshTargetDefinition& proto() const { return proto_; }

    // Geometry counts
    size_t num_vertices() const;
    size_t num_faces() const;

    /**
     * @brief Get vertex position in world space.
     * Applies mesh transform (rotation then translation).
     */
    Eigen::Vector3d vertex_position(size_t index) const;

    /**
     * @brief Get face centroid in world space.
     */
    Eigen::Vector3d face_centroid(size_t index) const;

    /**
     * @brief Get face normal in world space.
     * Uses explicit face_normal if provided, otherwise computes from vertices.
     */
    Eigen::Vector3d face_normal(size_t index) const;

    /**
     * @brief Compute face area in square meters.
     */
    double face_area(size_t index) const;

    /**
     * @brief Check if face is a triangle (3 vertices).
     */
    bool is_triangle(size_t index) const;

    /**
     * @brief Check if face is a quad (4 vertices).
     */
    bool is_quad(size_t index) const;

    // Bounding geometry

    /**
     * @brief Get mesh centroid in world space.
     */
    Eigen::Vector3d centroid() const;

    /**
     * @brief Get bounding sphere radius for quick culling.
     */
    double bounding_radius() const;

    /**
     * @brief Get range from origin to mesh centroid.
     */
    double range_m() const;

    // Velocity

    /**
     * @brief Get mesh velocity.
     */
    Eigen::Vector3d velocity() const;

    /**
     * @brief Compute radial velocity (toward/away from origin).
     */
    double radial_velocity_mps() const;

    // Surface properties

    /**
     * @brief Get default surface properties.
     */
    SurfacePropertiesView surface() const;

    /**
     * @brief Get surface properties for a specific face.
     * Returns override_surface if set, otherwise default surface.
     */
    SurfacePropertiesView face_surface(size_t index) const;

    /**
     * @brief Validate a MeshTargetDefinition proto.
     * @throws std::invalid_argument if any field is invalid
     */
    static void validate(const MeshTargetDefinition& proto);

  private:
    const MeshTargetDefinition& proto_;
    Eigen::Matrix3d rotation_matrix_;  // Cached from euler angles

    // Compute rotation matrix from euler angles (XYZ order)
    static Eigen::Matrix3d compute_rotation_matrix(const aurora::shared::Vector3& euler_rad);

    // Compute normal from triangle vertices
    Eigen::Vector3d compute_triangle_normal(size_t face_index) const;

    // Compute area of triangle
    double compute_triangle_area(size_t face_index) const;

    // Compute area of quad (split into two triangles)
    double compute_quad_area(size_t face_index) const;
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_VIEWS_H_
