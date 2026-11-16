#ifndef AURORA_SIMULATION_VISIBILITY_CHECKER_H_
#define AURORA_SIMULATION_VISIBILITY_CHECKER_H_

/**
 * @file visibility_checker.h
 * @brief Determines target visibility based on FOV, surface orientation, and occlusion.
 *
 * The VisibilityChecker class provides visibility determination for targets in a scene,
 * considering:
 * - Field of view (FOV) transmission from the PupilModel
 * - Back-face culling based on surface orientation
 * - Occlusion modes (NONE, BOUNDING_SPHERE, RAY_CAST)
 *
 * Supported occlusion modes:
 * - OCCLUSION_NONE: No occlusion checking (fastest)
 * - OCCLUSION_BOUNDING_SPHERE: Fast approximate occlusion using bounding spheres
 * - OCCLUSION_RAY_CAST: Accurate per-facet ray-cast occlusion
 *
 * @see PupilModel for FOV and transmission computation
 * @see Scene for target storage
 * @see OcclusionMode enum in sim_config.proto
 */

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "sim_config.pb.h"
#include "simulation/pupil_model.h"
#include "simulation/scene.h"

namespace aurora::simulation {

/**
 * @brief Determines target visibility considering FOV, surface orientation, and occlusion.
 *
 * Uses OcclusionMode enum from sim_config.proto directly.
 */
class VisibilityChecker {
  public:
    /**
     * @brief Information about a visible target.
     */
    struct VisibleTarget {
        size_t target_index;                   ///< Index of the target in the scene
        double transmission;                   ///< Combined FOV + occlusion factor [0, 1]
        std::vector<size_t> visible_faces;     ///< For meshes: which faces are visible
    };

    /**
     * @brief Construct a visibility checker.
     *
     * @param scene The scene containing targets
     * @param mode Occlusion mode (must not be OCCLUSION_UNSPECIFIED)
     * @throws std::invalid_argument if mode is OCCLUSION_UNSPECIFIED
     */
    VisibilityChecker(const Scene& scene, OcclusionMode mode);

    /**
     * @brief Check if a surface is front-facing (visible from origin).
     *
     * A surface is front-facing if its normal points toward the viewer.
     * The viewer is assumed to be at the origin looking along +z.
     *
     * @param surface_normal The surface normal vector (should be normalized)
     * @param surface_position The surface position (to compute view direction)
     * @return true if dot(normal, -view_direction) > 0
     */
    static bool is_front_facing(const Eigen::Vector3d& surface_normal,
                                const Eigen::Vector3d& surface_position);

    /**
     * @brief Get all visible point targets.
     *
     * Point targets are always visible if in FOV (no surface normal for back-face
     * culling). Applies FOV transmission check only.
     *
     * @param pupil The pupil model for FOV computation
     * @return Vector of visible targets with their transmission values
     */
    std::vector<VisibleTarget> compute_visible_points(const PupilModel& pupil) const;

    /**
     * @brief Get all visible plane targets.
     *
     * Applies FOV check and back-face culling. Plane targets facing away from
     * the origin (back-facing) are not visible.
     *
     * @param pupil The pupil model for FOV computation
     * @return Vector of visible targets with their transmission values
     */
    std::vector<VisibleTarget> compute_visible_planes(const PupilModel& pupil) const;

    /**
     * @brief Get all visible mesh targets with per-face visibility.
     *
     * Applies FOV check on mesh centroid, back-face culling per face, and
     * occlusion checking (when mode is OCCLUSION_BOUNDING_SPHERE).
     *
     * @param pupil The pupil model for FOV computation
     * @return Vector of visible targets with their transmission values and visible faces
     */
    std::vector<VisibleTarget> compute_visible_meshes(const PupilModel& pupil) const;

    /**
     * @brief Get the occlusion mode.
     * @return The configured occlusion mode
     */
    OcclusionMode mode() const { return mode_; }

  private:
    const Scene& scene_;
    OcclusionMode mode_;

    /// Minimum transmission threshold below which targets are considered not visible
    static constexpr double kTransmissionThreshold = 0.01;

    // =========================================================================
    // Bounding sphere occlusion check methods
    // =========================================================================

    /**
     * @brief Check if a point is occluded by any closer target (plane or mesh).
     *
     * Uses bounding sphere approximation for fast occlusion testing.
     *
     * @param point The point position to check
     * @return true if point is occluded by any closer target
     */
    bool is_point_occluded(const Eigen::Vector3d& point) const;

    /**
     * @brief Check if a plane is occluded by any closer mesh.
     *
     * Uses bounding sphere approximation. Planes can be occluded by meshes.
     *
     * @param plane_center The plane center position
     * @param plane_bounding_radius Bounding radius of the plane
     * @return true if plane is occluded by any closer mesh
     */
    bool is_plane_occluded(const Eigen::Vector3d& plane_center,
                           double plane_bounding_radius) const;

    /**
     * @brief Check if a mesh face is occluded by any closer target.
     *
     * Uses bounding sphere approximation.
     *
     * @param face_center The face centroid position
     * @param mesh_index Index of the mesh containing this face (to avoid self-occlusion)
     * @return true if face is occluded by any closer target
     */
    bool is_face_occluded(const Eigen::Vector3d& face_center,
                          size_t mesh_index) const;

    /**
     * @brief Check if a target is occluded by a plane's bounding sphere.
     *
     * @param target_position Position of the target being checked
     * @param target_range Range of the target from origin
     * @param target_dir Normalized direction to target from origin
     * @param plane PlaneTargetView of the potential occluder
     * @return true if target is within plane's shadow cone
     */
    bool is_occluded_by_plane_sphere(
        const Eigen::Vector3d& target_position,
        double target_range,
        const Eigen::Vector3d& target_dir,
        const PlaneTargetView& plane) const;

    /**
     * @brief Check if a target is occluded by a mesh's bounding sphere.
     *
     * @param target_position Position of the target being checked
     * @param target_range Range of the target from origin
     * @param target_dir Normalized direction to target from origin
     * @param mesh MeshTargetView of the potential occluder
     * @return true if target is within mesh's shadow cone
     */
    bool is_occluded_by_mesh_sphere(
        const Eigen::Vector3d& target_position,
        double target_range,
        const Eigen::Vector3d& target_dir,
        const MeshTargetView& mesh) const;

    // =========================================================================
    // Ray-cast occlusion check methods
    // =========================================================================

    /**
     * @brief Result of a ray intersection test.
     */
    struct RayHit {
        bool hit;           ///< Whether intersection occurred
        double t;           ///< Distance along ray to intersection point
        Eigen::Vector3d point;  ///< Intersection point in world space
    };

    /**
     * @brief Compute ray-triangle intersection using Möller–Trumbore algorithm.
     *
     * @param ray_origin Origin of the ray (typically origin)
     * @param ray_dir Normalized ray direction
     * @param v0 First triangle vertex
     * @param v1 Second triangle vertex
     * @param v2 Third triangle vertex
     * @return RayHit with intersection result
     */
    static RayHit ray_triangle_intersection(
        const Eigen::Vector3d& ray_origin,
        const Eigen::Vector3d& ray_dir,
        const Eigen::Vector3d& v0,
        const Eigen::Vector3d& v1,
        const Eigen::Vector3d& v2);

    /**
     * @brief Compute ray-plane intersection for bounded rectangular plane.
     *
     * @param ray_origin Origin of the ray
     * @param ray_dir Normalized ray direction
     * @param plane PlaneTargetView for the plane
     * @return RayHit with intersection result
     */
    static RayHit ray_plane_intersection(
        const Eigen::Vector3d& ray_origin,
        const Eigen::Vector3d& ray_dir,
        const PlaneTargetView& plane);

    /**
     * @brief Check if a point is occluded using ray-cast method.
     *
     * Traces a ray from origin to the point and checks for intersections
     * with other geometry (planes and mesh faces).
     *
     * @param point The point position to check
     * @return true if point is occluded by any geometry
     */
    bool is_point_occluded_raycast(const Eigen::Vector3d& point) const;

    /**
     * @brief Check if a plane is occluded using ray-cast method.
     *
     * Traces a ray from origin to the plane center and checks for intersections
     * with mesh faces.
     *
     * @param plane_center The plane center position
     * @return true if plane is occluded by any mesh
     */
    bool is_plane_occluded_raycast(const Eigen::Vector3d& plane_center) const;

    /**
     * @brief Check if a mesh face is occluded using ray-cast method.
     *
     * Traces a ray from origin to the face centroid and checks for intersections
     * with other geometry, excluding only this exact face while allowing
     * self-occlusion from other faces of the same mesh.
     *
     * @param face_center The face centroid position
     * @param mesh_index Index of the mesh containing this face
     * @param face_index Index of the face within the mesh
     * @return true if face is occluded by any other geometry
     */
    bool is_face_occluded_raycast(
        const Eigen::Vector3d& face_center,
        size_t mesh_index,
        size_t face_index) const;
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_VISIBILITY_CHECKER_H_
