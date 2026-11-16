#include "simulation/visibility_checker.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "simulation/views.h"

namespace aurora::simulation {

VisibilityChecker::VisibilityChecker(const Scene& scene, OcclusionMode mode)
    : scene_(scene), mode_(mode) {
    if (mode == OCCLUSION_UNSPECIFIED) {
        throw std::invalid_argument(
            "VisibilityChecker: mode must not be OCCLUSION_UNSPECIFIED");
    }
}

bool VisibilityChecker::is_front_facing(const Eigen::Vector3d& surface_normal,
                                        const Eigen::Vector3d& surface_position) {
    // View direction is from surface to origin (camera at origin)
    Eigen::Vector3d view_direction = -surface_position.normalized();

    // Surface is front-facing if normal points toward viewer
    return surface_normal.dot(view_direction) > 0;
}

std::vector<VisibilityChecker::VisibleTarget>
VisibilityChecker::compute_visible_points(const PupilModel& pupil) const {
    std::vector<VisibleTarget> visible;
    const auto& targets = scene_.get_point_targets();

    for (size_t i = 0; i < targets.size(); ++i) {
        PointTargetView target(targets[i]);

        double transmission = pupil.compute_transmission(target.position());
        if (transmission < kTransmissionThreshold) {
            continue;
        }

        // Occlusion check based on mode
        if (mode_ == OCCLUSION_BOUNDING_SPHERE) {
            if (is_point_occluded(target.position())) {
                continue;
            }
        } else if (mode_ == OCCLUSION_RAY_CAST) {
            if (is_point_occluded_raycast(target.position())) {
                continue;
            }
        }

        // Point targets have no surface normal — always "front-facing"
        visible.push_back({
            .target_index = i,
            .transmission = transmission,
            .visible_faces = {}});
    }

    return visible;
}

std::vector<VisibilityChecker::VisibleTarget>
VisibilityChecker::compute_visible_planes(const PupilModel& pupil) const {
    std::vector<VisibleTarget> visible;
    const auto& targets = scene_.get_plane_targets();

    for (size_t i = 0; i < targets.size(); ++i) {
        PlaneTargetView target(targets[i]);

        // FOV check
        double transmission = pupil.compute_transmission(target.center());
        if (transmission < kTransmissionThreshold) {
            continue;
        }

        // Back-face culling: skip if surface faces away from origin
        if (!is_front_facing(target.normal(), target.center())) {
            continue;
        }

        // Occlusion check based on mode (planes can be occluded by meshes)
        if (mode_ == OCCLUSION_BOUNDING_SPHERE) {
            // Approximate plane as sphere with radius = half-diagonal
            double plane_radius = 0.5 * std::hypot(target.width_m(), target.height_m());
            if (is_plane_occluded(target.center(), plane_radius)) {
                continue;
            }
        } else if (mode_ == OCCLUSION_RAY_CAST) {
            if (is_plane_occluded_raycast(target.center())) {
                continue;
            }
        }

        visible.push_back({
            .target_index = i,
            .transmission = transmission,
            .visible_faces = {}});
    }

    return visible;
}

std::vector<VisibilityChecker::VisibleTarget>
VisibilityChecker::compute_visible_meshes(const PupilModel& pupil) const {
    std::vector<VisibleTarget> visible;
    const auto& targets = scene_.get_mesh_targets();

    for (size_t mesh_idx = 0; mesh_idx < targets.size(); ++mesh_idx) {
        MeshTargetView mesh(targets[mesh_idx]);

        // Quick FOV check on mesh centroid
        double transmission = pupil.compute_transmission(mesh.centroid());
        if (transmission < kTransmissionThreshold) {
            continue;
        }

        // Collect visible faces (front-facing and not occluded)
        std::vector<size_t> visible_faces;
        for (size_t face_idx = 0; face_idx < mesh.num_faces(); ++face_idx) {
            Eigen::Vector3d face_center = mesh.face_centroid(face_idx);
            Eigen::Vector3d face_normal = mesh.face_normal(face_idx);

            if (!is_front_facing(face_normal, face_center)) {
                continue;
            }

            // Occlusion check based on mode for mesh faces
            if (mode_ == OCCLUSION_BOUNDING_SPHERE) {
                if (is_face_occluded(face_center, mesh_idx)) {
                    continue;
                }
            } else if (mode_ == OCCLUSION_RAY_CAST) {
                if (is_face_occluded_raycast(face_center, mesh_idx, face_idx)) {
                    continue;
                }
            }

            visible_faces.push_back(face_idx);
        }

        if (visible_faces.empty()) {
            continue;  // All faces back-facing or occluded
        }

        visible.push_back({
            .target_index = mesh_idx,
            .transmission = transmission,
            .visible_faces = std::move(visible_faces)});
    }

    return visible;
}

// =============================================================================
// Bounding sphere occlusion check implementations
// =============================================================================

bool VisibilityChecker::is_point_occluded(const Eigen::Vector3d& point) const {
    double point_range = point.norm();
    if (point_range < 1e-10) {
        return false;  // Point at origin can't be occluded
    }
    Eigen::Vector3d point_dir = point / point_range;

    // Check against plane targets
    for (const auto& plane_proto : scene_.get_plane_targets()) {
        PlaneTargetView plane(plane_proto);
        if (is_occluded_by_plane_sphere(point, point_range, point_dir, plane)) {
            return true;
        }
    }

    // Check against mesh targets
    for (const auto& mesh_proto : scene_.get_mesh_targets()) {
        MeshTargetView mesh(mesh_proto);
        if (is_occluded_by_mesh_sphere(point, point_range, point_dir, mesh)) {
            return true;
        }
    }

    return false;
}

bool VisibilityChecker::is_plane_occluded(const Eigen::Vector3d& plane_center,
                                          double plane_bounding_radius) const {
    double plane_range = plane_center.norm();
    if (plane_range < 1e-10) {
        return false;
    }
    Eigen::Vector3d plane_dir = plane_center / plane_range;

    // Planes can only be occluded by meshes (not by other planes in this mode)
    for (const auto& mesh_proto : scene_.get_mesh_targets()) {
        MeshTargetView mesh(mesh_proto);

        double mesh_range = mesh.range_m();
        if (mesh_range < 1e-10) {
            continue;  // Degenerate or extremely close mesh; skip to avoid numerical issues
        }
        if (mesh_range >= plane_range) {
            continue;  // Mesh is behind plane
        }

        Eigen::Vector3d mesh_center = mesh.centroid();
        double mesh_radius = mesh.bounding_radius();
        Eigen::Vector3d mesh_dir = mesh_center / mesh_range;

        // Angular separation
        double cos_angle = plane_dir.dot(mesh_dir);
        double angle = std::acos(std::clamp(cos_angle, -1.0, 1.0));

        // Angular radii
        double mesh_angular_radius = std::atan(mesh_radius / mesh_range);
        double plane_angular_radius = std::atan(plane_bounding_radius / plane_range);

        // Plane is occluded if within mesh's shadow cone (considering plane's size)
        if (angle < mesh_angular_radius + plane_angular_radius) {
            return true;
        }
    }

    return false;
}

bool VisibilityChecker::is_face_occluded(const Eigen::Vector3d& face_center,
                                         size_t mesh_index) const {
    double face_range = face_center.norm();
    if (face_range < 1e-10) {
        return false;
    }
    Eigen::Vector3d face_dir = face_center / face_range;

    // Check against plane targets
    for (const auto& plane_proto : scene_.get_plane_targets()) {
        PlaneTargetView plane(plane_proto);
        if (is_occluded_by_plane_sphere(face_center, face_range, face_dir, plane)) {
            return true;
        }
    }

    // Check against mesh targets (excluding self)
    const auto& mesh_targets = scene_.get_mesh_targets();
    for (size_t i = 0; i < mesh_targets.size(); ++i) {
        if (i == mesh_index) {
            continue;  // Skip self-occlusion check
        }
        MeshTargetView mesh(mesh_targets[i]);
        if (is_occluded_by_mesh_sphere(face_center, face_range, face_dir, mesh)) {
            return true;
        }
    }

    return false;
}

bool VisibilityChecker::is_occluded_by_plane_sphere(
    const Eigen::Vector3d& /* target_position */,
    double target_range,
    const Eigen::Vector3d& target_dir,
    const PlaneTargetView& plane) const {
    double plane_range = plane.range_m();
    if (plane_range < 1e-10) {
        return false;  // Plane at origin; skip to avoid numerical issues
    }
    if (plane_range >= target_range) {
        return false;  // Plane is behind target
    }

    // Approximate plane as sphere with radius = half-diagonal
    double plane_radius = 0.5 * std::hypot(plane.width_m(), plane.height_m());

    Eigen::Vector3d plane_center = plane.center();
    Eigen::Vector3d plane_dir = plane_center / plane_range;

    // Angular separation between target and plane center
    double cos_angle = target_dir.dot(plane_dir);
    double angle = std::acos(std::clamp(cos_angle, -1.0, 1.0));

    // Angular radius of plane's bounding sphere from origin
    double plane_angular_radius = std::atan(plane_radius / plane_range);

    // Target is occluded if within plane's shadow cone
    return angle < plane_angular_radius;
}

bool VisibilityChecker::is_occluded_by_mesh_sphere(
    const Eigen::Vector3d& /* target_position */,
    double target_range,
    const Eigen::Vector3d& target_dir,
    const MeshTargetView& mesh) const {
    double mesh_range = mesh.range_m();
    if (mesh_range < 1e-10) {
        return false;  // Mesh at origin; skip to avoid numerical issues
    }
    if (mesh_range >= target_range) {
        return false;  // Mesh is behind target
    }

    Eigen::Vector3d mesh_center = mesh.centroid();
    double mesh_radius = mesh.bounding_radius();
    Eigen::Vector3d mesh_dir = mesh_center / mesh_range;

    // Angular separation between target and mesh center
    double cos_angle = target_dir.dot(mesh_dir);
    double angle = std::acos(std::clamp(cos_angle, -1.0, 1.0));

    // Angular radius of mesh's bounding sphere from origin
    double mesh_angular_radius = std::atan(mesh_radius / mesh_range);

    // Target is occluded if within mesh's shadow cone
    return angle < mesh_angular_radius;
}

// =============================================================================
// Ray-cast occlusion check implementations
// =============================================================================

VisibilityChecker::RayHit VisibilityChecker::ray_triangle_intersection(
    const Eigen::Vector3d& ray_origin,
    const Eigen::Vector3d& ray_dir,
    const Eigen::Vector3d& v0,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2) {
    RayHit result{false, 0.0, Eigen::Vector3d::Zero()};

    constexpr double kEpsilon = 1e-10;

    Eigen::Vector3d edge1 = v1 - v0;
    Eigen::Vector3d edge2 = v2 - v0;
    Eigen::Vector3d h = ray_dir.cross(edge2);
    double a = edge1.dot(h);

    if (std::abs(a) < kEpsilon) {
        return result;  // Ray parallel to triangle
    }

    double f = 1.0 / a;
    Eigen::Vector3d s = ray_origin - v0;
    double u = f * s.dot(h);

    if (u < 0.0 || u > 1.0) {
        return result;
    }

    Eigen::Vector3d q = s.cross(edge1);
    double v = f * ray_dir.dot(q);

    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    double t = f * edge2.dot(q);

    if (t > kEpsilon) {
        result.hit = true;
        result.t = t;
        result.point = ray_origin + t * ray_dir;
    }

    return result;
}

VisibilityChecker::RayHit VisibilityChecker::ray_plane_intersection(
    const Eigen::Vector3d& ray_origin,
    const Eigen::Vector3d& ray_dir,
    const PlaneTargetView& plane) {
    RayHit result{false, 0.0, Eigen::Vector3d::Zero()};

    constexpr double kEpsilon = 1e-10;

    Eigen::Vector3d plane_normal = plane.normal().normalized();
    Eigen::Vector3d plane_center = plane.center();

    double denom = plane_normal.dot(ray_dir);
    if (std::abs(denom) < kEpsilon) {
        return result;  // Ray parallel to plane
    }

    double t = plane_normal.dot(plane_center - ray_origin) / denom;
    if (t < kEpsilon) {
        return result;  // Intersection behind ray origin
    }

    Eigen::Vector3d hit_point = ray_origin + t * ray_dir;

    // Check if hit point is within plane bounds
    // Create local coordinate system on the plane
    Eigen::Vector3d u_axis, v_axis;
    if (std::abs(plane_normal.x()) < 0.9) {
        u_axis = Eigen::Vector3d::UnitX();
    } else {
        u_axis = Eigen::Vector3d::UnitY();
    }
    u_axis = (u_axis - u_axis.dot(plane_normal) * plane_normal).normalized();
    v_axis = plane_normal.cross(u_axis).normalized();

    // Project hit point to local coordinates
    Eigen::Vector3d local = hit_point - plane_center;
    double local_u = local.dot(u_axis);
    double local_v = local.dot(v_axis);

    double half_width = plane.width_m() / 2.0;
    double half_height = plane.height_m() / 2.0;

    if (std::abs(local_u) <= half_width && std::abs(local_v) <= half_height) {
        result.hit = true;
        result.t = t;
        result.point = hit_point;
    }

    return result;
}

bool VisibilityChecker::is_point_occluded_raycast(
    const Eigen::Vector3d& point) const {
    double target_distance = point.norm();
    if (target_distance < 1e-10) {
        return false;  // Point at origin can't be occluded
    }
    Eigen::Vector3d ray_dir = point / target_distance;
    Eigen::Vector3d ray_origin = Eigen::Vector3d::Zero();

    constexpr double kDistanceEpsilon = 1e-6;

    // Check against plane targets
    for (const auto& plane_proto : scene_.get_plane_targets()) {
        PlaneTargetView plane(plane_proto);

        // Only check front-facing planes as occluders
        if (!is_front_facing(plane.normal(), plane.center())) {
            continue;
        }

        auto hit = ray_plane_intersection(ray_origin, ray_dir, plane);
        if (hit.hit && hit.t < target_distance - kDistanceEpsilon) {
            return true;  // Occluded by plane
        }
    }

    // Check against mesh faces
    for (const auto& mesh_proto : scene_.get_mesh_targets()) {
        MeshTargetView mesh(mesh_proto);

        for (size_t face_idx = 0; face_idx < mesh.num_faces(); ++face_idx) {
            // Only check front-facing faces as occluders
            if (!is_front_facing(mesh.face_normal(face_idx),
                                 mesh.face_centroid(face_idx))) {
                continue;
            }

            const auto& face = mesh_proto.faces(static_cast<int>(face_idx));

            if (mesh.is_triangle(face_idx)) {
                auto hit = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    mesh.vertex_position(face.vertex_indices(0)),
                    mesh.vertex_position(face.vertex_indices(1)),
                    mesh.vertex_position(face.vertex_indices(2)));

                if (hit.hit && hit.t < target_distance - kDistanceEpsilon) {
                    return true;
                }
            } else {
                // Quad: split into two triangles (0,1,2) and (0,2,3)
                auto hit1 = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    mesh.vertex_position(face.vertex_indices(0)),
                    mesh.vertex_position(face.vertex_indices(1)),
                    mesh.vertex_position(face.vertex_indices(2)));

                if (hit1.hit && hit1.t < target_distance - kDistanceEpsilon) {
                    return true;
                }

                auto hit2 = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    mesh.vertex_position(face.vertex_indices(0)),
                    mesh.vertex_position(face.vertex_indices(2)),
                    mesh.vertex_position(face.vertex_indices(3)));

                if (hit2.hit && hit2.t < target_distance - kDistanceEpsilon) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool VisibilityChecker::is_plane_occluded_raycast(
    const Eigen::Vector3d& plane_center) const {
    double target_distance = plane_center.norm();
    if (target_distance < 1e-10) {
        return false;
    }
    Eigen::Vector3d ray_dir = plane_center / target_distance;
    Eigen::Vector3d ray_origin = Eigen::Vector3d::Zero();

    constexpr double kDistanceEpsilon = 1e-6;

    // Planes can only be occluded by mesh faces (not by other planes)
    for (const auto& mesh_proto : scene_.get_mesh_targets()) {
        MeshTargetView mesh(mesh_proto);

        for (size_t face_idx = 0; face_idx < mesh.num_faces(); ++face_idx) {
            // Only check front-facing faces as occluders
            if (!is_front_facing(mesh.face_normal(face_idx),
                                 mesh.face_centroid(face_idx))) {
                continue;
            }

            const auto& face = mesh_proto.faces(static_cast<int>(face_idx));

            if (mesh.is_triangle(face_idx)) {
                auto hit = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    mesh.vertex_position(face.vertex_indices(0)),
                    mesh.vertex_position(face.vertex_indices(1)),
                    mesh.vertex_position(face.vertex_indices(2)));

                if (hit.hit && hit.t < target_distance - kDistanceEpsilon) {
                    return true;
                }
            } else {
                // Quad: split into two triangles
                auto hit1 = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    mesh.vertex_position(face.vertex_indices(0)),
                    mesh.vertex_position(face.vertex_indices(1)),
                    mesh.vertex_position(face.vertex_indices(2)));

                if (hit1.hit && hit1.t < target_distance - kDistanceEpsilon) {
                    return true;
                }

                auto hit2 = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    mesh.vertex_position(face.vertex_indices(0)),
                    mesh.vertex_position(face.vertex_indices(2)),
                    mesh.vertex_position(face.vertex_indices(3)));

                if (hit2.hit && hit2.t < target_distance - kDistanceEpsilon) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool VisibilityChecker::is_face_occluded_raycast(
    const Eigen::Vector3d& face_center,
    size_t mesh_index,
    size_t face_index) const {
    double target_distance = face_center.norm();
    if (target_distance < 1e-10) {
        return false;
    }
    Eigen::Vector3d ray_dir = face_center / target_distance;
    Eigen::Vector3d ray_origin = Eigen::Vector3d::Zero();

    constexpr double kDistanceEpsilon = 1e-6;

    // Check against plane targets
    for (const auto& plane_proto : scene_.get_plane_targets()) {
        PlaneTargetView plane(plane_proto);

        // Only check front-facing planes as occluders
        if (!is_front_facing(plane.normal(), plane.center())) {
            continue;
        }

        auto hit = ray_plane_intersection(ray_origin, ray_dir, plane);
        if (hit.hit && hit.t < target_distance - kDistanceEpsilon) {
            return true;
        }
    }

    // Check against mesh faces (excluding self)
    const auto& mesh_targets = scene_.get_mesh_targets();
    for (size_t other_mesh_idx = 0; other_mesh_idx < mesh_targets.size();
         ++other_mesh_idx) {
        MeshTargetView other_mesh(mesh_targets[other_mesh_idx]);

        for (size_t other_face_idx = 0; other_face_idx < other_mesh.num_faces();
             ++other_face_idx) {
            // Skip self
            if (other_mesh_idx == mesh_index && other_face_idx == face_index) {
                continue;
            }

            // Only check front-facing faces as occluders
            if (!is_front_facing(other_mesh.face_normal(other_face_idx),
                                 other_mesh.face_centroid(other_face_idx))) {
                continue;
            }

            const auto& face = mesh_targets[other_mesh_idx].faces(
                static_cast<int>(other_face_idx));

            if (other_mesh.is_triangle(other_face_idx)) {
                auto hit = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    other_mesh.vertex_position(face.vertex_indices(0)),
                    other_mesh.vertex_position(face.vertex_indices(1)),
                    other_mesh.vertex_position(face.vertex_indices(2)));

                if (hit.hit && hit.t < target_distance - kDistanceEpsilon) {
                    return true;
                }
            } else {
                // Quad: split into two triangles
                auto hit1 = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    other_mesh.vertex_position(face.vertex_indices(0)),
                    other_mesh.vertex_position(face.vertex_indices(1)),
                    other_mesh.vertex_position(face.vertex_indices(2)));

                if (hit1.hit && hit1.t < target_distance - kDistanceEpsilon) {
                    return true;
                }

                auto hit2 = ray_triangle_intersection(
                    ray_origin, ray_dir,
                    other_mesh.vertex_position(face.vertex_indices(0)),
                    other_mesh.vertex_position(face.vertex_indices(2)),
                    other_mesh.vertex_position(face.vertex_indices(3)));

                if (hit2.hit && hit2.t < target_distance - kDistanceEpsilon) {
                    return true;
                }
            }
        }
    }

    return false;
}

}  // namespace aurora::simulation
