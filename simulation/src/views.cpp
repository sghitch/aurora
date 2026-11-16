#include "simulation/views.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <sstream>

namespace aurora::simulation {

namespace {
constexpr double kWavelength905nm = 905e-9;
constexpr double kWavelength1550nm = 1550e-9;
}  // namespace

// =============================================================================
// SurfacePropertiesView
// =============================================================================

SurfacePropertiesView::SurfacePropertiesView(const SurfaceProperties& proto)
    : proto_(proto) {
    validate(proto_);
}

void SurfacePropertiesView::validate(const SurfaceProperties& proto) {
    if (proto.reflectivity_905nm() <= 0 || proto.reflectivity_905nm() > 1) {
        throw std::invalid_argument(
            "reflectivity_905nm must be in (0, 1]");
    }
    if (proto.reflectivity_1550nm() <= 0 || proto.reflectivity_1550nm() > 1) {
        throw std::invalid_argument(
            "reflectivity_1550nm must be in (0, 1]");
    }
    if (proto.roughness_rms_m() < 0) {
        throw std::invalid_argument(
            "roughness_rms_m must be non-negative");
    }
    if (proto.correlation_length_m() <= 0) {
        throw std::invalid_argument(
            "correlation_length_m must be positive");
    }
}

double SurfacePropertiesView::intrinsic_reflectivity(double wavelength_m) const {
    double t = (wavelength_m - kWavelength905nm) /
               (kWavelength1550nm - kWavelength905nm);
    t = std::clamp(t, 0.0, 1.0);

    return proto_.reflectivity_905nm() +
           t * (proto_.reflectivity_1550nm() - proto_.reflectivity_905nm());
}

double SurfacePropertiesView::specular_fraction(double wavelength_m) const {
    if (proto_.has_specular_fraction_override()) {
        return proto_.specular_fraction_override();
    }
    // Rayleigh criterion: α² = exp(-(4πσ/λ)²)
    double x = 4.0 * std::numbers::pi * proto_.roughness_rms_m() / wavelength_m;
    return std::exp(-x * x);
}

// =============================================================================
// PlaneTargetView
// =============================================================================

PlaneTargetView::PlaneTargetView(const PlaneTargetDefinition& proto)
    : proto_(proto) {
    validate(proto_);
}

double PlaneTargetView::area_m2() const {
    return proto_.width_m() * proto_.height_m();
}

double PlaneTargetView::range_m() const {
    return center().norm();
}

Eigen::Vector3d PlaneTargetView::center() const {
    return aurora::shared::to_eigen(proto_.center_m());
}

Eigen::Vector3d PlaneTargetView::normal() const {
    return aurora::shared::to_eigen(proto_.normal());
}

Eigen::Vector3d PlaneTargetView::velocity() const {
    return aurora::shared::to_eigen(proto_.velocity_mps());
}

SurfacePropertiesView PlaneTargetView::surface() const {
    return SurfacePropertiesView(proto_.surface());
}

void PlaneTargetView::validate(const PlaneTargetDefinition& proto) {
    if (!proto.has_surface()) {
        throw std::invalid_argument(
            "PlaneTargetDefinition must have surface properties");
    }
    SurfacePropertiesView::validate(proto.surface());

    if (proto.width_m() <= 0) {
        throw std::invalid_argument("width_m must be positive");
    }
    if (proto.height_m() <= 0) {
        throw std::invalid_argument("height_m must be positive");
    }

    // Validate normal vector is not zero
    Eigen::Vector3d n = aurora::shared::to_eigen(proto.normal());
    if (n.norm() < 1e-10) {
        throw std::invalid_argument("normal vector must be non-zero");
    }
}

// =============================================================================
// PointTargetView
// =============================================================================

PointTargetView::PointTargetView(const PointTargetDefinition& proto)
    : proto_(proto) {
    validate(proto_);
}

double PointTargetView::range_m() const {
    return position().norm();
}

double PointTargetView::radial_velocity_mps() const {
    double range = range_m();
    if (range < 1e-10) {
        return 0.0;
    }

    // Unit vector from origin to target
    Eigen::Vector3d unit_to_target = position() / range;

    // Radial velocity is projection of velocity onto the line-of-sight
    // Positive when moving away from origin
    return velocity().dot(unit_to_target);
}

Eigen::Vector3d PointTargetView::position() const {
    return aurora::shared::to_eigen(proto_.position_m());
}

Eigen::Vector3d PointTargetView::velocity() const {
    return aurora::shared::to_eigen(proto_.velocity_mps());
}

SurfacePropertiesView PointTargetView::surface() const {
    return SurfacePropertiesView(proto_.surface());
}

void PointTargetView::validate(const PointTargetDefinition& proto) {
    if (!proto.has_surface()) {
        throw std::invalid_argument(
            "PointTargetDefinition must have surface properties");
    }
    SurfacePropertiesView::validate(proto.surface());
}

// =============================================================================
// MeshTargetView
// =============================================================================

MeshTargetView::MeshTargetView(const MeshTargetDefinition& proto)
    : proto_(proto) {
    validate(proto);
    rotation_matrix_ = compute_rotation_matrix(proto.rotation_rad());
}

size_t MeshTargetView::num_vertices() const {
    return static_cast<size_t>(proto_.vertices_size());
}

size_t MeshTargetView::num_faces() const {
    return static_cast<size_t>(proto_.faces_size());
}

Eigen::Matrix3d MeshTargetView::compute_rotation_matrix(
    const aurora::shared::Vector3& euler_rad) {
    // Euler angles XYZ order (roll, pitch, yaw)
    double rx = euler_rad.x();
    double ry = euler_rad.y();
    double rz = euler_rad.z();

    Eigen::Matrix3d Rx, Ry, Rz;
    Rx = Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX());
    Ry = Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY());
    Rz = Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ());

    return Rz * Ry * Rx;  // XYZ order applied right-to-left
}

Eigen::Vector3d MeshTargetView::vertex_position(size_t index) const {
    const auto& v = proto_.vertices(static_cast<int>(index));
    Eigen::Vector3d local = aurora::shared::to_eigen(v.position_m());

    // Apply rotation then translation
    Eigen::Vector3d translation = aurora::shared::to_eigen(proto_.position_m());

    return rotation_matrix_ * local + translation;
}

Eigen::Vector3d MeshTargetView::face_centroid(size_t index) const {
    const auto& face = proto_.faces(static_cast<int>(index));
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();

    for (int i = 0; i < face.vertex_indices_size(); ++i) {
        sum += vertex_position(face.vertex_indices(i));
    }

    return sum / face.vertex_indices_size();
}

Eigen::Vector3d MeshTargetView::face_normal(size_t index) const {
    const auto& face = proto_.faces(static_cast<int>(index));

    // Use explicit normal if provided
    if (face.has_face_normal()) {
        Eigen::Vector3d local_normal = aurora::shared::to_eigen(face.face_normal());
        // Rotate normal to world space (normals transform differently)
        return (rotation_matrix_ * local_normal).normalized();
    }

    const int vertex_count = face.vertex_indices_size();

    // Compute from vertices
    if (vertex_count == 3) {
        // Triangle face: use existing triangle normal computation
        return compute_triangle_normal(index);
    }

    if (vertex_count == 4) {
        // Quad face: compute normal using all four vertices by averaging two
        // triangle normals (0,1,2) and (0,2,3). This is more accurate for
        // non-planar quads than using only the first three vertices.
        Eigen::Vector3d v0 = vertex_position(face.vertex_indices(0));
        Eigen::Vector3d v1 = vertex_position(face.vertex_indices(1));
        Eigen::Vector3d v2 = vertex_position(face.vertex_indices(2));
        Eigen::Vector3d v3 = vertex_position(face.vertex_indices(3));

        // Triangle (v0, v1, v2)
        Eigen::Vector3d n0 = (v1 - v0).cross(v2 - v0);
        // Triangle (v0, v2, v3)
        Eigen::Vector3d n1 = (v2 - v0).cross(v3 - v0);

        Eigen::Vector3d n = n0 + n1;
        if (n.squaredNorm() > 0.0) {
            return n.normalized();
        }

        // Degenerate case: fall back to triangle-based normal
        return compute_triangle_normal(index);
    }

    // Fallback for other vertex counts: preserve existing behavior
    return compute_triangle_normal(index);
}

Eigen::Vector3d MeshTargetView::compute_triangle_normal(size_t face_index) const {
    const auto& face = proto_.faces(static_cast<int>(face_index));

    Eigen::Vector3d v0 = vertex_position(face.vertex_indices(0));
    Eigen::Vector3d v1 = vertex_position(face.vertex_indices(1));
    Eigen::Vector3d v2 = vertex_position(face.vertex_indices(2));

    // Right-hand rule: (v1-v0) x (v2-v0)
    Eigen::Vector3d edge1 = v1 - v0;
    Eigen::Vector3d edge2 = v2 - v0;

    return edge1.cross(edge2).normalized();
}

double MeshTargetView::face_area(size_t index) const {
    if (is_triangle(index)) {
        return compute_triangle_area(index);
    } else {
        return compute_quad_area(index);
    }
}

double MeshTargetView::compute_triangle_area(size_t face_index) const {
    const auto& face = proto_.faces(static_cast<int>(face_index));

    Eigen::Vector3d v0 = vertex_position(face.vertex_indices(0));
    Eigen::Vector3d v1 = vertex_position(face.vertex_indices(1));
    Eigen::Vector3d v2 = vertex_position(face.vertex_indices(2));

    // Area = 0.5 * |cross product|
    return 0.5 * (v1 - v0).cross(v2 - v0).norm();
}

double MeshTargetView::compute_quad_area(size_t face_index) const {
    const auto& face = proto_.faces(static_cast<int>(face_index));

    Eigen::Vector3d v0 = vertex_position(face.vertex_indices(0));
    Eigen::Vector3d v1 = vertex_position(face.vertex_indices(1));
    Eigen::Vector3d v2 = vertex_position(face.vertex_indices(2));
    Eigen::Vector3d v3 = vertex_position(face.vertex_indices(3));

    // Split into two triangles along diagonal (0,2): triangles (0,1,2) and (0,2,3).
    // This produces accurate results for planar, convex quads. For non-planar
    // or concave quads, the result is an approximation.
    double area1 = 0.5 * (v1 - v0).cross(v2 - v0).norm();
    double area2 = 0.5 * (v2 - v0).cross(v3 - v0).norm();

    return area1 + area2;
}

bool MeshTargetView::is_triangle(size_t index) const {
    return proto_.faces(static_cast<int>(index)).vertex_indices_size() == 3;
}

bool MeshTargetView::is_quad(size_t index) const {
    return proto_.faces(static_cast<int>(index)).vertex_indices_size() == 4;
}

Eigen::Vector3d MeshTargetView::centroid() const {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < num_vertices(); ++i) {
        sum += vertex_position(i);
    }
    return sum / num_vertices();
}

double MeshTargetView::bounding_radius() const {
    Eigen::Vector3d c = centroid();
    double max_dist_sq = 0;
    for (size_t i = 0; i < num_vertices(); ++i) {
        double dist_sq = (vertex_position(i) - c).squaredNorm();
        max_dist_sq = std::max(max_dist_sq, dist_sq);
    }
    return std::sqrt(max_dist_sq);
}

double MeshTargetView::range_m() const {
    return centroid().norm();
}

Eigen::Vector3d MeshTargetView::velocity() const {
    return aurora::shared::to_eigen(proto_.velocity_mps());
}

double MeshTargetView::radial_velocity_mps() const {
    Eigen::Vector3d c = centroid();
    double range = c.norm();
    if (range < 1e-10) {
        return 0.0;
    }
    Eigen::Vector3d v = velocity();
    Eigen::Vector3d dir = c / range;
    return v.dot(dir);
}

SurfacePropertiesView MeshTargetView::surface() const {
    return SurfacePropertiesView(proto_.surface());
}

SurfacePropertiesView MeshTargetView::face_surface(size_t index) const {
    const auto& face = proto_.faces(static_cast<int>(index));
    if (face.has_override_surface()) {
        return SurfacePropertiesView(face.override_surface());
    }
    return surface();
}

void MeshTargetView::validate(const MeshTargetDefinition& proto) {
    if (proto.vertices_size() < 3) {
        throw std::invalid_argument("Mesh must have at least 3 vertices");
    }

    if (proto.faces_size() < 1) {
        throw std::invalid_argument("Mesh must have at least 1 face");
    }

    for (int fi = 0; fi < proto.faces_size(); ++fi) {
        const auto& face = proto.faces(fi);
        int num_indices = face.vertex_indices_size();

        if (num_indices < 3 || num_indices > 4) {
            throw std::invalid_argument(
                "Face " + std::to_string(fi) + " must have 3 or 4 vertices, got " +
                std::to_string(num_indices));
        }

        for (int vi = 0; vi < num_indices; ++vi) {
            if (face.vertex_indices(vi) >= static_cast<uint32_t>(proto.vertices_size())) {
                throw std::invalid_argument(
                    "Face " + std::to_string(fi) + " vertex index " +
                    std::to_string(face.vertex_indices(vi)) + " out of range");
            }
        }

        if (face.has_override_surface()) {
            SurfacePropertiesView::validate(face.override_surface());
        }
    }

    if (!proto.has_surface()) {
        throw std::invalid_argument(
            "MeshTargetDefinition must have surface properties");
    }
    SurfacePropertiesView::validate(proto.surface());
}

}  // namespace aurora::simulation
