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

#include <gtest/gtest.h>
#include "simulation/views.h"

#include <cmath>
#include <numbers>

namespace aurora::simulation {
namespace {

class MeshTargetViewTest : public ::testing::Test {
  protected:
    MeshTargetDefinition make_unit_triangle_at_z100() {
        MeshTargetDefinition mesh;

        // Triangle in XY plane
        auto* v0 = mesh.add_vertices();
        v0->mutable_position_m()->set_x(-0.5);
        v0->mutable_position_m()->set_y(-0.5);

        auto* v1 = mesh.add_vertices();
        v1->mutable_position_m()->set_x(0.5);
        v1->mutable_position_m()->set_y(-0.5);

        auto* v2 = mesh.add_vertices();
        v2->mutable_position_m()->set_x(0.0);
        v2->mutable_position_m()->set_y(0.5);

        auto* face = mesh.add_faces();
        face->add_vertex_indices(0);
        face->add_vertex_indices(1);
        face->add_vertex_indices(2);

        mesh.mutable_position_m()->set_z(100);

        mesh.mutable_surface()->set_reflectivity_905nm(0.5);
        mesh.mutable_surface()->set_reflectivity_1550nm(0.5);
        mesh.mutable_surface()->set_roughness_rms_m(0);
        mesh.mutable_surface()->set_correlation_length_m(0.01);

        return mesh;
    }

    MeshTargetDefinition make_unit_cube_at_origin() {
        MeshTargetDefinition mesh;

        // 8 vertices of unit cube centered at origin
        double h = 0.5;
        std::vector<std::tuple<double, double, double>> verts = {
            {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},  // back face
            {-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h}    // front face
        };

        for (const auto& [x, y, z] : verts) {
            auto* v = mesh.add_vertices();
            v->mutable_position_m()->set_x(x);
            v->mutable_position_m()->set_y(y);
            v->mutable_position_m()->set_z(z);
        }

        // 6 faces (quads)
        std::vector<std::vector<int>> faces = {
            {0, 1, 2, 3},  // back
            {4, 7, 6, 5},  // front
            {0, 4, 5, 1},  // bottom
            {2, 6, 7, 3},  // top
            {0, 3, 7, 4},  // left
            {1, 5, 6, 2}   // right
        };

        for (const auto& f : faces) {
            auto* face = mesh.add_faces();
            for (int idx : f) {
                face->add_vertex_indices(idx);
            }
        }

        mesh.mutable_surface()->set_reflectivity_905nm(0.5);
        mesh.mutable_surface()->set_reflectivity_1550nm(0.5);
        mesh.mutable_surface()->set_roughness_rms_m(0);
        mesh.mutable_surface()->set_correlation_length_m(0.01);

        return mesh;
    }
};

TEST_F(MeshTargetViewTest, ConstructsFromValidProto) {
    auto proto = make_unit_triangle_at_z100();
    EXPECT_NO_THROW(MeshTargetView view(proto));
}

TEST_F(MeshTargetViewTest, RejectsEmptyMesh) {
    MeshTargetDefinition proto;
    proto.mutable_surface()->set_reflectivity_905nm(0.5);
    proto.mutable_surface()->set_reflectivity_1550nm(0.5);
    proto.mutable_surface()->set_roughness_rms_m(0);
    proto.mutable_surface()->set_correlation_length_m(0.01);

    EXPECT_THROW(MeshTargetView view(proto), std::invalid_argument);
}

TEST_F(MeshTargetViewTest, NumVerticesAndFaces) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    EXPECT_EQ(view.num_vertices(), 3);
    EXPECT_EQ(view.num_faces(), 1);
}

TEST_F(MeshTargetViewTest, VertexPositionAppliesTranslation) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    Eigen::Vector3d v0 = view.vertex_position(0);

    EXPECT_NEAR(v0.x(), -0.5, 1e-9);
    EXPECT_NEAR(v0.y(), -0.5, 1e-9);
    EXPECT_NEAR(v0.z(), 100.0, 1e-9);  // Translation applied
}

TEST_F(MeshTargetViewTest, VertexPositionAppliesRotation) {
    MeshTargetDefinition proto;

    auto* v = proto.add_vertices();
    v->mutable_position_m()->set_x(1.0);  // Point on X axis

    proto.add_vertices();  // Need at least 3 vertices
    proto.add_vertices();

    auto* face = proto.add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);

    // Rotate 90 degrees around Z axis
    proto.mutable_rotation_rad()->set_z(std::numbers::pi / 2);

    proto.mutable_surface()->set_reflectivity_905nm(0.5);
    proto.mutable_surface()->set_reflectivity_1550nm(0.5);
    proto.mutable_surface()->set_roughness_rms_m(0);
    proto.mutable_surface()->set_correlation_length_m(0.01);

    MeshTargetView view(proto);
    Eigen::Vector3d v0 = view.vertex_position(0);

    EXPECT_NEAR(v0.x(), 0.0, 1e-9);
    EXPECT_NEAR(v0.y(), 1.0, 1e-9);  // X became Y after 90 deg Z rotation
}

TEST_F(MeshTargetViewTest, FaceNormalComputedFromVertices) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    Eigen::Vector3d normal = view.face_normal(0);

    // Triangle in XY plane, normal should be +Z or -Z
    EXPECT_NEAR(std::abs(normal.z()), 1.0, 1e-9);
    EXPECT_NEAR(normal.x(), 0.0, 1e-9);
    EXPECT_NEAR(normal.y(), 0.0, 1e-9);
}

TEST_F(MeshTargetViewTest, FaceNormalUsesExplicitIfProvided) {
    auto proto = make_unit_triangle_at_z100();
    proto.mutable_faces(0)->mutable_face_normal()->set_x(1.0);
    proto.mutable_faces(0)->mutable_face_normal()->set_y(0.0);
    proto.mutable_faces(0)->mutable_face_normal()->set_z(0.0);

    MeshTargetView view(proto);
    Eigen::Vector3d normal = view.face_normal(0);

    EXPECT_NEAR(normal.x(), 1.0, 1e-9);
}

TEST_F(MeshTargetViewTest, TriangleAreaCalculation) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    // Triangle with base 1.0, height 1.0 -> area = 0.5
    double area = view.face_area(0);
    EXPECT_NEAR(area, 0.5, 1e-9);
}

TEST_F(MeshTargetViewTest, QuadAreaCalculation) {
    auto proto = make_unit_cube_at_origin();
    MeshTargetView view(proto);

    // Each face of unit cube has area 1.0
    double area = view.face_area(0);
    EXPECT_NEAR(area, 1.0, 1e-9);
}

TEST_F(MeshTargetViewTest, IsTriangleAndIsQuad) {
    auto proto = make_unit_cube_at_origin();
    MeshTargetView view(proto);

    EXPECT_FALSE(view.is_triangle(0));
    EXPECT_TRUE(view.is_quad(0));
}

TEST_F(MeshTargetViewTest, Centroid) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    Eigen::Vector3d c = view.centroid();

    EXPECT_NEAR(c.x(), 0.0, 1e-9);
    EXPECT_NEAR(c.y(), (-0.5 - 0.5 + 0.5) / 3.0, 1e-9);
    EXPECT_NEAR(c.z(), 100.0, 1e-9);
}

TEST_F(MeshTargetViewTest, RangeM) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    double range = view.range_m();
    EXPECT_NEAR(range, 100.0, 0.1);  // Approximately 100m
}

TEST_F(MeshTargetViewTest, Velocity) {
    auto proto = make_unit_triangle_at_z100();
    proto.mutable_velocity_mps()->set_x(10);
    proto.mutable_velocity_mps()->set_y(5);
    proto.mutable_velocity_mps()->set_z(-2);

    MeshTargetView view(proto);
    Eigen::Vector3d v = view.velocity();

    EXPECT_DOUBLE_EQ(v.x(), 10);
    EXPECT_DOUBLE_EQ(v.y(), 5);
    EXPECT_DOUBLE_EQ(v.z(), -2);
}

TEST_F(MeshTargetViewTest, RadialVelocity) {
    auto proto = make_unit_triangle_at_z100();
    proto.mutable_velocity_mps()->set_z(5);  // Moving away from origin

    MeshTargetView view(proto);
    double vr = view.radial_velocity_mps();

    EXPECT_NEAR(vr, 5.0, 0.1);  // Moving radially at ~5 m/s
}

TEST_F(MeshTargetViewTest, FaceSurfaceDefault) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    auto surface = view.face_surface(0);
    EXPECT_DOUBLE_EQ(surface.intrinsic_reflectivity(905e-9), 0.5);
}

TEST_F(MeshTargetViewTest, FaceSurfaceOverride) {
    auto proto = make_unit_triangle_at_z100();
    proto.mutable_faces(0)->mutable_override_surface()->set_reflectivity_905nm(0.9);
    proto.mutable_faces(0)->mutable_override_surface()->set_reflectivity_1550nm(0.9);
    proto.mutable_faces(0)->mutable_override_surface()->set_roughness_rms_m(0);
    proto.mutable_faces(0)->mutable_override_surface()->set_correlation_length_m(0.01);

    MeshTargetView view(proto);

    auto surface = view.face_surface(0);
    EXPECT_DOUBLE_EQ(surface.intrinsic_reflectivity(905e-9), 0.9);
}

TEST_F(MeshTargetViewTest, ValidationRejectsInvalidVertexIndex) {
    MeshTargetDefinition proto;

    auto* v = proto.add_vertices();
    v->mutable_position_m()->set_z(0);
    proto.add_vertices();
    proto.add_vertices();

    auto* face = proto.add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(99);  // Invalid index

    proto.mutable_surface()->set_reflectivity_905nm(0.5);
    proto.mutable_surface()->set_reflectivity_1550nm(0.5);
    proto.mutable_surface()->set_roughness_rms_m(0);
    proto.mutable_surface()->set_correlation_length_m(0.01);

    EXPECT_THROW(MeshTargetView view(proto), std::invalid_argument);
}

TEST_F(MeshTargetViewTest, ValidationRejectsFaceWithTwoVertices) {
    MeshTargetDefinition proto;

    proto.add_vertices();
    proto.add_vertices();
    proto.add_vertices();

    auto* face = proto.add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);  // Only 2 vertices

    proto.mutable_surface()->set_reflectivity_905nm(0.5);
    proto.mutable_surface()->set_reflectivity_1550nm(0.5);
    proto.mutable_surface()->set_roughness_rms_m(0);
    proto.mutable_surface()->set_correlation_length_m(0.01);

    EXPECT_THROW(MeshTargetView view(proto), std::invalid_argument);
}

TEST_F(MeshTargetViewTest, ValidationRejectsMissingSurface) {
    MeshTargetDefinition proto;

    proto.add_vertices();
    proto.add_vertices();
    proto.add_vertices();

    auto* face = proto.add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);

    // No surface set

    EXPECT_THROW(MeshTargetView view(proto), std::invalid_argument);
}

TEST_F(MeshTargetViewTest, BoundingRadius) {
    auto proto = make_unit_cube_at_origin();
    MeshTargetView view(proto);

    // Cube centered at origin with half-side 0.5
    // Distance from center to corner is sqrt(3) * 0.5 ~ 0.866
    double radius = view.bounding_radius();
    EXPECT_NEAR(radius, std::sqrt(3.0) * 0.5, 1e-9);
}

TEST_F(MeshTargetViewTest, FaceCentroid) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    Eigen::Vector3d fc = view.face_centroid(0);

    // Centroid of the triangle
    EXPECT_NEAR(fc.x(), 0.0, 1e-9);
    EXPECT_NEAR(fc.y(), (-0.5 - 0.5 + 0.5) / 3.0, 1e-9);
    EXPECT_NEAR(fc.z(), 100.0, 1e-9);
}

TEST_F(MeshTargetViewTest, ProtoAccessor) {
    auto proto = make_unit_triangle_at_z100();
    MeshTargetView view(proto);

    EXPECT_EQ(view.proto().vertices_size(), 3);
    EXPECT_EQ(view.proto().faces_size(), 1);
}

}  // namespace
}  // namespace aurora::simulation
