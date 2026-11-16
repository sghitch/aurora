#include <gtest/gtest.h>
#include "scene.pb.h"

using namespace aurora::simulation;

TEST(MeshProtoTest, CanCreateVertex) {
    Vertex v;
    v.mutable_position_m()->set_x(1.0);
    v.mutable_position_m()->set_y(2.0);
    v.mutable_position_m()->set_z(3.0);

    EXPECT_DOUBLE_EQ(v.position_m().x(), 1.0);
    EXPECT_DOUBLE_EQ(v.position_m().y(), 2.0);
    EXPECT_DOUBLE_EQ(v.position_m().z(), 3.0);
}

TEST(MeshProtoTest, CanCreateVertexWithNormal) {
    Vertex v;
    v.mutable_position_m()->set_z(1.0);
    v.mutable_normal()->set_z(-1.0);

    EXPECT_TRUE(v.has_normal());
    EXPECT_DOUBLE_EQ(v.normal().z(), -1.0);
}

TEST(MeshProtoTest, CanCreateTriangleFace) {
    Face f;
    f.add_vertex_indices(0);
    f.add_vertex_indices(1);
    f.add_vertex_indices(2);

    EXPECT_EQ(f.vertex_indices_size(), 3);
    EXPECT_EQ(f.vertex_indices(0), 0);
    EXPECT_EQ(f.vertex_indices(1), 1);
    EXPECT_EQ(f.vertex_indices(2), 2);
}

TEST(MeshProtoTest, CanCreateQuadFace) {
    Face f;
    f.add_vertex_indices(0);
    f.add_vertex_indices(1);
    f.add_vertex_indices(2);
    f.add_vertex_indices(3);

    EXPECT_EQ(f.vertex_indices_size(), 4);
}

TEST(MeshProtoTest, FaceCanHaveNormal) {
    Face f;
    f.add_vertex_indices(0);
    f.add_vertex_indices(1);
    f.add_vertex_indices(2);
    f.mutable_face_normal()->set_z(1.0);

    EXPECT_TRUE(f.has_face_normal());
    EXPECT_DOUBLE_EQ(f.face_normal().z(), 1.0);
}

TEST(MeshProtoTest, FaceCanHaveOverrideSurface) {
    Face f;
    f.add_vertex_indices(0);
    f.add_vertex_indices(1);
    f.add_vertex_indices(2);
    f.mutable_override_surface()->set_reflectivity_905nm(0.9);

    EXPECT_TRUE(f.has_override_surface());
    EXPECT_DOUBLE_EQ(f.override_surface().reflectivity_905nm(), 0.9);
}

TEST(MeshProtoTest, CanCreateSimpleMesh) {
    MeshTargetDefinition mesh;

    // Create a triangle at z=100
    auto* v0 = mesh.add_vertices();
    v0->mutable_position_m()->set_x(-0.5);
    v0->mutable_position_m()->set_y(-0.5);
    v0->mutable_position_m()->set_z(0);

    auto* v1 = mesh.add_vertices();
    v1->mutable_position_m()->set_x(0.5);
    v1->mutable_position_m()->set_y(-0.5);
    v1->mutable_position_m()->set_z(0);

    auto* v2 = mesh.add_vertices();
    v2->mutable_position_m()->set_x(0);
    v2->mutable_position_m()->set_y(0.5);
    v2->mutable_position_m()->set_z(0);

    auto* face = mesh.add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);

    // Position at z=100
    mesh.mutable_position_m()->set_z(100);

    // Surface properties
    mesh.mutable_surface()->set_reflectivity_905nm(0.5);
    mesh.mutable_surface()->set_reflectivity_1550nm(0.5);
    mesh.mutable_surface()->set_roughness_rms_m(0);
    mesh.mutable_surface()->set_correlation_length_m(0.01);

    EXPECT_EQ(mesh.vertices_size(), 3);
    EXPECT_EQ(mesh.faces_size(), 1);
    EXPECT_DOUBLE_EQ(mesh.position_m().z(), 100);
}

TEST(MeshProtoTest, CanAddMeshToScene) {
    SceneDefinition scene;

    auto* mesh = scene.add_mesh_targets();
    mesh->mutable_position_m()->set_z(100);
    mesh->mutable_surface()->set_reflectivity_905nm(0.5);
    mesh->mutable_surface()->set_reflectivity_1550nm(0.5);
    mesh->mutable_surface()->set_roughness_rms_m(0);
    mesh->mutable_surface()->set_correlation_length_m(0.01);

    EXPECT_EQ(scene.mesh_targets_size(), 1);
}

TEST(MeshProtoTest, MeshCanHaveVelocity) {
    MeshTargetDefinition mesh;
    mesh.mutable_velocity_mps()->set_x(10);
    mesh.mutable_velocity_mps()->set_y(5);
    mesh.mutable_velocity_mps()->set_z(-2);

    EXPECT_DOUBLE_EQ(mesh.velocity_mps().x(), 10);
    EXPECT_DOUBLE_EQ(mesh.velocity_mps().y(), 5);
    EXPECT_DOUBLE_EQ(mesh.velocity_mps().z(), -2);
}

TEST(MeshProtoTest, MeshCanHaveRotation) {
    MeshTargetDefinition mesh;
    mesh.mutable_rotation_rad()->set_x(0.1);  // Roll
    mesh.mutable_rotation_rad()->set_y(0.2);  // Pitch
    mesh.mutable_rotation_rad()->set_z(0.3);  // Yaw

    EXPECT_DOUBLE_EQ(mesh.rotation_rad().x(), 0.1);
    EXPECT_DOUBLE_EQ(mesh.rotation_rad().y(), 0.2);
    EXPECT_DOUBLE_EQ(mesh.rotation_rad().z(), 0.3);
}

TEST(MeshProtoTest, MeshCanHaveMeshId) {
    MeshTargetDefinition mesh;
    mesh.set_mesh_id("vehicle_001");

    EXPECT_TRUE(mesh.has_mesh_id());
    EXPECT_EQ(mesh.mesh_id(), "vehicle_001");
}

TEST(MeshProtoTest, MeshCanHaveSurfaceType) {
    MeshTargetDefinition mesh;
    mesh.set_surface_type(SURFACE_METAL);

    EXPECT_TRUE(mesh.has_surface_type());
    EXPECT_EQ(mesh.surface_type(), SURFACE_METAL);
}
