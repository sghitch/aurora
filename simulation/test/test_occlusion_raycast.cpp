#include "simulation/visibility_checker.h"

#include <gtest/gtest.h>

#include <cmath>

#include "simulation/pupil_model.h"
#include "simulation/scene.h"

#include "optics_config.pb.h"
#include "scene.pb.h"
#include "sim_config.pb.h"

namespace aurora::simulation {
namespace {

class RayCastOcclusionTest : public ::testing::Test {
  protected:
    /**
     * @brief Create a pupil model with wide FOV for testing.
     */
    PupilModel make_wide_fov() {
        aurora::forward::PupilConfig config;
        config.set_fov_half_angle_rad(1.0);  // ~60 degrees
        config.set_rolloff_width_rad(0.05);
        config.set_shape(aurora::forward::PupilConfig::PUPIL_CIRCULAR);
        return PupilModel(config);
    }

    /**
     * @brief Helper to set surface properties on a proto.
     */
    void set_default_surface(SurfaceProperties* surface) {
        surface->set_reflectivity_905nm(0.5);
        surface->set_reflectivity_1550nm(0.5);
        surface->set_roughness_rms_m(0);
        surface->set_correlation_length_m(0.01);
    }
};

// =============================================================================
// Point Target Occlusion by Planes
// =============================================================================

TEST_F(RayCastOcclusionTest, PointOccludedByPlane) {
    SceneDefinition scene_def;

    // Plane at z=50 (facing origin)
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(0);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(50);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(10);
    plane->set_height_m(10);
    set_default_surface(plane->mutable_surface());

    // Point at z=100 (on-axis, behind plane)
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_points(make_wide_fov());
    EXPECT_TRUE(visible.empty());
}

TEST_F(RayCastOcclusionTest, PointNotOccludedWhenOffPlane) {
    SceneDefinition scene_def;

    // Small plane at z=50
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(0);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(50);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(2);
    plane->set_height_m(2);
    set_default_surface(plane->mutable_surface());

    // Point at z=100, x=10 (off to the side, ray misses plane)
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(10);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_points(make_wide_fov());
    ASSERT_EQ(visible.size(), 1);
}

TEST_F(RayCastOcclusionTest, PointNotOccludedWhenInFrontOfPlane) {
    SceneDefinition scene_def;

    // Plane at z=100 (farther)
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(0);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(100);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(10);
    plane->set_height_m(10);
    set_default_surface(plane->mutable_surface());

    // Point at z=50 (in front of plane)
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(50);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_points(make_wide_fov());
    ASSERT_EQ(visible.size(), 1);
}

// =============================================================================
// Point Target Occlusion by Meshes
// =============================================================================

TEST_F(RayCastOcclusionTest, TriangleMeshOccludesPoint) {
    SceneDefinition scene_def;

    // Triangle mesh at z=50
    auto* mesh = scene_def.add_mesh_targets();
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-5);
    v0->mutable_position_m()->set_y(-5);
    v0->mutable_position_m()->set_z(0);
    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(5);
    v1->mutable_position_m()->set_y(-5);
    v1->mutable_position_m()->set_z(0);
    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_x(0);
    v2->mutable_position_m()->set_y(5);
    v2->mutable_position_m()->set_z(0);
    auto* face = mesh->add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);
    face->mutable_face_normal()->set_x(0);
    face->mutable_face_normal()->set_y(0);
    face->mutable_face_normal()->set_z(-1);
    mesh->mutable_position_m()->set_x(0);
    mesh->mutable_position_m()->set_y(0);
    mesh->mutable_position_m()->set_z(50);
    set_default_surface(mesh->mutable_surface());

    // Point behind triangle (on-axis)
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_points(make_wide_fov());
    EXPECT_TRUE(visible.empty());
}

TEST_F(RayCastOcclusionTest, PointNotOccludedByBackFacingMesh) {
    SceneDefinition scene_def;

    // Triangle mesh at z=50 with back-facing normal
    auto* mesh = scene_def.add_mesh_targets();
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-5);
    v0->mutable_position_m()->set_y(-5);
    v0->mutable_position_m()->set_z(0);
    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(5);
    v1->mutable_position_m()->set_y(-5);
    v1->mutable_position_m()->set_z(0);
    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_x(0);
    v2->mutable_position_m()->set_y(5);
    v2->mutable_position_m()->set_z(0);
    auto* face = mesh->add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);
    // Back-facing normal (facing away from origin)
    face->mutable_face_normal()->set_x(0);
    face->mutable_face_normal()->set_y(0);
    face->mutable_face_normal()->set_z(1);
    mesh->mutable_position_m()->set_x(0);
    mesh->mutable_position_m()->set_y(0);
    mesh->mutable_position_m()->set_z(50);
    set_default_surface(mesh->mutable_surface());

    // Point behind mesh
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    // Point should be visible (back-facing mesh doesn't occlude)
    auto visible = checker.compute_visible_points(make_wide_fov());
    ASSERT_EQ(visible.size(), 1);
}

// =============================================================================
// Mesh Self-Occlusion Tests
// =============================================================================

TEST_F(RayCastOcclusionTest, MeshSelfOcclusion) {
    SceneDefinition scene_def;

    // Cube mesh - front face should occlude back face
    auto* mesh = scene_def.add_mesh_targets();

    // 8 vertices of unit cube centered at origin
    double h = 0.5;
    // Front face vertices (z = -h)
    auto* v0 = mesh->add_vertices();  // 0: front bottom-left
    v0->mutable_position_m()->set_x(-h);
    v0->mutable_position_m()->set_y(-h);
    v0->mutable_position_m()->set_z(-h);
    auto* v1 = mesh->add_vertices();  // 1: front bottom-right
    v1->mutable_position_m()->set_x(h);
    v1->mutable_position_m()->set_y(-h);
    v1->mutable_position_m()->set_z(-h);
    auto* v2 = mesh->add_vertices();  // 2: front top-right
    v2->mutable_position_m()->set_x(h);
    v2->mutable_position_m()->set_y(h);
    v2->mutable_position_m()->set_z(-h);
    auto* v3 = mesh->add_vertices();  // 3: front top-left
    v3->mutable_position_m()->set_x(-h);
    v3->mutable_position_m()->set_y(h);
    v3->mutable_position_m()->set_z(-h);

    // Back face vertices (z = h)
    auto* v4 = mesh->add_vertices();  // 4: back bottom-left
    v4->mutable_position_m()->set_x(-h);
    v4->mutable_position_m()->set_y(-h);
    v4->mutable_position_m()->set_z(h);
    auto* v5 = mesh->add_vertices();  // 5: back bottom-right
    v5->mutable_position_m()->set_x(h);
    v5->mutable_position_m()->set_y(-h);
    v5->mutable_position_m()->set_z(h);
    auto* v6 = mesh->add_vertices();  // 6: back top-right
    v6->mutable_position_m()->set_x(h);
    v6->mutable_position_m()->set_y(h);
    v6->mutable_position_m()->set_z(h);
    auto* v7 = mesh->add_vertices();  // 7: back top-left
    v7->mutable_position_m()->set_x(-h);
    v7->mutable_position_m()->set_y(h);
    v7->mutable_position_m()->set_z(h);

    // Front face (facing origin when at z=50, normal points to -z)
    auto* front = mesh->add_faces();
    front->add_vertex_indices(0);
    front->add_vertex_indices(1);
    front->add_vertex_indices(2);
    front->add_vertex_indices(3);
    front->mutable_face_normal()->set_x(0);
    front->mutable_face_normal()->set_y(0);
    front->mutable_face_normal()->set_z(-1);

    // Back face (facing away from origin, normal points to +z)
    auto* back = mesh->add_faces();
    back->add_vertex_indices(5);
    back->add_vertex_indices(4);
    back->add_vertex_indices(7);
    back->add_vertex_indices(6);
    back->mutable_face_normal()->set_x(0);
    back->mutable_face_normal()->set_y(0);
    back->mutable_face_normal()->set_z(1);

    mesh->mutable_position_m()->set_x(0);
    mesh->mutable_position_m()->set_y(0);
    mesh->mutable_position_m()->set_z(50);
    set_default_surface(mesh->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_meshes(make_wide_fov());

    // Should have visible mesh with only front face visible
    // (back face is back-facing, so excluded by culling, not by ray-cast occlusion)
    ASSERT_EQ(visible.size(), 1);
    EXPECT_EQ(visible[0].visible_faces.size(), 1);
    EXPECT_EQ(visible[0].visible_faces[0], 0);  // Front face only
}

TEST_F(RayCastOcclusionTest, MeshSelfOcclusionWithRayCast) {
    // In ray-cast mode, mesh faces CAN occlude other faces in the same mesh
    // (this is physically correct self-occlusion behavior)
    SceneDefinition scene_def;

    // Single mesh with two front-facing faces at different z positions
    // The front face SHOULD occlude the back face
    auto* mesh = scene_def.add_mesh_targets();

    // Front face vertices (z = 0)
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-1);
    v0->mutable_position_m()->set_y(-1);
    v0->mutable_position_m()->set_z(0);
    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(1);
    v1->mutable_position_m()->set_y(-1);
    v1->mutable_position_m()->set_z(0);
    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_x(0);
    v2->mutable_position_m()->set_y(1);
    v2->mutable_position_m()->set_z(0);

    // Back face vertices (z = 5, directly behind front face)
    auto* v3 = mesh->add_vertices();
    v3->mutable_position_m()->set_x(-1);
    v3->mutable_position_m()->set_y(-1);
    v3->mutable_position_m()->set_z(5);
    auto* v4 = mesh->add_vertices();
    v4->mutable_position_m()->set_x(1);
    v4->mutable_position_m()->set_y(-1);
    v4->mutable_position_m()->set_z(5);
    auto* v5 = mesh->add_vertices();
    v5->mutable_position_m()->set_x(0);
    v5->mutable_position_m()->set_y(1);
    v5->mutable_position_m()->set_z(5);

    // Front face
    auto* face1 = mesh->add_faces();
    face1->add_vertex_indices(0);
    face1->add_vertex_indices(1);
    face1->add_vertex_indices(2);
    face1->mutable_face_normal()->set_x(0);
    face1->mutable_face_normal()->set_y(0);
    face1->mutable_face_normal()->set_z(-1);

    // Back face (also front-facing toward origin)
    auto* face2 = mesh->add_faces();
    face2->add_vertex_indices(3);
    face2->add_vertex_indices(4);
    face2->add_vertex_indices(5);
    face2->mutable_face_normal()->set_x(0);
    face2->mutable_face_normal()->set_y(0);
    face2->mutable_face_normal()->set_z(-1);

    mesh->mutable_position_m()->set_x(0);
    mesh->mutable_position_m()->set_y(0);
    mesh->mutable_position_m()->set_z(50);
    set_default_surface(mesh->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_meshes(make_wide_fov());

    // Only front face should be visible (back face is occluded by front face)
    // This is the correct ray-cast self-occlusion behavior
    ASSERT_EQ(visible.size(), 1);
    EXPECT_EQ(visible[0].visible_faces.size(), 1);
    EXPECT_EQ(visible[0].visible_faces[0], 0);  // Front face only
}

// =============================================================================
// Plane Occlusion by Meshes
// =============================================================================

TEST_F(RayCastOcclusionTest, PlaneOccludedByMesh) {
    SceneDefinition scene_def;

    // Large mesh at z=30
    auto* mesh = scene_def.add_mesh_targets();
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-5);
    v0->mutable_position_m()->set_y(-5);
    v0->mutable_position_m()->set_z(0);
    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(5);
    v1->mutable_position_m()->set_y(-5);
    v1->mutable_position_m()->set_z(0);
    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_x(0);
    v2->mutable_position_m()->set_y(5);
    v2->mutable_position_m()->set_z(0);
    auto* face = mesh->add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);
    face->mutable_face_normal()->set_x(0);
    face->mutable_face_normal()->set_y(0);
    face->mutable_face_normal()->set_z(-1);
    mesh->mutable_position_m()->set_x(0);
    mesh->mutable_position_m()->set_y(0);
    mesh->mutable_position_m()->set_z(30);
    set_default_surface(mesh->mutable_surface());

    // Plane behind mesh at z=60
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(0);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(60);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(2);
    plane->set_height_m(2);
    set_default_surface(plane->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_planes(make_wide_fov());

    // Plane should be occluded by mesh
    EXPECT_TRUE(visible.empty());
}

TEST_F(RayCastOcclusionTest, PlaneNotOccludedByPlane) {
    SceneDefinition scene_def;

    // Plane at z=30 (closer)
    auto* plane1 = scene_def.add_plane_targets();
    plane1->mutable_center_m()->set_x(0);
    plane1->mutable_center_m()->set_y(0);
    plane1->mutable_center_m()->set_z(30);
    plane1->mutable_normal()->set_x(0);
    plane1->mutable_normal()->set_y(0);
    plane1->mutable_normal()->set_z(-1);
    plane1->set_width_m(10);
    plane1->set_height_m(10);
    set_default_surface(plane1->mutable_surface());

    // Plane at z=60 (behind, on-axis)
    auto* plane2 = scene_def.add_plane_targets();
    plane2->mutable_center_m()->set_x(0);
    plane2->mutable_center_m()->set_y(0);
    plane2->mutable_center_m()->set_z(60);
    plane2->mutable_normal()->set_x(0);
    plane2->mutable_normal()->set_y(0);
    plane2->mutable_normal()->set_z(-1);
    plane2->set_width_m(2);
    plane2->set_height_m(2);
    set_default_surface(plane2->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    // In ray-cast mode (same as bounding sphere), planes don't occlude other planes
    auto visible = checker.compute_visible_planes(make_wide_fov());

    // Both planes should be visible
    EXPECT_EQ(visible.size(), 2);
}

// =============================================================================
// Mesh Face Occlusion
// =============================================================================

TEST_F(RayCastOcclusionTest, MeshFaceOccludedByPlane) {
    SceneDefinition scene_def;

    // Occluding plane at z=30
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(0);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(30);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(10);
    plane->set_height_m(10);
    set_default_surface(plane->mutable_surface());

    // Mesh behind plane at z=60
    auto* mesh = scene_def.add_mesh_targets();
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-1);
    v0->mutable_position_m()->set_y(-1);
    v0->mutable_position_m()->set_z(0);
    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(1);
    v1->mutable_position_m()->set_y(-1);
    v1->mutable_position_m()->set_z(0);
    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_x(0);
    v2->mutable_position_m()->set_y(1);
    v2->mutable_position_m()->set_z(0);
    auto* face = mesh->add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);
    face->mutable_face_normal()->set_x(0);
    face->mutable_face_normal()->set_y(0);
    face->mutable_face_normal()->set_z(-1);
    mesh->mutable_position_m()->set_x(0);
    mesh->mutable_position_m()->set_y(0);
    mesh->mutable_position_m()->set_z(60);
    set_default_surface(mesh->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_meshes(make_wide_fov());

    // Mesh should be occluded by plane
    EXPECT_TRUE(visible.empty());
}

TEST_F(RayCastOcclusionTest, MeshFaceOccludedByOtherMesh) {
    SceneDefinition scene_def;

    // Closer mesh at z=30
    auto* mesh1 = scene_def.add_mesh_targets();
    auto* v0 = mesh1->add_vertices();
    v0->mutable_position_m()->set_x(-5);
    v0->mutable_position_m()->set_y(-5);
    v0->mutable_position_m()->set_z(0);
    auto* v1 = mesh1->add_vertices();
    v1->mutable_position_m()->set_x(5);
    v1->mutable_position_m()->set_y(-5);
    v1->mutable_position_m()->set_z(0);
    auto* v2 = mesh1->add_vertices();
    v2->mutable_position_m()->set_x(0);
    v2->mutable_position_m()->set_y(5);
    v2->mutable_position_m()->set_z(0);
    auto* face1 = mesh1->add_faces();
    face1->add_vertex_indices(0);
    face1->add_vertex_indices(1);
    face1->add_vertex_indices(2);
    face1->mutable_face_normal()->set_x(0);
    face1->mutable_face_normal()->set_y(0);
    face1->mutable_face_normal()->set_z(-1);
    mesh1->mutable_position_m()->set_x(0);
    mesh1->mutable_position_m()->set_y(0);
    mesh1->mutable_position_m()->set_z(30);
    set_default_surface(mesh1->mutable_surface());

    // Farther mesh at z=60
    auto* mesh2 = scene_def.add_mesh_targets();
    auto* v3 = mesh2->add_vertices();
    v3->mutable_position_m()->set_x(-1);
    v3->mutable_position_m()->set_y(-1);
    v3->mutable_position_m()->set_z(0);
    auto* v4 = mesh2->add_vertices();
    v4->mutable_position_m()->set_x(1);
    v4->mutable_position_m()->set_y(-1);
    v4->mutable_position_m()->set_z(0);
    auto* v5 = mesh2->add_vertices();
    v5->mutable_position_m()->set_x(0);
    v5->mutable_position_m()->set_y(1);
    v5->mutable_position_m()->set_z(0);
    auto* face2 = mesh2->add_faces();
    face2->add_vertex_indices(0);
    face2->add_vertex_indices(1);
    face2->add_vertex_indices(2);
    face2->mutable_face_normal()->set_x(0);
    face2->mutable_face_normal()->set_y(0);
    face2->mutable_face_normal()->set_z(-1);
    mesh2->mutable_position_m()->set_x(0);
    mesh2->mutable_position_m()->set_y(0);
    mesh2->mutable_position_m()->set_z(60);
    set_default_surface(mesh2->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_meshes(make_wide_fov());

    // First mesh should be visible, second mesh should be occluded
    ASSERT_EQ(visible.size(), 1);
    EXPECT_EQ(visible[0].target_index, 0);
}

// =============================================================================
// Ray-Cast Accuracy vs Bounding Sphere
// =============================================================================

TEST_F(RayCastOcclusionTest, RayCastMoreAccurateThanBoundingSphere) {
    SceneDefinition scene_def;

    // Small plane at z=50, off-center
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(3);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(50);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(1);
    plane->set_height_m(1);
    set_default_surface(plane->mutable_surface());

    // Point at z=100, on-axis (not behind the off-center plane)
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);

    // Ray-cast should correctly show point is visible (ray doesn't hit plane)
    VisibilityChecker checker_rc(scene, OCCLUSION_RAY_CAST);
    auto visible_rc = checker_rc.compute_visible_points(make_wide_fov());

    // Ray-cast should find point visible (accurate)
    ASSERT_EQ(visible_rc.size(), 1);
}

TEST_F(RayCastOcclusionTest, RayCastPreciseEdgeDetection) {
    SceneDefinition scene_def;

    // Plane at z=50 with width=10 (x from -5 to 5)
    // Edge at x=5 at z=50 projects to x=10 at z=100 (ratio 100/50 = 2)
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(0);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(50);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(10);
    plane->set_height_m(10);
    set_default_surface(plane->mutable_surface());

    // Point just inside plane's shadow at z=100
    // Ray to (9.9, 0, 100) hits plane at (4.95, 0, 50) - inside
    auto* point1 = scene_def.add_point_targets();
    point1->mutable_position_m()->set_x(9.9);
    point1->mutable_position_m()->set_y(0);
    point1->mutable_position_m()->set_z(100);
    set_default_surface(point1->mutable_surface());

    // Point just outside plane's shadow at z=100
    // Ray to (10.1, 0, 100) hits plane at (5.05, 0, 50) - outside
    auto* point2 = scene_def.add_point_targets();
    point2->mutable_position_m()->set_x(10.1);
    point2->mutable_position_m()->set_y(0);
    point2->mutable_position_m()->set_z(100);
    set_default_surface(point2->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_points(make_wide_fov());

    // Point at 9.9 should be occluded (ray hits at 4.95 < 5)
    // Point at 10.1 should be visible (ray hits at 5.05 > 5)
    ASSERT_EQ(visible.size(), 1);
    EXPECT_EQ(visible[0].target_index, 1);  // Only point2 visible
}

// =============================================================================
// Quad Face Tests
// =============================================================================

TEST_F(RayCastOcclusionTest, QuadMeshOccludesPoint) {
    SceneDefinition scene_def;

    // Quad mesh at z=50
    auto* mesh = scene_def.add_mesh_targets();
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-5);
    v0->mutable_position_m()->set_y(-5);
    v0->mutable_position_m()->set_z(0);
    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(5);
    v1->mutable_position_m()->set_y(-5);
    v1->mutable_position_m()->set_z(0);
    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_x(5);
    v2->mutable_position_m()->set_y(5);
    v2->mutable_position_m()->set_z(0);
    auto* v3 = mesh->add_vertices();
    v3->mutable_position_m()->set_x(-5);
    v3->mutable_position_m()->set_y(5);
    v3->mutable_position_m()->set_z(0);
    auto* face = mesh->add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);
    face->add_vertex_indices(3);
    face->mutable_face_normal()->set_x(0);
    face->mutable_face_normal()->set_y(0);
    face->mutable_face_normal()->set_z(-1);
    mesh->mutable_position_m()->set_x(0);
    mesh->mutable_position_m()->set_y(0);
    mesh->mutable_position_m()->set_z(50);
    set_default_surface(mesh->mutable_surface());

    // Point behind quad
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    auto visible = checker.compute_visible_points(make_wide_fov());
    EXPECT_TRUE(visible.empty());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(RayCastOcclusionTest, EmptyScene) {
    SceneDefinition scene_def;
    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_RAY_CAST);

    EXPECT_TRUE(checker.compute_visible_points(pupil).empty());
    EXPECT_TRUE(checker.compute_visible_planes(pupil).empty());
    EXPECT_TRUE(checker.compute_visible_meshes(pupil).empty());
}

TEST_F(RayCastOcclusionTest, OcclusionNoneStillWorksWithRayCastInScene) {
    SceneDefinition scene_def;

    // Occluding plane at z=50
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(0);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(50);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(10);
    plane->set_height_m(10);
    set_default_surface(plane->mutable_surface());

    // Point at z=100 (behind plane)
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    // With OCCLUSION_NONE, point should be visible
    VisibilityChecker checker_none(scene, OCCLUSION_NONE);
    auto visible_none = checker_none.compute_visible_points(pupil);
    EXPECT_EQ(visible_none.size(), 1);

    // With OCCLUSION_RAY_CAST, point should be occluded
    VisibilityChecker checker_rc(scene, OCCLUSION_RAY_CAST);
    auto visible_rc = checker_rc.compute_visible_points(pupil);
    EXPECT_TRUE(visible_rc.empty());
}

}  // namespace
}  // namespace aurora::simulation
