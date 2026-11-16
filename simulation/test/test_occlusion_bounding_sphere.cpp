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

class BoundingSphereOcclusionTest : public ::testing::Test {
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

    /**
     * @brief Create a scene with a large occluding plane and a point target behind it.
     */
    SceneDefinition make_scene_with_occluder_and_target() {
        SceneDefinition scene;

        // Occluding plane at z=50 (closer)
        auto* plane = scene.add_plane_targets();
        plane->mutable_center_m()->set_x(0);
        plane->mutable_center_m()->set_y(0);
        plane->mutable_center_m()->set_z(50);
        plane->mutable_normal()->set_x(0);
        plane->mutable_normal()->set_y(0);
        plane->mutable_normal()->set_z(-1);
        plane->set_width_m(10);   // Large occluder
        plane->set_height_m(10);
        set_default_surface(plane->mutable_surface());

        // Point target at z=100 (behind plane, on-axis)
        auto* point = scene.add_point_targets();
        point->mutable_position_m()->set_x(0);
        point->mutable_position_m()->set_y(0);
        point->mutable_position_m()->set_z(100);
        set_default_surface(point->mutable_surface());

        return scene;
    }
};

// =============================================================================
// Point Target Occlusion by Planes
// =============================================================================

TEST_F(BoundingSphereOcclusionTest, OnAxisTargetOccludedByCloserPlane) {
    auto scene_def = make_scene_with_occluder_and_target();
    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_points = checker.compute_visible_points(pupil);

    // Point should be occluded by the plane
    EXPECT_TRUE(visible_points.empty());
}

TEST_F(BoundingSphereOcclusionTest, OffAxisTargetNotOccluded) {
    SceneDefinition scene_def;

    // Occluding plane at z=50, on-axis
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(0);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(50);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(2);  // Small occluder
    plane->set_height_m(2);
    set_default_surface(plane->mutable_surface());

    // Point target at z=100, off-axis (outside shadow cone)
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(20);  // Far off-axis
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_points = checker.compute_visible_points(pupil);

    // Point should NOT be occluded (off-axis)
    ASSERT_EQ(visible_points.size(), 1);
}

TEST_F(BoundingSphereOcclusionTest, TargetInFrontNotOccluded) {
    SceneDefinition scene_def;

    // "Occluder" plane at z=100 (farther)
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

    // Point target at z=50 (in front of plane)
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(50);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_points = checker.compute_visible_points(pupil);

    // Point is in front, should be visible
    ASSERT_EQ(visible_points.size(), 1);
}

// =============================================================================
// Point Target Occlusion by Meshes
// =============================================================================

TEST_F(BoundingSphereOcclusionTest, MeshOccludesPointTarget) {
    SceneDefinition scene_def;

    // Large mesh at z=50
    auto* mesh = scene_def.add_mesh_targets();
    // Create a large triangle
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

    // Point behind mesh
    auto* point = scene_def.add_point_targets();
    point->mutable_position_m()->set_x(0);
    point->mutable_position_m()->set_y(0);
    point->mutable_position_m()->set_z(100);
    set_default_surface(point->mutable_surface());

    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_points = checker.compute_visible_points(pupil);

    EXPECT_TRUE(visible_points.empty());
}

// =============================================================================
// OCCLUSION_NONE Mode (Control Test)
// =============================================================================

TEST_F(BoundingSphereOcclusionTest, OcclusionNone_NoOcclusion) {
    auto scene_def = make_scene_with_occluder_and_target();
    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    // With OCCLUSION_NONE, point should be visible
    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible_points = checker.compute_visible_points(pupil);

    ASSERT_EQ(visible_points.size(), 1);
}

// =============================================================================
// Plane Occlusion by Meshes
// =============================================================================

TEST_F(BoundingSphereOcclusionTest, PlaneOccludedByMesh) {
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
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_planes = checker.compute_visible_planes(pupil);

    // Plane should be occluded by mesh
    EXPECT_TRUE(visible_planes.empty());
}

TEST_F(BoundingSphereOcclusionTest, PlaneNotOccludedByPlane) {
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

    // Plane at z=60 (behind)
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
    PupilModel pupil = make_wide_fov();

    // In bounding sphere mode, planes don't occlude other planes
    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_planes = checker.compute_visible_planes(pupil);

    // Both planes should be visible (planes don't occlude each other)
    EXPECT_EQ(visible_planes.size(), 2);
}

// =============================================================================
// Mesh Face Occlusion
// =============================================================================

TEST_F(BoundingSphereOcclusionTest, MeshFaceOccludedByPlane) {
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
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_meshes = checker.compute_visible_meshes(pupil);

    // Mesh should be occluded by plane
    EXPECT_TRUE(visible_meshes.empty());
}

TEST_F(BoundingSphereOcclusionTest, MeshFaceOccludedByOtherMesh) {
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
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_meshes = checker.compute_visible_meshes(pupil);

    // First mesh should be visible, second mesh should be occluded
    ASSERT_EQ(visible_meshes.size(), 1);
    EXPECT_EQ(visible_meshes[0].target_index, 0);
}

TEST_F(BoundingSphereOcclusionTest, MeshNoSelfOcclusion) {
    SceneDefinition scene_def;

    // Single mesh with two front-facing faces at different z positions
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

    // Back face vertices (z = 5)
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
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_meshes = checker.compute_visible_meshes(pupil);

    // Both faces should be visible (no self-occlusion)
    ASSERT_EQ(visible_meshes.size(), 1);
    EXPECT_EQ(visible_meshes[0].visible_faces.size(), 2);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(BoundingSphereOcclusionTest, EmptyScene) {
    SceneDefinition scene_def;
    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);

    EXPECT_TRUE(checker.compute_visible_points(pupil).empty());
    EXPECT_TRUE(checker.compute_visible_planes(pupil).empty());
    EXPECT_TRUE(checker.compute_visible_meshes(pupil).empty());
}

TEST_F(BoundingSphereOcclusionTest, MultiplePointsPartialOcclusion) {
    SceneDefinition scene_def;

    // Occluding plane at z=50, off-axis at x=5
    // Large enough to create a shadow cone that covers the on-axis point behind it
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_x(5);
    plane->mutable_center_m()->set_y(0);
    plane->mutable_center_m()->set_z(50);
    plane->mutable_normal()->set_x(0);
    plane->mutable_normal()->set_y(0);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(10);  // Large enough to create shadow cone
    plane->set_height_m(10);
    set_default_surface(plane->mutable_surface());

    // Point 1: on-axis at z=100, far from plane's shadow cone, not occluded
    auto* point1 = scene_def.add_point_targets();
    point1->mutable_position_m()->set_x(-10);  // Far off-axis from plane
    point1->mutable_position_m()->set_y(0);
    point1->mutable_position_m()->set_z(100);
    set_default_surface(point1->mutable_surface());

    // Point 2: directly behind plane center (on line from origin through plane center)
    // Plane center at (5, 0, 50), so line from origin passes through (10, 0, 100) at z=100
    auto* point2 = scene_def.add_point_targets();
    point2->mutable_position_m()->set_x(10);  // On line: (5/50)*100 = 10
    point2->mutable_position_m()->set_y(0);
    point2->mutable_position_m()->set_z(100);
    set_default_surface(point2->mutable_surface());

    // Point 3: in front of plane, not occluded
    auto* point3 = scene_def.add_point_targets();
    point3->mutable_position_m()->set_x(5);
    point3->mutable_position_m()->set_y(0);
    point3->mutable_position_m()->set_z(30);
    set_default_surface(point3->mutable_surface());

    Scene scene(scene_def);
    PupilModel pupil = make_wide_fov();

    VisibilityChecker checker(scene, OCCLUSION_BOUNDING_SPHERE);
    auto visible_points = checker.compute_visible_points(pupil);

    // Points 1 and 3 should be visible, point 2 should be occluded
    ASSERT_EQ(visible_points.size(), 2);

    // Verify the correct points are visible (indices 0 and 2)
    bool found_0 = false;
    bool found_2 = false;
    for (const auto& vt : visible_points) {
        if (vt.target_index == 0) found_0 = true;
        if (vt.target_index == 2) found_2 = true;
    }
    EXPECT_TRUE(found_0);
    EXPECT_TRUE(found_2);
}

}  // namespace
}  // namespace aurora::simulation
