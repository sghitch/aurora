#include <gtest/gtest.h>

#include <cmath>

#include "simulation/fmcw_simulator.h"
#include "simulation/scene.h"
#include "simulation/visibility_checker.h"
#include "simulation/views.h"

#include "optics_config.pb.h"
#include "scene.pb.h"
#include "sim_config.pb.h"

namespace aurora::simulation {
namespace {

class MeshIntegrationTest : public ::testing::Test {
  protected:
    static constexpr double kTolerance = 1e-9;

    /**
     * @brief Create a scene with a single front-facing triangle mesh at z=50m.
     */
    SceneDefinition make_scene_with_mesh() {
        SceneDefinition scene;

        auto* mesh = scene.add_mesh_targets();

        // Simple triangle at z=50m
        auto* v0 = mesh->add_vertices();
        v0->mutable_position_m()->set_x(-0.5);
        v0->mutable_position_m()->set_y(-0.5);
        v0->mutable_position_m()->set_z(0.0);

        auto* v1 = mesh->add_vertices();
        v1->mutable_position_m()->set_x(0.5);
        v1->mutable_position_m()->set_y(-0.5);
        v1->mutable_position_m()->set_z(0.0);

        auto* v2 = mesh->add_vertices();
        v2->mutable_position_m()->set_x(0.0);
        v2->mutable_position_m()->set_y(0.5);
        v2->mutable_position_m()->set_z(0.0);

        auto* face = mesh->add_faces();
        face->add_vertex_indices(0);
        face->add_vertex_indices(1);
        face->add_vertex_indices(2);
        // Normal facing toward origin (negative z in local coords becomes
        // negative z in world coords after translation)
        face->mutable_face_normal()->set_z(-1);

        mesh->mutable_position_m()->set_z(50);
        mesh->mutable_surface()->set_reflectivity_905nm(0.8);
        mesh->mutable_surface()->set_reflectivity_1550nm(0.8);
        mesh->mutable_surface()->set_roughness_rms_m(0);
        mesh->mutable_surface()->set_correlation_length_m(0.01);

        return scene;
    }

    /**
     * @brief Create a pupil model with wide FOV.
     */
    PupilModel make_wide_fov_pupil() {
        aurora::forward::PupilConfig config;
        config.set_fov_half_angle_rad(0.5);  // ~30 degrees
        config.set_rolloff_width_rad(0.05);
        config.set_shape(aurora::forward::PupilConfig::PUPIL_CIRCULAR);
        return PupilModel(config);
    }
};

// =============================================================================
// Scene Mesh Target Tests
// =============================================================================

TEST_F(MeshIntegrationTest, SceneStoresMeshTargets) {
    auto scene_def = make_scene_with_mesh();
    Scene scene(scene_def);

    EXPECT_EQ(scene.num_mesh_targets(), 1);
    EXPECT_EQ(scene.get_mesh_targets().size(), 1);
}

TEST_F(MeshIntegrationTest, SceneStoresMultipleMeshTargets) {
    SceneDefinition scene_def;

    // Add two mesh targets
    for (int m = 0; m < 2; ++m) {
        auto* mesh = scene_def.add_mesh_targets();

        auto* v0 = mesh->add_vertices();
        v0->mutable_position_m()->set_x(-0.5);
        v0->mutable_position_m()->set_y(-0.5);

        auto* v1 = mesh->add_vertices();
        v1->mutable_position_m()->set_x(0.5);
        v1->mutable_position_m()->set_y(-0.5);

        auto* v2 = mesh->add_vertices();
        v2->mutable_position_m()->set_x(0.0);
        v2->mutable_position_m()->set_y(0.5);

        auto* face = mesh->add_faces();
        face->add_vertex_indices(0);
        face->add_vertex_indices(1);
        face->add_vertex_indices(2);
        face->mutable_face_normal()->set_z(-1);

        mesh->mutable_position_m()->set_z(50 + m * 10);
        mesh->mutable_surface()->set_reflectivity_905nm(0.8);
        mesh->mutable_surface()->set_reflectivity_1550nm(0.8);
        mesh->mutable_surface()->set_roughness_rms_m(0);
        mesh->mutable_surface()->set_correlation_length_m(0.01);
    }

    Scene scene(scene_def);
    EXPECT_EQ(scene.num_mesh_targets(), 2);
}

TEST_F(MeshIntegrationTest, SceneUpdatesMeshPosition) {
    auto scene_def = make_scene_with_mesh();
    scene_def.mutable_mesh_targets(0)->mutable_velocity_mps()->set_z(10);  // 10 m/s

    Scene scene(scene_def);
    double initial_z = MeshTargetView(scene.get_mesh_targets()[0]).centroid().z();

    scene.update(1.0);  // 1 second

    double final_z = MeshTargetView(scene.get_mesh_targets()[0]).centroid().z();
    EXPECT_NEAR(final_z - initial_z, 10.0, 0.1);
}

TEST_F(MeshIntegrationTest, SceneMeshTargetAccessor) {
    auto scene_def = make_scene_with_mesh();
    Scene scene(scene_def);

    auto mesh = scene.mesh_target(0);
    EXPECT_EQ(mesh.num_vertices(), 3);
    EXPECT_EQ(mesh.num_faces(), 1);
}

TEST_F(MeshIntegrationTest, MeshReflectivity905nm) {
    auto scene_def = make_scene_with_mesh();
    scene_def.mutable_mesh_targets(0)->mutable_surface()->set_reflectivity_905nm(0.7);
    scene_def.mutable_mesh_targets(0)->mutable_surface()->set_reflectivity_1550nm(0.3);

    Scene scene(scene_def);
    auto mesh = scene.mesh_target(0);

    double rho = Scene::get_reflectivity(mesh, Scene::kReferenceWavelength905nm);
    EXPECT_NEAR(rho, 0.7, kTolerance);
}

TEST_F(MeshIntegrationTest, MeshReflectivity1550nm) {
    auto scene_def = make_scene_with_mesh();
    scene_def.mutable_mesh_targets(0)->mutable_surface()->set_reflectivity_905nm(0.7);
    scene_def.mutable_mesh_targets(0)->mutable_surface()->set_reflectivity_1550nm(0.3);

    Scene scene(scene_def);
    auto mesh = scene.mesh_target(0);

    double rho = Scene::get_reflectivity(mesh, Scene::kSecondaryWavelength1550nm);
    EXPECT_NEAR(rho, 0.3, kTolerance);
}

// =============================================================================
// Visibility Checker Mesh Tests
// =============================================================================

TEST_F(MeshIntegrationTest, VisibilityChecker_FrontFacingMeshVisible) {
    auto scene_def = make_scene_with_mesh();
    Scene scene(scene_def);

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto pupil = make_wide_fov_pupil();

    auto visible = checker.compute_visible_meshes(pupil);

    ASSERT_EQ(visible.size(), 1);
    EXPECT_EQ(visible[0].target_index, 0);
    ASSERT_EQ(visible[0].visible_faces.size(), 1);
    EXPECT_EQ(visible[0].visible_faces[0], 0);
}

TEST_F(MeshIntegrationTest, VisibilityChecker_BackFacingMeshNotVisible) {
    auto scene_def = make_scene_with_mesh();
    // Change normal to face away from origin
    scene_def.mutable_mesh_targets(0)->mutable_faces(0)->mutable_face_normal()->set_z(1);

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto pupil = make_wide_fov_pupil();

    auto visible = checker.compute_visible_meshes(pupil);

    // Mesh should have no visible faces
    EXPECT_TRUE(visible.empty());
}

TEST_F(MeshIntegrationTest, VisibilityChecker_MixedFacesMesh) {
    SceneDefinition scene_def;
    auto* mesh = scene_def.add_mesh_targets();

    // Create a mesh with two triangles: one front-facing, one back-facing
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-0.5);
    v0->mutable_position_m()->set_y(-0.5);

    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(0.5);
    v1->mutable_position_m()->set_y(-0.5);

    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_x(0.0);
    v2->mutable_position_m()->set_y(0.5);

    auto* v3 = mesh->add_vertices();
    v3->mutable_position_m()->set_x(0.0);
    v3->mutable_position_m()->set_y(0.0);
    v3->mutable_position_m()->set_z(0.5);

    // Face 0: front-facing
    auto* face0 = mesh->add_faces();
    face0->add_vertex_indices(0);
    face0->add_vertex_indices(1);
    face0->add_vertex_indices(2);
    face0->mutable_face_normal()->set_z(-1);

    // Face 1: back-facing
    auto* face1 = mesh->add_faces();
    face1->add_vertex_indices(0);
    face1->add_vertex_indices(2);
    face1->add_vertex_indices(3);
    face1->mutable_face_normal()->set_z(1);

    mesh->mutable_position_m()->set_z(50);
    mesh->mutable_surface()->set_reflectivity_905nm(0.8);
    mesh->mutable_surface()->set_reflectivity_1550nm(0.8);
    mesh->mutable_surface()->set_roughness_rms_m(0);
    mesh->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(scene_def);
    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto pupil = make_wide_fov_pupil();

    auto visible = checker.compute_visible_meshes(pupil);

    ASSERT_EQ(visible.size(), 1);
    // Only face 0 should be visible
    ASSERT_EQ(visible[0].visible_faces.size(), 1);
    EXPECT_EQ(visible[0].visible_faces[0], 0);
}

TEST_F(MeshIntegrationTest, VisibilityChecker_TransmissionPassedThrough) {
    auto scene_def = make_scene_with_mesh();
    Scene scene(scene_def);

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto pupil = make_wide_fov_pupil();

    auto visible = checker.compute_visible_meshes(pupil);

    ASSERT_EQ(visible.size(), 1);
    EXPECT_GT(visible[0].transmission, 0.0);
    EXPECT_LE(visible[0].transmission, 1.0);
}

TEST_F(MeshIntegrationTest, VisibilityChecker_EmptyScene) {
    SceneDefinition scene_def;
    Scene scene(scene_def);

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto pupil = make_wide_fov_pupil();

    auto visible = checker.compute_visible_meshes(pupil);
    EXPECT_TRUE(visible.empty());
}

// =============================================================================
// Mixed Scene Tests
// =============================================================================

TEST_F(MeshIntegrationTest, MixedScene_AllTargetTypesStored) {
    SceneDefinition scene_def;

    // Add point target
    auto* pt = scene_def.add_point_targets();
    pt->mutable_position_m()->set_z(30);
    pt->mutable_surface()->set_reflectivity_905nm(0.5);
    pt->mutable_surface()->set_reflectivity_1550nm(0.5);
    pt->mutable_surface()->set_roughness_rms_m(0);
    pt->mutable_surface()->set_correlation_length_m(0.01);

    // Add plane target
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_z(40);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(1);
    plane->set_height_m(1);
    plane->mutable_surface()->set_reflectivity_905nm(0.5);
    plane->mutable_surface()->set_reflectivity_1550nm(0.5);
    plane->mutable_surface()->set_roughness_rms_m(0);
    plane->mutable_surface()->set_correlation_length_m(0.01);

    // Add mesh target (from helper)
    auto mesh_scene = make_scene_with_mesh();
    *scene_def.add_mesh_targets() = mesh_scene.mesh_targets(0);

    Scene scene(scene_def);

    EXPECT_EQ(scene.point_target_count(), 1);
    EXPECT_EQ(scene.plane_target_count(), 1);
    EXPECT_EQ(scene.num_mesh_targets(), 1);
}

TEST_F(MeshIntegrationTest, MixedScene_AllUpdatedWithVelocity) {
    SceneDefinition scene_def;

    // Point target with velocity
    auto* pt = scene_def.add_point_targets();
    pt->mutable_position_m()->set_z(30);
    pt->mutable_velocity_mps()->set_z(5);
    pt->mutable_surface()->set_reflectivity_905nm(0.5);
    pt->mutable_surface()->set_reflectivity_1550nm(0.5);
    pt->mutable_surface()->set_roughness_rms_m(0);
    pt->mutable_surface()->set_correlation_length_m(0.01);

    // Plane target with velocity
    auto* plane = scene_def.add_plane_targets();
    plane->mutable_center_m()->set_z(40);
    plane->mutable_velocity_mps()->set_z(10);
    plane->mutable_normal()->set_z(-1);
    plane->set_width_m(1);
    plane->set_height_m(1);
    plane->mutable_surface()->set_reflectivity_905nm(0.5);
    plane->mutable_surface()->set_reflectivity_1550nm(0.5);
    plane->mutable_surface()->set_roughness_rms_m(0);
    plane->mutable_surface()->set_correlation_length_m(0.01);

    // Mesh target with velocity
    auto* mesh = scene_def.add_mesh_targets();
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-0.5);
    v0->mutable_position_m()->set_y(-0.5);
    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(0.5);
    v1->mutable_position_m()->set_y(-0.5);
    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_x(0.0);
    v2->mutable_position_m()->set_y(0.5);
    auto* face = mesh->add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);
    face->mutable_face_normal()->set_z(-1);
    mesh->mutable_position_m()->set_z(50);
    mesh->mutable_velocity_mps()->set_z(15);
    mesh->mutable_surface()->set_reflectivity_905nm(0.8);
    mesh->mutable_surface()->set_reflectivity_1550nm(0.8);
    mesh->mutable_surface()->set_roughness_rms_m(0);
    mesh->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(scene_def);

    // Record initial positions
    double initial_point_z = scene.point_target(0).position().z();
    double initial_plane_z = scene.plane_target(0).center().z();
    double initial_mesh_z = scene.mesh_target(0).centroid().z();

    // Update by 1 second
    scene.update(1.0);

    // Check positions updated correctly
    EXPECT_NEAR(scene.point_target(0).position().z() - initial_point_z, 5.0, kTolerance);
    EXPECT_NEAR(scene.plane_target(0).center().z() - initial_plane_z, 10.0, kTolerance);
    EXPECT_NEAR(scene.mesh_target(0).centroid().z() - initial_mesh_z, 15.0, kTolerance);
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST_F(MeshIntegrationTest, SceneRejectsMeshWithoutSurface) {
    SceneDefinition scene_def;
    auto* mesh = scene_def.add_mesh_targets();

    // Add vertices and face but no surface
    auto* v0 = mesh->add_vertices();
    v0->mutable_position_m()->set_x(-0.5);
    v0->mutable_position_m()->set_y(-0.5);
    auto* v1 = mesh->add_vertices();
    v1->mutable_position_m()->set_x(0.5);
    v1->mutable_position_m()->set_y(-0.5);
    auto* v2 = mesh->add_vertices();
    v2->mutable_position_m()->set_y(0.5);
    auto* face = mesh->add_faces();
    face->add_vertex_indices(0);
    face->add_vertex_indices(1);
    face->add_vertex_indices(2);
    face->mutable_face_normal()->set_z(-1);  // Provide explicit normal
    mesh->mutable_position_m()->set_z(50);
    // No surface set

    bool threw_invalid_argument = false;
    try {
        Scene scene(scene_def);
    } catch (const std::invalid_argument&) {
        threw_invalid_argument = true;
    }
    EXPECT_TRUE(threw_invalid_argument);
}

}  // namespace
}  // namespace aurora::simulation
