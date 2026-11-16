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

class VisibilityCheckerTest : public ::testing::Test {
  protected:
    /**
     * @brief Create a pupil model with wide FOV for testing.
     *
     * FOV is large enough to see all test targets within reasonable angles.
     */
    PupilModel make_wide_fov_pupil() {
        aurora::forward::PupilConfig config;
        config.set_fov_half_angle_rad(0.5);  // ~30 degrees
        config.set_rolloff_width_rad(0.05);
        config.set_shape(aurora::forward::PupilConfig::PUPIL_CIRCULAR);
        return PupilModel(config);
    }

    /**
     * @brief Create a pupil model with narrow FOV for testing edge cases.
     */
    PupilModel make_narrow_fov_pupil() {
        aurora::forward::PupilConfig config;
        config.set_fov_half_angle_rad(0.05);  // ~3 degrees
        config.set_rolloff_width_rad(0.01);
        config.set_shape(aurora::forward::PupilConfig::PUPIL_CIRCULAR);
        return PupilModel(config);
    }

    /**
     * @brief Create a scene with two plane targets: one front-facing, one back-facing.
     */
    Scene make_scene_with_planes() {
        SceneDefinition def;

        // Plane 1: facing origin (front-facing)
        auto* p1 = def.add_plane_targets();
        p1->mutable_center_m()->set_x(0.0);
        p1->mutable_center_m()->set_y(0.0);
        p1->mutable_center_m()->set_z(100.0);
        p1->mutable_normal()->set_x(0.0);
        p1->mutable_normal()->set_y(0.0);
        p1->mutable_normal()->set_z(-1.0);  // Points toward origin
        p1->set_width_m(1.0);
        p1->set_height_m(1.0);
        p1->mutable_surface()->set_reflectivity_905nm(0.5);
        p1->mutable_surface()->set_reflectivity_1550nm(0.5);
        p1->mutable_surface()->set_roughness_rms_m(0);
        p1->mutable_surface()->set_correlation_length_m(0.01);

        // Plane 2: facing away (back-facing)
        auto* p2 = def.add_plane_targets();
        p2->mutable_center_m()->set_x(0.0);
        p2->mutable_center_m()->set_y(0.0);
        p2->mutable_center_m()->set_z(100.0);
        p2->mutable_normal()->set_x(0.0);
        p2->mutable_normal()->set_y(0.0);
        p2->mutable_normal()->set_z(1.0);  // Points away from origin
        p2->set_width_m(1.0);
        p2->set_height_m(1.0);
        p2->mutable_surface()->set_reflectivity_905nm(0.5);
        p2->mutable_surface()->set_reflectivity_1550nm(0.5);
        p2->mutable_surface()->set_roughness_rms_m(0);
        p2->mutable_surface()->set_correlation_length_m(0.01);

        return Scene(def);
    }

    /**
     * @brief Create an empty scene for basic tests.
     */
    Scene make_empty_scene() {
        SceneDefinition def;
        return Scene(def);
    }
};

// =============================================================================
// Constructor Tests
// =============================================================================

TEST_F(VisibilityCheckerTest, RejectsUnspecifiedMode) {
    Scene scene = make_empty_scene();
    EXPECT_THROW(VisibilityChecker(scene, OCCLUSION_UNSPECIFIED),
                 std::invalid_argument);
}

TEST_F(VisibilityCheckerTest, AcceptsOcclusionNone) {
    Scene scene = make_empty_scene();
    EXPECT_NO_THROW(VisibilityChecker(scene, OCCLUSION_NONE));
}

TEST_F(VisibilityCheckerTest, AcceptsOcclusionBoundingSphere) {
    Scene scene = make_empty_scene();
    EXPECT_NO_THROW(VisibilityChecker(scene, OCCLUSION_BOUNDING_SPHERE));
}

TEST_F(VisibilityCheckerTest, AcceptsOcclusionRayCast) {
    Scene scene = make_empty_scene();
    EXPECT_NO_THROW(VisibilityChecker(scene, OCCLUSION_RAY_CAST));
}

TEST_F(VisibilityCheckerTest, ModeAccessor) {
    Scene scene = make_empty_scene();
    VisibilityChecker checker(scene, OCCLUSION_NONE);
    EXPECT_EQ(checker.mode(), OCCLUSION_NONE);
}

// =============================================================================
// is_front_facing() Static Method Tests
// =============================================================================

TEST_F(VisibilityCheckerTest, IsFrontFacing_FacingOrigin) {
    // Surface at z=100 with normal pointing toward origin (-z direction)
    Eigen::Vector3d normal(0, 0, -1);
    Eigen::Vector3d position(0, 0, 100);

    EXPECT_TRUE(VisibilityChecker::is_front_facing(normal, position));
}

TEST_F(VisibilityCheckerTest, IsFrontFacing_FacingAway) {
    // Surface at z=100 with normal pointing away from origin (+z direction)
    Eigen::Vector3d normal(0, 0, 1);
    Eigen::Vector3d position(0, 0, 100);

    EXPECT_FALSE(VisibilityChecker::is_front_facing(normal, position));
}

TEST_F(VisibilityCheckerTest, IsFrontFacing_Perpendicular) {
    // Surface at z=100 with normal perpendicular to view (x direction)
    Eigen::Vector3d normal(1, 0, 0);
    Eigen::Vector3d position(0, 0, 100);

    // Perpendicular should be false (grazing angle, not visible)
    EXPECT_FALSE(VisibilityChecker::is_front_facing(normal, position));
}

TEST_F(VisibilityCheckerTest, IsFrontFacing_OffAxisPosition) {
    // Surface at an off-axis position
    Eigen::Vector3d position(10, 0, 100);  // Off-axis
    Eigen::Vector3d view_dir = -position.normalized();

    // Normal pointing toward viewer
    Eigen::Vector3d normal = view_dir;
    EXPECT_TRUE(VisibilityChecker::is_front_facing(normal, position));

    // Normal pointing away from viewer
    normal = -view_dir;
    EXPECT_FALSE(VisibilityChecker::is_front_facing(normal, position));
}

TEST_F(VisibilityCheckerTest, IsFrontFacing_AngledNormal) {
    // Surface at z=100, tilted 45 degrees toward viewer
    Eigen::Vector3d position(0, 0, 100);
    double angle_rad = M_PI / 4.0;  // 45 degrees
    Eigen::Vector3d normal(std::sin(angle_rad), 0, -std::cos(angle_rad));

    // Tilted toward viewer - should still be front-facing
    EXPECT_TRUE(VisibilityChecker::is_front_facing(normal, position));

    // Tilted 135 degrees (away from viewer)
    normal = Eigen::Vector3d(std::sin(angle_rad), 0, std::cos(angle_rad));
    EXPECT_FALSE(VisibilityChecker::is_front_facing(normal, position));
}

// =============================================================================
// compute_visible_planes() Tests
// =============================================================================

TEST_F(VisibilityCheckerTest, ComputeVisiblePlanes_BackFaceCulling) {
    Scene scene = make_scene_with_planes();
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_planes(pupil);

    // Only front-facing plane should be visible
    ASSERT_EQ(visible.size(), 1);
    EXPECT_EQ(visible[0].target_index, 0);  // First plane (front-facing)
}

TEST_F(VisibilityCheckerTest, ComputeVisiblePlanes_EmptyScene) {
    Scene scene = make_empty_scene();
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_planes(pupil);

    EXPECT_TRUE(visible.empty());
}

TEST_F(VisibilityCheckerTest, ComputeVisiblePlanes_AllFrontFacing) {
    SceneDefinition def;

    // Add multiple front-facing planes
    for (int i = 0; i < 3; ++i) {
        auto* p = def.add_plane_targets();
        p->mutable_center_m()->set_x(static_cast<double>(i) * 0.5);
        p->mutable_center_m()->set_y(0.0);
        p->mutable_center_m()->set_z(100.0);
        p->mutable_normal()->set_x(0.0);
        p->mutable_normal()->set_y(0.0);
        p->mutable_normal()->set_z(-1.0);  // All front-facing
        p->set_width_m(0.1);
        p->set_height_m(0.1);
        p->mutable_surface()->set_reflectivity_905nm(0.5);
        p->mutable_surface()->set_reflectivity_1550nm(0.5);
        p->mutable_surface()->set_roughness_rms_m(0);
        p->mutable_surface()->set_correlation_length_m(0.01);
    }

    Scene scene(def);
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_planes(pupil);

    EXPECT_EQ(visible.size(), 3);
}

TEST_F(VisibilityCheckerTest, ComputeVisiblePlanes_OutsideFOV) {
    SceneDefinition def;

    // Add plane that's far outside the FOV (60 degrees off-axis)
    auto* p = def.add_plane_targets();
    p->mutable_center_m()->set_x(173.0);  // tan(60°) * 100
    p->mutable_center_m()->set_y(0.0);
    p->mutable_center_m()->set_z(100.0);
    p->mutable_normal()->set_x(0.0);
    p->mutable_normal()->set_y(0.0);
    p->mutable_normal()->set_z(-1.0);
    p->set_width_m(1.0);
    p->set_height_m(1.0);
    p->mutable_surface()->set_reflectivity_905nm(0.5);
    p->mutable_surface()->set_reflectivity_1550nm(0.5);
    p->mutable_surface()->set_roughness_rms_m(0);
    p->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(def);
    PupilModel pupil = make_narrow_fov_pupil();  // ~3 degree FOV

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_planes(pupil);

    // Should not be visible (outside FOV)
    EXPECT_TRUE(visible.empty());
}

// =============================================================================
// compute_visible_points() Tests
// =============================================================================

TEST_F(VisibilityCheckerTest, ComputeVisiblePoints_NoBackFaceCulling) {
    SceneDefinition def;

    // Add a point target
    auto* pt = def.add_point_targets();
    pt->mutable_position_m()->set_x(0.0);
    pt->mutable_position_m()->set_y(0.0);
    pt->mutable_position_m()->set_z(100.0);
    pt->mutable_surface()->set_reflectivity_905nm(0.5);
    pt->mutable_surface()->set_reflectivity_1550nm(0.5);
    pt->mutable_surface()->set_roughness_rms_m(0);
    pt->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(def);
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_points(pupil);

    // Point targets are always visible if in FOV (no back-face culling)
    ASSERT_EQ(visible.size(), 1);
}

TEST_F(VisibilityCheckerTest, ComputeVisiblePoints_EmptyScene) {
    Scene scene = make_empty_scene();
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_points(pupil);

    EXPECT_TRUE(visible.empty());
}

TEST_F(VisibilityCheckerTest, ComputeVisiblePoints_MultipleTargets) {
    SceneDefinition def;

    // Add multiple point targets at different positions
    for (int i = 0; i < 5; ++i) {
        auto* pt = def.add_point_targets();
        pt->mutable_position_m()->set_x(static_cast<double>(i) * 0.5);
        pt->mutable_position_m()->set_y(0.0);
        pt->mutable_position_m()->set_z(100.0);
        pt->mutable_surface()->set_reflectivity_905nm(0.5);
        pt->mutable_surface()->set_reflectivity_1550nm(0.5);
        pt->mutable_surface()->set_roughness_rms_m(0);
        pt->mutable_surface()->set_correlation_length_m(0.01);
    }

    Scene scene(def);
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_points(pupil);

    EXPECT_EQ(visible.size(), 5);
}

TEST_F(VisibilityCheckerTest, ComputeVisiblePoints_OutsideFOV) {
    SceneDefinition def;

    // Point target far outside FOV
    auto* pt = def.add_point_targets();
    pt->mutable_position_m()->set_x(173.0);  // ~60 degrees off-axis
    pt->mutable_position_m()->set_y(0.0);
    pt->mutable_position_m()->set_z(100.0);
    pt->mutable_surface()->set_reflectivity_905nm(0.5);
    pt->mutable_surface()->set_reflectivity_1550nm(0.5);
    pt->mutable_surface()->set_roughness_rms_m(0);
    pt->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(def);
    PupilModel pupil = make_narrow_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_points(pupil);

    EXPECT_TRUE(visible.empty());
}

// =============================================================================
// Transmission Tests
// =============================================================================

TEST_F(VisibilityCheckerTest, TransmissionPassedThrough_Points) {
    SceneDefinition def;

    auto* pt = def.add_point_targets();
    pt->mutable_position_m()->set_x(0.0);
    pt->mutable_position_m()->set_y(0.0);
    pt->mutable_position_m()->set_z(100.0);
    pt->mutable_surface()->set_reflectivity_905nm(0.5);
    pt->mutable_surface()->set_reflectivity_1550nm(0.5);
    pt->mutable_surface()->set_roughness_rms_m(0);
    pt->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(def);
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_points(pupil);

    ASSERT_EQ(visible.size(), 1);
    EXPECT_GT(visible[0].transmission, 0.0);
    EXPECT_LE(visible[0].transmission, 1.0);
}

TEST_F(VisibilityCheckerTest, TransmissionPassedThrough_Planes) {
    SceneDefinition def;

    auto* p = def.add_plane_targets();
    p->mutable_center_m()->set_x(0.0);
    p->mutable_center_m()->set_y(0.0);
    p->mutable_center_m()->set_z(100.0);
    p->mutable_normal()->set_x(0.0);
    p->mutable_normal()->set_y(0.0);
    p->mutable_normal()->set_z(-1.0);
    p->set_width_m(1.0);
    p->set_height_m(1.0);
    p->mutable_surface()->set_reflectivity_905nm(0.5);
    p->mutable_surface()->set_reflectivity_1550nm(0.5);
    p->mutable_surface()->set_roughness_rms_m(0);
    p->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(def);
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_planes(pupil);

    ASSERT_EQ(visible.size(), 1);
    EXPECT_GT(visible[0].transmission, 0.0);
    EXPECT_LE(visible[0].transmission, 1.0);
}

TEST_F(VisibilityCheckerTest, TransmissionNearUnity_OnAxis) {
    SceneDefinition def;

    // On-axis point target should have transmission ~1.0
    auto* pt = def.add_point_targets();
    pt->mutable_position_m()->set_x(0.0);
    pt->mutable_position_m()->set_y(0.0);
    pt->mutable_position_m()->set_z(100.0);
    pt->mutable_surface()->set_reflectivity_905nm(0.5);
    pt->mutable_surface()->set_reflectivity_1550nm(0.5);
    pt->mutable_surface()->set_roughness_rms_m(0);
    pt->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(def);
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_points(pupil);

    ASSERT_EQ(visible.size(), 1);
    EXPECT_GT(visible[0].transmission, 0.95);
}

// =============================================================================
// compute_visible_meshes() Tests
// =============================================================================

TEST_F(VisibilityCheckerTest, ComputeVisibleMeshes_ReturnsEmpty) {
    // Mesh support not yet implemented - should return empty
    Scene scene = make_empty_scene();
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);
    auto visible = checker.compute_visible_meshes(pupil);

    EXPECT_TRUE(visible.empty());
}

// =============================================================================
// Mixed Scene Tests
// =============================================================================

TEST_F(VisibilityCheckerTest, MixedScene_PointsAndPlanes) {
    SceneDefinition def;

    // Add point target
    auto* pt = def.add_point_targets();
    pt->mutable_position_m()->set_x(0.0);
    pt->mutable_position_m()->set_y(0.0);
    pt->mutable_position_m()->set_z(50.0);
    pt->mutable_surface()->set_reflectivity_905nm(0.5);
    pt->mutable_surface()->set_reflectivity_1550nm(0.5);
    pt->mutable_surface()->set_roughness_rms_m(0);
    pt->mutable_surface()->set_correlation_length_m(0.01);

    // Add front-facing plane
    auto* p1 = def.add_plane_targets();
    p1->mutable_center_m()->set_x(0.0);
    p1->mutable_center_m()->set_y(0.0);
    p1->mutable_center_m()->set_z(100.0);
    p1->mutable_normal()->set_x(0.0);
    p1->mutable_normal()->set_y(0.0);
    p1->mutable_normal()->set_z(-1.0);
    p1->set_width_m(1.0);
    p1->set_height_m(1.0);
    p1->mutable_surface()->set_reflectivity_905nm(0.5);
    p1->mutable_surface()->set_reflectivity_1550nm(0.5);
    p1->mutable_surface()->set_roughness_rms_m(0);
    p1->mutable_surface()->set_correlation_length_m(0.01);

    // Add back-facing plane
    auto* p2 = def.add_plane_targets();
    p2->mutable_center_m()->set_x(0.0);
    p2->mutable_center_m()->set_y(0.0);
    p2->mutable_center_m()->set_z(100.0);
    p2->mutable_normal()->set_x(0.0);
    p2->mutable_normal()->set_y(0.0);
    p2->mutable_normal()->set_z(1.0);  // Back-facing
    p2->set_width_m(1.0);
    p2->set_height_m(1.0);
    p2->mutable_surface()->set_reflectivity_905nm(0.5);
    p2->mutable_surface()->set_reflectivity_1550nm(0.5);
    p2->mutable_surface()->set_roughness_rms_m(0);
    p2->mutable_surface()->set_correlation_length_m(0.01);

    Scene scene(def);
    PupilModel pupil = make_wide_fov_pupil();

    VisibilityChecker checker(scene, OCCLUSION_NONE);

    auto visible_points = checker.compute_visible_points(pupil);
    auto visible_planes = checker.compute_visible_planes(pupil);

    EXPECT_EQ(visible_points.size(), 1);
    EXPECT_EQ(visible_planes.size(), 1);  // Only front-facing
}

}  // namespace
}  // namespace aurora::simulation
