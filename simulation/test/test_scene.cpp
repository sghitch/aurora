#include "simulation/scene.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

namespace aurora::simulation {
namespace {

// Helper to create a simple scene definition with a single point target
SceneDefinition create_single_point_scene(
    double x, double y, double z,
    double vx, double vy, double vz,
    double reflectivity_905nm,
    double reflectivity_1550nm) {
    SceneDefinition definition;

    auto* point = definition.add_point_targets();
    point->mutable_position_m()->set_x(x);
    point->mutable_position_m()->set_y(y);
    point->mutable_position_m()->set_z(z);
    point->mutable_velocity_mps()->set_x(vx);
    point->mutable_velocity_mps()->set_y(vy);
    point->mutable_velocity_mps()->set_z(vz);

    auto* surface = point->mutable_surface();
    surface->set_reflectivity_905nm(reflectivity_905nm);
    surface->set_reflectivity_1550nm(reflectivity_1550nm);
    surface->set_roughness_rms_m(1e-6);
    surface->set_correlation_length_m(10e-6);

    return definition;
}

// Helper to create a scene with a plane target
SceneDefinition create_single_plane_scene(
    double cx, double cy, double cz,
    double nx, double ny, double nz,
    double width, double height,
    double reflectivity_905nm,
    double reflectivity_1550nm) {
    SceneDefinition definition;

    auto* plane = definition.add_plane_targets();
    plane->mutable_center_m()->set_x(cx);
    plane->mutable_center_m()->set_y(cy);
    plane->mutable_center_m()->set_z(cz);
    plane->mutable_normal()->set_x(nx);
    plane->mutable_normal()->set_y(ny);
    plane->mutable_normal()->set_z(nz);
    plane->set_width_m(width);
    plane->set_height_m(height);

    auto* surface = plane->mutable_surface();
    surface->set_reflectivity_905nm(reflectivity_905nm);
    surface->set_reflectivity_1550nm(reflectivity_1550nm);
    surface->set_roughness_rms_m(1e-6);
    surface->set_correlation_length_m(10e-6);

    return definition;
}

class SceneTest : public ::testing::Test {
  protected:
    static constexpr double kTolerance = 1e-9;
};

// Test point target range calculation
TEST_F(SceneTest, PointTargetRange) {
    auto definition = create_single_point_scene(
        3.0, 4.0, 0.0,   // position (3, 4, 0) -> range = 5
        0.0, 0.0, 0.0,   // stationary
        0.8, 1.0);

    Scene scene(definition);
    ASSERT_EQ(scene.point_target_count(), 1);

    auto target = scene.point_target(0);
    EXPECT_NEAR(target.range_m(), 5.0, kTolerance);
}

// Test point target range in 3D
TEST_F(SceneTest, PointTargetRange3D) {
    // Position at (1, 2, 2) -> range = 3
    auto definition = create_single_point_scene(
        1.0, 2.0, 2.0,
        0.0, 0.0, 0.0,
        0.8, 1.0);

    Scene scene(definition);
    auto target = scene.point_target(0);
    EXPECT_NEAR(target.range_m(), 3.0, kTolerance);
}

// Test radial velocity calculation
TEST_F(SceneTest, RadialVelocity) {
    // Target at (10, 0, 0), moving in +x direction
    // Radial velocity should equal velocity magnitude
    auto definition = create_single_point_scene(
        10.0, 0.0, 0.0,
        5.0, 0.0, 0.0,   // moving away at 5 m/s
        0.8, 1.0);

    Scene scene(definition);
    auto target = scene.point_target(0);
    EXPECT_NEAR(target.radial_velocity_mps(), 5.0, kTolerance);
}

// Test radial velocity with perpendicular motion
TEST_F(SceneTest, RadialVelocityPerpendicular) {
    // Target at (10, 0, 0), moving in +y direction
    // Radial velocity should be zero
    auto definition = create_single_point_scene(
        10.0, 0.0, 0.0,
        0.0, 5.0, 0.0,   // moving perpendicular
        0.8, 1.0);

    Scene scene(definition);
    auto target = scene.point_target(0);
    EXPECT_NEAR(target.radial_velocity_mps(), 0.0, kTolerance);
}

// Test radial velocity with diagonal motion
TEST_F(SceneTest, RadialVelocityDiagonal) {
    // Target at (1, 1, 0), moving in (1, 1, 0) direction at sqrt(2) m/s
    // Radial velocity should be sqrt(2) m/s (unit vector is (1/sqrt(2), 1/sqrt(2), 0))
    auto definition = create_single_point_scene(
        1.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        0.8, 1.0);

    Scene scene(definition);
    auto target = scene.point_target(0);
    EXPECT_NEAR(target.radial_velocity_mps(), std::sqrt(2.0), kTolerance);
}

// Test position update with velocity
TEST_F(SceneTest, PositionUpdate) {
    auto definition = create_single_point_scene(
        10.0, 20.0, 30.0,
        1.0, 2.0, 3.0,   // velocity
        0.8, 1.0);

    Scene scene(definition);

    // Update by 1 second
    scene.update(1.0);

    auto target = scene.point_target(0);
    Eigen::Vector3d pos = target.position();
    EXPECT_NEAR(pos.x(), 11.0, kTolerance);
    EXPECT_NEAR(pos.y(), 22.0, kTolerance);
    EXPECT_NEAR(pos.z(), 33.0, kTolerance);
}

// Test position update with fractional time
TEST_F(SceneTest, PositionUpdateFractional) {
    auto definition = create_single_point_scene(
        0.0, 0.0, 0.0,
        10.0, 20.0, 30.0,
        0.8, 1.0);

    Scene scene(definition);

    // Update by 0.1 seconds
    scene.update(0.1);

    auto target = scene.point_target(0);
    Eigen::Vector3d pos = target.position();
    EXPECT_NEAR(pos.x(), 1.0, kTolerance);
    EXPECT_NEAR(pos.y(), 2.0, kTolerance);
    EXPECT_NEAR(pos.z(), 3.0, kTolerance);
}

// Test reflectivity at 905nm
TEST_F(SceneTest, Reflectivity905nm) {
    auto definition = create_single_point_scene(
        10.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.8,   // reflectivity at 905nm
        0.4);  // reflectivity at 1550nm

    Scene scene(definition);
    auto target = scene.point_target(0);

    // At 905nm, should return 905nm reflectivity
    double rho = Scene::get_reflectivity(target, Scene::kReferenceWavelength905nm);
    EXPECT_NEAR(rho, 0.8, kTolerance);
}

// Test reflectivity at 1550nm
TEST_F(SceneTest, Reflectivity1550nm) {
    auto definition = create_single_point_scene(
        10.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.8,   // reflectivity at 905nm
        0.4);  // reflectivity at 1550nm

    Scene scene(definition);
    auto target = scene.point_target(0);

    // At 1550nm, should return 1550nm reflectivity
    double rho = Scene::get_reflectivity(target, Scene::kSecondaryWavelength1550nm);
    EXPECT_NEAR(rho, 0.4, kTolerance);
}

// Test empty scene
TEST_F(SceneTest, EmptyScene) {
    SceneDefinition definition;
    Scene scene(definition);

    EXPECT_EQ(scene.point_target_count(), 0);
    EXPECT_EQ(scene.plane_target_count(), 0);
}

// Test environment parameters
TEST_F(SceneTest, EnvironmentParameters) {
    SceneDefinition definition;
    definition.set_ambient_temperature_k(300.0);
    definition.set_atmospheric_extinction_per_m(0.001);
    definition.set_background_radiance_w_per_m2_sr(0.01);

    Scene scene(definition);

    EXPECT_NEAR(scene.ambient_temperature_k(), 300.0, kTolerance);
    EXPECT_NEAR(scene.atmospheric_extinction_per_m(), 0.001, kTolerance);
    EXPECT_NEAR(scene.background_radiance_w_per_m2_sr(), 0.01, kTolerance);
}

// Test that environment parameters read from config (no defaults assumed)
TEST_F(SceneTest, DefaultEnvironmentParameters) {
    SceneDefinition definition;
    Scene scene(definition);

    // With no config values set, environment parameters should be 0
    // (no defaults assumed - values come directly from config)
    EXPECT_EQ(scene.ambient_temperature_k(), 0.0);
    EXPECT_EQ(scene.atmospheric_extinction_per_m(), 0.0);
    EXPECT_EQ(scene.background_radiance_w_per_m2_sr(), 0.0);
}

// Test radial velocity at origin (edge case)
TEST_F(SceneTest, RadialVelocityAtOrigin) {
    auto definition = create_single_point_scene(
        0.0, 0.0, 0.0,   // at origin
        1.0, 0.0, 0.0,   // moving
        0.8, 1.0);

    Scene scene(definition);
    auto target = scene.point_target(0);

    // Radial velocity should be zero when at origin (undefined direction)
    EXPECT_NEAR(target.radial_velocity_mps(), 0.0, kTolerance);
}

// Test plane position update
TEST_F(SceneTest, PlanePositionUpdate) {
    auto definition = create_single_plane_scene(
        10.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 2.0,
        0.8, 1.0);

    // Add velocity to the plane
    definition.mutable_plane_targets(0)->mutable_velocity_mps()->set_x(5.0);

    Scene scene(definition);

    // Update by 1 second
    scene.update(1.0);

    auto plane = scene.plane_target(0);
    EXPECT_NEAR(plane.center().x(), 15.0, kTolerance);  // 10 + 5*1
}

// Test multiple point targets
TEST_F(SceneTest, MultiplePointTargets) {
    SceneDefinition definition;

    for (int i = 0; i < 5; ++i) {
        auto* point = definition.add_point_targets();
        point->mutable_position_m()->set_x(static_cast<double>(i) * 10.0);
        point->mutable_position_m()->set_y(0.0);
        point->mutable_position_m()->set_z(50.0);

        auto* surface = point->mutable_surface();
        surface->set_reflectivity_905nm(0.8);
        surface->set_reflectivity_1550nm(0.8);
        surface->set_roughness_rms_m(1e-6);
        surface->set_correlation_length_m(10e-6);
    }

    Scene scene(definition);
    EXPECT_EQ(scene.point_target_count(), 5);
}

}  // namespace
}  // namespace aurora::simulation
