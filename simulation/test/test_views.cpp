#include "simulation/views.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

namespace aurora::simulation {
namespace {

// =============================================================================
// SurfacePropertiesView Tests
// =============================================================================

class SurfacePropertiesViewTest : public ::testing::Test {
  protected:
    // Helper to create a valid SurfaceProperties proto
    static SurfaceProperties create_valid_surface() {
        SurfaceProperties surface;
        surface.set_reflectivity_905nm(0.8);
        surface.set_reflectivity_1550nm(0.6);
        surface.set_roughness_rms_m(1e-6);
        surface.set_correlation_length_m(10e-6);
        return surface;
    }
};

TEST_F(SurfacePropertiesViewTest, ValidProtoPassesValidation) {
    auto surface = create_valid_surface();
    EXPECT_NO_THROW(SurfacePropertiesView view(surface));
}

TEST_F(SurfacePropertiesViewTest, ProtoAccessor) {
    auto surface = create_valid_surface();
    SurfacePropertiesView view(surface);

    EXPECT_DOUBLE_EQ(view.proto().reflectivity_905nm(), 0.8);
    EXPECT_DOUBLE_EQ(view.proto().reflectivity_1550nm(), 0.6);
}

TEST_F(SurfacePropertiesViewTest, ZeroReflectivity905nmThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.0);  // Invalid: must be > 0
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(10e-6);

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, NegativeReflectivity905nmThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(-0.1);  // Invalid: must be > 0
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(10e-6);

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, GreaterThanOneReflectivity905nmThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(1.1);  // Invalid: must be <= 1
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(10e-6);

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, GreaterThanOneReflectivity1550nmThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(1.1);  // Invalid: must be <= 1
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(10e-6);

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, ZeroReflectivity1550nmThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.0);  // Invalid: must be > 0
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(10e-6);

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, NegativeReflectivity1550nmThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(-0.1);  // Invalid: must be > 0
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(10e-6);

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, NegativeRoughnessThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(-1e-6);  // Invalid: must be >= 0
    surface.set_correlation_length_m(10e-6);

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, ZeroRoughnessIsValid) {
    // Zero roughness represents perfectly specular surface
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(0.0);  // Valid: perfectly specular
    surface.set_correlation_length_m(10e-6);

    EXPECT_NO_THROW(SurfacePropertiesView view(surface));
}

TEST_F(SurfacePropertiesViewTest, ZeroCorrelationLengthThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(0.0);  // Invalid: must be > 0

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, NegativeCorrelationLengthThrows) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(-10e-6);  // Invalid: must be > 0

    EXPECT_THROW(SurfacePropertiesView view(surface), std::invalid_argument);
}

TEST_F(SurfacePropertiesViewTest, BoundaryReflectivityOneIsValid) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(1.0);  // Boundary: exactly 1 is valid
    surface.set_reflectivity_1550nm(1.0);
    surface.set_roughness_rms_m(1e-6);
    surface.set_correlation_length_m(10e-6);

    EXPECT_NO_THROW(SurfacePropertiesView view(surface));
}

// =============================================================================
// SurfacePropertiesView - intrinsic_reflectivity Tests
// =============================================================================

TEST_F(SurfacePropertiesViewTest, IntrinsicReflectivityAt905nm) {
    auto surface = create_valid_surface();
    SurfacePropertiesView view(surface);

    constexpr double kWavelength905nm = 905e-9;
    EXPECT_DOUBLE_EQ(view.intrinsic_reflectivity(kWavelength905nm), 0.8);
}

TEST_F(SurfacePropertiesViewTest, IntrinsicReflectivityAt1550nm) {
    auto surface = create_valid_surface();
    SurfacePropertiesView view(surface);

    constexpr double kWavelength1550nm = 1550e-9;
    EXPECT_DOUBLE_EQ(view.intrinsic_reflectivity(kWavelength1550nm), 0.6);
}

TEST_F(SurfacePropertiesViewTest, IntrinsicReflectivityAtMidpoint) {
    auto surface = create_valid_surface();
    SurfacePropertiesView view(surface);

    constexpr double kWavelength905nm = 905e-9;
    constexpr double kWavelength1550nm = 1550e-9;
    double midpoint = (kWavelength905nm + kWavelength1550nm) / 2.0;

    double expected = (0.8 + 0.6) / 2.0;
    EXPECT_NEAR(view.intrinsic_reflectivity(midpoint), expected, 1e-9);
}

TEST_F(SurfacePropertiesViewTest, IntrinsicReflectivityClampsBelowRange) {
    auto surface = create_valid_surface();
    SurfacePropertiesView view(surface);

    // Wavelength below 905nm should clamp to 905nm reflectivity
    double wavelength_below = 800e-9;
    EXPECT_DOUBLE_EQ(view.intrinsic_reflectivity(wavelength_below), 0.8);
}

TEST_F(SurfacePropertiesViewTest, IntrinsicReflectivityClampsAboveRange) {
    auto surface = create_valid_surface();
    SurfacePropertiesView view(surface);

    // Wavelength above 1550nm should clamp to 1550nm reflectivity
    double wavelength_above = 2000e-9;
    EXPECT_DOUBLE_EQ(view.intrinsic_reflectivity(wavelength_above), 0.6);
}

// =============================================================================
// SurfacePropertiesView - specular_fraction Tests
// =============================================================================

TEST_F(SurfacePropertiesViewTest, SpecularFractionZeroRoughness) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(0.0);  // Perfectly specular
    surface.set_correlation_length_m(10e-6);

    SurfacePropertiesView view(surface);
    EXPECT_DOUBLE_EQ(view.specular_fraction(905e-9), 1.0);
}

TEST_F(SurfacePropertiesViewTest, SpecularFractionLargeRoughness) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(10e-6);  // Very rough (10 μm >> λ)
    surface.set_correlation_length_m(10e-6);

    SurfacePropertiesView view(surface);
    // For roughness >> wavelength, specular fraction approaches 0
    EXPECT_LT(view.specular_fraction(905e-9), 0.01);
}

TEST_F(SurfacePropertiesViewTest, SpecularFractionWavelengthDependence) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(100e-9);  // 100nm roughness
    surface.set_correlation_length_m(10e-6);

    SurfacePropertiesView view(surface);

    // Longer wavelength should have higher specular fraction (smoother)
    double specular_905 = view.specular_fraction(905e-9);
    double specular_1550 = view.specular_fraction(1550e-9);
    EXPECT_GT(specular_1550, specular_905);
}

TEST_F(SurfacePropertiesViewTest, SpecularFractionOverrideTakesPrecedence) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(0.0);  // Would give 1.0 from Rayleigh
    surface.set_correlation_length_m(10e-6);
    surface.set_specular_fraction_override(0.5);

    SurfacePropertiesView view(surface);
    EXPECT_DOUBLE_EQ(view.specular_fraction(905e-9), 0.5);
}

TEST_F(SurfacePropertiesViewTest, SpecularFractionRayleighCriterionCorrect) {
    SurfaceProperties surface;
    surface.set_reflectivity_905nm(0.8);
    surface.set_reflectivity_1550nm(0.6);
    surface.set_roughness_rms_m(50e-9);  // 50nm roughness
    surface.set_correlation_length_m(10e-6);

    SurfacePropertiesView view(surface);

    // Rayleigh criterion: α² = exp(-(4πσ/λ)²)
    double lambda = 905e-9;
    double sigma = 50e-9;
    double x = 4.0 * std::numbers::pi * sigma / lambda;
    double expected = std::exp(-x * x);

    EXPECT_NEAR(view.specular_fraction(lambda), expected, 1e-9);
}

// =============================================================================
// PlaneTargetView Tests
// =============================================================================

class PlaneTargetViewTest : public ::testing::Test {
  protected:
    static constexpr double kTolerance = 1e-9;

    // Helper to create a valid PlaneTargetDefinition proto
    static PlaneTargetDefinition create_valid_plane() {
        PlaneTargetDefinition plane;
        plane.mutable_center_m()->set_x(0.0);
        plane.mutable_center_m()->set_y(0.0);
        plane.mutable_center_m()->set_z(50.0);
        plane.mutable_normal()->set_x(0.0);
        plane.mutable_normal()->set_y(0.0);
        plane.mutable_normal()->set_z(-1.0);
        plane.set_width_m(2.0);
        plane.set_height_m(3.0);
        plane.mutable_velocity_mps()->set_x(0.0);
        plane.mutable_velocity_mps()->set_y(0.0);
        plane.mutable_velocity_mps()->set_z(0.0);

        auto* surface = plane.mutable_surface();
        surface->set_reflectivity_905nm(0.8);
        surface->set_reflectivity_1550nm(0.6);
        surface->set_roughness_rms_m(1e-6);
        surface->set_correlation_length_m(10e-6);

        return plane;
    }
};

TEST_F(PlaneTargetViewTest, ValidProtoPassesValidation) {
    auto plane = create_valid_plane();
    EXPECT_NO_THROW(PlaneTargetView view(plane));
}

TEST_F(PlaneTargetViewTest, AreaComputation) {
    auto plane = create_valid_plane();
    PlaneTargetView view(plane);

    EXPECT_NEAR(view.area_m2(), 6.0, kTolerance);  // 2.0 * 3.0
}

TEST_F(PlaneTargetViewTest, RangeComputation) {
    auto plane = create_valid_plane();
    PlaneTargetView view(plane);

    EXPECT_NEAR(view.range_m(), 50.0, kTolerance);
}

TEST_F(PlaneTargetViewTest, CenterVectorConversion) {
    auto plane = create_valid_plane();
    plane.mutable_center_m()->set_x(10.0);
    plane.mutable_center_m()->set_y(20.0);
    plane.mutable_center_m()->set_z(30.0);

    PlaneTargetView view(plane);
    auto center = view.center();

    EXPECT_NEAR(center.x(), 10.0, kTolerance);
    EXPECT_NEAR(center.y(), 20.0, kTolerance);
    EXPECT_NEAR(center.z(), 30.0, kTolerance);
}

TEST_F(PlaneTargetViewTest, NormalVectorConversion) {
    auto plane = create_valid_plane();
    PlaneTargetView view(plane);
    auto normal = view.normal();

    EXPECT_NEAR(normal.x(), 0.0, kTolerance);
    EXPECT_NEAR(normal.y(), 0.0, kTolerance);
    EXPECT_NEAR(normal.z(), -1.0, kTolerance);
}

TEST_F(PlaneTargetViewTest, VelocityVectorConversion) {
    auto plane = create_valid_plane();
    plane.mutable_velocity_mps()->set_x(1.0);
    plane.mutable_velocity_mps()->set_y(2.0);
    plane.mutable_velocity_mps()->set_z(3.0);

    PlaneTargetView view(plane);
    auto velocity = view.velocity();

    EXPECT_NEAR(velocity.x(), 1.0, kTolerance);
    EXPECT_NEAR(velocity.y(), 2.0, kTolerance);
    EXPECT_NEAR(velocity.z(), 3.0, kTolerance);
}

TEST_F(PlaneTargetViewTest, SurfaceAccessor) {
    auto plane = create_valid_plane();
    PlaneTargetView view(plane);
    auto surface = view.surface();

    EXPECT_DOUBLE_EQ(surface.proto().reflectivity_905nm(), 0.8);
    EXPECT_DOUBLE_EQ(surface.proto().reflectivity_1550nm(), 0.6);
}

TEST_F(PlaneTargetViewTest, MissingSurfaceThrows) {
    PlaneTargetDefinition plane;
    plane.mutable_center_m()->set_z(50.0);
    plane.mutable_normal()->set_z(-1.0);
    plane.set_width_m(2.0);
    plane.set_height_m(3.0);
    // No surface set

    EXPECT_THROW(PlaneTargetView view(plane), std::invalid_argument);
}

TEST_F(PlaneTargetViewTest, ZeroWidthThrows) {
    auto plane = create_valid_plane();
    plane.set_width_m(0.0);

    EXPECT_THROW(PlaneTargetView view(plane), std::invalid_argument);
}

TEST_F(PlaneTargetViewTest, NegativeWidthThrows) {
    auto plane = create_valid_plane();
    plane.set_width_m(-1.0);

    EXPECT_THROW(PlaneTargetView view(plane), std::invalid_argument);
}

TEST_F(PlaneTargetViewTest, ZeroHeightThrows) {
    auto plane = create_valid_plane();
    plane.set_height_m(0.0);

    EXPECT_THROW(PlaneTargetView view(plane), std::invalid_argument);
}

TEST_F(PlaneTargetViewTest, NegativeHeightThrows) {
    auto plane = create_valid_plane();
    plane.set_height_m(-1.0);

    EXPECT_THROW(PlaneTargetView view(plane), std::invalid_argument);
}

TEST_F(PlaneTargetViewTest, ZeroNormalThrows) {
    auto plane = create_valid_plane();
    plane.mutable_normal()->set_x(0.0);
    plane.mutable_normal()->set_y(0.0);
    plane.mutable_normal()->set_z(0.0);

    EXPECT_THROW(PlaneTargetView view(plane), std::invalid_argument);
}

TEST_F(PlaneTargetViewTest, IsGeometryOnlyWrapper) {
    // This test documents the API contract: PlaneTargetView provides
    // geometry accessors only, no physics computation.
    // Physics computation is handled by FmcwSimulator.

    auto plane = create_valid_plane();
    PlaneTargetView view(plane);

    // Geometry accessors work
    EXPECT_DOUBLE_EQ(view.width_m(), 2.0);
    EXPECT_DOUBLE_EQ(view.height_m(), 3.0);
    EXPECT_DOUBLE_EQ(view.area_m2(), 6.0);
    EXPECT_DOUBLE_EQ(view.range_m(), 50.0);

    // Verify center and normal accessors
    auto center = view.center();
    EXPECT_NEAR(center.z(), 50.0, kTolerance);

    auto normal = view.normal();
    EXPECT_NEAR(normal.z(), -1.0, kTolerance);

    // Verify surface accessor provides surface properties
    auto surface = view.surface();
    EXPECT_DOUBLE_EQ(surface.proto().reflectivity_905nm(), 0.8);
}

// =============================================================================
// PointTargetView Tests
// =============================================================================

class PointTargetViewTest : public ::testing::Test {
  protected:
    static constexpr double kTolerance = 1e-9;

    // Helper to create a valid PointTargetDefinition proto
    static PointTargetDefinition create_valid_point() {
        PointTargetDefinition point;
        point.mutable_position_m()->set_x(3.0);
        point.mutable_position_m()->set_y(4.0);
        point.mutable_position_m()->set_z(0.0);
        point.mutable_velocity_mps()->set_x(0.0);
        point.mutable_velocity_mps()->set_y(0.0);
        point.mutable_velocity_mps()->set_z(0.0);

        auto* surface = point.mutable_surface();
        surface->set_reflectivity_905nm(0.8);
        surface->set_reflectivity_1550nm(0.6);
        surface->set_roughness_rms_m(1e-6);
        surface->set_correlation_length_m(10e-6);

        return point;
    }
};

TEST_F(PointTargetViewTest, ValidProtoPassesValidation) {
    auto point = create_valid_point();
    EXPECT_NO_THROW(PointTargetView view(point));
}

TEST_F(PointTargetViewTest, RangeComputation) {
    auto point = create_valid_point();
    PointTargetView view(point);

    // (3, 4, 0) -> range = 5
    EXPECT_NEAR(view.range_m(), 5.0, kTolerance);
}

TEST_F(PointTargetViewTest, Range3DComputation) {
    auto point = create_valid_point();
    point.mutable_position_m()->set_x(1.0);
    point.mutable_position_m()->set_y(2.0);
    point.mutable_position_m()->set_z(2.0);

    PointTargetView view(point);

    // (1, 2, 2) -> range = 3
    EXPECT_NEAR(view.range_m(), 3.0, kTolerance);
}

TEST_F(PointTargetViewTest, RadialVelocityAwayFromOrigin) {
    auto point = create_valid_point();
    point.mutable_position_m()->set_x(10.0);
    point.mutable_position_m()->set_y(0.0);
    point.mutable_position_m()->set_z(0.0);
    point.mutable_velocity_mps()->set_x(5.0);  // Moving away in +x
    point.mutable_velocity_mps()->set_y(0.0);
    point.mutable_velocity_mps()->set_z(0.0);

    PointTargetView view(point);

    EXPECT_NEAR(view.radial_velocity_mps(), 5.0, kTolerance);
}

TEST_F(PointTargetViewTest, RadialVelocityPerpendicular) {
    auto point = create_valid_point();
    point.mutable_position_m()->set_x(10.0);
    point.mutable_position_m()->set_y(0.0);
    point.mutable_position_m()->set_z(0.0);
    point.mutable_velocity_mps()->set_x(0.0);
    point.mutable_velocity_mps()->set_y(5.0);  // Moving perpendicular
    point.mutable_velocity_mps()->set_z(0.0);

    PointTargetView view(point);

    EXPECT_NEAR(view.radial_velocity_mps(), 0.0, kTolerance);
}

TEST_F(PointTargetViewTest, RadialVelocityDiagonal) {
    auto point = create_valid_point();
    point.mutable_position_m()->set_x(1.0);
    point.mutable_position_m()->set_y(1.0);
    point.mutable_position_m()->set_z(0.0);
    point.mutable_velocity_mps()->set_x(1.0);
    point.mutable_velocity_mps()->set_y(1.0);
    point.mutable_velocity_mps()->set_z(0.0);

    PointTargetView view(point);

    // Velocity (1,1,0) along direction (1,1,0)/sqrt(2) = sqrt(2)
    EXPECT_NEAR(view.radial_velocity_mps(), std::sqrt(2.0), kTolerance);
}

TEST_F(PointTargetViewTest, RadialVelocityAtOrigin) {
    auto point = create_valid_point();
    point.mutable_position_m()->set_x(0.0);
    point.mutable_position_m()->set_y(0.0);
    point.mutable_position_m()->set_z(0.0);
    point.mutable_velocity_mps()->set_x(1.0);

    PointTargetView view(point);

    // At origin, radial velocity is undefined -> returns 0
    EXPECT_NEAR(view.radial_velocity_mps(), 0.0, kTolerance);
}

TEST_F(PointTargetViewTest, PositionVectorConversion) {
    auto point = create_valid_point();
    PointTargetView view(point);
    auto position = view.position();

    EXPECT_NEAR(position.x(), 3.0, kTolerance);
    EXPECT_NEAR(position.y(), 4.0, kTolerance);
    EXPECT_NEAR(position.z(), 0.0, kTolerance);
}

TEST_F(PointTargetViewTest, VelocityVectorConversion) {
    auto point = create_valid_point();
    point.mutable_velocity_mps()->set_x(1.0);
    point.mutable_velocity_mps()->set_y(2.0);
    point.mutable_velocity_mps()->set_z(3.0);

    PointTargetView view(point);
    auto velocity = view.velocity();

    EXPECT_NEAR(velocity.x(), 1.0, kTolerance);
    EXPECT_NEAR(velocity.y(), 2.0, kTolerance);
    EXPECT_NEAR(velocity.z(), 3.0, kTolerance);
}

TEST_F(PointTargetViewTest, SurfaceAccessor) {
    auto point = create_valid_point();
    PointTargetView view(point);
    auto surface = view.surface();

    EXPECT_DOUBLE_EQ(surface.proto().reflectivity_905nm(), 0.8);
    EXPECT_DOUBLE_EQ(surface.proto().reflectivity_1550nm(), 0.6);
}

TEST_F(PointTargetViewTest, MissingSurfaceThrows) {
    PointTargetDefinition point;
    point.mutable_position_m()->set_x(10.0);
    point.mutable_position_m()->set_y(0.0);
    point.mutable_position_m()->set_z(0.0);
    // No surface set

    EXPECT_THROW(PointTargetView view(point), std::invalid_argument);
}

}  // namespace
}  // namespace aurora::simulation
