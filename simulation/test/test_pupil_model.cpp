#include "simulation/pupil_model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

namespace aurora::simulation {
namespace {

// Create a PupilConfig with default values
aurora::forward::PupilConfig create_default_config() {
    aurora::forward::PupilConfig config;
    config.set_fov_half_angle_rad(0.524);   // 30 degrees
    config.set_rolloff_width_rad(0.087);    // 5 degrees
    config.set_shape(aurora::forward::PupilConfig::PUPIL_CIRCULAR);
    return config;
}

// Create a rectangular PupilConfig
aurora::forward::PupilConfig create_rectangular_config(double aspect_ratio) {
    aurora::forward::PupilConfig config;
    config.set_fov_half_angle_rad(0.524);   // 30 degrees
    config.set_rolloff_width_rad(0.087);    // 5 degrees
    config.set_shape(aurora::forward::PupilConfig::PUPIL_RECTANGULAR);
    config.set_aspect_ratio(aspect_ratio);
    return config;
}

class PupilModelTest : public ::testing::Test {
  protected:
    static constexpr double kTolerance = 1e-9;
    static constexpr double kLooseTolerance = 1e-6;
};

// Test: Center of FOV has full transmission
TEST_F(PupilModelTest, CenterTransmission) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // On-axis target (azimuth = 0, elevation = 0)
    double transmission = pupil.compute_transmission(0.0, 0.0);
    EXPECT_NEAR(transmission, 1.0, kTolerance);
}

// Test: Center via 3D position
TEST_F(PupilModelTest, CenterTransmissionFromPosition) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Target directly ahead at 100m
    Eigen::Vector3d position(0.0, 0.0, 100.0);
    double transmission = pupil.compute_transmission(position);
    EXPECT_NEAR(transmission, 1.0, kTolerance);
}

// Test: Edge of FOV (at fov_half_angle) has zero transmission
TEST_F(PupilModelTest, EdgeTransmission) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // At exact FOV boundary
    double transmission = pupil.compute_transmission(0.524, 0.0);
    EXPECT_NEAR(transmission, 0.0, kTolerance);
}

// Test: Outside FOV has zero transmission
TEST_F(PupilModelTest, OutsideTransmission) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Well beyond FOV boundary
    double transmission = pupil.compute_transmission(1.0, 0.0);
    EXPECT_NEAR(transmission, 0.0, kTolerance);
}

// Test: Rolloff region has intermediate transmission
TEST_F(PupilModelTest, RolloffTransmission) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Edge angle = 0.524 - 0.087 = 0.437 rad (25 degrees)
    // Midpoint of rolloff = 0.437 + 0.0435 = 0.4805 rad
    double midpoint_angle = 0.524 - 0.087 / 2.0;
    double transmission = pupil.compute_transmission(midpoint_angle, 0.0);

    // At midpoint of cosine rolloff, transmission should be 0.5
    EXPECT_NEAR(transmission, 0.5, 0.01);
}

// Test: Just inside rolloff edge has full transmission
TEST_F(PupilModelTest, JustInsideRolloffEdge) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Just inside the rolloff region (edge_angle - epsilon)
    double edge_angle = 0.524 - 0.087;  // 0.437 rad
    double transmission = pupil.compute_transmission(edge_angle - 0.001, 0.0);

    EXPECT_NEAR(transmission, 1.0, kLooseTolerance);
}

// Test: Rolloff is smooth (no discontinuities)
TEST_F(PupilModelTest, RolloffSmoothness) {
    auto config = create_default_config();
    PupilModel pupil(config);

    double edge_angle = 0.524 - 0.087;  // 0.437 rad
    double prev_transmission = pupil.compute_transmission(edge_angle - 0.01, 0.0);

    // Sample through rolloff region
    for (double theta = edge_angle; theta <= 0.524; theta += 0.001) {
        double transmission = pupil.compute_transmission(theta, 0.0);

        // Transmission should decrease monotonically
        EXPECT_LE(transmission, prev_transmission + kLooseTolerance);

        // Step size should be reasonable (no jumps > 0.05)
        EXPECT_LT(prev_transmission - transmission, 0.05);

        prev_transmission = transmission;
    }
}

// Test: Rolloff follows cosine function
TEST_F(PupilModelTest, RolloffCosineShape) {
    auto config = create_default_config();
    PupilModel pupil(config);

    double edge_angle = 0.524 - 0.087;  // 0.437 rad

    // Test at t = 0.25, 0.5, 0.75 in the rolloff region
    for (double t : {0.25, 0.5, 0.75}) {
        double theta = edge_angle + t * 0.087;
        double actual = pupil.compute_transmission(theta, 0.0);
        double expected = 0.5 * (1.0 + std::cos(std::numbers::pi * t));
        EXPECT_NEAR(actual, expected, 0.01);
    }
}

// Test: Rectangular FOV corners
TEST_F(PupilModelTest, RectangularFovCorners) {
    auto config = create_rectangular_config(2.0);  // 2:1 aspect ratio
    PupilModel pupil(config);

    // Half-width = 0.524 rad, half-height = 0.262 rad (due to aspect ratio)
    // Inside the corner
    double transmission = pupil.compute_transmission(0.3, 0.15);
    EXPECT_GT(transmission, 0.0);

    // Outside horizontally
    transmission = pupil.compute_transmission(0.6, 0.0);
    EXPECT_NEAR(transmission, 0.0, kTolerance);

    // Outside vertically
    transmission = pupil.compute_transmission(0.0, 0.35);
    EXPECT_NEAR(transmission, 0.0, kTolerance);
}

// Test: Target behind sensor (negative z) has zero transmission
TEST_F(PupilModelTest, NegativeRangeTransmission) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Target behind the sensor
    Eigen::Vector3d position(0.0, 0.0, -10.0);
    double transmission = pupil.compute_transmission(position);
    EXPECT_NEAR(transmission, 0.0, kTolerance);
}

// Test: Target at z = 0 (in sensor plane) has zero transmission
TEST_F(PupilModelTest, ZeroRangeTransmission) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Target in the sensor plane
    Eigen::Vector3d position(1.0, 1.0, 0.0);
    double transmission = pupil.compute_transmission(position);
    EXPECT_NEAR(transmission, 0.0, kTolerance);
}

// Test: is_in_fov matches compute_transmission
TEST_F(PupilModelTest, IsInFovConsistency) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Inside FOV
    Eigen::Vector3d inside_pos(0.0, 0.0, 100.0);
    EXPECT_TRUE(pupil.is_in_fov(inside_pos));
    EXPECT_GT(pupil.compute_transmission(inside_pos), 0.0);

    // Outside FOV
    Eigen::Vector3d outside_pos(100.0, 0.0, 50.0);  // Large angle
    EXPECT_FALSE(pupil.is_in_fov(outside_pos));
    EXPECT_NEAR(pupil.compute_transmission(outside_pos), 0.0, kTolerance);
}

// Test: Angular symmetry (circular FOV)
TEST_F(PupilModelTest, CircularSymmetry) {
    auto config = create_default_config();
    PupilModel pupil(config);

    double fixed_angle = 0.3;  // Inside FOV

    // Test at multiple azimuthal angles
    for (double phi = 0; phi < 2 * std::numbers::pi; phi += 0.1) {
        double az = fixed_angle * std::cos(phi);
        double el = fixed_angle * std::sin(phi);
        double transmission = pupil.compute_transmission(az, el);

        // All should have same transmission (within tolerance)
        double expected = pupil.compute_transmission(fixed_angle, 0.0);
        EXPECT_NEAR(transmission, expected, kLooseTolerance);
    }
}

// Test: Position to angle conversion
TEST_F(PupilModelTest, PositionToAngleConversion) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Target at 45 degrees off-axis in azimuth
    double range = 100.0;
    double x_offset = range * std::tan(0.3);  // 0.3 rad ~ 17 degrees
    Eigen::Vector3d position(x_offset, 0.0, range);

    // Should be equivalent to angular transmission at 0.3 rad
    double from_position = pupil.compute_transmission(position);
    double from_angle = pupil.compute_transmission(0.3, 0.0);

    EXPECT_NEAR(from_position, from_angle, 0.01);
}

// Test: FOV boundary generation (circular)
TEST_F(PupilModelTest, FovBoundaryCircular) {
    auto config = create_default_config();
    PupilModel pupil(config);

    auto boundary = pupil.get_fov_boundary(100);

    EXPECT_EQ(boundary.size(), 100);

    // All points should be at the FOV half-angle radius
    for (const auto& [az, el] : boundary) {
        double radius = std::sqrt(az * az + el * el);
        EXPECT_NEAR(radius, 0.524, kLooseTolerance);
    }
}

// Test: FOV boundary generation (rectangular)
TEST_F(PupilModelTest, FovBoundaryRectangular) {
    auto config = create_rectangular_config(2.0);
    PupilModel pupil(config);

    auto boundary = pupil.get_fov_boundary(100);

    EXPECT_GE(boundary.size(), 96);  // 4 sides * ~25 points

    // Check that boundary contains corners
    double half_width = 0.524;
    double half_height = 0.524 / 2.0;

    bool found_top_right = false;
    for (const auto& [az, el] : boundary) {
        if (std::abs(az - half_width) < 0.1 &&
            std::abs(el - half_height) < 0.1) {
            found_top_right = true;
            break;
        }
    }
    EXPECT_TRUE(found_top_right);
}

// Test: Getters return correct values
TEST_F(PupilModelTest, Getters) {
    auto config = create_default_config();
    PupilModel pupil(config);

    EXPECT_NEAR(pupil.fov_half_angle_rad(), 0.524, kTolerance);
    EXPECT_NEAR(pupil.rolloff_width_rad(), 0.087, kTolerance);
    EXPECT_TRUE(pupil.is_circular());
}

// Test: Rectangular shape flag
TEST_F(PupilModelTest, RectangularShapeFlag) {
    auto config = create_rectangular_config(1.5);
    PupilModel pupil(config);

    EXPECT_FALSE(pupil.is_circular());
}

// Test: Zero rolloff width (hard edge)
TEST_F(PupilModelTest, ZeroRolloffWidth) {
    aurora::forward::PupilConfig config;
    config.set_fov_half_angle_rad(0.5);
    config.set_rolloff_width_rad(0.0);  // Hard edge
    config.set_shape(aurora::forward::PupilConfig::PUPIL_CIRCULAR);

    PupilModel pupil(config);

    // Just inside FOV
    EXPECT_NEAR(pupil.compute_transmission(0.499, 0.0), 1.0, kTolerance);

    // Just outside FOV
    EXPECT_NEAR(pupil.compute_transmission(0.501, 0.0), 0.0, kTolerance);
}

// Test: Very small FOV
TEST_F(PupilModelTest, SmallFov) {
    aurora::forward::PupilConfig config;
    config.set_fov_half_angle_rad(0.01);   // ~0.57 degrees
    config.set_rolloff_width_rad(0.002);
    config.set_shape(aurora::forward::PupilConfig::PUPIL_CIRCULAR);

    PupilModel pupil(config);

    // Center should have full transmission
    EXPECT_NEAR(pupil.compute_transmission(0.0, 0.0), 1.0, kTolerance);

    // Just outside
    EXPECT_NEAR(pupil.compute_transmission(0.02, 0.0), 0.0, kTolerance);
}

// Test: Diagonal target in circular FOV
TEST_F(PupilModelTest, DiagonalTarget) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Target at 45-degree diagonal
    double angle = 0.3;  // Inside FOV
    double az = angle / std::sqrt(2.0);
    double el = angle / std::sqrt(2.0);

    double transmission = pupil.compute_transmission(az, el);

    // Should equal transmission at same radial distance
    double radial_transmission = pupil.compute_transmission(angle, 0.0);
    EXPECT_NEAR(transmission, radial_transmission, kLooseTolerance);
}

// Test: Large range doesn't affect angular calculation
TEST_F(PupilModelTest, LargeRangeAngularCalculation) {
    auto config = create_default_config();
    PupilModel pupil(config);

    // Same angular position at different ranges
    double angle = 0.2;

    for (double range : {10.0, 100.0, 1000.0, 10000.0}) {
        double x = range * std::tan(angle);
        Eigen::Vector3d pos(x, 0.0, range);
        double transmission = pupil.compute_transmission(pos);
        double expected = pupil.compute_transmission(angle, 0.0);

        EXPECT_NEAR(transmission, expected, 0.01);
    }
}

}  // namespace
}  // namespace aurora::simulation
