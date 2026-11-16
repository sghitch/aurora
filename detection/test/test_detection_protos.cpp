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

/**
 * @file test_detection_protos.cpp
 * @brief Unit tests for the detection protobuf messages.
 *
 * Tests round-trip serialization and configuration defaults.
 */

#include <gtest/gtest.h>

#include "detection.pb.h"
#include "detection_config.pb.h"
#include "detection_frame.pb.h"
#include "doppler_result.pb.h"
#include "vector.pb.h"

namespace aurora::detection {
namespace {

// Test fixture for detection proto tests
class DetectionProtoTest : public ::testing::Test {
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ----------------------------------------------------------------------------
// Detection Proto Tests
// ----------------------------------------------------------------------------

TEST_F(DetectionProtoTest, DetectionRoundtrip) {
    // Create a Detection message with all fields populated
    Detection detection;
    detection.set_range_m(50.5);
    detection.set_azimuth_rad(0.15);

    auto* position = detection.mutable_position_m();
    position->set_x(49.8);
    position->set_y(7.5);
    position->set_z(0.0);

    auto* doppler = detection.mutable_doppler();
    doppler->set_velocity_mps(-12.5);
    doppler->set_confidence(0.95);
    doppler->set_ambiguous(false);
    doppler->set_max_unambiguous_velocity_mps(50.0);

    detection.set_snr_db(25.3);
    detection.set_phase_residual_rad(0.02);

    // Serialize to string
    std::string serialized;
    ASSERT_TRUE(detection.SerializeToString(&serialized));
    ASSERT_FALSE(serialized.empty());

    // Deserialize from string
    Detection restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    // Verify all fields preserved
    EXPECT_DOUBLE_EQ(restored.range_m(), 50.5);
    EXPECT_DOUBLE_EQ(restored.azimuth_rad(), 0.15);

    EXPECT_DOUBLE_EQ(restored.position_m().x(), 49.8);
    EXPECT_DOUBLE_EQ(restored.position_m().y(), 7.5);
    EXPECT_DOUBLE_EQ(restored.position_m().z(), 0.0);

    EXPECT_DOUBLE_EQ(restored.doppler().velocity_mps(), -12.5);
    EXPECT_DOUBLE_EQ(restored.doppler().confidence(), 0.95);
    EXPECT_FALSE(restored.doppler().ambiguous());
    EXPECT_DOUBLE_EQ(restored.doppler().max_unambiguous_velocity_mps(), 50.0);

    EXPECT_DOUBLE_EQ(restored.snr_db(), 25.3);
    EXPECT_DOUBLE_EQ(restored.phase_residual_rad(), 0.02);
}

TEST_F(DetectionProtoTest, DetectionDefaultValues) {
    // Verify default values for an empty Detection
    Detection detection;

    EXPECT_DOUBLE_EQ(detection.range_m(), 0.0);
    EXPECT_DOUBLE_EQ(detection.azimuth_rad(), 0.0);
    EXPECT_DOUBLE_EQ(detection.snr_db(), 0.0);
    EXPECT_DOUBLE_EQ(detection.phase_residual_rad(), 0.0);

    // Nested message should have default values
    EXPECT_DOUBLE_EQ(detection.position_m().x(), 0.0);
    EXPECT_DOUBLE_EQ(detection.doppler().velocity_mps(), 0.0);
}

// ----------------------------------------------------------------------------
// DetectionFrame Proto Tests
// ----------------------------------------------------------------------------

TEST_F(DetectionProtoTest, DetectionFrameRoundtrip) {
    // Create a DetectionFrame with multiple detections
    DetectionFrame frame;
    frame.set_frame_id(12345);
    frame.set_timestamp_s(1704672000.5);

    // Add first detection
    auto* det1 = frame.add_detections();
    det1->set_range_m(30.0);
    det1->set_azimuth_rad(0.1);
    det1->set_snr_db(20.0);

    // Add second detection
    auto* det2 = frame.add_detections();
    det2->set_range_m(45.0);
    det2->set_azimuth_rad(-0.2);
    det2->set_snr_db(18.5);

    // Add third detection
    auto* det3 = frame.add_detections();
    det3->set_range_m(60.0);
    det3->set_azimuth_rad(0.05);
    det3->set_snr_db(22.0);

    frame.set_num_active_range_bins(3);
    frame.set_processing_time_ms(1.25);

    // Serialize to string
    std::string serialized;
    ASSERT_TRUE(frame.SerializeToString(&serialized));
    ASSERT_FALSE(serialized.empty());

    // Deserialize from string
    DetectionFrame restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    // Verify frame metadata
    EXPECT_EQ(restored.frame_id(), 12345u);
    EXPECT_DOUBLE_EQ(restored.timestamp_s(), 1704672000.5);
    EXPECT_EQ(restored.num_active_range_bins(), 3u);
    EXPECT_DOUBLE_EQ(restored.processing_time_ms(), 1.25);

    // Verify detection count
    ASSERT_EQ(restored.detections_size(), 3);

    // Verify first detection
    EXPECT_DOUBLE_EQ(restored.detections(0).range_m(), 30.0);
    EXPECT_DOUBLE_EQ(restored.detections(0).azimuth_rad(), 0.1);
    EXPECT_DOUBLE_EQ(restored.detections(0).snr_db(), 20.0);

    // Verify second detection
    EXPECT_DOUBLE_EQ(restored.detections(1).range_m(), 45.0);
    EXPECT_DOUBLE_EQ(restored.detections(1).azimuth_rad(), -0.2);
    EXPECT_DOUBLE_EQ(restored.detections(1).snr_db(), 18.5);

    // Verify third detection
    EXPECT_DOUBLE_EQ(restored.detections(2).range_m(), 60.0);
    EXPECT_DOUBLE_EQ(restored.detections(2).azimuth_rad(), 0.05);
    EXPECT_DOUBLE_EQ(restored.detections(2).snr_db(), 22.0);
}

TEST_F(DetectionProtoTest, DetectionFrameEmpty) {
    // Test empty detection frame (no detections)
    DetectionFrame frame;
    frame.set_frame_id(1);
    frame.set_timestamp_s(0.0);
    frame.set_num_active_range_bins(0);
    frame.set_processing_time_ms(0.5);

    // Serialize and restore
    std::string serialized;
    ASSERT_TRUE(frame.SerializeToString(&serialized));

    DetectionFrame restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    EXPECT_EQ(restored.detections_size(), 0);
    EXPECT_EQ(restored.num_active_range_bins(), 0u);
}

// ----------------------------------------------------------------------------
// DopplerResult Proto Tests
// ----------------------------------------------------------------------------

TEST_F(DetectionProtoTest, DopplerResultRoundtrip) {
    // Create a DopplerResult message with all fields
    DopplerResult doppler;
    doppler.set_velocity_mps(15.7);
    doppler.set_confidence(0.92);
    doppler.set_ambiguous(true);
    doppler.set_max_unambiguous_velocity_mps(31.0);

    // Serialize to string
    std::string serialized;
    ASSERT_TRUE(doppler.SerializeToString(&serialized));
    ASSERT_FALSE(serialized.empty());

    // Deserialize from string
    DopplerResult restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    // Verify all fields preserved
    EXPECT_DOUBLE_EQ(restored.velocity_mps(), 15.7);
    EXPECT_DOUBLE_EQ(restored.confidence(), 0.92);
    EXPECT_TRUE(restored.ambiguous());
    EXPECT_DOUBLE_EQ(restored.max_unambiguous_velocity_mps(), 31.0);
}

TEST_F(DetectionProtoTest, DopplerResultDefaultValues) {
    // Verify default values for an empty DopplerResult
    DopplerResult doppler;

    EXPECT_DOUBLE_EQ(doppler.velocity_mps(), 0.0);
    EXPECT_DOUBLE_EQ(doppler.confidence(), 0.0);
    EXPECT_FALSE(doppler.ambiguous());
    EXPECT_DOUBLE_EQ(doppler.max_unambiguous_velocity_mps(), 0.0);
}

TEST_F(DetectionProtoTest, DopplerResultNegativeVelocity) {
    // Test negative velocity (receding target)
    DopplerResult doppler;
    doppler.set_velocity_mps(-8.3);
    doppler.set_confidence(0.85);
    doppler.set_ambiguous(false);
    doppler.set_max_unambiguous_velocity_mps(31.0);

    // Serialize and restore
    std::string serialized;
    ASSERT_TRUE(doppler.SerializeToString(&serialized));

    DopplerResult restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    // Verify negative velocity preserved correctly
    EXPECT_DOUBLE_EQ(restored.velocity_mps(), -8.3);
    EXPECT_LT(restored.velocity_mps(), 0.0);
}

// ----------------------------------------------------------------------------
// DetectionConfig Proto Tests
// ----------------------------------------------------------------------------

TEST_F(DetectionProtoTest, DetectionConfigDefaults) {
    // Verify default values for DetectionConfig
    DetectionConfig config;

    // All numeric defaults should be 0/false
    EXPECT_DOUBLE_EQ(config.magnitude_threshold_db(), 0.0);
    EXPECT_EQ(config.num_angular_bins(), 0u);
    EXPECT_EQ(config.min_pixels_for_gradient(), 0u);
    EXPECT_DOUBLE_EQ(config.max_phase_residual_rad(), 0.0);
    EXPECT_DOUBLE_EQ(config.max_condition_number(), 0.0);
    EXPECT_FALSE(config.enable_doppler());
    EXPECT_DOUBLE_EQ(config.min_velocity_mps(), 0.0);
}

TEST_F(DetectionProtoTest, DetectionConfigRoundtrip) {
    // Create config with typical values
    DetectionConfig config;
    config.set_magnitude_threshold_db(10.0);
    config.set_num_angular_bins(16);
    config.set_min_pixels_for_gradient(4);
    config.set_max_phase_residual_rad(0.3);
    config.set_max_condition_number(100.0);
    config.set_enable_doppler(true);
    config.set_min_velocity_mps(0.5);

    // Serialize and restore
    std::string serialized;
    ASSERT_TRUE(config.SerializeToString(&serialized));

    DetectionConfig restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    // Verify all fields
    EXPECT_DOUBLE_EQ(restored.magnitude_threshold_db(), 10.0);
    EXPECT_EQ(restored.num_angular_bins(), 16u);
    EXPECT_EQ(restored.min_pixels_for_gradient(), 4u);
    EXPECT_DOUBLE_EQ(restored.max_phase_residual_rad(), 0.3);
    EXPECT_DOUBLE_EQ(restored.max_condition_number(), 100.0);
    EXPECT_TRUE(restored.enable_doppler());
    EXPECT_DOUBLE_EQ(restored.min_velocity_mps(), 0.5);
}

TEST_F(DetectionProtoTest, DetectionConfigReasonableValues) {
    // Test that config can hold reasonable operational values
    DetectionConfig config;

    // Set typical operational values
    config.set_magnitude_threshold_db(15.0);     // 15 dB above noise floor
    config.set_num_angular_bins(4096);           // Cap on angle-search bins
    config.set_min_pixels_for_gradient(4);       // Need at least 4 pixels
    config.set_max_phase_residual_rad(0.1);      // ~5.7 degrees max residual
    config.set_max_condition_number(50.0);       // Numerical stability limit
    config.set_enable_doppler(true);             // Enable velocity estimation
    config.set_min_velocity_mps(0.1);            // 0.1 m/s minimum detection

    // Verify values are set correctly
    EXPECT_GT(config.magnitude_threshold_db(), 0.0);
    EXPECT_GT(config.num_angular_bins(), 0u);
    EXPECT_GE(config.min_pixels_for_gradient(), 3u);
    EXPECT_LT(config.max_phase_residual_rad(), M_PI);
    EXPECT_GT(config.max_condition_number(), 1.0);
}

}  // namespace
}  // namespace aurora::detection
