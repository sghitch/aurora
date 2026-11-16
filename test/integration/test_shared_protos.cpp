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
#include <cmath>
#include <string>

#include "complex.pb.h"
#include "vector.pb.h"
#include "axis.pb.h"
#include "measurement_tensor.pb.h"
#include "calibration_artifact.pb.h"

namespace aurora::shared {

TEST(ComplexProtoTest, SerializationRoundtrip) {
    Complex original;
    original.set_real(3.14159265358979);
    original.set_imag(2.71828182845905);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    Complex deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_DOUBLE_EQ(original.real(), deserialized.real());
    EXPECT_DOUBLE_EQ(original.imag(), deserialized.imag());
}

TEST(ComplexProtoTest, DefaultValues) {
    Complex c;
    EXPECT_DOUBLE_EQ(0.0, c.real());
    EXPECT_DOUBLE_EQ(0.0, c.imag());
}

TEST(Vector3ProtoTest, SerializationRoundtrip) {
    Vector3 original;
    original.set_x(1.0);
    original.set_y(2.0);
    original.set_z(3.0);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    Vector3 deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_DOUBLE_EQ(original.x(), deserialized.x());
    EXPECT_DOUBLE_EQ(original.y(), deserialized.y());
    EXPECT_DOUBLE_EQ(original.z(), deserialized.z());
}

TEST(Vector3ProtoTest, DefaultValues) {
    Vector3 v;
    EXPECT_DOUBLE_EQ(0.0, v.x());
    EXPECT_DOUBLE_EQ(0.0, v.y());
    EXPECT_DOUBLE_EQ(0.0, v.z());
}

TEST(AxisProtoTest, SerializationRoundtrip) {
    Axis original;
    original.set_unit("m");
    original.set_start(0.0);
    original.set_step(0.5);
    original.set_count(1024);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    Axis deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_EQ(original.unit(), deserialized.unit());
    EXPECT_DOUBLE_EQ(original.start(), deserialized.start());
    EXPECT_DOUBLE_EQ(original.step(), deserialized.step());
    EXPECT_EQ(original.count(), deserialized.count());
}

TEST(AxisProtoTest, RangeAxisExample) {
    // Example: Range axis for FMCW radar
    // 1024 range bins, 0.15m resolution, starting at 0m
    Axis range_axis;
    range_axis.set_unit("m");
    range_axis.set_start(0.0);
    range_axis.set_step(0.15);
    range_axis.set_count(1024);

    // Verify we can compute the range extent
    double max_range = range_axis.start() + range_axis.step() * (range_axis.count() - 1);
    EXPECT_NEAR(153.45, max_range, 0.01);
}

TEST(AxisProtoTest, DopplerAxisExample) {
    // Example: Doppler axis
    // 256 Doppler bins, centered at 0 m/s
    Axis doppler_axis;
    doppler_axis.set_unit("m/s");
    doppler_axis.set_start(-50.0);
    doppler_axis.set_step(0.390625);  // 100 m/s / 256 bins
    doppler_axis.set_count(256);

    std::string serialized;
    ASSERT_TRUE(doppler_axis.SerializeToString(&serialized));

    Axis deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_EQ("m/s", deserialized.unit());
    EXPECT_EQ(256u, deserialized.count());
}

TEST(KVectorProtoTest, SerializationRoundtrip) {
    KVector original;
    original.set_kx(1500.0);  // rad/m
    original.set_ky(-750.0);  // rad/m

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    KVector deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_DOUBLE_EQ(original.kx(), deserialized.kx());
    EXPECT_DOUBLE_EQ(original.ky(), deserialized.ky());
}

TEST(KVectorProtoTest, DefaultValues) {
    KVector k;
    EXPECT_DOUBLE_EQ(0.0, k.kx());
    EXPECT_DOUBLE_EQ(0.0, k.ky());
}

TEST(DetectorChannelProtoTest, SerializationRoundtrip) {
    DetectorChannel original;
    original.set_detector_index(3);

    auto* k = original.mutable_k_vector();
    k->set_kx(1000.0);
    k->set_ky(500.0);

    // Add some range data (4 range bins)
    for (int i = 0; i < 4; ++i) {
        auto* c = original.add_range_data();
        c->set_real(static_cast<double>(i) * 0.1);
        c->set_imag(static_cast<double>(i) * 0.2);
    }

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    DetectorChannel deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_EQ(3u, deserialized.detector_index());
    EXPECT_DOUBLE_EQ(1000.0, deserialized.k_vector().kx());
    EXPECT_DOUBLE_EQ(500.0, deserialized.k_vector().ky());
    ASSERT_EQ(4, deserialized.range_data_size());
    // range_data(1) corresponds to i=1: real=0.1, imag=0.2
    EXPECT_DOUBLE_EQ(0.1, deserialized.range_data(1).real());
    EXPECT_DOUBLE_EQ(0.2, deserialized.range_data(1).imag());
}

TEST(WavelengthChannelProtoTest, SerializationRoundtrip) {
    WavelengthChannel original;
    original.set_wavelength_m(905e-9);  // 905 nm

    // Add 2 detectors
    for (uint32_t d = 0; d < 2; ++d) {
        auto* det = original.add_detectors();
        det->set_detector_index(d);
        det->mutable_k_vector()->set_kx(1000.0 * (d + 1));
        det->mutable_k_vector()->set_ky(0.0);
    }

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    WavelengthChannel deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_DOUBLE_EQ(905e-9, deserialized.wavelength_m());
    ASSERT_EQ(2, deserialized.detectors_size());
    EXPECT_EQ(0u, deserialized.detectors(0).detector_index());
    EXPECT_EQ(1u, deserialized.detectors(1).detector_index());
}

TEST(MeasurementTensorProtoTest, SerializationRoundtrip) {
    MeasurementTensor original;

    // Set up range axis: 256 bins, 0.15m resolution
    auto* range_axis = original.mutable_range_axis();
    range_axis->set_unit("m");
    range_axis->set_start(0.0);
    range_axis->set_step(0.15);
    range_axis->set_count(256);

    // Set metadata
    original.set_burst_id(1001);
    original.set_timestamp_s(1702459200.0);  // Example epoch time

    // Add one wavelength channel
    auto* wl_ch = original.add_data();
    wl_ch->set_wavelength_m(1550e-9);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    MeasurementTensor deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_EQ("m", deserialized.range_axis().unit());
    EXPECT_DOUBLE_EQ(0.15, deserialized.range_axis().step());
    EXPECT_EQ(256u, deserialized.range_axis().count());
    EXPECT_EQ(1001u, deserialized.burst_id());
    EXPECT_DOUBLE_EQ(1702459200.0, deserialized.timestamp_s());
    ASSERT_EQ(1, deserialized.data_size());
    EXPECT_DOUBLE_EQ(1550e-9, deserialized.data(0).wavelength_m());
}

TEST(MeasurementTensorProtoTest, OptionalDopplerAxis) {
    // Test with Doppler axis present
    MeasurementTensor with_doppler;
    with_doppler.mutable_range_axis()->set_unit("m");
    with_doppler.mutable_range_axis()->set_count(256);

    auto* doppler = with_doppler.mutable_doppler_axis();
    doppler->set_unit("m/s");
    doppler->set_start(-50.0);
    doppler->set_step(0.78125);
    doppler->set_count(128);

    EXPECT_TRUE(with_doppler.has_doppler_axis());
    EXPECT_EQ("m/s", with_doppler.doppler_axis().unit());
    EXPECT_EQ(128u, with_doppler.doppler_axis().count());

    // Test without Doppler axis
    MeasurementTensor without_doppler;
    without_doppler.mutable_range_axis()->set_unit("m");
    without_doppler.mutable_range_axis()->set_count(256);

    EXPECT_FALSE(without_doppler.has_doppler_axis());
}

TEST(MeasurementTensorProtoTest, RealisticMeasurement) {
    // Simulate a realistic measurement:
    // - 2 wavelengths (905nm, 1550nm), 8 detectors, 64 range bins
    constexpr int kNumWavelengths = 2;
    constexpr int kNumDetectors = 8;
    constexpr int kNumRangeBins = 64;
    constexpr double kWavelengths[] = {905e-9, 1550e-9};

    MeasurementTensor tensor;

    // Range axis
    auto* range_axis = tensor.mutable_range_axis();
    range_axis->set_unit("m");
    range_axis->set_start(0.0);
    range_axis->set_step(0.15);
    range_axis->set_count(kNumRangeBins);

    tensor.set_burst_id(42);
    tensor.set_timestamp_s(0.0);

    // Build the full tensor structure
    for (int w = 0; w < kNumWavelengths; ++w) {
        auto* wl_ch = tensor.add_data();
        wl_ch->set_wavelength_m(kWavelengths[w]);

        for (int d = 0; d < kNumDetectors; ++d) {
            auto* det_ch = wl_ch->add_detectors();
            det_ch->set_detector_index(d);

            // K-vector depends on detector position and wavelength
            double det_pos = (d - 3.5) * 0.5e-3;  // 0.5mm pitch, centered
            double focal_length = 0.1;            // 100mm
            double k = 2.0 * 3.14159265358979 * det_pos / (focal_length * kWavelengths[w]);
            det_ch->mutable_k_vector()->set_kx(k);
            det_ch->mutable_k_vector()->set_ky(0.0);

            // Add range data
            for (int r = 0; r < kNumRangeBins; ++r) {
                auto* c = det_ch->add_range_data();
                c->set_real(0.0);
                c->set_imag(0.0);
            }
        }
    }

    // Serialize and verify structure
    std::string serialized;
    ASSERT_TRUE(tensor.SerializeToString(&serialized));

    MeasurementTensor deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    // Verify dimensions
    ASSERT_EQ(kNumWavelengths, deserialized.data_size());
    ASSERT_EQ(kNumDetectors, deserialized.data(0).detectors_size());
    ASSERT_EQ(kNumRangeBins, deserialized.data(0).detectors(0).range_data_size());

    // Verify K-vector scaling with wavelength
    // K should be larger for shorter wavelength (905nm vs 1550nm)
    double k_905 = deserialized.data(0).detectors(7).k_vector().kx();
    double k_1550 = deserialized.data(1).detectors(7).k_vector().kx();
    EXPECT_GT(std::abs(k_905), std::abs(k_1550));
    EXPECT_NEAR(k_905 / k_1550, 1550e-9 / 905e-9, 0.001);
}

// ============================================================================
// CalibrationArtifact Tests
// ============================================================================

TEST(DetectorCalibrationProtoTest, SerializationRoundtrip) {
    DetectorCalibration original;
    original.set_detector_index(3);
    original.mutable_complex_gain()->set_real(0.95);
    original.mutable_complex_gain()->set_imag(0.05);
    original.set_timing_skew_s(1.5e-9);  // 1.5 ns timing offset

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    DetectorCalibration deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_EQ(3u, deserialized.detector_index());
    EXPECT_DOUBLE_EQ(0.95, deserialized.complex_gain().real());
    EXPECT_DOUBLE_EQ(0.05, deserialized.complex_gain().imag());
    EXPECT_DOUBLE_EQ(1.5e-9, deserialized.timing_skew_s());
}

TEST(DetectorCalibrationProtoTest, DefaultValues) {
    DetectorCalibration dc;
    EXPECT_EQ(0u, dc.detector_index());
    EXPECT_FALSE(dc.has_complex_gain());
    EXPECT_DOUBLE_EQ(0.0, dc.timing_skew_s());
}

TEST(KSpaceCalibrationProtoTest, SerializationRoundtrip) {
    KSpaceCalibration original;
    original.set_wavelength_m(905e-9);  // 905nm
    original.set_wavelength_scale(1.002);  // 0.2% scaling correction

    // Add per-detector kx/ky offsets for 8 detectors
    for (int d = 0; d < 8; ++d) {
        original.add_kx_offsets(static_cast<double>(d) * 10.0);  // rad/m
        original.add_ky_offsets(static_cast<double>(d) * -5.0);  // rad/m
    }

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    KSpaceCalibration deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_DOUBLE_EQ(905e-9, deserialized.wavelength_m());
    EXPECT_DOUBLE_EQ(1.002, deserialized.wavelength_scale());
    ASSERT_EQ(8, deserialized.kx_offsets_size());
    ASSERT_EQ(8, deserialized.ky_offsets_size());
    EXPECT_DOUBLE_EQ(30.0, deserialized.kx_offsets(3));
    EXPECT_DOUBLE_EQ(-15.0, deserialized.ky_offsets(3));
}

TEST(KSpaceCalibrationProtoTest, DefaultValues) {
    KSpaceCalibration ksc;
    EXPECT_DOUBLE_EQ(0.0, ksc.wavelength_m());
    EXPECT_EQ(0, ksc.kx_offsets_size());
    EXPECT_EQ(0, ksc.ky_offsets_size());
    EXPECT_DOUBLE_EQ(0.0, ksc.wavelength_scale());
}

TEST(CalibrationArtifactProtoTest, SerializationRoundtrip) {
    CalibrationArtifact original;
    original.set_calibration_id("cal-2024-001");
    original.set_timestamp_s(1702459200.0);

    // Add detector calibrations for 8 detectors
    for (uint32_t d = 0; d < 8; ++d) {
        auto* det_cal = original.add_detector_calibrations();
        det_cal->set_detector_index(d);
        det_cal->mutable_complex_gain()->set_real(1.0 - d * 0.01);
        det_cal->mutable_complex_gain()->set_imag(d * 0.005);
        det_cal->set_timing_skew_s(static_cast<double>(d) * 0.5e-9);
    }

    // Add K-space calibrations for 2 wavelengths
    constexpr double kWavelengths[] = {905e-9, 1550e-9};
    for (uint32_t w = 0; w < 2; ++w) {
        auto* k_cal = original.add_k_space_calibrations();
        k_cal->set_wavelength_m(kWavelengths[w]);
        k_cal->set_wavelength_scale(1.0 + w * 0.001);
        for (int d = 0; d < 8; ++d) {
            k_cal->add_kx_offsets(static_cast<double>(d + w) * 5.0);
            k_cal->add_ky_offsets(static_cast<double>(d - w) * 2.0);
        }
    }

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    CalibrationArtifact deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_EQ("cal-2024-001", deserialized.calibration_id());
    EXPECT_DOUBLE_EQ(1702459200.0, deserialized.timestamp_s());
    ASSERT_EQ(8, deserialized.detector_calibrations_size());
    ASSERT_EQ(2, deserialized.k_space_calibrations_size());

    // Verify detector calibration
    EXPECT_EQ(3u, deserialized.detector_calibrations(3).detector_index());
    EXPECT_DOUBLE_EQ(0.97, deserialized.detector_calibrations(3).complex_gain().real());

    // Verify K-space calibration
    EXPECT_DOUBLE_EQ(1550e-9, deserialized.k_space_calibrations(1).wavelength_m());
    EXPECT_DOUBLE_EQ(1.001, deserialized.k_space_calibrations(1).wavelength_scale());
}

TEST(CalibrationArtifactProtoTest, EmptyCalibration) {
    // Edge case: calibration artifact with no corrections
    CalibrationArtifact empty_cal;
    empty_cal.set_calibration_id("empty");
    empty_cal.set_timestamp_s(0.0);

    std::string serialized;
    ASSERT_TRUE(empty_cal.SerializeToString(&serialized));

    CalibrationArtifact deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    EXPECT_EQ("empty", deserialized.calibration_id());
    EXPECT_EQ(0, deserialized.detector_calibrations_size());
    EXPECT_EQ(0, deserialized.k_space_calibrations_size());
}

TEST(CalibrationArtifactProtoTest, RealisticCalibration) {
    // Simulate a realistic calibration for 8 detectors and 2 wavelengths
    constexpr int kNumDetectors = 8;
    constexpr int kNumWavelengths = 2;
    constexpr double kWavelengths[] = {905e-9, 1550e-9};

    CalibrationArtifact cal;
    cal.set_calibration_id("factory-cal-v1.0");
    cal.set_timestamp_s(1702459200.0);

    // Detector calibrations: slight gain variations and timing skews
    for (uint32_t d = 0; d < kNumDetectors; ++d) {
        auto* det_cal = cal.add_detector_calibrations();
        det_cal->set_detector_index(d);

        // Simulated gain: amplitude varies by +/- 5%, phase by +/- 0.1 rad
        double amplitude = 1.0 + (static_cast<double>(d) - 3.5) * 0.01;
        double phase = (static_cast<double>(d) - 3.5) * 0.02;
        det_cal->mutable_complex_gain()->set_real(amplitude * std::cos(phase));
        det_cal->mutable_complex_gain()->set_imag(amplitude * std::sin(phase));

        // Timing skew: +/- 2 ns relative to center
        det_cal->set_timing_skew_s((static_cast<double>(d) - 3.5) * 0.5e-9);
    }

    // K-space calibrations: per-wavelength corrections
    for (uint32_t w = 0; w < kNumWavelengths; ++w) {
        auto* k_cal = cal.add_k_space_calibrations();
        k_cal->set_wavelength_m(kWavelengths[w]);

        // Wavelength scaling: correct for actual vs nominal wavelength
        k_cal->set_wavelength_scale(1.0 + (w == 0 ? 0.001 : -0.0005));

        // Per-detector K-vector offsets (due to optical misalignment)
        for (int d = 0; d < kNumDetectors; ++d) {
            // Small random-ish offsets based on detector position
            double kx_offset = (d - 3.5) * 2.0;   // rad/m
            double ky_offset = (d % 2) * 1.0;     // rad/m
            k_cal->add_kx_offsets(kx_offset);
            k_cal->add_ky_offsets(ky_offset);
        }
    }

    // Serialize and verify
    std::string serialized;
    ASSERT_TRUE(cal.SerializeToString(&serialized));

    CalibrationArtifact deserialized;
    ASSERT_TRUE(deserialized.ParseFromString(serialized));

    // Verify structure
    EXPECT_EQ("factory-cal-v1.0", deserialized.calibration_id());
    ASSERT_EQ(kNumDetectors, deserialized.detector_calibrations_size());
    ASSERT_EQ(kNumWavelengths, deserialized.k_space_calibrations_size());

    // Verify detector calibration for middle detector
    const auto& det4 = deserialized.detector_calibrations(4);
    EXPECT_EQ(4u, det4.detector_index());
    EXPECT_TRUE(det4.has_complex_gain());
    EXPECT_NEAR(det4.timing_skew_s(), 0.25e-9, 1e-12);

    // Verify K-space calibration structure
    const auto& k_cal_905 = deserialized.k_space_calibrations(0);
    EXPECT_DOUBLE_EQ(905e-9, k_cal_905.wavelength_m());
    EXPECT_DOUBLE_EQ(1.001, k_cal_905.wavelength_scale());
    ASSERT_EQ(kNumDetectors, k_cal_905.kx_offsets_size());
    ASSERT_EQ(kNumDetectors, k_cal_905.ky_offsets_size());
}

}  // namespace aurora::shared
