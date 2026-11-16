#include <gtest/gtest.h>

#include <string>

#include "bandwidth_config.pb.h"
#include "detector_geometry.pb.h"
#include "forward_config.pb.h"
#include "k_space_geometry.pb.h"
#include "k_space_slice.pb.h"
#include "optics_config.pb.h"

namespace aurora::forward {
namespace {

// =============================================================================
// Detector K-Sample Serialization Tests
// =============================================================================

DetectorKSample make_sample(uint32_t detector_index) {
    DetectorKSample sample;
    sample.set_detector_index(detector_index);
    sample.set_wavelength_m(1550e-9);
    sample.set_kx_eff(10000.0 + detector_index * 1000.0);
    sample.set_ky_eff(500.0);
    sample.set_kx_bandwidth(100.0);
    sample.set_ky_bandwidth(50.0);
    return sample;
}

TEST(DetectorKSampleRoundtripTest, DetectorKSample) {
    DetectorKSample original = make_sample(3);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    DetectorKSample restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    EXPECT_EQ(restored.detector_index(), 3u);
    EXPECT_DOUBLE_EQ(restored.wavelength_m(), 1550e-9);
    EXPECT_DOUBLE_EQ(restored.kx_eff(), 13000.0);
    EXPECT_DOUBLE_EQ(restored.ky_eff(), 500.0);
    EXPECT_DOUBLE_EQ(restored.kx_bandwidth(), 100.0);
    EXPECT_DOUBLE_EQ(restored.ky_bandwidth(), 50.0);
}

TEST(DetectorKSampleRoundtripTest, DefaultsAreZero) {
    DetectorKSample original;

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    DetectorKSample restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    EXPECT_EQ(restored.detector_index(), 0u);
    EXPECT_DOUBLE_EQ(restored.kx_bandwidth(), 0.0);
    EXPECT_DOUBLE_EQ(restored.ky_bandwidth(), 0.0);
}

// =============================================================================
// Detector Geometry Serialization Tests
// =============================================================================

TEST(DetectorGeometryRoundtripTest, DetectorArray) {
    DetectorGeometry original;
    auto* array = original.add_arrays();
    array->set_array_id("reference_detector");
    array->set_nominal_pitch_m(0.5e-3);
    array->set_bandwidth_hz(150e6);
    array->set_responsivity_a_per_w(0.9);
    array->set_dark_current_a(1e-9);
    array->set_nep_w_per_sqrt_hz(1e-14);
    array->add_wavelengths_m(905e-9);
    array->add_wavelengths_m(1550e-9);
    array->set_sensitivity_type(DETECTOR_SENSITIVITY_SINC);

    for (uint32_t i = 0; i < 8; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((i - 3.5) * 0.5e-3);
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.4e-3);
        pixel->set_height_m(0.4e-3);
    }

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    DetectorGeometry restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    ASSERT_EQ(restored.arrays_size(), 1);
    const auto& restored_array = restored.arrays(0);
    EXPECT_EQ(restored_array.array_id(), "reference_detector");
    EXPECT_DOUBLE_EQ(restored_array.nominal_pitch_m(), 0.5e-3);
    EXPECT_DOUBLE_EQ(restored_array.bandwidth_hz(), 150e6);
    EXPECT_DOUBLE_EQ(restored_array.responsivity_a_per_w(), 0.9);
    EXPECT_DOUBLE_EQ(restored_array.dark_current_a(), 1e-9);
    EXPECT_DOUBLE_EQ(restored_array.nep_w_per_sqrt_hz(), 1e-14);
    ASSERT_EQ(restored_array.wavelengths_m_size(), 2);
    EXPECT_DOUBLE_EQ(restored_array.wavelengths_m(1), 1550e-9);
    EXPECT_EQ(restored_array.sensitivity_type(), DETECTOR_SENSITIVITY_SINC);

    ASSERT_EQ(restored_array.pixels_size(), 8);
    EXPECT_EQ(restored_array.pixels(7).index(), 7u);
    EXPECT_DOUBLE_EQ(restored_array.pixels(7).x_m(), 3.5 * 0.5e-3);
    EXPECT_DOUBLE_EQ(restored_array.pixels(7).width_m(), 0.4e-3);
    EXPECT_DOUBLE_EQ(restored_array.pixels(7).height_m(), 0.4e-3);
}

// =============================================================================
// Optics Config Serialization Tests
// =============================================================================

TEST(OpticsConfigRoundtripTest, OpticsConfigWithPupilAndLoTilt) {
    OpticsConfig original;
    original.set_fourier_lens_focal_m(0.05);
    original.set_fourier_lens_diameter_m(0.025);
    original.add_wavelengths_m(905e-9);
    original.add_wavelengths_m(1550e-9);
    original.set_system_fov_x_rad(1.047);
    original.set_system_fov_y_rad(0.524);
    original.set_aperture_diameter_m(0.02);
    original.set_k_max_rad_per_m(2.0e5);
    original.set_lo_tilt_x_rad(0.02);
    original.set_lo_tilt_y_rad(-0.01);

    auto* pupil = original.mutable_pupil();
    pupil->set_fov_half_angle_rad(0.524);
    pupil->set_rolloff_width_rad(0.087);
    pupil->set_shape(PupilConfig::PUPIL_RECTANGULAR);
    pupil->set_aspect_ratio(1.5);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    OpticsConfig restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    EXPECT_DOUBLE_EQ(restored.fourier_lens_focal_m(), 0.05);
    EXPECT_DOUBLE_EQ(restored.fourier_lens_diameter_m(), 0.025);
    ASSERT_EQ(restored.wavelengths_m_size(), 2);
    EXPECT_DOUBLE_EQ(restored.wavelengths_m(0), 905e-9);
    EXPECT_DOUBLE_EQ(restored.wavelengths_m(1), 1550e-9);
    EXPECT_DOUBLE_EQ(restored.system_fov_x_rad(), 1.047);
    EXPECT_DOUBLE_EQ(restored.system_fov_y_rad(), 0.524);
    EXPECT_DOUBLE_EQ(restored.aperture_diameter_m(), 0.02);
    EXPECT_DOUBLE_EQ(restored.k_max_rad_per_m(), 2.0e5);
    EXPECT_DOUBLE_EQ(restored.lo_tilt_x_rad(), 0.02);
    EXPECT_DOUBLE_EQ(restored.lo_tilt_y_rad(), -0.01);

    ASSERT_TRUE(restored.has_pupil());
    EXPECT_DOUBLE_EQ(restored.pupil().fov_half_angle_rad(), 0.524);
    EXPECT_DOUBLE_EQ(restored.pupil().rolloff_width_rad(), 0.087);
    EXPECT_EQ(restored.pupil().shape(), PupilConfig::PUPIL_RECTANGULAR);
    ASSERT_TRUE(restored.pupil().has_aspect_ratio());
    EXPECT_DOUBLE_EQ(restored.pupil().aspect_ratio(), 1.5);
}

TEST(OpticsConfigRoundtripTest, CircularPupilLeavesAspectRatioUnset) {
    OpticsConfig original;
    original.set_fourier_lens_focal_m(0.05);
    auto* pupil = original.mutable_pupil();
    pupil->set_fov_half_angle_rad(0.524);
    pupil->set_shape(PupilConfig::PUPIL_CIRCULAR);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    OpticsConfig restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    ASSERT_TRUE(restored.has_pupil());
    EXPECT_EQ(restored.pupil().shape(), PupilConfig::PUPIL_CIRCULAR);
    EXPECT_FALSE(restored.pupil().has_aspect_ratio());
    EXPECT_DOUBLE_EQ(restored.lo_tilt_x_rad(), 0.0);
    EXPECT_DOUBLE_EQ(restored.lo_tilt_y_rad(), 0.0);
}

// =============================================================================
// Forward Config Serialization Tests
// =============================================================================

TEST(ForwardConfigRoundtripTest, FullForwardConfig) {
    ForwardConfig original;

    auto* array = original.mutable_detector()->add_arrays();
    array->set_array_id("fwd-config-detector");
    array->set_sensitivity_type(DETECTOR_SENSITIVITY_POINT);
    auto* pixel = array->add_pixels();
    pixel->set_index(0);
    pixel->set_x_m(0.0);

    auto* optics = original.mutable_optics();
    optics->set_fourier_lens_focal_m(0.05);
    optics->set_k_max_rad_per_m(2.0e5);
    optics->set_lo_tilt_x_rad(0.02);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    ForwardConfig restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    ASSERT_TRUE(restored.has_detector());
    ASSERT_EQ(restored.detector().arrays_size(), 1);
    EXPECT_EQ(restored.detector().arrays(0).array_id(), "fwd-config-detector");
    EXPECT_EQ(restored.detector().arrays(0).sensitivity_type(), DETECTOR_SENSITIVITY_POINT);
    EXPECT_EQ(restored.detector().arrays(0).pixels_size(), 1);

    ASSERT_TRUE(restored.has_optics());
    EXPECT_DOUBLE_EQ(restored.optics().fourier_lens_focal_m(), 0.05);
    EXPECT_DOUBLE_EQ(restored.optics().k_max_rad_per_m(), 2.0e5);
    EXPECT_DOUBLE_EQ(restored.optics().lo_tilt_x_rad(), 0.02);
}

// =============================================================================
// K-Space Slice Serialization Tests
// =============================================================================

TEST(KSpaceSliceRoundtripTest, KSpaceSliceWithGridHints) {
    aurora::shared::KSpaceSlice original;
    original.set_range_bin(10);
    original.set_wavelength_m(1550e-9);
    original.set_burst_id(12345);
    original.set_timestamp_s(1.5);

    auto* with_bandwidth = original.add_samples();
    with_bandwidth->mutable_value()->set_real(1.0);
    with_bandwidth->mutable_value()->set_imag(-0.5);
    with_bandwidth->set_kx(-100.0);
    with_bandwidth->set_ky(200.0);
    with_bandwidth->set_weight(0.8);
    with_bandwidth->set_detector_index(3);
    with_bandwidth->set_kx_bandwidth(10.0);
    with_bandwidth->set_ky_bandwidth(15.0);

    auto* without_bandwidth = original.add_samples();
    without_bandwidth->mutable_value()->set_real(0.25);
    without_bandwidth->set_kx(100.0);
    without_bandwidth->set_ky(-200.0);
    without_bandwidth->set_weight(0.7);
    without_bandwidth->set_detector_index(4);

    auto* hints = original.mutable_grid_hints();
    hints->set_kx_min(-100.0);
    hints->set_kx_max(100.0);
    hints->set_ky_min(-200.0);
    hints->set_ky_max(200.0);
    hints->set_suggested_nx(64);
    hints->set_suggested_ny(128);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    aurora::shared::KSpaceSlice restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    EXPECT_EQ(restored.range_bin(), 10u);
    EXPECT_DOUBLE_EQ(restored.wavelength_m(), 1550e-9);
    ASSERT_TRUE(restored.has_burst_id());
    EXPECT_EQ(restored.burst_id(), 12345u);
    ASSERT_TRUE(restored.has_timestamp_s());
    EXPECT_DOUBLE_EQ(restored.timestamp_s(), 1.5);

    ASSERT_EQ(restored.samples_size(), 2);
    const auto& s0 = restored.samples(0);
    EXPECT_DOUBLE_EQ(s0.value().real(), 1.0);
    EXPECT_DOUBLE_EQ(s0.value().imag(), -0.5);
    EXPECT_DOUBLE_EQ(s0.kx(), -100.0);
    EXPECT_DOUBLE_EQ(s0.ky(), 200.0);
    EXPECT_DOUBLE_EQ(s0.weight(), 0.8);
    EXPECT_EQ(s0.detector_index(), 3u);
    ASSERT_TRUE(s0.has_kx_bandwidth());
    ASSERT_TRUE(s0.has_ky_bandwidth());
    EXPECT_DOUBLE_EQ(s0.kx_bandwidth(), 10.0);
    EXPECT_DOUBLE_EQ(s0.ky_bandwidth(), 15.0);

    const auto& s1 = restored.samples(1);
    EXPECT_EQ(s1.detector_index(), 4u);
    EXPECT_FALSE(s1.has_kx_bandwidth());
    EXPECT_FALSE(s1.has_ky_bandwidth());

    ASSERT_TRUE(restored.has_grid_hints());
    EXPECT_DOUBLE_EQ(restored.grid_hints().kx_min(), -100.0);
    EXPECT_DOUBLE_EQ(restored.grid_hints().kx_max(), 100.0);
    EXPECT_DOUBLE_EQ(restored.grid_hints().ky_min(), -200.0);
    EXPECT_DOUBLE_EQ(restored.grid_hints().ky_max(), 200.0);
    EXPECT_EQ(restored.grid_hints().suggested_nx(), 64u);
    EXPECT_EQ(restored.grid_hints().suggested_ny(), 128u);
}

TEST(KSpaceSliceRoundtripTest, OptionalFieldsUnsetByDefault) {
    aurora::shared::KSpaceSlice original;
    original.set_range_bin(3);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    aurora::shared::KSpaceSlice restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    EXPECT_EQ(restored.range_bin(), 3u);
    EXPECT_FALSE(restored.has_burst_id());
    EXPECT_FALSE(restored.has_timestamp_s());
    EXPECT_FALSE(restored.has_grid_hints());
    EXPECT_EQ(restored.samples_size(), 0);
}

TEST(KSpaceSliceRoundtripTest, KSpaceSliceCollection) {
    aurora::shared::KSpaceSliceCollection original;
    original.set_wavelength_m(905e-9);

    for (uint32_t bin = 0; bin < 3; ++bin) {
        auto* slice = original.add_slices();
        slice->set_range_bin(bin);
        slice->set_wavelength_m(905e-9);
        auto* sample = slice->add_samples();
        sample->mutable_value()->set_real(static_cast<double>(bin));
        sample->set_detector_index(bin + 1);
    }

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    aurora::shared::KSpaceSliceCollection restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    EXPECT_DOUBLE_EQ(restored.wavelength_m(), 905e-9);
    ASSERT_EQ(restored.slices_size(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(restored.slices(i).range_bin(), static_cast<uint32_t>(i));
        ASSERT_EQ(restored.slices(i).samples_size(), 1);
        EXPECT_DOUBLE_EQ(restored.slices(i).samples(0).value().real(), static_cast<double>(i));
        EXPECT_EQ(restored.slices(i).samples(0).detector_index(), static_cast<uint32_t>(i + 1));
    }
}

// =============================================================================
// Bandwidth Config Serialization Tests
// =============================================================================

TEST(BandwidthConfigRoundtripTest, BandwidthConfig) {
    BandwidthConfig original;
    original.set_window_type(BANDWIDTH_WINDOW_GAUSSIAN);
    original.set_truncation_lobes(3);
    original.set_weight_threshold(1e-6);
    original.set_quadrature_order(11);

    std::string serialized;
    ASSERT_TRUE(original.SerializeToString(&serialized));

    BandwidthConfig restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    EXPECT_EQ(restored.window_type(), BANDWIDTH_WINDOW_GAUSSIAN);
    EXPECT_EQ(restored.truncation_lobes(), 3u);
    EXPECT_DOUBLE_EQ(restored.weight_threshold(), 1e-6);
    EXPECT_EQ(restored.quadrature_order(), 11u);
}

}  // namespace
}  // namespace aurora::forward
