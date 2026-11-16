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

#include "dsp/measurement_builder.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <optional>

#include "forward/forward_model.h"
#include "test_forward_model_builder.h"

#include "detector_geometry.pb.h"
#include "optics_config.pb.h"

namespace aurora::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

/// Create test detector geometry with 8 pixels.
aurora::forward::DetectorGeometry create_test_detector_geometry() {
    aurora::forward::DetectorGeometry detector;
    auto* array = detector.add_arrays();
    array->set_array_id("test_array");
    array->set_nominal_pitch_m(0.5e-3);
    array->set_bandwidth_hz(150e6);
    array->add_wavelengths_m(1550e-9);
    array->add_wavelengths_m(905e-9);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_SINC);

    for (uint32_t i = 0; i < 8; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((static_cast<double>(i) - 3.5) * 0.5e-3);
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.4e-3);
        pixel->set_height_m(0.4e-3);
    }
    return detector;
}

/// Create test optics configuration.
aurora::forward::OpticsConfig create_test_optics_config() {
    aurora::forward::OpticsConfig optics;
    optics.set_fourier_lens_focal_m(0.05);
    optics.set_fourier_lens_diameter_m(0.025);
    optics.add_wavelengths_m(1550e-9);
    optics.add_wavelengths_m(905e-9);
    return optics;
}

/// Fill one wavelength's range data for all detectors with a peak at bin 100,
/// scaled by `scale` so up/down/averaged profiles can be told apart.
void fill_wavelength_range_data(WavelengthRangeData* wl_data,
                                double wavelength_m,
                                int num_range_bins,
                                int num_detectors,
                                double scale = 1.0) {
    wl_data->set_wavelength_m(wavelength_m);

    for (int det_idx = 0; det_idx < num_detectors; ++det_idx) {
        auto* det_data = wl_data->add_detectors();
        det_data->set_detector_index(det_idx);

        for (int bin = 0; bin < num_range_bins; ++bin) {
            auto* out_bin = det_data->add_bins();
            if (bin == 100) {
                out_bin->set_real(1.0 * scale);
                out_bin->set_imag(0.5 * scale);
            } else {
                out_bin->set_real(0.01 * std::cos(0.1 * bin) * scale);
                out_bin->set_imag(0.01 * std::sin(0.1 * bin) * scale);
            }
        }
    }
}

/// Create a test RangeCube with specified parameters (chirp-averaged data only).
RangeCube create_test_range_cube(
    uint64_t burst_id = 1,
    double timestamp_s = 0.0,
    double wavelength_m = 1550e-9,
    int num_range_bins = 1024,
    int num_detectors = 8) {

    RangeCube cube;
    cube.set_burst_id(burst_id);
    cube.set_timestamp_s(timestamp_s);

    // Set up range axis
    auto* range_axis = cube.mutable_range_axis();
    range_axis->set_unit("m");
    range_axis->set_start(0.0);
    range_axis->set_step(0.75);  // ~200 MHz bandwidth
    range_axis->set_count(num_range_bins);

    fill_wavelength_range_data(cube.add_wavelengths(), wavelength_m,
                               num_range_bins, num_detectors);

    return cube;
}

/// Up/down profile scales used by the Doppler cube helper.
constexpr double kUpScale = 2.0;
constexpr double kDownScale = 4.0;

/// Create a test RangeCube that also carries up/down chirp profiles, as
/// process_doppler() does. The averaged data has unit scale, the up-chirp
/// profile is scaled by kUpScale and the down-chirp profile by kDownScale.
RangeCube create_test_doppler_range_cube(
    uint64_t burst_id = 1,
    double wavelength_m = 1550e-9,
    int num_range_bins = 256,
    int num_detectors = 8) {

    RangeCube cube = create_test_range_cube(burst_id, 0.0, wavelength_m,
                                            num_range_bins, num_detectors);

    fill_wavelength_range_data(cube.add_up_chirps(), wavelength_m,
                               num_range_bins, num_detectors, kUpScale);
    fill_wavelength_range_data(cube.add_down_chirps(), wavelength_m,
                               num_range_bins, num_detectors, kDownScale);
    return cube;
}

/// Test fixture that sets up ForwardModel once per test suite.
class MeasurementBuilderTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        detector_ = create_test_detector_geometry();
        optics_ = create_test_optics_config();

        // Create ForwardModel from geometry
        forward_model_.emplace(detector_, optics_);
    }

    static void TearDownTestSuite() {
        forward_model_.reset();
    }

    const aurora::forward::ForwardModel& forward_model() const { return *forward_model_; }

    static aurora::forward::DetectorGeometry detector_;
    static aurora::forward::OpticsConfig optics_;
    static std::optional<aurora::forward::ForwardModel> forward_model_;
};

// Static member definitions
aurora::forward::DetectorGeometry MeasurementBuilderTest::detector_;
aurora::forward::OpticsConfig MeasurementBuilderTest::optics_;
std::optional<aurora::forward::ForwardModel> MeasurementBuilderTest::forward_model_;


// Test: K-vectors are annotated from ForwardModel
TEST_F(MeasurementBuilderTest, KVectorsAnnotated) {
    MeasurementBuilder builder(forward_model());
    auto cube = create_test_range_cube();
    auto tensor = builder.build(cube);

    ASSERT_EQ(tensor.data_size(), 1);
    ASSERT_GT(tensor.data(0).detectors_size(), 0);

    // Each detector should have a K-vector
    for (const auto& detector : tensor.data(0).detectors()) {
        EXPECT_TRUE(detector.has_k_vector());
        // K-vectors should be finite (not NaN or Inf)
        EXPECT_TRUE(std::isfinite(detector.k_vector().kx()));
        EXPECT_TRUE(std::isfinite(detector.k_vector().ky()));
    }
}

// Test: K-vectors equal the forward model's effective K per detector
TEST_F(MeasurementBuilderTest, KVectorsMatchGeometry) {
    MeasurementBuilder builder(forward_model());
    const double wavelength_m = 1550e-9;
    auto cube = create_test_range_cube(1, 0.0, wavelength_m);
    auto tensor = builder.build(cube);

    ASSERT_EQ(tensor.data_size(), 1);
    const double f = optics_.fourier_lens_focal_m();

    for (const auto& detector : tensor.data(0).detectors()) {
        const auto& pixel = detector_.arrays(0).pixels(detector.detector_index());
        const double expected_kx = kTwoPi * pixel.x_m() / (f * wavelength_m);

        auto k_scene = forward_model().get_effective_k(detector.detector_index(), wavelength_m);
        EXPECT_DOUBLE_EQ(detector.k_vector().kx(), k_scene.x());
        EXPECT_DOUBLE_EQ(detector.k_vector().ky(), k_scene.y());
        EXPECT_NEAR(detector.k_vector().kx(), expected_kx, 1e-6);
    }
}

// Test: Tensor structure is correct
TEST_F(MeasurementBuilderTest, TensorStructureCorrect) {
    MeasurementBuilder builder(forward_model());
    auto cube = create_test_range_cube(12345, 42.5, 1550e-9, 1024, 8);
    auto tensor = builder.build(cube);

    // Check metadata
    EXPECT_EQ(tensor.burst_id(), 12345u);
    EXPECT_DOUBLE_EQ(tensor.timestamp_s(), 42.5);

    // Check structure: one wavelength channel with one channel per detector
    ASSERT_EQ(tensor.data_size(), 1);
    EXPECT_DOUBLE_EQ(tensor.data(0).wavelength_m(), 1550e-9);
    EXPECT_EQ(tensor.data(0).detectors_size(), 8);

    // No up/down chirp data when the cube carries none
    EXPECT_EQ(tensor.up_chirp_data_size(), 0);
    EXPECT_EQ(tensor.down_chirp_data_size(), 0);
}

// Test: Range axis matches DSP output
TEST_F(MeasurementBuilderTest, RangeAxisMatches) {
    MeasurementBuilder builder(forward_model());
    auto cube = create_test_range_cube();
    auto tensor = builder.build(cube);

    EXPECT_EQ(tensor.range_axis().unit(), cube.range_axis().unit());
    EXPECT_DOUBLE_EQ(tensor.range_axis().start(), cube.range_axis().start());
    EXPECT_DOUBLE_EQ(tensor.range_axis().step(), cube.range_axis().step());
    EXPECT_EQ(tensor.range_axis().count(), cube.range_axis().count());
}

// Test: Range data is preserved
TEST_F(MeasurementBuilderTest, RangeDataPreserved) {
    MeasurementBuilder builder(forward_model());
    auto cube = create_test_range_cube();
    auto tensor = builder.build(cube);

    const auto& cube_det = cube.wavelengths(0).detectors(0);
    const auto& tensor_det = tensor.data(0).detectors(0);

    ASSERT_EQ(cube_det.bins_size(), tensor_det.range_data_size());

    // Check all range bins match
    for (int i = 0; i < cube_det.bins_size(); ++i) {
        EXPECT_DOUBLE_EQ(cube_det.bins(i).real(), tensor_det.range_data(i).real())
            << "Real mismatch at bin " << i;
        EXPECT_DOUBLE_EQ(cube_det.bins(i).imag(), tensor_det.range_data(i).imag())
            << "Imag mismatch at bin " << i;
    }
}

// Test: Up/down chirp profiles are carried into up_chirp_data/down_chirp_data
TEST_F(MeasurementBuilderTest, UpDownChirpDataCarriedThrough) {
    MeasurementBuilder builder(forward_model());
    auto cube = create_test_doppler_range_cube();
    auto tensor = builder.build(cube);

    ASSERT_EQ(tensor.data_size(), 1);
    ASSERT_EQ(tensor.up_chirp_data_size(), 1);
    ASSERT_EQ(tensor.down_chirp_data_size(), 1);

    EXPECT_DOUBLE_EQ(tensor.up_chirp_data(0).wavelength_m(), 1550e-9);
    EXPECT_DOUBLE_EQ(tensor.down_chirp_data(0).wavelength_m(), 1550e-9);
    EXPECT_EQ(tensor.up_chirp_data(0).detectors_size(), 8);
    EXPECT_EQ(tensor.down_chirp_data(0).detectors_size(), 8);

    const auto& avg_det = tensor.data(0).detectors(0);
    const auto& up_det = tensor.up_chirp_data(0).detectors(0);
    const auto& down_det = tensor.down_chirp_data(0).detectors(0);

    ASSERT_EQ(up_det.range_data_size(), avg_det.range_data_size());
    ASSERT_EQ(down_det.range_data_size(), avg_det.range_data_size());

    // Up/down channels carry their own data (not a copy of the average)
    // and the same K-vector annotation as the averaged channel.
    EXPECT_TRUE(up_det.has_k_vector());
    EXPECT_TRUE(down_det.has_k_vector());
    EXPECT_DOUBLE_EQ(up_det.k_vector().kx(), avg_det.k_vector().kx());
    EXPECT_DOUBLE_EQ(down_det.k_vector().kx(), avg_det.k_vector().kx());

    for (int b = 0; b < avg_det.range_data_size(); ++b) {
        EXPECT_DOUBLE_EQ(up_det.range_data(b).real(), avg_det.range_data(b).real() * kUpScale)
            << "Up-chirp real mismatch at bin " << b;
        EXPECT_DOUBLE_EQ(up_det.range_data(b).imag(), avg_det.range_data(b).imag() * kUpScale)
            << "Up-chirp imag mismatch at bin " << b;
        EXPECT_DOUBLE_EQ(down_det.range_data(b).real(), avg_det.range_data(b).real() * kDownScale)
            << "Down-chirp real mismatch at bin " << b;
        EXPECT_DOUBLE_EQ(down_det.range_data(b).imag(), avg_det.range_data(b).imag() * kDownScale)
            << "Down-chirp imag mismatch at bin " << b;
    }
}

// Test: Empty RangeCube produces empty tensor
TEST_F(MeasurementBuilderTest, EmptyRangeCubeProducesEmptyTensor) {
    MeasurementBuilder builder(forward_model());
    RangeCube empty_cube;
    empty_cube.set_burst_id(42);
    empty_cube.set_timestamp_s(1.5);

    auto tensor = builder.build(empty_cube);

    EXPECT_EQ(tensor.burst_id(), 42u);
    EXPECT_DOUBLE_EQ(tensor.timestamp_s(), 1.5);
    EXPECT_EQ(tensor.data_size(), 0);
    EXPECT_EQ(tensor.up_chirp_data_size(), 0);
    EXPECT_EQ(tensor.down_chirp_data_size(), 0);
}

// Test: Empty vector of RangeCubes returns empty tensor
TEST_F(MeasurementBuilderTest, EmptyVectorReturnsEmptyTensor) {
    MeasurementBuilder builder(forward_model());
    std::vector<RangeCube> empty_cubes;
    auto tensor = builder.build(empty_cubes);

    EXPECT_EQ(tensor.burst_id(), 0u);
    EXPECT_EQ(tensor.data_size(), 0);
}

// Test: Single-element vector behaves like single cube
TEST_F(MeasurementBuilderTest, SingleElementVectorBehavesLikeSingleCube) {
    MeasurementBuilder builder(forward_model());
    auto cube = create_test_range_cube(999, 123.456);
    std::vector<RangeCube> cubes = {cube};

    auto tensor_single = builder.build(cube);
    auto tensor_vector = builder.build(cubes);

    // Metadata should match
    EXPECT_EQ(tensor_single.burst_id(), tensor_vector.burst_id());
    EXPECT_DOUBLE_EQ(tensor_single.timestamp_s(), tensor_vector.timestamp_s());

    // Data should match
    ASSERT_EQ(tensor_single.data_size(), tensor_vector.data_size());
    for (int w = 0; w < tensor_single.data_size(); ++w) {
        auto& single_det = tensor_single.data(w).detectors(0);
        auto& vector_det = tensor_vector.data(w).detectors(0);
        ASSERT_EQ(single_det.range_data_size(), vector_det.range_data_size());

        for (int b = 0; b < single_det.range_data_size(); ++b) {
            EXPECT_DOUBLE_EQ(single_det.range_data(b).real(),
                            vector_det.range_data(b).real());
            EXPECT_DOUBLE_EQ(single_det.range_data(b).imag(),
                            vector_det.range_data(b).imag());
        }
    }
}

// Test: Multiple RangeCubes are coherently averaged
TEST_F(MeasurementBuilderTest, MultipleCubesAveraged) {
    MeasurementBuilder builder(forward_model());

    // Create two identical cubes
    auto cube1 = create_test_range_cube(1, 0.0);
    auto cube2 = create_test_range_cube(2, 1.0);

    // Triple the values in cube2
    for (auto& wavelength : *cube2.mutable_wavelengths()) {
        for (auto& detector : *wavelength.mutable_detectors()) {
            for (auto& bin : *detector.mutable_bins()) {
                bin.set_real(bin.real() * 3.0);  // Original + 3x -> average 2x
                bin.set_imag(bin.imag() * 3.0);
            }
        }
    }

    std::vector<RangeCube> cubes = {cube1, cube2};
    auto tensor = builder.build(cubes);

    // Result should be average: (1 + 3) / 2 = 2x original
    auto cube_original = create_test_range_cube();
    auto& orig_det = cube_original.wavelengths(0).detectors(0);
    auto& result_det = tensor.data(0).detectors(0);

    for (int b = 0; b < orig_det.bins_size(); ++b) {
        double expected_real = orig_det.bins(b).real() * 2.0;
        double expected_imag = orig_det.bins(b).imag() * 2.0;
        EXPECT_NEAR(result_det.range_data(b).real(), expected_real, 1e-9)
            << "Real mismatch at bin " << b;
        EXPECT_NEAR(result_det.range_data(b).imag(), expected_imag, 1e-9)
            << "Imag mismatch at bin " << b;
    }
}

// Test: Up/down chirp data is averaged across cubes like the main data
TEST_F(MeasurementBuilderTest, MultipleCubesAverageUpDownChirpData) {
    MeasurementBuilder builder(forward_model());

    auto cube1 = create_test_doppler_range_cube(1);
    auto cube2 = create_test_doppler_range_cube(2);

    std::vector<RangeCube> cubes = {cube1, cube2};
    auto tensor = builder.build(cubes);

    ASSERT_EQ(tensor.up_chirp_data_size(), 1);
    ASSERT_EQ(tensor.down_chirp_data_size(), 1);

    // Two identical cubes average to the original values
    const auto& up_orig = cube1.up_chirps(0).detectors(0);
    const auto& down_orig = cube1.down_chirps(0).detectors(0);
    const auto& up_det = tensor.up_chirp_data(0).detectors(0);
    const auto& down_det = tensor.down_chirp_data(0).detectors(0);

    ASSERT_EQ(up_det.range_data_size(), up_orig.bins_size());
    ASSERT_EQ(down_det.range_data_size(), down_orig.bins_size());

    for (int b = 0; b < up_orig.bins_size(); ++b) {
        EXPECT_NEAR(up_det.range_data(b).real(), up_orig.bins(b).real(), 1e-9);
        EXPECT_NEAR(up_det.range_data(b).imag(), up_orig.bins(b).imag(), 1e-9);
        EXPECT_NEAR(down_det.range_data(b).real(), down_orig.bins(b).real(), 1e-9);
        EXPECT_NEAR(down_det.range_data(b).imag(), down_orig.bins(b).imag(), 1e-9);
    }
}

// Test: Incompatible cubes throw exception
TEST_F(MeasurementBuilderTest, IncompatibleCubesThrow) {
    MeasurementBuilder builder(forward_model());
    auto cube1 = create_test_range_cube(1, 0.0, 1550e-9, 1024, 8);
    auto cube2 = create_test_range_cube(2, 0.0, 1550e-9, 512, 8);  // Different bin count

    std::vector<RangeCube> cubes = {cube1, cube2};
    EXPECT_THROW(builder.build(cubes), std::invalid_argument);
}

// Test: Cubes with and without up/down chirp data cannot be combined
TEST_F(MeasurementBuilderTest, MixedUpDownPresenceThrows) {
    MeasurementBuilder builder(forward_model());
    auto with_updown = create_test_doppler_range_cube(1, 1550e-9, 256, 8);
    auto without_updown = create_test_range_cube(2, 0.0, 1550e-9, 256, 8);

    std::vector<RangeCube> cubes = {with_updown, without_updown};
    EXPECT_THROW(builder.build(cubes), std::invalid_argument);
}

// Test: All fields are properly populated
TEST_F(MeasurementBuilderTest, AllFieldsPopulated) {
    MeasurementBuilder builder(forward_model());
    auto cube = create_test_range_cube();
    auto tensor = builder.build(cube);

    // Check that all required fields are set
    EXPECT_TRUE(tensor.has_range_axis());
    EXPECT_GT(tensor.range_axis().count(), 0u);

    ASSERT_GT(tensor.data_size(), 0);
    auto& wl_channel = tensor.data(0);

    EXPECT_GT(wl_channel.wavelength_m(), 0.0);
    ASSERT_GT(wl_channel.detectors_size(), 0);

    auto& det_channel = wl_channel.detectors(0);
    EXPECT_TRUE(det_channel.has_k_vector());
    EXPECT_GT(det_channel.range_data_size(), 0);
}

}  // namespace
}  // namespace aurora::dsp
