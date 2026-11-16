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

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "forward/forward_model.h"
#include "test_forward_model_builder.h"

namespace aurora::forward {

namespace {

constexpr double kTwoPi = 2.0 * M_PI;

// ============================================================================
// ForwardModel Tests
// ============================================================================
//
// Each detector pixel j samples one K-space position per wavelength:
//   k_j(lambda) = 2*pi * x_j / (f * lambda)
// with a K-space acceptance bandwidth set by the pixel size:
//   dk_j(lambda) = 2*pi * w_j / (f * lambda)
// The LO tilt adds a constant scene-space offset (see get_effective_k).

class ForwardModelTest : public ::testing::Test {
  protected:
    static constexpr uint32_t kNumDetectors = 8;
    static constexpr double kPitch = 0.5e-3;        // m
    static constexpr double kPixelWidth = 0.4e-3;   // m
    static constexpr double kPixelHeight = 0.3e-3;  // m (distinct from width)
    static constexpr double kFocalLength = 0.05;    // m
    static constexpr double kWavelength = 905e-9;   // m
    static constexpr double kOtherWavelength = 1550e-9;  // m

    void SetUp() override {
        // Set up detector geometry (8 pixels, 0.5mm pitch)
        auto* array = detector_.add_arrays();
        array->set_array_id("test_detector");
        array->set_nominal_pitch_m(kPitch);
        array->add_wavelengths_m(kWavelength);
        array->add_wavelengths_m(kOtherWavelength);
        array->set_sensitivity_type(DETECTOR_SENSITIVITY_SINC);

        for (uint32_t i = 0; i < kNumDetectors; ++i) {
            auto* pixel = array->add_pixels();
            pixel->set_index(i);
            pixel->set_x_m((i - (kNumDetectors - 1) / 2.0) * kPitch);
            pixel->set_y_m(0.0);
            pixel->set_width_m(kPixelWidth);
            pixel->set_height_m(kPixelHeight);
        }

        // Set up optics configuration
        optics_.set_fourier_lens_focal_m(kFocalLength);
        optics_.add_wavelengths_m(kWavelength);
        optics_.add_wavelengths_m(kOtherWavelength);
    }

    /// Expected geometric K-position of pixel `d`: k = 2*pi * x_d / (f * lambda)
    double expected_kx(uint32_t d, double wavelength) const {
        const auto& pixel = detector_.arrays(0).pixels(static_cast<int>(d));
        return kTwoPi * pixel.x_m() / (kFocalLength * wavelength);
    }

    /// Build a MeasurementTensor with one WavelengthChannel record, all detectors,
    /// and `num_bins` range bins of unit-real data.
    static aurora::shared::MeasurementTensor make_tensor(double wavelength,
                                                         int num_bins = 1) {
        aurora::shared::MeasurementTensor tensor;
        auto* wavelength_channel = tensor.add_data();
        wavelength_channel->set_wavelength_m(wavelength);

        for (uint32_t d = 0; d < kNumDetectors; ++d) {
            auto* det_channel = wavelength_channel->add_detectors();
            det_channel->set_detector_index(d);
            for (int bin = 0; bin < num_bins; ++bin) {
                auto* range_data = det_channel->add_range_data();
                range_data->set_real(1.0);
                range_data->set_imag(0.0);
            }
        }
        return tensor;
    }

    DetectorGeometry detector_;
    OpticsConfig optics_;
};

// ----------------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------------

// Test that ForwardModel constructs successfully
TEST_F(ForwardModelTest, ConstructsSuccessfully) {
    EXPECT_NO_THROW({
        ForwardModel model(detector_, optics_);
    });
}

// Test that ForwardModel constructs from a ForwardConfig
TEST_F(ForwardModelTest, ConstructsFromForwardConfig) {
    ForwardConfig config;
    *config.mutable_detector() = detector_;
    *config.mutable_optics() = optics_;

    ForwardModel model(config);

    EXPECT_EQ(model.num_detectors(), kNumDetectors);
    EXPECT_NEAR(model.optics().fourier_lens_focal_m(), kFocalLength, 1e-10);
    EXPECT_EQ(model.detector().arrays(0).array_id(), "test_detector");
}

// Test that a geometry without any detector array is rejected
TEST_F(ForwardModelTest, ThrowsWithoutDetectorArray) {
    DetectorGeometry empty;
    EXPECT_THROW(ForwardModel(empty, optics_), std::invalid_argument);

    ForwardConfig config;
    *config.mutable_optics() = optics_;
    EXPECT_THROW(ForwardModel{config}, std::invalid_argument);
}

// Test that an array without pixels is rejected
TEST_F(ForwardModelTest, ThrowsWithoutPixels) {
    DetectorGeometry no_pixels;
    auto* array = no_pixels.add_arrays();
    array->set_array_id("empty");
    array->set_sensitivity_type(DETECTOR_SENSITIVITY_SINC);

    EXPECT_THROW(ForwardModel(no_pixels, optics_), std::invalid_argument);
}

// Test that a non-positive focal length is rejected
TEST_F(ForwardModelTest, ThrowsWithoutPositiveFocalLength) {
    OpticsConfig zero_focal = optics_;
    zero_focal.set_fourier_lens_focal_m(0.0);
    EXPECT_THROW(ForwardModel(detector_, zero_focal), std::invalid_argument);

    OpticsConfig negative_focal = optics_;
    negative_focal.set_fourier_lens_focal_m(-kFocalLength);
    EXPECT_THROW(ForwardModel(detector_, negative_focal), std::invalid_argument);
}

// Test that the model is movable (needed for std::optional / emplace)
TEST_F(ForwardModelTest, IsMovable) {
    ForwardModel model(detector_, optics_);
    ForwardModel moved(std::move(model));

    EXPECT_EQ(moved.num_detectors(), kNumDetectors);
}

// ----------------------------------------------------------------------------
// K-space mapping
// ----------------------------------------------------------------------------

// Test number of detectors accessor
TEST_F(ForwardModelTest, NumDetectors) {
    ForwardModel model(detector_, optics_);
    EXPECT_EQ(model.num_detectors(), kNumDetectors);
}

// Test get_k_sample returns k = 2*pi * x_j / (f * lambda) and the pixel's
// K-space acceptance bandwidth dk = 2*pi * w / (f * lambda)
TEST_F(ForwardModelTest, GetKSampleReturnsCorrectValues) {
    ForwardModel model(detector_, optics_);

    const uint32_t det_idx = 3;
    auto sample = model.get_k_sample(det_idx, kWavelength);

    EXPECT_EQ(sample.detector_index(), det_idx);
    EXPECT_NEAR(sample.wavelength_m(), kWavelength, 1e-15);
    EXPECT_NEAR(sample.kx_eff(), expected_kx(det_idx, kWavelength), 1e-6);
    EXPECT_NEAR(sample.ky_eff(), 0.0, 1e-6);

    const double expected_kx_bw = kTwoPi * kPixelWidth / (kFocalLength * kWavelength);
    const double expected_ky_bw = kTwoPi * kPixelHeight / (kFocalLength * kWavelength);
    EXPECT_NEAR(sample.kx_bandwidth(), expected_kx_bw, 1e-6);
    EXPECT_NEAR(sample.ky_bandwidth(), expected_ky_bw, 1e-6);
}

// Test K-samples scale as 1/lambda
TEST_F(ForwardModelTest, GetKSampleScalesInverselyWithWavelength) {
    ForwardModel model(detector_, optics_);

    const uint32_t det_idx = 7;
    auto short_wl = model.get_k_sample(det_idx, kWavelength);
    auto long_wl = model.get_k_sample(det_idx, kOtherWavelength);

    const double ratio = kOtherWavelength / kWavelength;
    EXPECT_NEAR(short_wl.kx_eff(), long_wl.kx_eff() * ratio, 1e-6);
    EXPECT_NEAR(short_wl.kx_bandwidth(), long_wl.kx_bandwidth() * ratio, 1e-6);
}

// Test get_effective_k equals the geometric K when no LO tilt is configured
TEST_F(ForwardModelTest, GetEffectiveK) {
    ForwardModel model(detector_, optics_);

    const uint32_t det_idx = 5;
    auto k = model.get_effective_k(det_idx, kOtherWavelength);

    EXPECT_NEAR(k.x(), expected_kx(det_idx, kOtherWavelength), 1e-6);
    EXPECT_NEAR(k.y(), 0.0, 1e-6);
}

// Test get_effective_k adds the LO tilt K-offset: k_scene = k_det + k_LO
TEST_F(ForwardModelTest, GetEffectiveKIncludesLoTilt) {
    const double lo_tilt_x = 0.02;  // rad
    const double lo_tilt_y = 0.01;  // rad
    const uint32_t det_idx = 5;

    OpticsConfig tilted_optics = optics_;
    tilted_optics.set_lo_tilt_x_rad(lo_tilt_x);
    tilted_optics.set_lo_tilt_y_rad(lo_tilt_y);

    ForwardModel model(detector_, tilted_optics);

    auto sample = model.get_k_sample(det_idx, kOtherWavelength);
    auto k = model.get_effective_k(det_idx, kOtherWavelength);

    // k_LO = 2π sin(θ) / λ
    const double expected_x =
        sample.kx_eff() + kTwoPi * std::sin(lo_tilt_x) / kOtherWavelength;
    const double expected_y =
        sample.ky_eff() + kTwoPi * std::sin(lo_tilt_y) / kOtherWavelength;

    EXPECT_NEAR(k.x(), expected_x, 1e-6);
    EXPECT_NEAR(k.y(), expected_y, 1e-6);

    // The K-sample itself is the geometric detector K, unaffected by LO tilt
    EXPECT_NEAR(sample.kx_eff(), expected_kx(det_idx, kOtherWavelength), 1e-6);
    EXPECT_NEAR(sample.ky_eff(), 0.0, 1e-6);
}

// Test get_all_k_samples returns one sample per detector, in order
TEST_F(ForwardModelTest, GetAllKSamples) {
    ForwardModel model(detector_, optics_);

    auto samples = model.get_all_k_samples(kWavelength);

    ASSERT_EQ(samples.size(), static_cast<size_t>(kNumDetectors));
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto d = static_cast<uint32_t>(i);
        EXPECT_EQ(samples[i].detector_index(), d);
        EXPECT_NEAR(samples[i].wavelength_m(), kWavelength, 1e-15);
        EXPECT_NEAR(samples[i].kx_eff(), expected_kx(d, kWavelength), 1e-6);
    }
}

// Test get_k_sample / get_effective_k throw for a detector index past the array
TEST_F(ForwardModelTest, GetKSampleThrowsForInvalidDetector) {
    ForwardModel model(detector_, optics_);

    EXPECT_THROW(model.get_k_sample(kNumDetectors, kWavelength), std::out_of_range);
    EXPECT_THROW(model.get_effective_k(kNumDetectors, kWavelength), std::out_of_range);
}

// Test get_k_sample throws for a non-positive wavelength
TEST_F(ForwardModelTest, GetKSampleThrowsForInvalidWavelength) {
    ForwardModel model(detector_, optics_);

    EXPECT_THROW(model.get_k_sample(0, 0.0), std::invalid_argument);
    EXPECT_THROW(model.get_k_sample(0, -kWavelength), std::invalid_argument);
    EXPECT_THROW(model.get_all_k_samples(0.0), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// Physics utilities
// ----------------------------------------------------------------------------

// Test compute_phase computes -k . r correctly
TEST_F(ForwardModelTest, ComputePhase) {
    Eigen::Vector2d k(1000.0, 2000.0);
    Eigen::Vector2d position(0.01, 0.02);

    double phase = ForwardModel::compute_phase(k, position);

    // Expected: -(kx * x + ky * y) = -(1000 * 0.01 + 2000 * 0.02) = -(10 + 40) = -50
    EXPECT_NEAR(phase, -50.0, 1e-10);
}

// Test compute_phasor returns unit magnitude with correct phase
TEST_F(ForwardModelTest, ComputePhasor) {
    Eigen::Vector2d k(M_PI / 2.0, 0.0);  // k such that phase at x=1 is -pi/2
    Eigen::Vector2d position(1.0, 0.0);

    auto phasor = ForwardModel::compute_phasor(k, position);

    EXPECT_NEAR(std::abs(phasor), 1.0, 1e-10);
    EXPECT_NEAR(std::arg(phasor), -M_PI / 2.0, 1e-10);
}

// ----------------------------------------------------------------------------
// K-space extraction from measurements
// ----------------------------------------------------------------------------

// Test extract_kspace_slice returns empty for empty measurements
TEST_F(ForwardModelTest, ExtractKSpaceSliceEmpty) {
    ForwardModel model(detector_, optics_);

    aurora::shared::MeasurementTensor empty_tensor;
    auto slice = model.extract_kspace_slice(empty_tensor, 0, kWavelength);

    EXPECT_EQ(slice.samples_size(), 0);
    EXPECT_EQ(slice.range_bin(), 0u);
    EXPECT_NEAR(slice.wavelength_m(), kWavelength, 1e-15);
    EXPECT_FALSE(slice.has_grid_hints());
    EXPECT_FALSE(slice.has_burst_id());
    EXPECT_FALSE(slice.has_timestamp_s());
}

// Test extract_kspace_slice with actual measurements
TEST_F(ForwardModelTest, ExtractKSpaceSliceWithData) {
    ForwardModel model(detector_, optics_);

    // Create a MeasurementTensor with data
    aurora::shared::MeasurementTensor tensor;
    tensor.set_burst_id(12345);
    tensor.set_timestamp_s(1.0);

    auto* wavelength_channel = tensor.add_data();
    wavelength_channel->set_wavelength_m(kWavelength);

    for (uint32_t d = 0; d < kNumDetectors; ++d) {
        auto* det_channel = wavelength_channel->add_detectors();
        det_channel->set_detector_index(d);

        // Add range data for 4 bins
        for (int bin = 0; bin < 4; ++bin) {
            auto* range_data = det_channel->add_range_data();
            range_data->set_real(1.0 + bin * 0.1 + d * 0.01);
            range_data->set_imag(0.5 + bin * 0.05 + d * 0.005);
        }
    }

    // Extract slice for range bin 2
    auto slice = model.extract_kspace_slice(tensor, 2, kWavelength);

    ASSERT_EQ(slice.samples_size(), static_cast<int>(kNumDetectors));
    EXPECT_EQ(slice.range_bin(), 2u);
    EXPECT_NEAR(slice.wavelength_m(), kWavelength, 1e-15);
    EXPECT_EQ(slice.burst_id(), 12345u);
    EXPECT_NEAR(slice.timestamp_s(), 1.0, 1e-10);

    // Check first sample
    EXPECT_EQ(slice.samples(0).detector_index(), 0u);
    EXPECT_NEAR(slice.samples(0).kx(), expected_kx(0, kWavelength), 1e-6);
    EXPECT_NEAR(slice.samples(0).ky(), 0.0, 1e-6);
    EXPECT_DOUBLE_EQ(slice.samples(0).weight(), 1.0);  // unit sample weight
    EXPECT_NEAR(slice.samples(0).value().real(), 1.2, 1e-6);  // 1.0 + 2*0.1 + 0*0.01
    EXPECT_NEAR(slice.samples(0).value().imag(), 0.6, 1e-6);  // 0.5 + 2*0.05 + 0*0.005

    // Check last sample
    const auto& last = slice.samples(static_cast<int>(kNumDetectors) - 1);
    EXPECT_EQ(last.detector_index(), kNumDetectors - 1);
    EXPECT_NEAR(last.kx(), expected_kx(kNumDetectors - 1, kWavelength), 1e-6);
    EXPECT_NEAR(last.value().real(), 1.2 + (kNumDetectors - 1) * 0.01, 1e-6);
}

// Test extract_kspace_slice only picks the requested wavelength record
TEST_F(ForwardModelTest, ExtractKSpaceSliceSelectsWavelength) {
    ForwardModel model(detector_, optics_);

    auto tensor = make_tensor(kWavelength);
    *tensor.add_data() = make_tensor(kOtherWavelength).data(0);

    auto slice = model.extract_kspace_slice(tensor, 0, kOtherWavelength);

    ASSERT_EQ(slice.samples_size(), static_cast<int>(kNumDetectors));
    for (const auto& sample : slice.samples()) {
        EXPECT_NEAR(sample.kx(), expected_kx(sample.detector_index(), kOtherWavelength), 1e-6);
    }
}

// Test extract_kspace_slice skips detector records outside the geometry
TEST_F(ForwardModelTest, ExtractKSpaceSliceSkipsUnknownDetector) {
    ForwardModel model(detector_, optics_);

    aurora::shared::MeasurementTensor tensor;
    auto* wavelength_channel = tensor.add_data();
    wavelength_channel->set_wavelength_m(kWavelength);
    auto* det_channel = wavelength_channel->add_detectors();
    det_channel->set_detector_index(kNumDetectors);  // Not in the array
    det_channel->add_range_data()->set_real(1.0);

    auto slice = model.extract_kspace_slice(tensor, 0, kWavelength);

    EXPECT_EQ(slice.samples_size(), 0);
}

// Test extract_kspace_slice skips detector records without the requested range bin
TEST_F(ForwardModelTest, ExtractKSpaceSliceSkipsMissingRangeBin) {
    ForwardModel model(detector_, optics_);

    auto tensor = make_tensor(kWavelength, 2);
    auto slice = model.extract_kspace_slice(tensor, 2, kWavelength);

    EXPECT_EQ(slice.samples_size(), 0);
}

// Test extract_kspace_slice uses scene K (LO tilt applied) for sample coordinates
TEST_F(ForwardModelTest, ExtractKSpaceSliceAppliesLoTilt) {
    const double lo_tilt_x = 0.02;  // rad

    OpticsConfig tilted_optics = optics_;
    tilted_optics.set_lo_tilt_x_rad(lo_tilt_x);

    ForwardModel model(detector_, tilted_optics);

    auto tensor = make_tensor(kWavelength);
    auto slice = model.extract_kspace_slice(tensor, 0, kWavelength);

    ASSERT_EQ(slice.samples_size(), static_cast<int>(kNumDetectors));
    for (const auto& sample : slice.samples()) {
        auto expected = model.get_effective_k(sample.detector_index(), kWavelength);
        EXPECT_NEAR(sample.kx(), expected.x(), 1e-6);
        EXPECT_NEAR(sample.ky(), expected.y(), 1e-6);
        EXPECT_GT(sample.kx(), expected_kx(sample.detector_index(), kWavelength));
    }
}

// Test extract_kspace_slice propagates bandwidth to every sample
TEST_F(ForwardModelTest, ExtractKSpaceSlicePropagatesBandwidth) {
    ForwardModel model(detector_, optics_);

    auto tensor = make_tensor(kWavelength);
    auto slice = model.extract_kspace_slice(tensor, 0, kWavelength);

    // Verify all samples have bandwidth
    ASSERT_EQ(slice.samples_size(), static_cast<int>(kNumDetectors));

    for (int i = 0; i < slice.samples_size(); ++i) {
        const auto& sample = slice.samples(i);
        EXPECT_TRUE(sample.has_kx_bandwidth())
            << "Sample " << i << " missing kx_bandwidth";
        EXPECT_TRUE(sample.has_ky_bandwidth())
            << "Sample " << i << " missing ky_bandwidth";
        EXPECT_GT(sample.kx_bandwidth(), 0.0)
            << "Sample " << i << " has zero kx_bandwidth";
        EXPECT_GT(sample.ky_bandwidth(), 0.0)
            << "Sample " << i << " has zero ky_bandwidth";
    }
}

// Test extract_kspace_slice bandwidth values match the pixel size
TEST_F(ForwardModelTest, ExtractKSpaceSliceBandwidthMatchesPixelSize) {
    ForwardModel model(detector_, optics_);

    auto tensor = make_tensor(kWavelength);
    auto slice = model.extract_kspace_slice(tensor, 0, kWavelength);

    ASSERT_EQ(slice.samples_size(), static_cast<int>(kNumDetectors));

    const double expected_kx_bw = kTwoPi * kPixelWidth / (kFocalLength * kWavelength);
    const double expected_ky_bw = kTwoPi * kPixelHeight / (kFocalLength * kWavelength);

    for (int i = 0; i < slice.samples_size(); ++i) {
        const auto& sample = slice.samples(i);
        uint32_t det_idx = sample.detector_index();

        EXPECT_NEAR(sample.kx_bandwidth(), expected_kx_bw, 1e-6)
            << "Detector " << det_idx << " kx_bandwidth mismatch";
        EXPECT_NEAR(sample.ky_bandwidth(), expected_ky_bw, 1e-6)
            << "Detector " << det_idx << " ky_bandwidth mismatch";

        auto k_sample = model.get_k_sample(det_idx, kWavelength);
        EXPECT_DOUBLE_EQ(sample.kx_bandwidth(), k_sample.kx_bandwidth());
        EXPECT_DOUBLE_EQ(sample.ky_bandwidth(), k_sample.ky_bandwidth());
    }
}

// Test extract_kspace_slice throws when a pixel has zero width (no K bandwidth)
TEST_F(ForwardModelTest, ExtractKSpaceSliceThrowsOnZeroPixelWidth) {
    DetectorGeometry zero_width = detector_;
    zero_width.mutable_arrays(0)->mutable_pixels(0)->set_width_m(0.0);

    ForwardModel model(zero_width, optics_);

    auto tensor = make_tensor(kWavelength);
    EXPECT_THROW(model.extract_kspace_slice(tensor, 0, kWavelength), std::runtime_error);
}

// Test extract_kspace_slice throws when a pixel has zero height (no K bandwidth)
TEST_F(ForwardModelTest, ExtractKSpaceSliceThrowsOnZeroPixelHeight) {
    DetectorGeometry zero_height = detector_;
    zero_height.mutable_arrays(0)->mutable_pixels(2)->set_height_m(0.0);

    ForwardModel model(zero_height, optics_);

    auto tensor = make_tensor(kWavelength);
    EXPECT_THROW(model.extract_kspace_slice(tensor, 0, kWavelength), std::runtime_error);
}

// Test extract_kspace_slice computes grid hints spanning the detector array
TEST_F(ForwardModelTest, ExtractKSpaceSliceGridHints) {
    ForwardModel model(detector_, optics_);

    auto tensor = make_tensor(kWavelength);
    auto slice = model.extract_kspace_slice(tensor, 0, kWavelength);

    ASSERT_EQ(slice.samples_size(), static_cast<int>(kNumDetectors));
    EXPECT_TRUE(slice.has_grid_hints());
    const auto& hints = slice.grid_hints();

    // K bounds should span the first to the last pixel (linear array along x)
    EXPECT_NEAR(hints.kx_min(), expected_kx(0, kWavelength), 1e-6);
    EXPECT_NEAR(hints.kx_max(), expected_kx(kNumDetectors - 1, kWavelength), 1e-6);
    EXPECT_NEAR(hints.ky_min(), 0.0, 1e-6);
    EXPECT_NEAR(hints.ky_max(), 0.0, 1e-6);
    EXPECT_LT(hints.kx_min(), hints.kx_max());
    EXPECT_GE(hints.suggested_nx(), 8u);
    EXPECT_GE(hints.suggested_ny(), 8u);
}

// Test extract_all_range_slices
TEST_F(ForwardModelTest, ExtractAllRangeSlices) {
    ForwardModel model(detector_, optics_);

    // Create tensor with 4 range bins on a single detector
    aurora::shared::MeasurementTensor tensor;
    auto* wl = tensor.add_data();
    wl->set_wavelength_m(kWavelength);
    auto* det = wl->add_detectors();
    det->set_detector_index(0);
    for (int i = 0; i < 4; ++i) {
        det->add_range_data()->set_real(1.0 + i);
    }

    auto collection = model.extract_all_range_slices(tensor, kWavelength);

    EXPECT_NEAR(collection.wavelength_m(), kWavelength, 1e-15);
    ASSERT_EQ(collection.slices_size(), 4);

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(collection.slices(i).range_bin(), static_cast<uint32_t>(i));
        ASSERT_EQ(collection.slices(i).samples_size(), 1);
        EXPECT_NEAR(collection.slices(i).samples(0).value().real(), 1.0 + i, 1e-10);
    }
}

// Test extract_all_range_slices on an empty tensor yields no slices
TEST_F(ForwardModelTest, ExtractAllRangeSlicesEmpty) {
    ForwardModel model(detector_, optics_);

    aurora::shared::MeasurementTensor empty_tensor;
    auto collection = model.extract_all_range_slices(empty_tensor, kWavelength);

    EXPECT_NEAR(collection.wavelength_m(), kWavelength, 1e-15);
    EXPECT_EQ(collection.slices_size(), 0);
}

// ----------------------------------------------------------------------------
// Accessors
// ----------------------------------------------------------------------------

// Test ForwardModel accessors
TEST_F(ForwardModelTest, Accessors) {
    ForwardModel model(detector_, optics_);

    EXPECT_EQ(model.detector().arrays_size(), 1);
    EXPECT_EQ(model.detector().arrays(0).pixels_size(), static_cast<int>(kNumDetectors));
    EXPECT_NEAR(model.optics().fourier_lens_focal_m(), kFocalLength, 1e-10);
    EXPECT_EQ(model.geometry().num_detectors(), kNumDetectors);
    EXPECT_NEAR(model.geometry().focal_length_m(), kFocalLength, 1e-10);
}

// ============================================================================
// Integration Test: ForwardModel with the test geometry builder
// ============================================================================

// Verifies that the 8-pixel detector from the test helper yields
// k_j = 2*pi * x_j / (f * lambda) per pixel.
TEST(ForwardModelBuilderIntegrationTest, ForwardModelUsesGeometryKValues) {
    DetectorGeometry detector = aurora::test::create_reference_detector();

    OpticsConfig optics;
    optics.set_fourier_lens_focal_m(0.05);
    optics.add_wavelengths_m(1550e-9);

    const double wavelength = 1550e-9;

    ForwardModel model = aurora::test::TestForwardModelBuilder::build(detector, optics);

    const auto& array = detector.arrays(0);
    ASSERT_EQ(model.num_detectors(), static_cast<uint32_t>(array.pixels_size()));

    const double f = optics.fourier_lens_focal_m();
    for (int d = 0; d < array.pixels_size(); ++d) {
        const auto& pixel = array.pixels(d);
        const double k_det = kTwoPi * pixel.x_m() / (f * wavelength);
        const double k_bw = kTwoPi * pixel.width_m() / (f * wavelength);

        auto sample = model.get_k_sample(static_cast<uint32_t>(d), wavelength);

        EXPECT_NEAR(sample.kx_eff(), k_det, 1e-6);
        EXPECT_NEAR(sample.ky_eff(), 0.0, 1e-6);
        EXPECT_NEAR(sample.kx_bandwidth(), k_bw, 1e-6);
    }
}

}  // namespace
}  // namespace aurora::forward
