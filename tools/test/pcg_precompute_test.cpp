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
 * @file pcg_precompute_test.cpp
 * @brief Tests for the PCG precompute CLI tool.
 *
 * Tests protobuf loading and PCG precomputation functionality.
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <set>
#include <memory>
#include <stdexcept>
#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "reconstruction/direct_operator.h"
#include "reconstruction/pcg_precomputation.h"
#include "forward/forward_model.h"
#include "calibration_artifact.pb.h"
#include "detector_geometry.pb.h"
#include "forward_config.pb.h"
#include "k_space_slice.pb.h"
#include "optics_config.pb.h"
#include "pcg_config.pb.h"
#include "pcg_precomputed_data.pb.h"

namespace aurora::tools {
namespace {

constexpr uint32_t kNumDetectors = 8;
constexpr double kFocalLength = 0.05;  // m
constexpr double kPixelFillFactor = 0.8;

// Adjacent pixels are kKSpacing apart in K-space at kReferenceWavelength, so
// the receiver's K-samples (up to +-3.5 * kKSpacing) sit well inside the
// Nyquist band of the 0.01 m test grids (pi / 0.01 ~ 314 rad/m).
constexpr double kKSpacing = 40.0;  // rad/m
constexpr double kReferenceWavelength = 905e-9;

// RAII helper for temporary file cleanup
class TempFile {
  public:
    explicit TempFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {}

    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    std::string path() const { return path_.string(); }

  private:
    std::filesystem::path path_;
};

// Fourier-plane pitch that places adjacent pixels k_spacing apart in K-space:
// k = 2*pi * x / (f * lambda)  =>  x = k * f * lambda / (2*pi)
double pitch_for_k_spacing(double k_spacing, double focal_length_m, double wavelength_m) {
    return k_spacing * focal_length_m * wavelength_m / (2.0 * M_PI);
}

// Create the ForwardConfig (8-pixel detector array + optics) the tool loads
forward::ForwardConfig create_test_forward_config(
    const std::string& geometry_id,
    const std::vector<double>& wavelengths) {

    forward::ForwardConfig config;
    const double pitch = pitch_for_k_spacing(
        kKSpacing, kFocalLength, kReferenceWavelength);

    auto* array = config.mutable_detector()->add_arrays();
    array->set_array_id(geometry_id);
    array->set_nominal_pitch_m(pitch);
    array->set_sensitivity_type(forward::DETECTOR_SENSITIVITY_SINC);
    for (double wavelength : wavelengths) {
        array->add_wavelengths_m(wavelength);
    }
    for (uint32_t i = 0; i < kNumDetectors; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((i - (kNumDetectors - 1) / 2.0) * pitch);
        pixel->set_y_m(0.0);
        pixel->set_width_m(kPixelFillFactor * pitch);
        pixel->set_height_m(kPixelFillFactor * pitch);
    }

    auto* optics = config.mutable_optics();
    optics->set_fourier_lens_focal_m(kFocalLength);
    for (double wavelength : wavelengths) {
        optics->add_wavelengths_m(wavelength);
    }
    optics->set_k_max_rad_per_m(1e5);

    return config;
}

// Receiver physics model built from a ForwardConfig
struct Receiver {
    explicit Receiver(const forward::ForwardConfig& config) : model(config) {}

    forward::ForwardModel model;
};

// Create a KSpaceSlice (no measurement values) from the receiver's K-samples
shared::KSpaceSlice create_k_space_slice(
    const forward::ForwardModel& forward_model,
    double wavelength_m) {

    shared::KSpaceSlice slice;
    slice.set_wavelength_m(wavelength_m);
    slice.set_range_bin(0);

    for (const auto& s : forward_model.get_all_k_samples(wavelength_m)) {
        Eigen::Vector2d k = forward_model.get_effective_k(s.detector_index(), wavelength_m);
        auto* sample = slice.add_samples();
        sample->set_kx(k.x());
        sample->set_ky(k.y());
        sample->set_weight(1.0);
        sample->set_detector_index(s.detector_index());
        sample->set_kx_bandwidth(s.kx_bandwidth());
        sample->set_ky_bandwidth(s.ky_bandwidth());
    }
    return slice;
}

// Measurement weights: the per-sample weight of each K-sample
Eigen::VectorXd sample_weights(const shared::KSpaceSlice& slice) {
    Eigen::VectorXd weights(slice.samples_size());
    for (int i = 0; i < slice.samples_size(); ++i) {
        weights(i) = slice.samples(i).weight();
    }
    return weights;
}

// Create a DirectOperator over the receiver's K-samples for a grid
std::unique_ptr<reconstruction::DirectOperator> create_direct_operator(
    const shared::KSpaceSlice& slice,
    const Receiver& receiver,
    const reconstruction::GridSpec& grid) {

    Eigen::Vector2i grid_size(
        static_cast<int>(grid.size_x()),
        static_cast<int>(grid.size_y()));
    Eigen::Vector2d grid_spacing(grid.spacing_x_m(), grid.spacing_y_m());
    Eigen::Vector2d grid_origin(grid.origin_x_m(), grid.origin_y_m());

    forward::BandwidthConfig bw_config;
    bw_config.set_window_type(forward::BANDWIDTH_WINDOW_SINC);
    bw_config.set_quadrature_order(4);

    return std::make_unique<reconstruction::DirectOperator>(
        slice, receiver.model,
        bw_config,
        grid_size, grid_spacing, grid_origin);
}

// Helper to create a test calibration artifact
shared::CalibrationArtifact create_test_calibration(
    const std::string& calibration_id,
    const std::vector<double>& wavelengths,
    int num_detectors = 8) {

    shared::CalibrationArtifact calibration;
    calibration.set_calibration_id(calibration_id);
    calibration.set_timestamp_s(
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // Add detector calibrations
    for (int d = 0; d < num_detectors; ++d) {
        auto* det_cal = calibration.add_detector_calibrations();
        det_cal->set_detector_index(static_cast<uint32_t>(d));
        det_cal->mutable_complex_gain()->set_real(1.0);
        det_cal->mutable_complex_gain()->set_imag(0.0);
        det_cal->set_timing_skew_s(0.0);
    }

    // Add K-space calibrations for each wavelength
    for (double wavelength : wavelengths) {
        auto* k_cal = calibration.add_k_space_calibrations();
        k_cal->set_wavelength_m(wavelength);
        k_cal->set_wavelength_scale(1.0);
        for (int d = 0; d < num_detectors; ++d) {
            k_cal->add_kx_offsets(0.0);
            k_cal->add_ky_offsets(0.0);
        }
    }

    return calibration;
}

// Helper to save a protobuf to file
template <typename T>
void save_protobuf(const T& message, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    if (!message.SerializeToOstream(&file)) {
        throw std::runtime_error("Failed to serialize protobuf to file: " + path);
    }
}

// Extract the scene K-vectors of the receiver's K-samples at one wavelength
std::vector<Eigen::Vector2d> extract_k_vectors(
    const forward::ForwardModel& forward_model,
    double wavelength) {

    std::vector<Eigen::Vector2d> k_vectors;
    for (const auto& sample : forward_model.get_all_k_samples(wavelength)) {
        k_vectors.push_back(
            forward_model.get_effective_k(sample.detector_index(), wavelength));
    }
    return k_vectors;
}

class PcgPrecomputeToolTest : public ::testing::Test {
  protected:
    void SetUp() override {
        wavelengths_ = {905e-9, 1550e-9};
        geometry_id_ = "test_geometry_v1";
        calibration_id_ = "test_calibration_v1";

        // Create test data files
        forward_config_file_ = std::make_unique<TempFile>("test_forward_config.pb");
        calibration_file_ = std::make_unique<TempFile>("test_calibration.pb");
        output_file_ = std::make_unique<TempFile>("test_output.pb");

        auto forward_config = create_test_forward_config(geometry_id_, wavelengths_);
        auto calibration = create_test_calibration(
            calibration_id_, wavelengths_);

        save_protobuf(forward_config, forward_config_file_->path());
        save_protobuf(calibration, calibration_file_->path());
    }

    // Load the forward configuration written by SetUp
    forward::ForwardConfig load_forward_config() const {
        std::ifstream file(forward_config_file_->path(), std::ios::binary);
        forward::ForwardConfig config;
        if (!file.good() || !config.ParseFromIstream(&file)) {
            throw std::runtime_error("Failed to load forward configuration");
        }
        return config;
    }

    // Build the receiver model from the forward configuration written by SetUp
    Receiver load_receiver() const {
        return Receiver(load_forward_config());
    }

    std::vector<double> wavelengths_;
    std::string geometry_id_;
    std::string calibration_id_;
    std::unique_ptr<TempFile> forward_config_file_;
    std::unique_ptr<TempFile> calibration_file_;
    std::unique_ptr<TempFile> output_file_;
};

// =============================================================================
// Forward Configuration Loading Tests
// =============================================================================

TEST_F(PcgPrecomputeToolTest, ForwardConfig_LoadsCorrectly) {
    std::ifstream file(forward_config_file_->path(), std::ios::binary);
    ASSERT_TRUE(file.good());

    forward::ForwardConfig config;
    ASSERT_TRUE(config.ParseFromIstream(&file));

    // Detector geometry: one 8-pixel array identified by the geometry id
    ASSERT_EQ(config.detector().arrays_size(), 1);
    const auto& array = config.detector().arrays(0);
    EXPECT_EQ(array.array_id(), geometry_id_);
    EXPECT_EQ(array.pixels_size(), static_cast<int>(kNumDetectors));
    for (const auto& pixel : array.pixels()) {
        EXPECT_GT(pixel.width_m(), 0.0);
        EXPECT_GT(pixel.height_m(), 0.0);
    }

    // Optics: focal length and both wavelengths are present
    EXPECT_DOUBLE_EQ(config.optics().fourier_lens_focal_m(), kFocalLength);
    std::set<double> found_wavelengths(config.optics().wavelengths_m().begin(),
                                       config.optics().wavelengths_m().end());
    EXPECT_EQ(found_wavelengths.size(), 2u);
    EXPECT_TRUE(found_wavelengths.count(905e-9) > 0);
    EXPECT_TRUE(found_wavelengths.count(1550e-9) > 0);
}

TEST_F(PcgPrecomputeToolTest, Calibration_LoadsCorrectly) {
    std::ifstream file(calibration_file_->path(), std::ios::binary);
    ASSERT_TRUE(file.good());

    shared::CalibrationArtifact calibration;
    ASSERT_TRUE(calibration.ParseFromIstream(&file));

    EXPECT_EQ(calibration.calibration_id(), calibration_id_);
    EXPECT_EQ(calibration.detector_calibrations_size(), 8);
    EXPECT_EQ(calibration.k_space_calibrations_size(), 2);
}

// =============================================================================
// K-Sample Extraction Tests
// =============================================================================

TEST_F(PcgPrecomputeToolTest, ExtractKSamples_SingleWavelength) {
    auto receiver = load_receiver();

    // Extract K-samples for 905nm wavelength
    auto k_vectors = extract_k_vectors(receiver.model, 905e-9);

    // One K-sample per detector pixel
    EXPECT_EQ(k_vectors.size(), static_cast<size_t>(kNumDetectors));

    // Adjacent pixels are kKSpacing apart in kx at the reference wavelength
    for (size_t i = 1; i < k_vectors.size(); ++i) {
        EXPECT_NEAR(k_vectors[i].x() - k_vectors[i - 1].x(), kKSpacing, 1e-6);
        EXPECT_NEAR(k_vectors[i].y(), 0.0, 1e-9);
    }
}

// =============================================================================
// Precomputation Pipeline Tests
// =============================================================================

TEST_F(PcgPrecomputeToolTest, Precomputation_ProducesValidOutput) {
    auto receiver = load_receiver();

    // Set up grid
    reconstruction::GridSpec grid;
    grid.set_size_x(16);
    grid.set_size_y(16);
    grid.set_spacing_x_m(0.01);
    grid.set_spacing_y_m(0.01);
    grid.set_origin_x_m(-0.08);
    grid.set_origin_y_m(-0.08);

    // K-samples for first wavelength
    const double wavelength = 905e-9;
    auto slice = create_k_space_slice(receiver.model, wavelength);
    ASSERT_GT(slice.samples_size(), 0);

    // Build operator
    auto op = create_direct_operator(slice, receiver, grid);
    Eigen::VectorXd weights = sample_weights(slice);

    // Compute precomputed data
    auto data = reconstruction::PcgPrecomputation::compute(
        *op, weights, grid, wavelength,
        reconstruction::REGULARIZER_GRADIENT,
        reconstruction::PRECONDITIONER_HYBRID,
        geometry_id_, calibration_id_);

    // Verify output structure
    EXPECT_EQ(data.inputs().geometry_id(), geometry_id_);
    EXPECT_EQ(data.inputs().calibration_id(), calibration_id_);
    EXPECT_DOUBLE_EQ(data.inputs().wavelength_m(), wavelength);
    EXPECT_EQ(data.inputs().regularizer_type(),
              reconstruction::REGULARIZER_GRADIENT);
    EXPECT_EQ(data.inputs().preconditioner_type(),
              reconstruction::PRECONDITIONER_HYBRID);

    // Check array sizes
    size_t grid_size = grid.size_x() * grid.size_y();
    EXPECT_EQ(static_cast<size_t>(data.psf_size()), 2 * grid_size);
    EXPECT_EQ(static_cast<size_t>(data.transfer_function_size()), 2 * grid_size);
    EXPECT_EQ(static_cast<size_t>(data.regularizer_spectrum_size()), grid_size);
    EXPECT_EQ(static_cast<size_t>(data.operator_diagonal_size()), grid_size);
}

TEST_F(PcgPrecomputeToolTest, Precomputation_MultiWavelength) {
    auto receiver = load_receiver();

    // Set up grid
    reconstruction::GridSpec grid;
    grid.set_size_x(16);
    grid.set_size_y(16);
    grid.set_spacing_x_m(0.01);
    grid.set_spacing_y_m(0.01);
    grid.set_origin_x_m(-0.08);
    grid.set_origin_y_m(-0.08);

    reconstruction::PcgPrecomputedDataSet result;
    result.set_geometry_id(geometry_id_);
    result.set_calibration_id(calibration_id_);
    *result.mutable_grid() = grid;

    for (double wavelength : wavelengths_) {
        auto slice = create_k_space_slice(receiver.model, wavelength);
        if (slice.samples_size() == 0) continue;

        auto op = create_direct_operator(slice, receiver, grid);
        Eigen::VectorXd weights = sample_weights(slice);

        auto data = reconstruction::PcgPrecomputation::compute(
            *op, weights, grid, wavelength,
            reconstruction::REGULARIZER_GRADIENT,
            reconstruction::PRECONDITIONER_HYBRID,
            geometry_id_, calibration_id_);

        *result.add_wavelength_data() = data;
    }

    EXPECT_EQ(result.wavelength_data_size(), 2);
    EXPECT_NE(result.wavelength_data(0).inputs().wavelength_m(),
              result.wavelength_data(1).inputs().wavelength_m());
}

TEST_F(PcgPrecomputeToolTest, Precomputation_SaveLoadRoundtrip) {
    auto receiver = load_receiver();

    // Set up grid
    reconstruction::GridSpec grid;
    grid.set_size_x(16);
    grid.set_size_y(16);
    grid.set_spacing_x_m(0.01);
    grid.set_spacing_y_m(0.01);
    grid.set_origin_x_m(-0.08);
    grid.set_origin_y_m(-0.08);

    // Create data set
    reconstruction::PcgPrecomputedDataSet data_set;
    data_set.set_geometry_id(geometry_id_);
    data_set.set_calibration_id(calibration_id_);
    *data_set.mutable_grid() = grid;

    // Compute for one wavelength
    const double wavelength = 905e-9;
    auto slice = create_k_space_slice(receiver.model, wavelength);

    auto op = create_direct_operator(slice, receiver, grid);
    Eigen::VectorXd weights = sample_weights(slice);

    auto data = reconstruction::PcgPrecomputation::compute(
        *op, weights, grid, wavelength,
        reconstruction::REGULARIZER_GRADIENT,
        reconstruction::PRECONDITIONER_HYBRID,
        geometry_id_, calibration_id_);

    *data_set.add_wavelength_data() = data;

    // Save
    reconstruction::PcgPrecomputation::save(data_set, output_file_->path());

    // Load
    auto loaded = reconstruction::PcgPrecomputation::load(output_file_->path());

    // Verify
    EXPECT_EQ(loaded.wavelength_data_size(), 1);
    EXPECT_EQ(loaded.geometry_id(), geometry_id_);
    EXPECT_EQ(loaded.calibration_id(), calibration_id_);
    EXPECT_EQ(loaded.grid().size_x(), grid.size_x());
    EXPECT_EQ(loaded.grid().size_y(), grid.size_y());

    // Verify wavelength data
    const auto& loaded_data = loaded.wavelength_data(0);
    EXPECT_DOUBLE_EQ(loaded_data.inputs().wavelength_m(), wavelength);
    EXPECT_EQ(loaded_data.psf_size(), data.psf_size());
    EXPECT_EQ(loaded_data.transfer_function_size(), data.transfer_function_size());
}

// =============================================================================
// Different Regularizer Tests
// =============================================================================

TEST_F(PcgPrecomputeToolTest, Precomputation_AllRegularizerTypes) {
    auto receiver = load_receiver();

    reconstruction::GridSpec grid;
    grid.set_size_x(8);
    grid.set_size_y(8);
    grid.set_spacing_x_m(0.01);
    grid.set_spacing_y_m(0.01);
    grid.set_origin_x_m(-0.04);
    grid.set_origin_y_m(-0.04);

    const double wavelength = 905e-9;
    auto slice = create_k_space_slice(receiver.model, wavelength);

    auto op = create_direct_operator(slice, receiver, grid);
    Eigen::VectorXd weights = sample_weights(slice);

    // Test all regularizer types
    std::vector<reconstruction::RegularizerType> types = {
        reconstruction::REGULARIZER_TIKHONOV,
        reconstruction::REGULARIZER_GRADIENT,
        reconstruction::REGULARIZER_LAPLACIAN
    };

    for (auto reg_type : types) {
        SCOPED_TRACE("Regularizer: " + std::to_string(reg_type));

        auto data = reconstruction::PcgPrecomputation::compute(
            *op, weights, grid, wavelength,
            reg_type,
            reconstruction::PRECONDITIONER_JACOBI,
            geometry_id_, calibration_id_);

        EXPECT_EQ(data.inputs().regularizer_type(), reg_type);
        EXPECT_GT(data.regularizer_spectrum_size(), 0);
    }
}

// =============================================================================
// Different Preconditioner Tests
// =============================================================================

TEST_F(PcgPrecomputeToolTest, Precomputation_AllPreconditionerTypes) {
    auto receiver = load_receiver();

    reconstruction::GridSpec grid;
    grid.set_size_x(8);
    grid.set_size_y(8);
    grid.set_spacing_x_m(0.01);
    grid.set_spacing_y_m(0.01);
    grid.set_origin_x_m(-0.04);
    grid.set_origin_y_m(-0.04);

    const double wavelength = 905e-9;
    auto slice = create_k_space_slice(receiver.model, wavelength);

    auto op = create_direct_operator(slice, receiver, grid);
    Eigen::VectorXd weights = sample_weights(slice);

    // Test all preconditioner types
    std::vector<reconstruction::PreconditionerType> types = {
        reconstruction::PRECONDITIONER_NONE,
        reconstruction::PRECONDITIONER_JACOBI,
        reconstruction::PRECONDITIONER_CIRCULANT,
        reconstruction::PRECONDITIONER_HYBRID
    };

    for (auto prec_type : types) {
        SCOPED_TRACE("Preconditioner: " + std::to_string(prec_type));

        auto data = reconstruction::PcgPrecomputation::compute(
            *op, weights, grid, wavelength,
            reconstruction::REGULARIZER_GRADIENT,
            prec_type,
            geometry_id_, calibration_id_);

        EXPECT_EQ(data.inputs().preconditioner_type(), prec_type);
    }
}

// =============================================================================
// File Size and Performance Tests
// =============================================================================

TEST_F(PcgPrecomputeToolTest, OutputFile_HasReasonableSize) {
    auto receiver = load_receiver();

    reconstruction::GridSpec grid;
    grid.set_size_x(32);
    grid.set_size_y(32);
    grid.set_spacing_x_m(0.001);
    grid.set_spacing_y_m(0.001);
    grid.set_origin_x_m(-0.016);
    grid.set_origin_y_m(-0.016);

    reconstruction::PcgPrecomputedDataSet data_set;
    data_set.set_geometry_id(geometry_id_);
    data_set.set_calibration_id(calibration_id_);
    *data_set.mutable_grid() = grid;

    for (double wavelength : wavelengths_) {
        auto slice = create_k_space_slice(receiver.model, wavelength);
        if (slice.samples_size() == 0) continue;

        auto op = create_direct_operator(slice, receiver, grid);
        Eigen::VectorXd weights = sample_weights(slice);

        auto data = reconstruction::PcgPrecomputation::compute(
            *op, weights, grid, wavelength,
            reconstruction::REGULARIZER_GRADIENT,
            reconstruction::PRECONDITIONER_HYBRID,
            geometry_id_, calibration_id_);

        *data_set.add_wavelength_data() = data;
    }

    reconstruction::PcgPrecomputation::save(data_set, output_file_->path());

    // Check file size is reasonable
    // For 32x32 grid with 2 wavelengths:
    // - PSF: 32*32*2*8 = 16KB per wavelength (complex double)
    // - Transfer: 16KB per wavelength
    // - Regularizer spectrum: 32*32*8 = 8KB per wavelength
    // - Diagonal: 8KB per wavelength
    // Total: ~96KB for 2 wavelengths + overhead
    auto file_size = std::filesystem::file_size(output_file_->path());

    EXPECT_GT(file_size, 50000u);  // At least 50KB
    EXPECT_LT(file_size, 500000u);  // Less than 500KB
}

}  // namespace
}  // namespace aurora::tools
