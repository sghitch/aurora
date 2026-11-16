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

#include "reconstruction/pcg_precomputation.h"

#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "forward/fft2d.h"
#include "forward/forward_model.h"
#include "reconstruction/direct_operator.h"
#include "test_helpers.h"

#include "detector_geometry.pb.h"
#include "k_space_slice.pb.h"
#include "optics_config.pb.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aurora::reconstruction {

namespace {

constexpr uint32_t kNumDetectors = 8;   // point-detector geometry size
constexpr double kPitch = 0.5e-3;  // m

// Helper to create a KSpaceSlice from K-vectors and sample weights.
// detector_index is left at 0: with the point-detector geometry this maps
// every sample to a zero-bandwidth window, i.e. a point sample at (kx, ky).
aurora::shared::KSpaceSlice create_slice(
    const std::vector<Eigen::Vector2d>& k_vectors,
    const std::vector<double>& sample_weights) {
    aurora::shared::KSpaceSlice slice;
    slice.set_wavelength_m(1550e-9);
    slice.set_range_bin(0);

    for (size_t i = 0; i < k_vectors.size(); ++i) {
        auto* sample = slice.add_samples();
        sample->set_kx(k_vectors[i].x());
        sample->set_ky(k_vectors[i].y());
        sample->set_weight(i < sample_weights.size() ? sample_weights[i] : 1.0);
    }
    return slice;
}

}  // namespace

// RAII helper for temporary file cleanup
class TempFile {
  public:
    explicit TempFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {}
    ~TempFile() { std::filesystem::remove(path_); }
    std::string path() const { return path_.string(); }
  private:
    std::filesystem::path path_;
};

class PcgPrecomputationTest : public ::testing::Test {
  protected:
    static constexpr double kWavelength = 1550e-9;

    void SetUp() override {
        // Point-detector geometry (zero pixel size): no K-space bandwidth, so
        // the operator treats the random K-vectors in each slice as point
        // samples rather than integrating over a K-window.
        detector_geom_ = aurora::reconstruction::test::create_point_detector_geometry(
            "test_detector", kNumDetectors, kPitch, kWavelength);

        // Set up optics configuration
        optics_.set_fourier_lens_focal_m(0.05);
        optics_.add_wavelengths_m(kWavelength);
        optics_.set_k_max_rad_per_m(1e5);

        // Create ForwardModel
        forward_model_ = std::make_unique<aurora::forward::ForwardModel>(
            detector_geom_, optics_);

        // Default grid parameters
        grid_.set_size_x(16);
        grid_.set_size_y(16);
        grid_.set_spacing_x_m(0.01);
        grid_.set_spacing_y_m(0.01);
        grid_.set_origin_x_m(-0.08);
        grid_.set_origin_y_m(-0.08);

        wavelength_m_ = kWavelength;
        geometry_id_ = "test_geometry_v1";
        calibration_id_ = "test_calibration_v1";

        // Create a simple operator with random K-vectors
        create_test_operator(64, 42);
    }

    void create_test_operator(size_t num_samples, unsigned seed) {
        std::mt19937 rng(seed);
        // K-vector range [-100, 100] rad/m is chosen to be well-sampled by
        // the 16x16 grid with 0.01m spacing (Nyquist ~314 rad/m)
        std::uniform_real_distribution<double> k_dist(-100.0, 100.0);
        std::uniform_real_distribution<double> weight_dist(0.5, 1.0);

        k_vectors_.clear();
        sample_weights_.clear();
        weights_.resize(static_cast<Eigen::Index>(num_samples));

        for (size_t i = 0; i < num_samples; ++i) {
            k_vectors_.emplace_back(k_dist(rng), k_dist(rng));
            sample_weights_.push_back(weight_dist(rng));
            weights_(static_cast<Eigen::Index>(i)) = weight_dist(rng);
        }

        auto slice = create_slice(k_vectors_, sample_weights_);
        Eigen::Vector2i grid_size(
            static_cast<int>(grid_.size_x()),
            static_cast<int>(grid_.size_y()));
        Eigen::Vector2d grid_spacing(grid_.spacing_x_m(), grid_.spacing_y_m());
        Eigen::Vector2d grid_origin(grid_.origin_x_m(), grid_.origin_y_m());

        aurora::forward::BandwidthConfig bw_config;
        bw_config.set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
        bw_config.set_quadrature_order(4);

        op_ = std::make_unique<DirectOperator>(
            slice, *forward_model_,
            bw_config,
            grid_size, grid_spacing, grid_origin);
    }

    aurora::forward::DetectorGeometry detector_geom_;
    aurora::forward::OpticsConfig optics_;
    std::unique_ptr<aurora::forward::ForwardModel> forward_model_;

    GridSpec grid_;
    double wavelength_m_;
    std::string geometry_id_;
    std::string calibration_id_;
    std::vector<Eigen::Vector2d> k_vectors_;
    std::vector<double> sample_weights_;
    Eigen::VectorXd weights_;
    std::unique_ptr<DirectOperator> op_;
};

// =============================================================================
// PSF Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, Precomputation_PsfPeakAtCenter) {
    auto psf = PcgPrecomputation::compute_psf(*op_, weights_, grid_);

    int nx = static_cast<int>(grid_.size_x());
    int ny = static_cast<int>(grid_.size_y());

    // Find maximum magnitude
    double max_mag = 0.0;
    int max_i = 0, max_j = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double mag = std::abs(psf(j, i));
            if (mag > max_mag) {
                max_mag = mag;
                max_i = i;
                max_j = j;
            }
        }
    }

    // Maximum should be at center where impulse was placed
    EXPECT_EQ(max_i, nx / 2);
    EXPECT_EQ(max_j, ny / 2);
}

TEST_F(PcgPrecomputationTest, Precomputation_PsfSymmetric) {
    // For a real-valued operator (symmetric K-vectors with conjugate pairs),
    // the PSF should be conjugate-symmetric
    // Here we just verify the PSF is computed without error and has
    // reasonable properties

    auto psf = PcgPrecomputation::compute_psf(*op_, weights_, grid_);

    int nx = static_cast<int>(grid_.size_x());
    int ny = static_cast<int>(grid_.size_y());

    EXPECT_EQ(psf.rows(), ny);
    EXPECT_EQ(psf.cols(), nx);

    // PSF should have non-zero values
    double total_energy = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            total_energy += std::norm(psf(j, i));
        }
    }
    EXPECT_GT(total_energy, 0.0);
}

TEST_F(PcgPrecomputationTest, Precomputation_PsfIsHermitian) {
    // Create operator with conjugate-symmetric K-vectors for Hermitian PSF
    std::vector<Eigen::Vector2d> sym_k_vectors;
    std::vector<double> sym_weights;

    // Add symmetric pairs: k and -k
    for (int i = 0; i < 16; ++i) {
        double kx = (i - 8) * 10.0;
        double ky = (i - 8) * 5.0;
        sym_k_vectors.emplace_back(kx, ky);
        sym_k_vectors.emplace_back(-kx, -ky);
        sym_weights.push_back(1.0);
        sym_weights.push_back(1.0);
    }

    auto sym_slice = create_slice(sym_k_vectors, sym_weights);
    Eigen::Vector2i grid_size(
        static_cast<int>(grid_.size_x()),
        static_cast<int>(grid_.size_y()));
    Eigen::Vector2d grid_spacing(grid_.spacing_x_m(), grid_.spacing_y_m());
    Eigen::Vector2d grid_origin(grid_.origin_x_m(), grid_.origin_y_m());

    aurora::forward::BandwidthConfig bw_config;
    bw_config.set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    bw_config.set_quadrature_order(4);

    DirectOperator sym_op(sym_slice, *forward_model_,
                          bw_config,
                          grid_size, grid_spacing, grid_origin);
    Eigen::VectorXd unit_weights = Eigen::VectorXd::Ones(
        static_cast<Eigen::Index>(sym_k_vectors.size()));

    auto psf = PcgPrecomputation::compute_psf(sym_op, unit_weights, grid_);

    int nx = static_cast<int>(grid_.size_x());
    int ny = static_cast<int>(grid_.size_y());

    // DC component should be real for Hermitian arrays
    // Use relative tolerance for large values
    double dc_mag = std::abs(psf(0, 0));
    EXPECT_LT(std::abs(psf(0, 0).imag()), dc_mag * 1e-6 + 1e-6);

    // Check Hermitian symmetry: PSF(i,j) = conj(PSF(N-i, N-j))
    // With quadrature integration, small numerical asymmetries are expected
    for (int j = 1; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            auto val = psf(j, i);
            auto conj_val = std::conj(psf(ny - j, nx - i));
            double mag = std::max(std::abs(val), std::abs(conj_val));
            double tol = mag * 1e-6 + 1e-6;  // Relative + absolute tolerance
            EXPECT_NEAR(val.real(), conj_val.real(), tol)
                << "Hermitian mismatch at (" << i << ", " << j << ")";
            EXPECT_NEAR(val.imag(), conj_val.imag(), tol)
                << "Hermitian mismatch at (" << i << ", " << j << ")";
        }
    }
}

// =============================================================================
// Transfer Function Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, Precomputation_TransferMatchesPsf) {
    auto psf = PcgPrecomputation::compute_psf(*op_, weights_, grid_);
    auto transfer = PcgPrecomputation::compute_transfer_function(psf);

    int nx = static_cast<int>(grid_.size_x());
    int ny = static_cast<int>(grid_.size_y());

    // IFFT of transfer function should match PSF
    aurora::forward::Fft2d ifft(nx, ny, true);  // inverse FFT
    auto recovered = ifft.execute(transfer);

    // FFT scaling: need to divide by N
    double scale = 1.0 / (nx * ny);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            auto expected = psf(j, i);
            auto actual = recovered(j, i) * scale;
            double mag = std::max(std::abs(expected), std::abs(actual));
            double tol = mag * 1e-6 + 1e-6;  // Relative + absolute tolerance
            EXPECT_NEAR(actual.real(), expected.real(), tol)
                << "IFFT{Ψ} != PSF at (" << i << ", " << j << ")";
            EXPECT_NEAR(actual.imag(), expected.imag(), tol)
                << "IFFT{Ψ} != PSF at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(PcgPrecomputationTest, Precomputation_TransferDCMatchesPsfSum) {
    auto psf = PcgPrecomputation::compute_psf(*op_, weights_, grid_);
    auto transfer = PcgPrecomputation::compute_transfer_function(psf);

    // DC component Ψ(0,0) should equal sum of PSF
    std::complex<double> psf_sum{0.0, 0.0};
    for (int j = 0; j < psf.rows(); ++j) {
        for (int i = 0; i < psf.cols(); ++i) {
            psf_sum += psf(j, i);
        }
    }

    double mag = std::max(std::abs(transfer(0, 0)), std::abs(psf_sum));
    double tol = mag * 1e-6 + 1e-6;  // Relative + absolute tolerance
    EXPECT_NEAR(transfer(0, 0).real(), psf_sum.real(), tol);
    EXPECT_NEAR(transfer(0, 0).imag(), psf_sum.imag(), tol);
}

// =============================================================================
// Diagonal Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, Precomputation_DiagonalAllPositive) {
    auto diagonal = PcgPrecomputation::compute_diagonal_approximate(
        *op_, weights_, grid_);

    int nx = static_cast<int>(grid_.size_x());
    int ny = static_cast<int>(grid_.size_y());

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            EXPECT_GT(diagonal(j, i), 0.0)
                << "Diagonal not positive at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(PcgPrecomputationTest, Precomputation_DiagonalApproxReasonable) {
    auto diagonal = PcgPrecomputation::compute_diagonal_approximate(
        *op_, weights_, grid_);

    // Check that diagonal values are in a sane range
    // For random K-vectors, diagonal should be positive and bounded
    double max_diag = diagonal.maxCoeff();
    double min_diag = diagonal.minCoeff();

    // Sanity checks that do not assume specific conditioning
    EXPECT_GT(min_diag, 0.0) << "Minimum diagonal entry should be positive";
    EXPECT_GE(max_diag, min_diag)
        << "Maximum diagonal entry should be at least the minimum";
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, Precomputation_ValidationRejectsWrongGeometryId) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_TIKHONOV);
    config.set_preconditioner_type(PRECONDITIONER_JACOBI);

    EXPECT_THROW(
        PcgPrecomputation::validate(
            data, "different_geometry", calibration_id_,
            grid_, wavelength_m_, config),
        std::invalid_argument);
}

TEST_F(PcgPrecomputationTest, Precomputation_ValidationRejectsWrongCalibration) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_TIKHONOV);
    config.set_preconditioner_type(PRECONDITIONER_JACOBI);

    EXPECT_THROW(
        PcgPrecomputation::validate(
            data, geometry_id_, "different_calibration",
            grid_, wavelength_m_, config),
        std::invalid_argument);
}

TEST_F(PcgPrecomputationTest, Precomputation_ValidationRejectsWrongGrid) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    GridSpec wrong_grid = grid_;
    wrong_grid.set_size_x(32);  // Different size

    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_TIKHONOV);
    config.set_preconditioner_type(PRECONDITIONER_JACOBI);

    EXPECT_THROW(
        PcgPrecomputation::validate(
            data, geometry_id_, calibration_id_,
            wrong_grid, wavelength_m_, config),
        std::invalid_argument);
}

TEST_F(PcgPrecomputationTest, Precomputation_ValidationRejectsWrongWavelength) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_TIKHONOV);
    config.set_preconditioner_type(PRECONDITIONER_JACOBI);

    EXPECT_THROW(
        PcgPrecomputation::validate(
            data, geometry_id_, calibration_id_,
            grid_, 905e-9, config),  // Different wavelength
        std::invalid_argument);
}

TEST_F(PcgPrecomputationTest, Precomputation_ValidationRejectsWrongRegularizer) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_GRADIENT);  // Different
    config.set_preconditioner_type(PRECONDITIONER_JACOBI);

    EXPECT_THROW(
        PcgPrecomputation::validate(
            data, geometry_id_, calibration_id_,
            grid_, wavelength_m_, config),
        std::invalid_argument);
}

TEST_F(PcgPrecomputationTest, Precomputation_ValidationRejectsWrongPreconditioner) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_TIKHONOV);
    config.set_preconditioner_type(PRECONDITIONER_HYBRID);  // Different

    EXPECT_THROW(
        PcgPrecomputation::validate(
            data, geometry_id_, calibration_id_,
            grid_, wavelength_m_, config),
        std::invalid_argument);
}

TEST_F(PcgPrecomputationTest, Precomputation_ValidationRejectsCorruptedData) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    // Corrupt the PSF array
    data.mutable_psf()->RemoveLast();

    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_TIKHONOV);
    config.set_preconditioner_type(PRECONDITIONER_JACOBI);

    EXPECT_THROW(
        PcgPrecomputation::validate(
            data, geometry_id_, calibration_id_,
            grid_, wavelength_m_, config),
        std::invalid_argument);
}

TEST_F(PcgPrecomputationTest, Precomputation_ValidationAcceptsValid) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_TIKHONOV);
    config.set_preconditioner_type(PRECONDITIONER_JACOBI);

    // Should not throw
    EXPECT_NO_THROW(
        PcgPrecomputation::validate(
            data, geometry_id_, calibration_id_,
            grid_, wavelength_m_, config));
}

// =============================================================================
// I/O Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, Precomputation_SaveLoadRoundtrip) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_GRADIENT, PRECONDITIONER_HYBRID,
        geometry_id_, calibration_id_);

    PcgPrecomputedDataSet data_set;
    *data_set.add_wavelength_data() = data;
    data_set.set_geometry_id(geometry_id_);
    data_set.set_calibration_id(calibration_id_);
    *data_set.mutable_grid() = grid_;

    // Use RAII for cleanup
    TempFile temp_file("test_pcg_precomputation.pb");

    // Save
    PcgPrecomputation::save(data_set, temp_file.path());

    // Load
    auto loaded = PcgPrecomputation::load(temp_file.path());

    // Verify
    EXPECT_EQ(loaded.wavelength_data_size(), 1);
    EXPECT_EQ(loaded.geometry_id(), geometry_id_);
    EXPECT_EQ(loaded.calibration_id(), calibration_id_);
    EXPECT_EQ(loaded.grid().size_x(), grid_.size_x());
    EXPECT_EQ(loaded.grid().size_y(), grid_.size_y());

    const auto& loaded_data = loaded.wavelength_data(0);
    EXPECT_EQ(loaded_data.psf_size(), data.psf_size());
    EXPECT_EQ(loaded_data.transfer_function_size(), data.transfer_function_size());
    EXPECT_EQ(loaded_data.regularizer_spectrum_size(), data.regularizer_spectrum_size());
    EXPECT_EQ(loaded_data.operator_diagonal_size(), data.operator_diagonal_size());

    // Check values match
    for (int i = 0; i < data.psf_size(); ++i) {
        EXPECT_DOUBLE_EQ(loaded_data.psf(i), data.psf(i));
    }
}

TEST_F(PcgPrecomputationTest, Precomputation_LoadRejectsCorruptedFile) {
    // Use RAII for cleanup
    TempFile temp_file("test_corrupted.pb");

    // Write garbage
    std::ofstream file(temp_file.path(), std::ios::binary);
    file << "not a valid protobuf";
    file.close();

    EXPECT_THROW(PcgPrecomputation::load(temp_file.path()), std::runtime_error);
}

TEST_F(PcgPrecomputationTest, Precomputation_LoadRejectsMissingFile) {
    EXPECT_THROW(PcgPrecomputation::load("/nonexistent/path.pb"),
                 std::runtime_error);
}

// =============================================================================
// Multi-wavelength Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, Precomputation_MultiWavelength) {
    double wavelength1 = 905e-9;
    double wavelength2 = 1550e-9;

    auto data1 = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength1,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    auto data2 = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength2,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    // Both should be valid
    PcgConfig config;
    config.set_regularizer_type(REGULARIZER_TIKHONOV);
    config.set_preconditioner_type(PRECONDITIONER_JACOBI);

    EXPECT_NO_THROW(
        PcgPrecomputation::validate(
            data1, geometry_id_, calibration_id_,
            grid_, wavelength1, config));
    EXPECT_NO_THROW(
        PcgPrecomputation::validate(
            data2, geometry_id_, calibration_id_,
            grid_, wavelength2, config));

    // PSFs should be the same (K-vectors don't depend on wavelength in test)
    // but different wavelengths should be stored correctly
    EXPECT_NE(data1.inputs().wavelength_m(), data2.inputs().wavelength_m());
}

// =============================================================================
// Array Packing/Unpacking Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, PackUnpackComplex_Roundtrip) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    Eigen::MatrixXcd original(8, 12);
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 12; ++i) {
            original(j, i) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    google::protobuf::RepeatedField<double> packed;
    PcgPrecomputation::pack_complex_array(original, &packed);

    auto unpacked = PcgPrecomputation::unpack_complex_array(packed, 8, 12);

    EXPECT_EQ(unpacked.rows(), original.rows());
    EXPECT_EQ(unpacked.cols(), original.cols());

    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 12; ++i) {
            EXPECT_DOUBLE_EQ(unpacked(j, i).real(), original(j, i).real());
            EXPECT_DOUBLE_EQ(unpacked(j, i).imag(), original(j, i).imag());
        }
    }
}

TEST_F(PcgPrecomputationTest, PackUnpackReal_Roundtrip) {
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    Eigen::MatrixXd original(6, 10);
    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 10; ++i) {
            original(j, i) = dist(rng);
        }
    }

    google::protobuf::RepeatedField<double> packed;
    PcgPrecomputation::pack_real_array(original, &packed);

    auto unpacked = PcgPrecomputation::unpack_real_array(packed, 6, 10);

    EXPECT_EQ(unpacked.rows(), original.rows());
    EXPECT_EQ(unpacked.cols(), original.cols());

    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 10; ++i) {
            EXPECT_DOUBLE_EQ(unpacked(j, i), original(j, i));
        }
    }
}

TEST_F(PcgPrecomputationTest, UnpackComplex_RejectsInvalidDimensions) {
    google::protobuf::RepeatedField<double> data;
    data.Resize(24, 0.0);

    EXPECT_THROW(PcgPrecomputation::unpack_complex_array(data, 0, 12),
                 std::invalid_argument);
    EXPECT_THROW(PcgPrecomputation::unpack_complex_array(data, 4, -1),
                 std::invalid_argument);
}

TEST_F(PcgPrecomputationTest, UnpackReal_RejectsInvalidDimensions) {
    google::protobuf::RepeatedField<double> data;
    data.Resize(12, 0.0);

    EXPECT_THROW(PcgPrecomputation::unpack_real_array(data, 0, 12),
                 std::invalid_argument);
    EXPECT_THROW(PcgPrecomputation::unpack_real_array(data, 4, -1),
                 std::invalid_argument);
}

// =============================================================================
// Regularizer Spectrum Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, RegularizerSpectrum_Tikhonov) {
    auto spectrum = PcgPrecomputation::compute_regularizer_spectrum(
        grid_, REGULARIZER_TIKHONOV);

    int nx = static_cast<int>(grid_.size_x());
    int ny = static_cast<int>(grid_.size_y());

    // Tikhonov spectrum should be all ones
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            EXPECT_DOUBLE_EQ(spectrum(j, i), 1.0);
        }
    }
}

TEST_F(PcgPrecomputationTest, RegularizerSpectrum_Gradient) {
    auto spectrum = PcgPrecomputation::compute_regularizer_spectrum(
        grid_, REGULARIZER_GRADIENT);

    // DC should be zero
    EXPECT_NEAR(spectrum(0, 0), 0.0, 1e-10);

    // Should be non-negative everywhere
    for (int j = 0; j < spectrum.rows(); ++j) {
        for (int i = 0; i < spectrum.cols(); ++i) {
            EXPECT_GE(spectrum(j, i), 0.0);
        }
    }
}

TEST_F(PcgPrecomputationTest, RegularizerSpectrum_Laplacian) {
    auto spectrum = PcgPrecomputation::compute_regularizer_spectrum(
        grid_, REGULARIZER_LAPLACIAN);

    // DC should be zero
    EXPECT_NEAR(spectrum(0, 0), 0.0, 1e-10);

    // Should be non-negative everywhere
    for (int j = 0; j < spectrum.rows(); ++j) {
        for (int i = 0; i < spectrum.cols(); ++i) {
            EXPECT_GE(spectrum(j, i), 0.0);
        }
    }
}

// =============================================================================
// Compute Function Tests
// =============================================================================

TEST_F(PcgPrecomputationTest, Compute_SetsAllFields) {
    auto data = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_GRADIENT, PRECONDITIONER_HYBRID,
        geometry_id_, calibration_id_);

    // Check inputs
    EXPECT_EQ(data.inputs().geometry_id(), geometry_id_);
    EXPECT_EQ(data.inputs().calibration_id(), calibration_id_);
    EXPECT_EQ(data.inputs().grid().size_x(), grid_.size_x());
    EXPECT_EQ(data.inputs().grid().size_y(), grid_.size_y());
    EXPECT_DOUBLE_EQ(data.inputs().wavelength_m(), wavelength_m_);
    EXPECT_EQ(data.inputs().regularizer_type(), REGULARIZER_GRADIENT);
    EXPECT_EQ(data.inputs().preconditioner_type(), PRECONDITIONER_HYBRID);

    // Check arrays are populated
    size_t grid_size = grid_.size_x() * grid_.size_y();
    EXPECT_EQ(static_cast<size_t>(data.psf_size()), 2 * grid_size);
    EXPECT_EQ(static_cast<size_t>(data.transfer_function_size()), 2 * grid_size);
    EXPECT_EQ(static_cast<size_t>(data.regularizer_spectrum_size()), grid_size);
    EXPECT_EQ(static_cast<size_t>(data.operator_diagonal_size()), grid_size);

    // Check metadata
    EXPECT_GT(data.generation_timestamp_s(), 0.0);
    EXPECT_FALSE(data.generator_version().empty());
    EXPECT_GT(data.computation_time_s(), 0.0);
}

TEST_F(PcgPrecomputationTest, Compute_DifferentRegularizers) {
    auto data_tik = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_TIKHONOV, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    auto data_grad = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_GRADIENT, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    auto data_lap = PcgPrecomputation::compute(
        *op_, weights_, grid_, wavelength_m_,
        REGULARIZER_LAPLACIAN, PRECONDITIONER_JACOBI,
        geometry_id_, calibration_id_);

    // PSF should be the same (independent of regularizer)
    EXPECT_EQ(data_tik.psf_size(), data_grad.psf_size());
    EXPECT_EQ(data_tik.psf_size(), data_lap.psf_size());
    for (int i = 0; i < data_tik.psf_size(); ++i) {
        EXPECT_DOUBLE_EQ(data_tik.psf(i), data_grad.psf(i));
        EXPECT_DOUBLE_EQ(data_tik.psf(i), data_lap.psf(i));
    }

    // Regularizer spectrum should differ
    // Tikhonov should be all 1.0, others should have 0 at DC
    auto spectrum_tik = PcgPrecomputation::unpack_real_array(
        data_tik.regularizer_spectrum(),
        static_cast<int>(grid_.size_y()),
        static_cast<int>(grid_.size_x()));
    auto spectrum_grad = PcgPrecomputation::unpack_real_array(
        data_grad.regularizer_spectrum(),
        static_cast<int>(grid_.size_y()),
        static_cast<int>(grid_.size_x()));

    EXPECT_DOUBLE_EQ(spectrum_tik(0, 0), 1.0);
    EXPECT_NEAR(spectrum_grad(0, 0), 0.0, 1e-10);
}

}  // namespace aurora::reconstruction
