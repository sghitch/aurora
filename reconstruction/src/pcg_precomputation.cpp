#include "reconstruction/pcg_precomputation.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "forward/fft2d.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aurora::reconstruction {

namespace {

// Version string for tracking generator code version
constexpr const char* AURORA_VERSION = "0.1.0";

}  // namespace

PcgPrecomputedData PcgPrecomputation::compute(
    const ReconstructionOperator& op,
    const Eigen::VectorXd& weights,
    const GridSpec& grid,
    double wavelength_m,
    RegularizerType regularizer_type,
    PreconditionerType preconditioner_type,
    const std::string& geometry_id,
    const std::string& calibration_id) {

    auto start = std::chrono::steady_clock::now();

    // Compute PSF and transfer function
    auto psf = compute_psf(op, weights, grid);
    auto transfer = compute_transfer_function(psf);

    // Compute regularizer spectrum
    auto reg_spectrum = compute_regularizer_spectrum(grid, regularizer_type);

    // Compute diagonal approximation
    auto diagonal = compute_diagonal_approximate(op, weights, grid);

    auto end = std::chrono::steady_clock::now();
    double compute_time = std::chrono::duration<double>(end - start).count();

    // Pack into protobuf
    PcgPrecomputedData result;

    // Set inputs
    auto* inputs = result.mutable_inputs();
    inputs->set_geometry_id(geometry_id);
    inputs->set_calibration_id(calibration_id);
    *inputs->mutable_grid() = grid;
    inputs->set_wavelength_m(wavelength_m);
    inputs->set_regularizer_type(regularizer_type);
    inputs->set_preconditioner_type(preconditioner_type);

    // Pack arrays
    pack_complex_array(psf, result.mutable_psf());
    pack_complex_array(transfer, result.mutable_transfer_function());
    pack_real_array(reg_spectrum, result.mutable_regularizer_spectrum());
    pack_real_array(diagonal, result.mutable_operator_diagonal());

    // Metadata
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    double timestamp = std::chrono::duration<double>(epoch).count();
    result.set_generation_timestamp_s(timestamp);
    result.set_generator_version(AURORA_VERSION);
    result.set_computation_time_s(compute_time);

    return result;
}

void PcgPrecomputation::validate(
    const PcgPrecomputedData& data,
    const std::string& geometry_id,
    const std::string& calibration_id,
    const GridSpec& grid,
    double wavelength_m,
    const PcgConfig& config) {

    const auto& inputs = data.inputs();

    // Check identifiers
    if (inputs.geometry_id() != geometry_id) {
        throw std::invalid_argument("geometry_id mismatch");
    }
    if (inputs.calibration_id() != calibration_id) {
        throw std::invalid_argument("calibration_id mismatch");
    }

    // Check grid
    if (inputs.grid().size_x() != grid.size_x() ||
        inputs.grid().size_y() != grid.size_y()) {
        throw std::invalid_argument("grid size mismatch");
    }
    if (std::abs(inputs.grid().spacing_x_m() - grid.spacing_x_m()) > 1e-9 ||
        std::abs(inputs.grid().spacing_y_m() - grid.spacing_y_m()) > 1e-9) {
        throw std::invalid_argument("grid spacing mismatch");
    }
    if (std::abs(inputs.grid().origin_x_m() - grid.origin_x_m()) > 1e-9 ||
        std::abs(inputs.grid().origin_y_m() - grid.origin_y_m()) > 1e-9) {
        throw std::invalid_argument("grid origin mismatch");
    }

    // Check wavelength
    if (std::abs(inputs.wavelength_m() - wavelength_m) > 1e-12) {
        throw std::invalid_argument("wavelength mismatch");
    }

    // Check types
    if (inputs.regularizer_type() != config.regularizer_type()) {
        throw std::invalid_argument("regularizer_type mismatch");
    }
    if (inputs.preconditioner_type() != config.preconditioner_type()) {
        throw std::invalid_argument("preconditioner_type mismatch");
    }

    // Check array sizes
    size_t grid_size = static_cast<size_t>(grid.size_x()) * grid.size_y();
    size_t expected_complex = 2 * grid_size;
    size_t expected_real = grid_size;

    if (static_cast<size_t>(data.psf_size()) != expected_complex) {
        throw std::invalid_argument("psf array size corrupted");
    }
    if (static_cast<size_t>(data.transfer_function_size()) != expected_complex) {
        throw std::invalid_argument("transfer_function array size corrupted");
    }
    if (static_cast<size_t>(data.regularizer_spectrum_size()) != expected_real) {
        throw std::invalid_argument("regularizer_spectrum array size corrupted");
    }
    if (static_cast<size_t>(data.operator_diagonal_size()) != expected_real) {
        throw std::invalid_argument("operator_diagonal array size corrupted");
    }
}

PcgPrecomputedDataSet PcgPrecomputation::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    PcgPrecomputedDataSet data;
    if (!data.ParseFromIstream(&file)) {
        file.close();
        throw std::runtime_error("Failed to parse protobuf from file: " + path);
    }

    file.close();
    return data;
}

void PcgPrecomputation::save(const PcgPrecomputedDataSet& data,
                              const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    if (!data.SerializeToOstream(&file)) {
        file.close();
        throw std::runtime_error("Failed to serialize protobuf to file: " + path);
    }

    file.close();
    if (!file.good()) {
        throw std::runtime_error("Failed to flush/close file: " + path);
    }
}

Eigen::MatrixXcd PcgPrecomputation::compute_psf(
    const ReconstructionOperator& op,
    const Eigen::VectorXd& weights,
    const GridSpec& grid) {

    int nx = static_cast<int>(grid.size_x());
    int ny = static_cast<int>(grid.size_y());

    Eigen::Vector2d origin(grid.origin_x_m(), grid.origin_y_m());
    Eigen::Vector2d spacing(grid.spacing_x_m(), grid.spacing_y_m());
    Eigen::Vector2i size(nx, ny);

    // Create impulse at center
    Eigen::MatrixXcd delta = Eigen::MatrixXcd::Zero(ny, nx);
    delta(ny / 2, nx / 2) = 1.0;

    // Forward: δ → y = Aδ
    auto y = op.forward(delta, origin, spacing);

    // Weight: y → Wy
    // Empty weights means uniform weighting (all weights = 1)
    if (weights.size() > 0) {
        if (weights.size() != y.size()) {
            throw std::invalid_argument(
                "Weights size mismatch: expected " + std::to_string(y.size()) +
                ", got " + std::to_string(weights.size()));
        }
        y = (weights.array() * y.array()).matrix();
    }

    // Adjoint: Wy → A†Wy = PSF
    return op.adjoint(y, origin, spacing, size);
}

Eigen::MatrixXcd PcgPrecomputation::compute_transfer_function(
    const Eigen::MatrixXcd& psf) {

    int nx = static_cast<int>(psf.cols());
    int ny = static_cast<int>(psf.rows());

    aurora::forward::Fft2d fft(nx, ny, false);  // forward FFT
    return fft.execute(psf);
}

Eigen::MatrixXd PcgPrecomputation::compute_regularizer_spectrum(
    const GridSpec& grid,
    RegularizerType type) {

    auto reg_variant = create_regularizer(type);
    const Regularizer& reg = get_regularizer(reg_variant);

    Eigen::Vector2i size(static_cast<int>(grid.size_x()),
                         static_cast<int>(grid.size_y()));
    Eigen::Vector2d spacing(grid.spacing_x_m(), grid.spacing_y_m());

    return reg.frequency_response(size, spacing);
}

Eigen::MatrixXd PcgPrecomputation::compute_diagonal_approximate(
    const ReconstructionOperator& op,
    const Eigen::VectorXd& weights,
    const GridSpec& grid) {

    int nx = static_cast<int>(grid.size_x());
    int ny = static_cast<int>(grid.size_y());

    Eigen::Vector2d origin(grid.origin_x_m(), grid.origin_y_m());
    Eigen::Vector2d spacing(grid.spacing_x_m(), grid.spacing_y_m());
    Eigen::Vector2i size(nx, ny);

    // Apply A†WA to ones vector as proxy for diagonal energy
    Eigen::MatrixXcd ones = Eigen::MatrixXcd::Ones(ny, nx);

    auto y = op.forward(ones, origin, spacing);

    // Weight: y → Wy
    // Empty weights means uniform weighting (all weights = 1)
    if (weights.size() > 0) {
        if (weights.size() != y.size()) {
            throw std::invalid_argument(
                "Weights size mismatch: expected " + std::to_string(y.size()) +
                ", got " + std::to_string(weights.size()));
        }
        y = (weights.array() * y.array()).matrix();
    }

    auto result = op.adjoint(y, origin, spacing, size);

    // Take magnitude as diagonal approximation
    return result.cwiseAbs();
}

void PcgPrecomputation::pack_complex_array(
    const Eigen::MatrixXcd& matrix,
    google::protobuf::RepeatedField<double>* output) {

    int64_t ny = matrix.rows();
    int64_t nx = matrix.cols();
    int64_t total_size = 2LL * ny * nx;
    output->Resize(static_cast<int>(total_size), 0.0);

    int idx = 0;
    for (int64_t j = 0; j < ny; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            (*output)[idx++] = matrix(j, i).real();
            (*output)[idx++] = matrix(j, i).imag();
        }
    }
}

Eigen::MatrixXcd PcgPrecomputation::unpack_complex_array(
    const google::protobuf::RepeatedField<double>& input,
    int rows, int cols) {

    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Invalid dimensions: rows and cols must be positive");
    }

    int64_t expected_size = 2LL * rows * cols;
    if (input.size() != expected_size) {
        throw std::invalid_argument(
            "Array size mismatch: expected " + std::to_string(expected_size) +
            ", got " + std::to_string(input.size()));
    }

    Eigen::MatrixXcd result(rows, cols);
    int idx = 0;
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            result(j, i) = std::complex<double>(input[idx], input[idx + 1]);
            idx += 2;
        }
    }
    return result;
}

void PcgPrecomputation::pack_real_array(
    const Eigen::MatrixXd& matrix,
    google::protobuf::RepeatedField<double>* output) {

    int64_t ny = matrix.rows();
    int64_t nx = matrix.cols();
    int64_t total_size = ny * nx;
    output->Resize(static_cast<int>(total_size), 0.0);

    int idx = 0;
    for (int64_t j = 0; j < ny; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            (*output)[idx++] = matrix(j, i);
        }
    }
}

Eigen::MatrixXd PcgPrecomputation::unpack_real_array(
    const google::protobuf::RepeatedField<double>& input,
    int rows, int cols) {

    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Invalid dimensions: rows and cols must be positive");
    }

    int64_t expected_size = static_cast<int64_t>(rows) * cols;
    if (input.size() != expected_size) {
        throw std::invalid_argument(
            "Array size mismatch: expected " + std::to_string(expected_size) +
            ", got " + std::to_string(input.size()));
    }

    Eigen::MatrixXd result(rows, cols);
    int idx = 0;
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            result(j, i) = input[idx++];
        }
    }
    return result;
}

}  // namespace aurora::reconstruction
