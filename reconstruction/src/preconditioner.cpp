#include "reconstruction/preconditioner.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <kiss_fft.h>

namespace aurora::reconstruction {

// =============================================================================
// JacobiPreconditioner
// =============================================================================

JacobiPreconditioner::JacobiPreconditioner(const std::vector<double>& diagonal,
                                           int rows,
                                           int cols)
    : inv_diagonal_() {

    // Validate dimensions before allocating
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "JacobiPreconditioner: rows and cols must be positive");
    }

    if (static_cast<int>(diagonal.size()) != rows * cols) {
        throw std::invalid_argument(
            "JacobiPreconditioner: diagonal size mismatch with grid dimensions");
    }

    inv_diagonal_.resize(rows, cols);

    // Find maximum for clamping threshold
    double max_val = 0.0;
    for (double d : diagonal) {
        max_val = std::max(max_val, std::abs(d));
    }

    // Clamp minimum to avoid division by near-zero
    const double min_val = max_val * 1e-10;
    const double clamp_val = (max_val > 0.0) ? min_val : 1.0;

    // Compute inverse diagonal
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            double d = diagonal[j * cols + i];
            inv_diagonal_(j, i) = 1.0 / std::max(d, clamp_val);
        }
    }
}

JacobiPreconditioner::JacobiPreconditioner(const std::vector<double>& operator_diagonal,
                                           const std::vector<double>& regularizer_spectrum,
                                           double lambda,
                                           int rows,
                                           int cols)
    : inv_diagonal_() {

    // Validate dimensions before allocating
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "JacobiPreconditioner: rows and cols must be positive");
    }

    const int total_size = rows * cols;

    if (static_cast<int>(operator_diagonal.size()) != total_size) {
        throw std::invalid_argument(
            "JacobiPreconditioner: operator_diagonal size mismatch");
    }
    if (static_cast<int>(regularizer_spectrum.size()) != total_size) {
        throw std::invalid_argument(
            "JacobiPreconditioner: regularizer_spectrum size mismatch");
    }

    inv_diagonal_.resize(rows, cols);

    // Compute combined diagonal: diag(A†WA) + λ·diag(L†L)
    // Note: For Jacobi, we use the regularizer spectrum evaluated at DC (which is
    // approximately uniform for Tikhonov, but varies for gradient/Laplacian).
    // In practice, the operator_diagonal already contains spatial weighting.
    std::vector<double> combined(rows * cols);
    double max_val = 0.0;
    for (int idx = 0; idx < rows * cols; ++idx) {
        combined[idx] = operator_diagonal[idx] + lambda * regularizer_spectrum[idx];
        max_val = std::max(max_val, std::abs(combined[idx]));
    }

    // Clamp minimum to avoid division by near-zero
    const double min_val = max_val * 1e-10;
    const double clamp_val = (max_val > 0.0) ? min_val : 1.0;

    // Compute inverse diagonal
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            double d = combined[j * cols + i];
            inv_diagonal_(j, i) = 1.0 / std::max(d, clamp_val);
        }
    }
}

Eigen::MatrixXcd JacobiPreconditioner::apply(const Eigen::MatrixXcd& r) const {
    // Validate dimensions
    if (r.rows() != inv_diagonal_.rows() || r.cols() != inv_diagonal_.cols()) {
        throw std::invalid_argument(
            "JacobiPreconditioner::apply: input matrix dimension mismatch");
    }

    // Element-wise multiplication with inverse diagonal
    Eigen::MatrixXcd result(r.rows(), r.cols());
    for (int j = 0; j < r.rows(); ++j) {
        for (int i = 0; i < r.cols(); ++i) {
            result(j, i) = r(j, i) * inv_diagonal_(j, i);
        }
    }
    return result;
}

// =============================================================================
// CirculantPreconditioner - Pimpl for FFT state
// =============================================================================

struct CirculantPreconditioner::Impl {
    kiss_fft_cfg fft_row_plan = nullptr;
    kiss_fft_cfg fft_col_plan = nullptr;
    kiss_fft_cfg ifft_row_plan = nullptr;
    kiss_fft_cfg ifft_col_plan = nullptr;

    // Temporary buffers for row/column transforms
    // Note: KissFFT does not support in-place transforms, so we need separate
    // input and output buffers for each transform
    mutable std::vector<kiss_fft_cpx> row_buffer;
    mutable std::vector<kiss_fft_cpx> col_buffer;
    mutable std::vector<kiss_fft_cpx> temp_buffer;  // Output buffer for transforms
    mutable std::vector<kiss_fft_cpx> work_buffer;

    ~Impl() {
        if (fft_row_plan) kiss_fft_free(fft_row_plan);
        if (fft_col_plan) kiss_fft_free(fft_col_plan);
        if (ifft_row_plan) kiss_fft_free(ifft_row_plan);
        if (ifft_col_plan) kiss_fft_free(ifft_col_plan);
    }
};

CirculantPreconditioner::CirculantPreconditioner(
    const std::vector<double>& transfer_function,
    const std::vector<double>& regularizer_spectrum,
    double lambda,
    double epsilon,
    int rows,
    int cols)
    : impl_(std::make_unique<Impl>()),
      inv_filter_(),
      rows_(rows),
      cols_(cols) {

    // Validate dimensions before allocating matrices or FFT plans
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "CirculantPreconditioner: rows and cols must be positive");
    }

    const int total_size = rows * cols;

    // Validate input sizes
    // transfer_function is complex interleaved: 2 * rows * cols elements
    if (static_cast<int>(transfer_function.size()) != 2 * total_size) {
        throw std::invalid_argument(
            "CirculantPreconditioner: transfer_function size mismatch "
            "(expected complex interleaved 2*Ny*Nx)");
    }
    if (static_cast<int>(regularizer_spectrum.size()) != total_size) {
        throw std::invalid_argument(
            "CirculantPreconditioner: regularizer_spectrum size mismatch");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument(
            "CirculantPreconditioner: epsilon must be > 0");
    }

    inv_filter_.resize(rows, cols);

    // Compute inverse filter: 1 / (Ψ + λΦ + ε)
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            int idx = j * cols + i;
            // Extract complex transfer function value
            double tf_re = transfer_function[2 * idx];
            double tf_im = transfer_function[2 * idx + 1];
            std::complex<double> psi(tf_re, tf_im);

            // Regularizer spectrum is real
            double phi = regularizer_spectrum[idx];

            // Denominator: Ψ + λΦ + ε
            std::complex<double> denom = psi + lambda * phi + epsilon;

            // Inverse with magnitude clamping to prevent blowup
            double mag = std::abs(denom);
            if (mag < epsilon) {
                // Clamp small denominators
                inv_filter_(j, i) = std::complex<double>(1.0 / epsilon, 0.0);
            } else {
                inv_filter_(j, i) = 1.0 / denom;
            }
        }
    }

    // Allocate FFT plans
    impl_->fft_row_plan = kiss_fft_alloc(cols, 0, nullptr, nullptr);
    impl_->fft_col_plan = kiss_fft_alloc(rows, 0, nullptr, nullptr);
    impl_->ifft_row_plan = kiss_fft_alloc(cols, 1, nullptr, nullptr);
    impl_->ifft_col_plan = kiss_fft_alloc(rows, 1, nullptr, nullptr);

    if (!impl_->fft_row_plan || !impl_->fft_col_plan ||
        !impl_->ifft_row_plan || !impl_->ifft_col_plan) {
        throw std::runtime_error("Failed to allocate KissFFT plans");
    }

    // Allocate work buffers
    impl_->row_buffer.resize(cols);
    impl_->col_buffer.resize(rows);
    impl_->temp_buffer.resize(std::max(rows, cols));  // For FFT output (non-in-place)
    impl_->work_buffer.resize(rows * cols);
}

CirculantPreconditioner::~CirculantPreconditioner() = default;

CirculantPreconditioner::CirculantPreconditioner(CirculantPreconditioner&&) noexcept = default;

CirculantPreconditioner& CirculantPreconditioner::operator=(CirculantPreconditioner&&) noexcept = default;

Eigen::MatrixXcd CirculantPreconditioner::apply(const Eigen::MatrixXcd& r) const {
    // P⁻¹r = IFFT(FFT(r) .* inv_filter)
    //
    // Separable 2D FFT: transform rows, then columns

    const int Ny = rows_;
    const int Nx = cols_;

    // Validate input dimensions
    if (r.rows() != Ny || r.cols() != Nx) {
        throw std::invalid_argument(
            "CirculantPreconditioner::apply: input matrix dimension mismatch");
    }

    // Step 1: Forward FFT of r
    // First transform rows
    for (int j = 0; j < Ny; ++j) {
        // Copy row to buffer
        for (int i = 0; i < Nx; ++i) {
            impl_->row_buffer[i].r = r(j, i).real();
            impl_->row_buffer[i].i = r(j, i).imag();
        }
        // FFT row (row_buffer -> work_buffer, KissFFT needs separate in/out)
        kiss_fft(impl_->fft_row_plan, impl_->row_buffer.data(),
                 impl_->work_buffer.data() + j * Nx);
    }

    // Then transform columns
    // Note: KissFFT does not support in-place transforms, use temp_buffer as output
    for (int i = 0; i < Nx; ++i) {
        // Copy column to buffer
        for (int j = 0; j < Ny; ++j) {
            impl_->col_buffer[j] = impl_->work_buffer[j * Nx + i];
        }
        // FFT column (col_buffer -> temp_buffer)
        kiss_fft(impl_->fft_col_plan, impl_->col_buffer.data(), impl_->temp_buffer.data());
        // Copy back from temp_buffer
        for (int j = 0; j < Ny; ++j) {
            impl_->work_buffer[j * Nx + i] = impl_->temp_buffer[j];
        }
    }

    // Step 2: Multiply by inverse filter in frequency domain
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            auto& w = impl_->work_buffer[j * Nx + i];
            std::complex<double> val(w.r, w.i);
            std::complex<double> filtered = val * inv_filter_(j, i);
            w.r = filtered.real();
            w.i = filtered.imag();
        }
    }

    // Step 3: Inverse FFT
    // First inverse transform columns (using temp_buffer as output)
    for (int i = 0; i < Nx; ++i) {
        // Copy column to buffer
        for (int j = 0; j < Ny; ++j) {
            impl_->col_buffer[j] = impl_->work_buffer[j * Nx + i];
        }
        // IFFT column (col_buffer -> temp_buffer)
        kiss_fft(impl_->ifft_col_plan, impl_->col_buffer.data(), impl_->temp_buffer.data());
        // Copy back from temp_buffer
        for (int j = 0; j < Ny; ++j) {
            impl_->work_buffer[j * Nx + i] = impl_->temp_buffer[j];
        }
    }

    // Then inverse transform rows and normalize
    Eigen::MatrixXcd result(Ny, Nx);
    const double norm = 1.0 / (Nx * Ny);

    for (int j = 0; j < Ny; ++j) {
        // IFFT row (work_buffer -> row_buffer)
        kiss_fft(impl_->ifft_row_plan,
                 impl_->work_buffer.data() + j * Nx,
                 impl_->row_buffer.data());
        // Copy to result with normalization
        for (int i = 0; i < Nx; ++i) {
            result(j, i) = std::complex<double>(
                impl_->row_buffer[i].r * norm,
                impl_->row_buffer[i].i * norm);
        }
    }

    return result;
}

// =============================================================================
// HybridPreconditioner
// =============================================================================

HybridPreconditioner::HybridPreconditioner(
    const std::vector<double>& operator_diagonal,
    const std::vector<double>& transfer_function,
    const std::vector<double>& regularizer_spectrum,
    double lambda,
    double epsilon,
    int rows,
    int cols)
    : inv_sqrt_diagonal_() {

    // Validate dimensions before allocating
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "HybridPreconditioner: rows and cols must be positive");
    }

    const int total_size = rows * cols;

    if (static_cast<int>(operator_diagonal.size()) != total_size) {
        throw std::invalid_argument(
            "HybridPreconditioner: operator_diagonal size mismatch");
    }

    if (static_cast<int>(regularizer_spectrum.size()) != total_size) {
        throw std::invalid_argument(
            "HybridPreconditioner: regularizer_spectrum size mismatch");
    }

    inv_sqrt_diagonal_.resize(rows, cols);

    // Compute D^{-1/2} from operator diagonal + lambda * regularizer
    // (The diagonal part of the full H matrix)
    double max_val = 0.0;
    for (int idx = 0; idx < total_size; ++idx) {
        double d = operator_diagonal[idx] + lambda * regularizer_spectrum[idx];
        max_val = std::max(max_val, std::abs(d));
    }

    const double min_val = max_val * 1e-10;
    const double clamp_val = (max_val > 0.0) ? min_val : 1.0;

    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            int idx = j * cols + i;
            double d = operator_diagonal[idx] + lambda * regularizer_spectrum[idx];
            inv_sqrt_diagonal_(j, i) = 1.0 / std::sqrt(std::max(d, clamp_val));
        }
    }

    // Create circulant preconditioner C for the unnormalized system H
    // The full hybrid preconditioner P⁻¹ = D^{-1/2} · C⁻¹ · D^{-1/2}
    // acts on the normalized system D^{-1/2} H D^{-1/2}, which is closer to identity
    circulant_ = std::make_unique<CirculantPreconditioner>(
        transfer_function, regularizer_spectrum, lambda, epsilon, rows, cols);
}

HybridPreconditioner::~HybridPreconditioner() = default;

HybridPreconditioner::HybridPreconditioner(HybridPreconditioner&&) noexcept = default;

HybridPreconditioner& HybridPreconditioner::operator=(HybridPreconditioner&&) noexcept = default;

Eigen::MatrixXcd HybridPreconditioner::apply(const Eigen::MatrixXcd& r) const {
    // P⁻¹ = D^{-1/2} · C⁻¹ · D^{-1/2}

    const int Ny = static_cast<int>(inv_sqrt_diagonal_.rows());
    const int Nx = static_cast<int>(inv_sqrt_diagonal_.cols());

    // Validate input dimensions
    if (r.rows() != Ny || r.cols() != Nx) {
        throw std::invalid_argument(
            "HybridPreconditioner::apply: input matrix dimension mismatch");
    }

    // Step 1: Apply D^{-1/2}
    Eigen::MatrixXcd temp(Ny, Nx);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            temp(j, i) = r(j, i) * inv_sqrt_diagonal_(j, i);
        }
    }

    // Step 2: Apply circulant inverse
    temp = circulant_->apply(temp);

    // Step 3: Apply D^{-1/2} again
    Eigen::MatrixXcd result(Ny, Nx);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            result(j, i) = temp(j, i) * inv_sqrt_diagonal_(j, i);
        }
    }

    return result;
}

// =============================================================================
// Factory function
// =============================================================================

PreconditionerVariant create_preconditioner(
    PreconditionerType type,
    const std::vector<double>& operator_diagonal,
    const std::vector<double>& transfer_function,
    const std::vector<double>& regularizer_spectrum,
    double lambda,
    double epsilon,
    int rows,
    int cols) {

    switch (type) {
        case PRECONDITIONER_NONE:
            return IdentityPreconditioner{};

        case PRECONDITIONER_JACOBI:
            return JacobiPreconditioner(operator_diagonal, regularizer_spectrum,
                                        lambda, rows, cols);

        case PRECONDITIONER_CIRCULANT:
            return CirculantPreconditioner(transfer_function, regularizer_spectrum,
                                           lambda, epsilon, rows, cols);

        case PRECONDITIONER_HYBRID:
            return HybridPreconditioner(operator_diagonal, transfer_function,
                                        regularizer_spectrum, lambda, epsilon,
                                        rows, cols);

        case PRECONDITIONER_UNKNOWN:
        default:
            throw std::invalid_argument(
                "create_preconditioner: PRECONDITIONER_UNKNOWN is not valid");
    }
}

}  // namespace aurora::reconstruction
