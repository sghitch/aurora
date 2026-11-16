#include "forward/fft2d.h"

#include <kiss_fft.h>

#include <stdexcept>
#include <vector>

#ifdef AURORA_USE_OPENMP
#include <omp.h>
#endif

namespace aurora::forward {

struct Fft2d::Impl {
    kiss_fft_cfg row_plan = nullptr;
    kiss_fft_cfg col_plan = nullptr;

    // Temporary buffers for row/column processing
    mutable std::vector<kiss_fft_cpx> row_buffer;
    mutable std::vector<kiss_fft_cpx> col_buffer;
    mutable std::vector<kiss_fft_cpx> temp_buffer;

    ~Impl() {
        if (row_plan) {
            kiss_fft_free(row_plan);
        }
        if (col_plan) {
            kiss_fft_free(col_plan);
        }
    }
};

Fft2d::Fft2d(int nx, int ny, bool inverse)
    : nx_(nx), ny_(ny), inverse_(inverse), impl_(std::make_unique<Impl>()) {

    if (nx <= 0 || ny <= 0) {
        throw std::invalid_argument("FFT dimensions must be positive");
    }

    // Create row FFT plan (nx points)
    impl_->row_plan = kiss_fft_alloc(
        nx_,
        inverse_ ? 1 : 0,
        nullptr,
        nullptr);

    if (!impl_->row_plan) {
        throw std::runtime_error("Failed to allocate KissFFT row plan");
    }

    // Create column FFT plan (ny points)
    impl_->col_plan = kiss_fft_alloc(
        ny_,
        inverse_ ? 1 : 0,
        nullptr,
        nullptr);

    if (!impl_->col_plan) {
        throw std::runtime_error("Failed to allocate KissFFT column plan");
    }

    // Pre-allocate buffers
    impl_->row_buffer.resize(nx_);
    impl_->col_buffer.resize(ny_);
    impl_->temp_buffer.resize(std::max(nx_, ny_));
}

Fft2d::~Fft2d() = default;

Fft2d::Fft2d(Fft2d&&) noexcept = default;

Fft2d& Fft2d::operator=(Fft2d&&) noexcept = default;

void Fft2d::execute(const Eigen::MatrixXcd& input, Eigen::MatrixXcd& output) const {
    if (input.cols() != nx_ || input.rows() != ny_) {
        throw std::invalid_argument(
            "Input dimensions (" + std::to_string(input.rows()) + "x" +
            std::to_string(input.cols()) + ") don't match FFT dimensions (" +
            std::to_string(ny_) + "x" + std::to_string(nx_) + ")");
    }

    // Verify memory layout compatibility
    static_assert(sizeof(kiss_fft_cpx) == sizeof(std::complex<double>),
                  "kiss_fft_cpx and std::complex<double> must have same size");

    // Resize output if needed
    if (output.rows() != ny_ || output.cols() != nx_) {
        output.resize(ny_, nx_);
    }

    // Intermediate matrix for row-transformed data
    Eigen::MatrixXcd intermediate(ny_, nx_);

    // Step 1: Transform each row in parallel
    // Each thread gets its own buffers to avoid race conditions
#ifdef AURORA_USE_OPENMP
#pragma omp parallel
    {
        // Thread-local buffers
        std::vector<kiss_fft_cpx> row_buffer(nx_);
        std::vector<kiss_fft_cpx> temp_buffer(nx_);

#pragma omp for schedule(static)
        for (int row = 0; row < ny_; ++row) {
            // Copy row to buffer (Eigen is column-major, so we need explicit copy)
            for (int col = 0; col < nx_; ++col) {
                row_buffer[col].r = input(row, col).real();
                row_buffer[col].i = input(row, col).imag();
            }

            // Execute row FFT (plan is read-only, thread-safe)
            kiss_fft(impl_->row_plan, row_buffer.data(), temp_buffer.data());

            // Copy result to intermediate matrix
            for (int col = 0; col < nx_; ++col) {
                intermediate(row, col) = std::complex<double>(
                    temp_buffer[col].r, temp_buffer[col].i);
            }
        }
    }
#else
    // Sequential fallback
    for (int row = 0; row < ny_; ++row) {
        for (int col = 0; col < nx_; ++col) {
            impl_->row_buffer[col].r = input(row, col).real();
            impl_->row_buffer[col].i = input(row, col).imag();
        }
        kiss_fft(impl_->row_plan, impl_->row_buffer.data(), impl_->temp_buffer.data());
        for (int col = 0; col < nx_; ++col) {
            intermediate(row, col) = std::complex<double>(
                impl_->temp_buffer[col].r, impl_->temp_buffer[col].i);
        }
    }
#endif

    // Step 2: Transform each column in parallel
#ifdef AURORA_USE_OPENMP
#pragma omp parallel
    {
        // Thread-local buffers
        std::vector<kiss_fft_cpx> col_buffer(ny_);
        std::vector<kiss_fft_cpx> temp_buffer(ny_);

#pragma omp for schedule(static)
        for (int col = 0; col < nx_; ++col) {
            // Copy column to buffer
            for (int row = 0; row < ny_; ++row) {
                col_buffer[row].r = intermediate(row, col).real();
                col_buffer[row].i = intermediate(row, col).imag();
            }

            // Execute column FFT (plan is read-only, thread-safe)
            kiss_fft(impl_->col_plan, col_buffer.data(), temp_buffer.data());

            // Copy result to output matrix
            for (int row = 0; row < ny_; ++row) {
                output(row, col) = std::complex<double>(
                    temp_buffer[row].r, temp_buffer[row].i);
            }
        }
    }
#else
    // Sequential fallback
    for (int col = 0; col < nx_; ++col) {
        for (int row = 0; row < ny_; ++row) {
            impl_->col_buffer[row].r = intermediate(row, col).real();
            impl_->col_buffer[row].i = intermediate(row, col).imag();
        }
        kiss_fft(impl_->col_plan, impl_->col_buffer.data(), impl_->temp_buffer.data());
        for (int row = 0; row < ny_; ++row) {
            output(row, col) = std::complex<double>(
                impl_->temp_buffer[row].r, impl_->temp_buffer[row].i);
        }
    }
#endif
}

Eigen::MatrixXcd Fft2d::execute(const Eigen::MatrixXcd& input) const {
    Eigen::MatrixXcd output;
    execute(input, output);
    return output;
}

}  // namespace aurora::forward
