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

#include "reconstruction/nufft_operator.h"

#include <cmath>
#include <stdexcept>

#include <kiss_fft.h>

#include "forward/forward_model.h"
#include "forward/k_space_geometry.h"

namespace aurora::reconstruction {

namespace {

// Convert NufftWindowType to forward::BandwidthWindowType
forward::BandwidthWindowType convert_window_type(NufftWindowType type) {
    switch (type) {
        case NUFFT_WINDOW_SINC:
            return forward::BANDWIDTH_WINDOW_SINC;
        case NUFFT_WINDOW_GAUSSIAN:
            return forward::BANDWIDTH_WINDOW_GAUSSIAN;
        case NUFFT_WINDOW_KAISER:
            return forward::BANDWIDTH_WINDOW_KAISER;
        case NUFFT_WINDOW_UNSPECIFIED:
        default:
            throw std::invalid_argument(
                "NufftConfig: window_type must be explicitly set "
                "(NUFFT_WINDOW_SINC, NUFFT_WINDOW_GAUSSIAN, or NUFFT_WINDOW_KAISER)");
    }
}

// Compute Kaiser-Bessel beta from oversampling and kernel width.
// Uses the Beatty et al. formula. Validates inputs and throws on invalid combinations.
double compute_default_beta(double oversampling, int kernel_width) {
    if (oversampling <= 0.0) {
        throw std::invalid_argument(
            "compute_default_beta: oversampling must be > 0.0");
    }
    if (kernel_width < 1) {
        throw std::invalid_argument(
            "compute_default_beta: kernel_width must be >= 1");
    }

    double W = static_cast<double>(kernel_width);
    double t = 1.0 - 1.0 / (2.0 * oversampling);
    double radicand = W * W * t * t - 0.8;

    if (radicand <= 0.0) {
        throw std::domain_error(
            "compute_default_beta: invalid oversampling/kernel_width "
            "combination leading to negative sqrt argument");
    }

    return M_PI * std::sqrt(radicand);
}

}  // namespace

// =============================================================================
// Config validation
// =============================================================================

void NufftOperator::validate_config(const NufftConfig& config) {
    // Validate kernel type
    if (config.kernel() == KERNEL_UNSPECIFIED) {
        throw std::invalid_argument(
            "NufftConfig: kernel must be explicitly set "
            "(KERNEL_KAISER_BESSEL or KERNEL_GAUSSIAN)");
    }

    // Validate oversampling_factor
    if (!config.has_oversampling_factor()) {
        throw std::invalid_argument(
            "NufftConfig: oversampling_factor must be explicitly set");
    }
    if (config.oversampling_factor() < 1.0) {
        throw std::invalid_argument(
            "NufftConfig: oversampling_factor must be >= 1.0");
    }

    // Validate kernel_width
    if (!config.has_kernel_width()) {
        throw std::invalid_argument(
            "NufftConfig: kernel_width must be explicitly set");
    }
    if (config.kernel_width() < 1) {
        throw std::invalid_argument(
            "NufftConfig: kernel_width must be >= 1");
    }
}

// =============================================================================
// Pimpl for FFT state
// =============================================================================

struct NufftOperator::Impl {
    kiss_fft_cfg fft_row_plan = nullptr;
    kiss_fft_cfg fft_col_plan = nullptr;
    kiss_fft_cfg ifft_row_plan = nullptr;
    kiss_fft_cfg ifft_col_plan = nullptr;

    // Temporary buffers
    mutable std::vector<kiss_fft_cpx> row_buffer;
    mutable std::vector<kiss_fft_cpx> col_buffer;
    mutable std::vector<kiss_fft_cpx> temp_buffer;

    ~Impl() {
        if (fft_row_plan) kiss_fft_free(fft_row_plan);
        if (fft_col_plan) kiss_fft_free(fft_col_plan);
        if (ifft_row_plan) kiss_fft_free(ifft_row_plan);
        if (ifft_col_plan) kiss_fft_free(ifft_col_plan);
    }
};

// =============================================================================
// Constructor
// =============================================================================

NufftOperator::NufftOperator(
    const aurora::shared::KSpaceSlice& slice,
    const aurora::forward::ForwardModel& forward_model,
    const Eigen::Vector2i& grid_size,
    const Eigen::Vector2d& grid_spacing,
    const NufftConfig& config)
    : grid_size_(grid_size),
      grid_spacing_(grid_spacing),
      config_(config),
      impl_(std::make_unique<Impl>()) {

    // Validate config first (throws on error)
    validate_config(config_);

    // Validate grid parameters (before extracting samples)
    if (grid_size.x() <= 0 || grid_size.y() <= 0) {
        throw std::invalid_argument(
            "NufftOperator: grid_size must be positive");
    }

    if (grid_spacing.x() <= 0 || grid_spacing.y() <= 0) {
        throw std::invalid_argument(
            "NufftOperator: grid_spacing must be positive");
    }

    // Extract K-vectors and sample weights
    const int num = slice.samples_size();
    k_vectors_.reserve(num);
    weights_.reserve(num);

    for (int i = 0; i < num; ++i) {
        const auto& sample = slice.samples(i);
        k_vectors_.emplace_back(sample.kx(), sample.ky());
        weights_.push_back(sample.weight());
    }

    // Build the K-space window of each sample for physics weighting: centered
    // on the sample's K-position, bandwidth from its detector pixel size
    window_params_.reserve(num);
    bool any_valid_bandwidth = false;
    for (int i = 0; i < num; ++i) {
        const auto& sample = slice.samples(i);
        auto params = forward::build_window_params(
            forward_model, sample.detector_index(), slice.wavelength_m());
        params.k_center = k_vectors_[i];
        window_params_.push_back(params);

        // Check if this sample has valid (non-zero) bandwidth
        if (params.bandwidth_x > 0.0 && params.bandwidth_y > 0.0) {
            any_valid_bandwidth = true;
        }
    }

    // Only enable physics weighting if at least one sample has valid bandwidth
    use_physics_weighting_ = any_valid_bandwidth;

    // Require explicit window_type when physics weighting is enabled
    if (use_physics_weighting_) {
        if (!config.has_window_type() || config.window_type() == NUFFT_WINDOW_UNSPECIFIED) {
            throw std::invalid_argument(
                "NufftConfig: window_type must be explicitly set when bandwidth "
                "info is available (use NUFFT_WINDOW_SINC, NUFFT_WINDOW_GAUSSIAN, "
                "or NUFFT_WINDOW_KAISER)");
        }
        window_type_ = convert_window_type(config.window_type());
    }

    initialize();
}

NufftOperator::~NufftOperator() = default;

NufftOperator::NufftOperator(NufftOperator&&) noexcept = default;

NufftOperator& NufftOperator::operator=(NufftOperator&&) noexcept = default;

void NufftOperator::initialize() {
    // Compute oversampled grid size
    int os_nx = static_cast<int>(std::ceil(grid_size_.x() * config_.oversampling_factor()));
    int os_ny = static_cast<int>(std::ceil(grid_size_.y() * config_.oversampling_factor()));
    oversampled_size_ = Eigen::Vector2i(os_nx, os_ny);

    // Compute effective beta for Kaiser-Bessel kernel
    // If not specified or <= 0, compute from oversampling and width using Beatty formula
    double kernel_beta = config_.has_kernel_beta() ? config_.kernel_beta() : 0.0;
    if (config_.kernel() == KERNEL_KAISER_BESSEL && kernel_beta <= 0.0) {
        effective_beta_ = compute_default_beta(
            config_.oversampling_factor(),
            config_.kernel_width());
    } else {
        effective_beta_ = kernel_beta;
    }

    // Allocate FFT plans for oversampled grid
    impl_->fft_row_plan = kiss_fft_alloc(os_nx, 0, nullptr, nullptr);
    impl_->fft_col_plan = kiss_fft_alloc(os_ny, 0, nullptr, nullptr);
    impl_->ifft_row_plan = kiss_fft_alloc(os_nx, 1, nullptr, nullptr);
    impl_->ifft_col_plan = kiss_fft_alloc(os_ny, 1, nullptr, nullptr);

    if (!impl_->fft_row_plan || !impl_->fft_col_plan ||
        !impl_->ifft_row_plan || !impl_->ifft_col_plan) {
        throw std::runtime_error("Failed to allocate KissFFT plans");
    }

    // Pre-allocate buffers
    impl_->row_buffer.resize(os_nx);
    impl_->col_buffer.resize(os_ny);
    impl_->temp_buffer.resize(std::max(os_nx, os_ny));

    // Precompute gridding data (always precompute for correctness)
    precompute();
}

void NufftOperator::precompute() {
    compute_deapodization();
    compute_kernel_samples();
}

void NufftOperator::compute_deapodization() {
    // Compute the deapodization correction factors.
    // The deapodization is the FT of the interpolation kernel, evaluated at each
    // grid position. This corrects for the kernel's spreading effect.
    //
    // For a separable kernel: deap(y, x) = deap_y(y) * deap_x(x)
    // where deap_1d(n) = Σ_k kernel(k) * exp(2πi * n * k / os_N)
    //
    // Since we place the image at positions 0..N-1 of the oversampled grid,
    // and the kernel spreads each K-sample across nearby FFT bins,
    // we need to correct for this spreading when extracting the image.

    const int Nx = grid_size_.x();
    const int Ny = grid_size_.y();
    const int os_nx = oversampled_size_.x();
    const int os_ny = oversampled_size_.y();
    const int W = config_.kernel_width();
    const int half_W = W / 2;

    // Compute 1D deapodization for x direction
    std::vector<double> deap_x(Nx);
    for (int n = 0; n < Nx; ++n) {
        double sum = 0.0;
        for (int k = -half_W; k <= half_W; ++k) {
            double kern_val = evaluate_kernel(static_cast<double>(k));
            // exp(2πi * n * k / os_nx) - we only need the real part for symmetric kernel
            double angle = 2.0 * M_PI * n * k / static_cast<double>(os_nx);
            sum += kern_val * std::cos(angle);
        }
        deap_x[n] = sum;
    }

    // Compute 1D deapodization for y direction
    std::vector<double> deap_y(Ny);
    for (int n = 0; n < Ny; ++n) {
        double sum = 0.0;
        for (int k = -half_W; k <= half_W; ++k) {
            double kern_val = evaluate_kernel(static_cast<double>(k));
            double angle = 2.0 * M_PI * n * k / static_cast<double>(os_ny);
            sum += kern_val * std::cos(angle);
        }
        deap_y[n] = sum;
    }

    // 2D deapodization is the outer product
    deapodization_.resize(Ny, Nx);
    for (int iy = 0; iy < Ny; ++iy) {
        for (int ix = 0; ix < Nx; ++ix) {
            double deap = deap_y[iy] * deap_x[ix];
            // Avoid division by very small numbers
            if (std::abs(deap) < 1e-10) {
                deapodization_(iy, ix) = 1e-10;
            } else {
                deapodization_(iy, ix) = deap;
            }
        }
    }
}

void NufftOperator::compute_kernel_samples() {
    const int M = static_cast<int>(k_vectors_.size());
    const int W = config_.kernel_width();
    const int half_W = W / 2;

    kernel_samples_.resize(M);
    k_normalized_.resize(M);

    const int os_nx = oversampled_size_.x();
    const int os_ny = oversampled_size_.y();

    // The FFT bin spacing in physical K-space is:
    // delta_k = 2*pi / (N * spacing)
    // So K-vector k maps to fractional bin: bin = k / delta_k = k * N * spacing / (2*pi)
    const double scale_x = grid_spacing_.x() * os_nx / (2.0 * M_PI);
    const double scale_y = grid_spacing_.y() * os_ny / (2.0 * M_PI);

    // K-space delta per FFT bin (for physics weighting)
    const double dk_x = 2.0 * M_PI / (os_nx * grid_spacing_.x());
    const double dk_y = 2.0 * M_PI / (os_ny * grid_spacing_.y());

    // Half K-bandwidth of the FFT grid (for unwrapping negative frequencies)
    const double k_nyquist_x = M_PI / grid_spacing_.x();
    const double k_nyquist_y = M_PI / grid_spacing_.y();

    for (int m = 0; m < M; ++m) {
        // K-vector position in FFT bin units (can be fractional)
        double kx_bin = k_vectors_[m].x() * scale_x;
        double ky_bin = k_vectors_[m].y() * scale_y;

        k_normalized_[m] = Eigen::Vector2d(kx_bin, ky_bin);

        // Wrap to [0, N) range for FFT indexing
        // Negative frequencies wrap to high indices
        double kx_wrapped = std::fmod(kx_bin, static_cast<double>(os_nx));
        if (kx_wrapped < 0) kx_wrapped += os_nx;
        double ky_wrapped = std::fmod(ky_bin, static_cast<double>(os_ny));
        if (ky_wrapped < 0) ky_wrapped += os_ny;

        // Nearest grid point
        int nearest_gx = static_cast<int>(std::round(kx_wrapped));
        int nearest_gy = static_cast<int>(std::round(ky_wrapped));

        // Handle edge case where round gives N
        if (nearest_gx >= os_nx) nearest_gx = 0;
        if (nearest_gy >= os_ny) nearest_gy = 0;

        // Fractional offset from nearest grid point
        double frac_x = kx_wrapped - nearest_gx;
        double frac_y = ky_wrapped - nearest_gy;

        // Compute kernel weights for all grid points in support
        // The kernel has support [-W/2, W/2], so we need to check grid points
        // that could be within this distance of the fractional position.
        // NOTE: We do NOT normalize the weights here - the deapodization
        // correction handles the kernel's amplitude effect.
        kernel_samples_[m].clear();
        kernel_samples_[m].reserve((W + 1) * (W + 1));

        // Loop over grid points relative to the nearest integer grid point
        // We need enough points to cover the kernel support on both sides
        for (int dy = -half_W; dy <= half_W; ++dy) {
            // Distance from K-position to this grid point
            // If frac_y = 0.27 and dy = 0, grid point is at nearest_gy + 0,
            // and distance is 0.27 - 0 = 0.27
            double dist_y = frac_y - dy;
            double kernel_y = evaluate_kernel(dist_y);

            for (int dx = -half_W; dx <= half_W; ++dx) {
                double dist_x = frac_x - dx;
                double kernel_x = evaluate_kernel(dist_x);

                // NUFFT kernel weight (numerical interpolation)
                double weight = kernel_x * kernel_y;

                // Apply physics weighting
                // a_{m,j} = φ(k_j - k_m) * W_m(k_j)
                if (use_physics_weighting_ && weight > 1e-15) {
                    // Wrap grid indices
                    int gx = (nearest_gx + dx) % os_nx;
                    if (gx < 0) gx += os_nx;
                    int gy = (nearest_gy + dy) % os_ny;
                    if (gy < 0) gy += os_ny;

                    // Convert grid index to physical K-space position
                    double k_grid_x = gx * dk_x;
                    double k_grid_y = gy * dk_y;

                    // Unwrap: high indices are negative frequencies
                    if (k_grid_x > k_nyquist_x) {
                        k_grid_x -= 2.0 * k_nyquist_x;
                    }
                    if (k_grid_y > k_nyquist_y) {
                        k_grid_y -= 2.0 * k_nyquist_y;
                    }

                    Eigen::Vector2d k_grid(k_grid_x, k_grid_y);
                    double physics_weight = forward::MeasurementOperator::weight(
                        k_grid, window_params_[m], window_type_);

                    weight *= physics_weight;  // Composite weight
                }

                if (std::abs(weight) > 1e-15) {
                    KernelSample ks;
                    // Wrap grid indices (may have been computed above)
                    int gx = (nearest_gx + dx) % os_nx;
                    if (gx < 0) gx += os_nx;
                    int gy = (nearest_gy + dy) % os_ny;
                    if (gy < 0) gy += os_ny;

                    ks.grid_offset_x = gx;
                    ks.grid_offset_y = gy;
                    ks.weight = weight;
                    kernel_samples_[m].push_back(ks);
                }
            }
        }
    }
}

double NufftOperator::evaluate_kernel(double x) const {
    const int kernel_width = config_.kernel_width();
    switch (config_.kernel()) {
        case KERNEL_KAISER_BESSEL:
            return kaiser_bessel(x, effective_beta_, kernel_width);
        case KERNEL_GAUSSIAN:
            return gaussian_kernel(x, kernel_width / 3.0);
        default:
            // KERNEL_UNSPECIFIED should never get here (validated in constructor)
            return kaiser_bessel(x, effective_beta_, kernel_width);
    }
}

double NufftOperator::kaiser_bessel(double x, double beta, int width) const {
    const double w = width / 2.0;
    if (std::abs(x) >= w) {
        return 0.0;
    }

    const double ratio = x / w;
    const double arg = beta * std::sqrt(1.0 - ratio * ratio);
    return bessel_i0(arg) / bessel_i0(beta);
}

double NufftOperator::gaussian_kernel(double x, double sigma) const {
    return std::exp(-x * x / (2.0 * sigma * sigma));
}

double NufftOperator::bessel_i0(double x) {
    double ax = std::abs(x);

    if (ax < 3.75) {
        double t = x / 3.75;
        double t2 = t * t;
        return 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492 +
               t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
    } else {
        double t = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) *
               (0.39894228 + t * (0.01328592 + t * (0.00225319 +
               t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 +
               t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))));
    }
}

Eigen::VectorXcd NufftOperator::forward(
    const Eigen::MatrixXcd& scene,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing) const {

    const Eigen::Index Nx = scene.cols();
    const Eigen::Index Ny = scene.rows();
    const Eigen::Index M = static_cast<Eigen::Index>(k_vectors_.size());

    if (M == 0) {
        return Eigen::VectorXcd(0);
    }

    if (Nx != grid_size_.x() || Ny != grid_size_.y()) {
        throw std::invalid_argument(
            "Scene dimensions don't match NufftOperator grid_size");
    }

    const int os_nx = oversampled_size_.x();
    const int os_ny = oversampled_size_.y();

    // Step 1: Apodize (pre-divide by deapodization) and zero-pad to oversampled grid
    // The apodization corrects for the kernel's frequency response in advance.
    Eigen::MatrixXcd padded = Eigen::MatrixXcd::Zero(os_ny, os_nx);
    for (Eigen::Index iy = 0; iy < Ny; ++iy) {
        for (Eigen::Index ix = 0; ix < Nx; ++ix) {
            // Apodize: divide by deapodization factor
            padded(iy, ix) = scene(iy, ix) / deapodization_(iy, ix);
        }
    }

    // Step 2: Apply 2D FFT
    // Row FFTs
    Eigen::MatrixXcd intermediate(os_ny, os_nx);
    for (int row = 0; row < os_ny; ++row) {
        for (int col = 0; col < os_nx; ++col) {
            impl_->row_buffer[col].r = padded(row, col).real();
            impl_->row_buffer[col].i = padded(row, col).imag();
        }
        kiss_fft(impl_->fft_row_plan, impl_->row_buffer.data(), impl_->temp_buffer.data());
        for (int col = 0; col < os_nx; ++col) {
            intermediate(row, col) = std::complex<double>(
                impl_->temp_buffer[col].r, impl_->temp_buffer[col].i);
        }
    }

    // Column FFTs
    Eigen::MatrixXcd fft_result(os_ny, os_nx);
    for (int col = 0; col < os_nx; ++col) {
        for (int row = 0; row < os_ny; ++row) {
            impl_->col_buffer[row].r = intermediate(row, col).real();
            impl_->col_buffer[row].i = intermediate(row, col).imag();
        }
        kiss_fft(impl_->fft_col_plan, impl_->col_buffer.data(), impl_->temp_buffer.data());
        for (int row = 0; row < os_ny; ++row) {
            fft_result(row, col) = std::complex<double>(
                impl_->temp_buffer[row].r, impl_->temp_buffer[row].i);
        }
    }

    // Step 3: Interpolate at K-positions and apply phase corrections
    Eigen::VectorXcd result(M);

    for (Eigen::Index m = 0; m < M; ++m) {
        std::complex<double> sum{0.0, 0.0};

        for (const auto& ks : kernel_samples_[m]) {
            sum += fft_result(ks.grid_offset_y, ks.grid_offset_x) * ks.weight;
        }

        // The FFT computes: FFT[n] = Σ_i x[i] * exp(-i * 2π * n * i / N)
        // which corresponds to: y(k) = Σ_i x[i] * exp(-i * k * i * spacing)
        // where k = 2π * n / (N * spacing)
        //
        // But DirectOperator uses: y_m = Σ_i x[i] * exp(-i * k_m * r_i)
        // where r_i = origin + i * spacing
        //
        // So: y_m = Σ_i x[i] * exp(-i * k_m * origin) * exp(-i * k_m * i * spacing)
        //         = exp(-i * k_m * origin) * FFT_interpolated
        //
        // Therefore we need to apply: exp(-i * k · origin)
        double phase = -(k_vectors_[m].x() * grid_origin.x() +
                        k_vectors_[m].y() * grid_origin.y());
        sum *= std::polar(1.0, phase);

        // Apply sample weight
        result(m) = weights_[m] * sum;
    }

    return result;
}

Eigen::MatrixXcd NufftOperator::adjoint(
    const Eigen::VectorXcd& k_samples,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2i& grid_size) const {

    const Eigen::Index M = k_samples.size();
    const int Nx = grid_size.x();
    const int Ny = grid_size.y();

    // Validate input size matches operator's K-vector count
    if (static_cast<size_t>(M) != k_vectors_.size()) {
        throw std::invalid_argument(
            "NufftOperator::adjoint: k_samples size does not match number of k-vectors");
    }

    if (M == 0) {
        return Eigen::MatrixXcd::Zero(Ny, Nx);
    }

    const int os_nx = oversampled_size_.x();
    const int os_ny = oversampled_size_.y();

    // Step 1: Grid K-samples onto oversampled regular grid
    Eigen::MatrixXcd grid = Eigen::MatrixXcd::Zero(os_ny, os_nx);

    for (Eigen::Index m = 0; m < M; ++m) {
        // Apply phase shift for grid origin
        // Adjoint uses conjugate: exp(+i * k · origin)
        double phase = k_vectors_[m].x() * grid_origin.x() +
                      k_vectors_[m].y() * grid_origin.y();
        std::complex<double> value = weights_[m] * k_samples(m) *
                                     std::polar(1.0, phase);

        for (const auto& ks : kernel_samples_[m]) {
            grid(ks.grid_offset_y, ks.grid_offset_x) += value * ks.weight;
        }
    }

    // Step 2: Apply 2D inverse FFT
    // Row IFFTs
    Eigen::MatrixXcd intermediate(os_ny, os_nx);
    for (int row = 0; row < os_ny; ++row) {
        for (int col = 0; col < os_nx; ++col) {
            impl_->row_buffer[col].r = grid(row, col).real();
            impl_->row_buffer[col].i = grid(row, col).imag();
        }
        kiss_fft(impl_->ifft_row_plan, impl_->row_buffer.data(), impl_->temp_buffer.data());
        for (int col = 0; col < os_nx; ++col) {
            intermediate(row, col) = std::complex<double>(
                impl_->temp_buffer[col].r, impl_->temp_buffer[col].i);
        }
    }

    // Column IFFTs
    Eigen::MatrixXcd ifft_result(os_ny, os_nx);
    for (int col = 0; col < os_nx; ++col) {
        for (int row = 0; row < os_ny; ++row) {
            impl_->col_buffer[row].r = intermediate(row, col).real();
            impl_->col_buffer[row].i = intermediate(row, col).imag();
        }
        kiss_fft(impl_->ifft_col_plan, impl_->col_buffer.data(), impl_->temp_buffer.data());
        for (int row = 0; row < os_ny; ++row) {
            ifft_result(row, col) = std::complex<double>(
                impl_->temp_buffer[row].r, impl_->temp_buffer[row].i);
        }
    }

    // Step 3: Extract result (first Ny x Nx pixels) and deapodize
    // The deapodization corrects for the kernel's spreading effect.
    // KissFFT inverse doesn't normalize, so we also don't normalize here.
    Eigen::MatrixXcd result(Ny, Nx);
    for (int iy = 0; iy < Ny; ++iy) {
        for (int ix = 0; ix < Nx; ++ix) {
            result(iy, ix) = ifft_result(iy, ix) / deapodization_(iy, ix);
        }
    }

    return result;
}

}  // namespace aurora::reconstruction
