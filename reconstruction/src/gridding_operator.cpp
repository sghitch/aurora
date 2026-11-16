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

#include "reconstruction/gridding_operator.h"

#include <cmath>
#include <stdexcept>

#ifdef AURORA_USE_OPENMP
#include <omp.h>
#endif

#include "forward/fft2d.h"

namespace aurora::reconstruction {

// =============================================================================
// Pimpl for FFT state
// =============================================================================

struct GriddingOperator::Impl {
    std::unique_ptr<aurora::forward::Fft2d> fft_forward;
    std::unique_ptr<aurora::forward::Fft2d> fft_inverse;
};

// =============================================================================
// Constructors
// =============================================================================

GriddingOperator::GriddingOperator(
    std::vector<Eigen::Vector2d> k_vectors,
    std::vector<double> weights,
    const Eigen::Vector2i& grid_size,
    const Eigen::Vector2d& grid_spacing,
    const GriddingConfig& config)
    : k_vectors_(std::move(k_vectors)),
      weights_(std::move(weights)),
      grid_size_(grid_size),
      grid_spacing_(grid_spacing),
      config_(config),
      kernel_width_(0),
      gaussian_sigma_(1.0),
      kaiser_beta_(0.0),
      impl_(std::make_unique<Impl>()) {

    // Validate kernel is explicitly set when config is provided
    if (config_.kernel() == GRIDDING_KERNEL_UNSPECIFIED) {
        throw std::invalid_argument(
            "GriddingConfig: kernel must be explicitly set "
            "(GRIDDING_KERNEL_NEAREST, GRIDDING_KERNEL_GAUSSIAN, or GRIDDING_KERNEL_KAISER_BESSEL)");
    }

    if (k_vectors_.size() != weights_.size()) {
        throw std::invalid_argument(
            "GriddingOperator: k_vectors and weights must have same size");
    }
    if (grid_size.x() <= 0 || grid_size.y() <= 0) {
        throw std::invalid_argument(
            "GriddingOperator: grid_size must be positive");
    }
    if (grid_spacing.x() <= 0 || grid_spacing.y() <= 0) {
        throw std::invalid_argument(
            "GriddingOperator: grid_spacing must be positive");
    }

    setup_kernel_params();
    initialize();
}

GriddingOperator::GriddingOperator(
    const aurora::shared::KSpaceSlice& slice,
    const Eigen::Vector2i& grid_size,
    const Eigen::Vector2d& grid_spacing,
    const GriddingConfig& config)
    : grid_size_(grid_size),
      grid_spacing_(grid_spacing),
      config_(config),
      kernel_width_(0),
      gaussian_sigma_(1.0),
      kaiser_beta_(0.0),
      impl_(std::make_unique<Impl>()) {

    // Validate kernel is explicitly set when config is provided
    if (config_.kernel() == GRIDDING_KERNEL_UNSPECIFIED) {
        throw std::invalid_argument(
            "GriddingConfig: kernel must be explicitly set "
            "(GRIDDING_KERNEL_NEAREST, GRIDDING_KERNEL_GAUSSIAN, or GRIDDING_KERNEL_KAISER_BESSEL)");
    }

    // Extract K-vectors and sample weights from slice
    k_vectors_.reserve(slice.samples_size());
    weights_.reserve(slice.samples_size());

    for (const auto& sample : slice.samples()) {
        k_vectors_.emplace_back(sample.kx(), sample.ky());
        weights_.push_back(sample.weight());
    }

    if (grid_size.x() <= 0 || grid_size.y() <= 0) {
        throw std::invalid_argument(
            "GriddingOperator: grid_size must be positive");
    }
    if (grid_spacing.x() <= 0 || grid_spacing.y() <= 0) {
        throw std::invalid_argument(
            "GriddingOperator: grid_spacing must be positive");
    }

    setup_kernel_params();
    initialize();
}

GriddingOperator::~GriddingOperator() = default;

GriddingOperator::GriddingOperator(GriddingOperator&&) noexcept = default;
GriddingOperator& GriddingOperator::operator=(GriddingOperator&&) noexcept = default;

void GriddingOperator::initialize() {
    // Create FFT plans
    impl_->fft_forward = std::make_unique<aurora::forward::Fft2d>(
        grid_size_.x(), grid_size_.y(), false);  // forward FFT
    impl_->fft_inverse = std::make_unique<aurora::forward::Fft2d>(
        grid_size_.x(), grid_size_.y(), true);   // inverse FFT
}

void GriddingOperator::setup_kernel_params() {
    // Set up kernel parameters from config
    kernel_width_ = config_.has_kernel_width()
        ? static_cast<int>(config_.kernel_width())
        : 2;

    // Validate kernel_width is positive for non-NEAREST kernels
    if (config_.kernel() != GRIDDING_KERNEL_NEAREST && kernel_width_ <= 0) {
        throw std::invalid_argument(
            "GriddingConfig: kernel_width must be positive for GAUSSIAN and KAISER_BESSEL kernels");
    }

    // Gaussian sigma: fraction of kernel width
    double sigma_fraction = config_.has_gaussian_sigma_fraction()
        ? config_.gaussian_sigma_fraction()
        : 0.5;

    // Validate gaussian_sigma_fraction is positive for Gaussian kernel
    if (config_.kernel() == GRIDDING_KERNEL_GAUSSIAN && sigma_fraction <= 0.0) {
        throw std::invalid_argument(
            "GriddingConfig: gaussian_sigma_fraction must be positive for Gaussian kernel");
    }
    gaussian_sigma_ = kernel_width_ * sigma_fraction;

    // Kaiser-Bessel beta: auto-compute if not specified
    if (config_.has_kaiser_beta()) {
        kaiser_beta_ = config_.kaiser_beta();
    } else {
        // Use a reasonable default based on kernel width W.
        //
        // The following heuristic is adapted from common NUFFT / gridding practice,
        // where the Kaiser-Bessel beta parameter is chosen as an increasing function
        // of the kernel width and the oversampling factor to balance main-lobe width
        // and side-lobe suppression (see Beatty et al., "Rapid gridding reconstruction
        // with a minimal oversampling ratio," IEEE TMI, 2005).
        //
        // We approximate:
        //     beta ≈ π * sqrt(W^2 * t^2 - 0.8)
        //
        // where t encodes an assumed *effective* oversampling. We take:
        //     t = 1 - 1/3 ≈ 0.667
        //
        // corresponding to an assumed effective oversampling of ~1.5 in each dimension.
        //
        // For very small W, the radicand can become negative; we fall back to beta=2.34,
        // a typical value for small Kaiser-Bessel windows (see Jackson et al., "Selection
        // of a convolution function for Fourier inversion using gridding," IEEE TMI, 1991).
        double W = static_cast<double>(kernel_width_);
        double t = 1.0 - 1.0 / 3.0;
        double radicand = W * W * t * t - 0.8;
        kaiser_beta_ = (radicand > 0) ? M_PI * std::sqrt(radicand) : 2.34;
    }
}

double GriddingOperator::evaluate_kernel(double distance_sq) const {
    double distance = std::sqrt(distance_sq);

    switch (config_.kernel()) {
        case GRIDDING_KERNEL_GAUSSIAN: {
            // Gaussian: exp(-d^2 / (2 * sigma^2))
            double sigma_sq = gaussian_sigma_ * gaussian_sigma_;
            return std::exp(-distance_sq / (2.0 * sigma_sq));
        }

        case GRIDDING_KERNEL_KAISER_BESSEL: {
            // Kaiser-Bessel with compact support
            double w = static_cast<double>(kernel_width_);
            if (distance >= w) {
                return 0.0;
            }
            double ratio = distance / w;
            double arg = kaiser_beta_ * std::sqrt(1.0 - ratio * ratio);
            return bessel_i0(arg) / bessel_i0(kaiser_beta_);
        }

        case GRIDDING_KERNEL_NEAREST:
        default:
            // Nearest-neighbor: weight is 1 only if distance < 0.5
            // Note: UNSPECIFIED is rejected in constructor, so should never reach here
            return (distance < 0.5) ? 1.0 : 0.0;
    }
}

double GriddingOperator::bessel_i0(double x) {
    // Polynomial approximation of the modified Bessel function of the first kind, I0(x).
    // Coefficients from Abramowitz & Stegun, "Handbook of Mathematical Functions",
    // sections 9.8.1 (|x| < 3.75) and 9.8.2 (|x| >= 3.75).
    // Maximum relative error: ~1.6e-7 for |x| < 3.75, ~1.9e-7 for |x| >= 3.75.
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

// =============================================================================
// Forward operator
// =============================================================================

Eigen::VectorXcd GriddingOperator::forward(
    const Eigen::MatrixXcd& scene,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing) const {

    int nx = grid_size_.x();
    int ny = grid_size_.y();

    // K-space grid spacing (from image grid via Fourier relationship)
    double dk_x = 2.0 * M_PI / (nx * grid_spacing.x());
    double dk_y = 2.0 * M_PI / (ny * grid_spacing.y());

    // K-space grid origin (centered)
    double k_origin_x = -dk_x * nx / 2.0;
    double k_origin_y = -dk_y * ny / 2.0;

    int half_nx = nx / 2;
    int half_ny = ny / 2;

    // Inverse FFT shift on spatial input (move origin from center to corner)
    // This is the adjoint of the FFT shift applied after IFFT in adjoint()
    Eigen::MatrixXcd scene_shifted(ny, nx);
#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int src_x = (ix + half_nx) % nx;
            int src_y = (iy + half_ny) % ny;
            scene_shifted(iy, ix) = scene(src_y, src_x);
        }
    }

    // Apply FFT to shifted scene
    Eigen::MatrixXcd k_grid = impl_->fft_forward->execute(scene_shifted);

    // FFT shift to center DC
    Eigen::MatrixXcd k_grid_shifted(ny, nx);
#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int src_x = (ix + half_nx) % nx;
            int src_y = (iy + half_ny) % ny;
            k_grid_shifted(iy, ix) = k_grid(src_y, src_x);
        }
    }

    // Determine kernel radius for interpolation
    bool use_nearest = (config_.kernel() == GRIDDING_KERNEL_NEAREST);
    int kernel_radius = use_nearest ? 0 : kernel_width_;

    // Interpolate at K-positions using kernel.
    // Complexity: O(M * W^2) where M = num samples, W = kernel_width.
    // For GRIDDING_KERNEL_NEAREST (W=0), this is O(M).
    // Combined with FFT, total forward complexity is O(N log N + M * W^2).
    Eigen::VectorXcd result(k_vectors_.size());
    const int num_samples = static_cast<int>(k_vectors_.size());

#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < num_samples; ++m) {
        double kx = k_vectors_[m].x();
        double ky = k_vectors_[m].y();

        // Find fractional grid position
        double fx = (kx - k_origin_x) / dk_x;
        double fy = (ky - k_origin_y) / dk_y;

        // Nearest grid point (center of kernel)
        int cx = static_cast<int>(std::round(fx));
        int cy = static_cast<int>(std::round(fy));

        if (kernel_radius == 0) {
            // Nearest-neighbor: single cell interpolation
            if (cx >= 0 && cx < nx && cy >= 0 && cy < ny) {
                result(m) = weights_[m] * k_grid_shifted(cy, cx);
            } else {
                result(m) = std::complex<double>(0, 0);
            }
        } else {
            // Kernel interpolation: weighted sum of nearby cells
            std::complex<double> sum(0, 0);
            double weight_sum = 0.0;

            for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;

                for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= nx) continue;

                    // Distance from sample position to this grid cell (in grid units)
                    double dist_x = fx - ix;
                    double dist_y = fy - iy;
                    double dist_sq = dist_x * dist_x + dist_y * dist_y;

                    // Evaluate kernel weight
                    double w = evaluate_kernel(dist_sq);
                    if (w > 1e-10) {
                        sum += k_grid_shifted(iy, ix) * w;
                        weight_sum += w;
                    }
                }
            }

            // Normalize by total weight
            if (weight_sum > 0) {
                result(m) = weights_[m] * sum / weight_sum;
            } else {
                result(m) = std::complex<double>(0, 0);
            }
        }
    }

    return result;
}

// =============================================================================
// Adjoint operator
// =============================================================================

Eigen::MatrixXcd GriddingOperator::adjoint(
    const Eigen::VectorXcd& k_samples,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2i& grid_size) const {

    int nx = grid_size.x();
    int ny = grid_size.y();

    // K-space grid
    Eigen::MatrixXcd k_grid = Eigen::MatrixXcd::Zero(ny, nx);
    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(ny, nx);

    // K-space grid spacing (from image grid via Fourier relationship)
    double dk_x = 2.0 * M_PI / (nx * grid_spacing.x());
    double dk_y = 2.0 * M_PI / (ny * grid_spacing.y());

    // K-space grid origin (centered)
    double k_origin_x = -dk_x * nx / 2.0;
    double k_origin_y = -dk_y * ny / 2.0;

    // Determine kernel radius for spreading
    bool use_nearest = (config_.kernel() == GRIDDING_KERNEL_NEAREST);
    int kernel_radius = use_nearest ? 0 : kernel_width_;

    // Grid each sample with kernel spreading.
    // Complexity: O(M * W^2) where M = num samples, W = kernel_width.
    // For GRIDDING_KERNEL_NEAREST (W=0), this is O(M).
    // Combined with FFT, total adjoint complexity is O(N log N + M * W^2).
    for (size_t m = 0; m < k_vectors_.size(); ++m) {
        double kx = k_vectors_[m].x();
        double ky = k_vectors_[m].y();
        std::complex<double> value = k_samples(m);

        // Apply sample weight
        value *= weights_[m];

        // Find fractional grid position
        double fx = (kx - k_origin_x) / dk_x;
        double fy = (ky - k_origin_y) / dk_y;

        // Nearest grid point (center of kernel)
        int cx = static_cast<int>(std::round(fx));
        int cy = static_cast<int>(std::round(fy));

        if (kernel_radius == 0) {
            // Nearest-neighbor: single cell assignment
            if (cx >= 0 && cx < nx && cy >= 0 && cy < ny) {
                k_grid(cy, cx) += value;
                weights(cy, cx) += 1.0;
            }
        } else {
            // Kernel spreading: contribute to nearby cells
            for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;

                for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= nx) continue;

                    // Distance from sample position to this grid cell (in grid units)
                    double dist_x = fx - ix;
                    double dist_y = fy - iy;
                    double dist_sq = dist_x * dist_x + dist_y * dist_y;

                    // Evaluate kernel weight
                    double w = evaluate_kernel(dist_sq);
                    if (w > 1e-10) {
                        k_grid(iy, ix) += value * w;
                        weights(iy, ix) += w;
                    }
                }
            }
        }
    }

    // Normalize by weights
#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            if (weights(iy, ix) > 0) {
                k_grid(iy, ix) /= weights(iy, ix);
            }
        }
    }

    // Inverse FFT shift (move DC from center to corner)
    Eigen::MatrixXcd k_grid_shifted(ny, nx);
    int half_nx = nx / 2;
    int half_ny = ny / 2;
#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int dst_x = (ix + half_nx) % nx;
            int dst_y = (iy + half_ny) % ny;
            k_grid_shifted(dst_y, dst_x) = k_grid(iy, ix);
        }
    }

    // Apply inverse FFT (already parallelized in Fft2d)
    Eigen::MatrixXcd ifft_result = impl_->fft_inverse->execute(k_grid_shifted);

    // FFT shift to move spatial origin from corner to center
    // This ensures spatial (0,0) is at array index (nx/2, ny/2),
    // matching the grid_origin convention used by Reconstructor
    Eigen::MatrixXcd image(ny, nx);
    double norm = 1.0 / static_cast<double>(nx * ny);
#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int src_x = (ix + half_nx) % nx;
            int src_y = (iy + half_ny) % ny;
            image(iy, ix) = ifft_result(src_y, src_x) * norm;
        }
    }

    return image;
}

}  // namespace aurora::reconstruction
