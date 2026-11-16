#include "reconstruction/fast_online_operator.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef AURORA_USE_OPENMP
#include <omp.h>
#endif

#include "forward/fft2d.h"

namespace aurora::reconstruction {

// =============================================================================
// Pimpl for FFT state
// =============================================================================

struct FastOnlineOperator::Impl {
    std::unique_ptr<aurora::forward::Fft2d> fft_forward;
    std::unique_ptr<aurora::forward::Fft2d> fft_inverse;
};

// =============================================================================
// Constructors
// =============================================================================

FastOnlineOperator::FastOnlineOperator(
    std::vector<Eigen::Vector2d> k_vectors,
    std::vector<double> weights,
    std::vector<uint32_t> detector_indices,
    const Eigen::Vector2i& grid_size,
    const Eigen::Vector2d& grid_spacing,
    const FastOnlineConfig& config)
    : k_vectors_(std::move(k_vectors)),
      weights_(std::move(weights)),
      detector_indices_(std::move(detector_indices)),
      grid_size_(grid_size),
      grid_spacing_(grid_spacing),
      config_(config),
      num_detectors_(0),
      kernel_width_(0),
      gaussian_sigma_(1.0),
      kaiser_beta_(0.0),
      impl_(std::make_unique<Impl>()) {

    if (k_vectors_.size() != weights_.size()) {
        throw std::invalid_argument(
            "FastOnlineOperator: k_vectors and weights must have same size");
    }
    if (k_vectors_.size() != detector_indices_.size()) {
        throw std::invalid_argument(
            "FastOnlineOperator: k_vectors and detector_indices must have same size");
    }
    if (grid_size.x() <= 0 || grid_size.y() <= 0) {
        throw std::invalid_argument(
            "FastOnlineOperator: grid_size must be positive");
    }
    if (grid_spacing.x() <= 0 || grid_spacing.y() <= 0) {
        throw std::invalid_argument(
            "FastOnlineOperator: grid_spacing must be positive");
    }

    validate_config();
    setup_kernel_params();

    // Find max detector index to determine num_detectors_
    if (!detector_indices_.empty()) {
        num_detectors_ = *std::max_element(detector_indices_.begin(),
                                           detector_indices_.end()) + 1;
    } else {
        num_detectors_ = 1;  // No samples: a single-entry window suffices
    }

    compute_anti_gap_weights();
    initialize();
}

FastOnlineOperator::FastOnlineOperator(
    const aurora::shared::KSpaceSlice& slice,
    const Eigen::Vector2i& grid_size,
    const Eigen::Vector2d& grid_spacing,
    const FastOnlineConfig& config)
    : grid_size_(grid_size),
      grid_spacing_(grid_spacing),
      config_(config),
      num_detectors_(0),
      kernel_width_(0),
      gaussian_sigma_(1.0),
      kaiser_beta_(0.0),
      impl_(std::make_unique<Impl>()) {

    // Extract K-vectors, sample weights, and detector indices from slice
    k_vectors_.reserve(slice.samples_size());
    weights_.reserve(slice.samples_size());
    detector_indices_.reserve(slice.samples_size());

    for (const auto& sample : slice.samples()) {
        k_vectors_.emplace_back(sample.kx(), sample.ky());
        weights_.push_back(sample.weight());
        detector_indices_.push_back(sample.detector_index());
    }

    if (grid_size.x() <= 0 || grid_size.y() <= 0) {
        throw std::invalid_argument(
            "FastOnlineOperator: grid_size must be positive");
    }
    if (grid_spacing.x() <= 0 || grid_spacing.y() <= 0) {
        throw std::invalid_argument(
            "FastOnlineOperator: grid_spacing must be positive");
    }

    validate_config();
    setup_kernel_params();

    // Find max detector index to determine num_detectors_
    if (!detector_indices_.empty()) {
        num_detectors_ = *std::max_element(detector_indices_.begin(),
                                           detector_indices_.end()) + 1;
    } else {
        num_detectors_ = 1;  // No samples: a single-entry window suffices
    }

    compute_anti_gap_weights();
    initialize();
}

FastOnlineOperator::~FastOnlineOperator() = default;

FastOnlineOperator::FastOnlineOperator(FastOnlineOperator&&) noexcept = default;
FastOnlineOperator& FastOnlineOperator::operator=(FastOnlineOperator&&) noexcept = default;

// =============================================================================
// Initialization
// =============================================================================

void FastOnlineOperator::initialize() {
    // Create FFT plans
    impl_->fft_forward = std::make_unique<aurora::forward::Fft2d>(
        grid_size_.x(), grid_size_.y(), false);  // forward FFT
    impl_->fft_inverse = std::make_unique<aurora::forward::Fft2d>(
        grid_size_.x(), grid_size_.y(), true);   // inverse FFT
}

void FastOnlineOperator::validate_config() const {
    // Validate kernel
    if (config_.kernel() == GRIDDING_KERNEL_UNSPECIFIED) {
        throw std::invalid_argument(
            "FastOnlineConfig: kernel must be explicitly set");
    }
    if (config_.kernel_width() == 0) {
        throw std::invalid_argument(
            "FastOnlineConfig: kernel_width must be positive");
    }

    // Validate the K-sample axis window
    if (config_.anti_gap_window() == ANTI_GAP_WINDOW_UNSPECIFIED) {
        throw std::invalid_argument(
            "FastOnlineConfig: anti_gap_window must be explicitly set");
    }
    if (config_.anti_gap_window() == ANTI_GAP_WINDOW_TUKEY) {
        if (config_.anti_gap_tukey_alpha() < 0.0 ||
            config_.anti_gap_tukey_alpha() > 1.0) {
            throw std::invalid_argument(
                "FastOnlineConfig: anti_gap_tukey_alpha must be in [0, 1]");
        }
    }
    if (config_.anti_gap_window() == ANTI_GAP_WINDOW_KAISER) {
        if (config_.anti_gap_kaiser_beta() <= 0.0) {
            throw std::invalid_argument(
                "FastOnlineConfig: anti_gap_kaiser_beta must be positive");
        }
    }

    // Validate density normalization
    if (config_.density_normalization() == DENSITY_NORM_UNSPECIFIED) {
        throw std::invalid_argument(
            "FastOnlineConfig: density_normalization must be explicitly set");
    }
    if (config_.density_normalization() != DENSITY_NORM_NONE) {
        if (config_.density_epsilon() <= 0.0) {
            throw std::invalid_argument(
                "FastOnlineConfig: density_epsilon must be positive when normalization is enabled");
        }
    }

    // Validate post-processing
    if (config_.post_processing() == POST_PROC_UNSPECIFIED) {
        throw std::invalid_argument(
            "FastOnlineConfig: post_processing must be explicitly set");
    }
    if (config_.post_processing() == POST_PROC_LAPLACIAN_SHARPEN) {
        if (config_.laplacian_strength() < 0.0 ||
            config_.laplacian_strength() > 1.0) {
            throw std::invalid_argument(
                "FastOnlineConfig: laplacian_strength must be in [0, 1]");
        }
    }
    if (config_.post_processing() == POST_PROC_DIFFUSION) {
        if (config_.diffusion_coefficient() <= 0.0) {
            throw std::invalid_argument(
                "FastOnlineConfig: diffusion_coefficient must be positive");
        }
    }

    // Validate refinement
    if (config_.enable_refinement()) {
        if (config_.refinement_iterations() == 0) {
            throw std::invalid_argument(
                "FastOnlineConfig: refinement_iterations must be positive when refinement is enabled");
        }
        if (config_.refinement_alpha() <= 0.0) {
            throw std::invalid_argument(
                "FastOnlineConfig: refinement_alpha must be positive when refinement is enabled");
        }
        if (config_.refinement_lambda() < 0.0) {
            throw std::invalid_argument(
                "FastOnlineConfig: refinement_lambda must be non-negative when refinement is enabled");
        }
    }
}

void FastOnlineOperator::setup_kernel_params() {
    kernel_width_ = static_cast<int>(config_.kernel_width());

    // Gaussian sigma: use same fraction as GriddingOperator (0.5)
    gaussian_sigma_ = kernel_width_ * 0.5;

    // Kaiser-Bessel beta: auto-compute using Beatty formula
    double W = static_cast<double>(kernel_width_);
    double t = 1.0 - 1.0 / 3.0;
    double radicand = W * W * t * t - 0.8;
    kaiser_beta_ = (radicand > 0) ? M_PI * std::sqrt(radicand) : 2.34;
}

void FastOnlineOperator::compute_anti_gap_weights() {
    anti_gap_weights_.resize(num_detectors_);

    if (config_.anti_gap_window() == ANTI_GAP_WINDOW_NONE) {
        // No windowing: all weights are 1.0
        std::fill(anti_gap_weights_.begin(), anti_gap_weights_.end(), 1.0);
        return;
    }

    for (uint32_t j = 0; j < num_detectors_; ++j) {
        // Normalized position in [0, 1]
        double t = (num_detectors_ > 1)
            ? static_cast<double>(j) / (num_detectors_ - 1)
            : 0.5;

        switch (config_.anti_gap_window()) {
            case ANTI_GAP_WINDOW_HANN:
                // Hann: 0.5 * (1 - cos(2*pi*t))
                anti_gap_weights_[j] = 0.5 * (1.0 - std::cos(2.0 * M_PI * t));
                break;

            case ANTI_GAP_WINDOW_TUKEY: {
                // Tukey with alpha parameter
                double alpha = config_.anti_gap_tukey_alpha();
                if (alpha == 0.0) {
                    // Rectangular
                    anti_gap_weights_[j] = 1.0;
                } else if (t < alpha / 2.0) {
                    anti_gap_weights_[j] = 0.5 * (1.0 + std::cos(M_PI * (2.0 * t / alpha - 1.0)));
                } else if (t > 1.0 - alpha / 2.0) {
                    anti_gap_weights_[j] = 0.5 * (1.0 + std::cos(M_PI * (2.0 * t / alpha - 2.0 / alpha + 1.0)));
                } else {
                    anti_gap_weights_[j] = 1.0;
                }
                break;
            }

            case ANTI_GAP_WINDOW_KAISER: {
                // Kaiser: I0(beta * sqrt(1 - (2t-1)^2)) / I0(beta)
                double beta = config_.anti_gap_kaiser_beta();
                double x = 2.0 * t - 1.0;  // Map to [-1, 1]
                double arg_sq = 1.0 - x * x;
                if (arg_sq < 0.0) arg_sq = 0.0;  // Numerical safety
                double arg = beta * std::sqrt(arg_sq);
                anti_gap_weights_[j] = bessel_i0(arg) / bessel_i0(beta);
                break;
            }

            default:
                anti_gap_weights_[j] = 1.0;
                break;
        }
    }
}

// =============================================================================
// Kernel evaluation
// =============================================================================

double FastOnlineOperator::evaluate_kernel(double distance_sq) const {
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
            return (distance < 0.5) ? 1.0 : 0.0;
    }
}

double FastOnlineOperator::bessel_i0(double x) {
    // Polynomial approximation of I0(x) from Abramowitz & Stegun
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
// Core FORO operations
// =============================================================================

Eigen::VectorXcd FastOnlineOperator::apply_anti_gap_window(
    const Eigen::VectorXcd& k_samples) const {

    Eigen::VectorXcd windowed(k_samples.size());

    for (size_t m = 0; m < k_vectors_.size(); ++m) {
        uint32_t det_idx = detector_indices_[m];
        double weight = (det_idx < anti_gap_weights_.size())
            ? anti_gap_weights_[det_idx]
            : 1.0;
        windowed(m) = k_samples(m) * weight;
    }

    return windowed;
}

void FastOnlineOperator::grid_and_accumulate(
    const Eigen::VectorXcd& k_samples,
    Eigen::MatrixXcd& k_grid,
    Eigen::MatrixXd& density_map,
    const Eigen::Vector2d& grid_origin) const {

    int nx = grid_size_.x();
    int ny = grid_size_.y();

    k_grid = Eigen::MatrixXcd::Zero(ny, nx);
    density_map = Eigen::MatrixXd::Zero(ny, nx);

    // K-space grid spacing (from image grid via Fourier relationship)
    double dk_x = 2.0 * M_PI / (nx * grid_spacing_.x());
    double dk_y = 2.0 * M_PI / (ny * grid_spacing_.y());

    // K-space grid origin (centered)
    double k_origin_x = -dk_x * nx / 2.0;
    double k_origin_y = -dk_y * ny / 2.0;

    // Determine kernel radius
    bool use_nearest = (config_.kernel() == GRIDDING_KERNEL_NEAREST);
    int kernel_radius = use_nearest ? 0 : kernel_width_;

    // Single pass: grid each sample AND accumulate density
    for (size_t m = 0; m < k_vectors_.size(); ++m) {
        double kx = k_vectors_[m].x();
        double ky = k_vectors_[m].y();
        double w_i = weights_[m];
        std::complex<double> y_i = k_samples(m);

        // Find fractional grid position
        double fx = (kx - k_origin_x) / dk_x;
        double fy = (ky - k_origin_y) / dk_y;

        // Nearest grid point (center of kernel)
        int cx = static_cast<int>(std::round(fx));
        int cy = static_cast<int>(std::round(fy));

        if (kernel_radius == 0) {
            // Nearest-neighbor: single cell
            if (cx >= 0 && cx < nx && cy >= 0 && cy < ny) {
                k_grid(cy, cx) += w_i * y_i;
                density_map(cy, cx) += w_i;
            }
        } else {
            // Kernel spreading
            for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;

                for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= nx) continue;

                    // Distance from sample position to this grid cell
                    double dist_x = fx - ix;
                    double dist_y = fy - iy;
                    double dist_sq = dist_x * dist_x + dist_y * dist_y;

                    // Evaluate kernel weight
                    double K = evaluate_kernel(dist_sq);
                    if (K > 1e-10) {
                        // Accumulate complex K-grid: Y[k] += w_i * y_i * K
                        k_grid(iy, ix) += w_i * y_i * K;

                        // Accumulate density: D[k] += w_i * |K| or w_i * |K|^2
                        double density_contrib;
                        if (config_.density_normalization() == DENSITY_NORM_KERNEL_SQUARED) {
                            density_contrib = w_i * K * K;
                        } else {
                            density_contrib = w_i * std::abs(K);
                        }
                        density_map(iy, ix) += density_contrib;
                    }
                }
            }
        }
    }
}

void FastOnlineOperator::normalize_by_density(
    Eigen::MatrixXcd& k_grid,
    const Eigen::MatrixXd& density_map) const {

    if (config_.density_normalization() == DENSITY_NORM_NONE) {
        return;
    }

    double epsilon = config_.density_epsilon();
    int nx = k_grid.cols();
    int ny = k_grid.rows();

#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            double D = density_map(iy, ix);
            k_grid(iy, ix) /= (D + epsilon);
        }
    }
}

// =============================================================================
// Full reconstruction pipeline
// =============================================================================

Eigen::MatrixXcd FastOnlineOperator::reconstruct(
    const Eigen::VectorXcd& k_samples,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2i& grid_size) const {

    // Ensure runtime grid parameters match those used to construct the operator
    if (grid_size != grid_size_) {
        throw std::invalid_argument(
            "FastOnlineOperator::reconstruct: grid_size must match constructor grid_size");
    }
    if ((grid_spacing - grid_spacing_).norm() > 1e-12) {
        throw std::invalid_argument(
            "FastOnlineOperator::reconstruct: grid_spacing must match constructor grid_spacing");
    }

    // Step 1: Apply the K-sample axis window
    Eigen::VectorXcd windowed = apply_anti_gap_window(k_samples);

    // Step 2 & 3: Optional gap interpolation + gridding
    Eigen::MatrixXcd k_grid;
    Eigen::MatrixXd density_map;

    if (config_.enable_gap_interpolation()) {
        // Interpolate gaps and grid with extended K-vectors
        ExtendedKSpace extended = interpolate_gaps(windowed);
        grid_extended(extended, k_grid, density_map);
    } else {
        // Standard gridding with original K-vectors
        grid_and_accumulate(windowed, k_grid, density_map, grid_origin);
    }

    // Step 4: Normalize in K-space (online PSF correction)
    normalize_by_density(k_grid, density_map);

    // Step 5: FFT shift and IFFT
    int nx = grid_size.x();
    int ny = grid_size.y();
    int half_nx = nx / 2;
    int half_ny = ny / 2;

    // Inverse FFT shift (DC from center to corner)
    Eigen::MatrixXcd k_grid_shifted(ny, nx);
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

    // IFFT
    Eigen::MatrixXcd ifft_result = impl_->fft_inverse->execute(k_grid_shifted);

    // FFT shift (spatial origin from corner to center)
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

    // Step 6: Optional post-processing
    apply_post_processing(image);

    // Step 7: Optional iterative refinement
    if (config_.enable_refinement()) {
        for (uint32_t iter = 0; iter < config_.refinement_iterations(); ++iter) {
            image = refine(image, k_samples, grid_origin, grid_spacing);
        }
    }

    return image;
}

// =============================================================================
// Forward operator
// =============================================================================

Eigen::VectorXcd FastOnlineOperator::forward(
    const Eigen::MatrixXcd& scene,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing) const {

    int nx = grid_size_.x();
    int ny = grid_size_.y();

    // K-space grid spacing
    double dk_x = 2.0 * M_PI / (nx * grid_spacing.x());
    double dk_y = 2.0 * M_PI / (ny * grid_spacing.y());

    // K-space grid origin (centered)
    double k_origin_x = -dk_x * nx / 2.0;
    double k_origin_y = -dk_y * ny / 2.0;

    int half_nx = nx / 2;
    int half_ny = ny / 2;

    // Inverse FFT shift on spatial input
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

    // Apply FFT
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

    // Determine kernel radius
    bool use_nearest = (config_.kernel() == GRIDDING_KERNEL_NEAREST);
    int kernel_radius = use_nearest ? 0 : kernel_width_;

    // Interpolate at K-positions
    Eigen::VectorXcd result(k_vectors_.size());
    const int num_samples = static_cast<int>(k_vectors_.size());

#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < num_samples; ++m) {
        double kx = k_vectors_[m].x();
        double ky = k_vectors_[m].y();

        double fx = (kx - k_origin_x) / dk_x;
        double fy = (ky - k_origin_y) / dk_y;

        int cx = static_cast<int>(std::round(fx));
        int cy = static_cast<int>(std::round(fy));

        if (kernel_radius == 0) {
            if (cx >= 0 && cx < nx && cy >= 0 && cy < ny) {
                result(m) = weights_[m] * k_grid_shifted(cy, cx);
            } else {
                result(m) = std::complex<double>(0, 0);
            }
        } else {
            std::complex<double> sum(0, 0);
            double weight_sum = 0.0;

            for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;

                for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= nx) continue;

                    double dist_x = fx - ix;
                    double dist_y = fy - iy;
                    double dist_sq = dist_x * dist_x + dist_y * dist_y;

                    double w = evaluate_kernel(dist_sq);
                    if (w > 1e-10) {
                        sum += k_grid_shifted(iy, ix) * w;
                        weight_sum += w;
                    }
                }
            }

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

Eigen::MatrixXcd FastOnlineOperator::adjoint(
    const Eigen::VectorXcd& k_samples,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2i& grid_size) const {

    int nx = grid_size.x();
    int ny = grid_size.y();

    Eigen::MatrixXcd k_grid = Eigen::MatrixXcd::Zero(ny, nx);
    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(ny, nx);

    double dk_x = 2.0 * M_PI / (nx * grid_spacing.x());
    double dk_y = 2.0 * M_PI / (ny * grid_spacing.y());

    double k_origin_x = -dk_x * nx / 2.0;
    double k_origin_y = -dk_y * ny / 2.0;

    bool use_nearest = (config_.kernel() == GRIDDING_KERNEL_NEAREST);
    int kernel_radius = use_nearest ? 0 : kernel_width_;

    // Grid each sample with kernel spreading
    for (size_t m = 0; m < k_vectors_.size(); ++m) {
        double kx = k_vectors_[m].x();
        double ky = k_vectors_[m].y();
        std::complex<double> value = k_samples(m) * weights_[m];

        double fx = (kx - k_origin_x) / dk_x;
        double fy = (ky - k_origin_y) / dk_y;

        int cx = static_cast<int>(std::round(fx));
        int cy = static_cast<int>(std::round(fy));

        if (kernel_radius == 0) {
            if (cx >= 0 && cx < nx && cy >= 0 && cy < ny) {
                k_grid(cy, cx) += value;
                weights(cy, cx) += 1.0;
            }
        } else {
            for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;

                for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= nx) continue;

                    double dist_x = fx - ix;
                    double dist_y = fy - iy;
                    double dist_sq = dist_x * dist_x + dist_y * dist_y;

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

    // Inverse FFT shift
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

    // IFFT
    Eigen::MatrixXcd ifft_result = impl_->fft_inverse->execute(k_grid_shifted);

    // FFT shift
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

// =============================================================================
// Post-processing
// =============================================================================

void FastOnlineOperator::apply_post_processing(Eigen::MatrixXcd& image) const {
    switch (config_.post_processing()) {
        case POST_PROC_LAPLACIAN_SHARPEN:
            apply_laplacian_sharpen(image, config_.laplacian_strength());
            break;

        case POST_PROC_DIFFUSION:
            apply_diffusion(image, config_.diffusion_coefficient());
            break;

        case POST_PROC_NONE:
        default:
            // No post-processing
            break;
    }
}

void FastOnlineOperator::apply_laplacian_sharpen(
    Eigen::MatrixXcd& image, double strength) const {

    if (strength <= 0.0) return;

    // Compute Laplacian
    Eigen::MatrixXcd laplacian = compute_laplacian(image);

    // Unsharp filtering: image = image - strength * laplacian
    image -= strength * laplacian;
}

void FastOnlineOperator::apply_diffusion(
    Eigen::MatrixXcd& image, double coefficient) const {

    if (coefficient <= 0.0) return;

    int ny = image.rows();
    int nx = image.cols();

    // Simple 2D diffusion: one step of heat equation
    // u_{t+1} = u_t + dt * (u_xx + u_yy)
    // Using central differences with Neumann boundary conditions

    Eigen::MatrixXcd result = image;

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            // Neighbors with Neumann BC (reflect at boundaries)
            int ixm = (ix > 0) ? ix - 1 : ix + 1;
            int ixp = (ix < nx - 1) ? ix + 1 : ix - 1;
            int iym = (iy > 0) ? iy - 1 : iy + 1;
            int iyp = (iy < ny - 1) ? iy + 1 : iy - 1;

            // Laplacian (discrete)
            std::complex<double> lap =
                image(iy, ixm) + image(iy, ixp) +
                image(iym, ix) + image(iyp, ix) -
                4.0 * image(iy, ix);

            result(iy, ix) = image(iy, ix) + coefficient * lap;
        }
    }

    image = result;
}

Eigen::MatrixXcd FastOnlineOperator::compute_laplacian(
    const Eigen::MatrixXcd& image) const {

    int ny = image.rows();
    int nx = image.cols();

    Eigen::MatrixXcd laplacian(ny, nx);

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            // Neighbors with Neumann BC
            int ixm = (ix > 0) ? ix - 1 : ix + 1;
            int ixp = (ix < nx - 1) ? ix + 1 : ix - 1;
            int iym = (iy > 0) ? iy - 1 : iy + 1;
            int iyp = (iy < ny - 1) ? iy + 1 : iy - 1;

            laplacian(iy, ix) =
                image(iy, ixm) + image(iy, ixp) +
                image(iym, ix) + image(iyp, ix) -
                4.0 * image(iy, ix);
        }
    }

    return laplacian;
}

// =============================================================================
// Optional features
// =============================================================================

FastOnlineOperator::ExtendedKSpace FastOnlineOperator::interpolate_gaps(
    const Eigen::VectorXcd& k_samples) const {

    ExtendedKSpace result;

    // Copy original data
    result.k_vectors = k_vectors_;
    result.samples.resize(k_samples.size());
    for (Eigen::Index i = 0; i < k_samples.size(); ++i) {
        result.samples(i) = k_samples(i);
    }
    result.weights = weights_;

    if (num_detectors_ < 2 || k_vectors_.empty()) {
        return result;
    }

    // Group samples by approximate ky value (within tolerance)
    constexpr double ky_tolerance = 100.0;  // rad/m tolerance for grouping

    struct KyGroup {
        double ky_center;
        std::vector<size_t> indices;
    };
    std::vector<KyGroup> groups;

    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        double ky = k_vectors_[i].y();
        bool found = false;
        for (auto& g : groups) {
            if (std::abs(ky - g.ky_center) < ky_tolerance) {
                g.indices.push_back(i);
                g.ky_center = (g.ky_center * (g.indices.size() - 1) + ky) / g.indices.size();
                found = true;
                break;
            }
        }
        if (!found) {
            groups.push_back({ky, {i}});
        }
    }

    // K-space spacing estimate from grid
    double dk_x = 2.0 * M_PI / (grid_size_.x() * grid_spacing_.x());

    const int num_groups = static_cast<int>(groups.size());

    // Per-group results for parallel processing
    struct GroupResult {
        std::vector<Eigen::Vector2d> k_vectors;
        std::vector<std::complex<double>> samples;
        std::vector<double> weights;
    };
    std::vector<GroupResult> group_results(num_groups);

    // Process each ky group in parallel
#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int g = 0; g < num_groups; ++g) {
        const auto& group = groups[g];
        if (group.indices.size() < 2) continue;

        // Sort indices by kx
        std::vector<size_t> sorted_indices = group.indices;
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [this](size_t a, size_t b) {
                      return k_vectors_[a].x() < k_vectors_[b].x();
                  });

        auto& local = group_results[g];

        // Find pairs from adjacent detectors and interpolate
        for (size_t j = 0; j + 1 < sorted_indices.size(); ++j) {
            size_t idx_a = sorted_indices[j];
            size_t idx_b = sorted_indices[j + 1];

            uint32_t det_a = detector_indices_[idx_a];
            uint32_t det_b = detector_indices_[idx_b];

            // Only interpolate between adjacent detectors
            int det_diff = static_cast<int>(det_b) - static_cast<int>(det_a);
            if (std::abs(det_diff) != 1) continue;

            // Check if there's a significant gap in kx
            double kx_a = k_vectors_[idx_a].x();
            double kx_b = k_vectors_[idx_b].x();
            double kx_gap = kx_b - kx_a;

            // If gap is larger than ~1 grid cell, interpolate
            if (kx_gap > 1.0 * dk_x) {
                // Densely fill the gap (one sample per grid cell)
                int num_interp = std::max(1, static_cast<int>(kx_gap / dk_x));
                num_interp = std::min(num_interp, 10);  // Allow more points

                std::complex<double> val_a = k_samples(idx_a);
                std::complex<double> val_b = k_samples(idx_b);

                // Phase-aware interpolation
                double mag_a = std::abs(val_a);
                double mag_b = std::abs(val_b);
                double phase_a = std::arg(val_a);
                double phase_b = std::arg(val_b);

                // Unwrap phase
                double phase_diff = phase_b - phase_a;
                while (phase_diff > M_PI) phase_diff -= 2.0 * M_PI;
                while (phase_diff < -M_PI) phase_diff += 2.0 * M_PI;
                phase_b = phase_a + phase_diff;

                double ky_avg = (k_vectors_[idx_a].y() + k_vectors_[idx_b].y()) / 2.0;
                double weight_avg = (weights_[idx_a] + weights_[idx_b]) / 2.0;

                for (int k = 1; k <= num_interp; ++k) {
                    double t = static_cast<double>(k) / (num_interp + 1);

                    double kx_interp = kx_a + t * kx_gap;
                    double mag_interp = mag_a + t * (mag_b - mag_a);
                    double phase_interp = phase_a + t * (phase_b - phase_a);

                    local.k_vectors.emplace_back(kx_interp, ky_avg);
                    local.samples.push_back(std::polar(mag_interp, phase_interp));
                    local.weights.push_back(weight_avg);
                }
            }
        }
    }

    // Merge results from all groups
    std::vector<Eigen::Vector2d> new_k_vectors;
    std::vector<std::complex<double>> new_samples;
    std::vector<double> new_weights;

    // Calculate total size for reservation
    size_t total_new = 0;
    for (const auto& gr : group_results) {
        total_new += gr.samples.size();
    }
    new_k_vectors.reserve(total_new);
    new_samples.reserve(total_new);
    new_weights.reserve(total_new);

    for (const auto& gr : group_results) {
        new_k_vectors.insert(new_k_vectors.end(), gr.k_vectors.begin(), gr.k_vectors.end());
        new_samples.insert(new_samples.end(), gr.samples.begin(), gr.samples.end());
        new_weights.insert(new_weights.end(), gr.weights.begin(), gr.weights.end());
    }

    // Append new samples to result
    if (!new_samples.empty()) {
        size_t orig_size = result.k_vectors.size();
        size_t new_size = orig_size + new_samples.size();

        result.k_vectors.reserve(new_size);
        result.weights.reserve(new_size);

        Eigen::VectorXcd extended_samples(new_size);
        for (size_t i = 0; i < orig_size; ++i) {
            extended_samples(i) = result.samples(i);
        }

        for (size_t i = 0; i < new_samples.size(); ++i) {
            result.k_vectors.push_back(new_k_vectors[i]);
            result.weights.push_back(new_weights[i]);
            extended_samples(orig_size + i) = new_samples[i];
        }

        result.samples = extended_samples;
    }

    return result;
}

void FastOnlineOperator::grid_extended(
    const ExtendedKSpace& extended,
    Eigen::MatrixXcd& k_grid,
    Eigen::MatrixXd& density_map) const {

    int nx = grid_size_.x();
    int ny = grid_size_.y();

    k_grid = Eigen::MatrixXcd::Zero(ny, nx);
    density_map = Eigen::MatrixXd::Zero(ny, nx);

    // K-space grid spacing
    double dk_x = 2.0 * M_PI / (nx * grid_spacing_.x());
    double dk_y = 2.0 * M_PI / (ny * grid_spacing_.y());

    // K-space grid origin (centered)
    double k_origin_x = -dk_x * nx / 2.0;
    double k_origin_y = -dk_y * ny / 2.0;

    bool use_nearest = (config_.kernel() == GRIDDING_KERNEL_NEAREST);
    int kernel_radius = use_nearest ? 0 : kernel_width_;

    // Grid each sample
    for (size_t m = 0; m < extended.k_vectors.size(); ++m) {
        double kx = extended.k_vectors[m].x();
        double ky = extended.k_vectors[m].y();
        double w_i = extended.weights[m];
        std::complex<double> y_i = extended.samples(m);

        double fx = (kx - k_origin_x) / dk_x;
        double fy = (ky - k_origin_y) / dk_y;

        int cx = static_cast<int>(std::round(fx));
        int cy = static_cast<int>(std::round(fy));

        if (kernel_radius == 0) {
            if (cx >= 0 && cx < nx && cy >= 0 && cy < ny) {
                k_grid(cy, cx) += w_i * y_i;
                density_map(cy, cx) += w_i;
            }
        } else {
            for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
                int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;

                for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= nx) continue;

                    double dist_x = fx - ix;
                    double dist_y = fy - iy;
                    double dist_sq = dist_x * dist_x + dist_y * dist_y;

                    double K = evaluate_kernel(dist_sq);
                    if (K > 1e-10) {
                        k_grid(iy, ix) += w_i * y_i * K;

                        // Match density accumulation from grid_and_accumulate()
                        double density_contrib;
                        if (config_.density_normalization() == DENSITY_NORM_KERNEL_SQUARED) {
                            density_contrib = w_i * K * K;
                        } else {
                            density_contrib = w_i * std::abs(K);
                        }
                        density_map(iy, ix) += density_contrib;
                    }
                }
            }
        }
    }
}

Eigen::MatrixXcd FastOnlineOperator::refine(
    const Eigen::MatrixXcd& x0,
    const Eigen::VectorXcd& k_samples,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing) const {

    // Fixed-point refinement:
    // x_{t+1} = x_t + alpha * (A^H * W * (y - A*x_t) - lambda * L^H*L*x_t)

    double alpha = config_.refinement_alpha();
    double lambda = config_.refinement_lambda();

    // Compute residual: y - A*x_t
    Eigen::VectorXcd Ax = forward(x0, grid_origin, grid_spacing);
    Eigen::VectorXcd residual = k_samples - Ax;

    // Apply sample weights to residual (W * residual)
    Eigen::VectorXcd weighted_residual(k_samples.size());
    for (size_t m = 0; m < k_vectors_.size(); ++m) {
        weighted_residual(m) = weights_[m] * residual(m);
    }

    // Apply A^H to weighted residual
    Eigen::MatrixXcd gradient = adjoint(weighted_residual, grid_origin,
                                        grid_spacing, grid_size_);

    // Add regularization term: -lambda * L^H*L*x_t
    if (lambda > 0.0) {
        Eigen::MatrixXcd laplacian = compute_laplacian(x0);
        gradient -= lambda * laplacian;
    }

    // Update: x_{t+1} = x_t + alpha * gradient
    return x0 + alpha * gradient;
}

}  // namespace aurora::reconstruction
