#include "reconstruction/direct_operator.h"

#include <cmath>
#include <stdexcept>

#include "forward/forward_model.h"
#include "forward/measurement_operator.h"

#ifdef AURORA_USE_OPENMP
#include <omp.h>
#endif

namespace aurora::reconstruction {

DirectOperator::DirectOperator(
    const aurora::shared::KSpaceSlice& slice,
    const aurora::forward::ForwardModel& forward_model,
    const aurora::forward::BandwidthConfig& bandwidth_config,
    const Eigen::Vector2i& grid_size,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2d& grid_origin)
    : grid_size_(grid_size),
      grid_spacing_(grid_spacing),
      grid_origin_(grid_origin),
      window_type_(bandwidth_config.window_type()),
      quadrature_order_(static_cast<int>(bandwidth_config.quadrature_order())),
      bandwidth_calc_(slice, forward_model) {

    if (quadrature_order_ < 1) {
        throw std::invalid_argument(
            "DirectOperator: bandwidth_config.quadrature_order must be >= 1");
    }

    if (window_type_ == aurora::forward::BANDWIDTH_WINDOW_UNSPECIFIED) {
        throw std::invalid_argument(
            "DirectOperator: bandwidth_config.window_type must be explicitly specified");
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

    if (bandwidth_calc_.has_bandwidth()) {
        // Validate grid spacing
        if (!bandwidth_calc_.is_grid_valid(grid_spacing)) {
            throw std::invalid_argument(
                "DirectOperator: grid spacing is too coarse for bandwidth-limited resolution");
        }
    }
}

Eigen::VectorXcd DirectOperator::forward(
    const Eigen::MatrixXcd& scene,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing) const {

    using aurora::forward::MeasurementOperator;

    const int Nx = static_cast<int>(scene.cols());
    const int Ny = static_cast<int>(scene.rows());
    const int M = static_cast<int>(k_vectors_.size());

    Eigen::VectorXcd result(M);

    // Parallelize over K-samples (each is independent)
#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        const double w_m = weights_[m];
        std::complex<double> sum{0.0, 0.0};

        // K-space window of this sample
        auto params = bandwidth_calc_.get_window_params(m);
        auto bounds = MeasurementOperator::compute_support(params, 1e-6, window_type_);

        if (!bounds.is_valid()) {
            // No valid support - use point sample at k_m
            const Eigen::Vector2d& k = k_vectors_[m];
            for (int iy = 0; iy < Ny; ++iy) {
                const double ry = grid_origin.y() + iy * grid_spacing.y();
                for (int ix = 0; ix < Nx; ++ix) {
                    const double rx = grid_origin.x() + ix * grid_spacing.x();
                    const double phase = -k.x() * rx - k.y() * ry;
                    sum += scene(iy, ix) * std::polar(1.0, phase);
                }
            }
        } else {
            // Quadrature over K-space support:
            // y_m = Σ_q W_m(k_q) · X(k_q) · Δk
            // where X(k_q) = Σ_{i,j} x(r_{i,j}) exp(-i k_q · r_{i,j})
            const double dk_x = (bounds.k_max_x - bounds.k_min_x) / quadrature_order_;
            const double dk_y = (bounds.k_max_y - bounds.k_min_y) / quadrature_order_;
            const double dk_area = dk_x * dk_y;

            for (int qi = 0; qi < quadrature_order_; ++qi) {
                const double kx = bounds.k_min_x + (qi + 0.5) * dk_x;
                for (int qj = 0; qj < quadrature_order_; ++qj) {
                    const double ky = bounds.k_min_y + (qj + 0.5) * dk_y;
                    Eigen::Vector2d k_q{kx, ky};

                    double w = MeasurementOperator::weight(k_q, params, window_type_);
                    if (w < 1e-10) continue;

                    // Compute X(k_q) = Σ x(r) exp(-i k_q · r)
                    std::complex<double> X_kq{0.0, 0.0};
                    for (int iy = 0; iy < Ny; ++iy) {
                        const double ry = grid_origin.y() + iy * grid_spacing.y();
                        for (int ix = 0; ix < Nx; ++ix) {
                            const double rx = grid_origin.x() + ix * grid_spacing.x();
                            const double phase = -kx * rx - ky * ry;
                            X_kq += scene(iy, ix) * std::polar(1.0, phase);
                        }
                    }

                    sum += w * X_kq * dk_area;
                }
            }
        }

        // Apply sample weight
        result(m) = w_m * sum;
    }

    return result;
}

Eigen::MatrixXcd DirectOperator::adjoint(
    const Eigen::VectorXcd& k_samples,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2i& grid_size) const {

    using aurora::forward::MeasurementOperator;

    const int Nx = grid_size.x();
    const int Ny = grid_size.y();
    const int M = static_cast<int>(k_samples.size());

    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(Ny, Nx);

    // Parallelize over pixels (each pixel accumulates all K-samples independently)
#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int iy = 0; iy < Ny; ++iy) {
        const double ry = grid_origin.y() + iy * grid_spacing.y();
        for (int ix = 0; ix < Nx; ++ix) {
            const double rx = grid_origin.x() + ix * grid_spacing.x();
            std::complex<double> pixel_sum{0.0, 0.0};

            // Sum contributions from all K-samples
            for (int m = 0; m < M; ++m) {
                const double w_m = weights_[m];
                const std::complex<double>& y = k_samples(m);

                // K-space window of this sample
                auto params = bandwidth_calc_.get_window_params(m);
                auto bounds = MeasurementOperator::compute_support(params, 1e-6, window_type_);

                if (!bounds.is_valid()) {
                    // No valid support - use point sample at k_m
                    const Eigen::Vector2d& k = k_vectors_[m];
                    const double phase = k.x() * rx + k.y() * ry;
                    pixel_sum += w_m * y * std::polar(1.0, phase);
                } else {
                    // Quadrature-based adjoint:
                    // x̂(r) += Σ_q W_m(k_q) · y_m · exp(+i k_q · r) · Δk
                    const double dk_x = (bounds.k_max_x - bounds.k_min_x) / quadrature_order_;
                    const double dk_y = (bounds.k_max_y - bounds.k_min_y) / quadrature_order_;
                    const double dk_area = dk_x * dk_y;

                    for (int qi = 0; qi < quadrature_order_; ++qi) {
                        const double kx = bounds.k_min_x + (qi + 0.5) * dk_x;
                        for (int qj = 0; qj < quadrature_order_; ++qj) {
                            const double ky = bounds.k_min_y + (qj + 0.5) * dk_y;
                            Eigen::Vector2d k_q{kx, ky};

                            double w = MeasurementOperator::weight(k_q, params, window_type_);
                            if (w < 1e-10) continue;

                            // Adjoint: exp(+i k_q · r)
                            const double phase = kx * rx + ky * ry;
                            pixel_sum += w_m * w * y * std::polar(1.0, phase) * dk_area;
                        }
                    }
                }
            }

            result(iy, ix) = pixel_sum;
        }
    }

    return result;
}

}  // namespace aurora::reconstruction
