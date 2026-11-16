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

#include "reconstruction/bandwidth_weight_calculator.h"

#include <algorithm>
#include <cmath>

#include "forward/bandwidth_model.h"
#include "forward/forward_model.h"
#include "forward/measurement_operator.h"

namespace aurora::reconstruction {

BandwidthWeightCalculator::BandwidthWeightCalculator(
    const aurora::shared::KSpaceSlice& slice,
    const aurora::forward::ForwardModel& forward_model)
    : wavelength_m_(slice.wavelength_m()),
      focal_length_m_(forward_model.optics().fourier_lens_focal_m()) {

    const int num = slice.samples_size();
    k_centers_.reserve(num);
    kx_bandwidths_.reserve(num);
    ky_bandwidths_.reserve(num);
    detector_indices_.reserve(num);

    for (int i = 0; i < num; ++i) {
        const auto& sample = slice.samples(i);
        const uint32_t det_idx = sample.detector_index();

        // Window centered on the sample's K-position, with the K-space
        // bandwidth of the detector pixel that acquired it
        const auto params = aurora::forward::build_window_params(
            forward_model, det_idx, wavelength_m_);
        k_centers_.emplace_back(sample.kx(), sample.ky());
        kx_bandwidths_.push_back(params.bandwidth_x);
        ky_bandwidths_.push_back(params.bandwidth_y);
        detector_indices_.push_back(det_idx);

        if (params.bandwidth_x > 0 && params.bandwidth_y > 0) {
            has_bandwidth_ = true;
        }
    }
}

Eigen::Vector2d BandwidthWeightCalculator::average_bandwidth() const {
    if (kx_bandwidths_.empty()) return {0, 0};

    double sum_x = 0, sum_y = 0;
    int count = 0;
    for (size_t i = 0; i < kx_bandwidths_.size(); ++i) {
        if (kx_bandwidths_[i] > 0 && ky_bandwidths_[i] > 0) {
            sum_x += kx_bandwidths_[i];
            sum_y += ky_bandwidths_[i];
            count++;
        }
    }
    if (count == 0) return {0, 0};
    return {sum_x / count, sum_y / count};
}

Eigen::Vector2d BandwidthWeightCalculator::resolution_limit() const {
    auto avg_bw = average_bandwidth();
    return aurora::forward::BandwidthModel::resolution_limit(avg_bw.x(), avg_bw.y());
}

bool BandwidthWeightCalculator::is_grid_valid(const Eigen::Vector2d& grid_spacing,
                                               double tolerance_factor) const {
    auto limit = resolution_limit();
    if (limit.x() <= 0 || limit.y() <= 0) return true;

    return grid_spacing.x() <= limit.x() * tolerance_factor &&
           grid_spacing.y() <= limit.y() * tolerance_factor;
}

SparseWeights BandwidthWeightCalculator::compute_weights(
    const Eigen::Vector2i& grid_size,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2d& grid_origin,
    aurora::forward::BandwidthWindowType window_type,
    double weight_threshold) const {

    using aurora::forward::MeasurementOperator;

    SparseWeights result;
    if (!has_bandwidth_) return result;

    const int Nx = grid_size.x();
    const int Ny = grid_size.y();
    const size_t M = k_centers_.size();

    result.sample_to_pixels.resize(M);
    result.pixel_to_samples.resize(static_cast<size_t>(Nx) * static_cast<size_t>(Ny));

    // Spatial-to-K conversion factor: k = 2π * x / (f * λ)
    const double k_scale = 2.0 * M_PI / (focal_length_m_ * wavelength_m_);

    for (size_t m = 0; m < M; ++m) {
        if (kx_bandwidths_[m] <= 0 || ky_bandwidths_[m] <= 0) continue;

        const auto params = get_window_params(m);

        // Compute K-space support bounds
        auto support = MeasurementOperator::compute_support(params, weight_threshold, window_type);
        if (!support.is_valid()) continue;

        // Convert K-space bounds to spatial coordinates
        double x_min = support.k_min_x / k_scale;
        double x_max = support.k_max_x / k_scale;
        double y_min = support.k_min_y / k_scale;
        double y_max = support.k_max_y / k_scale;

        // Convert spatial bounds to pixel indices
        int ix_min = std::max(0, static_cast<int>(
            std::floor((x_min - grid_origin.x()) / grid_spacing.x())));
        int ix_max = std::min(Nx - 1, static_cast<int>(
            std::ceil((x_max - grid_origin.x()) / grid_spacing.x())));
        int iy_min = std::max(0, static_cast<int>(
            std::floor((y_min - grid_origin.y()) / grid_spacing.y())));
        int iy_max = std::min(Ny - 1, static_cast<int>(
            std::ceil((y_max - grid_origin.y()) / grid_spacing.y())));

        for (int iy = iy_min; iy <= iy_max; ++iy) {
            double ry = grid_origin.y() + iy * grid_spacing.y();
            for (int ix = ix_min; ix <= ix_max; ++ix) {
                double rx = grid_origin.x() + ix * grid_spacing.x();

                // Convert spatial position to K-space
                Eigen::Vector2d k_pixel(rx * k_scale, ry * k_scale);

                double w = MeasurementOperator::weight(k_pixel, params, window_type);
                if (w > weight_threshold) {
                    result.sample_to_pixels[m].push_back({ix, iy, w});
                    size_t pixel_idx = static_cast<size_t>(iy) * static_cast<size_t>(Nx) +
                                       static_cast<size_t>(ix);
                    result.pixel_to_samples[pixel_idx].push_back({m, w});
                }
            }
        }
    }

    return result;
}

double SparseWeights::average_sparsity() const {
    if (sample_to_pixels.empty()) return 0.0;

    size_t total = 0;
    for (const auto& pixels : sample_to_pixels) {
        total += pixels.size();
    }
    return static_cast<double>(total) / static_cast<double>(sample_to_pixels.size());
}

}  // namespace aurora::reconstruction
