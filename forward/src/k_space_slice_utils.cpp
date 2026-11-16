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

#include "forward/k_space_slice_utils.h"

#include <cmath>
#include <limits>

namespace aurora::forward {

Eigen::VectorXcd to_eigen_samples(const aurora::shared::KSpaceSlice& slice) {
    const int n = slice.samples_size();
    Eigen::VectorXcd result(n);

    for (int i = 0; i < n; ++i) {
        const auto& sample = slice.samples(i);
        result(i) = std::complex<double>(
            sample.value().real(),
            sample.value().imag());
    }

    return result;
}

std::vector<Eigen::Vector2d> to_k_vectors(const aurora::shared::KSpaceSlice& slice) {
    std::vector<Eigen::Vector2d> result;
    result.reserve(slice.samples_size());

    for (const auto& sample : slice.samples()) {
        result.emplace_back(sample.kx(), sample.ky());
    }

    return result;
}

std::vector<double> to_weights(const aurora::shared::KSpaceSlice& slice) {
    std::vector<double> result;
    result.reserve(slice.samples_size());

    for (const auto& sample : slice.samples()) {
        result.push_back(sample.weight());
    }

    return result;
}

std::tuple<double, double, double, double> compute_k_bounds(
    const aurora::shared::KSpaceSlice& slice) {

    if (slice.samples_size() == 0) {
        return {0.0, 0.0, 0.0, 0.0};
    }

    double kx_min = std::numeric_limits<double>::max();
    double kx_max = std::numeric_limits<double>::lowest();
    double ky_min = std::numeric_limits<double>::max();
    double ky_max = std::numeric_limits<double>::lowest();

    for (const auto& sample : slice.samples()) {
        kx_min = std::min(kx_min, sample.kx());
        kx_max = std::max(kx_max, sample.kx());
        ky_min = std::min(ky_min, sample.ky());
        ky_max = std::max(ky_max, sample.ky());
    }

    return {kx_min, kx_max, ky_min, ky_max};
}

Eigen::Vector2i suggest_grid_size(
    const aurora::shared::KSpaceSlice& slice,
    double oversampling) {

    if (slice.samples_size() == 0) {
        return {0, 0};
    }

    auto [kx_min, kx_max, ky_min, ky_max] = compute_k_bounds(slice);

    const double kx_extent = kx_max - kx_min;
    const double ky_extent = ky_max - ky_min;

    // Grid size based on K-space extent:
    // FOV = 2*pi / dk, so n = k_extent / dk = k_extent * FOV / (2*pi)
    // With oversampling, n = k_extent / (2*pi) * oversampling
    constexpr double kTwoPi = 2.0 * M_PI;

    int nx = static_cast<int>(std::ceil(kx_extent / kTwoPi * oversampling));
    int ny = static_cast<int>(std::ceil(ky_extent / kTwoPi * oversampling));

    // Ensure minimum grid size of 1
    nx = std::max(1, nx);
    ny = std::max(1, ny);

    return {nx, ny};
}

aurora::shared::KSpaceGridHints compute_grid_hints(
    const aurora::shared::KSpaceSlice& slice,
    double oversampling) {

    aurora::shared::KSpaceGridHints hints;

    auto [kx_min, kx_max, ky_min, ky_max] = compute_k_bounds(slice);
    hints.set_kx_min(kx_min);
    hints.set_kx_max(kx_max);
    hints.set_ky_min(ky_min);
    hints.set_ky_max(ky_max);

    auto grid_size = suggest_grid_size(slice, oversampling);
    hints.set_suggested_nx(static_cast<uint32_t>(grid_size.x()));
    hints.set_suggested_ny(static_cast<uint32_t>(grid_size.y()));

    return hints;
}

}  // namespace aurora::forward
