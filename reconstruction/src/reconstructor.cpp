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

#include "reconstruction/reconstructor.h"

#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>

#ifdef AURORA_USE_OPENMP
#include <omp.h>
#endif

#include "forward/k_space_slice_utils.h"
#include "reconstruction/k_space_preprocessor.h"
#include "reconstruction/reconstruction_error.h"

namespace aurora::reconstruction {

namespace {

constexpr uint32_t kDefaultGridSize = 64;
constexpr double kDefaultPixelSize = 0.001;  // 1 mm
constexpr double kSparsityThreshold = 0.01;   // 1% of max for sparsity calc
constexpr double kSignalCullThreshold = 1e-12;  // Skip bins below this total energy

/// Compute total signal energy in a K-space slice for culling decision
double compute_slice_energy(const aurora::shared::KSpaceSlice& slice) {
    double energy = 0.0;
    for (const auto& sample : slice.samples()) {
        double re = sample.value().real();
        double im = sample.value().imag();
        energy += re * re + im * im;
    }
    return energy;
}

/// Maximum |kx| and |ky| over the slice samples (for apodization normalization)
std::pair<double, double> compute_k_extent(const aurora::shared::KSpaceSlice& slice) {
    double k_max_x = 0.0, k_max_y = 0.0;
    for (const auto& sample : slice.samples()) {
        k_max_x = std::max(k_max_x, std::abs(sample.kx()));
        k_max_y = std::max(k_max_y, std::abs(sample.ky()));
    }
    return {k_max_x, k_max_y};
}

}  // namespace

Reconstructor::Reconstructor(
    const ReconstructionConfig& config,
    const aurora::forward::ForwardModel& forward_model)
    : config_(config),
      forward_model_(forward_model),
      algorithm_(config.algorithm()) {

    // Validate bandwidth_config is set
    if (!config.has_bandwidth_config()) {
        throw ReconstructionError(
            ReconstructionError::Code::CONFIGURATION_ERROR,
            "ReconstructionConfig.bandwidth_config must be set");
    }
    if (config.bandwidth_config().window_type() ==
            aurora::forward::BANDWIDTH_WINDOW_UNSPECIFIED) {
        throw ReconstructionError(
            ReconstructionError::Code::CONFIGURATION_ERROR,
            "ReconstructionConfig.bandwidth_config.window_type must be explicitly set");
    }
    if (config.bandwidth_config().quadrature_order() == 0) {
        throw ReconstructionError(
            ReconstructionError::Code::CONFIGURATION_ERROR,
            "ReconstructionConfig.bandwidth_config.quadrature_order must be >= 1");
    }

    validate_algorithm_config();

    // Set up grid parameters
    uint32_t nx = config.grid_size_x() > 0 ? config.grid_size_x() : kDefaultGridSize;
    uint32_t ny = config.grid_size_y() > 0 ? config.grid_size_y() : kDefaultGridSize;
    grid_size_ = Eigen::Vector2i(nx, ny);

    double pixel = config.pixel_size_m() > 0 ? config.pixel_size_m() : kDefaultPixelSize;
    grid_spacing_ = Eigen::Vector2d(pixel, pixel);

    // Origin centered on grid
    grid_origin_ = Eigen::Vector2d(
        -pixel * nx / 2.0,
        -pixel * ny / 2.0);
}

void Reconstructor::validate_algorithm_config() const {
    switch (algorithm_) {
        case ALGORITHM_DIRECT_ADJOINT:
            // direct_adjoint_config is checked when the slice is reconstructed
            break;

        case ALGORITHM_GRIDDING_IFFT:
            if (!config_.has_gridding_config()) {
                throw ReconstructionError(
                    ReconstructionError::Code::CONFIGURATION_ERROR,
                    "ALGORITHM_GRIDDING_IFFT requires gridding_config to be set");
            }
            if (config_.gridding_config().kernel() == GRIDDING_KERNEL_UNSPECIFIED) {
                throw ReconstructionError(
                    ReconstructionError::Code::CONFIGURATION_ERROR,
                    "gridding_config.kernel must be explicitly set (not UNSPECIFIED)");
            }
            break;

        case ALGORITHM_NUFFT_ADJOINT:
            if (!config_.has_nufft_config()) {
                throw ReconstructionError(
                    ReconstructionError::Code::CONFIGURATION_ERROR,
                    "ALGORITHM_NUFFT_ADJOINT requires nufft_config to be set");
            }
            break;

        case ALGORITHM_FAST_ONLINE:
            if (!config_.has_fast_online_config()) {
                throw ReconstructionError(
                    ReconstructionError::Code::CONFIGURATION_ERROR,
                    "ALGORITHM_FAST_ONLINE requires fast_online_config to be set");
            }
            // Additional validation is done in FastOnlineOperator constructor
            break;

        case ALGORITHM_PCG_REGULARIZED:
            if (!config_.has_pcg_config()) {
                throw ReconstructionError(
                    ReconstructionError::Code::CONFIGURATION_ERROR,
                    "ALGORITHM_PCG_REGULARIZED requires pcg_config to be set");
            }
            // Additional validation is done in PcgOperator via validate_pcg_config()
            break;

        default:
            throw ReconstructionError(
                ReconstructionError::Code::CONFIGURATION_ERROR,
                "Unsupported reconstruction algorithm value " +
                std::to_string(static_cast<int>(algorithm_)) +
                ". Supported: DIRECT_ADJOINT, GRIDDING_IFFT, NUFFT_ADJOINT, "
                "FAST_ONLINE, PCG_REGULARIZED");
    }
}

ReconstructedScene Reconstructor::reconstruct(
    const aurora::shared::MeasurementTensor& measurements) {

    auto start_time = std::chrono::steady_clock::now();

    // Extract wavelengths and range bins from the burst
    auto wavelengths = extract_wavelengths(measurements);
    uint32_t num_range_bins = get_num_range_bins(measurements);

    if (wavelengths.empty() || num_range_bins == 0) {
        throw ReconstructionError(
            ReconstructionError::Code::INVALID_INPUT,
            "No valid K-space samples in measurements");
    }

    // Reconstruct each wavelength and range bin
    // Process range bins in parallel for each wavelength
    std::vector<std::vector<Eigen::MatrixXcd>> wavelength_images(wavelengths.size());

    for (size_t wl_idx = 0; wl_idx < wavelengths.size(); ++wl_idx) {
        double wavelength_m = wavelengths[wl_idx];

        // Pre-allocate output images for this wavelength
        wavelength_images[wl_idx].resize(num_range_bins);

        // Process range bins in parallel
        const int num_bins = static_cast<int>(num_range_bins);

#ifdef AURORA_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int rb = 0; rb < num_bins; ++rb) {
            // Extract K-space slice for this range bin and wavelength
            auto slice = forward_model_.extract_kspace_slice(
                measurements, static_cast<uint32_t>(rb), wavelength_m);

            // Skip if no samples or negligible signal energy - output zero image
            if (slice.samples().empty() ||
                compute_slice_energy(slice) < kSignalCullThreshold) {
                wavelength_images[wl_idx][rb] =
                    Eigen::MatrixXcd::Zero(grid_size_.y(), grid_size_.x());
                continue;
            }

            wavelength_images[wl_idx][rb] = reconstruct_slice(slice);
        }
    }

    // Build output protobuf
    return build_scene(wavelength_images, wavelengths,
                       measurements.range_axis(), start_time);
}

Eigen::MatrixXcd Reconstructor::reconstruct_range_bin(
    const aurora::shared::MeasurementTensor& measurements,
    uint32_t range_bin,
    double wavelength_m) {

    auto slice = forward_model_.extract_kspace_slice(
        measurements, range_bin, wavelength_m);

    if (slice.samples().empty()) {
        return Eigen::MatrixXcd::Zero(grid_size_.y(), grid_size_.x());
    }

    return reconstruct_slice(slice);
}

Eigen::MatrixXcd Reconstructor::reconstruct_slice(
    const aurora::shared::KSpaceSlice& slice) const {

    // Operators handle the per-pixel K-space bandwidth via MeasurementOperator
    switch (algorithm_) {
        case ALGORITHM_DIRECT_ADJOINT:
            return adjoint_reconstruct(slice);
        case ALGORITHM_GRIDDING_IFFT:
            return gridding_reconstruct(slice);
        case ALGORITHM_NUFFT_ADJOINT:
            return nufft_reconstruct(slice);
        case ALGORITHM_FAST_ONLINE:
            return fast_online_reconstruct(slice);
        case ALGORITHM_PCG_REGULARIZED:
            return pcg_reconstruct(slice);
        default:
            throw ReconstructionError(
                ReconstructionError::Code::CONFIGURATION_ERROR,
                "reconstruct_slice: unsupported algorithm value " +
                std::to_string(static_cast<int>(algorithm_)));
    }
}

Eigen::MatrixXcd Reconstructor::adjoint_reconstruct(
    const aurora::shared::KSpaceSlice& slice) const {

    if (!config_.has_direct_adjoint_config()) {
        throw ReconstructionError(
            ReconstructionError::Code::CONFIGURATION_ERROR,
            "ALGORITHM_DIRECT_ADJOINT requires direct_adjoint_config to be set");
    }
    KSpacePreprocessor preprocessor(config_.direct_adjoint_config());

    // Apply preprocessing (density taper, apodization, DCF)
    const auto [k_max_x, k_max_y] = compute_k_extent(slice);
    aurora::shared::KSpaceSlice processed_slice = preprocessor.apply(slice, k_max_x, k_max_y);

    // Create DirectOperator with MeasurementOperator physics
    DirectOperator op(
        processed_slice,
        forward_model_,
        config_.bandwidth_config(),
        grid_size_,
        grid_spacing_,
        grid_origin_);

    return op.adjoint(aurora::forward::to_eigen_samples(processed_slice),
                      grid_origin_, grid_spacing_, grid_size_);
}

Eigen::MatrixXcd Reconstructor::gridding_reconstruct(
    const aurora::shared::KSpaceSlice& slice) const {

    if (!config_.has_direct_adjoint_config()) {
        throw ReconstructionError(
            ReconstructionError::Code::CONFIGURATION_ERROR,
            "ALGORITHM_GRIDDING_IFFT requires direct_adjoint_config to be set");
    }
    KSpacePreprocessor preprocessor(config_.direct_adjoint_config());

    // Apply preprocessing (density taper, apodization, DCF)
    const auto [k_max_x, k_max_y] = compute_k_extent(slice);
    aurora::shared::KSpaceSlice processed_slice = preprocessor.apply(slice, k_max_x, k_max_y);

    // gridding_config is required and validated in the constructor
    GriddingOperator op(processed_slice, grid_size_, grid_spacing_,
                        config_.gridding_config());

    return op.adjoint(aurora::forward::to_eigen_samples(processed_slice),
                      grid_origin_, grid_spacing_, grid_size_);
}

Eigen::MatrixXcd Reconstructor::nufft_reconstruct(
    const aurora::shared::KSpaceSlice& slice) const {

    if (!config_.has_direct_adjoint_config()) {
        throw ReconstructionError(
            ReconstructionError::Code::CONFIGURATION_ERROR,
            "ALGORITHM_NUFFT_ADJOINT requires direct_adjoint_config to be set");
    }
    KSpacePreprocessor preprocessor(config_.direct_adjoint_config());

    // Apply preprocessing (density taper, apodization, DCF)
    const auto [k_max_x, k_max_y] = compute_k_extent(slice);
    aurora::shared::KSpaceSlice processed_slice = preprocessor.apply(slice, k_max_x, k_max_y);

    // Create NufftOperator with MeasurementOperator physics
    NufftOperator op(
        processed_slice,
        forward_model_,
        grid_size_,
        grid_spacing_,
        config_.nufft_config());

    return op.adjoint(aurora::forward::to_eigen_samples(processed_slice),
                      grid_origin_, grid_spacing_, grid_size_);
}

Eigen::MatrixXcd Reconstructor::fast_online_reconstruct(
    const aurora::shared::KSpaceSlice& slice) const {

    // fast_online_config is required and validated in the constructor
    const FastOnlineConfig& fast_config = config_.fast_online_config();

    // The operator applies the K-sample axis window, gridding with density
    // accumulation, normalization, IFFT, post-processing and refinement
    FastOnlineOperator op(slice, grid_size_, grid_spacing_, fast_config);

    return op.reconstruct(aurora::forward::to_eigen_samples(slice),
                          grid_origin_, grid_spacing_, grid_size_);
}

Eigen::MatrixXcd Reconstructor::pcg_reconstruct(
    const aurora::shared::KSpaceSlice& slice) const {

    // pcg_config is required and validated in the constructor
    const PcgConfig& pcg_config = config_.pcg_config();

    // PcgOperator wraps an inner operator (NufftOperator or DirectOperator)
    // and performs the PCG iteration inside adjoint()
    PcgOperator op(
        slice,
        forward_model_,
        config_.bandwidth_config(),
        grid_size_,
        grid_spacing_,
        grid_origin_,
        pcg_config);

    return op.adjoint(aurora::forward::to_eigen_samples(slice),
                      grid_origin_, grid_spacing_, grid_size_);
}

ReconstructedScene Reconstructor::build_scene(
    const std::vector<std::vector<Eigen::MatrixXcd>>& wavelength_images,
    const std::vector<double>& wavelengths,
    const aurora::shared::Axis& range_axis,
    std::chrono::steady_clock::time_point start_time) const {

    ReconstructedScene scene;

    // Set identification
    static uint64_t frame_counter = 0;
    scene.set_frame_id(++frame_counter);

    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    double timestamp = std::chrono::duration<double>(epoch).count();
    scene.set_timestamp_s(timestamp);

    // Build per-wavelength reconstructions
    for (size_t wl_idx = 0; wl_idx < wavelengths.size(); ++wl_idx) {
        auto* wl_recon = scene.add_wavelengths();
        wl_recon->set_wavelength_m(wavelengths[wl_idx]);

        const auto& images = wavelength_images[wl_idx];
        for (size_t rb = 0; rb < images.size(); ++rb) {
            auto* img = wl_recon->add_images();
            img->set_range_bin(static_cast<uint32_t>(rb));

            // Compute range in meters
            double range_m = range_axis.start() + rb * range_axis.step();
            img->set_range_m(range_m);

            // Set axes
            auto* x_axis = img->mutable_x_axis();
            x_axis->set_start(grid_origin_.x());
            x_axis->set_step(grid_spacing_.x());
            x_axis->set_count(grid_size_.x());
            x_axis->set_unit("m");

            auto* y_axis = img->mutable_y_axis();
            y_axis->set_start(grid_origin_.y());
            y_axis->set_step(grid_spacing_.y());
            y_axis->set_count(grid_size_.y());
            y_axis->set_unit("m");

            // Store pixels (row-major)
            const auto& image = images[rb];
            for (int iy = 0; iy < image.rows(); ++iy) {
                for (int ix = 0; ix < image.cols(); ++ix) {
                    auto* pixel = img->add_pixels();
                    pixel->set_real(image(iy, ix).real());
                    pixel->set_imag(image(iy, ix).imag());
                }
            }

            // Compute metrics
            double peak, energy, sparsity;
            compute_metrics(image, peak, energy, sparsity);
            img->set_peak_value(peak);
            img->set_total_energy(energy);
            img->set_sparsity(sparsity);
        }
    }

    // Processing metrics
    scene.set_iterations(0);  // Non-iterative algorithms

    // Processing time
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(end_time - start_time);
    scene.set_processing_time_ms(elapsed.count());

    return scene;
}

void Reconstructor::compute_metrics(
    const Eigen::MatrixXcd& image,
    double& peak_value,
    double& total_energy,
    double& sparsity) const {

    double peak = 0.0;
    double energy = 0.0;
    int significant_pixels = 0;

    const int rows = static_cast<int>(image.rows());
    const int cols = static_cast<int>(image.cols());

    // First pass: compute peak and energy
    for (int iy = 0; iy < rows; ++iy) {
        for (int ix = 0; ix < cols; ++ix) {
            double mag = std::abs(image(iy, ix));
            peak = std::max(peak, mag);
            energy += mag * mag;
        }
    }
    peak_value = peak;
    total_energy = energy;

    // Count significant pixels (above threshold * peak)
    double threshold = kSparsityThreshold * peak_value;
    for (int iy = 0; iy < rows; ++iy) {
        for (int ix = 0; ix < cols; ++ix) {
            if (std::abs(image(iy, ix)) > threshold) {
                ++significant_pixels;
            }
        }
    }

    int total_pixels = rows * cols;
    sparsity = total_pixels > 0 ?
               static_cast<double>(significant_pixels) / total_pixels : 0.0;
}

std::vector<double> Reconstructor::extract_wavelengths(
    const aurora::shared::MeasurementTensor& measurements) const {

    std::set<double> wavelength_set;
    for (const auto& wl_channel : measurements.data()) {
        wavelength_set.insert(wl_channel.wavelength_m());
    }
    return std::vector<double>(wavelength_set.begin(), wavelength_set.end());
}

uint32_t Reconstructor::get_num_range_bins(
    const aurora::shared::MeasurementTensor& measurements) const {

    if (measurements.has_range_axis()) {
        return measurements.range_axis().count();
    }
    // Fallback: count from actual data
    for (const auto& wl_channel : measurements.data()) {
        for (const auto& det_channel : wl_channel.detectors()) {
            if (det_channel.range_data_size() > 0) {
                return det_channel.range_data_size();
            }
        }
    }
    return 0;
}

}  // namespace aurora::reconstruction
