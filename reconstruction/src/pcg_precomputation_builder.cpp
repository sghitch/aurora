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

/**
 * @file pcg_precomputation_builder.cpp
 * @brief Implementation of the PCG precomputation builder.
 */

#include "reconstruction/pcg_precomputation_builder.h"

#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

#include "forward/forward_model.h"
#include "reconstruction/direct_operator.h"
#include "k_space_slice.pb.h"

namespace aurora::reconstruction {

namespace {

// Build a KSpaceSlice (no measurement values) from the receiver's K-samples.
aurora::shared::KSpaceSlice create_slice(
    const forward::ForwardModel& forward_model,
    double wavelength_m) {
    aurora::shared::KSpaceSlice slice;
    slice.set_wavelength_m(wavelength_m);
    slice.set_range_bin(0);  // Not relevant for precomputation

    for (const auto& s : forward_model.get_all_k_samples(wavelength_m)) {
        Eigen::Vector2d k_scene = forward_model.get_effective_k(s.detector_index(), wavelength_m);
        auto* sample = slice.add_samples();
        sample->set_kx(k_scene.x());
        sample->set_ky(k_scene.y());
        sample->set_weight(1.0);
        sample->set_detector_index(s.detector_index());
        sample->set_kx_bandwidth(s.kx_bandwidth());
        sample->set_ky_bandwidth(s.ky_bandwidth());
    }
    return slice;
}

}  // namespace

PcgPrecomputationBuilder::PcgPrecomputationBuilder(
    const forward::ForwardModel& forward_model)
    : forward_model_(forward_model) {
    grid_.set_size_x(64);
    grid_.set_size_y(64);
    grid_.set_spacing_x_m(0.001);
    grid_.set_spacing_y_m(0.001);
    grid_.set_origin_x_m(0.0);
    grid_.set_origin_y_m(0.0);
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_grid_size(
    uint32_t size_x, uint32_t size_y) {
    grid_.set_size_x(size_x);
    grid_.set_size_y(size_y);
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_pixel_size(
    double pixel_size_m) {
    grid_.set_spacing_x_m(pixel_size_m);
    grid_.set_spacing_y_m(pixel_size_m);
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_pixel_spacing(
    double spacing_x_m, double spacing_y_m) {
    grid_.set_spacing_x_m(spacing_x_m);
    grid_.set_spacing_y_m(spacing_y_m);
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_origin(
    double origin_x_m, double origin_y_m) {
    grid_.set_origin_x_m(origin_x_m);
    grid_.set_origin_y_m(origin_y_m);
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_grid(const GridSpec& grid) {
    grid_ = grid;
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_regularizer(
    RegularizerType type) {
    regularizer_ = type;
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_preconditioner(
    PreconditionerType type) {
    preconditioner_ = type;
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_geometry_id(
    const std::string& id) {
    geometry_id_ = id;
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_calibration_id(
    const std::string& id) {
    calibration_id_ = id;
    return *this;
}

PcgPrecomputationBuilder& PcgPrecomputationBuilder::set_bandwidth_config(
    const forward::BandwidthConfig& config) {
    bandwidth_config_ = config;
    return *this;
}

void PcgPrecomputationBuilder::validate_bandwidth_config() const {
    if (bandwidth_config_.window_type() == forward::BANDWIDTH_WINDOW_UNSPECIFIED) {
        throw std::invalid_argument(
            "PcgPrecomputationBuilder: set_bandwidth_config() must be called "
            "before build(). window_type is required.");
    }
    if (bandwidth_config_.quadrature_order() == 0) {
        throw std::invalid_argument(
            "PcgPrecomputationBuilder: set_bandwidth_config() must be called "
            "before build(). quadrature_order must be >= 1.");
    }
}

PcgPrecomputedDataSet PcgPrecomputationBuilder::build(double wavelength_m) {
    return build(std::vector<double>{wavelength_m});
}

PcgPrecomputedDataSet PcgPrecomputationBuilder::build(
    const std::vector<double>& wavelengths_m) {

    if (wavelengths_m.empty()) {
        throw std::invalid_argument("At least one wavelength is required");
    }
    validate_bandwidth_config();

    PcgPrecomputedDataSet result;
    result.set_geometry_id(geometry_id_);
    result.set_calibration_id(calibration_id_);
    *result.mutable_grid() = grid_;

    for (double wavelength : wavelengths_m) {
        auto slice = create_slice(forward_model_, wavelength);

        Eigen::Vector2i grid_size(
            static_cast<int>(grid_.size_x()),
            static_cast<int>(grid_.size_y()));
        Eigen::Vector2d grid_spacing(grid_.spacing_x_m(), grid_.spacing_y_m());
        Eigen::Vector2d grid_origin(grid_.origin_x_m(), grid_.origin_y_m());

        DirectOperator op(
            slice,
            forward_model_,
            bandwidth_config_,
            grid_size,
            grid_spacing,
            grid_origin);

        Eigen::VectorXd weights(slice.samples_size());
        for (int i = 0; i < slice.samples_size(); ++i) {
            weights(i) = slice.samples(i).weight();
        }

        auto data = PcgPrecomputation::compute(
            op, weights, grid_, wavelength,
            regularizer_, preconditioner_,
            geometry_id_, calibration_id_);

        *result.add_wavelength_data() = data;
    }

    return result;
}

}  // namespace aurora::reconstruction
