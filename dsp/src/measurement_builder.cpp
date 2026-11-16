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

#include "dsp/measurement_builder.h"

#include <stdexcept>

namespace aurora::dsp {

MeasurementBuilder::MeasurementBuilder(
    const aurora::forward::ForwardModel& forward_model)
    : forward_model_(forward_model) {
}

MeasurementBuilder::~MeasurementBuilder() = default;

aurora::shared::KVector MeasurementBuilder::get_calibrated_k(
    double wavelength_m,
    uint32_t detector_index) const {

    // Scene K (detector K plus LO tilt) for this pixel at this wavelength.
    Eigen::Vector2d k_scene = forward_model_.get_effective_k(detector_index, wavelength_m);

    aurora::shared::KVector k_vec;
    k_vec.set_kx(k_scene.x());
    k_vec.set_ky(k_scene.y());
    return k_vec;
}

void MeasurementBuilder::convert_channels(
    const google::protobuf::RepeatedPtrField<WavelengthRangeData>& source,
    google::protobuf::RepeatedPtrField<aurora::shared::WavelengthChannel>* target) const {

    for (const auto& wavelength : source) {
        auto* wl_channel = target->Add();
        wl_channel->set_wavelength_m(wavelength.wavelength_m());

        for (const auto& detector : wavelength.detectors()) {
            auto* det_channel = wl_channel->add_detectors();
            det_channel->set_detector_index(detector.detector_index());
            *det_channel->mutable_k_vector() = get_calibrated_k(
                wavelength.wavelength_m(), detector.detector_index());

            for (const auto& bin : detector.bins()) {
                auto* out_data = det_channel->add_range_data();
                out_data->set_real(bin.real());
                out_data->set_imag(bin.imag());
            }
        }
    }
}

aurora::shared::MeasurementTensor MeasurementBuilder::build(
    const RangeCube& range_cube) {

    aurora::shared::MeasurementTensor tensor;
    *tensor.mutable_range_axis() = range_cube.range_axis();
    tensor.set_burst_id(range_cube.burst_id());
    tensor.set_timestamp_s(range_cube.timestamp_s());

    convert_channels(range_cube.wavelengths(), tensor.mutable_data());
    convert_channels(range_cube.up_chirps(), tensor.mutable_up_chirp_data());
    convert_channels(range_cube.down_chirps(), tensor.mutable_down_chirp_data());

    return tensor;
}

namespace {

/// Add the range bins of `source` into `target` element-wise (first step of
/// the coherent average over several bursts).
void sum_channels(
    const google::protobuf::RepeatedPtrField<aurora::shared::WavelengthChannel>& source,
    google::protobuf::RepeatedPtrField<aurora::shared::WavelengthChannel>* target) {

    if (source.size() != target->size()) {
        throw std::invalid_argument("RangeCubes have different number of wavelengths");
    }
    for (int w = 0; w < source.size(); ++w) {
        const auto& wavelength = source.Get(w);
        auto* tensor_wavelength = target->Mutable(w);
        if (wavelength.detectors_size() != tensor_wavelength->detectors_size()) {
            throw std::invalid_argument("RangeCubes have different number of detectors");
        }
        for (int d = 0; d < wavelength.detectors_size(); ++d) {
            const auto& detector = wavelength.detectors(d);
            auto* tensor_detector = tensor_wavelength->mutable_detectors(d);
            if (detector.range_data_size() != tensor_detector->range_data_size()) {
                throw std::invalid_argument("RangeCubes have different number of range bins");
            }
            for (int b = 0; b < detector.range_data_size(); ++b) {
                auto* out_bin = tensor_detector->mutable_range_data(b);
                out_bin->set_real(out_bin->real() + detector.range_data(b).real());
                out_bin->set_imag(out_bin->imag() + detector.range_data(b).imag());
            }
        }
    }
}

/// Scale every range bin by `scale` (second step of the coherent average).
void scale_channels(
    google::protobuf::RepeatedPtrField<aurora::shared::WavelengthChannel>* channels,
    double scale) {
    for (auto& wavelength : *channels) {
        for (auto& detector : *wavelength.mutable_detectors()) {
            for (auto& bin : *detector.mutable_range_data()) {
                bin.set_real(bin.real() * scale);
                bin.set_imag(bin.imag() * scale);
            }
        }
    }
}

}  // namespace

aurora::shared::MeasurementTensor MeasurementBuilder::build(
    const std::vector<RangeCube>& range_cubes) {

    if (range_cubes.empty()) {
        return aurora::shared::MeasurementTensor{};
    }
    if (range_cubes.size() == 1) {
        return build(range_cubes[0]);
    }

    // Coherent average of the per-burst tensors: sum the range bins, then
    // divide by the number of bursts. The K-vectors are identical for every
    // burst, so only the range data changes.
    aurora::shared::MeasurementTensor tensor = build(range_cubes[0]);

    for (size_t cube_idx = 1; cube_idx < range_cubes.size(); ++cube_idx) {
        aurora::shared::MeasurementTensor next = build(range_cubes[cube_idx]);
        sum_channels(next.data(), tensor.mutable_data());
        sum_channels(next.up_chirp_data(), tensor.mutable_up_chirp_data());
        sum_channels(next.down_chirp_data(), tensor.mutable_down_chirp_data());
    }

    double scale = 1.0 / static_cast<double>(range_cubes.size());
    scale_channels(tensor.mutable_data(), scale);
    scale_channels(tensor.mutable_up_chirp_data(), scale);
    scale_channels(tensor.mutable_down_chirp_data(), scale);

    return tensor;
}

}  // namespace aurora::dsp
