#include "forward/forward_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace aurora::forward {

namespace {

constexpr double kWavelengthTolerance = 1e-12;  // meters
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

const DetectorArray& first_array(const DetectorGeometry& detector) {
    if (detector.arrays_size() == 0) {
        throw std::invalid_argument("ForwardModel requires at least one detector array");
    }
    if (detector.arrays(0).pixels_size() == 0) {
        throw std::invalid_argument("ForwardModel requires a detector array with pixels");
    }
    return detector.arrays(0);
}

}  // namespace

ForwardModel::ForwardModel(const DetectorGeometry& detector,
                           const OpticsConfig& optics)
    : detector_(detector),
      optics_(optics),
      geometry_(optics, first_array(detector)) {
    if (optics_.fourier_lens_focal_m() <= 0.0) {
        throw std::invalid_argument("ForwardModel requires a positive fourier_lens_focal_m");
    }
}

ForwardModel::ForwardModel(const ForwardConfig& config)
    : ForwardModel(config.detector(), config.optics()) {}

ForwardModel::~ForwardModel() = default;

ForwardModel::ForwardModel(ForwardModel&&) noexcept = default;
ForwardModel& ForwardModel::operator=(ForwardModel&&) noexcept = default;

DetectorKSample ForwardModel::get_k_sample(uint32_t detector_idx,
                                           double wavelength_m) const {
    Eigen::Vector2d k_det = geometry_.detector_k(detector_idx, wavelength_m);

    DetectorKSample sample;
    sample.set_detector_index(detector_idx);
    sample.set_wavelength_m(wavelength_m);
    sample.set_kx_eff(k_det.x());
    sample.set_ky_eff(k_det.y());
    sample.set_kx_bandwidth(geometry_.detector_k_bandwidth_x(detector_idx, wavelength_m));
    sample.set_ky_bandwidth(geometry_.detector_k_bandwidth_y(detector_idx, wavelength_m));
    return sample;
}

Eigen::Vector2d ForwardModel::get_effective_k(uint32_t detector_idx,
                                              double wavelength_m) const {
    return geometry_.detector_k(detector_idx, wavelength_m) +
           geometry_.lo_tilt_k(wavelength_m);
}

std::vector<DetectorKSample> ForwardModel::get_all_k_samples(
    double wavelength_m) const {
    std::vector<DetectorKSample> result;
    result.reserve(num_detectors());
    for (uint32_t d = 0; d < num_detectors(); ++d) {
        result.push_back(get_k_sample(d, wavelength_m));
    }
    return result;
}

uint32_t ForwardModel::num_detectors() const {
    return geometry_.num_detectors();
}

aurora::shared::KSpaceSlice ForwardModel::extract_kspace_slice(
    const aurora::shared::MeasurementTensor& measurements,
    uint32_t range_bin,
    double wavelength_m) const {

    aurora::shared::KSpaceSlice slice;
    slice.set_range_bin(range_bin);
    slice.set_wavelength_m(wavelength_m);

    if (measurements.burst_id() != 0) {
        slice.set_burst_id(measurements.burst_id());
    }
    if (measurements.timestamp_s() != 0.0) {
        slice.set_timestamp_s(measurements.timestamp_s());
    }

    for (const auto& wavelength_channel : measurements.data()) {
        if (std::abs(wavelength_channel.wavelength_m() - wavelength_m)
            > kWavelengthTolerance) {
            continue;
        }

        for (const auto& detector_channel : wavelength_channel.detectors()) {
            const uint32_t det_idx = detector_channel.detector_index();

            if (det_idx >= num_detectors()) {
                continue;  // Detector index not part of this receiver geometry
            }
            if (range_bin >= static_cast<uint32_t>(detector_channel.range_data_size())) {
                continue;
            }

            DetectorKSample k_sample = get_k_sample(det_idx, wavelength_m);
            if (k_sample.kx_bandwidth() <= 0.0 || k_sample.ky_bandwidth() <= 0.0) {
                throw std::runtime_error(
                    "Detector " + std::to_string(det_idx) +
                    " has a zero K-space bandwidth (pixel width/height must be positive)");
            }

            // Scene K-space coordinates (includes LO tilt). The simulation
            // encodes phase using scene K, so reconstruction must match.
            Eigen::Vector2d k_scene = get_effective_k(det_idx, wavelength_m);

            const auto& range_data = detector_channel.range_data(range_bin);

            auto* sample = slice.add_samples();
            sample->mutable_value()->set_real(range_data.real());
            sample->mutable_value()->set_imag(range_data.imag());
            sample->set_kx(k_scene.x());
            sample->set_ky(k_scene.y());
            sample->set_weight(1.0);
            sample->set_detector_index(det_idx);
            sample->set_kx_bandwidth(k_sample.kx_bandwidth());
            sample->set_ky_bandwidth(k_sample.ky_bandwidth());
        }
    }

    compute_grid_hints(slice);
    return slice;
}

aurora::shared::KSpaceSliceCollection ForwardModel::extract_all_range_slices(
    const aurora::shared::MeasurementTensor& measurements,
    double wavelength_m) const {

    aurora::shared::KSpaceSliceCollection collection;
    collection.set_wavelength_m(wavelength_m);

    const uint32_t num_bins = get_num_range_bins(measurements);
    for (uint32_t bin = 0; bin < num_bins; ++bin) {
        *collection.add_slices() = extract_kspace_slice(measurements, bin, wavelength_m);
    }
    return collection;
}

double ForwardModel::compute_phase(const Eigen::Vector2d& k,
                                   const Eigen::Vector2d& position_xy) {
    // Phase encoding: φ = -k · r = -kx * x - ky * y
    return -(k.x() * position_xy.x() + k.y() * position_xy.y());
}

std::complex<double> ForwardModel::compute_phasor(
    const Eigen::Vector2d& k,
    const Eigen::Vector2d& position_xy) {
    return std::polar(1.0, compute_phase(k, position_xy));
}

void ForwardModel::compute_grid_hints(
    aurora::shared::KSpaceSlice& slice) const {

    if (slice.samples_size() == 0) {
        return;
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

    // Nyquist-based relationship: n = ceil(k_extent / (2*pi) * oversampling)
    constexpr double kOversampling = 2.0;
    constexpr uint32_t kMinGridSize = 8;
    const double kx_extent = kx_max - kx_min;
    const double ky_extent = ky_max - ky_min;

    uint32_t suggested_nx = static_cast<uint32_t>(
        std::ceil(kx_extent / kTwoPi * kOversampling));
    uint32_t suggested_ny = static_cast<uint32_t>(
        std::ceil(ky_extent / kTwoPi * kOversampling));

    suggested_nx = std::max(suggested_nx, kMinGridSize);
    suggested_ny = std::max(suggested_ny, kMinGridSize);

    auto* hints = slice.mutable_grid_hints();
    hints->set_kx_min(kx_min);
    hints->set_kx_max(kx_max);
    hints->set_ky_min(ky_min);
    hints->set_ky_max(ky_max);
    hints->set_suggested_nx(suggested_nx);
    hints->set_suggested_ny(suggested_ny);
}

uint32_t ForwardModel::get_num_range_bins(
    const aurora::shared::MeasurementTensor& measurements) {

    uint32_t max_bins = 0;
    for (const auto& wavelength_channel : measurements.data()) {
        for (const auto& detector_channel : wavelength_channel.detectors()) {
            max_bins = std::max(max_bins, static_cast<uint32_t>(
                detector_channel.range_data_size()));
        }
    }
    return max_bins;
}

}  // namespace aurora::forward
