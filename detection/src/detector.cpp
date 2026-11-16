#include "detection/detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <set>

namespace aurora::detection {

namespace {

double db_to_linear_power(double db) {
    return std::pow(10.0, db / 10.0);
}

double linear_power_to_db(double power) {
    if (power <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return 10.0 * std::log10(power);
}

/// Default cap on coarse angular search bins when the config leaves it unset.
constexpr uint32_t kDefaultAngularBins = 16384;

/// Index of the WavelengthChannel record used for angle estimation.
constexpr uint32_t kAngleWavelengthIndex = 0;

}  // namespace

Detector::Detector(const DetectionConfig& config,
                   const aurora::forward::ForwardConfig& forward_config,
                   const aurora::acquisition::SequenceSchedule& sequence_schedule)
    : config_(config),
      forward_config_(forward_config),
      sequence_schedule_(sequence_schedule) {
    initialize_analyzers();
}

void Detector::initialize_analyzers() {
    energy_threshold_ = db_to_linear_power(config_.magnitude_threshold_db());

    if (config_.has_cfar() && config_.cfar().mode() != CFAR_MODE_UNSPECIFIED &&
        config_.cfar().mode() != CFAR_MODE_FIXED) {
        cfar_detector_ = std::make_unique<CfarDetector>(config_.cfar());
    } else {
        cfar_detector_.reset();
    }

    if (!forward_config_.has_optics() || forward_config_.optics().wavelengths_m_size() == 0) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
            "forward_config must have optics with at least one wavelength");
    }
    if (!forward_config_.has_detector() || forward_config_.detector().arrays_size() == 0) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
            "forward_config must have detector with at least one array");
    }

    const auto& optics = forward_config_.optics();
    if (optics.fourier_lens_focal_m() <= 0.0) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
            "forward_config.optics must have a positive fourier_lens_focal_m");
    }

    // Field of view searched by the coarse stage: explicit system FOV if set,
    // otherwise derived from the pupil half-angle.
    double fov_x = optics.system_fov_x_rad();
    double fov_y = optics.system_fov_y_rad();
    if (fov_x <= 0.0 && optics.has_pupil()) {
        fov_x = 2.0 * optics.pupil().fov_half_angle_rad();
    }
    if (fov_y <= 0.0 && optics.has_pupil()) {
        fov_y = 2.0 * optics.pupil().fov_half_angle_rad();
    }
    if (fov_x <= 0.0) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
            "forward_config.optics must specify system_fov_x_rad or a pupil FOV");
    }

    uint32_t bins = config_.num_angular_bins() > 1
        ? config_.num_angular_bins() : kDefaultAngularBins;

    try {
        angle_estimator_ = std::make_unique<AngleEstimator>(
            optics.fourier_lens_focal_m(),
            db_to_linear_power(config_.noise_floor_db()),
            fov_x, fov_y, bins);
    } catch (const std::invalid_argument& e) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
            std::string("AngleEstimator initialization failed: ") + e.what());
    }

    if (!sequence_schedule_.has_chirp_schedule() ||
        sequence_schedule_.chirp_schedule().chirp_duration_s() <= 0.0) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
            "sequence_schedule must have chirp_schedule with valid chirp_duration_s");
    }

    doppler_extractor_ = std::make_unique<DopplerExtractor>(
        config_,
        optics.wavelengths_m(0),
        sequence_schedule_.chirp_schedule().chirp_duration_s());
}

void Detector::update_config(const DetectionConfig& config) {
    config_ = config;
    initialize_analyzers();
}

DetectionFrame Detector::detect(const aurora::shared::MeasurementTensor& measurements) {
    auto start_time = std::chrono::high_resolution_clock::now();

    DetectionFrame frame;
    frame.set_frame_id(next_frame_id_++);
    frame.set_timestamp_s(measurements.timestamp_s());

    if (measurements.data_size() == 0) {
        frame.set_num_active_range_bins(0);
        frame.set_processing_time_ms(0.0);
        return frame;
    }

    auto active_bins = find_active_range_bins(measurements);
    frame.set_num_active_range_bins(static_cast<uint32_t>(active_bins.size()));

    for (uint32_t range_bin : active_bins) {
        try {
            auto detection = process_range_bin(measurements, range_bin);
            if (!std::isfinite(detection.azimuth_uncertainty_rad())) {
                continue;
            }
            frame.add_detections()->CopyFrom(detection);
        } catch (const DetectionError&) {
            // Skip this bin; continue with the others
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    frame.set_processing_time_ms(
        std::chrono::duration<double, std::milli>(end_time - start_time).count());

    return frame;
}

std::vector<uint32_t> Detector::find_active_range_bins(
    const aurora::shared::MeasurementTensor& m) const {

    std::vector<uint32_t> active_bins;
    if (m.data_size() == 0) {
        return active_bins;
    }

    uint32_t num_range_bins = 0;
    if (m.data(0).detectors_size() > 0) {
        num_range_bins = static_cast<uint32_t>(m.data(0).detectors(0).range_data_size());
    }
    if (num_range_bins == 0) {
        return active_bins;
    }

    std::vector<double> bin_energies(num_range_bins, 0.0);
    for (const auto& wavelength_ch : m.data()) {
        for (const auto& detector : wavelength_ch.detectors()) {
            const uint32_t bins = std::min(
                num_range_bins, static_cast<uint32_t>(detector.range_data_size()));
            for (uint32_t bin = 0; bin < bins; ++bin) {
                const auto& sample = detector.range_data(bin);
                bin_energies[bin] += sample.real() * sample.real() +
                                     sample.imag() * sample.imag();
            }
        }
    }

    std::vector<uint32_t> candidate_bins;
    if (cfar_detector_ && cfar_detector_->is_enabled()) {
        auto cfar_detections = cfar_detector_->detect_1d(bin_energies);
        candidate_bins.reserve(cfar_detections.size());
        for (const auto& det : cfar_detections) {
            candidate_bins.push_back(det.bin_index());
        }
    } else {
        for (uint32_t bin = 0; bin < num_range_bins; ++bin) {
            if (bin_energies[bin] > energy_threshold_) {
                candidate_bins.push_back(bin);
            }
        }
    }

    return select_run_peaks(candidate_bins, bin_energies);
}

std::vector<uint32_t> Detector::select_run_peaks(
    const std::vector<uint32_t>& candidate_bins,
    const std::vector<double>& bin_energies) {

    // The range window spreads one target over a few adjacent bins. Keep the
    // energy peak of each run of consecutive candidate bins so a single
    // target yields a single detection.
    std::vector<uint32_t> peaks;
    size_t i = 0;
    while (i < candidate_bins.size()) {
        uint32_t best = candidate_bins[i];
        size_t j = i + 1;
        while (j < candidate_bins.size() && candidate_bins[j] == candidate_bins[j - 1] + 1) {
            if (bin_energies[candidate_bins[j]] > bin_energies[best]) {
                best = candidate_bins[j];
            }
            ++j;
        }
        peaks.push_back(best);
        i = j;
    }
    return peaks;
}

std::vector<AngleEstimator::Sample> Detector::gather_samples(
    const aurora::shared::MeasurementTensor& m,
    uint32_t range_bin,
    uint32_t wavelength_index) const {

    std::vector<AngleEstimator::Sample> samples;
    if (static_cast<int>(wavelength_index) >= m.data_size()) {
        return samples;
    }
    const auto& wavelength_ch = m.data(wavelength_index);
    for (const auto& detector : wavelength_ch.detectors()) {
        if (range_bin >= static_cast<uint32_t>(detector.range_data_size())) {
            continue;
        }
        if (!detector.has_k_vector()) {
            throw DetectionError(DetectionError::Code::INVALID_MEASUREMENT,
                "DetectorChannel is missing its k_vector annotation");
        }
        const auto& value = detector.range_data(range_bin);
        samples.push_back({
            Eigen::Vector2d(detector.k_vector().kx(), detector.k_vector().ky()),
            std::complex<double>(value.real(), value.imag())});
    }
    return samples;
}

Detection Detector::process_range_bin(
    const aurora::shared::MeasurementTensor& m,
    uint32_t range_bin) const {

    Detection det;

    double range_m = compute_range(m, range_bin);
    det.set_range_m(range_m);

    auto samples = gather_samples(m, range_bin, kAngleWavelengthIndex);
    const uint32_t min_samples = std::max<uint32_t>(2, config_.min_pixels_for_gradient());
    if (samples.size() < min_samples) {
        throw DetectionError(DetectionError::Code::INSUFFICIENT_SIGNAL,
            "Too few K-space samples for phase gradient fit");
    }

    AngleFitResult fit = angle_estimator_->estimate(samples);
    if (!fit.valid()) {
        throw DetectionError(DetectionError::Code::PHASE_GRADIENT_FAILED,
            "Phase gradient fit failed");
    }
    if (config_.max_phase_residual_rad() > 0.0 &&
        fit.residual_rad() > config_.max_phase_residual_rad()) {
        throw DetectionError(DetectionError::Code::PHASE_GRADIENT_FAILED,
            "Phase gradient residual exceeds configured maximum");
    }
    if (config_.max_condition_number() > 0.0 &&
        fit.condition_number() > config_.max_condition_number()) {
        throw DetectionError(DetectionError::Code::PHASE_GRADIENT_FAILED,
            "Phase gradient fit is ill-conditioned");
    }

    const double azimuth_rad = fit.azimuth_rad();
    const double elevation_rad = fit.elevation_rad();
    det.set_azimuth_rad(azimuth_rad);
    det.set_azimuth_uncertainty_rad(fit.azimuth_uncertainty_rad());
    det.set_elevation_rad(elevation_rad);
    det.set_elevation_uncertainty_rad(fit.elevation_uncertainty_rad());
    det.set_phase_residual_rad(fit.residual_rad());

    auto* position = det.mutable_position_m();
    position->set_x(range_m * std::cos(elevation_rad) * std::sin(azimuth_rad));
    position->set_y(range_m * std::sin(elevation_rad));
    position->set_z(range_m * std::cos(elevation_rad) * std::cos(azimuth_rad));

    if (doppler_extractor_->is_enabled()) {
        estimate_doppler(m, range_bin, det);
    }

    double energy = compute_range_bin_energy(m, range_bin, kAngleWavelengthIndex);
    det.set_snr_db(compute_snr_db(energy));

    std::set<uint32_t> detector_indices;
    for (const auto& detector : m.data(kAngleWavelengthIndex).detectors()) {
        detector_indices.insert(detector.detector_index());
    }
    for (uint32_t idx : detector_indices) {
        det.add_detector_indices(idx);
    }

    return det;
}

void Detector::estimate_doppler(
    const aurora::shared::MeasurementTensor& m,
    uint32_t range_bin,
    Detection& det) const {

    // Doppler needs the up-chirp and down-chirp range profiles, which the DSP
    // provides when the burst was processed with up/down chirps.
    if (static_cast<int>(kAngleWavelengthIndex) >= m.up_chirp_data_size() ||
        static_cast<int>(kAngleWavelengthIndex) >= m.down_chirp_data_size()) {
        return;
    }

    std::vector<std::complex<double>> up_samples;
    std::vector<std::complex<double>> down_samples;

    const auto& wl0 = m.up_chirp_data(kAngleWavelengthIndex);
    const auto& wl1 = m.down_chirp_data(kAngleWavelengthIndex);
    for (int d = 0; d < wl0.detectors_size() && d < wl1.detectors_size(); ++d) {
        const auto& det0 = wl0.detectors(d);
        const auto& det1 = wl1.detectors(d);
        if (range_bin < static_cast<uint32_t>(det0.range_data_size()) &&
            range_bin < static_cast<uint32_t>(det1.range_data_size())) {
            const auto& s0 = det0.range_data(range_bin);
            const auto& s1 = det1.range_data(range_bin);
            up_samples.emplace_back(s0.real(), s0.imag());
            down_samples.emplace_back(s1.real(), s1.imag());
        }
    }

    if (up_samples.empty() || up_samples.size() != down_samples.size()) {
        return;
    }

    try {
        auto doppler_result = doppler_extractor_->extract_velocity(up_samples, down_samples);
        det.mutable_doppler()->CopyFrom(doppler_result);
    } catch (const DetectionError&) {
        // Doppler extraction failed, continue without it
    }
}

double Detector::compute_range(
    const aurora::shared::MeasurementTensor& m,
    uint32_t range_bin) const {

    if (!m.has_range_axis()) {
        throw DetectionError(DetectionError::Code::INVALID_MEASUREMENT,
            "MeasurementTensor is missing its range_axis");
    }
    const auto& axis = m.range_axis();
    return axis.start() + range_bin * axis.step();
}

double Detector::compute_snr_db(double energy) const {
    double noise_floor_power = db_to_linear_power(config_.noise_floor_db());
    if (energy <= 0.0 || energy <= noise_floor_power) {
        return 0.0;
    }
    return linear_power_to_db(energy / noise_floor_power);
}

double Detector::compute_range_bin_energy(
    const aurora::shared::MeasurementTensor& m,
    uint32_t range_bin,
    uint32_t wavelength_index) const {

    double total_energy = 0.0;
    if (static_cast<int>(wavelength_index) >= m.data_size()) {
        return total_energy;
    }
    for (const auto& detector : m.data(wavelength_index).detectors()) {
        if (range_bin < static_cast<uint32_t>(detector.range_data_size())) {
            const auto& sample = detector.range_data(range_bin);
            total_energy += sample.real() * sample.real() +
                            sample.imag() * sample.imag();
        }
    }
    return total_energy;
}

}  // namespace aurora::detection
