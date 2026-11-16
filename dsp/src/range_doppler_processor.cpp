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

#include "dsp/range_doppler_processor.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

#include "dsp/fft_wrapper.h"
#include "dsp/iq_demodulator.h"
#include "dsp/measurement_builder.h"
#include "dsp/peak_doppler_processor.h"
#include "dsp/window_functions.h"
#include "forward/forward_model.h"

namespace aurora::dsp {

namespace {

/// Chirp parameters assumed when a burst's plan carries no chirps.
constexpr double kDefaultChirpBandwidthHz = 200e6;
constexpr double kDefaultChirpDurationS = 12.5e-6;

/// Wavelength assumed when a burst carries no wavelength information.
constexpr double kDefaultWavelengthM = 905e-9;
constexpr double kSpeedOfLight = 299792458.0;  // m/s

using Clock = std::chrono::high_resolution_clock;

double to_microseconds(Clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}
}  // namespace

namespace {

void validate_dsp_config(const DspConfig& config) {
    // Validate configuration
    if (config.range_fft_size() == 0) {
        throw std::invalid_argument("range_fft_size must be positive");
    }
    if ((config.range_fft_size() & (config.range_fft_size() - 1)) != 0) {
        throw std::invalid_argument("range_fft_size must be a power of 2");
    }

    // Validate ADC config
    if (!config.has_adc_config()) {
        throw std::invalid_argument("adc_config must be provided");
    }
    const auto& adc = config.adc_config();
    if (adc.sample_rate_hz() <= 0) {
        throw std::invalid_argument("adc_config.sample_rate_hz must be positive");
    }
    if (adc.bits() == 0) {
        throw std::invalid_argument("adc_config.bits must be positive");
    }
    if (adc.input_range_v() <= 0) {
        throw std::invalid_argument("adc_config.input_range_v must be positive");
    }

    // Validate IQ demod config
    if (!config.has_iq_demod_config()) {
        throw std::invalid_argument("iq_demod_config must be provided");
    }
    const auto& iq = config.iq_demod_config();
    if (iq.lpf_cutoff_hz() <= 0) {
        throw std::invalid_argument("iq_demod_config.lpf_cutoff_hz must be positive");
    }
    if (iq.lpf_order() == 0) {
        throw std::invalid_argument("iq_demod_config.lpf_order must be positive");
    }
}

}  // namespace

RangeDopplerProcessor::RangeDopplerProcessor(
    const DspConfig& config,
    const aurora::forward::ForwardModel& forward_model)
    : config_(config),
      forward_model_(forward_model),
      measurement_builder_(forward_model) {

    validate_dsp_config(config_);

    const auto& adc = config_.adc_config();
    const auto& iq = config_.iq_demod_config();

    // Compute ADC LSB from config
    adc_lsb_ = adc.input_range_v() / static_cast<double>(1 << adc.bits());
    sample_rate_hz_ = adc.sample_rate_hz();

    // Initialize FFT wrapper
    range_fft_ = std::make_unique<FftWrapper>(config_.range_fft_size(), false);

    // Initialize IQ demodulator with config parameters
    // Actual reference frequency will be set per-chirp based on chirp parameters
    IQDemodulator::Config iq_config{
        .sample_rate_hz = sample_rate_hz_,
        .reference_freq_hz = 0.0,  // Will be set per-chirp
        .lpf_cutoff_hz = iq.lpf_cutoff_hz(),
        .lpf_order = iq.lpf_order()
    };
    iq_demod_ = std::make_unique<IQDemodulator>(iq_config);
}

RangeDopplerProcessor::~RangeDopplerProcessor() = default;

double RangeDopplerProcessor::compute_range_resolution(double bandwidth_hz) {
    return kSpeedOfLight / (2.0 * bandwidth_hz);
}

double RangeDopplerProcessor::compute_range_per_bin(double bandwidth_hz,
                                                    double chirp_duration_s,
                                                    double sample_rate_hz,
                                                    uint32_t fft_size) {
    // Range per FFT bin = c * T * f_s / (2 * B * N)
    // where: c = speed of light, T = chirp duration, f_s = sample rate,
    //        B = bandwidth, N = FFT size
    //
    // Derivation:
    // - Beat frequency for target at range R: f_beat = 2 * R * B / (c * T)
    // - FFT bin k corresponds to frequency: f_k = k * f_s / N
    // - Therefore: R_k = k * c * T * f_s / (2 * B * N)
    // - Range step (per bin): delta_R = c * T * f_s / (2 * B * N)
    return kSpeedOfLight * chirp_duration_s * sample_rate_hz /
           (2.0 * bandwidth_hz * static_cast<double>(fft_size));
}

double RangeDopplerProcessor::compute_range_offset_from_reference_freq(
    double reference_freq_hz,
    double bandwidth_hz,
    double chirp_duration_s) {
    // IQ demodulation with non-zero reference frequency shifts the beat frequency
    // spectrum. The current implementation shifts UP by reference_freq_hz, so
    // we need a negative range offset to compensate.
    //
    // Range offset = -c * T * f_ref / (2 * B)
    //
    // This ensures that a target at range R still appears at the correct range
    // bin after IQ demodulation, regardless of reference frequency.
    if (reference_freq_hz == 0.0) {
        return 0.0;
    }
    return -kSpeedOfLight * chirp_duration_s * reference_freq_hz /
           (2.0 * bandwidth_hz);
}

std::vector<double> RangeDopplerProcessor::remove_dc(
    const std::vector<double>& signal) const {

    if (signal.empty()) {
        return {};
    }

    // Compute mean
    double mean = std::accumulate(signal.begin(), signal.end(), 0.0) /
                  static_cast<double>(signal.size());

    // Subtract mean
    std::vector<double> result(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        result[i] = signal[i] - mean;
    }

    return result;
}

std::vector<std::complex<double>> RangeDopplerProcessor::process_chirp(
    const std::vector<int32_t>& adc_samples,
    double chirp_bandwidth_hz,
    double chirp_duration_s) {

    if (adc_samples.empty()) {
        return {};
    }

    auto t_start = Clock::now();

    // 1. Convert ADC to voltage
    std::vector<double> signal(adc_samples.size());
    for (size_t i = 0; i < adc_samples.size(); ++i) {
        signal[i] = static_cast<double>(adc_samples[i]) * adc_lsb_;
    }

    // 2. DC removal (optional)
    auto t_preprocess = Clock::now();
    if (config_.enable_dc_removal()) {
        signal = remove_dc(signal);
    }
    latency_stats_.preprocess_us += to_microseconds(Clock::now() - t_preprocess);

    // 3. IQ demodulation
    auto t_iq = Clock::now();
    // Use reference frequency from config to produce analytic (complex) signal.
    // Non-zero ref_freq breaks Hermitian symmetry in K-space, preventing
    // mirroring artifacts in reconstruction. Set ref_freq near the center
    // of expected beat frequencies: f_beat = 2*R*B/(c*T).
    // For typical ranges (5-50m) with 200MHz bandwidth and 12.5µs chirp,
    // beat frequencies span ~0.5-5 MHz, so ref_freq ~2.5 MHz is appropriate.
    double ref_freq = config_.iq_demod_config().reference_freq_hz();
    auto complex_signal = iq_demod_->demodulate(signal, ref_freq);
    latency_stats_.iq_demod_us += to_microseconds(Clock::now() - t_iq);

    // 4. Apply window
    auto t_window = Clock::now();
    WindowFunctions::apply_inplace(complex_signal, config_.range_window());
    latency_stats_.window_us += to_microseconds(Clock::now() - t_window);

    // 5. Zero-pad if needed
    size_t fft_size = config_.range_fft_size();
    if (complex_signal.size() < fft_size) {
        complex_signal.resize(fft_size, std::complex<double>(0.0, 0.0));
    } else if (complex_signal.size() > fft_size) {
        complex_signal.resize(fft_size);
    }

    // 6. FFT
    auto t_fft = Clock::now();
    range_fft_->execute_inplace(complex_signal);
    latency_stats_.fft_us += to_microseconds(Clock::now() - t_fft);

    // 7. Return full spectrum
    //
    // FFT output layout for N bins:
    //   Bins [0, N/2):   positive frequencies (0 to +f_s/2)
    //   Bins [N/2, N):   negative frequencies (-f_s/2 to 0)
    //
    // For close-range targets with velocity, the down-chirp beat frequency
    // can go negative (when f_doppler > f_beat). These peaks appear in the
    // negative frequency bins [N/2, N).
    //
    // Callers that only need positive range bins should use bins [0, N/2).
    // Peak-based Doppler processing scans the full spectrum.

    // Update range resolution based on actual chirp bandwidth
    range_resolution_m_ = compute_range_resolution(chirp_bandwidth_hz);

    latency_stats_.total_us += to_microseconds(Clock::now() - t_start);

    return complex_signal;
}

RangeCube RangeDopplerProcessor::process_range(
    const aurora::simulation::Burst& burst) {

    // Reset latency stats
    latency_stats_ = LatencyStats{};
    auto t_total_start = Clock::now();

    RangeCube cube;
    cube.set_burst_id(burst.burst_id());
    cube.set_timestamp_s(burst.timestamp_s());
    *cube.mutable_plan() = burst.plan();

    if (burst.detectors().empty()) {
        return cube;
    }

    // Chirp parameters from the plan (defaults if the plan carries none)
    double chirp_bandwidth_hz = kDefaultChirpBandwidthHz;
    double chirp_duration_s = kDefaultChirpDurationS;
    if (burst.plan().chirps_size() > 0) {
        chirp_bandwidth_hz = burst.plan().chirps(0).chirp_bandwidth_hz();
        chirp_duration_s = burst.plan().chirps(0).chirp_duration_s();
    }

    setup_range_axis(cube, chirp_bandwidth_hz, chirp_duration_s);

    // Sum range profiles over the burst's chirps: wavelength -> detector -> bins
    std::map<double, std::map<uint32_t, std::vector<std::complex<double>>>> by_wavelength;
    std::map<double, std::map<uint32_t, size_t>> chirp_counts;

    for (const auto& det_burst : burst.detectors()) {
        uint32_t det_idx = det_burst.detector_index();

        for (const auto& chirp : det_burst.chirps()) {
            if (chirp.wavelengths_m().empty()) {
                throw std::invalid_argument(
                    "Chirp must have at least one wavelength specified");
            }

            // Process chirp - returns full N-bin FFT spectrum
            std::vector<int32_t> samples(chirp.samples().begin(), chirp.samples().end());
            auto full_fft = process_chirp(samples, chirp_bandwidth_hz, chirp_duration_s);

            // Bins [0, N/2) contain positive frequencies for range processing
            size_t num_range_bins = full_fft.size() / 2;

            for (double wl : chirp.wavelengths_m()) {
                auto& accum = by_wavelength[wl][det_idx];
                if (accum.empty()) {
                    accum.assign(num_range_bins, std::complex<double>(0.0, 0.0));
                }
                for (size_t i = 0; i < num_range_bins && i < accum.size(); ++i) {
                    accum[i] += full_fft[i];
                }
                chirp_counts[wl][det_idx]++;
            }
        }
    }

    // Convert summed data to protobuf structure (chirp-averaged)
    for (const auto& [wavelength, det_map] : by_wavelength) {
        auto* wl_data = cube.add_wavelengths();
        wl_data->set_wavelength_m(wavelength);

        for (const auto& [det_idx, bins] : det_map) {
            auto* det_data = wl_data->add_detectors();
            det_data->set_detector_index(det_idx);

            size_t chirp_count = chirp_counts[wavelength][det_idx];
            double scale = 1.0 / static_cast<double>(chirp_count > 0 ? chirp_count : 1);
            for (const auto& bin : bins) {
                auto* out_bin = det_data->add_bins();
                out_bin->set_real(bin.real() * scale);
                out_bin->set_imag(bin.imag() * scale);
            }
        }
    }

    latency_stats_.total_us = to_microseconds(Clock::now() - t_total_start);
    return cube;
}

void RangeDopplerProcessor::setup_range_axis(RangeCube& cube,
                                             double chirp_bandwidth_hz,
                                             double chirp_duration_s) const {
    auto* range_axis = cube.mutable_range_axis();
    range_axis->set_unit("m");
    double range_offset = compute_range_offset_from_reference_freq(
        config_.iq_demod_config().reference_freq_hz(),
        chirp_bandwidth_hz,
        chirp_duration_s);
    range_axis->set_start(range_offset);
    range_axis->set_step(compute_range_per_bin(chirp_bandwidth_hz,
                                               chirp_duration_s,
                                               sample_rate_hz_,
                                               config_.range_fft_size()));
    range_axis->set_count(config_.range_fft_size() / 2);
}

aurora::shared::MeasurementTensor RangeDopplerProcessor::to_measurement_tensor(
    const RangeCube& range_cube) {
    return measurement_builder_.build(range_cube);
}

double RangeDopplerProcessor::compute_velocity_resolution(
    double wavelength_m,
    double chirp_duration_s) {
    // Velocity resolution from triangular chirp:
    // The minimum detectable frequency shift is ~1/T_chirp
    // This maps to velocity: Δv = λ × Δf / 2 = λ / (2 × T_chirp)
    // For up/down comparison, we get factor of 2 improvement:
    // Δv = λ / (4 × T_chirp)
    return wavelength_m / (4.0 * chirp_duration_s);
}

// -----------------------------------------------------------------------------
// Doppler processing helpers
// -----------------------------------------------------------------------------

std::map<uint32_t, aurora::acquisition::ChirpPlan::Direction>
RangeDopplerProcessor::build_direction_map(
    const aurora::acquisition::SequencePlan& plan) {

    std::map<uint32_t, aurora::acquisition::ChirpPlan::Direction> directions;
    for (const auto& chirp_plan : plan.chirps()) {
        directions[chirp_plan.chirp_index()] = chirp_plan.direction();
    }
    return directions;
}

RangeDopplerProcessor::DirectionDataMap
RangeDopplerProcessor::sum_chirps_by_direction(
    const aurora::simulation::Burst& burst,
    const aurora::acquisition::ChirpPlan& chirp_plan) {

    using Direction = aurora::acquisition::ChirpPlan::Direction;
    DirectionDataMap result;

    auto directions = build_direction_map(burst.plan());

    for (const auto& det_burst : burst.detectors()) {
        uint32_t det_idx = det_burst.detector_index();

        for (const auto& chirp : det_burst.chirps()) {
            // Look up direction from plan
            Direction dir = Direction::ChirpPlan_Direction_DIRECTION_UNSPECIFIED;
            auto it = directions.find(chirp.chirp_index());
            if (it != directions.end()) {
                dir = it->second;
            }

            // Skip unspecified direction chirps
            if (dir == Direction::ChirpPlan_Direction_DIRECTION_UNSPECIFIED) {
                continue;
            }

            if (chirp.wavelengths_m().empty()) {
                throw std::invalid_argument(
                    "Chirp must have at least one wavelength specified");
            }

            // Process chirp - returns full N-bin FFT spectrum
            std::vector<int32_t> samples(chirp.samples().begin(), chirp.samples().end());
            auto full_fft = process_chirp(samples,
                                          chirp_plan.chirp_bandwidth_hz(),
                                          chirp_plan.chirp_duration_s());

            // Bins [0, N/2) contain positive frequencies for range processing
            size_t num_range_bins = full_fft.size() / 2;

            for (double wl : chirp.wavelengths_m()) {
                auto& accum = result[wl][det_idx][dir];
                if (accum.bins.empty()) {
                    accum.bins.resize(num_range_bins, std::complex<double>(0.0, 0.0));
                }
                for (size_t i = 0; i < num_range_bins && i < accum.bins.size(); ++i) {
                    accum.bins[i] += full_fft[i];
                }
                accum.count++;
            }
        }
    }

    return result;
}

void RangeDopplerProcessor::combine_up_down_chirps(
    const DirectionDataMap& data,
    RangeCube& cube) {

    using Direction = aurora::acquisition::ChirpPlan::Direction;

    auto average = [](const DirectionSum& acc, size_t i) {
        return acc.bins[i] / static_cast<double>(acc.count > 0 ? acc.count : 1);
    };

    for (const auto& [wl, det_map] : data) {
        auto* wl_data = cube.add_wavelengths();
        wl_data->set_wavelength_m(wl);
        auto* up_data = cube.add_up_chirps();
        up_data->set_wavelength_m(wl);
        auto* down_data = cube.add_down_chirps();
        down_data->set_wavelength_m(wl);

        for (const auto& [det_idx, dir_map] : det_map) {
            auto* det_data = wl_data->add_detectors();
            det_data->set_detector_index(det_idx);
            auto* up_det = up_data->add_detectors();
            up_det->set_detector_index(det_idx);
            auto* down_det = down_data->add_detectors();
            down_det->set_detector_index(det_idx);

            auto up_it = dir_map.find(Direction::ChirpPlan_Direction_DIRECTION_UP);
            auto down_it = dir_map.find(Direction::ChirpPlan_Direction_DIRECTION_DOWN);

            if (up_it == dir_map.end() || down_it == dir_map.end()) {
                throw std::invalid_argument(
                    "process_doppler requires both up and down chirps for each "
                    "wavelength/detector combination");
            }

            const auto& up = up_it->second;
            const auto& down = down_it->second;

            size_t num_bins = std::min(up.bins.size(), down.bins.size());
            for (size_t i = 0; i < num_bins; ++i) {
                std::complex<double> up_avg = average(up, i);
                std::complex<double> down_avg = average(down, i);

                // Doppler-corrected value is the average of up and down
                std::complex<double> corrected = (up_avg + down_avg) / 2.0;

                auto* out_bin = det_data->add_bins();
                out_bin->set_real(corrected.real());
                out_bin->set_imag(corrected.imag());

                auto* up_bin = up_det->add_bins();
                up_bin->set_real(up_avg.real());
                up_bin->set_imag(up_avg.imag());

                auto* down_bin = down_det->add_bins();
                down_bin->set_real(down_avg.real());
                down_bin->set_imag(down_avg.imag());
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Main Doppler processing entry point
// -----------------------------------------------------------------------------

RangeCube RangeDopplerProcessor::process_doppler(
    const aurora::simulation::Burst& burst) {

    latency_stats_ = LatencyStats{};
    auto t_total_start = Clock::now();

    RangeCube cube;
    cube.set_burst_id(burst.burst_id());
    cube.set_timestamp_s(burst.timestamp_s());
    *cube.mutable_plan() = burst.plan();

    if (burst.detectors().empty()) {
        return cube;
    }
    if (burst.plan().chirps_size() == 0) {
        throw std::invalid_argument("Burst plan must have chirp plans");
    }

    const auto& chirp_plan = burst.plan().chirps(0);
    setup_range_axis(cube, chirp_plan.chirp_bandwidth_hz(), chirp_plan.chirp_duration_s());

    auto direction_data = sum_chirps_by_direction(burst, chirp_plan);
    combine_up_down_chirps(direction_data, cube);

    latency_stats_.total_us = to_microseconds(Clock::now() - t_total_start);
    return cube;
}

// -----------------------------------------------------------------------------
// Peak-based Doppler processing
// -----------------------------------------------------------------------------

void RangeDopplerProcessor::set_peak_doppler_config(const PeakDopplerConfig& config) {
    peak_doppler_config_ = config;
    peak_doppler_processor_.reset();  // Force re-creation with new config
}

const PeakDopplerConfig& RangeDopplerProcessor::peak_doppler_config() const {
    return peak_doppler_config_;
}

std::vector<PeakDopplerResult> RangeDopplerProcessor::process_doppler_peaks(
    const aurora::simulation::Burst& burst,
    double wavelength_m) {

    latency_stats_ = LatencyStats{};
    auto t_total_start = Clock::now();

    std::vector<PeakDopplerResult> results;

    if (burst.detectors().empty()) {
        return results;
    }
    if (burst.plan().chirps_size() == 0) {
        throw std::invalid_argument("Burst plan must have chirp plans");
    }

    const auto& chirp_plan = burst.plan().chirps(0);
    double chirp_bandwidth = chirp_plan.chirp_bandwidth_hz();
    double chirp_duration = chirp_plan.chirp_duration_s();

    // Create or update PeakDopplerProcessor if needed
    if (!peak_doppler_processor_) {
        peak_doppler_processor_ = std::make_unique<PeakDopplerProcessor>(
            peak_doppler_config_, chirp_plan, sample_rate_hz_, wavelength_m);
    }

    using Direction = aurora::acquisition::ChirpPlan::Direction;
    auto directions = build_direction_map(burst.plan());

    // Find matched up/down chirps for each detector
    for (const auto& det_burst : burst.detectors()) {
        uint32_t det_idx = det_burst.detector_index();

        std::vector<std::vector<std::complex<double>>> up_ffts;
        std::vector<std::vector<std::complex<double>>> down_ffts;

        for (const auto& chirp : det_burst.chirps()) {
            Direction dir = Direction::ChirpPlan_Direction_DIRECTION_UNSPECIFIED;
            auto it = directions.find(chirp.chirp_index());
            if (it != directions.end()) {
                dir = it->second;
            }

            // Peak-based Doppler needs the full spectrum: close-range targets
            // with velocity can have negative down-chirp beat frequencies that
            // appear in bins [N/2, N).
            std::vector<int32_t> samples(chirp.samples().begin(), chirp.samples().end());
            auto full_fft = process_chirp(samples, chirp_bandwidth, chirp_duration);

            if (dir == Direction::ChirpPlan_Direction_DIRECTION_UP) {
                up_ffts.push_back(std::move(full_fft));
            } else if (dir == Direction::ChirpPlan_Direction_DIRECTION_DOWN) {
                down_ffts.push_back(std::move(full_fft));
            }
        }

        size_t num_pairs = std::min(up_ffts.size(), down_ffts.size());
        if (num_pairs > 0) {
            PeakDopplerResult result;
            result.set_detector_index(det_idx);
            result.set_wavelength_index(0);  // Single wavelength for now

            for (size_t i = 0; i < num_pairs; ++i) {
                auto peaks = peak_doppler_processor_->process(up_ffts[i], down_ffts[i]);
                for (auto& peak : peaks) {
                    *result.add_peaks() = std::move(peak);
                }
            }

            if (result.peaks_size() > 0) {
                results.push_back(std::move(result));
            }
        }
    }

    latency_stats_.total_us = to_microseconds(Clock::now() - t_total_start);
    return results;
}

RangeCube RangeDopplerProcessor::peaks_to_range_cube(
    const std::vector<PeakDopplerResult>& peak_results,
    const aurora::simulation::Burst& burst) {

    RangeCube cube;
    cube.set_burst_id(burst.burst_id());
    cube.set_timestamp_s(burst.timestamp_s());

    *cube.mutable_plan() = burst.plan();

    if (burst.detectors().empty() || peak_results.empty()) {
        return cube;
    }
    if (burst.plan().chirps_size() == 0) {
        throw std::invalid_argument("Burst plan must have chirp plans");
    }

    // Get chirp parameters for range axis setup
    const auto& chirp_plan = burst.plan().chirps(0);
    setup_range_axis(cube, chirp_plan.chirp_bandwidth_hz(), chirp_plan.chirp_duration_s());
    double range_per_bin = cube.range_axis().step();
    size_t num_range_bins = config_.range_fft_size() / 2;

    // Wavelength from the first chirp of the first detector
    double wavelength = kDefaultWavelengthM;
    if (!burst.detectors(0).chirps().empty() &&
        burst.detectors(0).chirps(0).wavelengths_m_size() > 0) {
        wavelength = burst.detectors(0).chirps(0).wavelengths_m(0);
    }

    auto* wl_data = cube.add_wavelengths();
    wl_data->set_wavelength_m(wavelength);

    // Collect all detectors from peak results
    std::map<uint32_t, std::vector<const DopplerCorrectedPeak*>> peaks_by_detector;
    for (const auto& result : peak_results) {
        for (int i = 0; i < result.peaks_size(); ++i) {
            peaks_by_detector[result.detector_index()].push_back(&result.peaks(i));
        }
    }

    // Create detector data with peaks in appropriate bins
    for (const auto& [det_idx, peaks] : peaks_by_detector) {
        auto* det_data = wl_data->add_detectors();
        det_data->set_detector_index(det_idx);

        // Initialize all bins to zero
        for (size_t bin = 0; bin < num_range_bins; ++bin) {
            auto* out_bin = det_data->add_bins();
            out_bin->set_real(0.0);
            out_bin->set_imag(0.0);
        }

        // Place peaks in appropriate bins
        for (const auto* peak : peaks) {
            // Convert range to bin index
            uint32_t bin_idx = static_cast<uint32_t>(peak->range_m() / range_per_bin);
            if (bin_idx < num_range_bins) {
                auto* bin = det_data->mutable_bins(bin_idx);
                // Sum if multiple peaks land in same bin
                bin->set_real(bin->real() + peak->amplitude().real());
                bin->set_imag(bin->imag() + peak->amplitude().imag());
            }
        }
    }

    return cube;
}

}  // namespace aurora::dsp
