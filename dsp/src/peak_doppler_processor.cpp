#include "dsp/peak_doppler_processor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace aurora::dsp {

namespace {
constexpr double kSpeedOfLight = 299792458.0;  // m/s
}  // namespace

PeakDopplerProcessor::PeakDopplerProcessor(const PeakDopplerConfig& config,
                                           const aurora::acquisition::ChirpPlan& chirp_plan,
                                           double sample_rate_hz,
                                           double wavelength_m)
    : config_(config), chirp_plan_(chirp_plan),
      sample_rate_hz_(sample_rate_hz), wavelength_m_(wavelength_m) {
    // Validate config - all required fields must be set
    if (config_.peak_threshold_db() <= 0) {
        throw std::invalid_argument(
            "PeakDopplerConfig.peak_threshold_db must be positive");
    }
    if (config_.min_peak_separation() == 0) {
        throw std::invalid_argument(
            "PeakDopplerConfig.min_peak_separation must be positive");
    }
    if (config_.max_velocity_mps() <= 0) {
        throw std::invalid_argument(
            "PeakDopplerConfig.max_velocity_mps must be positive");
    }
    if (config_.association_threshold() <= 0 || config_.association_threshold() > 1) {
        throw std::invalid_argument(
            "PeakDopplerConfig.association_threshold must be in (0, 1]");
    }

    // Validate chirp params - all fields must be set
    if (sample_rate_hz_ <= 0) {
        throw std::invalid_argument(
            "sample_rate_hz must be positive");
    }
    if (chirp_plan_.chirp_bandwidth_hz() <= 0) {
        throw std::invalid_argument(
            "ChirpPlan.chirp_bandwidth_hz must be positive");
    }
    if (chirp_plan_.chirp_duration_s() <= 0) {
        throw std::invalid_argument(
            "ChirpPlan.chirp_duration_s must be positive");
    }
    if (wavelength_m_ <= 0) {
        throw std::invalid_argument(
            "wavelength_m must be positive");
    }

    // Validate aliasing correction config
    if (config_.enable_aliasing_correction()) {
        if (!config_.has_aliasing_detection_ratio()) {
            throw std::invalid_argument(
                "PeakDopplerConfig.aliasing_detection_ratio must be set when "
                "enable_aliasing_correction is true");
        }
        if (!config_.has_aliasing_range_ratio_threshold()) {
            throw std::invalid_argument(
                "PeakDopplerConfig.aliasing_range_ratio_threshold must be set when "
                "enable_aliasing_correction is true");
        }
        if (config_.aliasing_detection_ratio() <= 0 ||
            config_.aliasing_detection_ratio() >= 1) {
            throw std::invalid_argument(
                "PeakDopplerConfig.aliasing_detection_ratio must be in (0, 1)");
        }
        if (config_.aliasing_range_ratio_threshold() <= 1) {
            throw std::invalid_argument(
                "PeakDopplerConfig.aliasing_range_ratio_threshold must be > 1");
        }
    }
}

std::vector<DopplerCorrectedPeak> PeakDopplerProcessor::process(
    std::span<const std::complex<double>> up_fft,
    std::span<const std::complex<double>> down_fft) const {

    // Step 1: Detect peaks in each chirp
    auto up_peaks = detect_peaks(up_fft);
    auto down_peaks = detect_peaks(down_fft);

    // Step 2: Associate peaks between up and down chirps
    auto pairs = associate_peaks(up_peaks, down_peaks);


    // Step 3: Compute range/velocity for each pair.
    //
    // The ADC samples a real-valued signal, so every beat tone at +f has a
    // conjugate image at -f. Pairing the two images with each other yields
    // the same target reflected through the origin (negative range, negated
    // velocity). Only the interpretation with positive range is physical.
    std::vector<DopplerCorrectedPeak> results;
    results.reserve(pairs.size());

    for (const auto& pair : pairs) {
        auto peak = compute_range_velocity(pair);
        if (peak.range_m() <= 0.0) {
            continue;
        }
        results.push_back(std::move(peak));
    }

    return results;
}

std::vector<ChirpPeak> PeakDopplerProcessor::detect_peaks(
    std::span<const std::complex<double>> fft_data) const {

    std::vector<ChirpPeak> peaks;
    const size_t n = fft_data.size();

    if (n == 0) {
        return peaks;
    }

    // Get configuration (validated in constructor)
    double threshold_db = config_.peak_threshold_db();
    uint32_t min_sep = config_.min_peak_separation();

    // Compute magnitude spectrum
    std::vector<double> magnitudes(n);
    for (size_t i = 0; i < n; ++i) {
        magnitudes[i] = std::abs(fft_data[i]);
    }

    // Estimate noise floor
    double noise_floor = estimate_noise_floor(magnitudes);

    // Threshold in linear scale (from dB relative to noise floor)
    double threshold = noise_floor * std::pow(10.0, threshold_db / 10.0);

    // Apply absolute minimum threshold if configured
    // (useful for noise-free simulations where noise floor is near zero)
    if (config_.min_peak_magnitude() > 0) {
        threshold = std::max(threshold, config_.min_peak_magnitude());
    }

    // Scan full spectrum (N bins) for peaks.
    //
    // FFT layout:
    //   Bins [0, N/2):   positive frequencies (0 to +f_s/2)
    //   Bins [N/2, N):   negative frequencies (-f_s/2 to 0)
    //
    // For close-range targets with velocity, the down-chirp beat frequency
    // can go negative (when f_doppler > f_beat), and those peaks appear
    // in the negative frequency region [N/2, N).
    //
    // Note: We skip the DC bin (0) and Nyquist bin (N/2) boundaries in the
    // positive region, and also handle the wrap-around at bin N-1 to bin 0.
    // However, since DC and Nyquist are typically at boundaries and peaks
    // at exact DC or Nyquist are rare in practice, we use a simpler approach:
    // we just exclude the first and last min_sep bins from peak candidacy
    // to avoid edge effects.

    // Find local maxima above threshold in both positive and negative regions
    for (size_t i = min_sep; i < n - min_sep; ++i) {
        if (magnitudes[i] < threshold) {
            continue;
        }

        // Check if local maximum
        bool is_max = true;
        for (size_t j = i - min_sep; j <= i + min_sep; ++j) {
            if (j != i && magnitudes[j] >= magnitudes[i]) {
                is_max = false;
                break;
            }
        }

        if (is_max) {
            ChirpPeak peak;
            peak.set_bin_index(static_cast<uint32_t>(i));
            // Use signed frequency for full-spectrum support
            peak.set_frequency_hz(bin_to_signed_frequency(static_cast<uint32_t>(i), n));
            peak.set_magnitude(magnitudes[i]);
            peak.set_phase_rad(std::arg(fft_data[i]));
            peak.mutable_value()->set_real(fft_data[i].real());
            peak.mutable_value()->set_imag(fft_data[i].imag());
            peaks.push_back(peak);
        }
    }

    return peaks;
}

std::vector<PeakPair> PeakDopplerProcessor::associate_peaks(
    const std::vector<ChirpPeak>& up_peaks,
    const std::vector<ChirpPeak>& down_peaks) const {

    const size_t n_up = up_peaks.size();
    const size_t n_down = down_peaks.size();

    if (n_up == 0 || n_down == 0) {
        return {};
    }

    // Get configuration (validated in constructor)
    double assoc_threshold = config_.association_threshold();

    // Compute score matrix
    // scores[i][j] = association score between up_peaks[i] and down_peaks[j]
    std::vector<std::vector<double>> scores(n_up, std::vector<double>(n_down, 0.0));

    for (size_t i = 0; i < n_up; ++i) {
        for (size_t j = 0; j < n_down; ++j) {
            scores[i][j] = compute_association_score(up_peaks[i], down_peaks[j]);
        }
    }

    // Greedy assignment: repeatedly pick best remaining pair
    std::vector<bool> up_used(n_up, false);
    std::vector<bool> down_used(n_down, false);
    std::vector<PeakPair> pairs;

    while (true) {
        // Find best unused pair
        double best_score = assoc_threshold;
        size_t best_i = 0, best_j = 0;
        bool found = false;

        for (size_t i = 0; i < n_up; ++i) {
            if (up_used[i]) {
                continue;
            }
            for (size_t j = 0; j < n_down; ++j) {
                if (down_used[j]) {
                    continue;
                }
                if (scores[i][j] > best_score) {
                    best_score = scores[i][j];
                    best_i = i;
                    best_j = j;
                    found = true;
                }
            }
        }

        if (!found) {
            break;
        }

        // Create pair
        PeakPair pair;
        *pair.mutable_up_peak() = up_peaks[best_i];
        *pair.mutable_down_peak() = down_peaks[best_j];
        pair.set_association_score(best_score);
        pairs.push_back(pair);

        up_used[best_i] = true;
        down_used[best_j] = true;
    }

    // Optionally handle unpaired peaks
    if (config_.allow_unpaired_peaks()) {
        // Create synthetic pairs with assumed zero velocity
        // (peak appears at same bin in both chirps)
        for (size_t i = 0; i < n_up; ++i) {
            if (!up_used[i]) {
                PeakPair pair;
                *pair.mutable_up_peak() = up_peaks[i];
                // Synthetic down peak at same frequency
                ChirpPeak synthetic_down;
                synthetic_down.set_bin_index(up_peaks[i].bin_index());
                synthetic_down.set_frequency_hz(up_peaks[i].frequency_hz());
                synthetic_down.set_magnitude(up_peaks[i].magnitude() * 0.5);  // Reduced confidence
                synthetic_down.set_phase_rad(up_peaks[i].phase_rad());
                *pair.mutable_down_peak() = synthetic_down;
                pair.set_association_score(0.3);  // Low confidence
                pairs.push_back(pair);
            }
        }
    }

    return pairs;
}

DopplerCorrectedPeak PeakDopplerProcessor::compute_range_velocity(
    const PeakPair& pair) const {

    double f_up = pair.up_peak().frequency_hz();
    double f_down = pair.down_peak().frequency_hz();
    bool aliasing_corrected = false;

    // Aliasing detection and correction for real-valued ADC sampling
    //
    // With real ADC, negative frequencies alias to positive: cos(-ωt) = cos(ωt)
    // For a target with velocity v at range R:
    //   f_up   = f_beat + f_doppler (always positive for positive range)
    //   f_down = f_beat - f_doppler (can go negative when f_doppler > f_beat)
    //
    // Aliasing occurs when: v > R * B * lambda / (c * T)
    // The observed f_down_aliased = |f_down_true| = f_doppler - f_beat (positive)
    //
    // Detection: Compare both interpretations and choose the more plausible one
    // Correction: use f_down_corrected = -f_down_observed
    if (config_.enable_aliasing_correction() && f_up > 0 && f_down >= 0) {
        // Compute range with both interpretations
        double range_positive = frequencies_to_range(f_up, f_down);
        double range_negative = frequencies_to_range(f_up, -f_down);

        // Only consider correction if the negative interpretation gives a
        // physically meaningful (positive) range
        if (range_negative > 0) {
            // Config values validated in constructor when aliasing correction enabled
            double aliasing_ratio = config_.aliasing_detection_ratio();
            double range_ratio_threshold = config_.aliasing_range_ratio_threshold();

            bool should_correct = false;

            // Heuristic 1: Range ratio check
            // If the positive interpretation gives a range significantly larger
            // than the negative interpretation, aliasing likely occurred.
            // The range ratio directly reflects the frequency aliasing:
            // range_positive/range_negative = (f_up + f_down)/(f_up - f_down)
            // For aliased signals where f_down should be negative, this ratio
            // will be much larger than for valid signals.
            // For non-aliased signals, ratio ≈ 1.0 (f_down and f_up similar magnitude)
            // For aliased signals, ratio > threshold (f_down much smaller than f_up)
            if (range_positive > range_ratio_threshold * range_negative) {
                should_correct = true;
            }

            // Heuristic 2: Frequency ratio gating (prevents false positives)
            // Only apply correction if f_down is not too close to f_up
            // This prevents spurious corrections when f_down ≈ f_up (stationary)
            // For aliased cases, typically f_down < f_up (unless severely aliased)
            // We use a generous threshold to catch most aliasing scenarios
            if (f_down > f_up * (1.0 - aliasing_ratio)) {
                // f_down is close to f_up - this doesn't look like aliasing
                // unless the range ratio is very compelling
                if (range_positive < 2.0 * range_negative) {
                    should_correct = false;
                }
            }

            // Heuristic 3: Max range check (if configured)
            // If positive interpretation exceeds max but negative doesn't,
            // the negative is more plausible
            if (config_.has_expected_max_range_m()) {
                if (range_positive > config_.expected_max_range_m() &&
                    range_negative <= config_.expected_max_range_m()) {
                    should_correct = true;
                }
            }

            if (should_correct) {
                f_down = -f_down;
                aliasing_corrected = true;
            }
        }
    }

    DopplerCorrectedPeak result;
    result.set_range_m(frequencies_to_range(f_up, f_down));
    result.set_velocity_mps(frequencies_to_velocity(f_up, f_down));
    result.set_aliasing_corrected(aliasing_corrected);

    // Amplitude: geometric mean
    double mag = std::sqrt(pair.up_peak().magnitude() * pair.down_peak().magnitude());
    double phase = 0.5 * (pair.up_peak().phase_rad() + pair.down_peak().phase_rad());
    result.mutable_amplitude()->set_real(mag * std::cos(phase));
    result.mutable_amplitude()->set_imag(mag * std::sin(phase));

    // SNR estimate (simplified - ratio to 1 as reference)
    result.set_snr_db(20.0 * std::log10(mag > 0 ? mag : 1e-12));

    result.set_association_score(pair.association_score());
    result.set_up_bin(pair.up_peak().bin_index());
    result.set_down_bin(pair.down_peak().bin_index());

    return result;
}

double PeakDopplerProcessor::compute_association_score(const ChirpPeak& up_peak,
                                                       const ChirpPeak& down_peak) const {

    // Get max velocity (validated in constructor)
    double max_vel = config_.max_velocity_mps();

    double f_up = up_peak.frequency_hz();
    double f_down = down_peak.frequency_hz();

    // Consider both aliased and non-aliased interpretations.
    // For real ADC sampling, negative frequencies alias to positive.
    // The down-chirp frequency might be aliased if f_doppler > f_beat.
    //
    // Non-aliased: velocity = (f_up - f_down) * λ / 4
    // Aliased:     velocity = (f_up + f_down) * λ / 4  (f_down was negative)
    //
    // We compute both and use the one that gives a valid, plausible velocity.

    double velocity_normal = frequencies_to_velocity(f_up, f_down);
    double velocity_aliased = frequencies_to_velocity(f_up, -f_down);

    // Check which interpretation(s) give valid velocities
    bool normal_valid = std::abs(velocity_normal) <= max_vel;
    bool aliased_valid = std::abs(velocity_aliased) <= max_vel && f_down >= 0;

    // If neither is valid, reject
    if (!normal_valid && !aliased_valid) {
        return 0.0;
    }

    // Pick the velocity interpretation
    // If both valid, prefer the one with higher velocity (more likely to be correct
    // when aliasing occurs, since non-aliased gives artificially low velocity)
    double velocity;
    if (normal_valid && aliased_valid) {
        // Both plausible - use range consistency to decide
        // For aliased case: range = (f_up - f_down) / 2 * c * T / (2B)
        // For normal case: range = (f_up + f_down) / 2 * c * T / (2B)
        // Choose the one giving a more reasonable range (positive, not too large)
        double range_normal = frequencies_to_range(f_up, f_down);
        double range_aliased = frequencies_to_range(f_up, -f_down);

        // Prefer aliased interpretation if it gives a smaller, positive range
        // (aliasing typically happens at close range)
        if (range_aliased > 0 && range_normal > 0) {
            velocity = (range_aliased < range_normal) ? velocity_aliased : velocity_normal;
        } else if (range_aliased > 0) {
            velocity = velocity_aliased;
        } else {
            velocity = velocity_normal;
        }
    } else if (aliased_valid) {
        velocity = velocity_aliased;
    } else {
        velocity = velocity_normal;
    }

    // Velocity plausibility score (favor lower velocities slightly)
    double vel_score = 1.0 - 0.5 * std::abs(velocity) / max_vel;

    // Magnitude similarity score
    double up_mag = up_peak.magnitude();
    double down_mag = down_peak.magnitude();
    double mag_ratio = std::min(up_mag, down_mag) / std::max(up_mag, down_mag);
    double mag_score = mag_ratio;  // 1.0 = identical, 0.0 = very different

    // Combined score
    return 0.4 * vel_score + 0.6 * mag_score;
}

double PeakDopplerProcessor::bin_to_frequency(uint32_t bin, size_t fft_size) const {
    // fft_size is the full FFT size (N bins).
    // f = bin * sample_rate / fft_size
    return static_cast<double>(bin) * sample_rate_hz_ /
           static_cast<double>(fft_size);
}

double PeakDopplerProcessor::bin_to_signed_frequency(uint32_t bin, size_t fft_size) const {
    // Convert bin to signed frequency for full-spectrum processing.
    //
    // FFT output layout for N bins:
    //   Bins [0, N/2):   positive frequencies (0 to +f_s/2)
    //   Bins [N/2, N):   negative frequencies (-f_s/2 to 0)
    //
    // For bins >= N/2, the frequency is negative:
    //   f = (bin - N) * sample_rate / N

    double freq = static_cast<double>(bin) * sample_rate_hz_ /
                  static_cast<double>(fft_size);

    // Bins >= N/2 represent negative frequencies
    if (bin >= fft_size / 2) {
        freq -= sample_rate_hz_;
    }

    return freq;
}

double PeakDopplerProcessor::frequency_resolution(size_t fft_size) const {
    // fft_size is the full FFT size (N bins).
    return sample_rate_hz_ / static_cast<double>(fft_size);
}

double PeakDopplerProcessor::frequencies_to_range(double f_up, double f_down) const {
    // R = (f_up + f_down) * c * T / (4B)
    // where T = chirp_duration, B = chirp_bandwidth
    double f_beat = (f_up + f_down) / 2.0;
    return f_beat * kSpeedOfLight * chirp_plan_.chirp_duration_s() /
           (2.0 * chirp_plan_.chirp_bandwidth_hz());
}

double PeakDopplerProcessor::frequencies_to_velocity(double f_up, double f_down) const {
    // v = (f_up - f_down) * lambda / 4
    double f_doppler = (f_up - f_down) / 2.0;
    return f_doppler * wavelength_m_ / 2.0;
}

double PeakDopplerProcessor::max_unambiguous_velocity() const {
    // Maximum unambiguous velocity is limited by the FFT bin spacing
    // For triangular chirp, max velocity where peaks don't wrap around
    // v_max = c * f_s / (4 * B) approximately
    return kSpeedOfLight * sample_rate_hz_ /
           (4.0 * chirp_plan_.chirp_bandwidth_hz());
}

double PeakDopplerProcessor::estimate_noise_floor(
    const std::vector<double>& magnitudes) const {

    if (magnitudes.empty()) {
        return 1e-12;  // Return small value for empty input
    }

    // Estimate noise floor as median of lower half of sorted magnitudes
    std::vector<double> sorted_mags = magnitudes;
    size_t n = sorted_mags.size();
    std::nth_element(sorted_mags.begin(), sorted_mags.begin() + n / 2, sorted_mags.end());
    double noise_floor = sorted_mags[n / 2];

    // Ensure minimum noise floor to avoid division by zero
    return std::max(noise_floor, 1e-12);
}

}  // namespace aurora::dsp
