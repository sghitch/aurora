#include "detection/doppler_extractor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace aurora::detection {

namespace {

// Normalize angle to [-pi, pi]
double normalize_angle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

}  // namespace

DopplerExtractor::DopplerExtractor(
    const DetectionConfig& config,
    double wavelength_m,
    double chirp_duration_s)
    : config_(config),
      wavelength_m_(wavelength_m),
      chirp_duration_s_(chirp_duration_s) {

    // Validate wavelength
    if (wavelength_m_ <= 0.0) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
                             "wavelength_m must be positive");
    }

    // Validate chirp duration
    if (chirp_duration_s_ <= 0.0) {
        throw DetectionError(DetectionError::Code::CONFIGURATION_ERROR,
                             "chirp_duration_s must be positive");
    }

    // Extract configuration
    enabled_ = config.enable_doppler();
    min_velocity_mps_ = config.min_velocity_mps();

    // Compute maximum unambiguous velocity
    // Phase difference wraps at ±π, so:
    // |Δφ| ≤ π  =>  |v| ≤ λ / (4 × T_chirp)
    max_unambiguous_velocity_ = wavelength_m_ / (4.0 * chirp_duration_s_);

    // Velocity resolution (minimum detectable velocity change)
    // Based on minimum detectable phase difference (~0.1 rad at high SNR)
    // v_resolution = Δφ_min × λ / (4π)
    constexpr double kMinPhaseResolution = 0.1;  // radians
    velocity_resolution_ = kMinPhaseResolution * wavelength_m_ / (4.0 * M_PI);
}

void DopplerExtractor::validate_inputs(
    const std::vector<std::complex<double>>& up_samples,
    const std::vector<std::complex<double>>& down_samples,
    const std::vector<double>& weights) const {

    if (up_samples.empty()) {
        throw DetectionError(DetectionError::Code::INVALID_MEASUREMENT,
                             "up_samples cannot be empty");
    }

    if (up_samples.size() != down_samples.size()) {
        throw DetectionError(DetectionError::Code::INVALID_MEASUREMENT,
                             "up_samples and down_samples must have same size");
    }

    if (up_samples.size() != weights.size()) {
        throw DetectionError(DetectionError::Code::INVALID_MEASUREMENT,
                             "samples and weights must have same size");
    }

    // Check for at least one non-zero weight
    double total_weight = 0.0;
    for (double w : weights) {
        if (w < 0.0) {
            throw DetectionError(DetectionError::Code::INVALID_MEASUREMENT,
                                 "weights must be non-negative");
        }
        total_weight += w;
    }

    if (total_weight <= 0.0) {
        throw DetectionError(DetectionError::Code::INSUFFICIENT_SIGNAL,
                             "total weight is zero");
    }
}

double DopplerExtractor::compute_phase_difference(
    const std::complex<double>& up_sample,
    const std::complex<double>& down_sample) const {

    // Phase difference = arg(up) - arg(down) = arg(up * conj(down))
    std::complex<double> product = up_sample * std::conj(down_sample);
    return std::arg(product);
}

double DopplerExtractor::phase_to_velocity(double phase_diff) const {
    // v = Δφ × λ / (4πT)
    // The phase difference accumulates over chirp duration, so T is in denominator
    return phase_diff * wavelength_m_ / (4.0 * M_PI * chirp_duration_s_);
}

double DopplerExtractor::velocity_to_phase(double velocity_mps) const {
    // Δφ = 4πvT / λ
    return 4.0 * M_PI * velocity_mps * chirp_duration_s_ / wavelength_m_;
}

void DopplerExtractor::compute_weighted_velocity(
    const std::vector<double>& phase_diffs,
    const std::vector<double>& weights,
    double& velocity_out,
    double& std_dev_out,
    double& total_weight_out) const {

    const size_t n = phase_diffs.size();

    // Compute total weight
    total_weight_out = 0.0;
    for (double w : weights) {
        total_weight_out += w;
    }

    // Weighted mean phase difference
    double weighted_phase_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        weighted_phase_sum += weights[i] * phase_diffs[i];
    }
    double mean_phase = weighted_phase_sum / total_weight_out;

    // Convert to velocity
    velocity_out = phase_to_velocity(mean_phase);

    // Weighted variance of velocity
    double weighted_sq_diff_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double v_i = phase_to_velocity(phase_diffs[i]);
        double diff = v_i - velocity_out;
        weighted_sq_diff_sum += weights[i] * diff * diff;
    }

    // Bessel-corrected weighted standard deviation
    if (n > 1) {
        // Sum of squared weights for correction
        double sum_weights_sq = 0.0;
        for (double w : weights) {
            sum_weights_sq += w * w;
        }
        double correction = total_weight_out -
                           sum_weights_sq / total_weight_out;

        if (correction > 0.0) {
            std_dev_out = std::sqrt(weighted_sq_diff_sum / correction);
        } else {
            std_dev_out = 0.0;
        }
    } else {
        std_dev_out = 0.0;
    }
}

bool DopplerExtractor::check_ambiguity(
    const std::vector<double>& phase_diffs,
    double mean_velocity) const {

    // Velocity is ambiguous if it's close to the ambiguity boundary
    // or if phase differences show high variance (suggesting wrapping)
    double threshold = 0.9 * max_unambiguous_velocity_;
    if (std::abs(mean_velocity) > threshold) {
        return true;
    }

    // Check for phase wrapping by looking at phase difference spread
    if (phase_diffs.size() < 2) {
        return false;
    }

    double min_phase = *std::min_element(phase_diffs.begin(), phase_diffs.end());
    double max_phase = *std::max_element(phase_diffs.begin(), phase_diffs.end());
    double phase_spread = max_phase - min_phase;

    // If phase spread is large (> π), likely have wrapping issues
    if (phase_spread > M_PI) {
        return true;
    }

    return false;
}

double DopplerExtractor::compute_confidence(
    double std_dev,
    double total_weight,
    size_t num_samples) const {

    // Base confidence from number of samples (more samples = higher confidence)
    double sample_factor = std::min(1.0, static_cast<double>(num_samples) / 8.0);

    // Weight factor (higher total weight = higher SNR = higher confidence)
    double weight_factor = 1.0 - std::exp(-total_weight / 4.0);

    // Consistency factor (lower std dev = more consistent = higher confidence)
    // Normalize std_dev by velocity resolution
    double normalized_std = std_dev / velocity_resolution_;
    double consistency_factor = std::exp(-normalized_std / 2.0);

    // Combine factors
    double confidence = sample_factor * weight_factor * consistency_factor;

    // Clamp to [0, 1]
    return std::max(0.0, std::min(1.0, confidence));
}

aurora::detection::DopplerResult DopplerExtractor::extract_velocity(
    const std::vector<std::complex<double>>& up_samples,
    const std::vector<std::complex<double>>& down_samples,
    const std::vector<double>& weights) const {

    // Validate inputs
    validate_inputs(up_samples, down_samples, weights);

    const size_t n = up_samples.size();

    // Compute phase differences for each sample pair
    std::vector<double> phase_diffs(n);
    for (size_t i = 0; i < n; ++i) {
        phase_diffs[i] = compute_phase_difference(up_samples[i], down_samples[i]);
    }

    // Compute weighted velocity estimate
    double velocity_mps = 0.0;
    double std_dev = 0.0;
    double total_weight = 0.0;
    compute_weighted_velocity(phase_diffs, weights, velocity_mps,
                              std_dev, total_weight);

    // Check for velocity ambiguity
    bool ambiguous = check_ambiguity(phase_diffs, velocity_mps);

    // Compute confidence
    double confidence = compute_confidence(std_dev, total_weight, n);

    // Build result
    aurora::detection::DopplerResult result;
    result.set_velocity_mps(velocity_mps);
    result.set_confidence(confidence);
    result.set_ambiguous(ambiguous);
    result.set_max_unambiguous_velocity_mps(max_unambiguous_velocity_);

    return result;
}

aurora::detection::DopplerResult DopplerExtractor::extract_velocity(
    const std::vector<std::complex<double>>& up_samples,
    const std::vector<std::complex<double>>& down_samples) const {

    // Use uniform weights
    std::vector<double> weights(up_samples.size(), 1.0);
    return extract_velocity(up_samples, down_samples, weights);
}

}  // namespace aurora::detection
