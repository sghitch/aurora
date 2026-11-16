/**
 * @file phase_gradient_estimator.cpp
 * @brief Implementation of the phase gradient estimation base class.
 */

#include "detection/phase_gradient_estimator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace aurora::detection {

PhaseGradientEstimator::PhaseGradientEstimator(
    double focal_length_m, double noise_floor_linear)
    : focal_length_m_(focal_length_m),
      noise_floor_linear_(noise_floor_linear) {}

GradientFitResult PhaseGradientEstimator::fit_phase_gradient(
    const std::vector<double>& phases,
    const std::vector<double>& k_values,
    const std::vector<double>& weights) const {

    GradientFitResult result;
    result.set_valid(false);

    const size_t n = phases.size();
    if (n < 2 || k_values.size() != n || weights.size() != n) {
        result.set_angle_rad(0.0);
        result.set_uncertainty_rad(M_PI);
        result.set_residual_rad(M_PI);
        result.set_r_squared(0.0);
        return result;
    }

    // Weighted least squares: phi = -k * x + c
    double sum_w = 0.0;
    double sum_wk = 0.0;
    double sum_wk2 = 0.0;
    double sum_wphi = 0.0;
    double sum_wkphi = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double w = weights[i] * weights[i];
        double k = k_values[i];
        double phi = phases[i];

        sum_w += w;
        sum_wk += w * k;
        sum_wk2 += w * k * k;
        sum_wphi += w * phi;
        sum_wkphi += w * k * phi;
    }

    double det = sum_w * sum_wk2 - sum_wk * sum_wk;
    if (std::abs(det) < 1e-12) {
        result.set_angle_rad(0.0);
        result.set_uncertainty_rad(M_PI);
        result.set_residual_rad(M_PI);
        result.set_r_squared(0.0);
        return result;
    }

    double c = (sum_wk2 * sum_wphi - sum_wk * sum_wkphi) / det;
    double neg_x = (sum_w * sum_wkphi - sum_wk * sum_wphi) / det;
    double x_position = -neg_x;

    double angle_rad = std::atan(x_position / focal_length_m_);

    // Compute per-sample residuals and statistics
    double ssr = 0.0;
    double sst = 0.0;
    double mean_phi = sum_wphi / sum_w;
    std::vector<double> per_sample_residuals(n);

    for (size_t i = 0; i < n; ++i) {
        double w = weights[i] * weights[i];
        double predicted = -k_values[i] * x_position + c;
        double residual = phases[i] - predicted;
        per_sample_residuals[i] = residual;
        ssr += w * residual * residual;
        sst += w * (phases[i] - mean_phi) * (phases[i] - mean_phi);
    }

    double residual_rms = std::sqrt(ssr / sum_w);
    double r_squared = (sst > 1e-12) ? (1.0 - ssr / sst) : 0.0;

    double k_variance = sum_wk2 / sum_w - (sum_wk / sum_w) * (sum_wk / sum_w);
    double sigma_x = (k_variance > 1e-12) ?
        residual_rms / std::sqrt(sum_w * k_variance) : M_PI;
    double uncertainty_rad = sigma_x / focal_length_m_;

    result.set_angle_rad(angle_rad);
    result.set_uncertainty_rad(std::min(uncertainty_rad, M_PI));
    result.set_residual_rad(residual_rms);
    result.set_r_squared(std::max(0.0, r_squared));
    for (double res : per_sample_residuals) {
        result.add_per_sample_residuals(res);
    }
    result.set_valid(true);

    return result;
}

std::vector<double> PhaseGradientEstimator::unwrap_phases(
    const std::vector<double>& wrapped_phases,
    const std::vector<double>& k_values,
    double prior_angle_rad) const {

    const size_t n = wrapped_phases.size();
    if (n == 0) return {};

    // Local sequential unwrapping: sort by K, unwrap continuously
    std::vector<size_t> sorted_indices(n);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
        [&k_values](size_t a, size_t b) { return k_values[a] < k_values[b]; });

    std::vector<double> unwrapped_seq(n);
    unwrapped_seq[sorted_indices[0]] = wrapped_phases[sorted_indices[0]];

    for (size_t i = 1; i < n; ++i) {
        size_t curr_idx = sorted_indices[i];
        size_t prev_idx = sorted_indices[i - 1];

        double prev_unwrapped = unwrapped_seq[prev_idx];
        double curr_wrapped = wrapped_phases[curr_idx];

        double diff = curr_wrapped - prev_unwrapped;
        int32_t num_wraps = static_cast<int32_t>(std::round(diff / (2.0 * M_PI)));
        unwrapped_seq[curr_idx] = curr_wrapped - num_wraps * 2.0 * M_PI;
    }

    // Compute K-span to decide if prior-based is needed
    double k_min = k_values[sorted_indices[0]];
    double k_max = k_values[sorted_indices[n - 1]];
    double k_span = k_max - k_min;
    double dk_per_sample = (n > 1) ? k_span / (n - 1) : 0.0;

    double max_resolvable_angle = (dk_per_sample > 0)
        ? M_PI / (dk_per_sample * focal_length_m_)
        : M_PI;

    if (std::abs(prior_angle_rad) > max_resolvable_angle * 0.5) {
        // Prior-based unwrapping for large angles
        double x_prior = focal_length_m_ * std::tan(prior_angle_rad);

        std::vector<double> unwrapped_prior(n);
        for (size_t i = 0; i < n; ++i) {
            double expected_phase = -k_values[i] * x_prior;
            double wrapped = wrapped_phases[i];

            double diff = wrapped - expected_phase;
            int32_t num_wraps = static_cast<int32_t>(std::round(diff / (2.0 * M_PI)));
            unwrapped_prior[i] = wrapped - num_wraps * 2.0 * M_PI;
        }
        return unwrapped_prior;
    }

    return unwrapped_seq;
}

double PhaseGradientEstimator::compute_uncertainty(
    double k_span,
    double snr_linear,
    uint32_t num_samples) const {

    if (k_span <= 0.0 || snr_linear <= 0.0 || num_samples == 0) {
        return M_PI;
    }

    // Cramer-Rao lower bound: σ_θ ≈ 1 / (SNR · k_span · f · sqrt(N))
    double uncertainty = 1.0 / (snr_linear * k_span * focal_length_m_ * std::sqrt(num_samples));
    return std::min(uncertainty, M_PI);
}

double PhaseGradientEstimator::estimate_extent_from_residuals(
    const std::vector<double>& residuals,
    const std::vector<double>& k_values,
    const std::vector<double>& magnitudes) const {

    if (residuals.empty() || k_values.empty() || magnitudes.empty()) {
        return 0.0;
    }

    const size_t n = residuals.size();
    if (k_values.size() != n || magnitudes.size() != n) {
        return 0.0;
    }

    double k_min = std::abs(k_values[0]);
    double k_max = std::abs(k_values[0]);
    for (size_t i = 1; i < n; ++i) {
        double k_abs = std::abs(k_values[i]);
        k_min = std::min(k_min, k_abs);
        k_max = std::max(k_max, k_abs);
    }

    double k_span = k_max - k_min;
    if (k_span < 1e-6) {
        return 0.0;
    }

    // Fit Gaussian envelope for magnitude roll-off
    double sum_w = 0.0;
    double sum_wk2 = 0.0;
    double sum_wk4 = 0.0;
    double sum_wlogm = 0.0;
    double sum_wk2logm = 0.0;

    double max_mag = *std::max_element(magnitudes.begin(), magnitudes.end());
    if (max_mag <= 0.0) {
        return 0.0;
    }

    for (size_t i = 0; i < n; ++i) {
        if (magnitudes[i] <= 0.0) continue;

        double k2 = k_values[i] * k_values[i];
        double log_m = std::log(magnitudes[i] / max_mag);
        double w = magnitudes[i];

        sum_w += w;
        sum_wk2 += w * k2;
        sum_wk4 += w * k2 * k2;
        sum_wlogm += w * log_m;
        sum_wk2logm += w * k2 * log_m;
    }

    if (sum_w <= 0.0) {
        return 0.0;
    }

    double det = sum_w * sum_wk4 - sum_wk2 * sum_wk2;
    if (std::abs(det) < 1e-20) {
        return 0.0;
    }

    double neg_b = (sum_w * sum_wk2logm - sum_wk2 * sum_wlogm) / det;

    if (neg_b >= 0.0) {
        return 0.0;  // Point target
    }

    double b = -neg_b;
    double extent_rad = std::sqrt(b) / (k_span + 1.0);

    return std::min(extent_rad, M_PI / 6.0);
}

double PhaseGradientEstimator::wrap_phase(double phase) {
    while (phase > M_PI) phase -= 2.0 * M_PI;
    while (phase < -M_PI) phase += 2.0 * M_PI;
    return phase;
}

}  // namespace aurora::detection
