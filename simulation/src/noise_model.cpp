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

#include "simulation/noise_model.h"

#include <cmath>

namespace aurora::simulation {

NoiseModel::NoiseModel(const NoiseModelConfig& config,
                       const AdcModelConfig& adc_config)
    : config_(config), adc_config_(adc_config) {}

std::vector<double> NoiseModel::add_noise(
    const std::vector<double>& signal,
    double optical_power_w,
    double bandwidth_hz,
    std::mt19937_64& rng) const {

    std::vector<double> noisy_signal = signal;

    double total_variance = compute_total_noise_variance(optical_power_w,
                                                          bandwidth_hz);

    if (total_variance <= 0.0) {
        return noisy_signal;
    }

    double total_std = std::sqrt(total_variance);
    std::normal_distribution<double> noise_dist(0.0, total_std);

    for (auto& sample : noisy_signal) {
        sample += noise_dist(rng);
    }

    return noisy_signal;
}

double NoiseModel::compute_shot_noise_std(double optical_power_w,
                                           double bandwidth_hz) const {
    if (!config_.enable_shot_noise()) {
        return 0.0;
    }

    // I_dc = R * P (photocurrent)
    double i_dc = config_.detector_responsivity_a_per_w() * optical_power_w;

    // sigma^2 = 2 * q * I_dc * B
    double variance = 2.0 * kElectronCharge * i_dc * bandwidth_hz;

    return std::sqrt(variance);
}

double NoiseModel::compute_thermal_noise_std(double bandwidth_hz) const {
    if (!config_.enable_thermal_noise()) {
        return 0.0;
    }

    if (config_.transimpedance_ohm() <= 0.0) {
        return 0.0;
    }

    // sigma^2 = 4 * k * T * B / R
    double variance = 4.0 * kBoltzmann * config_.temperature_k() *
                      bandwidth_hz / config_.transimpedance_ohm();

    return std::sqrt(variance);
}

double NoiseModel::compute_rin_noise_std(double optical_power_w,
                                          double bandwidth_hz) const {
    if (!config_.enable_rin_noise()) {
        return 0.0;
    }

    // Convert RIN from dB/Hz to linear
    double rin_linear = std::pow(10.0, config_.rin_db_per_hz() / 10.0);

    // sigma^2 = P^2 * RIN_linear * B
    double variance = optical_power_w * optical_power_w * rin_linear *
                      bandwidth_hz;

    return std::sqrt(variance);
}

double NoiseModel::compute_quantization_noise_std() const {
    if (!config_.enable_quantization_noise()) {
        return 0.0;
    }

    int adc_bits = static_cast<int>(adc_config_.bits());
    double voltage_range = adc_config_.input_range_v();

    if (adc_bits <= 0 || voltage_range <= 0.0) {
        return 0.0;
    }

    // V_LSB = V_range / 2^bits
    double v_lsb = voltage_range / static_cast<double>(1 << adc_bits);

    // sigma^2 = V_LSB^2 / 12
    double variance = v_lsb * v_lsb / 12.0;

    return std::sqrt(variance);
}

double NoiseModel::compute_dark_current_noise_std(double bandwidth_hz) const {
    if (!config_.enable_dark_current()) {
        return 0.0;
    }

    // sigma^2 = 2 * q * I_dark * B
    double variance = 2.0 * kElectronCharge *
                      config_.detector_dark_current_a() * bandwidth_hz;

    return std::sqrt(variance);
}

double NoiseModel::compute_total_noise_variance(double optical_power_w,
                                                 double bandwidth_hz) const {
    double shot_std = compute_shot_noise_std(optical_power_w, bandwidth_hz);
    double thermal_std = compute_thermal_noise_std(bandwidth_hz);
    double rin_std = compute_rin_noise_std(optical_power_w, bandwidth_hz);
    double quant_std = compute_quantization_noise_std();
    double dark_std = compute_dark_current_noise_std(bandwidth_hz);

    // Total variance = sum of individual variances
    return shot_std * shot_std +
           thermal_std * thermal_std +
           rin_std * rin_std +
           quant_std * quant_std +
           dark_std * dark_std;
}

double NoiseModel::compute_snr_db(double signal_power,
                                   double optical_power_w,
                                   double bandwidth_hz) const {
    double noise_variance = compute_total_noise_variance(optical_power_w,
                                                          bandwidth_hz);

    if (noise_variance <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    double snr_linear = signal_power / noise_variance;
    return 10.0 * std::log10(snr_linear);
}

}  // namespace aurora::simulation
