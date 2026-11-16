#include "simulation/adc_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aurora::simulation {

AdcModel::AdcModel(const AdcModelConfig& config) : config_(config) {
    if (config_.sample_rate_hz() <= 0) {
        throw std::invalid_argument("ADC sample rate must be positive");
    }
    if (config_.bits() == 0 || config_.bits() > 16) {
        throw std::invalid_argument("ADC bits must be between 1 and 16");
    }
    if (config_.input_range_v() <= 0) {
        throw std::invalid_argument("ADC input range must be positive");
    }
}

std::vector<int16_t> AdcModel::sample(
    const std::vector<double>& analog_signal,
    double signal_sample_rate_hz,
    std::mt19937_64& rng) const {

    if (analog_signal.empty()) {
        return {};
    }

    if (signal_sample_rate_hz <= 0) {
        throw std::invalid_argument("Signal sample rate must be positive");
    }

    // Compute signal duration and number of ADC samples
    double duration_s = static_cast<double>(analog_signal.size()) / signal_sample_rate_hz;
    uint32_t num_samples = samples_per_duration(duration_s);

    std::vector<int16_t> output(num_samples);
    double adc_period_s = 1.0 / config_.sample_rate_hz();

    // Get global impairments
    double dc_offset = config_.dc_offset_v();
    double gain_factor = 1.0 + config_.gain_error();

    for (uint32_t i = 0; i < num_samples; ++i) {
        double nominal_time = static_cast<double>(i) * adc_period_s;
        double sample_time = jittered_sample_time(nominal_time, rng);

        // Clamp sample time to valid range
        sample_time = std::clamp(sample_time, 0.0, duration_s);

        double voltage = interpolate(analog_signal, signal_sample_rate_hz, sample_time);
        output[i] = quantize(voltage, dc_offset, gain_factor);
    }

    return output;
}

std::vector<int16_t> AdcModel::sample_channel(
    const std::vector<double>& analog_signal,
    double signal_sample_rate_hz,
    uint32_t channel_index,
    std::mt19937_64& rng) const {

    if (analog_signal.empty()) {
        return {};
    }

    if (signal_sample_rate_hz <= 0) {
        throw std::invalid_argument("Signal sample rate must be positive");
    }

    // Compute signal duration and number of ADC samples
    double duration_s = static_cast<double>(analog_signal.size()) / signal_sample_rate_hz;
    uint32_t num_samples = samples_per_duration(duration_s);

    std::vector<int16_t> output(num_samples);
    double adc_period_s = 1.0 / config_.sample_rate_hz();

    // Get global impairments
    double dc_offset = config_.dc_offset_v();
    double gain_factor = 1.0 + config_.gain_error();

    // Apply per-channel variations if configured
    if (channel_index < static_cast<uint32_t>(config_.channel_offsets_v_size())) {
        dc_offset += config_.channel_offsets_v(channel_index);
    }
    if (channel_index < static_cast<uint32_t>(config_.channel_gains_size())) {
        gain_factor *= config_.channel_gains(channel_index);
    }

    // Get per-channel delay
    double channel_delay_s = 0.0;
    if (channel_index < static_cast<uint32_t>(config_.channel_delays_s_size())) {
        channel_delay_s = config_.channel_delays_s(channel_index);
    }

    for (uint32_t i = 0; i < num_samples; ++i) {
        double nominal_time = static_cast<double>(i) * adc_period_s;
        double sample_time = jittered_sample_time(nominal_time, rng);

        // Apply channel delay
        sample_time += channel_delay_s;

        // Clamp sample time to valid range
        sample_time = std::clamp(sample_time, 0.0, duration_s);

        double voltage = interpolate(analog_signal, signal_sample_rate_hz, sample_time);
        output[i] = quantize(voltage, dc_offset, gain_factor);
    }

    return output;
}

uint32_t AdcModel::samples_per_duration(double duration_s) const {
    if (duration_s <= 0) {
        return 0;
    }
    // Round up to ensure we capture the entire duration
    return static_cast<uint32_t>(std::ceil(duration_s * config_.sample_rate_hz()));
}

double AdcModel::lsb_voltage() const {
    // LSB = full scale range / 2^bits
    // For signed representation, range is +/- half the input range
    return config_.input_range_v() / static_cast<double>(1 << config_.bits());
}

double AdcModel::jittered_sample_time(double nominal_time, std::mt19937_64& rng) const {
    if (config_.aperture_jitter_s() <= 0) {
        return nominal_time;
    }

    std::normal_distribution<double> jitter_dist(0.0, config_.aperture_jitter_s());
    return nominal_time + jitter_dist(rng);
}

int16_t AdcModel::quantize(double voltage, double dc_offset, double gain_factor) const {
    // Remove DC offset and apply gain
    double adjusted = (voltage - dc_offset) * gain_factor;

    // Normalize to +/- 1 range (input_range_v is peak-to-peak)
    double half_range = config_.input_range_v() / 2.0;
    double normalized = adjusted / half_range;

    // Convert to ADC code
    // For N-bit signed ADC: range is [-2^(N-1), 2^(N-1) - 1]
    int32_t max_code = (1 << (config_.bits() - 1)) - 1;  // e.g., 8191 for 14-bit
    int32_t min_code = -(1 << (config_.bits() - 1));      // e.g., -8192 for 14-bit

    int32_t code = static_cast<int32_t>(std::round(normalized * (max_code + 1)));

    // Clamp to valid range
    code = std::clamp(code, min_code, max_code);

    return static_cast<int16_t>(code);
}

double AdcModel::interpolate(const std::vector<double>& signal,
                              double sample_rate_hz,
                              double time_s) const {
    if (signal.empty()) {
        return 0.0;
    }

    // Convert time to sample index (floating point)
    double sample_index = time_s * sample_rate_hz;

    // Handle boundary cases
    if (sample_index <= 0) {
        return signal.front();
    }
    if (sample_index >= static_cast<double>(signal.size() - 1)) {
        return signal.back();
    }

    // Linear interpolation between samples
    size_t idx_low = static_cast<size_t>(sample_index);
    size_t idx_high = idx_low + 1;
    double frac = sample_index - static_cast<double>(idx_low);

    return signal[idx_low] * (1.0 - frac) + signal[idx_high] * frac;
}

}  // namespace aurora::simulation
