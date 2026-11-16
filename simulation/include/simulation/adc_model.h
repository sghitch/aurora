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

#ifndef AURORA_SIMULATION_ADC_MODEL_H_
#define AURORA_SIMULATION_ADC_MODEL_H_

/**
 * @file adc_model.h
 * @brief ADC model for realistic signal digitization.
 *
 * The AdcModel class models ADC behavior including sampling, quantization,
 * DC offset, gain error, and timing jitter. All parameters are configurable
 * via AdcModelConfig protobuf.
 *
 * ### Quantization
 *
 * The ADC converts voltage to an N-bit signed integer:
 *   code = clamp((V - offset) * (1 + gain_error) * 2^(N-1) / V_range, min, max)
 *
 * ### Timing Jitter
 *
 * Aperture jitter adds Gaussian uncertainty to sample times:
 *   t_actual = t_nominal + N(0, jitter_rms)
 *
 * @see AdcModelConfig for protobuf configuration
 * @see NoiseModel for noise injection
 */

#include <cstdint>
#include <random>
#include <vector>

#include "adc_config.pb.h"

namespace aurora::simulation {

// Import AdcModelConfig from shared namespace for convenience
using aurora::shared::AdcModelConfig;

/**
 * @brief Models ADC digitization with realistic impairments.
 *
 * Converts analog signals to digital samples, applying DC offset,
 * gain error, and timing jitter. Supports multi-channel operation
 * with per-channel calibration differences.
 *
 * ### Example Usage
 * @code
 * AdcModelConfig config;
 * config.set_sample_rate_hz(125e6);
 * config.set_bits(14);
 * config.set_input_range_v(2.0);
 * config.set_dc_offset_v(0.001);    // 1 mV offset
 * config.set_gain_error(0.005);     // 0.5% gain error
 * config.set_aperture_jitter_s(100e-15);  // 100 fs jitter
 *
 * AdcModel adc(config);
 *
 * std::mt19937_64 rng(42);
 * auto samples = adc.sample(analog_signal, signal_sample_rate, rng);
 * @endcode
 */
class AdcModel {
  public:
    /**
     * @brief Construct an ADC model from protobuf configuration.
     * @param config ADC model configuration
     */
    explicit AdcModel(const AdcModelConfig& config);

    /**
     * @brief Convert analog signal to ADC samples.
     *
     * Resamples the input signal to the ADC sample rate, applies
     * impairments (DC offset, gain error, jitter), and quantizes
     * to the configured bit depth.
     *
     * @param analog_signal Input signal in Volts
     * @param signal_sample_rate_hz Sample rate of input signal in Hz
     * @param rng Random number generator for jitter (modified in place)
     * @return Vector of quantized samples
     */
    std::vector<int16_t> sample(
        const std::vector<double>& analog_signal,
        double signal_sample_rate_hz,
        std::mt19937_64& rng) const;

    /**
     * @brief Convert analog signal for a specific channel.
     *
     * Applies per-channel gain, offset, and delay variations
     * in addition to the global impairments.
     *
     * @param analog_signal Input signal in Volts
     * @param signal_sample_rate_hz Sample rate of input signal in Hz
     * @param channel_index Channel index (0-based)
     * @param rng Random number generator for jitter (modified in place)
     * @return Vector of quantized samples
     */
    std::vector<int16_t> sample_channel(
        const std::vector<double>& analog_signal,
        double signal_sample_rate_hz,
        uint32_t channel_index,
        std::mt19937_64& rng) const;

    /**
     * @brief Get ADC sample rate.
     * @return Sample rate in Hz
     */
    double sample_rate_hz() const { return config_.sample_rate_hz(); }

    /**
     * @brief Get ADC bit depth.
     * @return Number of bits
     */
    uint32_t bits() const { return config_.bits(); }

    /**
     * @brief Get ADC input range.
     * @return Full-scale input range in Volts (peak-to-peak)
     */
    double input_range_v() const { return config_.input_range_v(); }

    /**
     * @brief Compute number of samples for a given duration.
     * @param duration_s Duration in seconds
     * @return Number of samples (rounded up)
     */
    uint32_t samples_per_duration(double duration_s) const;

    /**
     * @brief Get the LSB (least significant bit) voltage.
     * @return LSB voltage in Volts
     */
    double lsb_voltage() const;

    /**
     * @brief Get the configuration.
     * @return Reference to the ADC configuration
     */
    const AdcModelConfig& config() const { return config_; }

  private:
    AdcModelConfig config_;

    /**
     * @brief Apply timing jitter to a sample time.
     * @param nominal_time Nominal sample time in seconds
     * @param rng Random number generator
     * @return Jittered sample time in seconds
     */
    double jittered_sample_time(double nominal_time, std::mt19937_64& rng) const;

    /**
     * @brief Quantize a voltage to an ADC code.
     * @param voltage Input voltage
     * @param dc_offset DC offset to apply
     * @param gain_factor Gain factor (1 + gain_error)
     * @return Quantized ADC code
     */
    int16_t quantize(double voltage, double dc_offset, double gain_factor) const;

    /**
     * @brief Interpolate signal value at a given time.
     * @param signal Input signal samples
     * @param sample_rate_hz Signal sample rate
     * @param time_s Time at which to interpolate
     * @return Interpolated signal value
     */
    double interpolate(const std::vector<double>& signal,
                       double sample_rate_hz,
                       double time_s) const;
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_ADC_MODEL_H_
