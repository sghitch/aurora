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

#ifndef AURORA_SIMULATION_NOISE_MODEL_H_
#define AURORA_SIMULATION_NOISE_MODEL_H_

/**
 * @file noise_model.h
 * @brief Noise model for realistic FMCW signal simulation.
 *
 * The NoiseModel class injects realistic noise sources into simulated
 * FMCW signals, including shot noise, thermal noise, relative intensity
 * noise (RIN), quantization noise, and dark current noise.
 *
 * ### Noise Sources
 *
 * | Source           | Model                      | Formula                    |
 * |------------------|----------------------------|----------------------------|
 * | Shot noise       | Poisson -> Gaussian        | sigma^2 = 2*q*I_dc*B       |
 * | Thermal noise    | Gaussian (Johnson-Nyquist) | sigma^2 = 4*k*T*B/R        |
 * | RIN              | Gaussian, power-dependent  | sigma^2 = P^2*RIN_lin*B    |
 * | Quantization     | Uniform                    | sigma^2 = V_LSB^2/12       |
 * | Dark current     | DC + shot noise            | sigma^2 = 2*q*I_dark*B     |
 *
 * ### SNR Calculation
 *
 * Signal-to-noise ratio:
 *   SNR = P_signal^2 / sigma^2_total
 *   SNR_dB = 10 * log10(SNR)
 *
 * @see NoiseModelConfig for protobuf configuration
 * @see FmcwSimulator for integration with signal simulation
 */

#include <random>
#include <vector>

#include "adc_config.pb.h"
#include "noise_model.pb.h"

namespace aurora::simulation {

// Import AdcModelConfig from shared namespace for convenience
using aurora::shared::AdcModelConfig;

/**
 * @brief Models realistic noise sources for FMCW signal simulation.
 *
 * Computes and injects noise from multiple sources: shot noise, thermal
 * noise, RIN, quantization, and dark current. Each source can be
 * individually enabled/disabled.
 *
 * ### Example Usage
 * @code
 * NoiseModelConfig config;
 * config.set_enable_shot_noise(true);
 * config.set_enable_thermal_noise(true);
 * config.set_enable_quantization_noise(true);
 * config.set_detector_responsivity_a_per_w(0.9);
 * config.set_transimpedance_ohm(50.0);
 * config.set_temperature_k(300.0);
 *
 * AdcModelConfig adc_config;
 * adc_config.set_bits(14);
 * adc_config.set_input_range_v(2.0);
 *
 * NoiseModel noise(config, adc_config);
 *
 * std::mt19937_64 rng(42);
 * auto noisy_signal = noise.add_noise(signal, 1e-6, 125e6, rng);
 * @endcode
 */
class NoiseModel {
  public:
    /// Boltzmann constant in J/K
    static constexpr double kBoltzmann = 1.380649e-23;

    /// Electron charge in Coulombs
    static constexpr double kElectronCharge = 1.602176634e-19;

    /**
     * @brief Construct a noise model from protobuf configurations.
     * @param config Noise model configuration
     * @param adc_config ADC configuration for quantization noise
     */
    NoiseModel(const NoiseModelConfig& config, const AdcModelConfig& adc_config);

    /**
     * @brief Add all enabled noise sources to a signal.
     *
     * Computes the total noise variance from all enabled sources and adds
     * Gaussian noise to each sample. Uses the provided RNG for reproducible
     * results.
     *
     * @param signal Input signal samples
     * @param optical_power_w Optical power at detector in Watts
     * @param bandwidth_hz Noise bandwidth in Hz
     * @param rng Random number generator (modified in place)
     * @return Signal with noise added
     */
    std::vector<double> add_noise(
        const std::vector<double>& signal,
        double optical_power_w,
        double bandwidth_hz,
        std::mt19937_64& rng) const;

    /**
     * @brief Compute shot noise standard deviation.
     *
     * Shot noise arises from the discrete nature of photon detection:
     *   sigma^2 = 2 * q * R * P * B
     *
     * @param optical_power_w Optical power in Watts
     * @param bandwidth_hz Noise bandwidth in Hz
     * @return Standard deviation in Amperes
     */
    double compute_shot_noise_std(double optical_power_w,
                                   double bandwidth_hz) const;

    /**
     * @brief Compute thermal (Johnson-Nyquist) noise standard deviation.
     *
     * Thermal noise from the transimpedance amplifier:
     *   sigma^2 = 4 * k * T * B / R_load
     *
     * @param bandwidth_hz Noise bandwidth in Hz
     * @return Standard deviation in Amperes
     */
    double compute_thermal_noise_std(double bandwidth_hz) const;

    /**
     * @brief Compute RIN noise standard deviation.
     *
     * Laser intensity fluctuations that scale with power:
     *   sigma^2 = P^2 * 10^(RIN_dB/10) * B
     *
     * @param optical_power_w Optical power in Watts
     * @param bandwidth_hz Noise bandwidth in Hz
     * @return Standard deviation in Watts
     */
    double compute_rin_noise_std(double optical_power_w,
                                  double bandwidth_hz) const;

    /**
     * @brief Compute quantization noise standard deviation.
     *
     * ADC quantization noise (uniform distribution):
     *   sigma^2 = V_LSB^2 / 12
     *
     * @return Standard deviation in Volts
     */
    double compute_quantization_noise_std() const;

    /**
     * @brief Compute dark current noise standard deviation.
     *
     * Dark current contributes shot noise even without light:
     *   sigma^2 = 2 * q * I_dark * B
     *
     * @param bandwidth_hz Noise bandwidth in Hz
     * @return Standard deviation in Amperes
     */
    double compute_dark_current_noise_std(double bandwidth_hz) const;

    /**
     * @brief Compute total noise variance.
     *
     * Sum of all enabled noise source variances.
     *
     * @param optical_power_w Optical power in Watts
     * @param bandwidth_hz Noise bandwidth in Hz
     * @return Total noise variance
     */
    double compute_total_noise_variance(double optical_power_w,
                                         double bandwidth_hz) const;

    /**
     * @brief Compute signal-to-noise ratio in dB.
     *
     * SNR = 10 * log10(signal_power / noise_variance)
     *
     * @param signal_power Signal power (same units as noise)
     * @param optical_power_w Optical power for noise calculation
     * @param bandwidth_hz Noise bandwidth in Hz
     * @return SNR in dB
     */
    double compute_snr_db(double signal_power,
                          double optical_power_w,
                          double bandwidth_hz) const;

    /**
     * @brief Get the noise model configuration.
     * @return Reference to the noise model configuration
     */
    const NoiseModelConfig& config() const { return config_; }

    /**
     * @brief Get the ADC configuration.
     * @return Reference to the ADC configuration
     */
    const AdcModelConfig& adc_config() const { return adc_config_; }

  private:
    NoiseModelConfig config_;
    AdcModelConfig adc_config_;
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_NOISE_MODEL_H_
