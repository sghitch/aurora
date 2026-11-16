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

#ifndef AURORA_SIMULATION_CHIRP_GENERATOR_H_
#define AURORA_SIMULATION_CHIRP_GENERATOR_H_

/**
 * @file chirp_generator.h
 * @brief FMCW chirp waveform generation.
 *
 * The ChirpGenerator class generates linear FMCW chirp waveforms for
 * simulation. It produces both the time-domain waveform and the
 * instantaneous phase needed for coherent detection.
 *
 * ### Linear FMCW Chirp
 *
 * Instantaneous frequency:
 *   f(t) = f_0 + (B/T) * t
 *
 * Phase:
 *   φ(t) = 2π * (f_0 * t + (B / 2T) * t²)
 *
 * Where:
 *   - f_0 = start frequency (optical carrier)
 *   - B = chirp bandwidth
 *   - T = chirp duration
 *
 * @see FmcwSimulator for full signal chain simulation
 */

#include <cstdint>
#include <vector>

#include "sequence_plan.pb.h"

namespace aurora::simulation {

/**
 * @brief Generates linear FMCW chirp waveforms.
 *
 * Creates time-domain chirp signals with configurable bandwidth, duration,
 * and sweep direction.
 */
class ChirpGenerator {
  public:
    /// Speed of light in m/s
    static constexpr double kSpeedOfLight = 299792458.0;

    /**
     * @brief Generate chirp phase samples.
     *
     * Computes the instantaneous phase φ(t) for each sample time.
     * The phase wraps modulo 2π for numerical stability.
     *
     * @param chirp_plan Chirp parameters (duration, bandwidth, direction)
     * @param sample_rate_hz Sample rate in Hz
     * @return Vector of phase values in radians
     */
    static std::vector<double> generate_phase(
        const aurora::acquisition::ChirpPlan& chirp_plan,
        double sample_rate_hz);

    /**
     * @brief Generate chirp amplitude samples (cosine of phase).
     *
     * Produces the real-valued chirp waveform cos(φ(t)).
     *
     * @param chirp_plan Chirp parameters
     * @param sample_rate_hz Sample rate in Hz
     * @return Vector of amplitude values in range [-1, 1]
     */
    static std::vector<double> generate_waveform(
        const aurora::acquisition::ChirpPlan& chirp_plan,
        double sample_rate_hz);

    /**
     * @brief Compute beat frequency for a given range.
     *
     * The beat frequency encodes target range:
     *   f_beat = (2 * B * R) / (c * T)
     *
     * @param range_m Target range in meters
     * @param bandwidth_hz Chirp bandwidth in Hz
     * @param duration_s Chirp duration in seconds
     * @return Beat frequency in Hz
     */
    static double compute_beat_frequency(double range_m,
                                          double bandwidth_hz,
                                          double duration_s);

    /**
     * @brief Compute range from beat frequency.
     *
     * Inverse of compute_beat_frequency:
     *   R = (c * T * f_beat) / (2 * B)
     *
     * @param beat_frequency_hz Beat frequency in Hz
     * @param bandwidth_hz Chirp bandwidth in Hz
     * @param duration_s Chirp duration in seconds
     * @return Range in meters
     */
    static double compute_range(double beat_frequency_hz,
                                 double bandwidth_hz,
                                 double duration_s);

    /**
     * @brief Compute Doppler frequency for a given velocity.
     *
     * Doppler shift from target velocity:
     *   f_d = 2 * v / λ
     *
     * @param velocity_mps Radial velocity in m/s (positive = moving away)
     * @param wavelength_m Wavelength in meters
     * @return Doppler frequency in Hz
     */
    static double compute_doppler_frequency(double velocity_mps,
                                             double wavelength_m);

    /**
     * @brief Compute number of samples for a chirp.
     *
     * @param duration_s Chirp duration in seconds
     * @param sample_rate_hz Sample rate in Hz
     * @return Number of samples (rounded up)
     */
    static uint32_t compute_num_samples(double duration_s,
                                         double sample_rate_hz);
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_CHIRP_GENERATOR_H_
