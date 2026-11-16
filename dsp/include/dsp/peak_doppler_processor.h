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

#pragma once

#include <complex>
#include <span>
#include <vector>

#include "doppler_processing.pb.h"
#include "sequence_plan.pb.h"

namespace aurora::dsp {

/// Peak-based Doppler processing for high-velocity targets.
///
/// The standard bin-by-bin averaging of up/down chirps fails when Doppler shift
/// causes peaks to appear at different bins. This processor:
/// 1. Detects peaks in each chirp's range FFT independently
/// 2. Associates peak pairs between up and down chirps
/// 3. Computes true range and velocity from peak positions
///
/// ### Physics Background
///
/// For FMCW with up-chirp and down-chirp:
/// ```
/// f_up   = f_beat + f_doppler = (2BR)/(cT) + (2v)/lambda
/// f_down = f_beat - f_doppler = (2BR)/(cT) - (2v)/lambda
/// ```
///
/// Solving for range and velocity:
/// ```
/// f_beat   = (f_up + f_down) / 2  ->  R = (f_up + f_down) * c * T / (4B)
/// f_doppler = (f_up - f_down) / 2  ->  v = (f_up - f_down) * lambda / 4
/// ```
class PeakDopplerProcessor {
  public:
    /// Construct with configuration and chirp parameters.
    ///
    /// @param config Peak detection and association parameters (all fields required)
    /// @param chirp_plan Chirp parameters (bandwidth, duration) from sequence plan
    /// @param sample_rate_hz ADC sample rate in Hz
    /// @param wavelength_m Operating wavelength in meters
    /// @throws std::invalid_argument if any required config/param field is zero or invalid
    PeakDopplerProcessor(const PeakDopplerConfig& config,
                         const aurora::acquisition::ChirpPlan& chirp_plan,
                         double sample_rate_hz,
                         double wavelength_m);

    /// Process up and down chirp FFTs into Doppler-corrected peaks.
    ///
    /// Peaks are detected over the full spectrum, paired between the two
    /// chirps, and converted to range and velocity. Pairs that resolve to a
    /// non-positive range (the conjugate image of a real-valued spectrum) are
    /// discarded.
    ///
    /// @param up_fft Complex FFT output from up-chirp
    /// @param down_fft Complex FFT output from down-chirp
    /// @return Vector of Doppler-corrected peaks with true range and velocity
    std::vector<DopplerCorrectedPeak> process(
        std::span<const std::complex<double>> up_fft,
        std::span<const std::complex<double>> down_fft) const;

    /// Detect peaks in a single chirp's FFT.
    ///
    /// Uses local maxima finding with SNR threshold.
    ///
    /// @param fft_data Complex FFT output
    /// @return Vector of detected peaks
    std::vector<ChirpPeak> detect_peaks(
        std::span<const std::complex<double>> fft_data) const;

    /// Associate peaks between up and down chirps.
    ///
    /// Uses greedy assignment based on velocity plausibility and magnitude
    /// similarity.
    ///
    /// @param up_peaks Peaks detected in up-chirp
    /// @param down_peaks Peaks detected in down-chirp
    /// @return Vector of matched peak pairs
    std::vector<PeakPair> associate_peaks(const std::vector<ChirpPeak>& up_peaks,
                                          const std::vector<ChirpPeak>& down_peaks) const;

    /// Convert peak pair to range/velocity.
    ///
    /// @param pair Matched up/down peak pair
    /// @return Doppler-corrected peak with true range and velocity
    DopplerCorrectedPeak compute_range_velocity(const PeakPair& pair) const;

    /// Get frequency resolution (Hz per bin).
    double frequency_resolution(size_t fft_size) const;

    /// Get maximum unambiguous velocity.
    double max_unambiguous_velocity() const;

    /// Update configuration.
    void set_config(const PeakDopplerConfig& config) { config_ = config; }

    /// Get current configuration.
    const PeakDopplerConfig& config() const { return config_; }

    /// Get chirp plan.
    const aurora::acquisition::ChirpPlan& chirp_plan() const { return chirp_plan_; }

    /// Get sample rate.
    double sample_rate_hz() const { return sample_rate_hz_; }

    /// Get wavelength.
    double wavelength_m() const { return wavelength_m_; }

  private:
    PeakDopplerConfig config_;
    aurora::acquisition::ChirpPlan chirp_plan_;
    double sample_rate_hz_;
    double wavelength_m_;

    /// Compute association score between two peaks.
    /// Based on magnitude similarity and velocity plausibility.
    double compute_association_score(const ChirpPeak& up_peak,
                                     const ChirpPeak& down_peak) const;

    /// Convert bin index to unsigned frequency (for positive bins only).
    ///
    /// @param bin FFT bin index (0 to fft_size-1)
    /// @param fft_size Full FFT size (N bins)
    /// @return Unsigned frequency in Hz (always positive)
    double bin_to_frequency(uint32_t bin, size_t fft_size) const;

    /// Convert bin index to signed frequency.
    ///
    /// Handles the full FFT spectrum where:
    ///   Bins [0, N/2):   positive frequencies (0 to +f_s/2)
    ///   Bins [N/2, N):   negative frequencies (-f_s/2 to 0)
    ///
    /// @param bin FFT bin index (0 to fft_size-1)
    /// @param fft_size Full FFT size (N bins)
    /// @return Signed frequency in Hz (positive for bins 0..N/2-1, negative for bins N/2..N-1)
    double bin_to_signed_frequency(uint32_t bin, size_t fft_size) const;

    /// Convert frequency pair to range.
    double frequencies_to_range(double f_up, double f_down) const;

    /// Convert frequency pair to velocity.
    double frequencies_to_velocity(double f_up, double f_down) const;

    /// Estimate noise floor from magnitude spectrum.
    double estimate_noise_floor(const std::vector<double>& magnitudes) const;
};

}  // namespace aurora::dsp
