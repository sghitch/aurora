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

#include <chrono>
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "burst.pb.h"
#include "sequence_plan.pb.h"
#include "doppler_processing.pb.h"
#include "dsp_config.pb.h"
#include "measurement_tensor.pb.h"
#include "range_cube.pb.h"

#include "dsp/measurement_builder.h"

namespace aurora::dsp {

class FftWrapper;
class IQDemodulator;
class PeakDopplerProcessor;

/// Orchestrates the complete DSP pipeline from raw ADC samples to MeasurementTensor.
///
/// This class is the main entry point for all DSP operations, processing raw
/// time-domain ADC samples through the following pipeline:
///
/// ```
/// ADC Samples -> DC Removal -> IQ Demodulation -> Windowing -> Range FFT -> RangeCube
///                                                                            |
///                                                                            v
///                                              K-Vector Annotation -> MeasurementTensor
/// ```
///
/// ### Physical Background
///
/// The range resolution is determined by chirp bandwidth:
///   ΔR = c / (2 × B)
///
/// The beat frequency maps to range:
///   R = (c × f_beat × T) / (2 × B)
///
/// FFT bin k maps to range:
///   R[k] = (c × k × f_s) / (2 × B × N_fft)
///
/// ### Example Usage
/// @code
///   // Create ForwardModel from detector geometry and optics
///   ForwardModel forward_model(detector, optics);
///
///   DspConfig dsp_config;
///   dsp_config.set_range_fft_size(2048);
///   dsp_config.mutable_range_window()->set_type(WindowConfig::WINDOW_HANN);
///
///   RangeDopplerProcessor processor(dsp_config, forward_model);
///
///   // Process a burst
///   RangeCube range_cube = processor.process_range(burst);
///
///   // Convert to MeasurementTensor with K-vectors
///   MeasurementTensor tensor = processor.to_measurement_tensor(range_cube);
/// @endcode
class RangeDopplerProcessor {
public:
    /// Processing latency statistics.
    struct LatencyStats {
        double preprocess_us = 0.0;   ///< Preprocessing time (DC removal) in microseconds
        double iq_demod_us = 0.0;     ///< IQ demodulation time in microseconds
        double window_us = 0.0;       ///< Windowing time in microseconds
        double fft_us = 0.0;          ///< FFT time in microseconds
        double total_us = 0.0;        ///< Total processing time in microseconds
    };

    /// Construct processor with configuration and ForwardModel.
    ///
    /// @param config DSP configuration (FFT sizes, windows, preprocessing options)
    /// @param forward_model Reference to ForwardModel for K-vector lookup
    /// @throws std::invalid_argument if configuration is invalid
    RangeDopplerProcessor(const DspConfig& config,
                          const aurora::forward::ForwardModel& forward_model);

    /// Destructor.
    ~RangeDopplerProcessor();

    // Non-copyable, non-movable (due to reference member)
    RangeDopplerProcessor(const RangeDopplerProcessor&) = delete;
    RangeDopplerProcessor& operator=(const RangeDopplerProcessor&) = delete;
    RangeDopplerProcessor(RangeDopplerProcessor&&) = delete;
    RangeDopplerProcessor& operator=(RangeDopplerProcessor&&) = delete;

    /// Process burst to range cube (primary interface).
    ///
    /// Performs the complete DSP pipeline:
    /// 1. DC removal (if enabled)
    /// 2. IQ demodulation
    /// 3. Window application
    /// 4. Range FFT
    ///
    /// @param burst Raw ADC samples organized by detector/chirp
    /// @return RangeCube with complex range profiles
    RangeCube process_range(const aurora::simulation::Burst& burst);

    /// Process burst with up/down chirp pairing for Doppler extraction.
    ///
    /// Uses the triangular FMCW technique: up-chirps and down-chirps see
    /// opposite Doppler shifts, allowing velocity extraction:
    ///
    ///   Up-chirp:   f_beat_up   = (2BR/cT) + (2v/λ)
    ///   Down-chirp: f_beat_down = (2BR/cT) - (2v/λ)
    ///
    /// By comparing:
    ///   Range (corrected):  f_avg = (f_up + f_down) / 2
    ///   Doppler frequency:  f_dop = (f_up - f_down) / 2
    ///   Velocity:           v = f_dop × λ / 2
    ///
    /// The returned RangeCube contains Doppler-corrected range data (averaging
    /// up and down to remove velocity bias).
    ///
    /// @param burst Raw ADC samples with both up and down chirps
    /// @return RangeCube with Doppler-corrected range data
    /// @throws std::invalid_argument if burst lacks up/down chirp pairs
    RangeCube process_doppler(const aurora::simulation::Burst& burst);

    /// Process burst with peak-based Doppler correction.
    ///
    /// For high-velocity targets where Doppler shift causes peaks to appear at
    /// different bins in up vs down chirps, use peak-based processing instead
    /// of bin-by-bin averaging.
    ///
    /// This method:
    /// 1. Detects peaks in each chirp's range FFT independently
    /// 2. Associates peak pairs between up and down chirps
    /// 3. Computes true range and velocity from peak positions
    ///
    /// @param burst Raw ADC samples with both up and down chirps
    /// @param wavelength_m Operating wavelength in meters
    /// @return Vector of PeakDopplerResult per detector/wavelength
    /// @throws std::invalid_argument if burst lacks up/down chirp pairs
    std::vector<PeakDopplerResult> process_doppler_peaks(
        const aurora::simulation::Burst& burst,
        double wavelength_m);

    /// Convert peak-based Doppler results to a RangeCube.
    ///
    /// Places each detected peak into the appropriate range bin for
    /// compatibility with downstream processing (MeasurementTensor).
    ///
    /// @param peak_results Peak-based Doppler results
    /// @param burst Original burst (for metadata)
    /// @return RangeCube with peak data in appropriate bins
    RangeCube peaks_to_range_cube(
        const std::vector<PeakDopplerResult>& peak_results,
        const aurora::simulation::Burst& burst);

    /// Set peak-based Doppler configuration.
    void set_peak_doppler_config(const PeakDopplerConfig& config);

    /// Get peak-based Doppler configuration.
    const PeakDopplerConfig& peak_doppler_config() const;

    /// Compute velocity resolution for given wavelength and chirp duration.
    ///
    /// Velocity resolution is determined by the ability to distinguish
    /// frequency shifts between up and down chirps:
    ///   Δv = λ / (4 × T_chirp)
    ///
    /// @param wavelength_m Wavelength in meters
    /// @param chirp_duration_s Chirp duration in seconds
    /// @return Velocity resolution in m/s
    static double compute_velocity_resolution(double wavelength_m,
                                              double chirp_duration_s);

    /// Convert range cube to measurement tensor with K-vector annotation.
    ///
    /// Annotates each measurement with its effective K-space coordinates
    /// from the forward model, including calibration corrections.
    ///
    /// @param range_cube Range-processed data
    /// @return MeasurementTensor ready for Detection/Reconstruction
    aurora::shared::MeasurementTensor to_measurement_tensor(const RangeCube& range_cube);

    /// Get processing latency statistics from last process_range call.
    /// @return LatencyStats with timing breakdown
    LatencyStats get_latency_stats() const { return latency_stats_; }

    /// Get computed range resolution in meters.
    /// @return Range resolution based on chirp bandwidth
    double range_resolution_m() const { return range_resolution_m_; }

    /// Get the DSP configuration.
    const DspConfig& config() const { return config_; }

private:
    DspConfig config_;
    const aurora::forward::ForwardModel& forward_model_;
    MeasurementBuilder measurement_builder_;

    // DSP components (pimpl to avoid header dependencies)
    std::unique_ptr<FftWrapper> range_fft_;
    std::unique_ptr<IQDemodulator> iq_demod_;

    // Peak-based Doppler processing
    PeakDopplerConfig peak_doppler_config_;
    mutable std::unique_ptr<PeakDopplerProcessor> peak_doppler_processor_;

    // Precomputed parameters
    double range_resolution_m_ = 0.0;
    double sample_rate_hz_ = 0.0;
    double adc_lsb_ = 0.0;  // ADC least significant bit in volts

    // Latency tracking
    mutable LatencyStats latency_stats_;

    /// Remove DC component from signal.
    /// @param signal Input signal
    /// @return DC-removed signal
    std::vector<double> remove_dc(const std::vector<double>& signal) const;

    /// Process a single chirp to range bins.
    ///
    /// Returns the full FFT spectrum (N bins) to support negative frequency
    /// detection for close-range Doppler processing.
    ///
    /// FFT output layout:
    ///   Bins [0, N/2):   positive frequencies (0 to +f_s/2)
    ///   Bins [N/2, N):   negative frequencies (-f_s/2 to 0)
    ///
    /// For standard range processing, use only bins [0, N/2).
    /// For Doppler processing where down-chirp beat frequency can go negative
    /// (when f_doppler > f_beat), check bins [N/2, N) for aliased peaks.
    ///
    /// @param adc_samples Raw ADC samples
    /// @param chirp_bandwidth_hz Chirp bandwidth in Hz
    /// @param chirp_duration_s Chirp duration in seconds
    /// @return Complex FFT bins (full N-bin spectrum)
    std::vector<std::complex<double>> process_chirp(
        const std::vector<int32_t>& adc_samples,
        double chirp_bandwidth_hz,
        double chirp_duration_s);

    /// Compute range resolution from chirp parameters.
    /// @param bandwidth_hz Chirp bandwidth in Hz
    /// @return Range resolution in meters
    static double compute_range_resolution(double bandwidth_hz);

    /// Compute range per FFT bin (range axis step size).
    /// This differs from range resolution when fft_size != samples_per_chirp.
    /// @param bandwidth_hz Chirp bandwidth in Hz
    /// @param chirp_duration_s Chirp duration in seconds
    /// @param sample_rate_hz ADC sample rate in Hz
    /// @param fft_size FFT size used for range processing
    /// @return Range per bin in meters
    static double compute_range_per_bin(double bandwidth_hz,
                                        double chirp_duration_s,
                                        double sample_rate_hz,
                                        uint32_t fft_size);

    /// Compute range axis offset due to IQ demodulation reference frequency.
    /// When IQ demodulation uses a non-zero reference frequency, it shifts the
    /// beat frequency spectrum. This function computes the compensating range
    /// offset needed for the range axis to correctly map bins to ranges.
    /// @param reference_freq_hz IQ demodulation reference frequency in Hz
    /// @param bandwidth_hz Chirp bandwidth in Hz
    /// @param chirp_duration_s Chirp duration in seconds
    /// @return Range offset in meters (negative when ref_freq > 0)
    static double compute_range_offset_from_reference_freq(
        double reference_freq_hz,
        double bandwidth_hz,
        double chirp_duration_s);

    // -------------------------------------------------------------------------
    // Doppler processing helpers
    // -------------------------------------------------------------------------

    /// Summed range data for a single chirp direction (averaged on output).
    struct DirectionSum {
        std::vector<std::complex<double>> bins;
        size_t count = 0;
    };

    /// Map type: wavelength -> detector -> direction -> summed range bins.
    using DirectionDataMap = std::map<double, std::map<uint32_t,
        std::map<aurora::acquisition::ChirpPlan::Direction, DirectionSum>>>;

    /// Build map of chirp index to direction from the burst plan.
    static std::map<uint32_t, aurora::acquisition::ChirpPlan::Direction>
    build_direction_map(const aurora::acquisition::SequencePlan& plan);

    /// Separate the burst's chirps by direction and sum their range bins.
    DirectionDataMap sum_chirps_by_direction(
        const aurora::simulation::Burst& burst,
        const aurora::acquisition::ChirpPlan& chirp_plan);

    /// Combine up/down chirps to produce Doppler-corrected range data.
    static void combine_up_down_chirps(
        const DirectionDataMap& data,
        RangeCube& cube);

    /// Fill the cube's range axis from chirp parameters and FFT size.
    void setup_range_axis(RangeCube& cube,
                          double chirp_bandwidth_hz,
                          double chirp_duration_s) const;
};

}  // namespace aurora::dsp
