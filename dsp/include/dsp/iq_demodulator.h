#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace aurora::dsp {

/// Time-domain IQ demodulator for extracting complex representation from
/// real-valued FMCW beat signals.
///
/// This class implements quadrature demodulation by:
/// 1. Multiplying the input signal by cos and sin references
/// 2. Low-pass filtering to remove sum frequency components
/// 3. Forming the complex I/Q output
///
/// Mathematical basis:
///   s(t) = A * cos(2*pi*f_beat*t + phi)
///   I(t) = LPF{ s(t) * cos(2*pi*f_ref*t) }
///   Q(t) = LPF{ s(t) * sin(2*pi*f_ref*t) }
///   z(t) = I(t) + j*Q(t)
///
/// IMPORTANT: For proper demodulation, the sum frequency (f_beat + f_ref)
/// must be above the LPF cutoff. When f_ref = f_beat (DC output case),
/// the sum = 2*f_beat, so you need lpf_cutoff_hz < f_beat.
class IQDemodulator {
public:
    struct Config {
        double sample_rate_hz;     ///< ADC sample rate (e.g., 125 MHz)
        double reference_freq_hz;  ///< Demodulation reference frequency
        double lpf_cutoff_hz;      ///< Low-pass filter cutoff frequency
        uint32_t lpf_order;        ///< Filter order (typically 4-8)
    };

    /// Construct demodulator with given configuration.
    /// @throws std::invalid_argument if config parameters are invalid
    explicit IQDemodulator(const Config& config);

    /// Demodulate real signal to complex I/Q using configured reference.
    /// @param real_signal Input real-valued signal samples
    /// @return Complex I/Q representation
    std::vector<std::complex<double>> demodulate(
        const std::vector<double>& real_signal) const;

    /// Demodulate with specified reference frequency (overrides config).
    /// Useful for variable beat frequency applications.
    /// @param real_signal Input real-valued signal samples
    /// @param reference_freq_hz Reference frequency for this demodulation
    /// @return Complex I/Q representation
    std::vector<std::complex<double>> demodulate(
        const std::vector<double>& real_signal,
        double reference_freq_hz) const;

    /// Get I and Q components separately (for debugging/analysis).
    /// @param real_signal Input real-valued signal samples
    /// @return Pair of (I components, Q components)
    std::pair<std::vector<double>, std::vector<double>> get_iq_components(
        const std::vector<double>& real_signal) const;

    /// Get the configured sample rate.
    double sample_rate_hz() const { return config_.sample_rate_hz; }

    /// Get the configured reference frequency.
    double reference_freq_hz() const { return config_.reference_freq_hz; }

    /// Get the configured LPF cutoff frequency.
    double lpf_cutoff_hz() const { return config_.lpf_cutoff_hz; }

    /// Get the configured LPF order.
    uint32_t lpf_order() const { return config_.lpf_order; }

private:
    Config config_;

    // Pre-computed Butterworth filter coefficients (second-order sections)
    // Each section: [b0, b1, b2, a1, a2] (a0 is normalized to 1)
    std::vector<std::array<double, 5>> sos_sections_;

    /// Design Butterworth low-pass filter and populate sos_sections_.
    void design_butterworth_lpf();

    /// Apply low-pass filter using cascaded second-order sections.
    /// @param signal Input signal
    /// @return Filtered signal
    std::vector<double> apply_lpf(const std::vector<double>& signal) const;

    /// Apply a single second-order section (biquad) filter.
    /// @param signal Input signal
    /// @param section Filter coefficients [b0, b1, b2, a1, a2]
    /// @return Filtered signal
    std::vector<double> apply_biquad(
        const std::vector<double>& signal,
        const std::array<double, 5>& section) const;
};

}  // namespace aurora::dsp
