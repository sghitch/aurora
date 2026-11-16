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

#include "dsp/iq_demodulator.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numeric>

namespace aurora::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

// Test configuration matching typical FMCW LiDAR parameters
IQDemodulator::Config make_test_config() {
    return {
        .sample_rate_hz = 125e6,     // 125 MHz ADC
        .reference_freq_hz = 0.0,    // Zero reference (baseband)
        .lpf_cutoff_hz = 40e6,       // 40 MHz cutoff
        .lpf_order = 6               // 6th order Butterworth
    };
}

// Generate a cosine signal with given amplitude, frequency, and phase
std::vector<double> generate_cosine(double amplitude, double freq_hz,
                                    double phase_rad, double sample_rate_hz,
                                    size_t num_samples) {
    std::vector<double> signal(num_samples);
    const double dt = 1.0 / sample_rate_hz;
    for (size_t i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) * dt;
        signal[i] = amplitude * std::cos(kTwoPi * freq_hz * t + phase_rad);
    }
    return signal;
}

// Calculate RMS of a signal
double rms(const std::vector<double>& signal) {
    if (signal.empty()) return 0.0;
    double sum_sq = 0.0;
    for (double x : signal) {
        sum_sq += x * x;
    }
    return std::sqrt(sum_sq / static_cast<double>(signal.size()));
}

// Calculate average magnitude of complex signal
double avg_magnitude(const std::vector<std::complex<double>>& signal) {
    if (signal.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& z : signal) {
        sum += std::abs(z);
    }
    return sum / static_cast<double>(signal.size());
}

// Calculate average phase of complex signal (unwrapped)
double avg_phase(const std::vector<std::complex<double>>& signal,
                 size_t skip_transient = 0) {
    if (signal.size() <= skip_transient) return 0.0;
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = skip_transient; i < signal.size(); ++i) {
        sum += std::arg(signal[i]);
        ++count;
    }
    return sum / static_cast<double>(count);
}

class IQDemodulatorTest : public ::testing::Test {
protected:
    IQDemodulator::Config config_ = make_test_config();
};

TEST_F(IQDemodulatorTest, ConstructorValidatesConfig) {
    // Valid config should not throw
    EXPECT_NO_THROW(IQDemodulator{config_});

    // Invalid sample rate
    auto bad_config = config_;
    bad_config.sample_rate_hz = 0;
    EXPECT_THROW(IQDemodulator{bad_config}, std::invalid_argument);

    bad_config.sample_rate_hz = -1;
    EXPECT_THROW(IQDemodulator{bad_config}, std::invalid_argument);

    // Invalid reference freq (negative)
    bad_config = config_;
    bad_config.reference_freq_hz = -1;
    EXPECT_THROW(IQDemodulator{bad_config}, std::invalid_argument);

    // Invalid LPF cutoff
    bad_config = config_;
    bad_config.lpf_cutoff_hz = 0;
    EXPECT_THROW(IQDemodulator{bad_config}, std::invalid_argument);

    // LPF cutoff above Nyquist
    bad_config = config_;
    bad_config.lpf_cutoff_hz = config_.sample_rate_hz;
    EXPECT_THROW(IQDemodulator{bad_config}, std::invalid_argument);

    // Invalid filter order
    bad_config = config_;
    bad_config.lpf_order = 0;
    EXPECT_THROW(IQDemodulator{bad_config}, std::invalid_argument);

    bad_config.lpf_order = 21;
    EXPECT_THROW(IQDemodulator{bad_config}, std::invalid_argument);
}

TEST_F(IQDemodulatorTest, EmptyInputReturnsEmpty) {
    IQDemodulator demod(config_);

    auto result = demod.demodulate({});
    EXPECT_TRUE(result.empty());

    auto [i_comp, q_comp] = demod.get_iq_components({});
    EXPECT_TRUE(i_comp.empty());
    EXPECT_TRUE(q_comp.empty());
}

TEST_F(IQDemodulatorTest, PhaseExtractionAccuracy) {
    // Test that phase is preserved within 0.05 rad
    // When reference = beat frequency, output is DC with phase = -input_phase
    // CRITICAL: LPF cutoff must be < beat freq so sum (2*beat) is rejected
    const double amplitude = 1.0;
    const double freq_hz = 30e6;  // 30 MHz beat frequency
    const size_t num_samples = 40000;

    // Test several phase values
    std::vector<double> test_phases = {0.0, kPi / 4, kPi / 2, -kPi / 4};

    for (double input_phase : test_phases) {
        // Use reference = beat, cutoff < beat so sum (60 MHz) is rejected
        auto config = config_;
        config.reference_freq_hz = freq_hz;
        config.lpf_cutoff_hz = 20e6;  // 20 MHz cutoff, sum at 60 MHz is rejected
        IQDemodulator demod(config);

        auto signal = generate_cosine(amplitude, freq_hz, input_phase,
                                      config.sample_rate_hz, num_samples);

        auto result = demod.demodulate(signal);

        // Skip transient (filter settling time)
        const size_t skip = num_samples * 3 / 4;

        // With ref = beat freq, output is DC with phase = -input_phase
        double expected_phase = -input_phase;
        if (expected_phase > kPi) expected_phase -= kTwoPi;
        if (expected_phase < -kPi) expected_phase += kTwoPi;

        // Average phase in steady state
        double sum_phase = 0.0;
        for (size_t i = skip; i < num_samples; ++i) {
            sum_phase += std::arg(result[i]);
        }
        double measured_phase = sum_phase / static_cast<double>(num_samples - skip);

        // Normalize phase difference to [-pi, pi]
        double phase_error = measured_phase - expected_phase;
        while (phase_error > kPi) phase_error -= kTwoPi;
        while (phase_error < -kPi) phase_error += kTwoPi;

        EXPECT_LT(std::abs(phase_error), 0.05)
            << "Phase error too large for input_phase=" << input_phase
            << ", expected=" << expected_phase << ", measured=" << measured_phase;
    }
}

TEST_F(IQDemodulatorTest, MagnitudePreservation) {
    // Test that magnitude is preserved within 1%
    // Use reference = beat frequency for DC output where we can measure amplitude
    // CRITICAL: cutoff < beat so sum frequency is rejected
    const double freq_hz = 30e6;  // 30 MHz beat
    const double phase = 0.0;
    const size_t num_samples = 40000;

    // Test several amplitude values
    std::vector<double> test_amplitudes = {0.1, 0.5, 1.0, 2.0, 10.0};

    for (double input_amplitude : test_amplitudes) {
        auto config = config_;
        config.reference_freq_hz = freq_hz;
        config.lpf_cutoff_hz = 20e6;  // Sum at 60 MHz is rejected
        IQDemodulator demod(config);

        auto signal = generate_cosine(input_amplitude, freq_hz, phase,
                                      config.sample_rate_hz, num_samples);

        auto result = demod.demodulate(signal);

        // Skip transient - use larger skip for filter settling
        const size_t skip = num_samples * 3 / 4;

        // Calculate average magnitude in steady state
        double sum_mag = 0.0;
        for (size_t i = skip; i < num_samples; ++i) {
            sum_mag += std::abs(result[i]);
        }
        double avg_mag = sum_mag / static_cast<double>(num_samples - skip);

        // Allow 1% error
        double error_fraction =
            std::abs(avg_mag - input_amplitude) / input_amplitude;
        EXPECT_LT(error_fraction, 0.01)
            << "Magnitude error too large for input=" << input_amplitude
            << ", measured=" << avg_mag;
    }
}

TEST_F(IQDemodulatorTest, FrequencyRangeSupport) {
    // Test that demodulator works for various beat frequencies
    // Use reference = beat frequency, cutoff < beat/2 for each test
    const double amplitude = 1.0;
    const double phase = kPi / 3;
    const size_t num_samples = 40000;

    // Test frequencies where sum can be properly rejected
    // Need cutoff < beat, so sum = 2*beat > 2*cutoff
    std::vector<std::pair<double, double>> freq_cutoff_pairs = {
        {25e6, 15e6},  // 25 MHz beat, 15 MHz cutoff, sum=50 MHz
        {30e6, 20e6},  // 30 MHz beat, 20 MHz cutoff, sum=60 MHz
        {40e6, 25e6},  // 40 MHz beat, 25 MHz cutoff, sum=80 MHz
    };

    for (const auto& [freq_hz, cutoff_hz] : freq_cutoff_pairs) {
        auto config = config_;
        config.reference_freq_hz = freq_hz;
        config.lpf_cutoff_hz = cutoff_hz;
        IQDemodulator demod(config);

        auto signal = generate_cosine(amplitude, freq_hz, phase,
                                      config.sample_rate_hz, num_samples);

        auto result = demod.demodulate(signal);

        // Skip transient
        const size_t skip = num_samples * 3 / 4;

        // Check magnitude is approximately correct
        double sum_mag = 0.0;
        for (size_t i = skip; i < num_samples; ++i) {
            sum_mag += std::abs(result[i]);
        }
        double avg_mag = sum_mag / static_cast<double>(num_samples - skip);

        // Allow up to 5% error for this broad test
        double error_fraction = std::abs(avg_mag - amplitude) / amplitude;
        EXPECT_LT(error_fraction, 0.05)
            << "Frequency " << freq_hz / 1e6 << " MHz failed, avg_mag=" << avg_mag;
    }
}

TEST_F(IQDemodulatorTest, LPFRejectsSumFrequency) {
    // Test that the LPF provides rejection of the sum frequency component.
    // When we demodulate at the exact beat frequency, output should be DC.
    // Sum frequency = 2*beat should be attenuated by LPF.

    // Use beat = reference = 30 MHz, sum = 60 MHz
    // Cutoff at 20 MHz strongly rejects 60 MHz
    auto config = config_;
    config.reference_freq_hz = 30e6;  // 30 MHz reference
    config.lpf_cutoff_hz = 20e6;      // 20 MHz cutoff - sum at 60 MHz is rejected
    IQDemodulator demod(config);

    const double beat_freq = 30e6;  // Same as reference -> DC output
    const double amplitude = 1.0;
    const size_t num_samples = 40000;

    auto signal = generate_cosine(amplitude, beat_freq, 0.0,
                                  config.sample_rate_hz, num_samples);

    auto result = demod.demodulate(signal, config.reference_freq_hz);

    // Skip transient
    const size_t skip = num_samples * 3 / 4;

    // Calculate the magnitude variation - if sum freq is rejected,
    // magnitude should be nearly constant (DC)
    std::vector<double> magnitudes;
    for (size_t i = skip; i < num_samples; ++i) {
        magnitudes.push_back(std::abs(result[i]));
    }

    double mean_mag = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0) /
                      static_cast<double>(magnitudes.size());

    // Calculate variance
    double variance = 0.0;
    for (double m : magnitudes) {
        variance += (m - mean_mag) * (m - mean_mag);
    }
    variance /= static_cast<double>(magnitudes.size());
    double std_dev = std::sqrt(variance);

    // With good sum frequency rejection, std_dev should be small
    EXPECT_LT(std_dev / mean_mag, 0.02)
        << "Magnitude variation too high, suggesting poor sum freq rejection"
        << " (std_dev=" << std_dev << ", mean=" << mean_mag << ")";
}

TEST_F(IQDemodulatorTest, KnownSignalDemodulation) {
    // Demodulate a known cosine with reference = beat frequency.
    // Output should be DC with magnitude = amplitude and phase = -input_phase.
    // CRITICAL: cutoff < beat so sum frequency is rejected

    auto config = config_;
    config.lpf_cutoff_hz = 20e6;  // Cutoff below beat freq
    IQDemodulator demod(config);

    const double freq_hz = 30e6;  // 30 MHz beat, sum = 60 MHz >> cutoff
    const double amplitude = 2.0;
    const double phase = kPi / 6;  // 30 degrees
    const size_t num_samples = 50000;

    auto signal = generate_cosine(amplitude, freq_hz, phase,
                                  config.sample_rate_hz, num_samples);

    // Demodulate at beat frequency for DC output
    auto result = demod.demodulate(signal, freq_hz);

    // Skip transient for the filter to settle
    const size_t skip = num_samples * 3 / 4;

    // Output should be DC with magnitude = amplitude
    double avg_mag = 0.0;
    for (size_t i = skip; i < num_samples; ++i) {
        avg_mag += std::abs(result[i]);
    }
    avg_mag /= static_cast<double>(num_samples - skip);

    EXPECT_NEAR(avg_mag, amplitude, amplitude * 0.02)
        << "Magnitude mismatch in known signal test";

    // Check phase (should be -input_phase for DC output)
    double avg_phase_val = 0.0;
    for (size_t i = skip; i < num_samples; ++i) {
        avg_phase_val += std::arg(result[i]);
    }
    avg_phase_val /= static_cast<double>(num_samples - skip);

    double expected_phase = -phase;
    double phase_error = avg_phase_val - expected_phase;
    while (phase_error > kPi) phase_error -= kTwoPi;
    while (phase_error < -kPi) phase_error += kTwoPi;

    EXPECT_LT(std::abs(phase_error), 0.05)
        << "Phase mismatch: expected " << expected_phase << ", got " << avg_phase_val;
}

TEST_F(IQDemodulatorTest, GetIQComponentsConsistency) {
    // Verify get_iq_components returns components consistent with demodulate
    IQDemodulator demod(config_);

    const double amplitude = 1.5;
    const double freq_hz = 8e6;
    const double phase = kPi / 5;
    const size_t num_samples = 5000;

    auto signal = generate_cosine(amplitude, freq_hz, phase,
                                  config_.sample_rate_hz, num_samples);

    auto complex_result = demod.demodulate(signal);
    auto [i_comp, q_comp] = demod.get_iq_components(signal);

    ASSERT_EQ(complex_result.size(), i_comp.size());
    ASSERT_EQ(complex_result.size(), q_comp.size());

    // Verify I and Q match the complex result
    for (size_t i = 0; i < complex_result.size(); ++i) {
        EXPECT_DOUBLE_EQ(complex_result[i].real(), i_comp[i]);
        EXPECT_DOUBLE_EQ(complex_result[i].imag(), q_comp[i]);
    }
}

TEST_F(IQDemodulatorTest, DifferentFilterOrders) {
    // Test that different filter orders work
    // Use reference = beat frequency, cutoff < beat for DC output
    std::vector<uint32_t> orders = {2, 4, 6, 8};

    const double amplitude = 1.0;
    const double freq_hz = 30e6;  // 30 MHz beat
    const size_t num_samples = 50000;

    for (uint32_t order : orders) {
        auto config = config_;
        config.lpf_order = order;
        config.reference_freq_hz = freq_hz;
        config.lpf_cutoff_hz = 20e6;  // Cutoff below beat, sum at 60 MHz is rejected
        IQDemodulator demod(config);

        auto signal = generate_cosine(amplitude, freq_hz, 0.0,
                                      config.sample_rate_hz, num_samples);

        auto result = demod.demodulate(signal);
        ASSERT_EQ(result.size(), num_samples);

        // Skip transient - larger skip for higher orders (slower settling)
        const size_t skip = num_samples * 4 / 5;
        double sum_mag = 0.0;
        for (size_t i = skip; i < num_samples; ++i) {
            sum_mag += std::abs(result[i]);
        }
        double avg_mag = sum_mag / static_cast<double>(num_samples - skip);

        // Should be close to input amplitude (within 5%)
        EXPECT_GT(avg_mag, 0.95 * amplitude)
            << "Order " << order << " filter gave too low magnitude: " << avg_mag;
        EXPECT_LT(avg_mag, 1.05 * amplitude)
            << "Order " << order << " filter gave too high magnitude: " << avg_mag;
    }
}

TEST_F(IQDemodulatorTest, VariableReferenceFrequency) {
    // Test that passing a different reference frequency to demodulate() works
    // Create demodulator with one reference, but demodulate with another
    // CRITICAL: cutoff must be < beat so sum frequency is rejected
    auto config = config_;
    config.reference_freq_hz = 5e6;   // Default reference (ignored in this test)
    config.lpf_cutoff_hz = 20e6;      // Cutoff below beat frequency
    IQDemodulator demod(config);

    const double amplitude = 1.0;
    const double beat_freq = 30e6;    // Sum = 60 MHz >> cutoff
    const size_t num_samples = 40000;

    auto signal = generate_cosine(amplitude, beat_freq, 0.0,
                                  config.sample_rate_hz, num_samples);

    // Demodulate with reference at beat frequency - output should be DC
    auto result = demod.demodulate(signal, beat_freq);

    const size_t skip = num_samples * 3 / 4;

    // With reference = beat, the difference frequency is 0 (DC)
    // The output should be a constant complex number - check phase stability
    double phase_variation = 0.0;
    double prev_phase = std::arg(result[skip]);
    size_t count = 0;
    for (size_t i = skip + 1; i < num_samples; ++i) {
        double curr_phase = std::arg(result[i]);
        double diff = curr_phase - prev_phase;
        // Handle phase wrapping
        while (diff > kPi) diff -= kTwoPi;
        while (diff < -kPi) diff += kTwoPi;
        phase_variation += std::abs(diff);
        prev_phase = curr_phase;
        ++count;
    }
    phase_variation /= static_cast<double>(count);

    // Phase should be nearly constant (DC output)
    EXPECT_LT(phase_variation, 0.01)
        << "Phase variation too high for DC demodulation: " << phase_variation;
}

TEST_F(IQDemodulatorTest, ConfigAccessors) {
    IQDemodulator demod(config_);

    EXPECT_DOUBLE_EQ(demod.sample_rate_hz(), config_.sample_rate_hz);
    EXPECT_DOUBLE_EQ(demod.reference_freq_hz(), config_.reference_freq_hz);
    EXPECT_DOUBLE_EQ(demod.lpf_cutoff_hz(), config_.lpf_cutoff_hz);
    EXPECT_EQ(demod.lpf_order(), config_.lpf_order);
}

}  // namespace
}  // namespace aurora::dsp
