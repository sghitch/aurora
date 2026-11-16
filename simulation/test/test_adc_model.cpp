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

#include "simulation/adc_model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

namespace aurora::simulation {
namespace {

// Create a default ADC config for testing
AdcModelConfig create_default_config() {
    AdcModelConfig config;
    config.set_sample_rate_hz(125e6);
    config.set_bits(14);
    config.set_input_range_v(2.0);
    config.set_dc_offset_v(0.0);
    config.set_gain_error(0.0);
    config.set_aperture_jitter_s(0.0);
    return config;
}

// Create a config with typical impairments
AdcModelConfig create_impaired_config() {
    AdcModelConfig config = create_default_config();
    config.set_dc_offset_v(0.001);        // 1 mV offset
    config.set_gain_error(0.01);          // 1% gain error
    config.set_aperture_jitter_s(100e-15); // 100 fs jitter
    return config;
}

// Create a multi-channel config
AdcModelConfig create_multichannel_config() {
    AdcModelConfig config = create_default_config();

    // Per-channel gains (slightly different from 1.0)
    config.add_channel_gains(1.005);
    config.add_channel_gains(0.995);
    config.add_channel_gains(1.002);
    config.add_channel_gains(0.998);

    // Per-channel offsets
    config.add_channel_offsets_v(0.0005);
    config.add_channel_offsets_v(-0.0003);
    config.add_channel_offsets_v(0.0001);
    config.add_channel_offsets_v(-0.0002);

    // Per-channel delays
    config.add_channel_delays_s(0.0);
    config.add_channel_delays_s(1e-9);
    config.add_channel_delays_s(2e-9);
    config.add_channel_delays_s(3e-9);

    return config;
}

class AdcModelTest : public ::testing::Test {
  protected:
    std::mt19937_64 rng_{42};
};

// Test sample count calculation
TEST_F(AdcModelTest, SampleCount) {
    AdcModel adc(create_default_config());

    // 12.5 us chirp at 125 MS/s = 1562.5 -> 1563 samples
    double chirp_duration = 12.5e-6;
    uint32_t expected = static_cast<uint32_t>(std::ceil(125e6 * chirp_duration));
    EXPECT_EQ(adc.samples_per_duration(chirp_duration), expected);
    EXPECT_EQ(adc.samples_per_duration(chirp_duration), 1563);

    // 1 ms duration = 125000 samples
    EXPECT_EQ(adc.samples_per_duration(1e-3), 125000);

    // Zero duration = 0 samples
    EXPECT_EQ(adc.samples_per_duration(0.0), 0);
}

// Test LSB voltage calculation
TEST_F(AdcModelTest, LsbVoltage) {
    AdcModel adc(create_default_config());

    // 14-bit ADC with 2V range: LSB = 2V / 2^14 = 122 uV
    double expected_lsb = 2.0 / (1 << 14);
    EXPECT_NEAR(adc.lsb_voltage(), expected_lsb, 1e-9);
    EXPECT_NEAR(adc.lsb_voltage(), 122e-6, 1e-6);
}

// Test that all output samples are within valid range
TEST_F(AdcModelTest, QuantizationRange) {
    AdcModel adc(create_default_config());

    // Create a signal that exceeds the input range
    std::vector<double> signal;
    double signal_sample_rate = 1e9;  // 1 GS/s input signal
    for (int i = 0; i < 10000; ++i) {
        // Vary from -2V to +2V (exceeds +/- 1V range)
        signal.push_back(2.0 * std::sin(2.0 * M_PI * i / 1000.0));
    }

    auto samples = adc.sample(signal, signal_sample_rate, rng_);

    // All samples should be in 14-bit signed range [-8192, 8191]
    for (int16_t s : samples) {
        EXPECT_GE(s, -8192);
        EXPECT_LE(s, 8191);
    }
}

// Test DC offset application
TEST_F(AdcModelTest, DcOffset) {
    AdcModelConfig config = create_default_config();
    config.set_dc_offset_v(0.1);  // 100 mV offset

    AdcModel adc(config);

    // Create a DC signal at 0V
    std::vector<double> signal(10000, 0.0);
    double signal_sample_rate = 125e6;

    auto samples = adc.sample(signal, signal_sample_rate, rng_);

    // With 100 mV offset subtracted, effective input is -100 mV
    // Expected code: -0.1V / 1V * 8192 = -819.2 -> -819
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    EXPECT_NEAR(mean, -819, 2);
}

// Test gain error application
TEST_F(AdcModelTest, GainError) {
    AdcModelConfig config = create_default_config();
    config.set_gain_error(0.1);  // 10% gain error

    AdcModel adc(config);

    // Create a 500 mV DC signal
    std::vector<double> signal(10000, 0.5);
    double signal_sample_rate = 125e6;

    auto samples = adc.sample(signal, signal_sample_rate, rng_);

    // With 10% gain: 0.5V * 1.1 = 0.55V
    // Expected code: 0.55V / 1V * 8192 = 4505.6 -> 4506
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    EXPECT_NEAR(mean, 4506, 2);
}

// Test timing jitter statistics
TEST_F(AdcModelTest, JitterStatistics) {
    AdcModelConfig config = create_default_config();
    config.set_aperture_jitter_s(1e-9);  // 1 ns jitter (large for testing)

    AdcModel adc(config);

    // Create a fast-changing signal where jitter will show up
    std::vector<double> signal;
    double signal_sample_rate = 10e9;  // 10 GS/s for high time resolution
    double freq = 100e6;  // 100 MHz sine wave

    for (int i = 0; i < 100000; ++i) {
        double t = static_cast<double>(i) / signal_sample_rate;
        signal.push_back(0.5 * std::sin(2.0 * M_PI * freq * t));
    }

    // Sample multiple times with different RNG seeds
    std::vector<double> sample_variances;
    for (int trial = 0; trial < 10; ++trial) {
        std::mt19937_64 trial_rng(trial);
        auto samples = adc.sample(signal, signal_sample_rate, trial_rng);

        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        double mean = sum / samples.size();

        double sq_sum = 0.0;
        for (int16_t s : samples) {
            sq_sum += (s - mean) * (s - mean);
        }
        sample_variances.push_back(sq_sum / samples.size());
    }

    // Check that variance across trials is non-zero (jitter has effect)
    double mean_var = std::accumulate(sample_variances.begin(), sample_variances.end(), 0.0) /
                      sample_variances.size();
    EXPECT_GT(mean_var, 0);
}

// Test per-channel variations
TEST_F(AdcModelTest, ChannelVariations) {
    AdcModel adc(create_multichannel_config());

    // Create a 500 mV DC signal
    std::vector<double> signal(10000, 0.5);
    double signal_sample_rate = 125e6;

    // Sample each channel and compute means
    std::vector<double> channel_means;
    for (uint32_t ch = 0; ch < 4; ++ch) {
        std::mt19937_64 ch_rng(ch);
        auto samples = adc.sample_channel(signal, signal_sample_rate, ch, ch_rng);

        double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        channel_means.push_back(mean);
    }

    // Each channel should have a different mean due to per-channel variations
    for (size_t i = 0; i < channel_means.size(); ++i) {
        for (size_t j = i + 1; j < channel_means.size(); ++j) {
            EXPECT_NE(channel_means[i], channel_means[j])
                << "Channels " << i << " and " << j << " have identical means";
        }
    }
}

// Test empty signal handling
TEST_F(AdcModelTest, EmptySignal) {
    AdcModel adc(create_default_config());

    std::vector<double> empty_signal;
    auto samples = adc.sample(empty_signal, 125e6, rng_);

    EXPECT_TRUE(samples.empty());
}

// Test invalid configuration
TEST_F(AdcModelTest, InvalidConfig) {
    AdcModelConfig config;
    config.set_sample_rate_hz(0);  // Invalid
    config.set_bits(14);
    config.set_input_range_v(2.0);

    EXPECT_THROW(AdcModel adc(config), std::invalid_argument);
}

// Test invalid sample rate in sample()
TEST_F(AdcModelTest, InvalidSignalSampleRate) {
    AdcModel adc(create_default_config());
    std::vector<double> signal(100, 0.0);

    EXPECT_THROW(adc.sample(signal, 0.0, rng_), std::invalid_argument);
    EXPECT_THROW(adc.sample(signal, -1.0, rng_), std::invalid_argument);
}

// Test accessors
TEST_F(AdcModelTest, Accessors) {
    AdcModel adc(create_default_config());

    EXPECT_DOUBLE_EQ(adc.sample_rate_hz(), 125e6);
    EXPECT_EQ(adc.bits(), 14);
    EXPECT_DOUBLE_EQ(adc.input_range_v(), 2.0);
}

// Test signal resampling (input rate different from ADC rate)
TEST_F(AdcModelTest, SignalResampling) {
    AdcModel adc(create_default_config());

    // Create a 1 kHz sine wave at 10 MS/s (10 samples per cycle)
    std::vector<double> signal;
    double signal_sample_rate = 10e6;
    double freq = 1e6;
    double duration = 10e-6;  // 10 cycles

    size_t num_input_samples = static_cast<size_t>(duration * signal_sample_rate);
    for (size_t i = 0; i < num_input_samples; ++i) {
        double t = static_cast<double>(i) / signal_sample_rate;
        signal.push_back(0.5 * std::sin(2.0 * M_PI * freq * t));
    }

    auto samples = adc.sample(signal, signal_sample_rate, rng_);

    // ADC at 125 MS/s for 10 us = 1250 samples
    EXPECT_EQ(samples.size(), 1250);

    // Signal should still have sinusoidal character
    // Check that we have both positive and negative samples
    bool has_positive = false;
    bool has_negative = false;
    for (int16_t s : samples) {
        if (s > 100) has_positive = true;
        if (s < -100) has_negative = true;
    }
    EXPECT_TRUE(has_positive);
    EXPECT_TRUE(has_negative);
}

// Test full scale signal
TEST_F(AdcModelTest, FullScaleSignal) {
    AdcModel adc(create_default_config());

    // Create signals at +1V and -1V (full scale)
    std::vector<double> positive_signal(1000, 1.0);
    std::vector<double> negative_signal(1000, -1.0);

    auto pos_samples = adc.sample(positive_signal, 125e6, rng_);
    auto neg_samples = adc.sample(negative_signal, 125e6, rng_);

    // +1V should give max positive code (8191 for 14-bit)
    // -1V should give max negative code (-8192 for 14-bit)
    EXPECT_EQ(pos_samples[0], 8191);
    EXPECT_EQ(neg_samples[0], -8192);
}

// Test determinism with same seed
TEST_F(AdcModelTest, Determinism) {
    AdcModelConfig config = create_impaired_config();
    AdcModel adc(config);

    std::vector<double> signal;
    double signal_sample_rate = 1e9;
    for (int i = 0; i < 10000; ++i) {
        signal.push_back(0.5 * std::sin(2.0 * M_PI * i / 100.0));
    }

    std::mt19937_64 rng1(12345);
    std::mt19937_64 rng2(12345);

    auto samples1 = adc.sample(signal, signal_sample_rate, rng1);
    auto samples2 = adc.sample(signal, signal_sample_rate, rng2);

    EXPECT_EQ(samples1.size(), samples2.size());
    for (size_t i = 0; i < samples1.size(); ++i) {
        EXPECT_EQ(samples1[i], samples2[i]) << "Mismatch at index " << i;
    }
}

}  // namespace
}  // namespace aurora::simulation
