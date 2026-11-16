#include "simulation/chirp_generator.h"

#include <gtest/gtest.h>

#include <cmath>

#include "sequence_plan.pb.h"

namespace aurora::simulation {
namespace {

// Create a chirp plan for testing
aurora::acquisition::ChirpPlan create_test_chirp_plan() {
    aurora::acquisition::ChirpPlan plan;
    plan.set_chirp_index(0);
    plan.set_chirp_start_s(0.0);
    plan.set_chirp_duration_s(12.5e-6);      // 12.5 us
    plan.set_chirp_start_hz(0.0);
    plan.set_chirp_bandwidth_hz(200e6);      // 200 MHz
    plan.set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);
    plan.add_wavelength_m(1550e-9);          // 1550 nm
    return plan;
}

class ChirpGeneratorTest : public ::testing::Test {};

// Test beat frequency calculation
TEST_F(ChirpGeneratorTest, BeatFrequency) {
    // For 100m range, 200 MHz bandwidth, 12.5 us duration:
    // f_beat = (2 * 200e6 * 100) / (3e8 * 12.5e-6) = 10.67 MHz
    double range_m = 100.0;
    double bandwidth_hz = 200e6;
    double duration_s = 12.5e-6;

    double f_beat = ChirpGenerator::compute_beat_frequency(
        range_m, bandwidth_hz, duration_s);

    double expected = (2.0 * bandwidth_hz * range_m) / (ChirpGenerator::kSpeedOfLight * duration_s);
    EXPECT_NEAR(f_beat, expected, 1.0);  // Within 1 Hz
    EXPECT_NEAR(f_beat, 10.67e6, 0.01e6);  // ~10.67 MHz
}

// Test range calculation (inverse of beat frequency)
TEST_F(ChirpGeneratorTest, RangeFromBeatFrequency) {
    double bandwidth_hz = 200e6;
    double duration_s = 12.5e-6;

    // Round-trip test
    for (double range : {10.0, 50.0, 100.0, 200.0, 300.0}) {
        double f_beat = ChirpGenerator::compute_beat_frequency(
            range, bandwidth_hz, duration_s);
        double recovered_range = ChirpGenerator::compute_range(
            f_beat, bandwidth_hz, duration_s);
        EXPECT_NEAR(recovered_range, range, 1e-6);
    }
}

// Test Doppler frequency calculation
TEST_F(ChirpGeneratorTest, DopplerFrequency) {
    // For 10 m/s velocity at 1550 nm:
    // f_d = 2 * 10 / 1550e-9 = 12.9 MHz
    double velocity_mps = 10.0;
    double wavelength_m = 1550e-9;

    double f_doppler = ChirpGenerator::compute_doppler_frequency(
        velocity_mps, wavelength_m);

    double expected = (2.0 * velocity_mps) / wavelength_m;
    EXPECT_NEAR(f_doppler, expected, 1.0);
    EXPECT_NEAR(f_doppler, 12.9e6, 0.1e6);  // ~12.9 MHz
}

// Test sample count calculation
TEST_F(ChirpGeneratorTest, NumSamples) {
    double duration_s = 12.5e-6;
    double sample_rate_hz = 125e6;

    uint32_t num_samples = ChirpGenerator::compute_num_samples(
        duration_s, sample_rate_hz);

    // 12.5 us at 125 MS/s = 1562.5 -> 1563 samples
    EXPECT_EQ(num_samples, 1563);
}

// Test chirp waveform generation
TEST_F(ChirpGeneratorTest, WaveformGeneration) {
    auto chirp_plan = create_test_chirp_plan();
    double sample_rate_hz = 125e6;

    auto waveform = ChirpGenerator::generate_waveform(chirp_plan, sample_rate_hz);

    // Check correct number of samples
    uint32_t expected_samples = ChirpGenerator::compute_num_samples(
        chirp_plan.chirp_duration_s(), sample_rate_hz);
    EXPECT_EQ(waveform.size(), expected_samples);

    // Check amplitude bounds
    for (double sample : waveform) {
        EXPECT_GE(sample, -1.0);
        EXPECT_LE(sample, 1.0);
    }
}

// Test beat frequency calculation round-trip
TEST_F(ChirpGeneratorTest, BeatFrequencyRoundTrip) {
    double original_range = 50.0;  // 50 meters
    double chirp_bandwidth = 1e9;  // 1 GHz
    double chirp_duration = 10e-6; // 10 microseconds

    double beat_freq = ChirpGenerator::compute_beat_frequency(
        original_range, chirp_bandwidth, chirp_duration);

    double recovered_range = ChirpGenerator::compute_range(
        beat_freq, chirp_bandwidth, chirp_duration);

    EXPECT_NEAR(recovered_range, original_range, 1e-6);  // 1 micron tolerance
}

// Test beat frequency calculation values
TEST_F(ChirpGeneratorTest, BeatFrequencyValue) {
    double range = 100.0;         // 100 meters
    double chirp_bandwidth = 1e9; // 1 GHz
    double chirp_duration = 10e-6;

    // f_beat = 2 * B * R / (c * T)
    // f_beat = 2 * 1e9 * 100 / (3e8 * 10e-6)
    // f_beat = 2e11 / 3e3 = 66.67 MHz (approximately)
    double expected_beat = (2.0 * chirp_bandwidth * range) /
                           (ChirpGenerator::kSpeedOfLight * chirp_duration);

    double actual_beat = ChirpGenerator::compute_beat_frequency(
        range, chirp_bandwidth, chirp_duration);

    EXPECT_NEAR(actual_beat, expected_beat, 1.0);  // 1 Hz tolerance
}

}  // namespace
}  // namespace aurora::simulation
