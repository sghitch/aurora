#include "dsp/peak_doppler_processor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "sequence_plan.pb.h"

namespace aurora::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

// Physical constants matching the detection example
constexpr double kSpeedOfLight = 299792458.0;
constexpr double kSampleRate = 125e6;       // 125 MHz
constexpr double kChirpBandwidth = 200e6;   // 200 MHz
constexpr double kChirpDuration = 12.5e-6;  // 12.5 us
constexpr double kWavelength1550nm = 1550e-9;

class PeakDopplerProcessorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Configuration - all required fields must be set
        config_.set_peak_threshold_db(10.0);
        config_.set_min_peak_separation(3);
        config_.set_max_velocity_mps(30.0);
        config_.set_association_threshold(0.3);
        config_.set_allow_unpaired_peaks(false);

        // Chirp plan - from sequence plan proto
        chirp_plan_.set_chirp_bandwidth_hz(kChirpBandwidth);
        chirp_plan_.set_chirp_duration_s(kChirpDuration);

        // Separate parameters
        sample_rate_hz_ = kSampleRate;
        wavelength_m_ = kWavelength1550nm;
    }

    // Generate a simulated range FFT with a peak at specified bin
    // fft_size is the full FFT size (N bins)
    std::vector<std::complex<double>> generate_peak_at_bin(size_t fft_size, size_t target_bin,
                                                           double magnitude = 100.0,
                                                           double noise_level = 0.1) {
        std::vector<std::complex<double>> fft_data(fft_size, std::complex<double>(0.0, 0.0));

        // Add peak
        double phase = 0.0;
        fft_data[target_bin] = std::polar(magnitude, phase);

        // Add some noise
        for (size_t i = 0; i < fft_size; ++i) {
            if (i != target_bin) {
                fft_data[i] += std::complex<double>(noise_level * (rand() / double(RAND_MAX)),
                                                     noise_level * (rand() / double(RAND_MAX)));
            }
        }

        return fft_data;
    }

    // Convert range to FFT bin
    // fft_size is the full FFT size (N bins)
    size_t range_to_bin(double range_m, size_t fft_size) {
        // f_beat = 2 * R * B / (c * T)
        // bin = f_beat * fft_size / sample_rate
        double f_beat = 2.0 * range_m * chirp_plan_.chirp_bandwidth_hz() /
                        (kSpeedOfLight * chirp_plan_.chirp_duration_s());
        return static_cast<size_t>(f_beat * fft_size / sample_rate_hz_);
    }

    // Convert velocity to bin offset (difference between up and down chirp bins)
    // fft_size is the full FFT size (N bins)
    double velocity_to_bin_offset(double velocity_mps, size_t fft_size) {
        // f_doppler = 2 * v / lambda
        // bin_offset = f_doppler * fft_size / sample_rate
        double f_doppler = 2.0 * velocity_mps / wavelength_m_;
        return f_doppler * fft_size / sample_rate_hz_;
    }

    PeakDopplerConfig config_;
    aurora::acquisition::ChirpPlan chirp_plan_;
    double sample_rate_hz_;
    double wavelength_m_;
};

// Test: Constructor validates parameters
TEST_F(PeakDopplerProcessorTest, ConstructorCreatesProcessor) {
    EXPECT_NO_THROW(PeakDopplerProcessor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_));
}

// Test: Peak detection finds single peak in noisy signal
TEST_F(PeakDopplerProcessorTest, DetectPeaksSingle) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    constexpr size_t kFftSize = 1024;
    constexpr size_t kTargetBin = 100;
    auto fft_data = generate_peak_at_bin(kFftSize, kTargetBin, 100.0, 0.5);

    auto peaks = processor.detect_peaks(fft_data);

    EXPECT_GE(peaks.size(), 1u) << "Should detect at least one peak";
    if (!peaks.empty()) {
        EXPECT_EQ(peaks[0].bin_index(), kTargetBin)
            << "Peak should be at expected bin";
    }
}

// Test: Peak detection finds multiple peaks
TEST_F(PeakDopplerProcessorTest, DetectPeaksMultiple) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    constexpr size_t kFftSize = 1024;
    std::vector<std::complex<double>> fft_data(kFftSize, std::complex<double>(0.0, 0.0));

    // Add three peaks at different bins
    fft_data[100] = std::complex<double>(80.0, 0.0);
    fft_data[300] = std::complex<double>(100.0, 0.0);
    fft_data[500] = std::complex<double>(60.0, 0.0);

    // Add low-level noise
    for (size_t i = 0; i < kFftSize; ++i) {
        if (i != 100 && i != 300 && i != 500) {
            fft_data[i] = std::complex<double>(0.5, 0.5);
        }
    }

    auto peaks = processor.detect_peaks(fft_data);

    EXPECT_EQ(peaks.size(), 3u) << "Should detect three peaks";
}

// Test: Stationary target (v=0) - peaks at same bin in up and down chirps
TEST_F(PeakDopplerProcessorTest, AssociateStationaryTarget) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Peaks at same bin = zero velocity
    // kFftSize is the full FFT size (N bins)
    // Frequency = bin * sample_rate / fft_size
    constexpr size_t kFftSize = 2048;  // Full FFT size
    ChirpPeak up_peak;
    up_peak.set_bin_index(100);
    up_peak.set_frequency_hz(100.0 * kSampleRate / kFftSize);
    up_peak.set_magnitude(100.0);
    up_peak.set_phase_rad(0.0);

    ChirpPeak down_peak;
    down_peak.set_bin_index(100);
    down_peak.set_frequency_hz(100.0 * kSampleRate / kFftSize);
    down_peak.set_magnitude(100.0);
    down_peak.set_phase_rad(0.0);

    auto pairs = processor.associate_peaks({up_peak}, {down_peak});

    ASSERT_EQ(pairs.size(), 1u) << "Should create one pair";
    EXPECT_EQ(pairs[0].up_peak().bin_index(), 100u);
    EXPECT_EQ(pairs[0].down_peak().bin_index(), 100u);
}

// Test: Moving target - peaks at different bins
TEST_F(PeakDopplerProcessorTest, AssociateMovingTarget) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Simulate up-chirp at bin 150, down-chirp at bin 100
    // This represents positive velocity
    // kFftSize is the full FFT size (N bins)
    constexpr size_t kFftSize = 2048;  // Full FFT size
    ChirpPeak up_peak;
    up_peak.set_bin_index(150);
    up_peak.set_frequency_hz(150.0 * kSampleRate / kFftSize);
    up_peak.set_magnitude(100.0);
    up_peak.set_phase_rad(0.0);

    ChirpPeak down_peak;
    down_peak.set_bin_index(100);
    down_peak.set_frequency_hz(100.0 * kSampleRate / kFftSize);
    down_peak.set_magnitude(90.0);  // Slightly different magnitude
    down_peak.set_phase_rad(0.0);

    auto pairs = processor.associate_peaks({up_peak}, {down_peak});

    ASSERT_EQ(pairs.size(), 1u) << "Should create one pair for moving target";
    EXPECT_EQ(pairs[0].up_peak().bin_index(), 150u);
    EXPECT_EQ(pairs[0].down_peak().bin_index(), 100u);
}

// Test: Range/velocity computation for stationary target
TEST_F(PeakDopplerProcessorTest, ComputeRangeVelocityStationary) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Create peak pair with same frequency (v=0)
    constexpr double kFrequency = 10e6;  // 10 MHz beat frequency

    PeakPair pair;
    pair.mutable_up_peak()->set_frequency_hz(kFrequency);
    pair.mutable_up_peak()->set_magnitude(100.0);
    pair.mutable_down_peak()->set_frequency_hz(kFrequency);
    pair.mutable_down_peak()->set_magnitude(100.0);

    auto result = processor.compute_range_velocity(pair);

    // Expected range: R = f_beat * c * T / (2 * B)
    double expected_range = kFrequency * kSpeedOfLight * kChirpDuration /
                            (2.0 * kChirpBandwidth);

    EXPECT_NEAR(result.velocity_mps(), 0.0, 0.1)
        << "Velocity should be ~0 for stationary target";
    EXPECT_NEAR(result.range_m(), expected_range, 0.5)
        << "Range should match expected value";
}

// Test: Range/velocity computation for moving target
TEST_F(PeakDopplerProcessorTest, ComputeRangeVelocityMoving) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Target at 50m with 3 m/s velocity
    constexpr double kTargetRange = 50.0;
    constexpr double kTargetVelocity = 3.0;

    // Compute expected frequencies
    // f_beat = 2 * R * B / (c * T) = 2 * 50 * 200e6 / (3e8 * 12.5e-6)
    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    // f_doppler = 2 * v / lambda = 2 * 3 / 1550e-9
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;

    double f_up = f_beat + f_doppler;
    double f_down = f_beat - f_doppler;

    PeakPair pair;
    pair.mutable_up_peak()->set_frequency_hz(f_up);
    pair.mutable_up_peak()->set_magnitude(100.0);
    pair.mutable_down_peak()->set_frequency_hz(f_down);
    pair.mutable_down_peak()->set_magnitude(100.0);

    auto result = processor.compute_range_velocity(pair);

    EXPECT_NEAR(result.range_m(), kTargetRange, 1.0)
        << "Range should be approximately " << kTargetRange << " m";
    EXPECT_NEAR(result.velocity_mps(), kTargetVelocity, 0.5)
        << "Velocity should be approximately " << kTargetVelocity << " m/s";
}

// Test: Full process() pipeline for stationary target
TEST_F(PeakDopplerProcessorTest, ProcessStationaryTarget) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    constexpr size_t kFftSize = 1024;
    constexpr double kTargetRange = 50.0;

    // Compute expected bin for stationary target
    size_t expected_bin = range_to_bin(kTargetRange, kFftSize);

    auto up_fft = generate_peak_at_bin(kFftSize, expected_bin, 100.0, 0.5);
    auto down_fft = generate_peak_at_bin(kFftSize, expected_bin, 100.0, 0.5);

    auto results = processor.process(up_fft, down_fft);

    ASSERT_GE(results.size(), 1u) << "Should find at least one target";
    if (!results.empty()) {
        EXPECT_NEAR(results[0].velocity_mps(), 0.0, 0.5)
            << "Velocity should be near zero for stationary target";
    }
}

// Test: Full process() pipeline for moving target
TEST_F(PeakDopplerProcessorTest, ProcessMovingTarget) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    constexpr size_t kFftSize = 1024;
    constexpr double kTargetRange = 50.0;
    constexpr double kTargetVelocity = 1.0;  // 1 m/s (lower velocity to keep bins positive)

    // Compute bins for up and down chirps
    size_t base_bin = range_to_bin(kTargetRange, kFftSize);
    double bin_offset = velocity_to_bin_offset(kTargetVelocity, kFftSize);

    // Make sure bins stay in valid range
    ASSERT_GT(base_bin, static_cast<size_t>(bin_offset))
        << "Base bin must be larger than offset for valid down_bin";

    size_t up_bin = base_bin + static_cast<size_t>(bin_offset);
    size_t down_bin = base_bin - static_cast<size_t>(bin_offset);

    // Ensure bins are within valid range and separated by min_peak_separation
    ASSERT_GT(up_bin, 3u);
    ASSERT_LT(up_bin, kFftSize - 3u);
    ASSERT_GT(down_bin, 3u);
    ASSERT_LT(down_bin, kFftSize - 3u);

    auto up_fft = generate_peak_at_bin(kFftSize, up_bin, 100.0, 0.5);
    auto down_fft = generate_peak_at_bin(kFftSize, down_bin, 100.0, 0.5);

    auto results = processor.process(up_fft, down_fft);

    ASSERT_GE(results.size(), 1u) << "Should find at least one target";
    if (!results.empty()) {
        // Range should be recovered correctly
        EXPECT_NEAR(results[0].range_m(), kTargetRange, 5.0)
            << "Range should be approximately correct";
        // Velocity direction should be correct (positive)
        EXPECT_GT(results[0].velocity_mps(), 0.0)
            << "Velocity should be positive (moving away)";
    }
}

// Test: Rejects impossible velocity pairings
TEST_F(PeakDopplerProcessorTest, RejectsHighVelocityPairings) {
    // Set low max velocity
    config_.set_max_velocity_mps(5.0);
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Create peaks that would imply very high velocity (100+ m/s)
    // kFftSize is the full FFT size (N bins)
    constexpr size_t kFftSize = 2048;  // Full FFT size
    ChirpPeak up_peak;
    up_peak.set_bin_index(500);
    up_peak.set_frequency_hz(500.0 * kSampleRate / kFftSize);
    up_peak.set_magnitude(100.0);

    ChirpPeak down_peak;
    down_peak.set_bin_index(100);  // Very different from up peak
    down_peak.set_frequency_hz(100.0 * kSampleRate / kFftSize);
    down_peak.set_magnitude(100.0);

    auto pairs = processor.associate_peaks({up_peak}, {down_peak});

    // Should reject this pairing due to implausible velocity
    EXPECT_EQ(pairs.size(), 0u)
        << "Should reject pairing that implies velocity > max_velocity_mps";
}

// Test: Handles empty input gracefully
TEST_F(PeakDopplerProcessorTest, HandlesEmptyInput) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    std::vector<std::complex<double>> empty_fft;

    auto peaks = processor.detect_peaks(empty_fft);
    EXPECT_EQ(peaks.size(), 0u);

    auto results = processor.process(empty_fft, empty_fft);
    EXPECT_EQ(results.size(), 0u);
}

// Test: Frequency resolution calculation
TEST_F(PeakDopplerProcessorTest, FrequencyResolution) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // kFftSize is the full FFT size (N bins)
    // Frequency resolution = sample_rate / fft_size
    constexpr size_t kFftSize = 2048;  // Full FFT size
    double expected_resolution = kSampleRate / kFftSize;
    double actual_resolution = processor.frequency_resolution(kFftSize);

    EXPECT_DOUBLE_EQ(actual_resolution, expected_resolution);
}

// Test: Max unambiguous velocity calculation
TEST_F(PeakDopplerProcessorTest, MaxUnambiguousVelocity) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    double max_vel = processor.max_unambiguous_velocity();

    // Should be positive
    // Note: The max_unambiguous_velocity() gives a theoretical upper bound
    // that can be quite high depending on chirp parameters. In practice,
    // velocity is limited by the max_velocity_mps config parameter.
    EXPECT_GT(max_vel, 0.0);
}

// Test: Configuration accessors
TEST_F(PeakDopplerProcessorTest, ConfigurationAccessors) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    EXPECT_DOUBLE_EQ(processor.config().peak_threshold_db(),
                     config_.peak_threshold_db());
    EXPECT_EQ(processor.config().min_peak_separation(),
              config_.min_peak_separation());
    EXPECT_DOUBLE_EQ(processor.config().max_velocity_mps(),
                     config_.max_velocity_mps());
}

// Test: Configuration update
TEST_F(PeakDopplerProcessorTest, ConfigurationUpdate) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    PeakDopplerConfig new_config;
    new_config.set_peak_threshold_db(20.0);
    new_config.set_max_velocity_mps(50.0);

    processor.set_config(new_config);

    EXPECT_DOUBLE_EQ(processor.config().peak_threshold_db(), 20.0);
    EXPECT_DOUBLE_EQ(processor.config().max_velocity_mps(), 50.0);
}

// =============================================================================
// Aliasing Detection and Correction Tests
// =============================================================================

// Test: Close-range high-velocity target with aliased down-chirp (30m @ 4 m/s)
// This is the primary aliasing failure case.
// Without correction, the 18m range error occurs because f_down aliases to positive.
TEST_F(PeakDopplerProcessorTest, AliasingCorrectionCloseRangeHighVelocity) {
    // Enable aliasing correction with required config values
    config_.set_enable_aliasing_correction(true);
    config_.set_expected_max_range_m(100.0);
    config_.set_aliasing_detection_ratio(0.3);
    config_.set_aliasing_range_ratio_threshold(1.15);
    config_.set_max_velocity_mps(10.0);

    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Target at 30m with 4 m/s velocity (moving away)
    constexpr double kTargetRange = 30.0;
    constexpr double kTargetVelocity = 4.0;

    // Compute expected frequencies
    // f_beat = 2 * R * B / (c * T) = 2 * 30 * 200e6 / (3e8 * 12.5e-6) ≈ 3.2 MHz
    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    // f_doppler = 2 * v / lambda = 2 * 4 / 1550e-9 ≈ 5.16 MHz
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;

    double f_up = f_beat + f_doppler;
    // f_down_true = f_beat - f_doppler (negative in this case!)
    double f_down_true = f_beat - f_doppler;

    // With real ADC sampling, negative frequency aliases to positive
    double f_down_aliased = std::abs(f_down_true);

    // Verify this is indeed an aliasing scenario (f_down_true < 0)
    EXPECT_LT(f_down_true, 0.0) << "This test requires f_down to be negative";

    // Create peak pair with aliased down frequency
    PeakPair pair;
    pair.mutable_up_peak()->set_frequency_hz(f_up);
    pair.mutable_up_peak()->set_magnitude(100.0);
    pair.mutable_down_peak()->set_frequency_hz(f_down_aliased);
    pair.mutable_down_peak()->set_magnitude(100.0);

    auto result = processor.compute_range_velocity(pair);

    // With aliasing correction enabled, range should be correct
    EXPECT_NEAR(result.range_m(), kTargetRange, 1.0)
        << "Range should be approximately " << kTargetRange << " m with aliasing correction";
    EXPECT_NEAR(result.velocity_mps(), kTargetVelocity, 0.5)
        << "Velocity should be approximately " << kTargetVelocity << " m/s";
    EXPECT_TRUE(result.aliasing_corrected())
        << "Aliasing correction flag should be set";
}

// Test: Far-range target should NOT trigger aliasing correction (50m @ 4 m/s)
// Regression test - ensure we don't false-positive on valid signals.
TEST_F(PeakDopplerProcessorTest, AliasingCorrectionFarRangeNoFalsePositive) {
    // Enable aliasing correction with required config values
    config_.set_enable_aliasing_correction(true);
    config_.set_expected_max_range_m(100.0);
    config_.set_aliasing_detection_ratio(0.3);
    config_.set_aliasing_range_ratio_threshold(1.15);
    config_.set_max_velocity_mps(10.0);

    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Target at 50m with 4 m/s velocity
    constexpr double kTargetRange = 50.0;
    constexpr double kTargetVelocity = 4.0;

    // Compute expected frequencies
    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;

    double f_up = f_beat + f_doppler;
    double f_down = f_beat - f_doppler;

    // Verify this is NOT an aliasing scenario (f_down > 0)
    EXPECT_GT(f_down, 0.0) << "This test requires f_down to be positive";

    // Create peak pair with correct (non-aliased) frequencies
    PeakPair pair;
    pair.mutable_up_peak()->set_frequency_hz(f_up);
    pair.mutable_up_peak()->set_magnitude(100.0);
    pair.mutable_down_peak()->set_frequency_hz(f_down);
    pair.mutable_down_peak()->set_magnitude(100.0);

    auto result = processor.compute_range_velocity(pair);

    // Should get correct result without triggering aliasing correction
    EXPECT_NEAR(result.range_m(), kTargetRange, 1.0)
        << "Range should be approximately " << kTargetRange << " m";
    EXPECT_NEAR(result.velocity_mps(), kTargetVelocity, 0.5)
        << "Velocity should be approximately " << kTargetVelocity << " m/s";
    EXPECT_FALSE(result.aliasing_corrected())
        << "Aliasing correction should NOT be triggered for valid signal";
}

// Test: Approaching target (negative velocity) - no aliasing case
// For approaching targets with moderate velocity, both f_up and f_down stay positive.
// This tests that aliasing correction doesn't interfere with valid approaching targets.
TEST_F(PeakDopplerProcessorTest, AliasingCorrectionApproachingTarget) {
    // Enable aliasing correction with required config values
    config_.set_enable_aliasing_correction(true);
    config_.set_expected_max_range_m(100.0);
    config_.set_aliasing_detection_ratio(0.3);
    config_.set_aliasing_range_ratio_threshold(1.15);
    config_.set_max_velocity_mps(10.0);

    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Target at 50m with -1 m/s velocity (approaching)
    // Using larger range and smaller velocity to ensure both frequencies stay positive
    // f_beat must be > |f_doppler| to keep f_up positive
    constexpr double kTargetRange = 50.0;
    constexpr double kTargetVelocity = -1.0;  // Negative = approaching

    // Compute expected frequencies
    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    // f_doppler sign follows velocity convention
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;  // Negative

    double f_up = f_beat + f_doppler;    // Lower frequency for approaching
    double f_down = f_beat - f_doppler;  // Higher frequency for approaching

    // Both should be positive for this configuration
    EXPECT_GT(f_up, 0.0) << "f_up should be positive (f_beat > |f_doppler|)";
    EXPECT_GT(f_down, 0.0) << "f_down should be positive";
    EXPECT_GT(f_down, f_up) << "For approaching targets, f_down > f_up";

    // Create peak pair
    PeakPair pair;
    pair.mutable_up_peak()->set_frequency_hz(f_up);
    pair.mutable_up_peak()->set_magnitude(100.0);
    pair.mutable_down_peak()->set_frequency_hz(f_down);
    pair.mutable_down_peak()->set_magnitude(100.0);

    auto result = processor.compute_range_velocity(pair);

    // Should work correctly without triggering aliasing correction
    // (aliasing correction only applies when f_down << f_up, not when f_down > f_up)
    EXPECT_NEAR(result.range_m(), kTargetRange, 1.0)
        << "Range should be approximately " << kTargetRange << " m";
    EXPECT_NEAR(result.velocity_mps(), kTargetVelocity, 0.5)
        << "Velocity should be approximately " << kTargetVelocity << " m/s (negative = approaching)";
    EXPECT_FALSE(result.aliasing_corrected())
        << "Aliasing correction should NOT be triggered for approaching target with f_down > f_up";
}

// Test: Aliasing correction disabled - verify old behavior preserved
TEST_F(PeakDopplerProcessorTest, AliasingCorrectionDisabledPreservesOldBehavior) {
    // Aliasing correction disabled (default)
    config_.set_enable_aliasing_correction(false);

    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Same scenario as the 30m @ 4 m/s test
    constexpr double kTargetRange = 30.0;
    constexpr double kTargetVelocity = 4.0;

    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;

    double f_up = f_beat + f_doppler;
    double f_down_true = f_beat - f_doppler;
    double f_down_aliased = std::abs(f_down_true);

    // Create peak pair with aliased down frequency
    PeakPair pair;
    pair.mutable_up_peak()->set_frequency_hz(f_up);
    pair.mutable_up_peak()->set_magnitude(100.0);
    pair.mutable_down_peak()->set_frequency_hz(f_down_aliased);
    pair.mutable_down_peak()->set_magnitude(100.0);

    auto result = processor.compute_range_velocity(pair);

    // Without correction, the computed range will be WRONG (much higher than actual)
    // This is expected behavior with correction disabled
    EXPECT_GT(result.range_m(), kTargetRange + 10.0)
        << "Without aliasing correction, range should be incorrectly high";
    EXPECT_FALSE(result.aliasing_corrected())
        << "Aliasing correction flag should be false";
}

// Test: Aliasing correction with ratio-based heuristic (no max range specified)
TEST_F(PeakDopplerProcessorTest, AliasingCorrectionRatioBasedHeuristic) {
    // Enable aliasing correction without explicit max range - use ratio-based heuristic
    config_.set_enable_aliasing_correction(true);
    // Don't set expected_max_range_m - rely on ratio-based detection
    config_.set_aliasing_detection_ratio(0.3);
    config_.set_aliasing_range_ratio_threshold(1.15);
    config_.set_max_velocity_mps(10.0);

    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Same 30m @ 4 m/s scenario
    constexpr double kTargetRange = 30.0;
    constexpr double kTargetVelocity = 4.0;

    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;

    double f_up = f_beat + f_doppler;
    double f_down_true = f_beat - f_doppler;
    double f_down_aliased = std::abs(f_down_true);

    // Create peak pair with aliased down frequency
    PeakPair pair;
    pair.mutable_up_peak()->set_frequency_hz(f_up);
    pair.mutable_up_peak()->set_magnitude(100.0);
    pair.mutable_down_peak()->set_frequency_hz(f_down_aliased);
    pair.mutable_down_peak()->set_magnitude(100.0);

    auto result = processor.compute_range_velocity(pair);

    // Ratio-based heuristic should also detect and correct the aliasing
    EXPECT_NEAR(result.range_m(), kTargetRange, 1.0)
        << "Ratio-based heuristic should correct aliased frequency";
    EXPECT_TRUE(result.aliasing_corrected());
}

// Test: Edge case - very low frequency ratio that should trigger correction
TEST_F(PeakDopplerProcessorTest, AliasingCorrectionEdgeCaseLowRatio) {
    config_.set_enable_aliasing_correction(true);
    config_.set_expected_max_range_m(100.0);
    config_.set_aliasing_detection_ratio(0.5);  // Higher threshold
    config_.set_aliasing_range_ratio_threshold(1.15);
    config_.set_max_velocity_mps(15.0);

    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Very close range, high velocity - severe aliasing
    constexpr double kTargetRange = 15.0;
    constexpr double kTargetVelocity = 5.0;

    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;

    double f_up = f_beat + f_doppler;
    double f_down_true = f_beat - f_doppler;
    double f_down_aliased = std::abs(f_down_true);

    EXPECT_LT(f_down_true, 0.0) << "Should be aliased scenario";

    PeakPair pair;
    pair.mutable_up_peak()->set_frequency_hz(f_up);
    pair.mutable_up_peak()->set_magnitude(100.0);
    pair.mutable_down_peak()->set_frequency_hz(f_down_aliased);
    pair.mutable_down_peak()->set_magnitude(100.0);

    auto result = processor.compute_range_velocity(pair);

    EXPECT_NEAR(result.range_m(), kTargetRange, 1.0)
        << "Should correctly recover range for severely aliased signal";
    EXPECT_TRUE(result.aliasing_corrected());
}

// =============================================================================
// Full-Spectrum Doppler Processing Tests
// =============================================================================

// Test: Detect peak in negative frequency region
// For close-range targets with velocity, f_down = f_beat - f_doppler can be negative.
// The peak appears in bins [N/2, N) which represent negative frequencies.
TEST_F(PeakDopplerProcessorTest, DetectPeakInNegativeFrequencyRegion) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Full FFT size (N bins)
    constexpr size_t kFftSize = 2048;

    // Place a peak in the negative frequency region (bins >= N/2)
    // Bin N/2 + 100 corresponds to a negative frequency
    constexpr size_t kNegativeBin = kFftSize / 2 + 100;
    auto fft_data = generate_peak_at_bin(kFftSize, kNegativeBin, 100.0, 0.5);

    auto peaks = processor.detect_peaks(fft_data);

    ASSERT_GE(peaks.size(), 1u) << "Should detect peak in negative frequency region";
    if (!peaks.empty()) {
        // Find the peak we placed
        bool found_neg_peak = false;
        for (const auto& peak : peaks) {
            if (peak.bin_index() == kNegativeBin) {
                found_neg_peak = true;
                // Verify frequency is negative
                EXPECT_LT(peak.frequency_hz(), 0.0)
                    << "Frequency should be negative for bins >= N/2";
                break;
            }
        }
        EXPECT_TRUE(found_neg_peak) << "Should find the peak at bin " << kNegativeBin;
    }
}

// Test: Close-range target with velocity causing negative down-chirp frequency
// Primary full-spectrum scenario: 10m range with 2 m/s velocity
TEST_F(PeakDopplerProcessorTest, CloseRangeMovingTargetNegativeDownChirp) {
    // Lower the threshold for this test since we're using clean signals
    config_.set_min_peak_magnitude(10.0);
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    // Target at 10m range with 2 m/s velocity
    // f_beat = 2 * R * B / (c * T) = 2 * 10 * 200e6 / (3e8 * 12.5e-6) ≈ 1.07 MHz
    // f_doppler = 2 * v / λ = 2 * 2 / 1550e-9 ≈ 2.58 MHz
    // f_up = f_beat + f_doppler ≈ 3.65 MHz (positive)
    // f_down = f_beat - f_doppler ≈ -1.51 MHz (NEGATIVE!)
    constexpr double kTargetRange = 10.0;
    constexpr double kTargetVelocity = 2.0;

    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;

    double f_up = f_beat + f_doppler;
    double f_down = f_beat - f_doppler;  // This is negative

    // Verify the scenario is as expected
    ASSERT_GT(f_up, 0.0) << "f_up should be positive";
    ASSERT_LT(f_down, 0.0) << "f_down should be negative for this test case";

    // Full FFT size (N bins)
    constexpr size_t kFftSize = 2048;

    // Calculate bin positions
    // For positive frequency: bin = f * N / f_s
    size_t up_bin = static_cast<size_t>(f_up * kFftSize / kSampleRate);

    // For negative frequency: bin = N + (f * N / f_s) since f is negative
    // This maps to bins [N/2, N)
    size_t down_bin = static_cast<size_t>(kFftSize + f_down * kFftSize / kSampleRate);

    // Verify bins are in expected regions
    ASSERT_LT(up_bin, kFftSize / 2) << "up_bin should be in positive frequency region";
    ASSERT_GE(down_bin, kFftSize / 2) << "down_bin should be in negative frequency region";
    ASSERT_LT(down_bin, kFftSize) << "down_bin should be within FFT size";

    // Generate FFT data with peaks at calculated positions
    auto up_fft = generate_peak_at_bin(kFftSize, up_bin, 100.0, 0.5);
    auto down_fft = generate_peak_at_bin(kFftSize, down_bin, 100.0, 0.5);

    auto results = processor.process(up_fft, down_fft);

    ASSERT_GE(results.size(), 1u) << "Should find at least one target";
    if (!results.empty()) {
        // Range should be recovered correctly
        EXPECT_NEAR(results[0].range_m(), kTargetRange, 2.0)
            << "Range should be approximately " << kTargetRange << " m";
        // Velocity should be recovered correctly
        EXPECT_NEAR(results[0].velocity_mps(), kTargetVelocity, 0.5)
            << "Velocity should be approximately " << kTargetVelocity << " m/s";
    }
}

// Test: Verify bin_to_signed_frequency() correctly handles both regions
TEST_F(PeakDopplerProcessorTest, BinToSignedFrequencyCorrectness) {
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    constexpr size_t kFftSize = 2048;

    // Test positive frequency bins (0 to N/2-1)
    // Bin 100 should give positive frequency = 100 * sample_rate / N
    constexpr size_t kPosBin = 100;
    auto pos_fft = generate_peak_at_bin(kFftSize, kPosBin, 100.0, 0.1);
    auto pos_peaks = processor.detect_peaks(pos_fft);

    ASSERT_GE(pos_peaks.size(), 1u);
    bool found_pos = false;
    for (const auto& peak : pos_peaks) {
        if (peak.bin_index() == kPosBin) {
            found_pos = true;
            double expected_freq = kPosBin * kSampleRate / kFftSize;
            EXPECT_NEAR(peak.frequency_hz(), expected_freq, 1.0)
                << "Positive bin should give positive frequency";
            EXPECT_GT(peak.frequency_hz(), 0.0);
            break;
        }
    }
    EXPECT_TRUE(found_pos);

    // Test negative frequency bins (N/2 to N-1)
    // Bin N/2+100 should give negative frequency
    constexpr size_t kNegBin = kFftSize / 2 + 100;
    auto neg_fft = generate_peak_at_bin(kFftSize, kNegBin, 100.0, 0.1);
    auto neg_peaks = processor.detect_peaks(neg_fft);

    ASSERT_GE(neg_peaks.size(), 1u);
    bool found_neg = false;
    for (const auto& peak : neg_peaks) {
        if (peak.bin_index() == kNegBin) {
            found_neg = true;
            // Expected: (bin - N) * sample_rate / N
            double expected_freq = (static_cast<double>(kNegBin) - kFftSize) * kSampleRate / kFftSize;
            EXPECT_NEAR(peak.frequency_hz(), expected_freq, 1.0)
                << "Negative bin should give negative frequency";
            EXPECT_LT(peak.frequency_hz(), 0.0);
            break;
        }
    }
    EXPECT_TRUE(found_neg);
}

// Test: Conjugate images of a real-valued spectrum do not produce mirror targets.
// A real ADC signal has every beat tone at +f mirrored at -f. Pairing the two
// mirror peaks with each other would give the same target at negative range.
TEST_F(PeakDopplerProcessorTest, ConjugateImagePairsAreDiscarded) {
    config_.set_min_peak_magnitude(10.0);
    PeakDopplerProcessor processor(config_, chirp_plan_, sample_rate_hz_, wavelength_m_);

    constexpr size_t kFftSize = 2048;
    constexpr double kTargetRange = 30.0;
    constexpr double kTargetVelocity = 1.0;

    double f_beat = 2.0 * kTargetRange * kChirpBandwidth /
                    (kSpeedOfLight * kChirpDuration);
    double f_doppler = 2.0 * kTargetVelocity / kWavelength1550nm;
    size_t up_bin = static_cast<size_t>((f_beat + f_doppler) * kFftSize / kSampleRate);
    size_t down_bin = static_cast<size_t>((f_beat - f_doppler) * kFftSize / kSampleRate);
    ASSERT_GT(down_bin, 0u);

    // Real-valued spectrum: peak at +f and its conjugate image at -f (bin N - b)
    auto up_fft = generate_peak_at_bin(kFftSize, up_bin, 100.0, 0.5);
    auto up_image = generate_peak_at_bin(kFftSize, kFftSize - up_bin, 100.0, 0.0);
    auto down_fft = generate_peak_at_bin(kFftSize, down_bin, 100.0, 0.5);
    auto down_image = generate_peak_at_bin(kFftSize, kFftSize - down_bin, 100.0, 0.0);
    for (size_t i = 0; i < kFftSize; ++i) {
        up_fft[i] += up_image[i];
        down_fft[i] += down_image[i];
    }

    auto results = processor.process(up_fft, down_fft);

    ASSERT_EQ(results.size(), 1u) << "Mirror pair must not appear as a second target";
    EXPECT_GT(results[0].range_m(), 0.0);
    EXPECT_NEAR(results[0].range_m(), kTargetRange, 1.0);
    EXPECT_NEAR(results[0].velocity_mps(), kTargetVelocity, 0.1);
}

}  // namespace
}  // namespace aurora::dsp
