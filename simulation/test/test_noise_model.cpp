#include "simulation/noise_model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

namespace aurora::simulation {
namespace {

// Create a default ADC config
AdcModelConfig create_default_adc_config() {
    AdcModelConfig config;
    config.set_bits(14);              // 14-bit ADC
    config.set_input_range_v(2.0);    // +/- 1V = 2V total
    return config;
}

// Create a config with all noise sources enabled
NoiseModelConfig create_full_config() {
    NoiseModelConfig config;
    config.set_enable_shot_noise(true);
    config.set_enable_thermal_noise(true);
    config.set_enable_rin_noise(true);
    config.set_enable_quantization_noise(true);
    config.set_enable_dark_current(true);

    config.set_detector_responsivity_a_per_w(0.9);     // Typical InGaAs
    config.set_detector_dark_current_a(1e-9);          // 1 nA
    config.set_transimpedance_ohm(50.0);               // 50 ohm
    config.set_temperature_k(300.0);                   // Room temperature

    config.set_rin_db_per_hz(-150.0);                  // Typical DFB laser

    return config;
}

// Create a config with only shot noise
NoiseModelConfig create_shot_only_config() {
    NoiseModelConfig config;
    config.set_enable_shot_noise(true);
    config.set_enable_thermal_noise(false);
    config.set_enable_rin_noise(false);
    config.set_enable_quantization_noise(false);
    config.set_enable_dark_current(false);

    config.set_detector_responsivity_a_per_w(0.9);

    return config;
}

// Create a config with only thermal noise
NoiseModelConfig create_thermal_only_config() {
    NoiseModelConfig config;
    config.set_enable_shot_noise(false);
    config.set_enable_thermal_noise(true);
    config.set_enable_rin_noise(false);
    config.set_enable_quantization_noise(false);
    config.set_enable_dark_current(false);

    config.set_transimpedance_ohm(50.0);
    config.set_temperature_k(300.0);

    return config;
}

// Create a config with no noise
NoiseModelConfig create_no_noise_config() {
    NoiseModelConfig config;
    config.set_enable_shot_noise(false);
    config.set_enable_thermal_noise(false);
    config.set_enable_rin_noise(false);
    config.set_enable_quantization_noise(false);
    config.set_enable_dark_current(false);

    return config;
}

class NoiseModelTest : public ::testing::Test {
  protected:
    static constexpr double kTolerance = 1e-12;
    static constexpr double kLooseTolerance = 0.05;  // 5% for statistical tests
};

// Test: Shot noise variance matches theory
TEST_F(NoiseModelTest, ShotNoiseVariance) {
    auto config = create_shot_only_config();
    NoiseModel noise(config, create_default_adc_config());

    double optical_power = 1e-6;   // 1 uW
    double bandwidth = 125e6;      // 125 MHz

    double shot_std = noise.compute_shot_noise_std(optical_power, bandwidth);

    // Expected: sigma^2 = 2 * q * R * P * B
    // sigma^2 = 2 * 1.602e-19 * 0.9 * 1e-6 * 125e6 = 3.60e-17
    // sigma = 1.90e-9 A
    double i_dc = 0.9 * optical_power;
    double expected_var = 2.0 * NoiseModel::kElectronCharge * i_dc * bandwidth;
    double expected_std = std::sqrt(expected_var);

    EXPECT_NEAR(shot_std, expected_std, expected_std * kLooseTolerance);
}

// Test: Thermal noise variance matches theory
TEST_F(NoiseModelTest, ThermalNoiseVariance) {
    auto config = create_thermal_only_config();
    NoiseModel noise(config, create_default_adc_config());

    double bandwidth = 125e6;  // 125 MHz

    double thermal_std = noise.compute_thermal_noise_std(bandwidth);

    // Expected: sigma^2 = 4 * k * T * B / R
    // sigma^2 = 4 * 1.38e-23 * 300 * 125e6 / 50 = 4.14e-14
    // sigma = 2.03e-7 A
    double expected_var = 4.0 * NoiseModel::kBoltzmann * 300.0 * bandwidth / 50.0;
    double expected_std = std::sqrt(expected_var);

    EXPECT_NEAR(thermal_std, expected_std, expected_std * kLooseTolerance);
}

// Test: RIN scales with power squared
TEST_F(NoiseModelTest, RinScalesWithPowerSquared) {
    NoiseModelConfig config;
    config.set_enable_rin_noise(true);
    config.set_rin_db_per_hz(-150.0);
    NoiseModel noise(config, create_default_adc_config());

    double bandwidth = 125e6;

    double rin_std_1mw = noise.compute_rin_noise_std(1e-3, bandwidth);
    double rin_std_2mw = noise.compute_rin_noise_std(2e-3, bandwidth);

    // RIN variance scales with P^2, so std scales with P
    // At 2x power, std should be 2x
    EXPECT_NEAR(rin_std_2mw / rin_std_1mw, 2.0, 0.01);
}

// Test: Quantization noise value
TEST_F(NoiseModelTest, QuantizationNoise) {
    NoiseModelConfig config;
    config.set_enable_quantization_noise(true);
    NoiseModel noise(config, create_default_adc_config());

    double quant_std = noise.compute_quantization_noise_std();

    // V_LSB = 2V / 2^14 = 122 uV
    // sigma^2 = V_LSB^2 / 12
    double v_lsb = 2.0 / (1 << 14);
    double expected_var = v_lsb * v_lsb / 12.0;
    double expected_std = std::sqrt(expected_var);

    EXPECT_NEAR(quant_std, expected_std, expected_std * 0.01);
}

// Test: Dark current noise
TEST_F(NoiseModelTest, DarkCurrentNoise) {
    NoiseModelConfig config;
    config.set_enable_dark_current(true);
    config.set_detector_dark_current_a(1e-9);  // 1 nA
    NoiseModel noise(config, create_default_adc_config());

    double bandwidth = 125e6;

    double dark_std = noise.compute_dark_current_noise_std(bandwidth);

    // sigma^2 = 2 * q * I_dark * B
    double expected_var = 2.0 * NoiseModel::kElectronCharge * 1e-9 * bandwidth;
    double expected_std = std::sqrt(expected_var);

    EXPECT_NEAR(dark_std, expected_std, expected_std * kLooseTolerance);
}

// Test: Total variance is sum of components
TEST_F(NoiseModelTest, TotalVarianceSumOfComponents) {
    auto config = create_full_config();
    NoiseModel noise(config, create_default_adc_config());

    double optical_power = 1e-6;
    double bandwidth = 125e6;

    double shot_std = noise.compute_shot_noise_std(optical_power, bandwidth);
    double thermal_std = noise.compute_thermal_noise_std(bandwidth);
    double rin_std = noise.compute_rin_noise_std(optical_power, bandwidth);
    double quant_std = noise.compute_quantization_noise_std();
    double dark_std = noise.compute_dark_current_noise_std(bandwidth);

    double expected_variance = shot_std * shot_std +
                               thermal_std * thermal_std +
                               rin_std * rin_std +
                               quant_std * quant_std +
                               dark_std * dark_std;

    double total_variance = noise.compute_total_noise_variance(optical_power,
                                                                bandwidth);

    EXPECT_NEAR(total_variance, expected_variance, expected_variance * 0.001);
}

// Test: Deterministic noise with same seed
TEST_F(NoiseModelTest, NoiseDeterminism) {
    auto config = create_full_config();
    NoiseModel noise(config, create_default_adc_config());

    std::vector<double> signal(1000, 1.0);
    double optical_power = 1e-6;
    double bandwidth = 125e6;

    // First run
    std::mt19937_64 rng1(42);
    auto noisy1 = noise.add_noise(signal, optical_power, bandwidth, rng1);

    // Second run with same seed
    std::mt19937_64 rng2(42);
    auto noisy2 = noise.add_noise(signal, optical_power, bandwidth, rng2);

    // Should be identical
    ASSERT_EQ(noisy1.size(), noisy2.size());
    for (size_t i = 0; i < noisy1.size(); ++i) {
        EXPECT_EQ(noisy1[i], noisy2[i]);
    }
}

// Test: Different seed gives different noise
TEST_F(NoiseModelTest, DifferentSeedDifferentNoise) {
    auto config = create_full_config();
    NoiseModel noise(config, create_default_adc_config());

    std::vector<double> signal(1000, 1.0);
    double optical_power = 1e-6;
    double bandwidth = 125e6;

    std::mt19937_64 rng1(42);
    auto noisy1 = noise.add_noise(signal, optical_power, bandwidth, rng1);

    std::mt19937_64 rng2(123);
    auto noisy2 = noise.add_noise(signal, optical_power, bandwidth, rng2);

    // Should be different
    bool all_same = true;
    for (size_t i = 0; i < noisy1.size(); ++i) {
        if (noisy1[i] != noisy2[i]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

// Test: Enable flags work (disabled sources = 0)
TEST_F(NoiseModelTest, DisabledSourcesZero) {
    auto config = create_no_noise_config();
    NoiseModel noise(config, create_default_adc_config());

    double optical_power = 1e-6;
    double bandwidth = 125e6;

    EXPECT_EQ(noise.compute_shot_noise_std(optical_power, bandwidth), 0.0);
    EXPECT_EQ(noise.compute_thermal_noise_std(bandwidth), 0.0);
    EXPECT_EQ(noise.compute_rin_noise_std(optical_power, bandwidth), 0.0);
    EXPECT_EQ(noise.compute_quantization_noise_std(), 0.0);
    EXPECT_EQ(noise.compute_dark_current_noise_std(bandwidth), 0.0);
    EXPECT_EQ(noise.compute_total_noise_variance(optical_power, bandwidth), 0.0);
}

// Test: No noise added when all sources disabled
TEST_F(NoiseModelTest, NoNoiseWhenDisabled) {
    auto config = create_no_noise_config();
    NoiseModel noise(config, create_default_adc_config());

    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0, 5.0};
    double optical_power = 1e-6;
    double bandwidth = 125e6;

    std::mt19937_64 rng(42);
    auto noisy = noise.add_noise(signal, optical_power, bandwidth, rng);

    ASSERT_EQ(noisy.size(), signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        EXPECT_EQ(noisy[i], signal[i]);
    }
}

// Test: SNR calculation
TEST_F(NoiseModelTest, SnrCalculation) {
    auto config = create_shot_only_config();
    NoiseModel noise(config, create_default_adc_config());

    double optical_power = 1e-6;
    double bandwidth = 125e6;
    double signal_power = 1e-12;  // 1 pW

    double snr_db = noise.compute_snr_db(signal_power, optical_power, bandwidth);

    // Manual calculation
    double noise_var = noise.compute_total_noise_variance(optical_power,
                                                           bandwidth);
    double expected_snr_db = 10.0 * std::log10(signal_power / noise_var);

    EXPECT_NEAR(snr_db, expected_snr_db, 0.1);  // 0.1 dB tolerance
}

// Test: SNR infinite when no noise
TEST_F(NoiseModelTest, SnrInfiniteWhenNoNoise) {
    auto config = create_no_noise_config();
    NoiseModel noise(config, create_default_adc_config());

    double snr_db = noise.compute_snr_db(1e-12, 1e-6, 125e6);

    EXPECT_TRUE(std::isinf(snr_db));
    EXPECT_GT(snr_db, 0.0);  // Positive infinity
}

// Test: Added noise has approximately correct variance (statistical test)
TEST_F(NoiseModelTest, AddedNoiseVariance) {
    auto config = create_shot_only_config();
    NoiseModel noise(config, create_default_adc_config());

    // Use many samples for statistical accuracy
    std::vector<double> signal(100000, 0.0);
    double optical_power = 1e-3;   // 1 mW for larger noise
    double bandwidth = 125e6;

    std::mt19937_64 rng(42);
    auto noisy = noise.add_noise(signal, optical_power, bandwidth, rng);

    // Compute sample variance
    double sum = 0.0;
    double sum_sq = 0.0;
    for (double val : noisy) {
        sum += val;
        sum_sq += val * val;
    }
    double n = static_cast<double>(noisy.size());
    double mean = sum / n;
    double sample_variance = (sum_sq - n * mean * mean) / (n - 1);

    double expected_variance = noise.compute_total_noise_variance(optical_power,
                                                                   bandwidth);

    // Allow 10% tolerance for statistical test
    EXPECT_NEAR(sample_variance, expected_variance, expected_variance * 0.1);
}

// Test: Zero optical power gives only non-power-dependent noise
TEST_F(NoiseModelTest, ZeroOpticalPower) {
    auto config = create_full_config();
    NoiseModel noise(config, create_default_adc_config());

    double bandwidth = 125e6;

    // With zero optical power, only thermal, quantization, and dark current
    // should contribute
    double shot_std = noise.compute_shot_noise_std(0.0, bandwidth);
    double rin_std = noise.compute_rin_noise_std(0.0, bandwidth);

    EXPECT_EQ(shot_std, 0.0);
    EXPECT_EQ(rin_std, 0.0);

    // But thermal and quant should still be non-zero
    double thermal_std = noise.compute_thermal_noise_std(bandwidth);
    double quant_std = noise.compute_quantization_noise_std();

    EXPECT_GT(thermal_std, 0.0);
    EXPECT_GT(quant_std, 0.0);
}

// Test: Config accessor
TEST_F(NoiseModelTest, ConfigAccessor) {
    auto config = create_full_config();
    NoiseModel noise(config, create_default_adc_config());

    EXPECT_TRUE(noise.config().enable_shot_noise());
    EXPECT_TRUE(noise.config().enable_thermal_noise());
    EXPECT_NEAR(noise.config().temperature_k(), 300.0, kTolerance);
}

// Test: Physical constants
TEST_F(NoiseModelTest, PhysicalConstants) {
    // Verify physical constants are correct
    EXPECT_NEAR(NoiseModel::kBoltzmann, 1.380649e-23, 1e-29);
    EXPECT_NEAR(NoiseModel::kElectronCharge, 1.602176634e-19, 1e-28);
}

// Test: Zero transimpedance handled gracefully
TEST_F(NoiseModelTest, ZeroTransimpedance) {
    NoiseModelConfig config;
    config.set_enable_thermal_noise(true);
    config.set_transimpedance_ohm(0.0);  // Invalid but should not crash
    config.set_temperature_k(300.0);

    NoiseModel noise(config, create_default_adc_config());

    // Should return 0 rather than inf/nan
    double thermal_std = noise.compute_thermal_noise_std(125e6);
    EXPECT_EQ(thermal_std, 0.0);
}

// Test: Empty signal
TEST_F(NoiseModelTest, EmptySignal) {
    auto config = create_full_config();
    NoiseModel noise(config, create_default_adc_config());

    std::vector<double> signal;
    std::mt19937_64 rng(42);

    auto noisy = noise.add_noise(signal, 1e-6, 125e6, rng);

    EXPECT_TRUE(noisy.empty());
}

}  // namespace
}  // namespace aurora::simulation
