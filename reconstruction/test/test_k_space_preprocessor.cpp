#include "reconstruction/k_space_preprocessor.h"

#include <cmath>

#include <gtest/gtest.h>

#include "k_space_slice.pb.h"
#include "reconstruction_config.pb.h"

namespace aurora::reconstruction {

class KSpacePreprocessorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a simple K-space slice with samples at various k magnitudes
        slice_.set_range_bin(0);
        slice_.set_wavelength_m(905e-9);

        // Add samples at different radii
        add_sample(0.0, 0.0);         // DC
        add_sample(1000.0, 0.0);      // Low k
        add_sample(2000.0, 0.0);      // Mid k
        add_sample(3000.0, 0.0);      // High k
        add_sample(4000.0, 0.0);      // Very high k
        add_sample(0.0, 1000.0);      // ky direction
        add_sample(0.0, 3000.0);      // High ky
        add_sample(2000.0, 2000.0);   // Diagonal
    }

    void add_sample(double kx, double ky) {
        auto* sample = slice_.add_samples();
        sample->set_kx(kx);
        sample->set_ky(ky);
        sample->mutable_value()->set_real(1.0);
        sample->mutable_value()->set_imag(0.0);
        sample->set_weight(1.0);
    }

    aurora::shared::KSpaceSlice slice_;
    double k_max_x_ = 5000.0;
    double k_max_y_ = 5000.0;
};

// Test default config (Hann apodization, no density taper)
TEST_F(KSpacePreprocessorTest, DefaultConstructorNoTaper) {
    DirectAdjointConfig config;
    KSpacePreprocessor preprocessor(config);

    auto result = preprocessor.apply(slice_, k_max_x_, k_max_y_);

    // All samples should be present
    EXPECT_EQ(result.samples_size(), slice_.samples_size());

    // DC sample should have non-zero weight (from Hann apodization)
    EXPECT_GT(std::abs(result.samples(0).value().real()), 0.0);
}

// Test isotropic taper with linear rolloff
TEST_F(KSpacePreprocessorTest, IsotropicLinearTaper) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);  // Disable apodization
    auto* taper = config.mutable_density_taper();
    auto* iso = taper->mutable_isotropic();
    iso->set_k_threshold(1500.0);   // Below this: full weight
    iso->set_k_max_taper(4500.0);   // Above this: retain_fraction
    iso->set_retain_fraction(0.0);  // Zero at high k
    iso->set_taper_function(TAPER_LINEAR);

    KSpacePreprocessor preprocessor(config);

    // Check density taper weights directly
    // DC (k=0): below threshold -> weight = 1.0
    double dc_weight = preprocessor.compute_density_taper(0.0, 0.0);
    EXPECT_DOUBLE_EQ(dc_weight, 1.0);

    // k=1000: below threshold -> weight = 1.0
    double low_k_weight = preprocessor.compute_density_taper(1000.0, 0.0);
    EXPECT_DOUBLE_EQ(low_k_weight, 1.0);

    // k=3000: in transition zone (t = (3000-1500)/(4500-1500) = 0.5)
    // Linear: weight = 1 - 0.5 = 0.5
    double mid_k_weight = preprocessor.compute_density_taper(3000.0, 0.0);
    EXPECT_NEAR(mid_k_weight, 0.5, 1e-10);

    // k=4500: at k_max_taper -> weight = retain_fraction = 0.0
    double high_k_weight = preprocessor.compute_density_taper(4500.0, 0.0);
    EXPECT_DOUBLE_EQ(high_k_weight, 0.0);
}

// Test isotropic taper with non-zero retain_fraction
TEST_F(KSpacePreprocessorTest, IsotropicRetainFraction) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();
    auto* iso = taper->mutable_isotropic();
    iso->set_k_threshold(1000.0);
    iso->set_k_max_taper(4000.0);
    iso->set_retain_fraction(0.2);  // 20% minimum weight
    iso->set_taper_function(TAPER_LINEAR);

    KSpacePreprocessor preprocessor(config);

    // At k_max_taper: weight should be retain_fraction
    double high_k_weight = preprocessor.compute_density_taper(4000.0, 0.0);
    EXPECT_DOUBLE_EQ(high_k_weight, 0.2);

    // Above k_max_taper: weight should still be retain_fraction
    double very_high_k_weight = preprocessor.compute_density_taper(5000.0, 0.0);
    EXPECT_DOUBLE_EQ(very_high_k_weight, 0.2);

    // Mid k: t = (2500-1000)/(4000-1000) = 0.5
    // Linear with retain=0.2: weight = 0.2 + 0.8*(1-0.5) = 0.6
    double mid_k_weight = preprocessor.compute_density_taper(2500.0, 0.0);
    EXPECT_NEAR(mid_k_weight, 0.6, 1e-10);
}

// Test cosine taper function
TEST_F(KSpacePreprocessorTest, CosineTaper) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();
    auto* iso = taper->mutable_isotropic();
    iso->set_k_threshold(1000.0);
    iso->set_k_max_taper(3000.0);
    iso->set_retain_fraction(0.0);
    iso->set_taper_function(TAPER_COSINE);

    KSpacePreprocessor preprocessor(config);

    // At midpoint t=0.5: cosine = 0.5*(1+cos(π*0.5)) = 0.5*(1+0) = 0.5
    double mid_k_weight = preprocessor.compute_density_taper(2000.0, 0.0);
    EXPECT_NEAR(mid_k_weight, 0.5, 1e-10);

    // At t=0 (threshold): cosine = 0.5*(1+cos(0)) = 1.0
    double threshold_weight = preprocessor.compute_density_taper(1000.0, 0.0);
    EXPECT_DOUBLE_EQ(threshold_weight, 1.0);

    // At t=1 (k_max_taper): cosine = 0.5*(1+cos(π)) = 0.0
    double max_weight = preprocessor.compute_density_taper(3000.0, 0.0);
    EXPECT_NEAR(max_weight, 0.0, 1e-10);
}

// Test quadratic taper function
TEST_F(KSpacePreprocessorTest, QuadraticTaper) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();
    auto* iso = taper->mutable_isotropic();
    iso->set_k_threshold(1000.0);
    iso->set_k_max_taper(3000.0);
    iso->set_retain_fraction(0.0);
    iso->set_taper_function(TAPER_QUADRATIC);
    iso->set_taper_exponent(2.0);

    KSpacePreprocessor preprocessor(config);

    // At t=0.5: quadratic = (1-0.5)^2 = 0.25
    double mid_k_weight = preprocessor.compute_density_taper(2000.0, 0.0);
    EXPECT_NEAR(mid_k_weight, 0.25, 1e-10);
}

// Test exponential taper function
TEST_F(KSpacePreprocessorTest, ExponentialTaper) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();
    auto* iso = taper->mutable_isotropic();
    iso->set_k_threshold(1000.0);
    iso->set_k_max_taper(3000.0);
    iso->set_retain_fraction(0.0);
    iso->set_taper_function(TAPER_EXPONENTIAL);
    iso->set_taper_exponent(2.0);

    KSpacePreprocessor preprocessor(config);

    // At t=0.5: exp(-2*0.5) = exp(-1) ≈ 0.368
    double mid_k_weight = preprocessor.compute_density_taper(2000.0, 0.0);
    EXPECT_NEAR(mid_k_weight, std::exp(-1.0), 1e-10);
}

// Test kx-only taper
TEST_F(KSpacePreprocessorTest, KxOnlyTaper) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();
    auto* kx_taper = taper->mutable_kx_taper();
    kx_taper->set_k_threshold(1000.0);
    kx_taper->set_k_max_taper(3000.0);
    kx_taper->set_retain_fraction(0.0);
    kx_taper->set_taper_function(TAPER_LINEAR);

    KSpacePreprocessor preprocessor(config);

    // High kx, zero ky: should be tapered
    double kx_weight = preprocessor.compute_density_taper(2000.0, 0.0);
    EXPECT_NEAR(kx_weight, 0.5, 1e-10);  // t=0.5, linear -> 0.5

    // Zero kx, high ky: should NOT be tapered (kx taper only)
    double ky_weight = preprocessor.compute_density_taper(0.0, 2000.0);
    EXPECT_DOUBLE_EQ(ky_weight, 1.0);  // No ky taper configured
}

// Test ky-only taper
TEST_F(KSpacePreprocessorTest, KyOnlyTaper) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();
    auto* ky_taper = taper->mutable_ky_taper();
    ky_taper->set_k_threshold(1000.0);
    ky_taper->set_k_max_taper(3000.0);
    ky_taper->set_retain_fraction(0.0);
    ky_taper->set_taper_function(TAPER_LINEAR);

    KSpacePreprocessor preprocessor(config);

    // High ky, zero kx: should be tapered
    double ky_weight = preprocessor.compute_density_taper(0.0, 2000.0);
    EXPECT_NEAR(ky_weight, 0.5, 1e-10);

    // Zero ky, high kx: should NOT be tapered
    double kx_weight = preprocessor.compute_density_taper(2000.0, 0.0);
    EXPECT_DOUBLE_EQ(kx_weight, 1.0);
}

// Test combined kx + ky tapers (multiplicative)
TEST_F(KSpacePreprocessorTest, CombinedKxKyTapers) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();

    auto* kx_taper = taper->mutable_kx_taper();
    kx_taper->set_k_threshold(1000.0);
    kx_taper->set_k_max_taper(3000.0);
    kx_taper->set_retain_fraction(0.0);
    kx_taper->set_taper_function(TAPER_LINEAR);

    auto* ky_taper = taper->mutable_ky_taper();
    ky_taper->set_k_threshold(1000.0);
    ky_taper->set_k_max_taper(3000.0);
    ky_taper->set_retain_fraction(0.0);
    ky_taper->set_taper_function(TAPER_LINEAR);

    KSpacePreprocessor preprocessor(config);

    // Both kx=2000 and ky=2000: each contributes 0.5, total = 0.5 * 0.5 = 0.25
    double combined_weight = preprocessor.compute_density_taper(2000.0, 2000.0);
    EXPECT_NEAR(combined_weight, 0.25, 1e-10);
}

// Test combined isotropic + axis tapers (multiplicative)
TEST_F(KSpacePreprocessorTest, CombinedIsotropicAndAxisTapers) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();

    auto* iso = taper->mutable_isotropic();
    iso->set_k_threshold(1000.0);
    iso->set_k_max_taper(5000.0);
    iso->set_retain_fraction(0.0);
    iso->set_taper_function(TAPER_LINEAR);

    auto* ky_taper = taper->mutable_ky_taper();
    ky_taper->set_k_threshold(500.0);
    ky_taper->set_k_max_taper(2500.0);
    ky_taper->set_retain_fraction(0.0);
    ky_taper->set_taper_function(TAPER_LINEAR);

    KSpacePreprocessor preprocessor(config);

    // kx=0, ky=1500:
    // - isotropic: k_mag=1500, t=(1500-1000)/(5000-1000)=0.125, weight=0.875
    // - ky: t=(1500-500)/(2500-500)=0.5, weight=0.5
    // - combined: 0.875 * 0.5 = 0.4375
    double combined_weight = preprocessor.compute_density_taper(0.0, 1500.0);
    EXPECT_NEAR(combined_weight, 0.4375, 1e-10);
}

// Test that taper is applied before apodization in apply()
TEST_F(KSpacePreprocessorTest, TaperAppliedInApply) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);  // No apodization
    auto* taper = config.mutable_density_taper();
    auto* iso = taper->mutable_isotropic();
    iso->set_k_threshold(500.0);
    iso->set_k_max_taper(4000.0);
    iso->set_retain_fraction(0.1);
    iso->set_taper_function(TAPER_LINEAR);

    KSpacePreprocessor preprocessor(config);
    auto result = preprocessor.apply(slice_, k_max_x_, k_max_y_);

    // DC sample (k=0, below threshold): should have highest relative weight
    // High-k sample (k=4000): should have lowest relative weight (0.1)
    double dc_magnitude = std::hypot(result.samples(0).value().real(),
                                      result.samples(0).value().imag());
    double high_k_magnitude = std::hypot(result.samples(4).value().real(),
                                          result.samples(4).value().imag());

    // DC should be much higher than high-k sample
    EXPECT_GT(dc_magnitude, high_k_magnitude * 5);  // At least 5x higher
}

// Test negative k values (should use absolute value)
TEST_F(KSpacePreprocessorTest, NegativeKValues) {
    DirectAdjointConfig config;
    config.set_apodization(DirectAdjointConfig::APOD_NONE);
    auto* taper = config.mutable_density_taper();
    auto* kx_taper = taper->mutable_kx_taper();
    kx_taper->set_k_threshold(1000.0);
    kx_taper->set_k_max_taper(3000.0);
    kx_taper->set_retain_fraction(0.0);
    kx_taper->set_taper_function(TAPER_LINEAR);

    KSpacePreprocessor preprocessor(config);

    // Positive and negative kx should give same weight
    double pos_weight = preprocessor.compute_density_taper(2000.0, 0.0);
    double neg_weight = preprocessor.compute_density_taper(-2000.0, 0.0);
    EXPECT_DOUBLE_EQ(pos_weight, neg_weight);
}

}  // namespace aurora::reconstruction
