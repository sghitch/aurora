#include "dsp/window_functions.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numeric>

namespace aurora::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

// Helper to create WindowConfig with a specific type
WindowConfig make_config(WindowConfig::WindowType type) {
    WindowConfig config;
    config.set_type(type);
    return config;
}

// Helper to create Kaiser WindowConfig with beta parameter
WindowConfig make_kaiser(double beta) {
    WindowConfig config;
    config.set_type(WindowConfig::WINDOW_KAISER);
    config.set_parameter(beta);
    return config;
}

class WindowFunctionsTest : public ::testing::Test {
protected:
    static constexpr size_t kDefaultLength = 1024;

    // Pre-built configs for common window types
    WindowConfig rectangular_ = make_config(WindowConfig::WINDOW_RECTANGULAR);
    WindowConfig hann_ = make_config(WindowConfig::WINDOW_HANN);
    WindowConfig hamming_ = make_config(WindowConfig::WINDOW_HAMMING);
    WindowConfig blackman_ = make_config(WindowConfig::WINDOW_BLACKMAN);
    WindowConfig kaiser_5_ = make_kaiser(5.0);
};

// Test: Empty window generation
TEST_F(WindowFunctionsTest, EmptyWindow) {
    auto window = WindowFunctions::generate(hann_, 0);
    EXPECT_TRUE(window.empty());
}

// Test: Single element window
TEST_F(WindowFunctionsTest, SingleElement) {
    std::vector<WindowConfig> configs = {rectangular_, hann_, hamming_, blackman_, kaiser_5_};

    for (const auto& config : configs) {
        auto window = WindowFunctions::generate(config, 1);
        ASSERT_EQ(window.size(), 1u);
        EXPECT_DOUBLE_EQ(window[0], 1.0);
    }
}

// Test: Rectangular window is all ones
TEST_F(WindowFunctionsTest, RectangularAllOnes) {
    auto window = WindowFunctions::generate(rectangular_, kDefaultLength);
    ASSERT_EQ(window.size(), kDefaultLength);

    for (size_t i = 0; i < kDefaultLength; ++i) {
        EXPECT_DOUBLE_EQ(window[i], 1.0) << "at index " << i;
    }
}

// Test: Hann window symmetry (w[n] = w[N-1-n])
TEST_F(WindowFunctionsTest, HannSymmetry) {
    auto window = WindowFunctions::generate(hann_, kDefaultLength);
    ASSERT_EQ(window.size(), kDefaultLength);

    for (size_t i = 0; i < kDefaultLength / 2; ++i) {
        EXPECT_NEAR(window[i], window[kDefaultLength - 1 - i], 1e-10)
            << "Symmetry failed at index " << i;
    }
}

// Test: Hamming window symmetry
TEST_F(WindowFunctionsTest, HammingSymmetry) {
    auto window = WindowFunctions::generate(hamming_, kDefaultLength);
    ASSERT_EQ(window.size(), kDefaultLength);

    for (size_t i = 0; i < kDefaultLength / 2; ++i) {
        EXPECT_NEAR(window[i], window[kDefaultLength - 1 - i], 1e-10)
            << "Symmetry failed at index " << i;
    }
}

// Test: Blackman window symmetry
TEST_F(WindowFunctionsTest, BlackmanSymmetry) {
    auto window = WindowFunctions::generate(blackman_, kDefaultLength);
    ASSERT_EQ(window.size(), kDefaultLength);

    for (size_t i = 0; i < kDefaultLength / 2; ++i) {
        EXPECT_NEAR(window[i], window[kDefaultLength - 1 - i], 1e-10)
            << "Symmetry failed at index " << i;
    }
}

// Test: Kaiser window symmetry
TEST_F(WindowFunctionsTest, KaiserSymmetry) {
    auto window = WindowFunctions::generate(kaiser_5_, kDefaultLength);
    ASSERT_EQ(window.size(), kDefaultLength);

    for (size_t i = 0; i < kDefaultLength / 2; ++i) {
        EXPECT_NEAR(window[i], window[kDefaultLength - 1 - i], 1e-10)
            << "Symmetry failed at index " << i;
    }
}

// Test: All window values are in range [0, 1] (with tolerance for floating point)
TEST_F(WindowFunctionsTest, WindowRange) {
    std::vector<WindowConfig> configs = {rectangular_, hann_, hamming_, blackman_, kaiser_5_};

    const double tolerance = 1e-15;  // Floating point tolerance
    for (const auto& config : configs) {
        auto window = WindowFunctions::generate(config, kDefaultLength);
        for (size_t i = 0; i < window.size(); ++i) {
            EXPECT_GE(window[i], -tolerance) << "Value below 0 at index " << i;
            EXPECT_LE(window[i], 1.0 + tolerance) << "Value above 1 at index " << i;
        }
    }
}

// Test: Hann window endpoints are zero
TEST_F(WindowFunctionsTest, HannEndpoints) {
    auto window = WindowFunctions::generate(hann_, kDefaultLength);
    EXPECT_NEAR(window[0], 0.0, 1e-10);
    EXPECT_NEAR(window[kDefaultLength - 1], 0.0, 1e-10);
}

// Test: Blackman window endpoints are approximately zero
TEST_F(WindowFunctionsTest, BlackmanEndpoints) {
    auto window = WindowFunctions::generate(blackman_, kDefaultLength);
    EXPECT_NEAR(window[0], 0.0, 1e-10);
    EXPECT_NEAR(window[kDefaultLength - 1], 0.0, 1e-10);
}

// Test: Hamming window endpoints are non-zero (0.08)
TEST_F(WindowFunctionsTest, HammingEndpoints) {
    auto window = WindowFunctions::generate(hamming_, kDefaultLength);
    EXPECT_NEAR(window[0], 0.08, 1e-10);
    EXPECT_NEAR(window[kDefaultLength - 1], 0.08, 1e-10);
}

// Test: Coherent gain matches theoretical values
TEST_F(WindowFunctionsTest, CoherentGain) {
    const size_t n = 2048;
    const double tolerance = 0.01;  // 1% tolerance

    // Rectangular CG = 1.0
    EXPECT_NEAR(WindowFunctions::coherent_gain(rectangular_, n), 1.0, tolerance);

    // Hann CG = 0.5
    EXPECT_NEAR(WindowFunctions::coherent_gain(hann_, n), 0.5, tolerance);

    // Hamming CG = 0.54
    EXPECT_NEAR(WindowFunctions::coherent_gain(hamming_, n), 0.54, tolerance);

    // Blackman CG = 0.42
    EXPECT_NEAR(WindowFunctions::coherent_gain(blackman_, n), 0.42, tolerance);
}

// Test: Equivalent noise bandwidth matches theoretical values
TEST_F(WindowFunctionsTest, NoiseBandwidth) {
    const size_t n = 2048;
    const double tolerance = 0.02;  // 2% tolerance

    // Rectangular ENBW = 1.0
    EXPECT_NEAR(WindowFunctions::noise_bandwidth(rectangular_, n), 1.0, tolerance);

    // Hann ENBW = 1.5
    EXPECT_NEAR(WindowFunctions::noise_bandwidth(hann_, n), 1.5, tolerance);

    // Hamming ENBW = 1.36
    EXPECT_NEAR(WindowFunctions::noise_bandwidth(hamming_, n), 1.36, tolerance);

    // Blackman ENBW = 1.73
    EXPECT_NEAR(WindowFunctions::noise_bandwidth(blackman_, n), 1.73, tolerance);
}

// Test: Kaiser without parameter throws exception
TEST_F(WindowFunctionsTest, KaiserRequiresParameter) {
    WindowConfig kaiser_no_param;
    kaiser_no_param.set_type(WindowConfig::WINDOW_KAISER);
    // No parameter set

    EXPECT_THROW(WindowFunctions::generate(kaiser_no_param, kDefaultLength),
                 std::invalid_argument);
    EXPECT_THROW(WindowFunctions::sidelobe_level_db(kaiser_no_param),
                 std::invalid_argument);
    EXPECT_THROW(WindowFunctions::main_lobe_width(kaiser_no_param),
                 std::invalid_argument);
}

// Test: Kaiser beta=0 produces rectangular-like window
TEST_F(WindowFunctionsTest, KaiserBetaZero) {
    auto kaiser_0 = make_kaiser(0.0);
    auto kaiser_window = WindowFunctions::generate(kaiser_0, kDefaultLength);
    auto rect_window = WindowFunctions::generate(rectangular_, kDefaultLength);

    for (size_t i = 0; i < kDefaultLength; ++i) {
        EXPECT_NEAR(kaiser_window[i], rect_window[i], 1e-10)
            << "Kaiser(beta=0) should equal rectangular at index " << i;
    }
}

// Test: Kaiser higher beta produces lower sidelobes (but we test CG/ENBW trends)
TEST_F(WindowFunctionsTest, KaiserBetaProgression) {
    const size_t n = 1024;

    auto kaiser_5 = make_kaiser(5.0);
    auto kaiser_7 = make_kaiser(7.0);
    auto kaiser_9 = make_kaiser(9.0);

    double cg_5 = WindowFunctions::coherent_gain(kaiser_5, n);
    double cg_7 = WindowFunctions::coherent_gain(kaiser_7, n);
    double cg_9 = WindowFunctions::coherent_gain(kaiser_9, n);

    // Coherent gain decreases with increasing beta
    EXPECT_GT(cg_5, cg_7);
    EXPECT_GT(cg_7, cg_9);

    double enbw_5 = WindowFunctions::noise_bandwidth(kaiser_5, n);
    double enbw_7 = WindowFunctions::noise_bandwidth(kaiser_7, n);
    double enbw_9 = WindowFunctions::noise_bandwidth(kaiser_9, n);

    // ENBW increases with increasing beta
    EXPECT_LT(enbw_5, enbw_7);
    EXPECT_LT(enbw_7, enbw_9);
}

// Test: Apply in-place for real signal
TEST_F(WindowFunctionsTest, ApplyInplaceReal) {
    const size_t n = 256;
    std::vector<double> signal(n, 2.0);

    auto window = WindowFunctions::generate(hann_, n);
    WindowFunctions::apply_inplace(signal, hann_);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(signal[i], 2.0 * window[i], 1e-10)
            << "In-place application failed at index " << i;
    }
}

// Test: Apply in-place for complex signal
TEST_F(WindowFunctionsTest, ApplyInplaceComplex) {
    const size_t n = 256;
    std::vector<std::complex<double>> signal(n, std::complex<double>(3.0, 4.0));

    auto window = WindowFunctions::generate(hamming_, n);
    WindowFunctions::apply_inplace(signal, hamming_);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(signal[i].real(), 3.0 * window[i], 1e-10)
            << "Real part in-place application failed at index " << i;
        EXPECT_NEAR(signal[i].imag(), 4.0 * window[i], 1e-10)
            << "Imaginary part in-place application failed at index " << i;
    }
}

// Test: Sidelobe levels are negative
TEST_F(WindowFunctionsTest, SidelobeLevels) {
    EXPECT_LT(WindowFunctions::sidelobe_level_db(rectangular_), 0.0);
    EXPECT_LT(WindowFunctions::sidelobe_level_db(hann_), 0.0);
    EXPECT_LT(WindowFunctions::sidelobe_level_db(hamming_), 0.0);
    EXPECT_LT(WindowFunctions::sidelobe_level_db(blackman_), 0.0);
    EXPECT_LT(WindowFunctions::sidelobe_level_db(kaiser_5_), 0.0);
}

// Test: Sidelobe level ordering (rect > hann > hamming > blackman)
TEST_F(WindowFunctionsTest, SidelobeLevelOrdering) {
    double rect = WindowFunctions::sidelobe_level_db(rectangular_);
    double hann = WindowFunctions::sidelobe_level_db(hann_);
    double hamming = WindowFunctions::sidelobe_level_db(hamming_);
    double blackman = WindowFunctions::sidelobe_level_db(blackman_);

    // Higher (less negative) sidelobe level = worse
    EXPECT_GT(rect, hann);
    EXPECT_GT(hann, hamming);
    EXPECT_GT(hamming, blackman);
}

// Test: Main lobe width ordering (rect < hamming < hann < blackman)
TEST_F(WindowFunctionsTest, MainLobeWidthOrdering) {
    double rect = WindowFunctions::main_lobe_width(rectangular_);
    double hann = WindowFunctions::main_lobe_width(hann_);
    double hamming = WindowFunctions::main_lobe_width(hamming_);
    double blackman = WindowFunctions::main_lobe_width(blackman_);

    // Narrower main lobe = better resolution
    EXPECT_LT(rect, hamming);
    EXPECT_LT(hamming, hann);
    EXPECT_LT(hann, blackman);
}

// Test: Main lobe width values are positive
TEST_F(WindowFunctionsTest, MainLobeWidthPositive) {
    EXPECT_GT(WindowFunctions::main_lobe_width(rectangular_), 0.0);
    EXPECT_GT(WindowFunctions::main_lobe_width(hann_), 0.0);
    EXPECT_GT(WindowFunctions::main_lobe_width(hamming_), 0.0);
    EXPECT_GT(WindowFunctions::main_lobe_width(blackman_), 0.0);
    EXPECT_GT(WindowFunctions::main_lobe_width(kaiser_5_), 0.0);
}

// Test: Coherent gain for empty window
TEST_F(WindowFunctionsTest, CoherentGainEmpty) {
    EXPECT_DOUBLE_EQ(WindowFunctions::coherent_gain(hann_, 0), 0.0);
}

// Test: Noise bandwidth for empty window
TEST_F(WindowFunctionsTest, NoiseBandwidthEmpty) {
    EXPECT_DOUBLE_EQ(WindowFunctions::noise_bandwidth(hann_, 0), 0.0);
}

// Test: Different window sizes work correctly
TEST_F(WindowFunctionsTest, DifferentSizes) {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};

    for (size_t size : sizes) {
        auto window = WindowFunctions::generate(hann_, size);
        EXPECT_EQ(window.size(), size) << "Size mismatch for " << size;

        // Verify symmetry
        for (size_t i = 0; i < size / 2; ++i) {
            EXPECT_NEAR(window[i], window[size - 1 - i], 1e-10)
                << "Symmetry failed for size " << size << " at index " << i;
        }
    }
}

// Test: Window center is maximum for symmetric windows
TEST_F(WindowFunctionsTest, CenterIsMaximum) {
    std::vector<WindowConfig> configs = {hann_, hamming_, blackman_, kaiser_5_};

    const size_t n = 1025;  // Odd size so center is exactly middle
    size_t center = n / 2;

    for (const auto& config : configs) {
        auto window = WindowFunctions::generate(config, n);

        for (size_t i = 0; i < n; ++i) {
            EXPECT_LE(window[i], window[center] + 1e-10)
                << "Center is not maximum for type at index " << i;
        }
    }
}

// Test: Bessel I0 for Kaiser window produces reasonable values
TEST_F(WindowFunctionsTest, KaiserValuesReasonable) {
    const size_t n = 512;
    auto kaiser_7 = make_kaiser(7.0);

    auto window = WindowFunctions::generate(kaiser_7, n);

    // Kaiser window should be 1.0 at center, tapering to edges
    size_t center = n / 2;
    EXPECT_NEAR(window[center], 1.0, 0.01);
    EXPECT_GT(window[center], window[0]);
    EXPECT_GT(window[center], window[n - 1]);
}

// Test: Theoretical sidelobe values match spec
TEST_F(WindowFunctionsTest, SidelobeLevelValues) {
    EXPECT_NEAR(WindowFunctions::sidelobe_level_db(rectangular_), -13.0, 1.0);
    EXPECT_NEAR(WindowFunctions::sidelobe_level_db(hann_), -31.0, 1.0);
    EXPECT_NEAR(WindowFunctions::sidelobe_level_db(hamming_), -42.0, 1.0);
    EXPECT_NEAR(WindowFunctions::sidelobe_level_db(blackman_), -58.0, 1.0);
}

// Test: Theoretical main lobe width values match spec
TEST_F(WindowFunctionsTest, MainLobeWidthValues) {
    EXPECT_NEAR(WindowFunctions::main_lobe_width(rectangular_), 0.89, 0.1);
    EXPECT_NEAR(WindowFunctions::main_lobe_width(hann_), 1.44, 0.1);
    EXPECT_NEAR(WindowFunctions::main_lobe_width(hamming_), 1.30, 0.1);
    EXPECT_NEAR(WindowFunctions::main_lobe_width(blackman_), 1.68, 0.1);
}

}  // namespace
}  // namespace aurora::dsp
