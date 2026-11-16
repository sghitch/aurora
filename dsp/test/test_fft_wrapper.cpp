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

#include "dsp/fft_wrapper.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numeric>

namespace aurora::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

class FftWrapperTest : public ::testing::Test {
protected:
    static constexpr uint32_t kDefaultSize = 1024;
};

TEST_F(FftWrapperTest, ConstructorValidatesSize) {
    // Valid sizes should not throw
    EXPECT_NO_THROW(FftWrapper(64));
    EXPECT_NO_THROW(FftWrapper(128));
    EXPECT_NO_THROW(FftWrapper(1024));
    EXPECT_NO_THROW(FftWrapper(2048));
    EXPECT_NO_THROW(FftWrapper(4096));

    // Zero size should throw
    EXPECT_THROW(FftWrapper(0), std::invalid_argument);
}

TEST_F(FftWrapperTest, ConstructorCreatesInversePlan) {
    FftWrapper forward(kDefaultSize, false);
    FftWrapper inverse(kDefaultSize, true);

    EXPECT_FALSE(forward.is_inverse());
    EXPECT_TRUE(inverse.is_inverse());
    EXPECT_EQ(forward.size(), kDefaultSize);
    EXPECT_EQ(inverse.size(), kDefaultSize);
}

TEST_F(FftWrapperTest, MoveConstructor) {
    FftWrapper original(kDefaultSize);
    uint32_t original_size = original.size();

    FftWrapper moved(std::move(original));
    EXPECT_EQ(moved.size(), original_size);
}

TEST_F(FftWrapperTest, MoveAssignment) {
    FftWrapper wrapper1(512);
    FftWrapper wrapper2(1024);

    wrapper1 = std::move(wrapper2);
    EXPECT_EQ(wrapper1.size(), 1024u);
}

TEST_F(FftWrapperTest, ExecuteValidatesInputSize) {
    FftWrapper fft(kDefaultSize);

    // Wrong size should throw
    std::vector<std::complex<double>> wrong_size(kDefaultSize / 2);
    EXPECT_THROW(fft.execute(wrong_size), std::invalid_argument);

    // Correct size should not throw
    std::vector<std::complex<double>> correct_size(kDefaultSize);
    EXPECT_NO_THROW(fft.execute(correct_size));
}

TEST_F(FftWrapperTest, ExecuteInplaceValidatesSize) {
    FftWrapper fft(kDefaultSize);

    std::vector<std::complex<double>> wrong_size(kDefaultSize / 2);
    EXPECT_THROW(fft.execute_inplace(wrong_size), std::invalid_argument);

    std::vector<std::complex<double>> correct_size(kDefaultSize);
    EXPECT_NO_THROW(fft.execute_inplace(correct_size));
}

TEST_F(FftWrapperTest, ExecuteRealValidatesSize) {
    FftWrapper fft(kDefaultSize);

    std::vector<double> wrong_size(kDefaultSize / 2);
    EXPECT_THROW(fft.execute_real(wrong_size), std::invalid_argument);

    std::vector<double> correct_size(kDefaultSize);
    EXPECT_NO_THROW(fft.execute_real(correct_size));
}

TEST_F(FftWrapperTest, ExecuteRealThrowsForInverseFft) {
    FftWrapper inverse_fft(kDefaultSize, true);
    std::vector<double> input(kDefaultSize);
    EXPECT_THROW(inverse_fft.execute_real(input), std::logic_error);
}

// Test: FFT of pure sine wave produces peak at correct bin
TEST_F(FftWrapperTest, SineWavePeakAtCorrectBin) {
    const uint32_t size = 1024;
    FftWrapper fft(size);

    // Generate sine wave at bin 10 (frequency = 10 * sample_rate / size)
    const int target_bin = 10;
    std::vector<std::complex<double>> input(size);
    for (uint32_t i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(size);
        double value = std::sin(kTwoPi * target_bin * t);
        input[i] = std::complex<double>(value, 0.0);
    }

    auto result = fft.execute(input);
    ASSERT_EQ(result.size(), size);

    // Find bin with maximum magnitude
    int max_bin = 0;
    double max_mag = 0.0;
    for (uint32_t i = 0; i < size / 2; ++i) {
        double mag = std::abs(result[i]);
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = static_cast<int>(i);
        }
    }

    EXPECT_EQ(max_bin, target_bin)
        << "FFT peak should be at bin " << target_bin;
}

// Test: Forward -> Inverse round-trip recovers original signal
TEST_F(FftWrapperTest, InverseRecovery) {
    const uint32_t size = 512;
    FftWrapper forward(size, false);
    FftWrapper inverse(size, true);

    // Create test signal
    std::vector<std::complex<double>> original(size);
    for (uint32_t i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(size);
        original[i] = std::complex<double>(
            std::sin(kTwoPi * 5 * t) + 0.5 * std::cos(kTwoPi * 13 * t),
            0.3 * std::sin(kTwoPi * 7 * t));
    }

    // Forward FFT
    auto freq_domain = forward.execute(original);

    // Inverse FFT
    auto recovered = inverse.execute(freq_domain);

    // IFFT needs normalization by 1/N
    for (auto& val : recovered) {
        val /= static_cast<double>(size);
    }

    // Check recovery within tolerance
    const double tolerance = 1e-10;
    for (uint32_t i = 0; i < size; ++i) {
        EXPECT_NEAR(recovered[i].real(), original[i].real(), tolerance)
            << "Real part mismatch at index " << i;
        EXPECT_NEAR(recovered[i].imag(), original[i].imag(), tolerance)
            << "Imaginary part mismatch at index " << i;
    }
}

// Test: In-place execution produces identical results to copy
TEST_F(FftWrapperTest, InplaceMatchesCopy) {
    const uint32_t size = 256;
    FftWrapper fft(size);

    // Create test signal
    std::vector<std::complex<double>> signal(size);
    for (uint32_t i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(size);
        signal[i] = std::complex<double>(
            std::sin(kTwoPi * 3 * t) + 0.7 * std::cos(kTwoPi * 11 * t),
            0.0);
    }

    // Execute copy
    auto copy_result = fft.execute(signal);

    // Execute in-place
    auto inplace_data = signal;  // Make a copy first
    fft.execute_inplace(inplace_data);

    // Results should be identical
    ASSERT_EQ(copy_result.size(), inplace_data.size());
    for (uint32_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(copy_result[i].real(), inplace_data[i].real())
            << "Real part mismatch at index " << i;
        EXPECT_DOUBLE_EQ(copy_result[i].imag(), inplace_data[i].imag())
            << "Imaginary part mismatch at index " << i;
    }
}

// Test: Real-to-complex FFT produces correct spectrum
TEST_F(FftWrapperTest, RealFftCorrectSpectrum) {
    const uint32_t size = 512;
    FftWrapper fft(size);

    // Generate real sine wave at bin 20
    const int target_bin = 20;
    std::vector<double> real_input(size);
    for (uint32_t i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(size);
        real_input[i] = std::sin(kTwoPi * target_bin * t);
    }

    auto result = fft.execute_real(real_input);

    // Real FFT returns size/2 + 1 complex values
    EXPECT_EQ(result.size(), size / 2 + 1);

    // Find peak
    int max_bin = 0;
    double max_mag = 0.0;
    for (size_t i = 0; i < result.size(); ++i) {
        double mag = std::abs(result[i]);
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = static_cast<int>(i);
        }
    }

    EXPECT_EQ(max_bin, target_bin)
        << "Real FFT peak should be at bin " << target_bin;
}

// Test: Real FFT matches complex FFT for real input
TEST_F(FftWrapperTest, RealFftMatchesComplexFft) {
    const uint32_t size = 256;
    FftWrapper fft(size);

    // Generate real signal
    std::vector<double> real_input(size);
    std::vector<std::complex<double>> complex_input(size);
    for (uint32_t i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(size);
        double value = std::sin(kTwoPi * 7 * t) + 0.5 * std::cos(kTwoPi * 23 * t);
        real_input[i] = value;
        complex_input[i] = std::complex<double>(value, 0.0);
    }

    auto real_result = fft.execute_real(real_input);
    auto complex_result = fft.execute(complex_input);

    // Compare first size/2 + 1 bins (the positive frequency half)
    const double tolerance = 1e-10;
    for (size_t i = 0; i < real_result.size(); ++i) {
        EXPECT_NEAR(real_result[i].real(), complex_result[i].real(), tolerance)
            << "Real part mismatch at bin " << i;
        EXPECT_NEAR(real_result[i].imag(), complex_result[i].imag(), tolerance)
            << "Imaginary part mismatch at bin " << i;
    }
}

// Test: Parseval's theorem - energy preservation
// Sum of |x[n]|^2 = (1/N) * Sum of |X[k]|^2
TEST_F(FftWrapperTest, ParsevalEnergyPreservation) {
    const uint32_t size = 512;
    FftWrapper fft(size);

    // Create test signal
    std::vector<std::complex<double>> signal(size);
    for (uint32_t i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(size);
        signal[i] = std::complex<double>(
            std::sin(kTwoPi * 4 * t) + 0.3 * std::cos(kTwoPi * 17 * t),
            0.2 * std::sin(kTwoPi * 9 * t));
    }

    // Calculate time-domain energy
    double time_energy = 0.0;
    for (const auto& x : signal) {
        time_energy += std::norm(x);  // |x|^2
    }

    // FFT and calculate frequency-domain energy
    auto spectrum = fft.execute(signal);
    double freq_energy = 0.0;
    for (const auto& X : spectrum) {
        freq_energy += std::norm(X);  // |X|^2
    }
    freq_energy /= static_cast<double>(size);  // Normalize

    // Energy should be preserved within tolerance
    const double tolerance = 1e-10;
    EXPECT_NEAR(time_energy, freq_energy, tolerance)
        << "Parseval's theorem: time energy = " << time_energy
        << ", freq energy/N = " << freq_energy;
}

// Test: FFT linearity - FFT(a + b) = FFT(a) + FFT(b)
TEST_F(FftWrapperTest, Linearity) {
    const uint32_t size = 256;
    FftWrapper fft(size);

    // Create two different signals
    std::vector<std::complex<double>> signal_a(size);
    std::vector<std::complex<double>> signal_b(size);
    std::vector<std::complex<double>> signal_sum(size);

    for (uint32_t i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(size);
        signal_a[i] = std::complex<double>(std::sin(kTwoPi * 5 * t), 0.0);
        signal_b[i] = std::complex<double>(std::cos(kTwoPi * 13 * t), 0.0);
        signal_sum[i] = signal_a[i] + signal_b[i];
    }

    // FFT of individual signals
    auto fft_a = fft.execute(signal_a);
    auto fft_b = fft.execute(signal_b);

    // FFT of sum
    auto fft_sum = fft.execute(signal_sum);

    // FFT(a) + FFT(b) should equal FFT(a + b)
    const double tolerance = 1e-10;
    for (uint32_t i = 0; i < size; ++i) {
        std::complex<double> sum_of_ffts = fft_a[i] + fft_b[i];
        EXPECT_NEAR(fft_sum[i].real(), sum_of_ffts.real(), tolerance)
            << "Linearity real part mismatch at bin " << i;
        EXPECT_NEAR(fft_sum[i].imag(), sum_of_ffts.imag(), tolerance)
            << "Linearity imaginary part mismatch at bin " << i;
    }
}

// Test: FFT of DC signal produces peak at bin 0
TEST_F(FftWrapperTest, DcSignal) {
    const uint32_t size = 128;
    FftWrapper fft(size);

    const double dc_value = 3.5;
    std::vector<std::complex<double>> signal(size, std::complex<double>(dc_value, 0.0));

    auto result = fft.execute(signal);

    // DC component should be at bin 0 with magnitude = N * dc_value
    EXPECT_NEAR(result[0].real(), size * dc_value, 1e-10);
    EXPECT_NEAR(result[0].imag(), 0.0, 1e-10);

    // All other bins should be zero
    for (uint32_t i = 1; i < size; ++i) {
        EXPECT_NEAR(std::abs(result[i]), 0.0, 1e-10)
            << "Non-zero value at bin " << i << " for DC signal";
    }
}

// Test: Different FFT sizes work correctly
TEST_F(FftWrapperTest, DifferentSizes) {
    std::vector<uint32_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};

    for (uint32_t size : sizes) {
        FftWrapper fft(size);
        EXPECT_EQ(fft.size(), size);

        // Test basic functionality
        std::vector<std::complex<double>> signal(size);
        for (uint32_t i = 0; i < size; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(size);
            signal[i] = std::complex<double>(std::sin(kTwoPi * 3 * t), 0.0);
        }

        auto result = fft.execute(signal);
        EXPECT_EQ(result.size(), size) << "FFT size " << size << " failed";
    }
}

// Test: Non-power-of-2 sizes still work (with potentially reduced performance)
TEST_F(FftWrapperTest, NonPowerOfTwoSize) {
    const uint32_t size = 300;  // Not a power of 2
    FftWrapper fft(size);
    EXPECT_EQ(fft.size(), size);

    std::vector<std::complex<double>> signal(size);
    for (uint32_t i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(size);
        signal[i] = std::complex<double>(std::sin(kTwoPi * 5 * t), 0.0);
    }

    auto result = fft.execute(signal);
    EXPECT_EQ(result.size(), size);
}

// Test: Multiple FFT sizes for range processing
TEST_F(FftWrapperTest, RangeProcessingSizes) {
    // Typical chirp FFT sizes
    std::vector<uint32_t> range_fft_sizes = {2048, 4096, 8192};

    for (uint32_t size : range_fft_sizes) {
        FftWrapper fft(size);
        EXPECT_EQ(fft.size(), size);

        // Verify real FFT output size
        std::vector<double> real_signal(size, 0.0);
        auto result = fft.execute_real(real_signal);
        EXPECT_EQ(result.size(), size / 2 + 1);
    }
}

}  // namespace
}  // namespace aurora::dsp
