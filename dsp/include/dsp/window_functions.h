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

#include <complex>
#include <cstddef>
#include <vector>

#include "dsp_config.pb.h"

namespace aurora::dsp {

/// Spectral window functions for range FFT processing.
///
/// Proper windowing reduces sidelobes at the cost of main lobe width.
/// Key tradeoffs:
/// - Rectangular: Best resolution, worst sidelobes (-13 dB)
/// - Hann/Hamming: Good balance for general use
/// - Blackman: Best sidelobe suppression (-58 dB), widest main lobe
/// - Kaiser: Adjustable tradeoff via beta parameter
///
/// Usage:
/// @code
///   WindowConfig config;
///   config.set_type(WindowConfig::WINDOW_HANN);
///   auto window = WindowFunctions::generate(config, 2048);
///   WindowFunctions::apply_inplace(signal, config);
/// @endcode
class WindowFunctions {
public:
    /// Generate window coefficients.
    /// @param config Window configuration (type and optional parameter)
    /// @param length Number of window samples
    /// @return Vector of window coefficients in range [0, 1]
    static std::vector<double> generate(const WindowConfig& config, size_t length);

    /// Apply window to real signal in-place.
    /// @param signal Real signal to window (modified in-place)
    /// @param config Window configuration (type and optional parameter)
    static void apply_inplace(std::vector<double>& signal, const WindowConfig& config);

    /// Apply window to complex signal in-place.
    /// @param signal Complex signal to window (modified in-place)
    /// @param config Window configuration (type and optional parameter)
    static void apply_inplace(std::vector<std::complex<double>>& signal,
                              const WindowConfig& config);

    /// Get coherent gain for a window configuration.
    /// CG = (1/N) * sum(w[n])
    /// Used for amplitude calibration: A_corrected = A_measured / CG
    /// @param config Window configuration
    /// @param length Window length
    /// @return Coherent gain (typically 0.4 to 1.0)
    static double coherent_gain(const WindowConfig& config, size_t length);

    /// Get equivalent noise bandwidth (ENBW) in bins.
    /// ENBW = N * sum(w[n]^2) / sum(w[n])^2
    /// Used to calculate noise floor after FFT.
    /// @param config Window configuration
    /// @param length Window length
    /// @return ENBW in bins (typically 1.0 to 2.0)
    static double noise_bandwidth(const WindowConfig& config, size_t length);

    /// Get theoretical peak sidelobe level in dB.
    /// @param config Window configuration
    /// @return Sidelobe level in dB (negative value)
    static double sidelobe_level_db(const WindowConfig& config);

    /// Get 3dB main lobe width in bins.
    /// @param config Window configuration
    /// @return Main lobe width in bins
    static double main_lobe_width(const WindowConfig& config);

private:
    // Window generation functions
    static std::vector<double> rectangular(size_t n);
    static std::vector<double> hann(size_t n);
    static std::vector<double> hamming(size_t n);
    static std::vector<double> blackman(size_t n);
    static std::vector<double> kaiser(size_t n, double beta);

    // Modified Bessel function I_0 for Kaiser window
    static double bessel_i0(double x);
};

}  // namespace aurora::dsp
