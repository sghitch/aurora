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

#include "dsp/window_functions.h"

#include <cmath>
#include <stdexcept>

namespace aurora::dsp {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
}  // namespace

std::vector<double> WindowFunctions::generate(const WindowConfig& config, size_t length) {
    if (length == 0) {
        return {};
    }

    switch (config.type()) {
        case WindowConfig::WINDOW_RECTANGULAR:
            return rectangular(length);
        case WindowConfig::WINDOW_HANN:
            return hann(length);
        case WindowConfig::WINDOW_HAMMING:
            return hamming(length);
        case WindowConfig::WINDOW_BLACKMAN:
            return blackman(length);
        case WindowConfig::WINDOW_KAISER:
            if (!config.has_parameter()) {
                throw std::invalid_argument("Kaiser window requires beta parameter");
            }
            return kaiser(length, config.parameter());
        default:
            throw std::invalid_argument("Unknown window type");
    }
}

void WindowFunctions::apply_inplace(std::vector<double>& signal, const WindowConfig& config) {
    auto window = generate(config, signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] *= window[i];
    }
}

void WindowFunctions::apply_inplace(std::vector<std::complex<double>>& signal,
                                    const WindowConfig& config) {
    auto window = generate(config, signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] *= window[i];
    }
}

double WindowFunctions::coherent_gain(const WindowConfig& config, size_t length) {
    if (length == 0) {
        return 0.0;
    }

    auto window = generate(config, length);
    double sum = 0.0;
    for (double w : window) {
        sum += w;
    }
    return sum / static_cast<double>(length);
}

double WindowFunctions::noise_bandwidth(const WindowConfig& config, size_t length) {
    if (length == 0) {
        return 0.0;
    }

    auto window = generate(config, length);
    double sum_w = 0.0;
    double sum_w2 = 0.0;
    for (double w : window) {
        sum_w += w;
        sum_w2 += w * w;
    }

    if (sum_w == 0.0) {
        return 0.0;
    }

    return static_cast<double>(length) * sum_w2 / (sum_w * sum_w);
}

double WindowFunctions::sidelobe_level_db(const WindowConfig& config) {
    switch (config.type()) {
        case WindowConfig::WINDOW_RECTANGULAR:
            return -13.0;
        case WindowConfig::WINDOW_HANN:
            return -31.0;
        case WindowConfig::WINDOW_HAMMING:
            return -42.0;
        case WindowConfig::WINDOW_BLACKMAN:
            return -58.0;
        case WindowConfig::WINDOW_KAISER: {
            if (!config.has_parameter()) {
                throw std::invalid_argument("Kaiser window requires beta parameter");
            }
            double beta = config.parameter();
            // Approximate sidelobe level for Kaiser window
            // Based on empirical formula: -14.36 * beta - 7.5
            // for beta > 0, capped at -13 dB for beta = 0
            if (beta <= 0.0) {
                return -13.0;
            }
            return -(14.36 * beta + 7.5);
        }
        default:
            return 0.0;
    }
}

double WindowFunctions::main_lobe_width(const WindowConfig& config) {
    switch (config.type()) {
        case WindowConfig::WINDOW_RECTANGULAR:
            return 0.89;
        case WindowConfig::WINDOW_HANN:
            return 1.44;
        case WindowConfig::WINDOW_HAMMING:
            return 1.30;
        case WindowConfig::WINDOW_BLACKMAN:
            return 1.68;
        case WindowConfig::WINDOW_KAISER: {
            if (!config.has_parameter()) {
                throw std::invalid_argument("Kaiser window requires beta parameter");
            }
            double beta = config.parameter();
            // Approximate main lobe width for Kaiser window
            // Width increases with beta approximately as 0.89 + 0.12 * beta
            if (beta <= 0.0) {
                return 0.89;
            }
            return 0.89 + 0.12 * beta;
        }
        default:
            return 1.0;
    }
}

std::vector<double> WindowFunctions::rectangular(size_t n) {
    return std::vector<double>(n, 1.0);
}

std::vector<double> WindowFunctions::hann(size_t n) {
    std::vector<double> w(n);
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }

    for (size_t i = 0; i < n; ++i) {
        w[i] = 0.5 * (1.0 - std::cos(kTwoPi * static_cast<double>(i) /
                                     static_cast<double>(n - 1)));
    }
    return w;
}

std::vector<double> WindowFunctions::hamming(size_t n) {
    std::vector<double> w(n);
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }

    for (size_t i = 0; i < n; ++i) {
        w[i] = 0.54 - 0.46 * std::cos(kTwoPi * static_cast<double>(i) /
                                      static_cast<double>(n - 1));
    }
    return w;
}

std::vector<double> WindowFunctions::blackman(size_t n) {
    std::vector<double> w(n);
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }

    for (size_t i = 0; i < n; ++i) {
        double t = kTwoPi * static_cast<double>(i) / static_cast<double>(n - 1);
        w[i] = 0.42 - 0.5 * std::cos(t) + 0.08 * std::cos(2.0 * t);
    }
    return w;
}

std::vector<double> WindowFunctions::kaiser(size_t n, double beta) {
    std::vector<double> w(n);
    if (n == 1) {
        w[0] = 1.0;
        return w;
    }

    double denom = bessel_i0(beta);
    double half = static_cast<double>(n - 1) / 2.0;

    for (size_t i = 0; i < n; ++i) {
        double x = (static_cast<double>(i) - half) / half;  // -1 to +1
        double arg = beta * std::sqrt(1.0 - x * x);
        w[i] = bessel_i0(arg) / denom;
    }
    return w;
}

double WindowFunctions::bessel_i0(double x) {
    // Series expansion for modified Bessel function I_0(x)
    // I_0(x) = sum_{k=0}^{inf} (x^2/4)^k / (k!)^2
    double sum = 1.0;
    double term = 1.0;
    double x2_4 = (x * x) / 4.0;

    for (int k = 1; k <= 25; ++k) {
        term *= x2_4 / static_cast<double>(k * k);
        sum += term;
        if (term < 1e-12 * sum) {
            break;
        }
    }
    return sum;
}

}  // namespace aurora::dsp
