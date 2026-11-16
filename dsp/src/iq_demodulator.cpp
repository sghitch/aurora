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

#include <cmath>
#include <stdexcept>

namespace aurora::dsp {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
}  // namespace

IQDemodulator::IQDemodulator(const Config& config) : config_(config) {
    if (config_.sample_rate_hz <= 0) {
        throw std::invalid_argument("sample_rate_hz must be positive");
    }
    if (config_.reference_freq_hz < 0) {
        throw std::invalid_argument("reference_freq_hz must be non-negative");
    }
    if (config_.lpf_cutoff_hz <= 0) {
        throw std::invalid_argument("lpf_cutoff_hz must be positive");
    }
    if (config_.lpf_cutoff_hz >= config_.sample_rate_hz / 2) {
        throw std::invalid_argument(
            "lpf_cutoff_hz must be less than Nyquist frequency");
    }
    if (config_.lpf_order == 0 || config_.lpf_order > 20) {
        throw std::invalid_argument("lpf_order must be between 1 and 20");
    }

    design_butterworth_lpf();
}

void IQDemodulator::design_butterworth_lpf() {
    // Design Butterworth low-pass filter using bilinear transform.
    // We use second-order sections (SOS) for numerical stability.

    const uint32_t order = config_.lpf_order;
    const double fs = config_.sample_rate_hz;
    const double fc = config_.lpf_cutoff_hz;

    // Pre-warp the cutoff frequency for bilinear transform
    const double wc = kTwoPi * fc / fs;
    const double wc_warped = 2.0 * fs * std::tan(wc / 2.0);

    // Number of second-order sections needed
    const uint32_t num_sections = (order + 1) / 2;
    sos_sections_.resize(num_sections);

    // Design each second-order section
    for (uint32_t k = 0; k < num_sections; ++k) {
        // Analog prototype pole angle for Butterworth
        // For order N, poles are at angles: pi/2 + (2k+1)*pi/(2N) for k=0..N-1
        // For SOS, we pair conjugate poles

        double theta1, theta2;
        bool is_first_order_section = false;

        if (order % 2 == 1 && k == num_sections - 1) {
            // Odd order: last section is first-order (single real pole)
            is_first_order_section = true;
            theta1 = kPi;  // Real pole at -1 (normalized)
        } else {
            // Second-order section with conjugate pole pair
            uint32_t pole_idx = k;
            if (order % 2 == 1) {
                // Odd order: skip the real pole index
            }
            theta1 = kPi / 2.0 + (2.0 * pole_idx + 1.0) * kPi / (2.0 * order);
            theta2 = kPi / 2.0 + (2.0 * (order - 1 - pole_idx) + 1.0) * kPi /
                                    (2.0 * order);
        }

        // Analog prototype poles (on unit circle in s-plane, scaled by wc_warped)
        // s = wc_warped * exp(j*theta)

        if (is_first_order_section) {
            // First-order section: H(s) = wc / (s + wc)
            // Bilinear transform: s = 2*fs*(z-1)/(z+1)
            // H(z) = wc*(z+1) / (2*fs*(z-1) + wc*(z+1))
            //      = wc*(z+1) / ((2*fs+wc)*z + (wc-2*fs))

            double K = wc_warped / (2.0 * fs);
            double b0 = K / (1.0 + K);
            double b1 = b0;
            double a1 = (K - 1.0) / (1.0 + K);

            // Pack as second-order with b2=0, a2=0
            sos_sections_[k] = {b0, b1, 0.0, a1, 0.0};
        } else {
            // Second-order section from conjugate pole pair
            // Analog: H(s) = wc^2 / (s^2 - 2*Re(p)*wc*s + wc^2)
            // where p = exp(j*theta) is the normalized pole

            double cos_theta = std::cos(theta1);
            double alpha = -2.0 * cos_theta * wc_warped;
            double beta = wc_warped * wc_warped;

            // Bilinear transform coefficients
            // Using s = 2*fs*(z-1)/(z+1)
            // Denominator: (2*fs)^2*(z-1)^2 + alpha*2*fs*(z-1)*(z+1) + beta*(z+1)^2

            double fs2 = fs * fs;
            double c0 = 4.0 * fs2 + 2.0 * alpha * fs + beta;
            double c1 = 2.0 * beta - 8.0 * fs2;
            double c2 = 4.0 * fs2 - 2.0 * alpha * fs + beta;

            // Numerator: beta*(z+1)^2 = beta*(z^2 + 2*z + 1)
            double b0 = beta / c0;
            double b1 = 2.0 * beta / c0;
            double b2 = beta / c0;
            double a1 = c1 / c0;
            double a2 = c2 / c0;

            sos_sections_[k] = {b0, b1, b2, a1, a2};
        }
    }
}

std::vector<double> IQDemodulator::apply_biquad(
    const std::vector<double>& signal,
    const std::array<double, 5>& section) const {

    const double b0 = section[0];
    const double b1 = section[1];
    const double b2 = section[2];
    const double a1 = section[3];
    const double a2 = section[4];

    std::vector<double> output(signal.size());

    // Direct Form II transposed implementation
    double w1 = 0.0, w2 = 0.0;

    for (size_t i = 0; i < signal.size(); ++i) {
        double x = signal[i];
        double y = b0 * x + w1;
        w1 = b1 * x - a1 * y + w2;
        w2 = b2 * x - a2 * y;
        output[i] = y;
    }

    return output;
}

std::vector<double> IQDemodulator::apply_lpf(
    const std::vector<double>& signal) const {

    std::vector<double> result = signal;

    // Apply each second-order section in cascade
    for (const auto& section : sos_sections_) {
        result = apply_biquad(result, section);
    }

    return result;
}

std::vector<std::complex<double>> IQDemodulator::demodulate(
    const std::vector<double>& real_signal) const {
    return demodulate(real_signal, config_.reference_freq_hz);
}

std::vector<std::complex<double>> IQDemodulator::demodulate(
    const std::vector<double>& real_signal, double reference_freq_hz) const {

    if (real_signal.empty()) {
        return {};
    }

    const size_t n = real_signal.size();
    const double dt = 1.0 / config_.sample_rate_hz;

    // Generate quadrature references and multiply in one pass
    std::vector<double> i_raw(n), q_raw(n);

    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * dt;
        const double phase = kTwoPi * reference_freq_hz * t;
        const double cos_ref = std::cos(phase);
        const double sin_ref = std::sin(phase);

        i_raw[i] = real_signal[i] * cos_ref;
        q_raw[i] = real_signal[i] * sin_ref;
    }

    // Low-pass filter to remove sum frequency
    auto i_filt = apply_lpf(i_raw);
    auto q_filt = apply_lpf(q_raw);

    // Form complex output (scale by 2 to recover original amplitude)
    std::vector<std::complex<double>> output(n);
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::complex<double>(2.0 * i_filt[i], 2.0 * q_filt[i]);
    }

    return output;
}

std::pair<std::vector<double>, std::vector<double>>
IQDemodulator::get_iq_components(const std::vector<double>& real_signal) const {

    if (real_signal.empty()) {
        return {{}, {}};
    }

    const size_t n = real_signal.size();
    const double dt = 1.0 / config_.sample_rate_hz;

    std::vector<double> i_raw(n), q_raw(n);

    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) * dt;
        const double phase = kTwoPi * config_.reference_freq_hz * t;
        i_raw[i] = real_signal[i] * std::cos(phase);
        q_raw[i] = real_signal[i] * std::sin(phase);
    }

    auto i_filt = apply_lpf(i_raw);
    auto q_filt = apply_lpf(q_raw);

    // Scale by 2 to recover original amplitude
    for (size_t i = 0; i < n; ++i) {
        i_filt[i] *= 2.0;
        q_filt[i] *= 2.0;
    }

    return {std::move(i_filt), std::move(q_filt)};
}

}  // namespace aurora::dsp
