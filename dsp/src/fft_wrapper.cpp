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

#include <kiss_fft.h>
#include <kiss_fftr.h>

#include <stdexcept>

namespace aurora::dsp {

struct FftWrapper::KissState {
    kiss_fft_cfg complex_plan = nullptr;
    kiss_fftr_cfg real_plan = nullptr;

    ~KissState() {
        if (complex_plan) {
            kiss_fft_free(complex_plan);
        }
        if (real_plan) {
            kiss_fft_free(real_plan);
        }
    }
};

FftWrapper::FftWrapper(uint32_t size, bool inverse)
    : size_(size), inverse_(inverse), state_(std::make_unique<KissState>()) {
    if (size == 0) {
        throw std::invalid_argument("FFT size must be greater than 0");
    }

    // Create complex FFT plan
    // kiss_fft_alloc returns nullptr on failure
    state_->complex_plan = kiss_fft_alloc(
        static_cast<int>(size_),
        inverse_ ? 1 : 0,  // inverse flag
        nullptr,           // use malloc
        nullptr);          // lenmem (unused when mem is nullptr)

    if (!state_->complex_plan) {
        throw std::runtime_error("Failed to allocate KissFFT complex plan");
    }

    // Create real FFT plan (only for forward transform)
    // Real FFT requires even size
    if (!inverse_ && (size_ % 2 == 0)) {
        state_->real_plan = kiss_fftr_alloc(
            static_cast<int>(size_),
            0,        // forward transform
            nullptr,  // use malloc
            nullptr); // lenmem

        if (!state_->real_plan) {
            throw std::runtime_error("Failed to allocate KissFFT real plan");
        }
    }
}

FftWrapper::~FftWrapper() = default;

FftWrapper::FftWrapper(FftWrapper&&) noexcept = default;

FftWrapper& FftWrapper::operator=(FftWrapper&&) noexcept = default;

std::vector<std::complex<double>> FftWrapper::execute(
    const std::vector<std::complex<double>>& input) const {
    if (input.size() != size_) {
        throw std::invalid_argument(
            "Input size (" + std::to_string(input.size()) +
            ") does not match FFT size (" + std::to_string(size_) + ")");
    }

    // Allocate output buffer
    std::vector<std::complex<double>> output(size_);

    // KissFFT uses kiss_fft_cpx which is {double, double} for real/imag
    // std::complex<double> has the same memory layout
    static_assert(sizeof(kiss_fft_cpx) == sizeof(std::complex<double>),
                  "kiss_fft_cpx and std::complex<double> must have same size");
    static_assert(alignof(kiss_fft_cpx) <= alignof(std::complex<double>),
                  "kiss_fft_cpx alignment must not exceed std::complex<double>");

    kiss_fft(state_->complex_plan,
             reinterpret_cast<const kiss_fft_cpx*>(input.data()),
             reinterpret_cast<kiss_fft_cpx*>(output.data()));

    return output;
}

void FftWrapper::execute_inplace(std::vector<std::complex<double>>& data) const {
    if (data.size() != size_) {
        throw std::invalid_argument(
            "Data size (" + std::to_string(data.size()) +
            ") does not match FFT size (" + std::to_string(size_) + ")");
    }

    // KissFFT does not support true in-place operation, so we need a temp buffer
    std::vector<std::complex<double>> temp(size_);

    kiss_fft(state_->complex_plan,
             reinterpret_cast<const kiss_fft_cpx*>(data.data()),
             reinterpret_cast<kiss_fft_cpx*>(temp.data()));

    data = std::move(temp);
}

std::vector<std::complex<double>> FftWrapper::execute_real(
    const std::vector<double>& input) const {
    if (input.size() != size_) {
        throw std::invalid_argument(
            "Input size (" + std::to_string(input.size()) +
            ") does not match FFT size (" + std::to_string(size_) + ")");
    }

    if (inverse_) {
        throw std::logic_error(
            "execute_real() cannot be called on an inverse FFT wrapper");
    }

    if (!state_->real_plan) {
        throw std::logic_error(
            "Real FFT plan not available (size must be even)");
    }

    // Real FFT output has size/2 + 1 complex values due to conjugate symmetry
    const size_t output_size = size_ / 2 + 1;
    std::vector<std::complex<double>> output(output_size);

    kiss_fftr(state_->real_plan,
              input.data(),
              reinterpret_cast<kiss_fft_cpx*>(output.data()));

    return output;
}

}  // namespace aurora::dsp
