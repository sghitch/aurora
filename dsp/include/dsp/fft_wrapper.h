#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace aurora::dsp {

/// Wrapper around KissFFT providing a clean C++ interface with pre-allocated
/// plans for real-time operation.
///
/// Key features:
/// - Pre-allocated FFT plans created at construction time
/// - Thread-safe execution (same plan can be used from multiple threads)
/// - In-place operation to minimize memory allocation
/// - Optimized real-to-complex FFT for real-valued input
///
/// Note: KissFFT is used instead of FFTW for BSD-3 licensing compatibility.
class FftWrapper {
public:
    /// Create FFT plan for given size.
    /// @param size FFT size (should be power of 2 for best performance)
    /// @param inverse If true, create inverse FFT plan
    /// @throws std::invalid_argument if size is 0
    explicit FftWrapper(uint32_t size, bool inverse = false);

    /// Destructor.
    ~FftWrapper();

    // Non-copyable
    FftWrapper(const FftWrapper&) = delete;
    FftWrapper& operator=(const FftWrapper&) = delete;

    // Movable
    FftWrapper(FftWrapper&&) noexcept;
    FftWrapper& operator=(FftWrapper&&) noexcept;

    /// Execute FFT on input data.
    /// @param input Complex input data (must have size() elements)
    /// @return Complex FFT result
    /// @throws std::invalid_argument if input size doesn't match FFT size
    std::vector<std::complex<double>> execute(
        const std::vector<std::complex<double>>& input) const;

    /// Execute FFT in-place.
    /// @param data Complex data to transform in-place (must have size() elements)
    /// @throws std::invalid_argument if data size doesn't match FFT size
    void execute_inplace(std::vector<std::complex<double>>& data) const;

    /// Execute real-to-complex FFT for real input.
    /// Uses optimized kiss_fftr for real-valued signals.
    /// @param input Real input data (must have size() elements)
    /// @return Complex FFT result (size()/2 + 1 elements due to conjugate symmetry)
    /// @throws std::invalid_argument if input size doesn't match FFT size
    std::vector<std::complex<double>> execute_real(
        const std::vector<double>& input) const;

    /// Get FFT size.
    uint32_t size() const { return size_; }

    /// Check if this is an inverse FFT.
    bool is_inverse() const { return inverse_; }

private:
    uint32_t size_;
    bool inverse_;

    // Pimpl for KissFFT state (avoids exposing KissFFT headers)
    struct KissState;
    std::unique_ptr<KissState> state_;
};

}  // namespace aurora::dsp
