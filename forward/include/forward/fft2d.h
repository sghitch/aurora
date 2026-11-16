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

#ifndef AURORA_FORWARD_FFT2D_H_
#define AURORA_FORWARD_FFT2D_H_

/**
 * @file fft2d.h
 * @brief 2D FFT wrapper for NUFFT operations.
 *
 * Provides a 2D FFT using KissFFT for the forward module's NUFFT implementation.
 * Uses separable 1D FFTs (row-wise then column-wise) for efficiency.
 */

#include <memory>

#include <Eigen/Core>

namespace aurora::forward {

/**
 * @brief 2D FFT wrapper using KissFFT.
 *
 * Implements 2D FFT via separable 1D transforms:
 * 1. Transform each row
 * 2. Transform each column of the result
 *
 * This is mathematically equivalent to a 2D DFT and is efficient
 * for the grid sizes used in NUFFT (typically 64-256).
 */
class Fft2d {
  public:
    /**
     * @brief Create 2D FFT plan for given dimensions.
     *
     * @param nx Number of columns (x dimension)
     * @param ny Number of rows (y dimension)
     * @param inverse If true, create inverse FFT plan
     * @throws std::invalid_argument if nx or ny is 0
     */
    Fft2d(int nx, int ny, bool inverse = false);

    /**
     * @brief Destructor.
     */
    ~Fft2d();

    // Non-copyable
    Fft2d(const Fft2d&) = delete;
    Fft2d& operator=(const Fft2d&) = delete;

    // Movable
    Fft2d(Fft2d&&) noexcept;
    Fft2d& operator=(Fft2d&&) noexcept;

    /**
     * @brief Execute 2D FFT.
     *
     * @param input Complex input matrix (ny rows, nx columns)
     * @param output Complex output matrix (will be resized if needed)
     * @throws std::invalid_argument if input dimensions don't match
     */
    void execute(const Eigen::MatrixXcd& input, Eigen::MatrixXcd& output) const;

    /**
     * @brief Execute 2D FFT and return result.
     *
     * @param input Complex input matrix (ny rows, nx columns)
     * @return Complex FFT result
     * @throws std::invalid_argument if input dimensions don't match
     */
    Eigen::MatrixXcd execute(const Eigen::MatrixXcd& input) const;

    /**
     * @brief Get x dimension (number of columns).
     */
    int nx() const { return nx_; }

    /**
     * @brief Get y dimension (number of rows).
     */
    int ny() const { return ny_; }

    /**
     * @brief Check if this is an inverse FFT.
     */
    bool is_inverse() const { return inverse_; }

  private:
    int nx_;
    int ny_;
    bool inverse_;

    // Pimpl for KissFFT state
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aurora::forward

#endif  // AURORA_FORWARD_FFT2D_H_
