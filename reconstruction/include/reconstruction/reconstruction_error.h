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

#ifndef AURORA_RECONSTRUCTION_RECONSTRUCTION_ERROR_H_
#define AURORA_RECONSTRUCTION_RECONSTRUCTION_ERROR_H_

/**
 * @file reconstruction_error.h
 * @brief Exception types for the reconstruction module.
 *
 * Provides fail-fast error handling for the reconstruction pipeline.
 * Exceptions are thrown when:
 * - Input data is invalid or insufficient
 * - ADMM solver fails to converge
 * - Configuration is invalid
 */

#include <stdexcept>
#include <string>

namespace aurora::reconstruction {

/**
 * @brief Exception thrown when reconstruction fails.
 *
 * Encapsulates error codes for different failure modes in the
 * reconstruction pipeline. Used to implement fail-fast behavior when
 * conditions prevent reliable reconstruction.
 *
 * ### Error Codes
 * - CONVERGENCE_FAILED: ADMM solver did not converge within max iterations
 * - INVALID_INPUT: Input data is malformed or inconsistent
 * - CONFIGURATION_ERROR: Configuration parameters are invalid
 * - NUMERICAL_ERROR: Numerical instability (NaN/Inf) encountered
 */
class ReconstructionError : public std::runtime_error {
  public:
    /**
     * @brief Error codes for different failure modes.
     */
    enum class Code {
        CONVERGENCE_FAILED,   ///< ADMM solver did not converge
        INVALID_INPUT,        ///< Input data is invalid or inconsistent
        CONFIGURATION_ERROR,  ///< Configuration parameters are invalid
        NUMERICAL_ERROR       ///< Numerical instability encountered
    };

    /**
     * @brief Construct a ReconstructionError with code and message.
     *
     * @param code The error code indicating the failure type
     * @param message Human-readable description of the error
     */
    ReconstructionError(Code code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    /**
     * @brief Get the error code.
     * @return The error code for this exception
     */
    Code code() const { return code_; }

  private:
    Code code_;
};

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_RECONSTRUCTION_ERROR_H_
