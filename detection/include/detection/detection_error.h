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

#ifndef AURORA_DETECTION_DETECTION_ERROR_H_
#define AURORA_DETECTION_DETECTION_ERROR_H_

/**
 * @file detection_error.h
 * @brief Exception types for the detection module.
 *
 * Provides fail-fast error handling for the detection pipeline.
 * Exceptions are thrown when:
 * - Input data is invalid or insufficient
 * - Phase gradient fitting fails
 * - Configuration is invalid
 */

#include <stdexcept>
#include <string>

namespace aurora::detection {

/**
 * @brief Exception thrown when detection fails.
 *
 * Encapsulates error codes for different failure modes in the
 * detection pipeline. Used to implement fail-fast behavior when
 * conditions prevent reliable detection.
 *
 * ### Error Codes
 * - INSUFFICIENT_SIGNAL: SNR too low for reliable detection
 * - PHASE_GRADIENT_FAILED: Phase fitting failed (ill-conditioned, high residual)
 * - INVALID_MEASUREMENT: Input data is malformed or inconsistent
 * - CONFIGURATION_ERROR: Configuration parameters are invalid
 *
 * ### Example Usage
 * @code
 * try {
 *     auto result = fitter.fit(phases, k_vectors, weights);
 * } catch (const DetectionError& e) {
 *     if (e.code() == DetectionError::Code::INSUFFICIENT_SIGNAL) {
 *         // Handle low SNR case
 *     }
 *     spdlog::warn("Detection failed: {}", e.what());
 * }
 * @endcode
 */
class DetectionError : public std::runtime_error {
  public:
    /**
     * @brief Error codes for different failure modes.
     */
    enum class Code {
        INSUFFICIENT_SIGNAL,    ///< SNR too low for reliable detection
        PHASE_GRADIENT_FAILED,  ///< Phase gradient fitting failed
        INVALID_MEASUREMENT,    ///< Input data is invalid or inconsistent
        CONFIGURATION_ERROR     ///< Configuration parameters are invalid
    };

    /**
     * @brief Construct a DetectionError with code and message.
     *
     * @param code The error code indicating the failure type
     * @param message Human-readable description of the error
     */
    DetectionError(Code code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    /**
     * @brief Get the error code.
     * @return The error code for this exception
     */
    Code code() const { return code_; }

  private:
    Code code_;
};

}  // namespace aurora::detection

#endif  // AURORA_DETECTION_DETECTION_ERROR_H_
