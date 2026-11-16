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
