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
