#ifndef AURORA_DETECTION_CFAR_DETECTOR_H_
#define AURORA_DETECTION_CFAR_DETECTOR_H_

/**
 * @file cfar_detector.h
 * @brief CFAR (Constant False Alarm Rate) adaptive threshold detector.
 *
 * Implements CFAR-style adaptive thresholding for range bin detection.
 * CFAR estimates the local noise floor around each cell under test and
 * sets a detection threshold relative to that estimate, maintaining
 * a constant false alarm rate regardless of absolute signal levels.
 *
 * ### Supported CFAR Modes
 *
 * - **CA-CFAR (Cell-Averaging)**: Average of all reference cells
 * - **GO-CFAR (Greatest-Of)**: Maximum of leading/lagging averages
 * - **SO-CFAR (Smallest-Of)**: Minimum of leading/lagging averages
 * - **OS-CFAR (Order-Statistic)**: K-th largest value in reference cells
 *
 * ### Window Structure
 *
 * ```
 * |--ref--|--guard--|  CUT  |--guard--|--ref--|
 *    N_r      N_g       ^        N_g      N_r
 *                       |
 *                Cell Under Test
 * ```
 */

#include <cstdint>
#include <span>
#include <vector>

#include "detection/detection_error.h"
#include "detection_config.pb.h"

namespace aurora::detection {

// CfarDetection is defined in detection_config.pb.h

/**
 * @brief CFAR-style adaptive threshold detector.
 *
 * Provides constant false alarm rate detection by estimating the local
 * noise floor around each cell under test and setting detection thresholds
 * relative to that estimate.
 *
 * ### Example Usage
 * @code
 * CfarConfig config;
 * config.set_mode(CFAR_MODE_CA);
 * config.set_guard_cells(3);
 * config.set_reference_cells(12);
 * config.set_threshold_factor(4.0);
 * config.set_min_threshold(1e-6);
 *
 * CfarDetector cfar(config);
 *
 * // Compute magnitude profile from measurement tensor
 * std::vector<double> magnitudes = compute_magnitudes(measurements);
 *
 * // Run CFAR detection
 * auto detections = cfar.detect_1d(magnitudes);
 *
 * for (const auto& det : detections) {
 *     std::cout << "Detection at bin " << det.bin_index
 *               << " with SNR " << det.snr_db << " dB\n";
 * }
 * @endcode
 */
class CfarDetector {
  public:
    /**
     * @brief Construct CFAR detector with configuration.
     *
     * Validates that all required fields are set for the selected CFAR mode.
     * For modes other than FIXED/UNSPECIFIED, requires:
     * - guard_cells > 0
     * - reference_cells > 0
     * - threshold_factor > 0
     * - min_threshold > 0
     * - os_rank > 0 (for OS-CFAR mode)
     *
     * @param config CFAR configuration with mode and parameters
     * @throws DetectionError if required fields are not set
     */
    explicit CfarDetector(const CfarConfig& config);

    /**
     * @brief Detect peaks in a 1D range profile.
     *
     * Applies CFAR thresholding to each cell in the magnitude array.
     * Returns detections where magnitude exceeds the adaptive threshold.
     *
     * @param magnitudes Pre-computed magnitude of each range bin (linear scale)
     * @return Vector of detections, sorted by bin index
     */
    std::vector<CfarDetection> detect_1d(
        std::span<const double> magnitudes) const;

    /**
     * @brief Get current configuration.
     * @return Current CFAR configuration
     */
    const CfarConfig& config() const { return config_; }

    /**
     * @brief Check if CFAR is enabled (mode != FIXED and != UNSPECIFIED).
     * @return True if adaptive CFAR thresholding is active
     */
    bool is_enabled() const;

  private:
    CfarConfig config_;

    /**
     * @brief Compute noise estimate for cell at index using CA-CFAR.
     *
     * Average of all reference cells (leading + lagging).
     *
     * @param magnitudes Magnitude array
     * @param cell_index Index of cell under test
     * @return Noise estimate
     */
    double estimate_noise_ca(
        std::span<const double> magnitudes,
        uint32_t cell_index) const;

    /**
     * @brief Compute noise estimate using GO-CFAR.
     *
     * Maximum of leading and lagging cell averages.
     * Better for multiple targets in close proximity.
     *
     * @param magnitudes Magnitude array
     * @param cell_index Index of cell under test
     * @return Noise estimate
     */
    double estimate_noise_go(
        std::span<const double> magnitudes,
        uint32_t cell_index) const;

    /**
     * @brief Compute noise estimate using SO-CFAR.
     *
     * Minimum of leading and lagging cell averages.
     * Better for targets near clutter edges.
     *
     * @param magnitudes Magnitude array
     * @param cell_index Index of cell under test
     * @return Noise estimate
     */
    double estimate_noise_so(
        std::span<const double> magnitudes,
        uint32_t cell_index) const;

    /**
     * @brief Compute noise estimate using OS-CFAR.
     *
     * Uses the K-th largest value from reference cells.
     * Robust to outliers (interfering targets).
     *
     * @param magnitudes Magnitude array
     * @param cell_index Index of cell under test
     * @return Noise estimate
     */
    double estimate_noise_os(
        std::span<const double> magnitudes,
        uint32_t cell_index) const;

    /**
     * @brief Handle edge cases where full reference window isn't available.
     *
     * Uses reduced window at array boundaries.
     *
     * @param magnitudes Magnitude array
     * @param cell_index Index of cell under test
     * @return Noise estimate with reduced window
     */
    double estimate_noise_edge(
        std::span<const double> magnitudes,
        uint32_t cell_index) const;

    /**
     * @brief Compute threshold for a given noise estimate.
     *
     * @param noise_estimate Local noise estimate
     * @return Detection threshold (max of CFAR threshold and min_threshold)
     */
    double compute_threshold(double noise_estimate) const;
};

}  // namespace aurora::detection

#endif  // AURORA_DETECTION_CFAR_DETECTOR_H_
