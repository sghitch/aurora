#ifndef AURORA_DETECTION_DOPPLER_EXTRACTOR_H_
#define AURORA_DETECTION_DOPPLER_EXTRACTOR_H_

/**
 * @file doppler_extractor.h
 * @brief Doppler velocity extraction from triangular chirp measurements.
 *
 * The DopplerExtractor extracts radial velocity from the phase difference
 * between up-chirp and down-chirp measurements. This uses the triangular
 * FMCW technique where opposite chirp directions see opposite Doppler shifts.
 *
 * ### Physical Model
 *
 * For a target with radial velocity v, the beat frequencies are:
 *
 *     f_beat_up   = (2BR/cT) + (2v/λ)
 *     f_beat_down = (2BR/cT) - (2v/λ)
 *
 * The phase accumulates over the chirp duration, giving:
 *
 *     Δφ = φ_up - φ_down = 4πvT/λ
 *     v = Δφ × λ / (4πT)
 *
 * ### Velocity Ambiguity
 *
 * Phase wraps at ±π, creating velocity ambiguity:
 *
 *     v_max_unambiguous = λ / (4 × T_chirp)
 *
 * For typical parameters (λ = 1550nm, T = 12.5μs):
 *     v_max ≈ 31 mm/s (0.031 m/s)
 *
 * For higher velocities, use longer chirp durations or multiple PRIs.
 *
 * @see AngleEstimator for position extraction
 */

#include <complex>
#include <vector>

#include "detection/detection_error.h"
#include "detection_config.pb.h"
#include "doppler_result.pb.h"

namespace aurora::detection {

/**
 * @brief Doppler velocity extraction from triangular chirp measurements.
 *
 * Extracts radial velocity by comparing phases between up-chirp and
 * down-chirp measurements. Provides velocity, confidence, and ambiguity
 * information.
 *
 * ### Fail-Fast Behavior
 *
 * The extractor throws DetectionError when:
 * - Input vectors are mismatched in size
 * - No valid measurements are provided
 * - Phase difference exceeds ambiguity threshold with low confidence
 *
 * ### Example Usage
 * @code
 * DetectionConfig config;
 * config.set_enable_doppler(true);
 * config.set_min_velocity_mps(0.001);  // 1 mm/s threshold
 *
 * DopplerExtractor extractor(config, 1550e-9, 12.5e-6);
 *
 * std::vector<std::complex<double>> up_samples = {...};
 * std::vector<std::complex<double>> down_samples = {...};
 * std::vector<double> weights = {...};
 *
 * try {
 *     auto result = extractor.extract_velocity(up_samples, down_samples, weights);
 *     double velocity = result.velocity_mps();
 *     if (result.ambiguous()) {
 *         // Handle velocity aliasing
 *     }
 * } catch (const DetectionError& e) {
 *     // Handle extraction failure
 * }
 * @endcode
 */
class DopplerExtractor {
  public:
    /**
     * @brief Construct analyzer with configuration and system parameters.
     *
     * @param config Detection configuration with Doppler settings
     * @param wavelength_m Wavelength in meters (e.g., 1550e-9)
     * @param chirp_duration_s Chirp duration in seconds (e.g., 12.5e-6)
     * @throws DetectionError if parameters are invalid
     */
    DopplerExtractor(const DetectionConfig& config,
                    double wavelength_m,
                    double chirp_duration_s);

    /**
     * @brief Extract velocity from up/down chirp phase difference.
     *
     * Computes weighted average of velocity estimates across all detector
     * pixels. Uses phase difference between up and down chirp samples.
     *
     * @param up_samples Complex samples from up-chirp measurements
     * @param down_samples Complex samples from down-chirp measurements
     * @param weights Weights for each measurement (typically from magnitude/SNR)
     * @return DopplerResult with velocity and quality metrics
     * @throws DetectionError if extraction fails
     */
    aurora::detection::DopplerResult extract_velocity(
        const std::vector<std::complex<double>>& up_samples,
        const std::vector<std::complex<double>>& down_samples,
        const std::vector<double>& weights) const;

    /**
     * @brief Extract velocity with uniform weights.
     *
     * Convenience method using uniform weights for all measurements.
     *
     * @param up_samples Complex samples from up-chirp measurements
     * @param down_samples Complex samples from down-chirp measurements
     * @return DopplerResult with velocity and quality metrics
     * @throws DetectionError if extraction fails
     */
    aurora::detection::DopplerResult extract_velocity(
        const std::vector<std::complex<double>>& up_samples,
        const std::vector<std::complex<double>>& down_samples) const;

    /**
     * @brief Compute phase difference between up and down samples.
     *
     * @param up_sample Complex up-chirp sample
     * @param down_sample Complex down-chirp sample
     * @return Phase difference in radians [-π, π]
     */
    double compute_phase_difference(
        const std::complex<double>& up_sample,
        const std::complex<double>& down_sample) const;

    /**
     * @brief Convert phase difference to velocity.
     *
     * Uses the relation: v = Δφ × λ / (4πT)
     *
     * @param phase_diff Phase difference in radians
     * @return Velocity in m/s (positive = approaching)
     */
    double phase_to_velocity(double phase_diff) const;

    /**
     * @brief Convert velocity to expected phase difference.
     *
     * Uses the relation: Δφ = 4πvT / λ
     *
     * @param velocity_mps Velocity in m/s
     * @return Expected phase difference in radians
     */
    double velocity_to_phase(double velocity_mps) const;

    /**
     * @brief Get maximum unambiguous velocity.
     *
     * Computed from wavelength and chirp duration:
     *     v_max = λ / (4 × T_chirp)
     *
     * @return Maximum unambiguous velocity in m/s
     */
    double max_unambiguous_velocity() const { return max_unambiguous_velocity_; }

    /**
     * @brief Get velocity resolution.
     *
     * Minimum detectable velocity change based on phase noise.
     *
     * @return Velocity resolution in m/s
     */
    double velocity_resolution() const { return velocity_resolution_; }

    /**
     * @brief Get wavelength.
     * @return Wavelength in meters
     */
    double wavelength_m() const { return wavelength_m_; }

    /**
     * @brief Get chirp duration.
     * @return Chirp duration in seconds
     */
    double chirp_duration_s() const { return chirp_duration_s_; }

    /**
     * @brief Check if Doppler extraction is enabled.
     * @return True if enabled in config
     */
    bool is_enabled() const { return enabled_; }

  private:
    DetectionConfig config_;
    double wavelength_m_;
    double chirp_duration_s_;
    double max_unambiguous_velocity_;
    double velocity_resolution_;
    double min_velocity_mps_;
    bool enabled_;

    /**
     * @brief Validate input vectors.
     *
     * @param up_samples Up-chirp samples
     * @param down_samples Down-chirp samples
     * @param weights Measurement weights
     * @throws DetectionError if validation fails
     */
    void validate_inputs(
        const std::vector<std::complex<double>>& up_samples,
        const std::vector<std::complex<double>>& down_samples,
        const std::vector<double>& weights) const;

    /**
     * @brief Compute weighted velocity estimate and statistics.
     *
     * @param phase_diffs Phase differences for each sample
     * @param weights Measurement weights
     * @param[out] velocity_out Weighted mean velocity
     * @param[out] std_dev_out Weighted standard deviation
     * @param[out] total_weight_out Sum of weights
     */
    void compute_weighted_velocity(
        const std::vector<double>& phase_diffs,
        const std::vector<double>& weights,
        double& velocity_out,
        double& std_dev_out,
        double& total_weight_out) const;

    /**
     * @brief Check if velocity is likely ambiguous (aliased).
     *
     * Examines phase difference variance to detect aliasing.
     *
     * @param phase_diffs Phase differences for each sample
     * @param mean_velocity Computed mean velocity
     * @return True if velocity may be aliased
     */
    bool check_ambiguity(
        const std::vector<double>& phase_diffs,
        double mean_velocity) const;

    /**
     * @brief Compute confidence score for velocity estimate.
     *
     * Based on phase consistency across detectors and SNR.
     *
     * @param std_dev Standard deviation of velocity estimates
     * @param total_weight Sum of measurement weights
     * @param num_samples Number of samples
     * @return Confidence score [0, 1]
     */
    double compute_confidence(
        double std_dev,
        double total_weight,
        size_t num_samples) const;
};

}  // namespace aurora::detection

#endif  // AURORA_DETECTION_DOPPLER_EXTRACTOR_H_
