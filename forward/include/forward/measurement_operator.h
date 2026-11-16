#ifndef AURORA_FORWARD_MEASUREMENT_OPERATOR_H_
#define AURORA_FORWARD_MEASUREMENT_OPERATOR_H_

/**
 * @file measurement_operator.h
 * @brief Measurement window shared by simulation and reconstruction.
 *
 * This module provides a single source of truth for K-space measurement
 * weights, ensuring consistency between simulation (forward measurement
 * generation) and reconstruction (inverse problem operators).
 *
 * ### Physical Model
 *
 * Each detector pixel measures a complex inner product in K-space:
 * @f[
 *   y_d = \int X(k) \cdot H_d(k - k_d) \, dk + \text{noise}
 * @f]
 *
 * Where:
 * - X(k): Scene complex spectrum for that range bin
 * - H_d(Δk): Acceptance window of pixel d, centered on its K-position k_d
 *   (detector geometry plus the LO tilt offset) with a width set by the
 *   pixel size through Fourier optics
 *
 * @see BandwidthModel for window function implementations
 * @see ForwardModel for K-sample retrieval
 * @see KSpaceGeometry for detector K-position calculations
 */

#include <complex>
#include <functional>

#include <Eigen/Core>

#include "bandwidth_config.pb.h"

namespace aurora::forward {

// Forward declarations
class ForwardModel;

/**
 * @brief Measurement window operator for simulation and reconstruction.
 *
 * Computes W_d(k) = H_d(k - k_d)
 *
 * All methods are static. This class cannot be instantiated.
 */
class MeasurementOperator {
  public:
    // Non-instantiable static utility class
    MeasurementOperator() = delete;
    MeasurementOperator(const MeasurementOperator&) = delete;
    MeasurementOperator& operator=(const MeasurementOperator&) = delete;

    /**
     * @brief Window parameters for one detector pixel at one wavelength.
     */
    struct WindowParams {
        Eigen::Vector2d k_center;  ///< Window center in scene K-space (rad/m)
        double bandwidth_x;        ///< Window bandwidth in kx (rad/m)
        double bandwidth_y;        ///< Window bandwidth in ky (rad/m)
    };

    /**
     * @brief Bounding box in K-space where W > threshold.
     */
    struct SupportBounds {
        double k_min_x;  ///< Minimum kx in support region
        double k_max_x;  ///< Maximum kx in support region
        double k_min_y;  ///< Minimum ky in support region
        double k_max_y;  ///< Maximum ky in support region

        /**
         * @brief Check if a K-vector is contained in the bounds.
         * @param k K-vector to test
         * @return true if k is within the bounding box
         */
        bool contains(const Eigen::Vector2d& k) const;

        /**
         * @brief Check if the support region is valid (non-empty).
         * @return true if k_min < k_max in both dimensions
         */
        bool is_valid() const;
    };

    // =========================================================================
    // WEIGHT COMPUTATION
    // =========================================================================

    /**
     * @brief Compute the window weight at query point k.
     *
     * W(k) = H(k - k_center), separable in kx and ky.
     *
     * @param k Query point in K-space (rad/m)
     * @param params Window parameters (center and bandwidths)
     * @param window_type Window function type (must be specified explicitly)
     * @return Weight with |W| ≤ 1 (can be negative for sinc sidelobes)
     */
    static double weight(const Eigen::Vector2d& k,
                         const WindowParams& params,
                         BandwidthWindowType window_type);

    /**
     * @brief Point target integrated response.
     *
     * A point target at spatial position r_target has K-space representation
     * X(k) = exp(i·k·r_target) for all k. The measurement integrates:
     *
     *   y = ∫ W(k) · exp(i·k·r_target) dk
     *
     * This is the Fourier transform of W evaluated at r_target.
     *
     * @param r_target Target spatial position (meters, in focal plane coords)
     * @param params Window parameters (center and bandwidths)
     * @param window_type Window function type (must be specified explicitly)
     * @param quadrature_order Points per axis for integration
     * @return Magnitude of integrated response |∫ W(k)·exp(i·k·r) dk|
     */
    static double point_target_weight(const Eigen::Vector2d& r_target,
                                      const WindowParams& params,
                                      BandwidthWindowType window_type,
                                      int quadrature_order);

    // =========================================================================
    // INTEGRATION FOR EXTENDED TARGETS
    // =========================================================================

    /**
     * @brief Integrate scene spectrum over the measurement window (general path).
     *
     * Used for plane/mesh targets where X(k) varies across the window.
     * Computes: y = ∫ X(k) · W(k) dk over the support region
     *
     * @warning X_k must return COMPLEX values, not magnitude.
     *          Integrating magnitude destroys phase information.
     *
     * @warning Do NOT normalize by ∫W — that would be forward normalization,
     *          which we explicitly avoid.
     *
     * @param X_k Scene spectrum function (complex-valued)
     * @param params Window parameters
     * @param window_type Window function type
     * @param quadrature_points Points per axis (use odd numbers):
     *        - 5×5 is TOO COARSE for sinc windows (misses sidelobes)
     *        - 9×9 minimum for sinc, 11×11 or 15×15 for high accuracy
     *        - Gaussian windows can use fewer points (5×5 often sufficient)
     * @return Complex measurement value
     */
    static std::complex<double> integrate_response(
        const std::function<std::complex<double>(const Eigen::Vector2d&)>& X_k,
        const WindowParams& params,
        BandwidthWindowType window_type,
        int quadrature_points);

    // =========================================================================
    // SUPPORT REGION COMPUTATION
    // =========================================================================

    /**
     * @brief Compute support region bounds for sparse weight computation.
     *
     * Returns the bounding box outside of which |W| falls below the
     * threshold. The threshold is applied per axis (using sqrt(threshold))
     * so that the separable product W = H_x · H_y exceeds the specified
     * threshold within the returned bounds.
     *
     * @param params Window parameters
     * @param threshold Minimum weight |W| to include in support
     * @param window_type Window function type
     * @return Bounding box
     */
    static SupportBounds compute_support(const WindowParams& params,
                                         double threshold,
                                         BandwidthWindowType window_type);

  private:
    /**
     * @brief Compute support factor for a given window type and threshold.
     *
     * Returns the number of bandwidth half-widths to include based on
     * the window type and threshold.
     *
     * @param window_type Window function type
     * @param threshold Minimum weight threshold
     * @return Multiplier for bandwidth to get half-support width
     */
    static double compute_support_factor(BandwidthWindowType window_type,
                                         double threshold);
};

/**
 * @brief Build WindowParams for a detector pixel from the ForwardModel.
 *
 * The window is centered on the pixel's scene K-position
 * (ForwardModel::get_effective_k) with the K-space bandwidth set by the
 * pixel size (ForwardModel::get_k_sample).
 *
 * @param forward_model ForwardModel for the receiver geometry
 * @param detector_idx Detector pixel index (0-based)
 * @param wavelength_m Operating wavelength in meters
 * @return WindowParams for the pixel
 *
 * @throws std::out_of_range if detector_idx >= num_detectors
 */
MeasurementOperator::WindowParams build_window_params(
    const ForwardModel& forward_model,
    uint32_t detector_idx,
    double wavelength_m);

}  // namespace aurora::forward

#endif  // AURORA_FORWARD_MEASUREMENT_OPERATOR_H_
