#ifndef AURORA_FORWARD_BANDWIDTH_MODEL_H_
#define AURORA_FORWARD_BANDWIDTH_MODEL_H_

/**
 * @file bandwidth_model.h
 * @brief Unified bandwidth model for simulation and reconstruction.
 *
 * This module provides the physics for K-space bandwidth effects:
 * - Each detector pixel integrates over its spatial aperture
 * - The finite aperture gives the pixel a finite K-space window (bandwidth)
 *
 * Both simulation (forward) and reconstruction (inverse) use this model
 * to ensure consistency.
 *
 * ### Physical Model
 *
 * In K-space:
 *   S(k) = sinc(k_x / Δk_x) · sinc(k_y / Δk_y)
 *
 * Where Δk is the K-space bandwidth set by the pixel aperture.
 *
 * In spatial domain (Fourier dual):
 *   S(r) = rect(r_x · Δk_x / 2π) ⊛ rect(r_y · Δk_y / 2π)
 *
 * The rect convolution produces a trapezoid sensitivity profile.
 */

#include <Eigen/Core>
#include "bandwidth_config.pb.h"

namespace aurora::forward {

/**
 * @brief Static utility class for bandwidth physics computations.
 *
 * All methods are static. This class cannot be instantiated.
 */
class BandwidthModel {
  public:
    // Non-instantiable static utility class
    BandwidthModel() = delete;
    BandwidthModel(const BandwidthModel&) = delete;
    BandwidthModel& operator=(const BandwidthModel&) = delete;

    // =========================================================================
    // K-SPACE DOMAIN (for reconstruction)
    // =========================================================================

    /**
     * @brief Compute K-space weight from 2D window function.
     *
     * Returns the combined window value based on the specified window type:
     * - SINC: sinc(Δk_x / bw_x) · sinc(Δk_y / bw_y)
     * - GAUSSIAN: exp(-0.5 · (Δk_x/bw_x)²) · exp(-0.5 · (Δk_y/bw_y)²)
     * - KAISER: kaiser(Δk_x, bw_x) · kaiser(Δk_y, bw_y)
     *
     * @param delta_k K-space offset from sample center (kx, ky) in rad/m
     * @param k_bandwidth_x K-space bandwidth in x direction (rad/m)
     * @param k_bandwidth_y K-space bandwidth in y direction (rad/m)
     * @param window_type Type of window function to use
     * @return Weight factor (typically in [0, 1], may be negative for SINC)
     */
    static double k_space_weight(const Eigen::Vector2d& delta_k,
                                 double k_bandwidth_x,
                                 double k_bandwidth_y,
                                 BandwidthWindowType window_type);

    /**
     * @brief Compute 1D window function value.
     *
     * @param delta_k K-space offset from center (rad/m)
     * @param bandwidth K-space bandwidth (rad/m)
     * @param window_type Type of window function to use
     * @return Window value
     * @throws std::invalid_argument if window_type is UNSPECIFIED or unknown
     */
    static double window_function(double delta_k,
                                  double bandwidth,
                                  BandwidthWindowType window_type);

    /**
     * @brief Compute 1D sinc window value.
     *
     * Returns sinc(Δk / bandwidth) = sin(π·Δk/bw) / (π·Δk/bw)
     *
     * @param delta_k K-space offset from center (rad/m)
     * @param bandwidth K-space bandwidth (rad/m)
     * @return Window value in [-1, 1]
     */
    static double sinc_window(double delta_k, double bandwidth);

    /**
     * @brief Compute 1D Gaussian window value.
     *
     * Returns exp(-0.5 · (Δk / bandwidth)²)
     *
     * @param delta_k K-space offset from center (rad/m)
     * @param bandwidth K-space bandwidth (rad/m)
     * @return Window value in [0, 1]
     */
    static double gaussian_window(double delta_k, double bandwidth);

    /**
     * @brief Compute 1D Kaiser window value.
     *
     * Returns I₀(β·√(1 - (Δk/bandwidth)²)) / I₀(β) for |Δk| ≤ bandwidth,
     * 0 otherwise. Uses β = π (standard NUFFT gridding value).
     *
     * @param delta_k K-space offset from center (rad/m)
     * @param bandwidth K-space bandwidth (rad/m)
     * @return Window value in [0, 1]
     */
    static double kaiser_window(double delta_k, double bandwidth);

    // =========================================================================
    // SPATIAL DOMAIN (for simulation)
    // =========================================================================

    /**
     * @brief Compute 1D rect convolution (trapezoid function).
     *
     * Convolution of two rect functions:
     * - Outside support [(w1+w2)/2]: returns 0
     * - Inside flat region [(w_max-w_min)/2]: returns 1
     * - Sloped region: linear interpolation
     *
     * @param x Position offset from center
     * @param width1 First rect width
     * @param width2 Second rect width
     * @return Convolution value in [0, 1]
     */
    static double rect_convolution(double x, double width1, double width2);

    // =========================================================================
    // CONVERSIONS
    // =========================================================================

    /**
     * @brief Convert K-space bandwidth to spatial width.
     *
     * Fourier relationship: spatial_width = 2π / k_bandwidth
     *
     * @param k_bandwidth K-space bandwidth (rad/m)
     * @return Spatial width (meters)
     */
    static double k_bandwidth_to_spatial_width(double k_bandwidth);

    /**
     * @brief Convert spatial width to K-space bandwidth.
     *
     * Fourier relationship: k_bandwidth = 2π / spatial_width
     *
     * @param spatial_width Spatial width (meters)
     * @return K-space bandwidth (rad/m)
     */
    static double spatial_width_to_k_bandwidth(double spatial_width);

    /**
     * @brief Convert K-space bandwidth to Fourier plane width.
     *
     * Maps K-bandwidth to physical width at Fourier plane:
     *   fourier_width = k_bandwidth · f · λ / (2π)
     *
     * @param k_bandwidth K-space bandwidth (rad/m)
     * @param wavelength_m Operating wavelength (meters)
     * @param focal_length_m Fourier lens focal length (meters)
     * @return Fourier plane width (meters)
     */
    static double k_bandwidth_to_fourier_width(double k_bandwidth,
                                                double wavelength_m,
                                                double focal_length_m);

    // =========================================================================
    // RESOLUTION LIMITS
    // =========================================================================

    /**
     * @brief Compute resolution limit from K-space bandwidth.
     *
     * Minimum resolvable spatial separation: Δr = 2π / k_bandwidth
     *
     * @param k_bandwidth_x K-space bandwidth in x (rad/m)
     * @param k_bandwidth_y K-space bandwidth in y (rad/m)
     * @return Resolution limit (dx, dy) in meters
     */
    static Eigen::Vector2d resolution_limit(double k_bandwidth_x,
                                            double k_bandwidth_y);

    /**
     * @brief Check if grid spacing is valid for given bandwidth.
     *
     * Grid spacing must be >= resolution limit for valid sampling:
     *   grid_spacing >= tolerance_factor × (2π / k_bandwidth)
     *
     * @param grid_spacing Reconstruction grid spacing (meters)
     * @param k_bandwidth K-space bandwidth (rad/m)
     * @param tolerance_factor Safety factor (default 0.9)
     * @return true if spacing is valid, false if undersampled
     */
    static bool is_spacing_valid(double grid_spacing,
                                 double k_bandwidth,
                                 double tolerance_factor = 0.9);

  private:
    /**
     * @brief Normalized sinc function: sinc(x) = sin(πx) / (πx), sinc(0) = 1
     * @param x Input value
     * @return sinc(x)
     */
    static double sinc(double x);

    /**
     * @brief Modified Bessel function of the first kind, order 0.
     * @param x Input value
     * @return I₀(x)
     */
    static double bessel_i0(double x);
};

}  // namespace aurora::forward

#endif  // AURORA_FORWARD_BANDWIDTH_MODEL_H_
