#ifndef AURORA_FORWARD_K_SPACE_GEOMETRY_H_
#define AURORA_FORWARD_K_SPACE_GEOMETRY_H_

/**
 * @file k_space_geometry.h
 * @brief K-space coordinate geometry for the coherent receiver.
 *
 * This module answers one question for the rest of the system: which
 * K-vector does each detector pixel sample, and how wide is its acceptance
 * window? Both follow from Fourier optics given the pixel geometry, the lens
 * focal length and the local-oscillator tilt.
 *
 * @see ForwardModel for the effective K-sample lookup
 */

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "detector_geometry.pb.h"
#include "forward/bandwidth_model.h"
#include "optics_config.pb.h"

namespace aurora::forward {

/**
 * @brief K-space geometry of the detector array.
 *
 * Provides K-space coordinate information for the coherent LiDAR system:
 * - Detector K-positions and acceptance bandwidths from Fourier optics
 * - LO tilt K-offset
 *
 * ### Physical Model
 *
 * Detector K-position from Fourier optics:
 * @f[
 *   k_{det} = \frac{2\pi \cdot (x_{det}, y_{det})}{f \cdot \lambda}
 * @f]
 *
 * LO tilt K-offset:
 * @f[
 *   k_{LO} = \frac{2\pi \cdot (\sin(\theta_x), \sin(\theta_y))}{\lambda}
 * @f]
 *
 * ### Example Usage
 * @code
 * OpticsConfig optics;
 * optics.set_fourier_lens_focal_m(0.05);
 *
 * DetectorArray detector = create_linear_array("test", 8, 0.5e-3, {1550e-9});
 *
 * KSpaceGeometry geometry(optics, detector);
 *
 * // K-positions and acceptance bandwidths for all pixels at 1550nm
 * auto k_positions = geometry.all_detector_k(1550e-9);
 * double bw_x = geometry.detector_k_bandwidth_x(0, 1550e-9);
 * @endcode
 */
class KSpaceGeometry {
  public:
    /**
     * @brief Construct K-space geometry from optical system components.
     *
     * @param optics Optical configuration (focal length, LO tilt)
     * @param detector Detector array with pixel positions
     */
    KSpaceGeometry(const OpticsConfig& optics,
                   const DetectorArray& detector);

    // =========================================================================
    // Detector K-positions
    // =========================================================================

    /**
     * @brief Get K-space x-coordinate for a detector pixel.
     *
     * Implements Fourier optics relationship:
     * @f[
     *   k_x = \frac{2\pi \cdot x_{det}}{f \cdot \lambda}
     * @f]
     *
     * @param detector_idx Index of detector pixel (0-based)
     * @param wavelength_m Operating wavelength in meters
     * @return K-space x-coordinate in rad/m
     *
     * @throws std::out_of_range if detector_idx >= num_detectors()
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    double detector_k_x(uint32_t detector_idx, double wavelength_m) const;

    /**
     * @brief Get K-space y-coordinate for a detector pixel.
     *
     * @param detector_idx Index of detector pixel (0-based)
     * @param wavelength_m Operating wavelength in meters
     * @return K-space y-coordinate in rad/m
     *
     * @throws std::out_of_range if detector_idx >= num_detectors()
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    double detector_k_y(uint32_t detector_idx, double wavelength_m) const;

    /**
     * @brief Get K-space position vector for a detector pixel.
     *
     * @param detector_idx Index of detector pixel (0-based)
     * @param wavelength_m Operating wavelength in meters
     * @return K-space position (kx, ky) in rad/m
     *
     * @throws std::out_of_range if detector_idx >= num_detectors()
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    Eigen::Vector2d detector_k(uint32_t detector_idx, double wavelength_m) const;

    /**
     * @brief Get K-space positions for all detector pixels at a wavelength.
     *
     * @param wavelength_m Operating wavelength in meters
     * @return Vector of K-positions (kx, ky) for each detector pixel
     *
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    std::vector<Eigen::Vector2d> all_detector_k(double wavelength_m) const;

    // =========================================================================
    // Detector K-space windows
    // =========================================================================

    /**
     * @brief Get K-space bandwidth for a detector pixel in x direction.
     *
     * Computes the K-space acceptance bandwidth from detector width:
     * @f[
     *   \Delta k_x = \frac{2\pi \cdot w_{det}}{f \cdot \lambda}
     * @f]
     *
     * The detector acts as a spatial frequency filter; its finite aperture
     * defines a K-space window centered on the detector's position.
     *
     * @param detector_idx Index of detector pixel (0-based)
     * @param wavelength_m Operating wavelength in meters
     * @return K-space bandwidth in rad/m
     *
     * @throws std::out_of_range if detector_idx >= num_detectors()
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    double detector_k_bandwidth_x(uint32_t detector_idx, double wavelength_m) const;

    /**
     * @brief Get K-space bandwidth for a detector pixel in y direction.
     *
     * @param detector_idx Index of detector pixel (0-based)
     * @param wavelength_m Operating wavelength in meters
     * @return K-space bandwidth in rad/m
     *
     * @throws std::out_of_range if detector_idx >= num_detectors()
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    double detector_k_bandwidth_y(uint32_t detector_idx, double wavelength_m) const;

    // =========================================================================
    // K-space coordinate conversions
    // =========================================================================

    /**
     * @brief Convert detector position to K-vector.
     *
     * Implements Fourier relationship:
     * @f[
     *   k = \frac{2\pi \cdot x}{f \cdot \lambda}
     * @f]
     *
     * @param x_m Detector position in meters
     * @param wavelength_m Operating wavelength in meters
     * @return K-vector magnitude in rad/m
     *
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    double detector_to_k(double x_m, double wavelength_m) const;

    // =========================================================================
    // LO tilt
    // =========================================================================

    /**
     * @brief Get LO tilt contribution to K-space offset.
     *
     * The LO beam tilt shifts all K-space samples:
     * @f[
     *   k_{LO} = \frac{2\pi \cdot (\sin(\theta_x), \sin(\theta_y))}{\lambda}
     * @f]
     *
     * @param wavelength_m Operating wavelength in meters
     * @return K-space offset (kx, ky) from LO tilt in rad/m
     *
     * @throws std::invalid_argument if wavelength_m <= 0
     */
    Eigen::Vector2d lo_tilt_k(double wavelength_m) const;

    // =========================================================================
    // Detector Sensitivity Functions
    // =========================================================================

    /**
     * @brief Compute detector-only sensitivity at K-offset from detector center.
     *
     * Returns S_det(delta_k) where delta_k = k - k_detector_center.
     *
     * For DETECTOR_SENSITIVITY_POINT: Returns 1.0 always (no spatial filtering).
     * For DETECTOR_SENSITIVITY_SINC:
     * @f[
     *   S_{det}(\Delta k) = \mathrm{sinc}(\Delta k_x \cdot w/2) \cdot \mathrm{sinc}(\Delta k_y \cdot h/2)
     * @f]
     * where w, h are detector width and height in K-space bandwidth units.
     *
     * @param detector_idx Index of detector pixel (0-based)
     * @param wavelength_m Operating wavelength in meters
     * @param delta_k K-offset from detector center (kx, ky) in rad/m
     * @return Sensitivity factor in [0, 1]
     *
     * @throws std::invalid_argument if sensitivity_type is UNSPECIFIED
     * @throws std::out_of_range if detector_idx >= num_detectors()
     */
    double detector_sensitivity(uint32_t detector_idx,
                                double wavelength_m,
                                const Eigen::Vector2d& delta_k) const;

    /**
     * @brief Get the detector sensitivity type.
     * @return The configured sensitivity type
     */
    DetectorSensitivityType sensitivity_type() const { return detector_.sensitivity_type(); }

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get number of detector pixels.
     * @return Number of pixels in the detector array
     */
    uint32_t num_detectors() const;

    /**
     * @brief Get Fourier lens focal length.
     * @return Focal length in meters
     */
    double focal_length_m() const;

    /**
     * @brief Get LO tilt angles.
     * @return (tilt_x, tilt_y) in radians
     */
    Eigen::Vector2d lo_tilt_rad() const;

  private:
    OpticsConfig optics_;
    DetectorArray detector_;

    /// Validate wavelength parameter
    void validate_wavelength(double wavelength_m) const;

    /// Validate detector index parameter
    void validate_detector_index(uint32_t detector_idx) const;

    /// Validate sensitivity_type is not UNSPECIFIED
    void validate_sensitivity_type() const;
};

}  // namespace aurora::forward

#endif  // AURORA_FORWARD_K_SPACE_GEOMETRY_H_
