#ifndef AURORA_SIMULATION_PUPIL_MODEL_H_
#define AURORA_SIMULATION_PUPIL_MODEL_H_

/**
 * @file pupil_model.h
 * @brief Pupil model for receive optics FOV projection and vignetting.
 *
 * The PupilModel class models the field of view (FOV) and vignetting of the
 * receive optics. This determines which targets are visible and with what
 * transmission efficiency.
 *
 * ### Physical Model
 *
 * The transmission function models vignetting with a smooth rolloff at the
 * FOV edge using a cosine transition:
 *
 *   T(theta) = 1.0                                 if theta < theta_edge
 *            = 0.5 * (1 + cos(pi * t))             if theta_edge <= theta < theta_fov
 *            = 0.0                                 if theta >= theta_fov
 *
 * Where:
 *   - theta = angle from optical axis
 *   - theta_fov = FOV half-angle
 *   - theta_edge = theta_fov - rolloff_width
 *   - t = (theta - theta_edge) / rolloff_width
 *
 * @see PupilConfig for protobuf configuration
 * @see FmcwSimulator for integration with signal simulation
 */

#include <Eigen/Dense>
#include <utility>
#include <vector>

#include "optics_config.pb.h"

namespace aurora::simulation {

/**
 * @brief Models receive optics FOV and vignetting.
 *
 * Computes transmission efficiency based on target angular position
 * relative to the optical axis. Supports both circular and rectangular
 * FOV shapes with smooth edge transitions.
 *
 * ### Example Usage
 * @code
 * aurora::forward::PupilConfig config;
 * config.set_fov_half_angle_rad(0.524);    // 30 degrees
 * config.set_rolloff_width_rad(0.087);     // 5 degrees
 * config.set_shape(aurora::forward::PupilConfig::PUPIL_CIRCULAR);
 *
 * PupilModel pupil(config);
 *
 * Eigen::Vector3d target_pos(1.0, 0.0, 10.0);  // 10m range, 5.7 deg off-axis
 * double transmission = pupil.compute_transmission(target_pos);
 * @endcode
 */
class PupilModel {
  public:
    /**
     * @brief Construct a pupil model from protobuf configuration.
     * @param config Pupil configuration with FOV and rolloff parameters
     */
    explicit PupilModel(const aurora::forward::PupilConfig& config);

    /**
     * @brief Compute transmission for a target at given 3D position.
     *
     * Converts the position to angular coordinates and computes the
     * vignetting transmission. Targets behind the sensor (z <= 0) have
     * zero transmission.
     *
     * @param target_position_m Target position in meters [x, y, z]
     *        where z is the range axis (optical axis)
     * @return Transmission factor in range [0.0, 1.0]
     */
    double compute_transmission(const Eigen::Vector3d& target_position_m) const;

    /**
     * @brief Compute transmission from angular coordinates.
     *
     * Computes vignetting transmission for a target at the specified
     * angular position relative to the optical axis.
     *
     * @param azimuth_rad Azimuth angle in radians (horizontal)
     * @param elevation_rad Elevation angle in radians (vertical)
     * @return Transmission factor in range [0.0, 1.0]
     */
    double compute_transmission(double azimuth_rad, double elevation_rad) const;

    /**
     * @brief Check if position is within FOV (any transmission > 0).
     * @param target_position_m Target position in meters
     * @return true if target has non-zero transmission
     */
    bool is_in_fov(const Eigen::Vector3d& target_position_m) const;

    /**
     * @brief Get FOV boundary points for visualization/debugging.
     *
     * Returns points on the FOV boundary in angular coordinates
     * (azimuth, elevation) for plotting or debugging purposes.
     *
     * @param num_points Number of points around the boundary
     * @return Vector of (azimuth, elevation) pairs in radians
     */
    std::vector<std::pair<double, double>> get_fov_boundary(
        int num_points = 100) const;

    /**
     * @brief Get the configured FOV half-angle.
     * @return FOV half-angle in radians
     */
    double fov_half_angle_rad() const { return fov_half_angle_rad_; }

    /**
     * @brief Get the configured rolloff width.
     * @return Rolloff width in radians
     */
    double rolloff_width_rad() const { return rolloff_width_rad_; }

    /**
     * @brief Check if the pupil shape is circular.
     * @return true if circular, false if rectangular
     */
    bool is_circular() const {
        return shape_ == aurora::forward::PupilConfig::PUPIL_CIRCULAR;
    }

  private:
    /**
     * @brief Convert 3D position to angular coordinates.
     * @param position_m Position in meters
     * @return Pair of (azimuth, elevation) in radians
     */
    std::pair<double, double> position_to_angles(
        const Eigen::Vector3d& position_m) const;

    double fov_half_angle_rad_;
    double rolloff_width_rad_;
    aurora::forward::PupilConfig::PupilShape shape_;
    double aspect_ratio_;  // For rectangular FOV (width/height)
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_PUPIL_MODEL_H_
