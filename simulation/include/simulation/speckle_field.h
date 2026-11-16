#ifndef AURORA_SIMULATION_SPECKLE_FIELD_H_
#define AURORA_SIMULATION_SPECKLE_FIELD_H_

/**
 * @file speckle_field.h
 * @brief Deterministic correlated complex Gaussian speckle generator.
 *
 * The SpeckleField class generates complex Gaussian speckle coefficients
 * for the diffuse component of plane target K-space response. The speckle
 * has spatial correlation determined by the surface correlation length.
 *
 * ### Physics Background
 *
 * Real rough surfaces produce speckle with spatial correlation determined
 * by the surface texture. The diffuse field in K-space is:
 *   E_diff(k) = E_spec(k) * xi(k)
 *
 * Where xi(k) is a complex Gaussian random field with correlation:
 *   <xi(k1) xi*(k2)> = exp(-|k1-k2|^2 * l_c^2 / 4)
 *
 * - Correlation length l_c = characteristic surface texture scale
 * - Larger l_c -> more correlated in K-space -> larger speckle grains
 * - Complex Gaussian: Re(xi) and Im(xi) are independent N(0, 1/sqrt(2))
 *
 * ### Implementation
 *
 * Uses smooth interpolation between grid points spaced at 2*pi/l_c. Random
 * values are assigned at each grid point, and intermediate K-vectors get
 * smoothly interpolated values using a smoothstep function for C1 continuity.
 * This produces a smooth correlation function without discontinuities.
 *
 * @see SurfaceProperties for correlation_length_m field
 * @see PlaneTargetView for diffuse response computation
 */

#include <Eigen/Dense>
#include <complex>
#include <cstdint>

namespace aurora::simulation {

/**
 * @brief Generates deterministic correlated complex Gaussian speckle.
 *
 * This is a static utility class for generating speckle coefficients
 * that are:
 * - Deterministic: same inputs always produce the same output
 * - Spatially correlated: nearby K-vectors have correlated values
 * - Complex Gaussian: unit variance with independent real/imaginary parts
 *
 * ### Example Usage
 * @code
 * // Generate speckle coefficient for a K-vector
 * Eigen::Vector2d k_vector(1000.0, 500.0);  // rad/m
 * double correlation_length = 1e-4;  // 100 um
 * uint64_t seed = SpeckleField::hash_combine(42, target_id);
 *
 * std::complex<double> xi = SpeckleField::sample(
 *     k_vector, correlation_length, seed);
 *
 * // Apply to specular response for diffuse component
 * auto E_diffuse = E_specular * xi;
 * @endcode
 */
class SpeckleField {
  public:
    /**
     * @brief Generate speckle coefficient for given K-vector.
     *
     * Returns a complex Gaussian sample with unit variance. Uses smooth
     * interpolation between grid points spaced at 2*pi/correlation_length_m,
     * producing continuous correlation without discontinuities.
     *
     * @param k_vector K-space position (kx, ky) in rad/m
     * @param correlation_length_m Surface correlation length in meters
     * @param seed Deterministic seed (e.g., hash of target_id + facet index)
     * @return Complex Gaussian sample with unit variance: E[|xi|^2] = 1
     */
    static std::complex<double> sample(
        const Eigen::Vector2d& k_vector,
        double correlation_length_m,
        uint64_t seed);

    /**
     * @brief Combine two seeds into one deterministic seed.
     *
     * Uses a boost-style hash combine algorithm for good distribution.
     *
     * @param seed Base seed value
     * @param value Value to combine
     * @return Combined seed
     */
    static uint64_t hash_combine(uint64_t seed, uint64_t value);

    /**
     * @brief Combine seed with K-space grid cell indices.
     *
     * @param seed Base seed value
     * @param kx_bin X grid cell index
     * @param ky_bin Y grid cell index
     * @return Combined seed for the grid cell
     */
    static uint64_t hash_combine(uint64_t seed, int64_t kx_bin, int64_t ky_bin);

  private:
    SpeckleField() = delete;  // Static-only class

    /**
     * @brief Generate speckle value at a specific grid point.
     *
     * @param kx_bin X grid cell index
     * @param ky_bin Y grid cell index
     * @param seed Deterministic seed
     * @return Complex Gaussian sample with unit variance
     */
    static std::complex<double> sample_at_grid_point(
        int64_t kx_bin, int64_t ky_bin, uint64_t seed);
};

}  // namespace aurora::simulation

#endif  // AURORA_SIMULATION_SPECKLE_FIELD_H_
