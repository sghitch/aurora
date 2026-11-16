#include "simulation/speckle_field.h"

#include <cmath>
#include <random>

namespace aurora::simulation {

namespace {

// Smoothstep interpolation for C1 continuity (no derivative discontinuities)
double smoothstep(double t) {
    return t * t * (3.0 - 2.0 * t);
}

}  // namespace

std::complex<double> SpeckleField::sample(
    const Eigen::Vector2d& k_vector,
    double correlation_length_m,
    uint64_t seed) {

    // Grid spacing based on correlation length.
    // The correlation function will smoothly decay with this characteristic scale.
    double k_grid = 2.0 * M_PI / correlation_length_m;

    // Normalized coordinates in grid units
    double kx_norm = k_vector.x() / k_grid;
    double ky_norm = k_vector.y() / k_grid;

    // Grid cell containing this point
    int64_t kx0 = static_cast<int64_t>(std::floor(kx_norm));
    int64_t ky0 = static_cast<int64_t>(std::floor(ky_norm));

    // Fractional position within cell [0, 1)
    double fx = kx_norm - static_cast<double>(kx0);
    double fy = ky_norm - static_cast<double>(ky0);

    // Smooth interpolation weights using smoothstep for C1 continuity.
    // This eliminates derivative discontinuities at cell boundaries.
    double wx = smoothstep(fx);
    double wy = smoothstep(fy);

    // Compute interpolation weights for all four corners
    double w00 = (1.0 - wx) * (1.0 - wy);
    double w10 = wx * (1.0 - wy);
    double w01 = (1.0 - wx) * wy;
    double w11 = wx * wy;

    // Sample at four corners of the grid cell
    auto v00 = sample_at_grid_point(kx0, ky0, seed);
    auto v10 = sample_at_grid_point(kx0 + 1, ky0, seed);
    auto v01 = sample_at_grid_point(kx0, ky0 + 1, seed);
    auto v11 = sample_at_grid_point(kx0 + 1, ky0 + 1, seed);

    // Bilinear interpolation
    auto result = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;

    // Normalize to preserve unit variance.
    // The variance of a weighted sum of independent unit-variance variables is
    // the sum of squared weights. We divide by sqrt of this to restore unit variance.
    double weight_norm = std::sqrt(w00*w00 + w10*w10 + w01*w01 + w11*w11);
    return result / weight_norm;
}

std::complex<double> SpeckleField::sample_at_grid_point(
    int64_t kx_bin, int64_t ky_bin, uint64_t seed) {

    // Deterministic seed from grid cell coordinates
    uint64_t cell_seed = hash_combine(seed, kx_bin, ky_bin);
    std::mt19937_64 rng(cell_seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Complex Gaussian with unit variance: E[|xi|^2] = 1
    // Re and Im each have variance 1/2, so we scale by 1/sqrt(2)
    double re = normal(rng) / std::sqrt(2.0);
    double im = normal(rng) / std::sqrt(2.0);

    return std::complex<double>(re, im);
}

uint64_t SpeckleField::hash_combine(uint64_t seed, uint64_t value) {
    // Boost-style hash combine for good distribution
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

uint64_t SpeckleField::hash_combine(uint64_t seed, int64_t kx_bin, int64_t ky_bin) {
    uint64_t h = hash_combine(seed, static_cast<uint64_t>(kx_bin));
    return hash_combine(h, static_cast<uint64_t>(ky_bin));
}

}  // namespace aurora::simulation
