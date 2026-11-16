#ifndef AURORA_FORWARD_K_SPACE_SLICE_UTILS_H_
#define AURORA_FORWARD_K_SPACE_SLICE_UTILS_H_

/**
 * @file k_space_slice_utils.h
 * @brief Utility functions for working with KSpaceSlice protobufs.
 *
 * This module provides convenience functions for converting KSpaceSlice
 * protobuf data to Eigen types and computing K-space sampling extent metrics.
 * These utilities facilitate interaction between the protobuf data contracts
 * and the numerical algorithms in the reconstruction operators.
 */

#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "k_space_slice.pb.h"

namespace aurora::forward {

/**
 * @brief Extract complex samples from a KSpaceSlice as an Eigen vector.
 *
 * Converts the repeated KSpaceSample messages into a dense Eigen vector
 * suitable for linear algebra operations.
 *
 * @param slice KSpaceSlice containing the samples
 * @return Eigen vector of complex samples, size = slice.samples_size()
 */
Eigen::VectorXcd to_eigen_samples(const aurora::shared::KSpaceSlice& slice);

/**
 * @brief Extract K-vectors from a KSpaceSlice.
 *
 * Returns a vector of 2D K-space positions (kx, ky) for each sample.
 *
 * @param slice KSpaceSlice containing the samples
 * @return Vector of 2D K-vectors in rad/m
 */
std::vector<Eigen::Vector2d> to_k_vectors(const aurora::shared::KSpaceSlice& slice);

/**
 * @brief Extract per-sample weights from a KSpaceSlice.
 *
 * @param slice KSpaceSlice containing the samples
 * @return Vector of sample weights (1.0 for unweighted samples)
 */
std::vector<double> to_weights(const aurora::shared::KSpaceSlice& slice);

/**
 * @brief Compute K-space bounds from a KSpaceSlice.
 *
 * Returns the minimum and maximum K-vector components across all samples.
 *
 * @param slice KSpaceSlice containing the samples
 * @return Tuple of (kx_min, kx_max, ky_min, ky_max) in rad/m
 */
std::tuple<double, double, double, double> compute_k_bounds(
    const aurora::shared::KSpaceSlice& slice);

/**
 * @brief Suggest reconstruction grid size based on the K-space sampling extent.
 *
 * Computes an appropriate grid size for image reconstruction based on
 * the K-space extent and an optional oversampling factor.
 *
 * The suggested grid size is:
 *   n = ceil(k_extent / (2*pi) * oversampling)
 *
 * where k_extent is the span from kmin to kmax.
 *
 * @param slice KSpaceSlice containing the samples
 * @param oversampling Oversampling factor (default: 2.0 for Nyquist)
 * @return Suggested grid size (nx, ny) as integer vector
 */
Eigen::Vector2i suggest_grid_size(
    const aurora::shared::KSpaceSlice& slice,
    double oversampling = 2.0);

/**
 * @brief Compute grid hints from a KSpaceSlice.
 *
 * Populates a KSpaceGridHints message with computed K-space bounds
 * and suggested grid dimensions.
 *
 * @param slice KSpaceSlice containing the samples
 * @param oversampling Oversampling factor for grid size suggestion
 * @return Populated KSpaceGridHints message
 */
aurora::shared::KSpaceGridHints compute_grid_hints(
    const aurora::shared::KSpaceSlice& slice,
    double oversampling = 2.0);

/**
 * @brief Check if a KSpaceSlice is empty.
 *
 * @param slice KSpaceSlice to check
 * @return true if the slice has no samples
 */
inline bool is_empty(const aurora::shared::KSpaceSlice& slice) {
    return slice.samples_size() == 0;
}

/**
 * @brief Get the number of samples in a KSpaceSlice.
 *
 * @param slice KSpaceSlice to query
 * @return Number of samples
 */
inline int num_samples(const aurora::shared::KSpaceSlice& slice) {
    return slice.samples_size();
}

}  // namespace aurora::forward

#endif  // AURORA_FORWARD_K_SPACE_SLICE_UTILS_H_
