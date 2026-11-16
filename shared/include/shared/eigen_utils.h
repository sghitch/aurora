// Copyright (C) 2026 Stephen Hitchcock
//
// This file is part of Aurora.
//
// Aurora is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// Aurora is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Aurora. If not, see <https://www.gnu.org/licenses/>.

#ifndef AURORA_SHARED_EIGEN_UTILS_H_
#define AURORA_SHARED_EIGEN_UTILS_H_

/**
 * @file eigen_utils.h
 * @brief Utility functions for Eigen type conversions.
 *
 * Provides helper functions for converting between protobuf vector types
 * and Eigen vector types used throughout the codebase.
 */

#include <Eigen/Dense>

#include "vector.pb.h"

namespace aurora::shared {

/**
 * @brief Convert a protobuf Vector3 to Eigen::Vector3d.
 * @param v Protobuf vector
 * @return Eigen vector
 */
inline Eigen::Vector3d to_eigen(const Vector3& v) {
    return Eigen::Vector3d(v.x(), v.y(), v.z());
}

/**
 * @brief Convert an Eigen::Vector3d to protobuf Vector3.
 * @param v Eigen vector
 * @return Protobuf vector
 */
inline Vector3 to_proto(const Eigen::Vector3d& v) {
    Vector3 result;
    result.set_x(v.x());
    result.set_y(v.y());
    result.set_z(v.z());
    return result;
}

}  // namespace aurora::shared

#endif  // AURORA_SHARED_EIGEN_UTILS_H_
