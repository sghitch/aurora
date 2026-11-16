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

#include "reconstruction/reconstruction_operator.h"

#include <algorithm>
#include <cmath>

namespace aurora::reconstruction {

double ReconstructionOperator::check_adjoint_consistency(
    const Eigen::MatrixXcd& x,
    const Eigen::VectorXcd& y,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing) const {
    // Compute <Ax, y>
    auto Ax = forward(x, grid_origin, grid_spacing);
    std::complex<double> inner_Ax_y{0.0, 0.0};
    for (Eigen::Index i = 0; i < Ax.size(); ++i) {
        inner_Ax_y += Ax(i) * std::conj(y(i));
    }

    // Compute <x, A*y>
    Eigen::Vector2i grid_size(static_cast<int>(x.cols()),
                               static_cast<int>(x.rows()));
    auto Asy = adjoint(y, grid_origin, grid_spacing, grid_size);
    std::complex<double> inner_x_Asy{0.0, 0.0};
    for (Eigen::Index iy = 0; iy < x.rows(); ++iy) {
        for (Eigen::Index ix = 0; ix < x.cols(); ++ix) {
            inner_x_Asy += x(iy, ix) * std::conj(Asy(iy, ix));
        }
    }

    // Return relative error
    double magnitude = std::max(std::abs(inner_Ax_y), std::abs(inner_x_Asy));
    if (magnitude < 1e-15) {
        return 0.0;
    }
    return std::abs(inner_Ax_y - inner_x_Asy) / magnitude;
}

Eigen::MatrixXcd ReconstructionOperator::apply_normal(
    const Eigen::MatrixXcd& x,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing) const {
    // A* A x = adjoint(forward(x))
    auto Ax = forward(x, grid_origin, grid_spacing);
    Eigen::Vector2i grid_size(static_cast<int>(x.cols()),
                               static_cast<int>(x.rows()));
    return adjoint(Ax, grid_origin, grid_spacing, grid_size);
}

}  // namespace aurora::reconstruction
