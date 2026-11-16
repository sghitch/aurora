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
