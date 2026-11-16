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

#include "reconstruction/regularizer.h"

#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aurora::reconstruction {

Eigen::MatrixXcd GradientRegularizer::apply_LtL(const Eigen::MatrixXcd& x) const {
    // L†L for gradient operator with periodic boundary conditions.
    // Using forward differences: (∂_x f)_i = f_{i+1} - f_i
    //
    // The adjoint of the forward difference is the negated difference taken
    // in the opposite direction.
    // L†L gives the 5-point stencil (negative discrete Laplacian):
    //   (L†Lx)_{i,j} = 4x_{i,j} - x_{i+1,j} - x_{i-1,j} - x_{i,j+1} - x_{i,j-1}
    //
    // This is self-adjoint and has eigenvalue 4 - 2cos(kx) - 2cos(ky).

    const int Ny = static_cast<int>(x.rows());
    const int Nx = static_cast<int>(x.cols());

    Eigen::MatrixXcd result(Ny, Nx);

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            // Periodic indices
            int ip1 = (i + 1) % Nx;
            int im1 = (i - 1 + Nx) % Nx;
            int jp1 = (j + 1) % Ny;
            int jm1 = (j - 1 + Ny) % Ny;

            // L†Lx = 4x - x_{i+1} - x_{i-1} - x_{j+1} - x_{j-1}
            result(j, i) = 4.0 * x(j, i) - x(j, ip1) - x(j, im1)
                           - x(jp1, i) - x(jm1, i);
        }
    }

    return result;
}

Eigen::MatrixXd GradientRegularizer::frequency_response(
    const Eigen::Vector2i& size,
    const Eigen::Vector2d& spacing) const {

    Eigen::MatrixXd spectrum(size.y(), size.x());

    for (int j = 0; j < size.y(); ++j) {
        for (int i = 0; i < size.x(); ++i) {
            // Normalized frequency in [0, 2π)
            double kx = 2.0 * M_PI * i / size.x();
            double ky = 2.0 * M_PI * j / size.y();

            // Wrap to [-π, π] for symmetric representation
            if (kx > M_PI) kx -= 2.0 * M_PI;
            if (ky > M_PI) ky -= 2.0 * M_PI;

            // Gradient L†L eigenvalue (5-point stencil):
            //   Φ(kx, ky) = 4 - 2cos(kx) - 2cos(ky)  (unit spacing)
            //
            // With general spacing, the finite difference 1/dx scaling gives:
            //   Φ(kx, ky) = (2 - 2cos(kx))/dx² + (2 - 2cos(ky))/dy²
            //
            // For consistency with apply_LtL (which doesn't scale by spacing),
            // we use unit spacing eigenvalues here.
            (void)spacing;  // Unused - operator doesn't scale by spacing
            spectrum(j, i) = 4.0 - 2.0 * std::cos(kx) - 2.0 * std::cos(ky);
        }
    }
    return spectrum;
}

Eigen::MatrixXcd LaplacianRegularizer::apply_LtL(const Eigen::MatrixXcd& x) const {
    // L = ∇² (discrete Laplacian using 5-point stencil)
    // Lx = x_{i+1,j} + x_{i-1,j} + x_{i,j+1} + x_{i,j-1} - 4x_{i,j}
    //
    // L† = L (Laplacian is self-adjoint)
    // L†L = ∇⁴ (biharmonic operator)
    //
    // We apply ∇² twice with periodic boundaries.

    const int Ny = static_cast<int>(x.rows());
    const int Nx = static_cast<int>(x.cols());

    // First pass: compute Laplacian
    Eigen::MatrixXcd laplacian(Ny, Nx);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int ip1 = (i + 1) % Nx;
            int im1 = (i - 1 + Nx) % Nx;
            int jp1 = (j + 1) % Ny;
            int jm1 = (j - 1 + Ny) % Ny;

            // ∇²x = neighbors - 4*center
            laplacian(j, i) = x(j, ip1) + x(j, im1) + x(jp1, i) + x(jm1, i)
                              - 4.0 * x(j, i);
        }
    }

    // Second pass: apply Laplacian again (∇⁴ = ∇²∇²)
    Eigen::MatrixXcd result(Ny, Nx);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int ip1 = (i + 1) % Nx;
            int im1 = (i - 1 + Nx) % Nx;
            int jp1 = (j + 1) % Ny;
            int jm1 = (j - 1 + Ny) % Ny;

            result(j, i) = laplacian(j, ip1) + laplacian(j, im1)
                           + laplacian(jp1, i) + laplacian(jm1, i)
                           - 4.0 * laplacian(j, i);
        }
    }

    return result;
}

Eigen::MatrixXd LaplacianRegularizer::frequency_response(
    const Eigen::Vector2i& size,
    const Eigen::Vector2d& spacing) const {

    Eigen::MatrixXd spectrum(size.y(), size.x());

    for (int j = 0; j < size.y(); ++j) {
        for (int i = 0; i < size.x(); ++i) {
            // Normalized frequency in [0, 2π)
            double kx = 2.0 * M_PI * i / size.x();
            double ky = 2.0 * M_PI * j / size.y();

            // Wrap to [-π, π] for symmetric representation
            if (kx > M_PI) kx -= 2.0 * M_PI;
            if (ky > M_PI) ky -= 2.0 * M_PI;

            // Laplacian eigenvalue (5-point stencil):
            //   λ = 4 - 2cos(kx) - 2cos(ky)
            //
            // For L†L = ∇⁴ (biharmonic), the eigenvalue is λ²:
            //   Φ(kx, ky) = (4 - 2cos(kx) - 2cos(ky))²
            (void)spacing;  // Unused - operator doesn't scale by spacing
            double lambda = 4.0 - 2.0 * std::cos(kx) - 2.0 * std::cos(ky);
            spectrum(j, i) = lambda * lambda;
        }
    }
    return spectrum;
}

RegularizerVariant create_regularizer(RegularizerType type) {
    switch (type) {
        case REGULARIZER_TIKHONOV:
            return TikhonovRegularizer{};
        case REGULARIZER_GRADIENT:
            return GradientRegularizer{};
        case REGULARIZER_LAPLACIAN:
            return LaplacianRegularizer{};
        case REGULARIZER_UNKNOWN:
        default:
            throw std::invalid_argument(
                "create_regularizer: REGULARIZER_UNKNOWN is not valid");
    }
}

}  // namespace aurora::reconstruction
