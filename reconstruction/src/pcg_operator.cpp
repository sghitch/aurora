#include "reconstruction/pcg_operator.h"

#include <cmath>
#include <stdexcept>

namespace aurora::reconstruction {

void validate_pcg_config(const PcgConfig& config) {
    // max_iterations must be at least 1
    if (config.max_iterations() < 1) {
        throw std::invalid_argument(
            "PcgConfig: max_iterations must be at least 1");
    }

    // tolerance must be in range (0, 1)
    if (config.tolerance() <= 0.0 || config.tolerance() >= 1.0) {
        throw std::invalid_argument(
            "PcgConfig: tolerance must be in range (0, 1)");
    }

    // regularizer_type must be explicitly set
    if (config.regularizer_type() == REGULARIZER_UNKNOWN) {
        throw std::invalid_argument(
            "PcgConfig: regularizer_type must be explicitly set");
    }

    // regularizer_lambda must be positive
    if (config.regularizer_lambda() <= 0.0) {
        throw std::invalid_argument(
            "PcgConfig: regularizer_lambda must be positive");
    }

    // preconditioner_type must be explicitly set
    if (config.preconditioner_type() == PRECONDITIONER_UNKNOWN) {
        throw std::invalid_argument(
            "PcgConfig: preconditioner_type must be explicitly set");
    }

    // preconditioner_epsilon required for CIRCULANT and HYBRID
    if ((config.preconditioner_type() == PRECONDITIONER_CIRCULANT ||
         config.preconditioner_type() == PRECONDITIONER_HYBRID) &&
        config.preconditioner_epsilon() <= 0.0) {
        throw std::invalid_argument(
            "PcgConfig: preconditioner_epsilon must be positive for "
            "CIRCULANT or HYBRID preconditioner");
    }

    // nufft_config required when use_nufft_operator is true
    if (config.use_nufft_operator() && !config.has_nufft_config()) {
        throw std::invalid_argument(
            "PcgConfig: nufft_config required when use_nufft_operator is true");
    }

    // Validate warm_start config if present
    if (config.has_warm_start()) {
        const auto& ws = config.warm_start();

        // change_threshold must be in range (0, 1]
        if (ws.change_threshold() <= 0.0 || ws.change_threshold() > 1.0) {
            throw std::invalid_argument(
                "PcgConfig: warm_start.change_threshold must be in range (0, 1]");
        }

        // cache_depth must be at least 1
        if (ws.cache_depth() < 1) {
            throw std::invalid_argument(
                "PcgConfig: warm_start.cache_depth must be at least 1");
        }
    }
}

PcgOperator::PcgOperator(const aurora::shared::KSpaceSlice& slice,
                         const aurora::forward::ForwardModel& forward_model,
                         const aurora::forward::BandwidthConfig& bandwidth_config,
                         const Eigen::Vector2i& grid_size,
                         const Eigen::Vector2d& grid_spacing,
                         const Eigen::Vector2d& grid_origin,
                         const PcgConfig& config)
    : config_(config),
      inner_operator_(DirectOperator(slice, forward_model, bandwidth_config,
                                     grid_size, grid_spacing, grid_origin)),
      grid_size_(grid_size),
      grid_spacing_(grid_spacing),
      grid_origin_(grid_origin) {

    validate_pcg_config(config_);

    // Create inner operator based on config
    if (config_.use_nufft_operator()) {
        inner_operator_ = NufftOperator(
            slice, forward_model, grid_size_, grid_spacing_, config_.nufft_config());
    }

    initialize();
}

void PcgOperator::initialize() {
    // Create regularizer
    regularizer_ = create_regularizer(config_.regularizer_type());
    lambda_ = config_.regularizer_lambda();
}

PcgOperator::~PcgOperator() = default;

PcgOperator::PcgOperator(PcgOperator&&) noexcept = default;
PcgOperator& PcgOperator::operator=(PcgOperator&&) noexcept = default;

Eigen::VectorXcd PcgOperator::forward(
    const Eigen::MatrixXcd& scene,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing) const {
    // Delegate to inner operator
    return get_inner_operator(inner_operator_).forward(scene, grid_origin, grid_spacing);
}

Eigen::MatrixXcd PcgOperator::adjoint(
    const Eigen::VectorXcd& k_samples,
    const Eigen::Vector2d& grid_origin,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2i& grid_size) const {

    // Unit weights (could be extended to accept weights)
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(k_samples.size());

    // Compute RHS: b = A†Wy using inner operator
    Eigen::VectorXcd weighted = (weights.array() * k_samples.array()).matrix();
    const auto& inner = get_inner_operator(inner_operator_);
    Eigen::MatrixXcd b = inner.adjoint(
        weighted, grid_origin, grid_spacing, grid_size);

    // Initialize x from dirty image (adjoint result)
    Eigen::MatrixXcd x = b;

    // Initial residual: r = b - Hx
    Eigen::MatrixXcd r = b - apply_H(x, weights, grid_origin, grid_spacing);
    double r0_norm = r.norm();

    // Handle trivial case
    if (r0_norm < 1e-15) {
        last_iterations_ = 0;
        last_residual_ = r0_norm;
        return x;
    }

    // Create identity preconditioner for now
    // (Full preconditioner support requires precomputed data)
    IdentityPreconditioner preconditioner;

    // Precondition: z = P⁻¹r
    Eigen::MatrixXcd z = preconditioner.apply(r);
    Eigen::MatrixXcd p = z;

    // ⟨r, z⟩ = sum of r* .* z
    std::complex<double> rz = (r.conjugate().cwiseProduct(z)).sum();

    // PCG iterations
    for (uint32_t k = 0; k < config_.max_iterations(); ++k) {
        // q = H·p
        Eigen::MatrixXcd q = apply_H(p, weights, grid_origin, grid_spacing);

        // α = ⟨r, z⟩ / ⟨p, q⟩
        std::complex<double> pq = (p.conjugate().cwiseProduct(q)).sum();

        // Avoid division by zero
        if (std::abs(pq) < 1e-30) {
            last_iterations_ = static_cast<int>(k + 1);
            last_residual_ = r.norm();
            return x;
        }

        std::complex<double> alpha = rz / pq;

        // x_{k+1} = x_k + α·p
        x += alpha * p;

        // r_{k+1} = r_k - α·q
        r -= alpha * q;

        // Check convergence
        double r_norm = r.norm();
        if (r_norm / r0_norm < config_.tolerance()) {
            last_iterations_ = static_cast<int>(k + 1);
            last_residual_ = r_norm;
            return x;
        }

        // z_{k+1} = P⁻¹·r_{k+1}
        z = preconditioner.apply(r);

        // β = ⟨r_{k+1}, z_{k+1}⟩ / ⟨r_k, z_k⟩
        std::complex<double> rz_new = (r.conjugate().cwiseProduct(z)).sum();

        // Avoid division by zero
        if (std::abs(rz) < 1e-30) {
            last_iterations_ = static_cast<int>(k + 1);
            last_residual_ = r_norm;
            return x;
        }

        std::complex<double> beta = rz_new / rz;
        rz = rz_new;

        // p_{k+1} = z_{k+1} + β·p_k
        p = z + beta * p;
    }

    last_iterations_ = static_cast<int>(config_.max_iterations());
    last_residual_ = r.norm();
    return x;
}

size_t PcgOperator::num_samples() const {
    return get_inner_operator(inner_operator_).num_samples();
}

const std::vector<Eigen::Vector2d>& PcgOperator::k_vectors() const {
    return get_inner_operator(inner_operator_).k_vectors();
}

const std::vector<double>& PcgOperator::weights() const {
    return get_inner_operator(inner_operator_).weights();
}

Eigen::MatrixXcd PcgOperator::apply_H(
    const Eigen::MatrixXcd& x,
    const Eigen::VectorXd& weights,
    const Eigen::Vector2d& origin,
    const Eigen::Vector2d& spacing) const {

    const auto& inner = get_inner_operator(inner_operator_);

    // A†WA part using inner operator
    auto y = inner.forward(x, origin, spacing);
    y = (weights.array() * y.array()).matrix();
    auto AwAx = inner.adjoint(
        y, origin, spacing,
        Eigen::Vector2i(static_cast<int>(x.cols()), static_cast<int>(x.rows())));

    // λL†L part
    const Regularizer& reg = get_regularizer(regularizer_);
    auto LtLx = reg.apply_LtL(x);

    return AwAx + lambda_ * LtLx;
}

}  // namespace aurora::reconstruction
