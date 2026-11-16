#include "reconstruction/k_space_preprocessor.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace aurora::reconstruction {

namespace {

// Validate that an AxisTaperConfig has a valid taper function set.
void validate_axis_taper(const AxisTaperConfig& config, const char* axis_name) {
    if (config.taper_function() == TAPER_NONE) {
        throw std::invalid_argument(
            std::string("AxisTaperConfig for ") + axis_name +
            " has TAPER_NONE set - must explicitly configure taper_function");
    }
    if (config.k_threshold() <= 0) {
        throw std::invalid_argument(
            std::string("AxisTaperConfig for ") + axis_name +
            " has invalid k_threshold <= 0");
    }
    if (config.k_max_taper() <= config.k_threshold()) {
        throw std::invalid_argument(
            std::string("AxisTaperConfig for ") + axis_name +
            " has k_max_taper <= k_threshold");
    }
    if ((config.taper_function() == TAPER_QUADRATIC ||
         config.taper_function() == TAPER_EXPONENTIAL) &&
        config.taper_exponent() <= 0) {
        throw std::invalid_argument(
            std::string("AxisTaperConfig for ") + axis_name +
            " has invalid taper_exponent <= 0 for quadratic/exponential taper_function");
    }
    if (config.retain_fraction() < 0.0 || config.retain_fraction() > 1.0) {
        throw std::invalid_argument(
            std::string("AxisTaperConfig for ") + axis_name +
            " has retain_fraction outside valid range [0, 1]");
    }
}

}  // namespace

KSpacePreprocessor::KSpacePreprocessor(const DirectAdjointConfig& config)
    : apodization_(config.apodization()),
      apod_param_(config.has_apod_param() ? config.apod_param() : 0.4),
      enable_dcf_(config.enable_dcf()),
      dcf_iterations_(config.dcf_iterations() > 0 ? config.dcf_iterations() : 15),
      normalize_output_(config.has_normalize_output() ? config.normalize_output() : true),
      has_density_taper_(false),
      density_taper_() {
    // Validate and store density taper configuration if present
    // Check directly against config before copying to preserve has_* semantics
    if (config.has_density_taper()) {
        const auto& taper = config.density_taper();
        bool has_any = false;

        if (taper.has_isotropic()) {
            validate_axis_taper(taper.isotropic(), "isotropic");
            has_any = true;
        }
        if (taper.has_kx_taper()) {
            validate_axis_taper(taper.kx_taper(), "kx");
            has_any = true;
        }
        if (taper.has_ky_taper()) {
            validate_axis_taper(taper.ky_taper(), "ky");
            has_any = true;
        }

        // At least one taper must be configured
        if (!has_any) {
            throw std::invalid_argument(
                "KSpaceDensityTaperConfig has no taper configured - "
                "must set at least one of isotropic, kx_taper, or ky_taper");
        }

        // Validation passed - store the config
        has_density_taper_ = true;
        density_taper_ = taper;
    }
}

KSpacePreprocessor::KSpacePreprocessor()
    : apodization_(DirectAdjointConfig::APOD_HANN),
      apod_param_(0.4),
      enable_dcf_(false),
      dcf_iterations_(15),
      normalize_output_(true),
      has_density_taper_(false),
      density_taper_() {}

double KSpacePreprocessor::compute_apodization(double kx, double ky,
                                                double k_max_x, double k_max_y) const {
    // Compute normalized radius (elliptical for anisotropic K-space)
    double rx = (k_max_x > 0) ? kx / k_max_x : 0.0;
    double ry = (k_max_y > 0) ? ky / k_max_y : 0.0;
    double r = std::sqrt(rx * rx + ry * ry);

    // Clamp to [0, 1] - samples beyond k_max get zero weight
    if (r > 1.0) {
        return 0.0;
    }

    switch (apodization_) {
        case DirectAdjointConfig::APOD_NONE:
            return 1.0;

        case DirectAdjointConfig::APOD_HANN:
            // Hann window: 0.5 * (1 + cos(π * r))
            return 0.5 * (1.0 + std::cos(M_PI * r));

        case DirectAdjointConfig::APOD_TUKEY: {
            // Tukey window: flat center, tapered edges
            // alpha = fraction that is tapered (0 = rect, 1 = Hann)
            double alpha = apod_param_;
            if (alpha <= 0.0) {
                return 1.0;  // Rectangular
            }
            if (alpha >= 1.0) {
                // Hann
                return 0.5 * (1.0 + std::cos(M_PI * r));
            }
            // Tukey: flat for r < (1-alpha), tapered beyond
            double r_taper = 1.0 - alpha;
            if (r <= r_taper) {
                return 1.0;
            }
            double t = (r - r_taper) / alpha;
            return 0.5 * (1.0 + std::cos(M_PI * t));
        }

        case DirectAdjointConfig::APOD_GAUSSIAN: {
            // Gaussian: exp(-(r/sigma)^2 / 2)
            double sigma = apod_param_ > 0 ? apod_param_ : 0.4;
            return std::exp(-0.5 * (r / sigma) * (r / sigma));
        }

        default:
            return 1.0;
    }
}

double KSpacePreprocessor::compute_axis_taper(double k_abs,
                                               const AxisTaperConfig& config) {
    // Below threshold: full weight
    if (k_abs <= config.k_threshold()) {
        return 1.0;
    }

    // Above max: minimum weight
    if (k_abs >= config.k_max_taper()) {
        return config.retain_fraction();
    }

    // In transition zone: apply taper function
    double t = (k_abs - config.k_threshold()) /
               (config.k_max_taper() - config.k_threshold());

    double taper_value;
    switch (config.taper_function()) {
        case TAPER_LINEAR:
            taper_value = 1.0 - t;
            break;

        case TAPER_COSINE:
            taper_value = 0.5 * (1.0 + std::cos(M_PI * t));
            break;

        case TAPER_QUADRATIC: {
            double exponent = config.has_taper_exponent() ? config.taper_exponent() : 2.0;
            taper_value = std::pow(1.0 - t, exponent);
            break;
        }

        case TAPER_EXPONENTIAL: {
            double exponent = config.has_taper_exponent() ? config.taper_exponent() : 2.0;
            taper_value = std::exp(-exponent * t);
            break;
        }

        default:
            // TAPER_NONE should have been caught in validation
            taper_value = 1.0;
            break;
    }

    // Blend between full weight (1.0) and retain_fraction
    double retain = config.retain_fraction();
    return retain + (1.0 - retain) * taper_value;
}

double KSpacePreprocessor::compute_density_taper(double kx, double ky) const {
    if (!has_density_taper_) {
        return 1.0;
    }

    double weight = 1.0;

    // Apply isotropic (radial) taper if configured
    if (density_taper_.has_isotropic()) {
        double k_mag = std::sqrt(kx * kx + ky * ky);
        weight *= compute_axis_taper(k_mag, density_taper_.isotropic());
    }

    // Apply kx taper if configured
    if (density_taper_.has_kx_taper()) {
        weight *= compute_axis_taper(std::abs(kx), density_taper_.kx_taper());
    }

    // Apply ky taper if configured
    if (density_taper_.has_ky_taper()) {
        weight *= compute_axis_taper(std::abs(ky), density_taper_.ky_taper());
    }

    return weight;
}

aurora::shared::KSpaceSlice KSpacePreprocessor::apply(
    const aurora::shared::KSpaceSlice& slice,
    double k_max_x,
    double k_max_y) const {

    aurora::shared::KSpaceSlice result;
    result.set_range_bin(slice.range_bin());
    result.set_wavelength_m(slice.wavelength_m());

    const int num_samples = slice.samples_size();

    // Compute density compensation weights if enabled
    std::vector<double> dcf_weights(num_samples, 1.0);
    if (enable_dcf_ && num_samples > 0) {
        // Simple Pipe-Menon iterative DCF
        // Initialize weights to 1
        std::vector<double> w(num_samples, 1.0);

        for (uint32_t iter = 0; iter < dcf_iterations_; ++iter) {
            // Compute local density at each sample
            std::vector<double> density(num_samples, 0.0);
            for (int i = 0; i < num_samples; ++i) {
                double ki_x = slice.samples(i).kx();
                double ki_y = slice.samples(i).ky();

                for (int j = 0; j < num_samples; ++j) {
                    double kj_x = slice.samples(j).kx();
                    double kj_y = slice.samples(j).ky();

                    // Kernel: sinc-like falloff based on distance
                    double dx = (ki_x - kj_x) / k_max_x;
                    double dy = (ki_y - kj_y) / k_max_y;
                    double d = std::sqrt(dx * dx + dy * dy);

                    // Use a Gaussian kernel for density estimation
                    double kernel_width = 0.1;  // ~10% of k_max
                    double kernel = std::exp(-d * d / (2.0 * kernel_width * kernel_width));
                    density[i] += w[j] * kernel;
                }
            }

            // Update weights: w = w / density
            for (int i = 0; i < num_samples; ++i) {
                if (density[i] > 1e-10) {
                    w[i] = w[i] / density[i];
                }
            }
        }

        // Normalize weights so mean = 1
        double w_sum = 0.0;
        for (int i = 0; i < num_samples; ++i) {
            w_sum += w[i];
        }
        double w_mean = w_sum / num_samples;
        for (int i = 0; i < num_samples; ++i) {
            dcf_weights[i] = w[i] / w_mean;
        }
    }

    // First pass: compute total weight for normalization
    // Order: density_taper * apodization * dcf
    double weight_sum = 0.0;
    std::vector<double> total_weights(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        const auto& sample = slice.samples(i);
        double taper = compute_density_taper(sample.kx(), sample.ky());
        double apod = compute_apodization(sample.kx(), sample.ky(), k_max_x, k_max_y);
        total_weights[i] = taper * apod * dcf_weights[i];
        weight_sum += total_weights[i];
    }

    // Compute normalization factor
    double norm_factor = 1.0;
    if (normalize_output_ && weight_sum > 1e-10) {
        // Normalize so sum of weights equals num_samples (preserves average magnitude)
        norm_factor = static_cast<double>(num_samples) / weight_sum;
    }

    // Second pass: apply density taper, apodization, DCF, and normalization to each sample
    for (int i = 0; i < num_samples; ++i) {
        const auto& sample = slice.samples(i);

        auto* out_sample = result.add_samples();
        out_sample->set_kx(sample.kx());
        out_sample->set_ky(sample.ky());
        out_sample->set_weight(sample.weight());
        out_sample->set_detector_index(sample.detector_index());

        // Apply weights and normalization to the measurement value
        double weight = total_weights[i] * norm_factor;
        double re = sample.value().real() * weight;
        double im = sample.value().imag() * weight;
        out_sample->mutable_value()->set_real(re);
        out_sample->mutable_value()->set_imag(im);
    }

    return result;
}

}  // namespace aurora::reconstruction
