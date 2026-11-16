#include "simulation/fmcw_simulator.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "forward/measurement_operator.h"
#include "simulation/speckle_field.h"
#include "simulation/visibility_checker.h"

#ifdef AURORA_USE_OPENMP
#include <omp.h>
#endif

namespace aurora::simulation {

namespace {

using aurora::forward::MeasurementOperator;

/**
 * @brief Convert target angular position to K-space position.
 *
 * For a target at angles (theta_x, theta_y), the K-space position is:
 *   k = (2π/λ) · (sin(theta_x), sin(theta_y))
 *
 * For small angles (sin(θ) ≈ θ), this simplifies to k ≈ (2π/λ) · (θ_x, θ_y).
 *
 * @param theta_x Azimuth angle in radians
 * @param theta_y Elevation angle in radians
 * @param wavelength_m Wavelength in meters
 * @return 2D K-vector (kx, ky) in rad/m
 */
Eigen::Vector2d angle_to_k(double theta_x, double theta_y, double wavelength_m) {
    double k_mag = 2.0 * std::numbers::pi / wavelength_m;
    return {k_mag * std::sin(theta_x), k_mag * std::sin(theta_y)};
}

/**
 * @brief Numerically stable sinc function: sin(x)/x with sinc(0) = 1.
 */
double safe_sinc(double x) {
    if (std::abs(x) < 1e-10) {
        return 1.0;
    }
    return std::sin(x) / x;
}

}  // namespace

FmcwSimulator::FmcwSimulator(const aurora::forward::ForwardModel& forward_model,
                             const SimConfig& config)
    : config_(config),
      forward_model_(forward_model),
      scene_(config.scene()),
      pupil_model_(forward_model.optics().pupil()),
      noise_model_(config.noise(), config.adc()),
      adc_model_(config.adc()),
      rng_(config.rng_seed()),
      burst_counter_(0) {
    // Validate required config fields
    if (!config.has_bandwidth_config()) {
        throw std::invalid_argument(
            "SimConfig.bandwidth_config must be set.");
    }
    if (config.bandwidth_config().window_type() ==
            aurora::forward::BANDWIDTH_WINDOW_UNSPECIFIED) {
        throw std::invalid_argument(
            "SimConfig.bandwidth_config.window_type must be explicitly set.");
    }
    if (config.bandwidth_config().quadrature_order() == 0) {
        throw std::invalid_argument(
            "SimConfig.bandwidth_config.quadrature_order must be set (recommended: 11).");
    }
    if (config.specular_threshold() <= 0.0) {
        throw std::invalid_argument(
            "SimConfig.specular_threshold must be set (recommended: 0.9999999999).");
    }

#ifdef AURORA_USE_OPENMP
    // Enable nested parallelism for detector + target loops
    omp_set_nested(1);
#endif
}

Burst FmcwSimulator::simulate_burst(const aurora::acquisition::SequencePlan& plan) {
    Burst burst;
    burst.set_burst_id(burst_counter_++);
    burst.set_timestamp_s(plan.start_timestamp_s());
    *burst.mutable_plan() = plan;

    // The ForwardModel maps the first detector array
    const uint32_t num_detectors = forward_model_.num_detectors();
    if (num_detectors == 0) {
        throw std::invalid_argument(
            "No detector pixels configured in forward_config");
    }

    const int num_det = static_cast<int>(num_detectors);

#ifdef AURORA_USE_OPENMP
    // Pre-generate RNG seeds for each detector (ensures determinism)
    std::vector<uint64_t> rng_seeds(num_det);
    for (int det = 0; det < num_det; ++det) {
        rng_seeds[det] = rng_();
    }

    // Storage for detector bursts (populated in parallel)
    std::vector<DetectorBurst> detector_bursts(num_det);

    // Process detectors in parallel
#pragma omp parallel for schedule(dynamic)
    for (int det = 0; det < num_det; ++det) {
        // Thread-local RNG seeded deterministically
        std::mt19937_64 local_rng(rng_seeds[det]);
        detector_bursts[det] = simulate_detector(plan, det, local_rng);
    }

    // Add detector bursts sequentially (protobuf not thread-safe)
    for (int det = 0; det < num_det; ++det) {
        *burst.add_detectors() = std::move(detector_bursts[det]);
    }
#else
    // Sequential fallback
    for (int det = 0; det < num_det; ++det) {
        *burst.add_detectors() = simulate_detector(plan, det, rng_);
    }
#endif

    return burst;
}

DetectorBurst FmcwSimulator::simulate_detector(
    const aurora::acquisition::SequencePlan& plan,
    uint32_t detector_index,
    std::mt19937_64& rng) const {

    DetectorBurst detector_burst;
    detector_burst.set_detector_index(detector_index);

    for (const auto& chirp_plan : plan.chirps()) {
        auto* chirp_samples = detector_burst.add_chirps();
        chirp_samples->set_chirp_index(chirp_plan.chirp_index());

        for (double wavelength_m : chirp_plan.wavelength_m()) {
            chirp_samples->add_wavelengths_m(wavelength_m);

            // Generate beat signal for this detector pixel
            auto beat_signal = generate_beat_signal(chirp_plan, detector_index, wavelength_m);

            // Compute optical power for noise calculation
            double optical_power_w = config_.tx_power_w() * config_.system_efficiency();

            // Apply noise and ADC
            auto digitized = apply_noise_and_adc(beat_signal, optical_power_w, rng);

            // Store samples (convert int16_t to int32 for protobuf)
            for (int16_t sample : digitized) {
                chirp_samples->add_samples(static_cast<int32_t>(sample));
            }
        }
    }
    return detector_burst;
}

void FmcwSimulator::update_scene(double delta_time_s) {
    scene_.update(delta_time_s);
}

void FmcwSimulator::reset_rng() {
    rng_.seed(config_.rng_seed());
}

std::vector<double> FmcwSimulator::generate_beat_signal(
    const aurora::acquisition::ChirpPlan& chirp_plan,
    uint32_t detector_index,
    double wavelength_m) const {

    // Get sample rate from ADC config
    double sample_rate_hz = adc_model_.sample_rate_hz();

    // Compute number of samples
    uint32_t num_samples = ChirpGenerator::compute_num_samples(
        chirp_plan.chirp_duration_s(), sample_rate_hz);

    // Initialize output signal to zeros
    std::vector<double> signal(num_samples, 0.0);

    // Acceptance window of this detector pixel: centered on its scene K
    // (detector K plus LO tilt) with bandwidth set by the pixel size.
    const MeasurementOperator::WindowParams window =
        aurora::forward::build_window_params(forward_model_, detector_index, wavelength_m);

    // Get references to target lists
    const auto& point_targets = scene_.get_point_targets();
    const int num_point_targets = static_cast<int>(point_targets.size());

#ifdef AURORA_USE_OPENMP
    // Parallelize over point targets using thread-local signal vectors
    // When nested, use sqrt(max_threads) to balance outer/inner parallelism
    if (num_point_targets > 0) {
        const int max_threads = omp_get_max_threads();
        const int nested_threads = omp_in_parallel() ?
            std::max(2, static_cast<int>(std::sqrt(max_threads))) : max_threads;

#pragma omp parallel num_threads(nested_threads)
        {
            // Thread-local signal accumulator
            std::vector<double> local_signal(num_samples, 0.0);

#pragma omp for schedule(static) nowait
            for (int t = 0; t < num_point_targets; ++t) {
                PointTargetView target(point_targets[t]);

                // Compute pupil transmission
                double transmission = pupil_model_.compute_transmission(target.position());

                // Skip targets outside FOV
                if (transmission < 0.01) {
                    continue;
                }

                // Compute target response (phase encoded by the pixel's scene K)
                auto response = compute_target_response(
                    target, chirp_plan, window, wavelength_m, transmission);

                // Add to thread-local signal
                for (size_t i = 0; i < local_signal.size() && i < response.size(); ++i) {
                    local_signal[i] += response[i];
                }
            }

            // Reduce thread-local signals into main signal
#pragma omp critical
            {
                for (size_t i = 0; i < signal.size(); ++i) {
                    signal[i] += local_signal[i];
                }
            }
        }
    }
#else
    // Sequential fallback for point targets
    for (const auto& target_proto : point_targets) {
        PointTargetView target(target_proto);
        double transmission = pupil_model_.compute_transmission(target.position());
        if (transmission < 0.01) {
            continue;
        }

        auto response = compute_target_response(
            target, chirp_plan, window, wavelength_m, transmission);

        for (size_t i = 0; i < signal.size() && i < response.size(); ++i) {
            signal[i] += response[i];
        }
    }
#endif

    // Process plane targets using analytic model
    const auto& plane_targets = scene_.get_plane_targets();
    const int num_plane_targets = static_cast<int>(plane_targets.size());

    for (int plane_idx = 0; plane_idx < num_plane_targets; ++plane_idx) {
        PlaneTargetView plane(plane_targets[plane_idx]);

        // Compute pupil transmission at plane center
        double transmission = pupil_model_.compute_transmission(plane.center());

        // Skip planes outside FOV
        if (transmission < 0.01) {
            continue;
        }

        // Generate deterministic speckle seed from simulation RNG seed and plane index.
        // The seed must NOT depend on the chirp or detector - speckle is a property
        // of the target surface, not the measurement. All detector pixels observing
        // the same target must see the same speckle for coherent imaging to work.
        uint64_t speckle_seed = SpeckleField::hash_combine(
            config_.rng_seed(), static_cast<uint64_t>(plane_idx));

        // Compute response using analytic model (specular + diffuse)
        auto response = compute_plane_target_response(
            plane, chirp_plan, window, wavelength_m, transmission, speckle_seed);

        // Add to main signal
        for (size_t i = 0; i < signal.size() && i < response.size(); ++i) {
            signal[i] += response[i];
        }
    }

    // Process mesh targets using visibility checker and analytic model
    const auto& mesh_targets = scene_.get_mesh_targets();
    if (!mesh_targets.empty()) {
        VisibilityChecker visibility(scene_, config_.occlusion_mode());
        auto visible_meshes = visibility.compute_visible_meshes(pupil_model_);

        for (const auto& vt : visible_meshes) {
            MeshTargetView mesh(mesh_targets[vt.target_index]);

            // Generate deterministic speckle seed from simulation RNG seed and mesh index.
            // Include a type discriminator to avoid seed collisions with plane targets.
            uint64_t speckle_seed = SpeckleField::hash_combine(
                SpeckleField::hash_combine(config_.rng_seed(), static_cast<uint64_t>(vt.target_index)),
                0x4D455348ULL);  // "MESH" as type discriminator

            // Compute response using analytic model (coherent sum of facets)
            auto response = compute_mesh_target_response(
                mesh, vt.visible_faces, chirp_plan, window, wavelength_m,
                vt.transmission, speckle_seed);

            // Add to main signal
            for (size_t i = 0; i < signal.size() && i < response.size(); ++i) {
                signal[i] += response[i];
            }
        }
    }

    return signal;
}

std::vector<double> FmcwSimulator::compute_target_response(
    const PointTargetView& target,
    const aurora::acquisition::ChirpPlan& chirp_plan,
    const MeasurementOperator::WindowParams& window,
    double wavelength_m,
    double transmission) const {

    const Eigen::Vector2d& k_vector = window.k_center;

    double sample_rate_hz = adc_model_.sample_rate_hz();
    uint32_t num_samples = ChirpGenerator::compute_num_samples(
        chirp_plan.chirp_duration_s(), sample_rate_hz);

    std::vector<double> response(num_samples);

    // Target parameters
    double range_m = target.range_m();
    double radial_velocity_mps = target.radial_velocity_mps();

    // Reflectivity at this wavelength using consistent logic with Scene
    double reflectivity = Scene::get_reflectivity(target, wavelength_m) * transmission;

    // Beat frequency from range
    double f_beat = ChirpGenerator::compute_beat_frequency(
        range_m,
        chirp_plan.chirp_bandwidth_hz(),
        chirp_plan.chirp_duration_s());

    // Doppler frequency from velocity
    double f_doppler = ChirpGenerator::compute_doppler_frequency(
        radial_velocity_mps, wavelength_m);

    // Total frequency: beat frequency + Doppler shift
    double f_total = f_beat + f_doppler;
    if (chirp_plan.direction() == aurora::acquisition::ChirpPlan::DIRECTION_DOWN) {
        f_total = f_beat - f_doppler;
    }

    // K-space phase encoding
    // The phase encodes the target's ANGULAR position.
    // φ_k = -k · x_fourier = -k · f · tan(θ)
    double focal_length = forward_model_.optics().fourier_lens_focal_m();
    Eigen::Vector3d pos = target.position();
    double theta_x = pos.x() / range_m;  // azimuth angle (small angle approx)
    double theta_y = pos.y() / range_m;  // elevation angle (small angle approx)
    double x_fourier = focal_length * theta_x;
    double y_fourier = focal_length * theta_y;
    double phi_k = -(k_vector.x() * x_fourier + k_vector.y() * y_fourier);

    // Range phase (from round-trip delay)
    double tau = 2.0 * range_m / ChirpGenerator::kSpeedOfLight;
    double f_optical = ChirpGenerator::kSpeedOfLight / wavelength_m;
    double phi_range = 2.0 * std::numbers::pi * f_optical * tau;

    // Total initial phase
    double phi_0 = phi_k + phi_range;

    // Measurement weight from the detector sensitivity type:
    // - POINT: y = X(k_center), a direct sample of the scene spectrum (weight 1)
    // - SINC:  y = ∫ W(k) X(k) dk, the spectrum integrated over the pixel's
    //          acceptance window (MeasurementOperator)
    double eff_sensitivity = 1.0;

    if (sensitivity_type() == aurora::forward::DETECTOR_SENSITIVITY_SINC) {
        // Convert target angular position to K-space position, then to the
        // focal-plane position used by the point-target window response:
        // k = 2π·r/(f·λ), so r = k·f·λ/(2π)
        Eigen::Vector2d k_target = angle_to_k(theta_x, theta_y, wavelength_m);
        Eigen::Vector2d r_target =
            k_target * focal_length * wavelength_m / (2.0 * std::numbers::pi);

        eff_sensitivity = MeasurementOperator::point_target_weight(
            r_target, window, config_.bandwidth_config().window_type(),
            static_cast<int>(config_.bandwidth_config().quadrature_order()));
    }

    // Amplitude (includes system efficiency, reflectivity, transmission, and sensitivity)
    double amplitude = reflectivity * config_.system_efficiency() * eff_sensitivity;

    // Generate samples
    // Note: This loop is embarrassingly parallel but is called from within
    // the target-parallelized loop in generate_beat_signal. To avoid excessive
    // nested parallelism overhead, only parallelize when there are many samples
    // and we're not already in a nested parallel region.
    double dt = 1.0 / sample_rate_hz;
    const int n_samples = static_cast<int>(num_samples);

#ifdef AURORA_USE_OPENMP
    // Only parallelize if enough samples to amortize thread overhead
    // and not already in a parallel region (omp_in_parallel is OpenMP 2.0 compatible)
    constexpr int kSampleParallelThreshold = 4096;
    if (n_samples >= kSampleParallelThreshold && !omp_in_parallel()) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n_samples; ++i) {
            double t = static_cast<double>(i) * dt;
            double phase = 2.0 * std::numbers::pi * f_total * t + phi_0;
            response[i] = amplitude * std::cos(phase);
        }
    } else {
        for (int i = 0; i < n_samples; ++i) {
            double t = static_cast<double>(i) * dt;
            double phase = 2.0 * std::numbers::pi * f_total * t + phi_0;
            response[i] = amplitude * std::cos(phase);
        }
    }
#else
    for (int i = 0; i < n_samples; ++i) {
        double t = static_cast<double>(i) * dt;
        double phase = 2.0 * std::numbers::pi * f_total * t + phi_0;
        response[i] = amplitude * std::cos(phase);
    }
#endif

    return response;
}

std::vector<double> FmcwSimulator::compute_plane_target_response(
    const PlaneTargetView& plane,
    const aurora::acquisition::ChirpPlan& chirp_plan,
    const MeasurementOperator::WindowParams& window,
    double wavelength_m,
    double amplitude_scale,
    uint64_t speckle_seed) const {

    const Eigen::Vector2d& k_vector = window.k_center;

    double sample_rate_hz = adc_model_.sample_rate_hz();
    uint32_t num_samples = ChirpGenerator::compute_num_samples(
        chirp_plan.chirp_duration_s(), sample_rate_hz);

    std::vector<double> response(num_samples);

    double focal_length = forward_model_.optics().fourier_lens_focal_m();
    double range_m = plane.range_m();

    // Get plane target properties for the scene spectrum
    double reflectivity = plane.surface().intrinsic_reflectivity(wavelength_m);
    double width_m = plane.width_m();
    double height_m = plane.height_m();
    Eigen::Vector3d center = plane.center();
    double area = width_m * height_m;

    // Geometric scaling factor for K-space
    double k_scale = focal_length / range_m;

    // Define scene spectrum X(k) for Fraunhofer diffraction (specular only).
    // The full response is: X(k) = ρ·A·sinc(kx'·W/2)·sinc(ky'·H/2)·exp(-i k'·r₀)
    // where k' = k * (f/R) is the apparent K-vector.
    //
    // Speckle is applied as a multiplicative factor AFTER integration,
    // since it represents a surface-scale coherence effect.
    auto X_k = [&](const Eigen::Vector2d& k) -> std::complex<double> {
        // Apparent K-vector accounting for range
        Eigen::Vector2d k_apparent = k * k_scale;

        // Sinc envelope from target geometry
        double sinc_x = safe_sinc(k_apparent.x() * width_m / 2.0);
        double sinc_y = safe_sinc(k_apparent.y() * height_m / 2.0);

        // Phase from target position: φ = -k' · r₀
        double phase = -(k_apparent.x() * center.x() + k_apparent.y() * center.y());

        return reflectivity * area * sinc_x * sinc_y *
               std::exp(std::complex<double>(0, phase));
    };

    // Measurement from the detector sensitivity type:
    // - POINT: y = X(k_center), a direct sample at the pixel's scene K
    // - SINC:  y = ∫ W(k) X(k) dk, integrated over the pixel's acceptance
    //          window with N×N midpoint quadrature
    std::complex<double> integrated_response;
    const auto sensitivity = sensitivity_type();

    if (sensitivity == aurora::forward::DETECTOR_SENSITIVITY_SINC) {
        integrated_response = MeasurementOperator::integrate_response(
            X_k, window, config_.bandwidth_config().window_type(),
            static_cast<int>(config_.bandwidth_config().quadrature_order()));
    } else if (sensitivity == aurora::forward::DETECTOR_SENSITIVITY_POINT) {
        integrated_response = X_k(k_vector);
    } else {
        throw std::runtime_error(
            "FmcwSimulator: unsupported detector sensitivity type.");
    }

    // Apply speckle as a multiplicative factor on the integrated response.
    // Speckle represents surface roughness effects that coherently modify
    // the entire integrated response. Use the pixel's scene K as the
    // reference point for spatially correlated speckle sampling.
    double specular_fraction = plane.surface().specular_fraction(wavelength_m);

    std::complex<double> k_response;
    if (specular_fraction >= config_.specular_threshold()) {
        k_response = integrated_response;
    } else {
        // Apply speckle mixing: E = α·E_spec + √(1-α²)·E_spec·ξ
        double correlation_length = plane.proto().surface().correlation_length_m();
        Eigen::Vector2d k_apparent_eff = k_vector * k_scale;
        std::complex<double> xi = SpeckleField::sample(
            k_apparent_eff, correlation_length, speckle_seed);

        double alpha = specular_fraction;
        double diffuse_weight = std::sqrt(1.0 - alpha * alpha);
        k_response = alpha * integrated_response + diffuse_weight * integrated_response * xi;
    }

    // Extract amplitude and phase from K-space response
    double k_amplitude = std::abs(k_response);
    double k_phase = std::arg(k_response);

    // If response is negligible, return zeros
    if (k_amplitude < 1e-15) {
        return response;
    }

    // Compute radial velocity from plane velocity
    Eigen::Vector3d pos = plane.center();
    Eigen::Vector3d vel = plane.velocity();
    double range = pos.norm();
    double radial_velocity_mps = 0.0;
    if (range > 1e-10) {
        Eigen::Vector3d unit_to_target = pos / range;
        radial_velocity_mps = vel.dot(unit_to_target);
    }

    // Beat frequency from range
    double f_beat = ChirpGenerator::compute_beat_frequency(
        range_m,
        chirp_plan.chirp_bandwidth_hz(),
        chirp_plan.chirp_duration_s());

    // Doppler frequency from velocity
    double f_doppler = ChirpGenerator::compute_doppler_frequency(
        radial_velocity_mps, wavelength_m);

    // Total frequency: beat frequency + Doppler shift
    double f_total = f_beat + f_doppler;
    if (chirp_plan.direction() == aurora::acquisition::ChirpPlan::DIRECTION_DOWN) {
        f_total = f_beat - f_doppler;
    }

    // Range phase (from round-trip delay)
    double tau = 2.0 * range_m / ChirpGenerator::kSpeedOfLight;
    double f_optical = ChirpGenerator::kSpeedOfLight / wavelength_m;
    double phi_range = 2.0 * std::numbers::pi * f_optical * tau;

    // Total initial phase: K-space phase (from integrated response) + range phase
    double phi_0 = k_phase + phi_range;

    // Amplitude (integrated response already includes reflectivity and sinc effects)
    // Note: No separate eff_sensitivity factor needed - it's incorporated in integrate_response
    double amplitude = k_amplitude * amplitude_scale * config_.system_efficiency();

    // Generate samples
    double dt = 1.0 / sample_rate_hz;
    const int n_samples = static_cast<int>(num_samples);

    for (int i = 0; i < n_samples; ++i) {
        double t = static_cast<double>(i) * dt;
        double phase = 2.0 * std::numbers::pi * f_total * t + phi_0;
        response[i] = amplitude * std::cos(phase);
    }

    return response;
}

std::vector<double> FmcwSimulator::compute_mesh_target_response(
    const MeshTargetView& mesh,
    const std::vector<size_t>& visible_faces,
    const aurora::acquisition::ChirpPlan& chirp_plan,
    const MeasurementOperator::WindowParams& window,
    double wavelength_m,
    double amplitude_scale,
    uint64_t speckle_seed) const {

    const Eigen::Vector2d& k_vector = window.k_center;

    double sample_rate_hz = adc_model_.sample_rate_hz();
    uint32_t num_samples = ChirpGenerator::compute_num_samples(
        chirp_plan.chirp_duration_s(), sample_rate_hz);

    std::vector<double> response(num_samples);

    if (visible_faces.empty()) {
        return response;
    }

    double focal_length = forward_model_.optics().fourier_lens_focal_m();
    double range_m = mesh.range_m();

    // Geometric scaling factor for K-space
    double k_scale = focal_length / range_m;

    // Measurement model selection (POINT sample vs. SINC window integration)
    const auto sensitivity = sensitivity_type();

    // Integrate the response of each visible facet and sum coherently.
    // Each facet contributes a Fraunhofer envelope weighted by the pixel's
    // acceptance window. Speckle is applied per facet after integration.
    std::complex<double> total_response{0.0, 0.0};

    for (size_t face_idx : visible_faces) {
        Eigen::Vector3d centroid = mesh.face_centroid(face_idx);
        SurfacePropertiesView face_surface = mesh.face_surface(face_idx);
        double reflectivity = face_surface.intrinsic_reflectivity(wavelength_m);
        double area = mesh.face_area(face_idx);

        // Per-facet speckle seed
        uint64_t facet_seed = SpeckleField::hash_combine(speckle_seed, face_idx);

        // Define scene spectrum X(k) for this facet's Fraunhofer envelope.
        // Uses equivalent circular aperture approximation for triangles,
        // and oriented rectangular aperture for quads.
        std::function<std::complex<double>(const Eigen::Vector2d&)> X_k_facet;

        if (mesh.is_triangle(face_idx)) {
            // Triangle: use equivalent circular aperture (Airy envelope)
            double r_equiv = std::sqrt(area / std::numbers::pi);

            X_k_facet = [=](const Eigen::Vector2d& k) -> std::complex<double> {
                Eigen::Vector2d k_apparent = k * k_scale;
                double k_mag = k_apparent.norm();

                // Airy-like shape factor: 2*J1(kr)/kr
                double shape_factor;
                double kr = k_mag * r_equiv;
                if (kr < 1e-10) {
                    // Limit kr -> 0 of 2*J1(kr)/kr is 1
                    shape_factor = 1.0;
                } else {
                    // Use C++17 standard library Bessel function for accuracy
                    double j1 = std::cyl_bessel_j(1, kr);
                    shape_factor = 2.0 * j1 / kr;
                }

                // Phase from centroid position
                double phase = -(k_apparent.x() * centroid.x() +
                                k_apparent.y() * centroid.y());

                return reflectivity * area * shape_factor *
                       std::exp(std::complex<double>(0, phase));
            };
        } else {
            // Quad: use oriented rectangular aperture
            // Get quad geometry (width, height, local axes)
            const auto& proto = mesh.proto();
            const auto& face = proto.faces(static_cast<int>(face_idx));

            Eigen::Vector3d v0 = mesh.vertex_position(face.vertex_indices(0));
            Eigen::Vector3d v1 = mesh.vertex_position(face.vertex_indices(1));
            Eigen::Vector3d v2 = mesh.vertex_position(face.vertex_indices(2));
            Eigen::Vector3d v3 = mesh.vertex_position(face.vertex_indices(3));

            // Compute edge dimensions for sinc envelope shape
            double width = ((v1 - v0).norm() + (v2 - v3).norm()) / 2.0;
            double height = ((v2 - v1).norm() + (v3 - v0).norm()) / 2.0;

            // Local axes projected to XY plane
            Eigen::Vector3d local_x_3d = ((v1 - v0).normalized() +
                                          (v2 - v3).normalized()).normalized();
            Eigen::Vector3d local_y_3d = ((v2 - v1).normalized() +
                                          (v3 - v0).normalized()).normalized();
            Eigen::Vector2d local_x(local_x_3d.x(), local_x_3d.y());
            Eigen::Vector2d local_y(local_y_3d.x(), local_y_3d.y());
            if (local_x.norm() > 1e-10) local_x.normalize();
            else local_x = Eigen::Vector2d(1, 0);
            if (local_y.norm() > 1e-10) local_y.normalize();
            else local_y = Eigen::Vector2d(0, 1);

            // Use actual geometric area for amplitude (handles skewed quads correctly)
            // Width/height are only used for sinc envelope shape
            double actual_area = area;

            X_k_facet = [=](const Eigen::Vector2d& k) -> std::complex<double> {
                Eigen::Vector2d k_apparent = k * k_scale;

                // Project k onto local frame for oriented sinc
                double k_local_x = k_apparent.dot(local_x);
                double k_local_y = k_apparent.dot(local_y);

                double sinc_x = safe_sinc(k_local_x * width / 2.0);
                double sinc_y = safe_sinc(k_local_y * height / 2.0);

                // Phase from centroid position
                double phase = -(k_apparent.x() * centroid.x() +
                                k_apparent.y() * centroid.y());

                return reflectivity * actual_area * sinc_x * sinc_y *
                       std::exp(std::complex<double>(0, phase));
            };
        }

        // Compute facet response based on sensitivity type
        std::complex<double> facet_integrated;
        if (sensitivity == aurora::forward::DETECTOR_SENSITIVITY_SINC) {
            // SINC sensitivity: integrate facet spectrum over the acceptance window
            facet_integrated = MeasurementOperator::integrate_response(
                X_k_facet, window, config_.bandwidth_config().window_type(),
                static_cast<int>(config_.bandwidth_config().quadrature_order()));
        } else if (sensitivity == aurora::forward::DETECTOR_SENSITIVITY_POINT) {
            // POINT sensitivity: sample facet spectrum at the pixel's scene K
            facet_integrated = X_k_facet(k_vector);
        } else {
            throw std::runtime_error(
                "FmcwSimulator: unsupported detector sensitivity type.");
        }

        // Apply per-facet speckle
        double specular_fraction = face_surface.specular_fraction(wavelength_m);

        std::complex<double> facet_response;
        if (specular_fraction >= config_.specular_threshold()) {
            facet_response = facet_integrated;
        } else {
            double correlation_length = face_surface.proto().correlation_length_m();
            Eigen::Vector2d k_apparent_eff = k_vector * k_scale;
            std::complex<double> xi = SpeckleField::sample(
                k_apparent_eff, correlation_length, facet_seed);

            double alpha = specular_fraction;
            double diffuse_weight = std::sqrt(1.0 - alpha * alpha);
            facet_response = alpha * facet_integrated +
                            diffuse_weight * facet_integrated * xi;
        }

        total_response += facet_response;
    }

    // Extract amplitude and phase from total K-space response
    double k_amplitude = std::abs(total_response);
    double k_phase = std::arg(total_response);

    // If response is negligible, return zeros
    if (k_amplitude < 1e-15) {
        return response;
    }

    // Compute radial velocity from mesh velocity
    Eigen::Vector3d centroid = mesh.centroid();
    Eigen::Vector3d vel = mesh.velocity();
    double radial_velocity_mps = 0.0;
    if (range_m > 1e-10) {
        Eigen::Vector3d unit_to_target = centroid / range_m;
        radial_velocity_mps = vel.dot(unit_to_target);
    }

    // Beat frequency from range
    double f_beat = ChirpGenerator::compute_beat_frequency(
        range_m,
        chirp_plan.chirp_bandwidth_hz(),
        chirp_plan.chirp_duration_s());

    // Doppler frequency from velocity
    double f_doppler = ChirpGenerator::compute_doppler_frequency(
        radial_velocity_mps, wavelength_m);

    // Total frequency: beat frequency + Doppler shift
    double f_total = f_beat + f_doppler;
    if (chirp_plan.direction() == aurora::acquisition::ChirpPlan::DIRECTION_DOWN) {
        f_total = f_beat - f_doppler;
    }

    // Range phase (from round-trip delay)
    double tau = 2.0 * range_m / ChirpGenerator::kSpeedOfLight;
    double f_optical = ChirpGenerator::kSpeedOfLight / wavelength_m;
    double phi_range = 2.0 * std::numbers::pi * f_optical * tau;

    // Total initial phase: K-space phase (from integrated response) + range phase
    double phi_0 = k_phase + phi_range;

    // Amplitude (integrated response already includes facet contributions)
    // Note: No separate eff_sensitivity factor needed - it's incorporated in integrate_response
    double amplitude = k_amplitude * amplitude_scale * config_.system_efficiency();

    // Generate samples
    double dt = 1.0 / sample_rate_hz;
    const int n_samples = static_cast<int>(num_samples);

    for (int i = 0; i < n_samples; ++i) {
        double t = static_cast<double>(i) * dt;
        double phase = 2.0 * std::numbers::pi * f_total * t + phi_0;
        response[i] = amplitude * std::cos(phase);
    }

    return response;
}

aurora::forward::DetectorSensitivityType FmcwSimulator::sensitivity_type() const {
    const auto& detector = forward_model_.detector();
    if (detector.arrays_size() == 0) {
        throw std::runtime_error(
            "FmcwSimulator requires at least one detector array.");
    }
    return detector.arrays(0).sensitivity_type();
}

std::vector<int16_t> FmcwSimulator::apply_noise_and_adc(
    const std::vector<double>& signal,
    double optical_power_w,
    std::mt19937_64& rng) const {

    // Apply noise with provided RNG
    double bandwidth_hz = adc_model_.sample_rate_hz() / 2.0;  // Nyquist bandwidth
    auto noisy_signal = noise_model_.add_noise(
        signal, optical_power_w, bandwidth_hz, rng);

    // Apply ADC with provided RNG
    return adc_model_.sample(noisy_signal, adc_model_.sample_rate_hz(), rng);
}

}  // namespace aurora::simulation
