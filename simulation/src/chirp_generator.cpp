#include "simulation/chirp_generator.h"

#include <cmath>

namespace aurora::simulation {

std::vector<double> ChirpGenerator::generate_phase(
    const aurora::acquisition::ChirpPlan& chirp_plan,
    double sample_rate_hz) {

    uint32_t num_samples = compute_num_samples(chirp_plan.chirp_duration_s(),
                                                sample_rate_hz);
    std::vector<double> phase(num_samples);

    double dt = 1.0 / sample_rate_hz;
    double T = chirp_plan.chirp_duration_s();
    double B = chirp_plan.chirp_bandwidth_hz();

    // Chirp rate (Hz/s)
    double chirp_rate = B / T;

    // Direction: down-chirp negates the chirp rate
    if (chirp_plan.direction() == aurora::acquisition::ChirpPlan::DIRECTION_DOWN) {
        chirp_rate = -chirp_rate;
    }

    // Generate phase samples
    // φ(t) = 2π * (f_0 * t + (chirp_rate / 2) * t²)
    // We only need the chirp phase, not the carrier (f_0 term), since
    // we're computing beat signals relative to the LO
    for (uint32_t i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) * dt;
        // Only the quadratic term (the chirp) matters for beat frequency
        double phi = M_PI * chirp_rate * t * t;
        // Wrap to [-π, π] for numerical stability
        phase[i] = std::fmod(phi, 2.0 * M_PI);
        if (phase[i] > M_PI) {
            phase[i] -= 2.0 * M_PI;
        }
    }

    return phase;
}

std::vector<double> ChirpGenerator::generate_waveform(
    const aurora::acquisition::ChirpPlan& chirp_plan,
    double sample_rate_hz) {

    auto phase = generate_phase(chirp_plan, sample_rate_hz);
    std::vector<double> waveform(phase.size());

    for (size_t i = 0; i < phase.size(); ++i) {
        waveform[i] = std::cos(phase[i]);
    }

    return waveform;
}

double ChirpGenerator::compute_beat_frequency(double range_m,
                                               double bandwidth_hz,
                                               double duration_s) {
    // f_beat = (2 * B * R) / (c * T)
    return (2.0 * bandwidth_hz * range_m) / (kSpeedOfLight * duration_s);
}

double ChirpGenerator::compute_range(double beat_frequency_hz,
                                      double bandwidth_hz,
                                      double duration_s) {
    // R = (c * T * f_beat) / (2 * B)
    return (kSpeedOfLight * duration_s * beat_frequency_hz) / (2.0 * bandwidth_hz);
}

double ChirpGenerator::compute_doppler_frequency(double velocity_mps,
                                                  double wavelength_m) {
    // f_d = 2 * v / λ
    return (2.0 * velocity_mps) / wavelength_m;
}

uint32_t ChirpGenerator::compute_num_samples(double duration_s,
                                              double sample_rate_hz) {
    return static_cast<uint32_t>(std::ceil(duration_s * sample_rate_hz));
}

}  // namespace aurora::simulation
