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

/**
 * @file detection_example.cpp
 * @brief End-to-end detection stream example.
 *
 * Scene -> FMCW simulation -> range/Doppler DSP -> MeasurementTensor -> Detector.
 *
 * Point targets are placed at known ranges, azimuths and radial velocities.
 * Each of the 8 detector pixels samples the scene at its own spatial
 * frequency, k_j = 2π·x_j/(f·λ); the Detector recovers azimuth from the
 * phase gradient across those samples, range from the beat frequency, and
 * velocity from the up/down chirp pair.
 *
 * Build and run:
 *   cmake --build build --target detection_example
 *   ./build/docs/examples/detection_example
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "acquisition/sequence_planner.h"
#include "detection/detector.h"
#include "dsp/range_doppler_processor.h"
#include "forward/forward_model.h"
#include "simulation/fmcw_simulator.h"

#include "detection_config.pb.h"
#include "forward_config.pb.h"
#include "sequence_schedule.pb.h"
#include "sim_config.pb.h"

namespace {

constexpr double kTwoPi = 2.0 * M_PI;

// Radar parameters
constexpr double kSampleRate = 125e6;        // Hz
constexpr double kChirpBandwidth = 200e6;    // Hz -> 0.75 m range resolution
constexpr double kChirpDuration = 12.5e-6;   // s
constexpr double kWavelength = 1550e-9;      // m
constexpr uint32_t kFftSize = 2048;
constexpr double kBurstPeriod = 100e-6;      // s

// Receiver optics
constexpr double kFocalLength = 0.2;         // m
constexpr double kDetectorPitch = 0.5e-3;    // m
constexpr uint32_t kNumDetectors = 8;
constexpr double kFieldOfView = 0.1;         // rad (full), searched for targets

struct Target {
    double range_m;
    double azimuth_rad;
    double radial_velocity_mps;
    double reflectivity;
};

/// Unambiguous azimuth span of the uniformly spaced array: 2π / (f·Δk).
double azimuth_period() {
    double dk = kTwoPi * kDetectorPitch / (kFocalLength * kWavelength);
    return kTwoPi / (kFocalLength * dk);
}

/// Detector array and optics: the K-space sampling follows from these alone.
aurora::forward::ForwardConfig create_forward_config() {
    aurora::forward::ForwardConfig config;

    auto* array = config.mutable_detector()->add_arrays();
    array->set_array_id("ingaas_8px");
    array->set_nominal_pitch_m(kDetectorPitch);
    array->set_bandwidth_hz(150e6);
    array->add_wavelengths_m(kWavelength);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_POINT);
    for (uint32_t i = 0; i < kNumDetectors; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((static_cast<double>(i) - (kNumDetectors - 1) / 2.0) * kDetectorPitch);
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.8 * kDetectorPitch);
        pixel->set_height_m(0.8 * kDetectorPitch);
    }

    auto* optics = config.mutable_optics();
    optics->set_fourier_lens_focal_m(kFocalLength);
    optics->set_fourier_lens_diameter_m(0.05);
    optics->add_wavelengths_m(kWavelength);
    optics->set_system_fov_x_rad(kFieldOfView);
    optics->set_system_fov_y_rad(kFieldOfView);
    auto* pupil = optics->mutable_pupil();
    pupil->set_fov_half_angle_rad(0.5);
    pupil->set_rolloff_width_rad(0.05);
    return config;
}

aurora::simulation::SimConfig create_sim_config(const std::vector<Target>& targets) {
    aurora::simulation::SimConfig config;

    auto* adc = config.mutable_adc();
    adc->set_sample_rate_hz(kSampleRate);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    auto* noise = config.mutable_noise();
    noise->set_enable_shot_noise(true);
    noise->set_enable_thermal_noise(true);
    noise->set_enable_quantization_noise(true);
    noise->set_detector_responsivity_a_per_w(1.0);
    noise->set_transimpedance_ohm(1e3);
    noise->set_temperature_k(300.0);
    noise->set_noise_bandwidth_hz(kSampleRate / 2.0);

    config.set_tx_power_w(1.0);
    config.set_lo_power_w(1.0);
    config.set_system_efficiency(1.0);
    config.set_rng_seed(42);
    config.set_occlusion_mode(aurora::simulation::OCCLUSION_NONE);
    config.set_specular_threshold(1.0 - 1e-10);

    auto* bw = config.mutable_bandwidth_config();
    bw->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    bw->set_quadrature_order(11);

    for (const auto& t : targets) {
        auto* point = config.mutable_scene()->add_point_targets();
        double x = t.range_m * std::sin(t.azimuth_rad);
        double z = t.range_m * std::cos(t.azimuth_rad);
        point->mutable_position_m()->set_x(x);
        point->mutable_position_m()->set_y(0.0);
        point->mutable_position_m()->set_z(z);
        point->mutable_velocity_mps()->set_x(t.radial_velocity_mps * x / t.range_m);
        point->mutable_velocity_mps()->set_y(0.0);
        point->mutable_velocity_mps()->set_z(t.radial_velocity_mps * z / t.range_m);
        point->mutable_surface()->set_reflectivity_1550nm(t.reflectivity);
        point->mutable_surface()->set_reflectivity_905nm(t.reflectivity);
        point->mutable_surface()->set_roughness_rms_m(1e-6);
        point->mutable_surface()->set_correlation_length_m(10e-6);
    }
    return config;
}

aurora::dsp::DspConfig create_dsp_config() {
    aurora::dsp::DspConfig config;
    config.set_range_fft_size(kFftSize);
    config.mutable_range_window()->set_type(aurora::dsp::WindowConfig::WINDOW_HANN);
    config.set_enable_dc_removal(true);

    auto* adc = config.mutable_adc_config();
    adc->set_sample_rate_hz(kSampleRate);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    auto* iq = config.mutable_iq_demod_config();
    iq->set_lpf_cutoff_hz(10e6);
    iq->set_lpf_order(4);
    return config;
}

aurora::detection::DetectionConfig create_detection_config() {
    aurora::detection::DetectionConfig config;
    config.set_min_pixels_for_gradient(3);
    config.set_max_phase_residual_rad(0.8);
    config.set_max_condition_number(1e12);
    config.set_enable_doppler(true);
    config.set_min_velocity_mps(0.05);
    config.set_noise_floor_db(0.0);

    auto* cfar = config.mutable_cfar();
    cfar->set_mode(aurora::detection::CFAR_MODE_CA);
    cfar->set_guard_cells(4);
    cfar->set_reference_cells(16);
    cfar->set_threshold_factor(20.0);
    cfar->set_min_threshold(1e-12);
    cfar->set_skip_bins(4);
    return config;
}

aurora::acquisition::SequenceSchedule create_sequence_schedule() {
    aurora::acquisition::SequenceSchedule schedule;
    schedule.set_burst_period_s(kBurstPeriod);
    auto* chirp = schedule.mutable_chirp_schedule();
    chirp->set_chirp_duration_s(kChirpDuration);
    chirp->set_chirp_bandwidth_hz(kChirpBandwidth);
    chirp->set_chirps_per_burst(2);
    chirp->set_mode(aurora::acquisition::ChirpSchedule::CHIRP_MODE_UP_DOWN);
    auto* wavelength = schedule.mutable_wavelength_schedule();
    wavelength->set_mode(aurora::acquisition::WavelengthSchedule::INTERLEAVING_SIMULTANEOUS);
    wavelength->add_wavelengths_m(kWavelength);
    return schedule;
}

}  // namespace

/// Range and velocity of fast targets by pairing up- and down-chirp peaks.
void print_peak_doppler(aurora::dsp::RangeDopplerProcessor& processor,
                        const aurora::simulation::Burst& burst) {
    aurora::dsp::PeakDopplerConfig peak_config;
    peak_config.set_peak_threshold_db(10.0);
    peak_config.set_min_peak_separation(3);
    peak_config.set_max_velocity_mps(30.0);
    peak_config.set_association_threshold(0.3);
    peak_config.set_allow_unpaired_peaks(false);
    processor.set_peak_doppler_config(peak_config);

    constexpr double kMinPeakSnrDb = 20.0;  // suppress noise-only pairings

    auto results = processor.process_doppler_peaks(burst, kWavelength);
    for (const auto& result : results) {
        if (result.detector_index() != kNumDetectors / 2) continue;  // one pixel is enough
        for (const auto& peak : result.peaks()) {
            if (peak.snr_db() < kMinPeakSnrDb) continue;
            std::cout << "  range " << std::setw(7) << peak.range_m() << " m, velocity "
                      << std::setw(6) << peak.velocity_mps() << " m/s, SNR "
                      << std::setprecision(1) << peak.snr_db() << " dB"
                      << std::setprecision(3) << "\n";
        }
    }
}

int main() {
    const double period = azimuth_period();

    // Slow targets: the Detector resolves range, azimuth and velocity from one
    // burst. Its phase-based Doppler is unambiguous for |v| < λ/(4T).
    const std::vector<Target> slow_targets = {
        {30.0, 0.30 * period, 0.020, 0.5},
        {52.0, -0.20 * period, -0.015, 0.3},
    };
    // Fast targets: at optical wavelengths a few m/s shifts the beat frequency
    // by MHz, so range and velocity come from pairing up/down chirp peaks.
    const std::vector<Target> fast_targets = {
        {30.0, 0.0, 1.0, 0.5},
        {52.0, 0.0, -1.5, 0.3},
    };

    std::cout << "Aurora detection example\n"
              << "  " << kNumDetectors << " detectors, up/down chirps, "
              << kChirpBandwidth / 1e6 << " MHz bandwidth\n"
              << "  unambiguous azimuth span: +/- " << std::fixed << std::setprecision(2)
              << period / 2.0 * 1e3 << " mrad\n\n";

    auto forward_config = create_forward_config();
    aurora::forward::ForwardModel forward_model(forward_config);

    auto schedule = create_sequence_schedule();
    aurora::acquisition::SequencePlanner planner(schedule);
    auto plan = planner.plan_burst(0, 0.0);

    aurora::dsp::RangeDopplerProcessor processor(create_dsp_config(), forward_model);
    aurora::detection::Detector detector(create_detection_config(), forward_config, schedule);

    // --- Slow targets through the full detection stream ---
    aurora::simulation::FmcwSimulator simulator(forward_model, create_sim_config(slow_targets));
    auto burst = simulator.simulate_burst(plan);
    auto tensor = processor.to_measurement_tensor(processor.process_doppler(burst));
    auto frame = detector.detect(tensor);

    std::cout << std::setprecision(3);
    std::cout << "Slow targets (truth):\n";
    for (const auto& t : slow_targets) {
        std::cout << "  range " << std::setw(7) << t.range_m << " m, azimuth "
                  << std::setw(7) << t.azimuth_rad * 1e3 << " mrad, velocity "
                  << std::setw(6) << t.radial_velocity_mps << " m/s\n";
    }
    std::cout << "Detections (" << frame.detections_size() << ", "
              << frame.processing_time_ms() << " ms):\n";
    for (const auto& det : frame.detections()) {
        std::cout << "  range " << std::setw(7) << det.range_m() << " m, azimuth "
                  << std::setw(7) << det.azimuth_rad() * 1e3 << " mrad"
                  << " (+/- " << det.azimuth_uncertainty_rad() * 1e3 << " mrad)";
        if (det.has_doppler()) {
            std::cout << ", velocity " << std::setw(6) << det.doppler().velocity_mps() << " m/s";
        }
        std::cout << ", SNR " << std::setprecision(1) << det.snr_db() << " dB"
                  << std::setprecision(3) << "\n";
    }

    // --- Fast targets through up/down peak pairing ---
    aurora::simulation::FmcwSimulator fast_simulator(forward_model,
                                                     create_sim_config(fast_targets));
    auto fast_burst = fast_simulator.simulate_burst(plan);

    std::cout << "\nFast targets (truth):\n";
    for (const auto& t : fast_targets) {
        std::cout << "  range " << std::setw(7) << t.range_m << " m, velocity "
                  << std::setw(6) << t.radial_velocity_mps << " m/s\n";
    }
    std::cout << "Paired up/down chirp peaks (center pixel):\n";
    print_peak_doppler(processor, fast_burst);
    return 0;
}
