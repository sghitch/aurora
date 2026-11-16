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
 * @file test_detection_integration.cpp
 * @brief End-to-end detection stream test: Scene -> FMCW simulation -> DSP ->
 *        MeasurementTensor -> Detector.
 *
 * Verifies that a point target placed in the scene is recovered with the
 * correct range, azimuth and radial velocity by the detection pipeline,
 * using only the public module contracts.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "detection/detector.h"
#include "dsp/range_doppler_processor.h"
#include "forward/forward_model.h"
#include "simulation/fmcw_simulator.h"

#include "detection_config.pb.h"
#include "forward_config.pb.h"
#include "sequence_plan.pb.h"
#include "sequence_schedule.pb.h"

namespace aurora::integration {
namespace {

constexpr double kTwoPi = 2.0 * M_PI;
constexpr double kSpeedOfLight = 299792458.0;

constexpr double kSampleRate = 125e6;         // Hz
constexpr double kChirpBandwidth = 200e6;     // Hz
constexpr double kChirpDuration = 12.5e-6;    // s
constexpr double kWavelength = 1550e-9;       // m
constexpr double kFocalLength = 0.2;          // m
constexpr double kDetectorPitch = 0.5e-3;     // m
constexpr uint32_t kNumDetectors = 8;
constexpr uint32_t kFftSize = 2048;
constexpr double kFovRad = 0.1;               // full field of view searched

/// Detector K-spacing at the operating wavelength: Δk = 2π·pitch/(f·λ).
double detector_k_step() {
    return kTwoPi * kDetectorPitch / (kFocalLength * kWavelength);
}

/// Unambiguous azimuth period of the uniformly spaced array: 2π/(f·Δk).
double azimuth_period() {
    return kTwoPi / (kFocalLength * detector_k_step());
}

/// Range represented by one FFT bin: (c·T·f_s) / (2·B·N_fft).
double range_per_bin() {
    return (kSpeedOfLight * kChirpDuration * kSampleRate) /
           (2.0 * kChirpBandwidth * kFftSize);
}

aurora::forward::ForwardConfig create_forward_config() {
    aurora::forward::ForwardConfig config;

    auto* array = config.mutable_detector()->add_arrays();
    array->set_array_id("main");
    array->set_nominal_pitch_m(kDetectorPitch);
    array->set_bandwidth_hz(150e6);
    array->add_wavelengths_m(kWavelength);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_POINT);
    for (uint32_t i = 0; i < kNumDetectors; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((static_cast<double>(i) - 3.5) * kDetectorPitch);
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.4e-3);
        pixel->set_height_m(0.4e-3);
    }

    auto* optics = config.mutable_optics();
    optics->set_fourier_lens_focal_m(kFocalLength);
    optics->set_fourier_lens_diameter_m(0.05);
    optics->add_wavelengths_m(kWavelength);
    optics->set_system_fov_x_rad(kFovRad);
    optics->set_system_fov_y_rad(kFovRad);
    auto* pupil = optics->mutable_pupil();
    pupil->set_fov_half_angle_rad(0.5);
    pupil->set_rolloff_width_rad(0.05);

    return config;
}

simulation::SimConfig create_sim_config(double target_x_m,
                                        double target_z_m,
                                        double radial_velocity_mps = 0.0) {
    simulation::SimConfig config;

    auto* adc = config.mutable_adc();
    adc->set_sample_rate_hz(kSampleRate);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    auto* noise = config.mutable_noise();
    noise->set_enable_shot_noise(false);
    noise->set_enable_thermal_noise(false);
    noise->set_enable_rin_noise(false);
    noise->set_enable_quantization_noise(false);
    noise->set_enable_dark_current(false);

    config.set_tx_power_w(1.0);
    config.set_lo_power_w(1.0);
    config.set_system_efficiency(1.0);
    config.set_rng_seed(42);
    config.set_occlusion_mode(simulation::OCCLUSION_NONE);
    config.set_specular_threshold(1.0 - 1e-10);

    auto* bw = config.mutable_bandwidth_config();
    bw->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    bw->set_quadrature_order(11);

    auto* target = config.mutable_scene()->add_point_targets();
    target->mutable_position_m()->set_x(target_x_m);
    target->mutable_position_m()->set_y(0.0);
    target->mutable_position_m()->set_z(target_z_m);
    // Velocity along the line of sight (positive = receding)
    double range = std::hypot(target_x_m, target_z_m);
    target->mutable_velocity_mps()->set_x(radial_velocity_mps * target_x_m / range);
    target->mutable_velocity_mps()->set_y(0.0);
    target->mutable_velocity_mps()->set_z(radial_velocity_mps * target_z_m / range);
    target->mutable_surface()->set_reflectivity_1550nm(0.5);
    target->mutable_surface()->set_reflectivity_905nm(0.5);
    target->mutable_surface()->set_roughness_rms_m(1e-6);
    target->mutable_surface()->set_correlation_length_m(10e-6);

    return config;
}

dsp::DspConfig create_dsp_config() {
    dsp::DspConfig config;
    config.set_range_fft_size(kFftSize);
    config.mutable_range_window()->set_type(dsp::WindowConfig::WINDOW_HANN);
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

detection::DetectionConfig create_detection_config(bool enable_doppler) {
    detection::DetectionConfig config;
    config.set_min_pixels_for_gradient(3);
    config.set_max_phase_residual_rad(0.5);
    config.set_max_condition_number(1e12);
    config.set_enable_doppler(enable_doppler);
    config.set_min_velocity_mps(0.05);
    config.set_noise_floor_db(0.0);

    auto* cfar = config.mutable_cfar();
    cfar->set_mode(detection::CFAR_MODE_CA);
    cfar->set_guard_cells(4);
    cfar->set_reference_cells(16);
    cfar->set_threshold_factor(20.0);
    cfar->set_min_threshold(1e-12);
    cfar->set_skip_bins(4);
    return config;
}

aurora::acquisition::SequenceSchedule create_sequence_schedule(bool up_down) {
    aurora::acquisition::SequenceSchedule schedule;
    schedule.set_burst_period_s(100e-6);
    auto* chirp = schedule.mutable_chirp_schedule();
    chirp->set_chirp_duration_s(kChirpDuration);
    chirp->set_chirp_bandwidth_hz(kChirpBandwidth);
    chirp->set_chirps_per_burst(up_down ? 2 : 1);
    chirp->set_mode(up_down ? aurora::acquisition::ChirpSchedule::CHIRP_MODE_UP_DOWN
                            : aurora::acquisition::ChirpSchedule::CHIRP_MODE_UP_ONLY);
    return schedule;
}

aurora::acquisition::SequencePlan create_plan(bool up_down) {
    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(1);
    plan.set_start_timestamp_s(0.0);

    const int num_chirps = up_down ? 2 : 1;
    for (int c = 0; c < num_chirps; ++c) {
        auto* chirp = plan.add_chirps();
        chirp->set_chirp_index(c);
        chirp->set_chirp_start_s(c * kChirpDuration);
        chirp->set_chirp_duration_s(kChirpDuration);
        chirp->set_chirp_bandwidth_hz(kChirpBandwidth);
        chirp->set_direction(c == 0 ? aurora::acquisition::ChirpPlan::DIRECTION_UP
                                    : aurora::acquisition::ChirpPlan::DIRECTION_DOWN);
        chirp->add_wavelength_m(kWavelength);
    }
    return plan;
}

detection::DetectionFrame run_pipeline(double target_x_m,
                                       double target_z_m,
                                       double radial_velocity_mps = 0.0,
                                       bool up_down = false) {
    auto forward_config = create_forward_config();
    aurora::forward::ForwardModel forward_model(forward_config);

    simulation::FmcwSimulator simulator(
        forward_model, create_sim_config(target_x_m, target_z_m, radial_velocity_mps));
    auto burst = simulator.simulate_burst(create_plan(up_down));

    dsp::RangeDopplerProcessor processor(create_dsp_config(), forward_model);
    auto cube = up_down ? processor.process_doppler(burst) : processor.process_range(burst);
    auto tensor = processor.to_measurement_tensor(cube);

    detection::Detector detector(create_detection_config(up_down), forward_config,
                                 create_sequence_schedule(up_down));
    return detector.detect(tensor);
}

/// Find the detection whose range is closest to the expected range.
const detection::Detection* closest_detection(const detection::DetectionFrame& frame,
                                              double expected_range_m) {
    const detection::Detection* best = nullptr;
    double best_error = std::numeric_limits<double>::infinity();
    for (const auto& det : frame.detections()) {
        double error = std::abs(det.range_m() - expected_range_m);
        if (error < best_error) {
            best_error = error;
            best = &det;
        }
    }
    return best;
}

}  // namespace

TEST(DetectionIntegrationTest, PointTargetRangeAndAzimuth) {
    const double range_m = 30.0;
    const double azimuth_rad = 0.3 * azimuth_period();  // inside the unambiguous range
    const double x_m = range_m * std::tan(azimuth_rad);
    const double true_range = std::hypot(x_m, range_m);

    auto frame = run_pipeline(x_m, range_m);
    ASSERT_GT(frame.detections_size(), 0) << "No detections produced";

    const auto* det = closest_detection(frame, true_range);
    ASSERT_NE(det, nullptr);
    EXPECT_NEAR(det->range_m(), true_range, 2.0 * range_per_bin());
    EXPECT_NEAR(det->azimuth_rad(), azimuth_rad, 2.0e-4);
    EXPECT_TRUE(std::isfinite(det->azimuth_uncertainty_rad()));
    EXPECT_LT(det->phase_residual_rad(), 0.5);
    EXPECT_EQ(det->detector_indices_size(), static_cast<int>(kNumDetectors));
}

TEST(DetectionIntegrationTest, NegativeAzimuthIsSigned) {
    const double range_m = 45.0;
    const double azimuth_rad = -0.25 * azimuth_period();
    const double x_m = range_m * std::tan(azimuth_rad);

    auto frame = run_pipeline(x_m, range_m);
    ASSERT_GT(frame.detections_size(), 0);

    const auto* det = closest_detection(frame, std::hypot(x_m, range_m));
    ASSERT_NE(det, nullptr);
    EXPECT_NEAR(det->azimuth_rad(), azimuth_rad, 2.0e-4);
    EXPECT_LT(det->position_m().x(), 0.0);
}

TEST(DetectionIntegrationTest, BoresightTarget) {
    auto frame = run_pipeline(0.0, 20.0);
    ASSERT_GT(frame.detections_size(), 0);

    const auto* det = closest_detection(frame, 20.0);
    ASSERT_NE(det, nullptr);
    EXPECT_NEAR(det->azimuth_rad(), 0.0, 2.0e-4);
    EXPECT_NEAR(det->range_m(), 20.0, 2.0 * range_per_bin());
}

TEST(DetectionIntegrationTest, SlowTargetVelocityFromPhase) {
    // The Detector's phase-based Doppler is unambiguous for |v| < λ/(4T),
    // about 0.03 m/s with these chirps.
    const double range_m = 30.0;
    const double velocity = 0.02;  // m/s receding

    auto frame = run_pipeline(0.0, range_m, velocity, /*up_down=*/true);
    ASSERT_GT(frame.detections_size(), 0);

    const auto* det = closest_detection(frame, range_m);
    ASSERT_NE(det, nullptr);
    ASSERT_TRUE(det->has_doppler());
    EXPECT_NEAR(std::abs(det->doppler().velocity_mps()), velocity, 0.005);
}

TEST(DetectionIntegrationTest, FastTargetRangeAndVelocityFromPeakPairing) {
    // At optical wavelengths a few m/s shifts the beat frequency by MHz, so
    // range and velocity are separated by pairing up- and down-chirp peaks.
    const double range_m = 30.0;
    const double velocity = 1.0;  // m/s: keeps both up and down beats positive

    auto forward_config = create_forward_config();
    aurora::forward::ForwardModel forward_model(forward_config);

    simulation::FmcwSimulator simulator(forward_model, create_sim_config(0.0, range_m, velocity));
    auto burst = simulator.simulate_burst(create_plan(/*up_down=*/true));

    dsp::RangeDopplerProcessor processor(create_dsp_config(), forward_model);
    dsp::PeakDopplerConfig peak_config;
    peak_config.set_peak_threshold_db(10.0);
    peak_config.set_min_peak_separation(3);
    peak_config.set_max_velocity_mps(30.0);
    peak_config.set_association_threshold(0.3);
    peak_config.set_allow_unpaired_peaks(false);
    processor.set_peak_doppler_config(peak_config);

    auto results = processor.process_doppler_peaks(burst, kWavelength);
    ASSERT_FALSE(results.empty());

    bool found = false;
    for (const auto& result : results) {
        for (const auto& peak : result.peaks()) {
            if (std::abs(peak.range_m() - range_m) < 2.0 * range_per_bin() &&
                std::abs(std::abs(peak.velocity_mps()) - velocity) < 0.3) {
                found = true;
            }
        }
    }
    EXPECT_TRUE(found) << "No paired peak matched range " << range_m
                       << " m and |velocity| " << velocity << " m/s";
}

TEST(DetectionIntegrationTest, DeterministicAcrossRuns) {
    auto a = run_pipeline(0.05, 25.0);
    auto b = run_pipeline(0.05, 25.0);
    ASSERT_EQ(a.detections_size(), b.detections_size());
    for (int i = 0; i < a.detections_size(); ++i) {
        EXPECT_DOUBLE_EQ(a.detections(i).range_m(), b.detections(i).range_m());
        EXPECT_DOUBLE_EQ(a.detections(i).azimuth_rad(), b.detections(i).azimuth_rad());
    }
}

}  // namespace aurora::integration
