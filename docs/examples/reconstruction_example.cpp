/**
 * @file reconstruction_example.cpp
 * @brief End-to-end reconstruction stream example.
 *
 * Scene -> FMCW simulation -> range DSP -> MeasurementTensor -> Reconstructor.
 *
 * The 8-pixel detector array samples eight spatial frequencies along kx at
 * each wavelength. The Reconstructor forms an azimuth profile of each range
 * bin from those K-space samples with the direct adjoint operator; this
 * example prints the profile of the range bin that contains the targets.
 *
 * Build and run:
 *   cmake --build build --target reconstruction_example
 *   ./build/docs/examples/reconstruction_example
 */

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "dsp/range_doppler_processor.h"
#include "forward/forward_model.h"
#include "reconstruction/reconstructor.h"
#include "simulation/fmcw_simulator.h"

#include "forward_config.pb.h"
#include "reconstruction_config.pb.h"
#include "sequence_plan.pb.h"
#include "sim_config.pb.h"

namespace {

constexpr double kTwoPi = 2.0 * M_PI;

constexpr double kSampleRate = 125e6;        // Hz
constexpr double kChirpBandwidth = 200e6;    // Hz
constexpr double kChirpDuration = 12.5e-6;   // s
constexpr double kWavelength = 1550e-9;      // m
constexpr uint32_t kFftSize = 2048;

constexpr double kFocalLength = 0.2;         // m
constexpr double kDetectorPitch = 0.5e-3;    // m
constexpr uint32_t kNumDetectors = 8;

// Image grid: 8 K-samples with spacing Δk give an unambiguous azimuth span of
// 2π/(f·Δk); oversample it 4x so the profile is easy to read.
constexpr int kOversampling = 4;
constexpr int kGridSizeX = static_cast<int>(kNumDetectors) * kOversampling;
constexpr int kGridSizeY = 1;
constexpr double kTargetRange = 30.0;        // m

struct Target {
    double azimuth_rad;
    double reflectivity;
};

double detector_k_step() {
    return kTwoPi * kDetectorPitch / (kFocalLength * kWavelength);
}

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
    optics->set_k_max_rad_per_m(1e6);
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
    noise->set_enable_shot_noise(false);
    noise->set_enable_thermal_noise(false);
    noise->set_enable_quantization_noise(false);

    config.set_tx_power_w(1.0);
    config.set_lo_power_w(1.0);
    config.set_system_efficiency(1.0);
    config.set_rng_seed(7);
    config.set_occlusion_mode(aurora::simulation::OCCLUSION_NONE);
    config.set_specular_threshold(1.0 - 1e-10);

    auto* bw = config.mutable_bandwidth_config();
    bw->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    bw->set_quadrature_order(11);  // odd order: SINC window with even orders samples its zeros

    for (const auto& t : targets) {
        auto* point = config.mutable_scene()->add_point_targets();
        point->mutable_position_m()->set_x(kTargetRange * std::sin(t.azimuth_rad));
        point->mutable_position_m()->set_y(0.0);
        point->mutable_position_m()->set_z(kTargetRange * std::cos(t.azimuth_rad));
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

aurora::reconstruction::ReconstructionConfig create_reconstruction_config() {
    aurora::reconstruction::ReconstructionConfig config;
    config.set_algorithm(aurora::reconstruction::ALGORITHM_DIRECT_ADJOINT);
    config.set_grid_size_x(kGridSizeX);
    config.set_grid_size_y(kGridSizeY);
    // Pixel size: one unambiguous period 2π/Δk split over the oversampled grid
    config.set_pixel_size_m(kTwoPi / (detector_k_step() * kGridSizeX));

    auto* bw = config.mutable_bandwidth_config();
    bw->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    bw->set_quadrature_order(11);

    config.mutable_direct_adjoint_config()->set_apodization(
        aurora::reconstruction::DirectAdjointConfig::APOD_NONE);
    return config;
}

aurora::acquisition::SequencePlan create_plan() {
    aurora::acquisition::SequencePlan plan;
    plan.set_plan_id(1);
    auto* chirp = plan.add_chirps();
    chirp->set_chirp_index(0);
    chirp->set_chirp_duration_s(kChirpDuration);
    chirp->set_chirp_bandwidth_hz(kChirpBandwidth);
    chirp->set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);
    chirp->add_wavelength_m(kWavelength);
    return plan;
}

/// Index of the range bin with the most energy across all samples.
uint32_t find_peak_range_bin(const aurora::shared::MeasurementTensor& tensor) {
    std::vector<double> energy;
    for (const auto& wl : tensor.data()) {
        for (const auto& det : wl.detectors()) {
            if (energy.size() < static_cast<size_t>(det.range_data_size())) {
                energy.resize(det.range_data_size(), 0.0);
            }
            for (int b = 0; b < det.range_data_size(); ++b) {
                energy[b] += std::norm(std::complex<double>(
                    det.range_data(b).real(), det.range_data(b).imag()));
            }
        }
    }
    uint32_t best = 0;
    for (size_t b = 1; b < energy.size(); ++b) {
        if (energy[b] > energy[best]) best = static_cast<uint32_t>(b);
    }
    return best;
}

void print_profile(const Eigen::MatrixXcd& image, double origin_m, double pixel_size_m) {
    constexpr int kBarWidth = 40;
    double max_mag = image.cwiseAbs().maxCoeff();
    if (max_mag <= 0.0) return;

    for (Eigen::Index ix = 0; ix < image.cols(); ++ix) {
        double azimuth_mrad = (origin_m + ix * pixel_size_m) / kFocalLength * 1e3;
        double v = std::abs(image(0, ix)) / max_mag;
        int bar = static_cast<int>(std::round(v * kBarWidth));
        std::cout << "  " << std::setw(7) << std::fixed << std::setprecision(2)
                  << azimuth_mrad << " mrad |" << std::string(bar, '#') << "\n";
    }
}

}  // namespace

int main() {
    const std::vector<Target> targets = {
        {0.8e-3, 0.5},
        {-0.6e-3, 0.3},
    };

    auto forward_config = create_forward_config();
    aurora::forward::ForwardModel forward_model(forward_config);

    std::cout << "Aurora reconstruction example\n"
              << "  " << kNumDetectors << " detectors = " << kNumDetectors
              << " K-space samples per wavelength\n\n";

    aurora::simulation::FmcwSimulator simulator(forward_model, create_sim_config(targets));
    auto burst = simulator.simulate_burst(create_plan());

    aurora::dsp::RangeDopplerProcessor processor(create_dsp_config(), forward_model);
    auto tensor = processor.to_measurement_tensor(processor.process_range(burst));

    aurora::reconstruction::Reconstructor reconstructor(create_reconstruction_config(),
                                                        forward_model);
    uint32_t bin = find_peak_range_bin(tensor);
    auto image = reconstructor.reconstruct_range_bin(tensor, bin, kWavelength);

    const auto& axis = tensor.range_axis();
    std::cout << std::fixed << std::setprecision(2)
              << "Range bin " << bin << " (" << axis.start() + bin * axis.step()
              << " m), targets at " << kTargetRange << " m:\n";
    for (const auto& t : targets) {
        std::cout << "  truth: azimuth " << t.azimuth_rad * 1e3 << " mrad\n";
    }

    Eigen::Index peak_row = 0, peak_col = 0;
    image.cwiseAbs().maxCoeff(&peak_row, &peak_col);
    Eigen::Vector2d origin = reconstructor.grid_origin_m();
    Eigen::Vector2d spacing = reconstructor.grid_spacing_m();
    std::cout << "  profile peak: azimuth "
              << (origin.x() + peak_col * spacing.x()) / kFocalLength * 1e3 << " mrad\n\n";

    std::cout << "Azimuth profile:\n";
    print_profile(image, origin.x(), spacing.x());
    return 0;
}
