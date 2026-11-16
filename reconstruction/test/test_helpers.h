#ifndef AURORA_RECONSTRUCTION_TEST_HELPERS_H
#define AURORA_RECONSTRUCTION_TEST_HELPERS_H

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <vector>

#include "detector_geometry.pb.h"
#include "reconstruction_config.pb.h"

namespace aurora::reconstruction::test {

// Pixel fill factor used for finite-size test pixels (80%, as in
// aurora::test::create_linear_array).
inline constexpr double kTestPixelFillFactor = 0.8;

// Fourier-plane pitch that places adjacent detector pixels `k_spacing` apart
// in K-space. From k = 2*pi * x / (f * lambda): x = k * f * lambda / (2*pi).
inline double pitch_for_k_spacing(double k_spacing_rad_m,
                                  double focal_length_m,
                                  double wavelength_m) {
    return k_spacing_rad_m * focal_length_m * wavelength_m / (2.0 * M_PI);
}

// Linear array of `num_pixels` point detectors (zero pixel width/height).
// A point detector has no K-space acceptance bandwidth, so the reconstruction
// operators treat each KSpaceSlice sample as a point sample at its own
// (kx, ky) instead of integrating over the detector's K-window. Use this for
// operator tests that hand-build arbitrary K-vectors; the sample's
// detector_index only needs to exist in the geometry.
inline aurora::forward::DetectorGeometry create_point_detector_geometry(
    const std::string& array_id,
    uint32_t num_pixels,
    double pitch_m,
    double wavelength_m) {

    aurora::forward::DetectorGeometry geometry;
    auto* array = geometry.add_arrays();
    array->set_array_id(array_id);
    array->set_nominal_pitch_m(pitch_m);
    array->add_wavelengths_m(wavelength_m);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_POINT);

    const double start = -static_cast<double>(num_pixels - 1) / 2.0 * pitch_m;
    for (uint32_t i = 0; i < num_pixels; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m(start + i * pitch_m);
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.0);
        pixel->set_height_m(0.0);
    }
    return geometry;
}

// Square `side` x `side` array of finite-size pixels centred on the optical
// axis. The receiver then samples a 2D K-space grid with spacing
// 2*pi * pitch / (f * lambda) in both kx and ky. Pixel index is row-major
// (index = row * side + col).
inline aurora::forward::DetectorGeometry create_square_detector_geometry(
    const std::string& array_id,
    uint32_t side,
    double pitch_m,
    double wavelength_m) {

    aurora::forward::DetectorGeometry geometry;
    auto* array = geometry.add_arrays();
    array->set_array_id(array_id);
    array->set_nominal_pitch_m(pitch_m);
    array->add_wavelengths_m(wavelength_m);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_SINC);

    const double start = -static_cast<double>(side - 1) / 2.0 * pitch_m;
    const double pixel_size = kTestPixelFillFactor * pitch_m;
    for (uint32_t row = 0; row < side; ++row) {
        for (uint32_t col = 0; col < side; ++col) {
            auto* pixel = array->add_pixels();
            pixel->set_index(row * side + col);
            pixel->set_x_m(start + col * pitch_m);
            pixel->set_y_m(start + row * pitch_m);
            pixel->set_width_m(pixel_size);
            pixel->set_height_m(pixel_size);
        }
    }
    return geometry;
}

// Generate synthetic K-space samples for known scene
// Computes y_j = sum_i scene(i) * exp(-i * k_j dot r_i)
inline std::vector<std::complex<double>> generate_k_samples(
    const Eigen::MatrixXcd& scene,
    const std::vector<Eigen::Vector2d>& k_vectors,
    const Eigen::Vector2d& grid_spacing,
    const Eigen::Vector2d& grid_origin) {

    std::vector<std::complex<double>> samples(k_vectors.size());
    int ny = scene.rows();
    int nx = scene.cols();

    for (size_t j = 0; j < k_vectors.size(); ++j) {
        std::complex<double> sum(0, 0);
        for (int iy = 0; iy < ny; ++iy) {
            double y = grid_origin.y() + iy * grid_spacing.y();
            for (int ix = 0; ix < nx; ++ix) {
                double x = grid_origin.x() + ix * grid_spacing.x();
                double phase = -(k_vectors[j].x() * x + k_vectors[j].y() * y);
                sum += scene(iy, ix) * std::exp(std::complex<double>(0, phase));
            }
        }
        samples[j] = sum;
    }
    return samples;
}

// Generate sparse scene with point targets at pixel indices
inline Eigen::MatrixXcd generate_sparse_scene(
    const Eigen::Vector2i& size,
    const std::vector<Eigen::Vector2i>& target_pixels,
    std::complex<double> amplitude = {1.0, 0.0}) {

    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Zero(size.y(), size.x());

    for (const auto& pix : target_pixels) {
        if (pix.x() >= 0 && pix.x() < size.x() &&
            pix.y() >= 0 && pix.y() < size.y()) {
            scene(pix.y(), pix.x()) = amplitude;
        }
    }
    return scene;
}

// Compute PSNR between two images (in dB)
// PSNR = 10 * log10(max_signal^2 / MSE)
inline double compute_psnr(
    const Eigen::MatrixXcd& reference,
    const Eigen::MatrixXcd& reconstructed) {

    if (reference.rows() != reconstructed.rows() ||
        reference.cols() != reconstructed.cols()) {
        return -std::numeric_limits<double>::infinity();
    }

    // Find max magnitude in reference for normalization
    double max_signal = 0.0;
    for (int i = 0; i < reference.rows(); ++i) {
        for (int j = 0; j < reference.cols(); ++j) {
            max_signal = std::max(max_signal, std::abs(reference(i, j)));
        }
    }

    if (max_signal < 1e-15) {
        return -std::numeric_limits<double>::infinity();
    }

    // Compute MSE
    double mse = 0.0;
    int count = reference.rows() * reference.cols();
    for (int i = 0; i < reference.rows(); ++i) {
        for (int j = 0; j < reference.cols(); ++j) {
            mse += std::norm(reference(i, j) - reconstructed(i, j));
        }
    }
    mse /= count;

    if (mse < 1e-20) {
        return std::numeric_limits<double>::infinity();  // Perfect reconstruction
    }

    return 10.0 * std::log10(max_signal * max_signal / mse);
}

// Compute total energy (sum of squared magnitudes)
inline double compute_energy(const Eigen::MatrixXcd& image) {
    double energy = 0.0;
    for (int i = 0; i < image.rows(); ++i) {
        for (int j = 0; j < image.cols(); ++j) {
            energy += std::norm(image(i, j));
        }
    }
    return energy;
}

// Compute total energy of K-space samples
inline double compute_k_space_energy(
    const std::vector<std::complex<double>>& k_data) {
    double energy = 0.0;
    for (const auto& sample : k_data) {
        energy += std::norm(sample);
    }
    return energy;
}

// Create test K-vectors with uniform grid sampling
inline std::vector<Eigen::Vector2d> create_test_k_vectors(
    uint32_t samples_per_side,
    double k_max = 3000.0) {

    std::vector<Eigen::Vector2d> k_vectors;
    k_vectors.reserve(samples_per_side * samples_per_side);

    double k_step = 2.0 * k_max / (samples_per_side - 1);
    for (uint32_t i = 0; i < samples_per_side; ++i) {
        double kx = -k_max + i * k_step;
        for (uint32_t j = 0; j < samples_per_side; ++j) {
            double ky = -k_max + j * k_step;
            k_vectors.emplace_back(kx, ky);
        }
    }
    return k_vectors;
}

}  // namespace aurora::reconstruction::test

#endif  // AURORA_RECONSTRUCTION_TEST_HELPERS_H
