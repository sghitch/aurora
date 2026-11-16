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

#include <gtest/gtest.h>

#include "reconstruction/bandwidth_weight_calculator.h"
#include "reconstruction/direct_operator.h"

#include <cmath>

#include "forward/bandwidth_model.h"
#include "forward/forward_model.h"
#include "forward/measurement_operator.h"
#include "test_helpers.h"

#include "bandwidth_config.pb.h"

using namespace aurora::reconstruction;
using aurora::forward::BandwidthWindowType;
using aurora::forward::MeasurementOperator;

// Helper function that provides a default BandwidthConfig for tests
aurora::forward::BandwidthConfig make_test_bandwidth_config() {
    aurora::forward::BandwidthConfig config;
    config.set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    config.set_quadrature_order(4);
    return config;
}

// Helper function that provides a BandwidthConfig with specified window type
aurora::forward::BandwidthConfig make_test_bandwidth_config(aurora::forward::BandwidthWindowType window_type) {
    aurora::forward::BandwidthConfig config;
    config.set_window_type(window_type);
    config.set_quadrature_order(4);
    return config;
}

// =============================================================================
// Test Fixture with ForwardModel
// =============================================================================

class BandwidthWeightCalculatorTest : public ::testing::Test {
  protected:
    static constexpr double kTolerance = 1e-6;
    static constexpr double kWavelength = 905e-9;
    static constexpr double kFocalLength = 0.05;   // m
    static constexpr uint32_t kNumDetectors = 8;
    static constexpr double kPitch = 0.5e-3;       // m
    static constexpr double kWeight = 0.8;
    static constexpr double kBandwidth = 30000.0;  // rad/m

    void SetUp() override {
        // Set up the detector geometry (kNumDetectors pixels, 0.5mm pitch).
        // The pixel size is chosen so the pixel's K-space acceptance bandwidth
        // 2*pi * w / (f * lambda) equals kBandwidth exactly.
        const double pixel_size = aurora::forward::BandwidthModel::
            k_bandwidth_to_fourier_width(kBandwidth, kWavelength, kFocalLength);

        auto* array = detector_geom_.add_arrays();
        array->set_array_id("test_detector");
        array->set_nominal_pitch_m(kPitch);
        array->add_wavelengths_m(kWavelength);
        array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_SINC);
        for (uint32_t i = 0; i < kNumDetectors; ++i) {
            auto* pixel = array->add_pixels();
            pixel->set_index(i);
            pixel->set_x_m((i - (kNumDetectors - 1) / 2.0) * kPitch);
            pixel->set_y_m(0.0);
            pixel->set_width_m(pixel_size);
            pixel->set_height_m(pixel_size);
        }

        // Set up optics configuration
        optics_.set_fourier_lens_focal_m(kFocalLength);
        optics_.add_wavelengths_m(kWavelength);

        // Create ForwardModel
        forward_model_ = std::make_unique<aurora::forward::ForwardModel>(
            detector_geom_, optics_);

        // Create KSpaceSlice with one sample per detector pixel at its scene K-vector
        slice_.set_wavelength_m(kWavelength);
        for (uint32_t i = 0; i < kNumDetectors; ++i) {
            Eigen::Vector2d k_scene = forward_model_->get_effective_k(i, kWavelength);
            auto* sample = slice_.add_samples();
            sample->set_detector_index(i);
            sample->set_kx(k_scene.x());
            sample->set_ky(k_scene.y());
            sample->set_kx_bandwidth(kBandwidth);
            sample->set_ky_bandwidth(kBandwidth);
            sample->set_weight(kWeight);
        }
    }

    // Window parameters for a detector pixel, built independently of the calculator
    MeasurementOperator::WindowParams window_params(uint32_t det_idx) const {
        return aurora::forward::build_window_params(*forward_model_, det_idx, kWavelength);
    }

    aurora::forward::DetectorGeometry detector_geom_;
    aurora::forward::OpticsConfig optics_;
    aurora::shared::KSpaceSlice slice_;
    std::unique_ptr<aurora::forward::ForwardModel> forward_model_;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(BandwidthWeightCalculatorTest, HasBandwidth) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);
    EXPECT_TRUE(calc.has_bandwidth());
}

TEST_F(BandwidthWeightCalculatorTest, NumSamples) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);
    EXPECT_EQ(calc.num_samples(), 8);
}

TEST_F(BandwidthWeightCalculatorTest, NoBandwidthWhenBandwidthFieldsMissing) {
    // Point detectors (zero pixel size) have no K-space acceptance bandwidth
    auto geom_no_bw = aurora::reconstruction::test::create_point_detector_geometry(
        "no_bw_detector", kNumDetectors, kPitch, kWavelength);

    aurora::forward::ForwardModel fm_no_bw(geom_no_bw, optics_);

    aurora::shared::KSpaceSlice slice_no_bw;
    slice_no_bw.set_wavelength_m(kWavelength);
    auto* sample = slice_no_bw.add_samples();
    sample->set_detector_index(0);
    sample->set_kx(1000.0);
    sample->set_ky(1000.0);

    BandwidthWeightCalculator calc(slice_no_bw, fm_no_bw);
    EXPECT_FALSE(calc.has_bandwidth());
}

TEST_F(BandwidthWeightCalculatorTest, BandwidthAccessors) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    for (size_t i = 0; i < calc.num_samples(); ++i) {
        EXPECT_NEAR(calc.kx_bandwidth(i), kBandwidth, kTolerance);
        EXPECT_NEAR(calc.ky_bandwidth(i), kBandwidth, kTolerance);
    }
}

TEST_F(BandwidthWeightCalculatorTest, AverageBandwidth) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);
    auto avg = calc.average_bandwidth();

    EXPECT_NEAR(avg.x(), kBandwidth, kTolerance);
    EXPECT_NEAR(avg.y(), kBandwidth, kTolerance);
}

TEST_F(BandwidthWeightCalculatorTest, ResolutionLimit) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);
    auto limit = calc.resolution_limit();

    // limit = 2*pi / bandwidth
    double expected = 2.0 * M_PI / kBandwidth;
    EXPECT_NEAR(limit.x(), expected, kTolerance);
    EXPECT_NEAR(limit.y(), expected, kTolerance);
}

TEST_F(BandwidthWeightCalculatorTest, IsGridValidFineSpacing) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    // Resolution limit = 2*pi / 30000 ~ 0.21mm
    // 0.1mm spacing is fine enough
    EXPECT_TRUE(calc.is_grid_valid(Eigen::Vector2d(0.1e-3, 0.1e-3)));
}

TEST_F(BandwidthWeightCalculatorTest, IsGridValidCoarseSpacing) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    // Resolution limit = 2*pi / 30000 ~ 0.21mm
    // 0.5mm spacing is too coarse
    EXPECT_FALSE(calc.is_grid_valid(Eigen::Vector2d(0.5e-3, 0.5e-3)));
}

TEST_F(BandwidthWeightCalculatorTest, ComputeWeightsNotEmpty) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    Eigen::Vector2i grid_size(64, 64);
    Eigen::Vector2d grid_spacing(0.1e-3, 0.1e-3);
    Eigen::Vector2d grid_origin(-3.2e-3, -3.2e-3);

    auto weights = calc.compute_weights(
        grid_size, grid_spacing, grid_origin,
        aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN, 1e-6);

    EXPECT_FALSE(weights.empty());
    EXPECT_EQ(weights.sample_to_pixels.size(), 8);
}

TEST_F(BandwidthWeightCalculatorTest, SparseWeightsConsistency) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    Eigen::Vector2i grid_size(64, 64);
    Eigen::Vector2d grid_spacing(0.1e-3, 0.1e-3);
    Eigen::Vector2d grid_origin(-3.2e-3, -3.2e-3);

    auto weights = calc.compute_weights(
        grid_size, grid_spacing, grid_origin,
        aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN, 1e-6);

    // Verify consistency: every entry in sample_to_pixels should have
    // a corresponding entry in pixel_to_samples
    for (size_t m = 0; m < weights.sample_to_pixels.size(); ++m) {
        for (const auto& pw : weights.sample_to_pixels[m]) {
            size_t pixel_idx = static_cast<size_t>(pw.iy) * static_cast<size_t>(grid_size.x()) +
                               static_cast<size_t>(pw.ix);
            const auto& contributions = weights.pixel_to_samples[pixel_idx];

            bool found = false;
            for (const auto& sc : contributions) {
                if (sc.sample_idx == m) {
                    EXPECT_NEAR(sc.weight, pw.weight, 1e-10);
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "Missing reverse mapping for sample " << m;
        }
    }
}

TEST_F(BandwidthWeightCalculatorTest, AverageSparsity) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    Eigen::Vector2i grid_size(64, 64);
    Eigen::Vector2d grid_spacing(0.1e-3, 0.1e-3);
    Eigen::Vector2d grid_origin(-3.2e-3, -3.2e-3);

    auto weights = calc.compute_weights(
        grid_size, grid_spacing, grid_origin,
        aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN, 1e-6);

    double sparsity = weights.average_sparsity();
    EXPECT_GT(sparsity, 0.0);
    EXPECT_LT(sparsity, grid_size.x() * grid_size.y());  // Must be sparse
}

TEST_F(BandwidthWeightCalculatorTest, DetectorIndexAccessor) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    // One sample per detector pixel, in pixel order
    for (size_t i = 0; i < calc.num_samples(); ++i) {
        EXPECT_EQ(calc.detector_index(i), static_cast<uint32_t>(i));
    }
}

TEST_F(BandwidthWeightCalculatorTest, WeightsArePositive) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    Eigen::Vector2i grid_size(64, 64);
    Eigen::Vector2d grid_spacing(0.1e-3, 0.1e-3);
    Eigen::Vector2d grid_origin(-3.2e-3, -3.2e-3);

    auto weights = calc.compute_weights(
        grid_size, grid_spacing, grid_origin,
        aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN, 1e-6);

    for (const auto& sample_weights : weights.sample_to_pixels) {
        for (const auto& pw : sample_weights) {
            EXPECT_GT(pw.weight, 0.0) << "All weights should be positive";
        }
    }
}

TEST_F(BandwidthWeightCalculatorTest, PixelIndicesInBounds) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    Eigen::Vector2i grid_size(64, 64);
    Eigen::Vector2d grid_spacing(0.1e-3, 0.1e-3);
    Eigen::Vector2d grid_origin(-3.2e-3, -3.2e-3);

    auto weights = calc.compute_weights(
        grid_size, grid_spacing, grid_origin,
        aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN, 1e-6);

    for (const auto& sample_weights : weights.sample_to_pixels) {
        for (const auto& pw : sample_weights) {
            EXPECT_GE(pw.ix, 0);
            EXPECT_LT(pw.ix, grid_size.x());
            EXPECT_GE(pw.iy, 0);
            EXPECT_LT(pw.iy, grid_size.y());
        }
    }
}

// =============================================================================
// MeasurementOperator Integration Tests
// =============================================================================

TEST_F(BandwidthWeightCalculatorTest, WeightsMatchMeasurementOperator) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    Eigen::Vector2i grid_size(64, 64);
    Eigen::Vector2d grid_spacing(0.1e-3, 0.1e-3);
    Eigen::Vector2d grid_origin(-3.2e-3, -3.2e-3);

    auto weights = calc.compute_weights(
        grid_size, grid_spacing, grid_origin,
        aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN, 1e-6);

    ASSERT_FALSE(weights.empty());

    // Spatial-to-K conversion factor
    double k_scale = 2.0 * M_PI / (optics_.fourier_lens_focal_m() * kWavelength);

    // Verify weights match direct MeasurementOperator::weight() calls
    for (size_t m = 0; m < std::min(size_t(3), weights.sample_to_pixels.size()); ++m) {
        auto params = window_params(slice_.samples(static_cast<int>(m)).detector_index());

        for (const auto& pw : weights.sample_to_pixels[m]) {
            double rx = grid_origin.x() + pw.ix * grid_spacing.x();
            double ry = grid_origin.y() + pw.iy * grid_spacing.y();
            Eigen::Vector2d k_pixel(rx * k_scale, ry * k_scale);

            double expected_w = MeasurementOperator::weight(
                k_pixel, params, aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN);

            EXPECT_NEAR(pw.weight, expected_w, 1e-12)
                << "Mismatch at sample " << m << ", pixel (" << pw.ix << ", " << pw.iy << ")";
        }
    }
}

TEST_F(BandwidthWeightCalculatorTest, AdjointConsistencyWithWindowWeights) {
    Eigen::Vector2i grid_size(64, 64);
    Eigen::Vector2d grid_spacing(0.1e-3, 0.1e-3);
    Eigen::Vector2d grid_origin(-3.2e-3, -3.2e-3);

    // DirectOperator over the same slice
    DirectOperator op(
        slice_, *forward_model_,
        make_test_bandwidth_config(aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN),
        grid_size, grid_spacing, grid_origin);

    // Random scene
    Eigen::MatrixXcd x = Eigen::MatrixXcd::Random(grid_size.y(), grid_size.x());

    // Random K-samples
    Eigen::VectorXcd y = Eigen::VectorXcd::Random(slice_.samples_size());

    // Compute <Ax, y>
    auto Ax = op.forward(x, grid_origin, grid_spacing);
    std::complex<double> lhs = (Ax.conjugate().array() * y.array()).sum();

    // Compute <x, A*y>
    auto Asy = op.adjoint(y, grid_origin, grid_spacing, grid_size);
    std::complex<double> rhs = (x.conjugate().array() * Asy.array()).sum();

    double rel_error = std::abs(lhs - rhs) / std::max(std::abs(lhs), std::abs(rhs));
    EXPECT_LT(rel_error, 1e-10) << "Adjoint consistency failed: lhs=" << lhs << ", rhs=" << rhs;
}

TEST_F(BandwidthWeightCalculatorTest, SupportBoundsCorrect) {
    BandwidthWeightCalculator calc(slice_, *forward_model_);

    Eigen::Vector2i grid_size(64, 64);
    Eigen::Vector2d grid_spacing(0.1e-3, 0.1e-3);
    Eigen::Vector2d grid_origin(-3.2e-3, -3.2e-3);

    double threshold = 1e-6;
    auto weights = calc.compute_weights(
        grid_size, grid_spacing, grid_origin,
        aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN, threshold);

    ASSERT_FALSE(weights.empty());

    // Spatial-to-K conversion factor
    double k_scale = 2.0 * M_PI / (optics_.fourier_lens_focal_m() * kWavelength);

    for (size_t m = 0; m < weights.sample_to_pixels.size(); ++m) {
        auto params = window_params(slice_.samples(static_cast<int>(m)).detector_index());

        auto support = MeasurementOperator::compute_support(
            params, threshold, aurora::forward::BANDWIDTH_WINDOW_GAUSSIAN);

        EXPECT_TRUE(support.is_valid()) << "Support bounds invalid for sample " << m;

        // All weights should be positive and above threshold
        for (const auto& pw : weights.sample_to_pixels[m]) {
            EXPECT_GT(pw.weight, 0.0) << "Weight should be positive";
            EXPECT_GT(pw.weight, threshold) << "Weight should be above threshold";
        }

        // The highest-weight pixel should be near the support center
        if (!weights.sample_to_pixels[m].empty()) {
            double max_weight = 0;
            int max_ix = 0, max_iy = 0;
            for (const auto& pw : weights.sample_to_pixels[m]) {
                if (pw.weight > max_weight) {
                    max_weight = pw.weight;
                    max_ix = pw.ix;
                    max_iy = pw.iy;
                }
            }
            double rx = grid_origin.x() + max_ix * grid_spacing.x();
            double ry = grid_origin.y() + max_iy * grid_spacing.y();
            Eigen::Vector2d k_max_pixel(rx * k_scale, ry * k_scale);

            EXPECT_TRUE(support.contains(k_max_pixel))
                << "Max-weight pixel should be within support for sample " << m;
        }
    }
}
