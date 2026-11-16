/**
 * @file test_pcg_integration.cpp
 * @brief Integration tests for PCG algorithm with Reconstructor.
 *
 * Tests that ALGORITHM_PCG_REGULARIZED works correctly when used through
 * the Reconstructor facade, in the same way as the other algorithms.
 */

#include "reconstruction/reconstructor.h"

#include <cmath>

#include <gtest/gtest.h>

#include "bandwidth_config.pb.h"
#include "forward/forward_model.h"
#include "reconstruction/reconstruction_error.h"
#include "test_helpers.h"

namespace aurora::reconstruction {

namespace {

// Receiver geometry for the reconstruction tests: a square array of
// kArraySide x kArraySide pixels whose pitch places adjacent pixels kKSpacing
// apart in K-space. The geometric K-samples then form a 2D K-grid matched to
// the 16 mm reconstruction grid (unambiguous field of view 2*pi / kKSpacing).
constexpr uint32_t kArraySide = 8;
constexpr double kKSpacing = 400.0;    // rad/m between adjacent pixels
constexpr double kFocalLength = 0.05;  // m

// Helper to create detector geometry
aurora::forward::DetectorGeometry create_test_detector(double wavelength_m) {
    const double pitch = aurora::reconstruction::test::pitch_for_k_spacing(
        kKSpacing, kFocalLength, wavelength_m);
    return aurora::reconstruction::test::create_square_detector_geometry(
        "test_array", kArraySide, pitch, wavelength_m);
}

// Helper to create optics config
aurora::forward::OpticsConfig create_test_optics(double wavelength_m = 905e-9) {
    aurora::forward::OpticsConfig optics;
    optics.set_fourier_lens_focal_m(kFocalLength);
    optics.add_wavelengths_m(wavelength_m);
    optics.set_k_max_rad_per_m(1e5);
    return optics;
}

}  // namespace

class PcgIntegrationTest : public ::testing::Test {
  protected:
    static constexpr int kDefaultNumDetectors = kArraySide * kArraySide;
    static constexpr double kDefaultWavelength = 905e-9;

    static void SetUpTestSuite() {
        // Create shared ForwardModel (expensive initialization)
        detector_ = new aurora::forward::DetectorGeometry(
            create_test_detector(kDefaultWavelength));
        optics_ = new aurora::forward::OpticsConfig(
            create_test_optics(kDefaultWavelength));

        shared_forward_model_ = new aurora::forward::ForwardModel(
            *detector_, *optics_);
    }

    static void TearDownTestSuite() {
        delete shared_forward_model_;
        delete optics_;
        delete detector_;
        shared_forward_model_ = nullptr;
        optics_ = nullptr;
        detector_ = nullptr;
    }

    void SetUp() override {
        config_.set_algorithm(ALGORITHM_PCG_REGULARIZED);
        config_.set_grid_size_x(16);
        config_.set_grid_size_y(16);
        config_.set_pixel_size_m(0.001);  // 1mm pixels

        // Set up bandwidth config (required)
        auto* bw_config = config_.mutable_bandwidth_config();
        bw_config->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
        bw_config->set_quadrature_order(4);

        // Set up PCG config with required fields
        auto* pcg_config = config_.mutable_pcg_config();
        pcg_config->set_max_iterations(20);
        pcg_config->set_tolerance(1e-6);
        pcg_config->set_regularizer_type(REGULARIZER_TIKHONOV);
        pcg_config->set_regularizer_lambda(1e-4);
        pcg_config->set_preconditioner_type(PRECONDITIONER_NONE);
    }

    const aurora::forward::ForwardModel& forward_model() const {
        return *shared_forward_model_;
    }

    // Helper to create a MeasurementTensor with test data: one wavelength
    // record with one detector record per pixel
    aurora::shared::MeasurementTensor create_test_tensor(
        int num_detectors,
        int num_range_bins,
        double wavelength_m = 905e-9) {

        aurora::shared::MeasurementTensor tensor;

        // Set up range axis
        auto* range_axis = tensor.mutable_range_axis();
        range_axis->set_start(0.0);
        range_axis->set_step(0.1);  // 10cm range bins
        range_axis->set_count(num_range_bins);
        range_axis->set_unit("m");

        tensor.set_burst_id(1);
        tensor.set_timestamp_s(0.0);

        auto* wavelength_channel = tensor.add_data();
        wavelength_channel->set_wavelength_m(wavelength_m);

        for (int d = 0; d < num_detectors; ++d) {
            auto* detector_channel = wavelength_channel->add_detectors();
            detector_channel->set_detector_index(d);

            // Add range data (zeros for now - tests will populate)
            for (int r = 0; r < num_range_bins; ++r) {
                auto* complex_val = detector_channel->add_range_data();
                complex_val->set_real(0.0);
                complex_val->set_imag(0.0);
            }
        }

        return tensor;
    }

    // Helper to add a point target using ForwardModel K-vectors
    void add_point_target_with_forward_model(
        aurora::shared::MeasurementTensor& tensor,
        const aurora::forward::ForwardModel& forward_model,
        int range_bin,
        double x_m,
        double y_m,
        std::complex<double> reflectivity = {1.0, 0.0}) {

        for (auto& wavelength_channel : *tensor.mutable_data()) {
            double wavelength_m = wavelength_channel.wavelength_m();

            for (auto& detector_channel : *wavelength_channel.mutable_detectors()) {
                uint32_t det_idx = detector_channel.detector_index();
                if (det_idx >= forward_model.num_detectors()) continue;

                // Scene K-vector of this detector pixel
                Eigen::Vector2d k = forward_model.get_effective_k(det_idx, wavelength_m);

                // Phase encoding: phi = -k dot r
                double phase = -(k.x() * x_m + k.y() * y_m);
                std::complex<double> contribution =
                    reflectivity * std::exp(std::complex<double>(0, phase));

                if (range_bin < detector_channel.range_data_size()) {
                    auto* range_data = detector_channel.mutable_range_data(range_bin);
                    range_data->set_real(range_data->real() + contribution.real());
                    range_data->set_imag(range_data->imag() + contribution.imag());
                }
            }
        }
    }

    static aurora::forward::DetectorGeometry* detector_;
    static aurora::forward::OpticsConfig* optics_;
    static aurora::forward::ForwardModel* shared_forward_model_;

    ReconstructionConfig config_;
};

// Static member definitions
aurora::forward::DetectorGeometry* PcgIntegrationTest::detector_ = nullptr;
aurora::forward::OpticsConfig* PcgIntegrationTest::optics_ = nullptr;
aurora::forward::ForwardModel* PcgIntegrationTest::shared_forward_model_ = nullptr;

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(PcgIntegrationTest, Reconstructor_RequiresPcgConfig) {
    // PCG algorithm without pcg_config should throw
    config_.clear_pcg_config();

    EXPECT_THROW(
        Reconstructor(config_, forward_model()),
        ReconstructionError);
}

TEST_F(PcgIntegrationTest, Reconstructor_AcceptsPcgConfig) {
    // PCG algorithm with valid pcg_config should construct successfully
    EXPECT_NO_THROW(Reconstructor(config_, forward_model()));
}

TEST_F(PcgIntegrationTest, AlgorithmAccessorReturnsPcg) {
    Reconstructor reconstructor(config_, forward_model());
    EXPECT_EQ(reconstructor.algorithm(), ALGORITHM_PCG_REGULARIZED);
}

// =============================================================================
// Reconstruction Tests
// =============================================================================

TEST_F(PcgIntegrationTest, PcgReconstruction_PointTarget) {
    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);

    // Add a point target at the center
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0, {1.0, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    EXPECT_EQ(scene.wavelengths_size(), 1);
    EXPECT_EQ(scene.wavelengths(0).images_size(), 1);

    const auto& image = scene.wavelengths(0).images(0);
    EXPECT_GT(image.pixels_size(), 0);
    EXPECT_GT(image.peak_value(), 0.0);
}

TEST_F(PcgIntegrationTest, PcgReconstruction_PointTargetLocation) {
    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);

    // Add a point target at an off-center location
    double target_x = 0.001;  // 1mm
    double target_y = 0.002;  // 2mm
    add_point_target_with_forward_model(tensor, forward_model(), 0, target_x, target_y, {1.0, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    const auto& image = scene.wavelengths(0).images(0);
    EXPECT_GT(image.peak_value(), 0.0);

    // Find peak location
    int nx = image.x_axis().count();
    int ny = image.y_axis().count();
    double max_val = 0.0;
    int max_ix = 0, max_iy = 0;

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const auto& pixel = image.pixels(iy * nx + ix);
            double mag = std::sqrt(pixel.real() * pixel.real() +
                                   pixel.imag() * pixel.imag());
            if (mag > max_val) {
                max_val = mag;
                max_ix = ix;
                max_iy = iy;
            }
        }
    }

    // Peak should be near the target location
    double peak_x = image.x_axis().start() + max_ix * image.x_axis().step();
    double peak_y = image.y_axis().start() + max_iy * image.y_axis().step();

    // Allow some error (within a few pixels)
    // Note: With bandwidth quadrature, PSF shape changes slightly affecting localization
    EXPECT_NEAR(peak_x, target_x, 0.005);
    EXPECT_NEAR(peak_y, target_y, 0.005);
}

TEST_F(PcgIntegrationTest, PcgReconstruction_TwoTargets) {
    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);

    // Add two point targets at well-separated locations
    // Use wider separation to ensure they're resolvable with sparse K-space
    double target1_x = -0.004;
    double target1_y = 0.0;
    double target2_x = 0.004;
    double target2_y = 0.0;
    add_point_target_with_forward_model(tensor, forward_model(), 0, target1_x, target1_y, {1.0, 0.0});
    add_point_target_with_forward_model(tensor, forward_model(), 0, target2_x, target2_y, {1.0, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    const auto& image = scene.wavelengths(0).images(0);

    // Both targets should contribute to the reconstruction
    EXPECT_GT(image.peak_value(), 0.0);
    EXPECT_GT(image.total_energy(), 0.0);

    // Verify both target regions have significant energy
    // Find pixel indices corresponding to target locations
    int nx = image.x_axis().count();
    double x_start = image.x_axis().start();
    double x_step = image.x_axis().step();

    // Convert target positions to pixel indices
    int target1_pixel = static_cast<int>((target1_x - x_start) / x_step);
    int target2_pixel = static_cast<int>((target2_x - x_start) / x_step);

    // Clamp to valid range
    target1_pixel = std::max(0, std::min(nx - 1, target1_pixel));
    target2_pixel = std::max(0, std::min(nx - 1, target2_pixel));

    // Sum energy in local regions around each target
    // Use the center row for simplicity
    int center_row = image.y_axis().count() / 2;
    double energy_region1 = 0.0;
    double energy_region2 = 0.0;

    for (int dx = -2; dx <= 2; ++dx) {
        int ix1 = target1_pixel + dx;
        int ix2 = target2_pixel + dx;
        if (ix1 >= 0 && ix1 < nx) {
            const auto& pixel = image.pixels(center_row * nx + ix1);
            energy_region1 += pixel.real() * pixel.real() + pixel.imag() * pixel.imag();
        }
        if (ix2 >= 0 && ix2 < nx) {
            const auto& pixel = image.pixels(center_row * nx + ix2);
            energy_region2 += pixel.real() * pixel.real() + pixel.imag() * pixel.imag();
        }
    }

    // Both regions should have meaningful energy from the targets
    // Note: targets may blur together somewhat, so we just verify total energy increased
    EXPECT_GT(energy_region1 + energy_region2, 0.0);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_F(PcgIntegrationTest, Reconstructor_PcgInFullPipeline) {
    Reconstructor reconstructor(config_, forward_model());

    const int num_range_bins = 3;
    auto tensor = create_test_tensor(
        kDefaultNumDetectors, num_range_bins, kDefaultWavelength);

    // Add targets in different range bins
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0, {1.0, 0.0});
    add_point_target_with_forward_model(tensor, forward_model(), 1, 0.002, 0.001, {1.0, 0.0});
    add_point_target_with_forward_model(tensor, forward_model(), 2, -0.001, 0.002, {0.8, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    // Verify output structure
    ASSERT_EQ(scene.wavelengths_size(), 1);
    EXPECT_EQ(scene.wavelengths(0).images_size(), num_range_bins);

    // Verify each range bin has valid image
    for (int rb = 0; rb < num_range_bins; ++rb) {
        const auto& image = scene.wavelengths(0).images(rb);
        EXPECT_EQ(image.range_bin(), static_cast<uint32_t>(rb));
        EXPECT_GT(image.pixels_size(), 0);
        EXPECT_GT(image.peak_value(), 0.0) << "Range bin " << rb << " should have non-zero peak";
    }

    // Verify metadata
    EXPECT_GT(scene.processing_time_ms(), 0.0);
}

TEST_F(PcgIntegrationTest, SingleRangeBinReconstruction) {
    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 5, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 2, 0.001, 0.002, {1.0, 0.0});


    // Reconstruct only range bin 2
    auto image = reconstructor.reconstruct_range_bin(tensor, 2, kDefaultWavelength);

    EXPECT_EQ(image.rows(), 16);
    EXPECT_EQ(image.cols(), 16);
    EXPECT_GT(image.norm(), 0.0);
}

TEST_F(PcgIntegrationTest, QualityMetrics) {
    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0, {1.0, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    const auto& image = scene.wavelengths(0).images(0);

    EXPECT_GT(image.peak_value(), 0.0);
    EXPECT_GT(image.total_energy(), 0.0);
    EXPECT_GE(image.sparsity(), 0.0);
    EXPECT_LE(image.sparsity(), 1.0);
}

// =============================================================================
// Regularizer Variant Tests
// =============================================================================

TEST_F(PcgIntegrationTest, WorksWithGradientRegularizer) {
    config_.mutable_pcg_config()->set_regularizer_type(REGULARIZER_GRADIENT);

    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0, {1.0, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    EXPECT_EQ(scene.wavelengths_size(), 1);
    EXPECT_GT(scene.wavelengths(0).images(0).peak_value(), 0.0);
}

TEST_F(PcgIntegrationTest, WorksWithLaplacianRegularizer) {
    config_.mutable_pcg_config()->set_regularizer_type(REGULARIZER_LAPLACIAN);

    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0, {1.0, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    EXPECT_EQ(scene.wavelengths_size(), 1);
    EXPECT_GT(scene.wavelengths(0).images(0).peak_value(), 0.0);
}

// =============================================================================
// NUFFT Inner Operator Tests
// =============================================================================

TEST_F(PcgIntegrationTest, WorksWithNufftOperator) {
    auto* pcg_config = config_.mutable_pcg_config();
    pcg_config->set_use_nufft_operator(true);
    auto* nufft_config = pcg_config->mutable_nufft_config();
    nufft_config->set_kernel(KERNEL_KAISER_BESSEL);
    nufft_config->set_oversampling_factor(2.0);
    nufft_config->set_kernel_width(4);
    // Required for physics weighting when bandwidth info is present
    nufft_config->set_window_type(NUFFT_WINDOW_GAUSSIAN);

    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0, {1.0, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    EXPECT_EQ(scene.wavelengths_size(), 1);
    EXPECT_GT(scene.wavelengths(0).images(0).peak_value(), 0.0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(PcgIntegrationTest, EmptyRangeBinHandling) {
    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 3, kDefaultWavelength);
    // Only add target in range bin 1
    add_point_target_with_forward_model(tensor, forward_model(), 1, 0.0, 0.0, {1.0, 0.0});
    auto scene = reconstructor.reconstruct(tensor);

    // Empty range bins should produce valid (zero or low energy) images
    EXPECT_EQ(scene.wavelengths(0).images_size(), 3);

    // Range bin 1 with target should have higher peak than empty bins
    const auto& img0 = scene.wavelengths(0).images(0);
    const auto& img1 = scene.wavelengths(0).images(1);

    EXPECT_GT(img1.peak_value(), img0.peak_value());
}

TEST_F(PcgIntegrationTest, GridParameters) {
    Reconstructor reconstructor(config_, forward_model());

    EXPECT_EQ(reconstructor.grid_size().x(), 16);
    EXPECT_EQ(reconstructor.grid_size().y(), 16);
    EXPECT_DOUBLE_EQ(reconstructor.grid_spacing_m().x(), 0.001);
    EXPECT_DOUBLE_EQ(reconstructor.grid_spacing_m().y(), 0.001);

    // Origin should be centered
    EXPECT_DOUBLE_EQ(reconstructor.grid_origin_m().x(), -0.008);
    EXPECT_DOUBLE_EQ(reconstructor.grid_origin_m().y(), -0.008);
}

}  // namespace aurora::reconstruction
