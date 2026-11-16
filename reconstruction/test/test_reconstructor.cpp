#include "reconstruction/reconstructor.h"

#include <cmath>

#include <gtest/gtest.h>

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

class ReconstructorTest : public ::testing::Test {
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
        config_.set_algorithm(ALGORITHM_DIRECT_ADJOINT);
        config_.set_grid_size_x(16);
        config_.set_grid_size_y(16);
        config_.set_pixel_size_m(0.001);  // 1mm pixels

        // Set up bandwidth config (required for DirectOperator/PcgOperator)
        auto* bw_config = config_.mutable_bandwidth_config();
        bw_config->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
        bw_config->set_quadrature_order(4);

        // Set required direct_adjoint_config for ALGORITHM_DIRECT_ADJOINT
        config_.mutable_direct_adjoint_config()->set_apodization(
            DirectAdjointConfig::APOD_NONE);
    }

    const aurora::forward::ForwardModel& forward_model() const {
        return *shared_forward_model_;
    }

    // Helper to create a MeasurementTensor for one burst: one wavelength
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
aurora::forward::DetectorGeometry* ReconstructorTest::detector_ = nullptr;
aurora::forward::OpticsConfig* ReconstructorTest::optics_ = nullptr;
aurora::forward::ForwardModel* ReconstructorTest::shared_forward_model_ = nullptr;

// Test gridding reconstruction with point target
TEST_F(ReconstructorTest, GriddingPointTarget) {
    config_.set_algorithm(ALGORITHM_GRIDDING_IFFT);
    auto* gridding_config = config_.mutable_gridding_config();
    gridding_config->set_kernel(GRIDDING_KERNEL_GAUSSIAN);
    gridding_config->set_kernel_width(3);

    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0, {1.0, 0.0});

    auto scene = reconstructor.reconstruct(tensor);

    EXPECT_EQ(scene.wavelengths_size(), 1);
    EXPECT_EQ(scene.wavelengths(0).images_size(), 1);

    const auto& image = scene.wavelengths(0).images(0);
    EXPECT_GT(image.pixels_size(), 0);
    EXPECT_GT(image.peak_value(), 0.0);
}

// Test direct adjoint reconstruction
TEST_F(ReconstructorTest, AdjointReconstruction) {
    config_.set_algorithm(ALGORITHM_DIRECT_ADJOINT);

    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);
    add_point_target_with_forward_model(
        tensor, forward_model(), 0, 0.001, 0.002, {1.0, 0.0});

    auto scene = reconstructor.reconstruct(tensor);

    EXPECT_EQ(scene.wavelengths_size(), 1);

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
    EXPECT_NEAR(peak_x, 0.001, 0.004);
    EXPECT_NEAR(peak_y, 0.002, 0.004);
}

// Test an algorithm value outside the enum throws a configuration error
TEST_F(ReconstructorTest, UnsupportedAlgorithmThrows) {
    config_.set_algorithm(static_cast<ReconstructionAlgorithm>(99));

    EXPECT_THROW(
        Reconstructor(config_, forward_model()),
        ReconstructionError);
}

// Test GRIDDING_IFFT without gridding_config throws error
TEST_F(ReconstructorTest, GriddingWithoutConfigThrows) {
    config_.set_algorithm(ALGORITHM_GRIDDING_IFFT);
    // No gridding_config set

    EXPECT_THROW(
        Reconstructor(config_, forward_model()),
        ReconstructionError);
}

// Test GRIDDING_IFFT with UNSPECIFIED kernel throws error
TEST_F(ReconstructorTest, GriddingWithUnspecifiedKernelThrows) {
    config_.set_algorithm(ALGORITHM_GRIDDING_IFFT);
    auto* gridding_config = config_.mutable_gridding_config();
    gridding_config->set_kernel(GRIDDING_KERNEL_UNSPECIFIED);

    EXPECT_THROW(
        Reconstructor(config_, forward_model()),
        ReconstructionError);
}

// Test empty input handling: a tensor without data has no K-space samples
TEST_F(ReconstructorTest, EmptyInput) {
    Reconstructor reconstructor(config_, forward_model());

    aurora::shared::MeasurementTensor empty_tensor;
    EXPECT_THROW(reconstructor.reconstruct(empty_tensor),
                 ReconstructionError);
}

// Test single range bin reconstruction
TEST_F(ReconstructorTest, SingleRangeBinReconstruction) {
    config_.set_algorithm(ALGORITHM_DIRECT_ADJOINT);

    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 5, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 2, 0.001, 0.002, {1.0, 0.0});
    add_point_target_with_forward_model(tensor, forward_model(), 4, -0.002, 0.001, {0.8, 0.2});

    // Reconstruct only range bin 2
    auto image = reconstructor.reconstruct_range_bin(tensor, 2, kDefaultWavelength);

    EXPECT_EQ(image.rows(), 16);
    EXPECT_EQ(image.cols(), 16);
    EXPECT_GT(image.norm(), 0.0);

    double max_val = 0.0;
    for (int i = 0; i < image.rows(); ++i) {
        for (int j = 0; j < image.cols(); ++j) {
            max_val = std::max(max_val, std::abs(image(i, j)));
        }
    }
    EXPECT_GT(max_val, 0.0);
}

// Test a range bin / wavelength without samples yields a zero image
TEST_F(ReconstructorTest, RangeBinWithoutSamplesIsZero) {
    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 2, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0);

    // No detector records at this wavelength
    auto image = reconstructor.reconstruct_range_bin(tensor, 0, 1550e-9);

    EXPECT_EQ(image.rows(), 16);
    EXPECT_EQ(image.cols(), 16);
    EXPECT_DOUBLE_EQ(image.norm(), 0.0);
}

// Test processing time tracking
TEST_F(ReconstructorTest, ProcessingTimeTracking) {
    config_.set_algorithm(ALGORITHM_DIRECT_ADJOINT);

    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 1, kDefaultWavelength);
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0);

    auto scene = reconstructor.reconstruct(tensor);

    EXPECT_GT(scene.processing_time_ms(), 0.0);
}

// Test grid parameters
TEST_F(ReconstructorTest, GridParameters) {
    Reconstructor reconstructor(config_, forward_model());

    EXPECT_EQ(reconstructor.grid_size().x(), 16);
    EXPECT_EQ(reconstructor.grid_size().y(), 16);
    EXPECT_DOUBLE_EQ(reconstructor.grid_spacing_m().x(), 0.001);
    EXPECT_DOUBLE_EQ(reconstructor.grid_spacing_m().y(), 0.001);

    // Origin should be centered
    EXPECT_DOUBLE_EQ(reconstructor.grid_origin_m().x(), -0.008);
    EXPECT_DOUBLE_EQ(reconstructor.grid_origin_m().y(), -0.008);
}

// Test reconstruction quality metrics
TEST_F(ReconstructorTest, QualityMetrics) {
    config_.set_algorithm(ALGORITHM_DIRECT_ADJOINT);

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

// Test algorithm accessor
TEST_F(ReconstructorTest, AlgorithmAccessor) {
    config_.set_algorithm(ALGORITHM_DIRECT_ADJOINT);
    Reconstructor reconstructor(config_, forward_model());
    EXPECT_EQ(reconstructor.algorithm(), ALGORITHM_DIRECT_ADJOINT);

    ReconstructionConfig config2;
    config2.set_algorithm(ALGORITHM_GRIDDING_IFFT);
    config2.set_grid_size_x(16);
    config2.set_grid_size_y(16);
    config2.set_pixel_size_m(0.001);
    auto* bw_config2 = config2.mutable_bandwidth_config();
    bw_config2->set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    bw_config2->set_quadrature_order(4);
    auto* gridding_config = config2.mutable_gridding_config();
    gridding_config->set_kernel(GRIDDING_KERNEL_GAUSSIAN);

    Reconstructor reconstructor2(config2, forward_model());
    EXPECT_EQ(reconstructor2.algorithm(), ALGORITHM_GRIDDING_IFFT);
}

// ============================================================
// Multi-Range-Bin Reconstruction Tests
// ============================================================

// Test that multi-range-bin reconstruction produces distinct images per bin
TEST_F(ReconstructorTest, MultiRangeBinReconstruction) {
    config_.set_algorithm(ALGORITHM_DIRECT_ADJOINT);

    Reconstructor reconstructor(config_, forward_model());

    auto tensor = create_test_tensor(
        kDefaultNumDetectors, 3, kDefaultWavelength);

    // Add target at different position for each range bin
    add_point_target_with_forward_model(tensor, forward_model(), 0, 0.0, 0.0, {1.0, 0.0});
    add_point_target_with_forward_model(tensor, forward_model(), 1, 0.003, 0.0, {1.0, 0.0});
    add_point_target_with_forward_model(tensor, forward_model(), 2, -0.003, 0.0, {1.0, 0.0});

    auto img0 = reconstructor.reconstruct_range_bin(tensor, 0, kDefaultWavelength);
    auto img1 = reconstructor.reconstruct_range_bin(tensor, 1, kDefaultWavelength);
    auto img2 = reconstructor.reconstruct_range_bin(tensor, 2, kDefaultWavelength);

    EXPECT_GT(img0.norm(), 0.0) << "Range bin 0 should have non-zero content";
    EXPECT_GT(img1.norm(), 0.0) << "Range bin 1 should have non-zero content";
    EXPECT_GT(img2.norm(), 0.0) << "Range bin 2 should have non-zero content";

    auto find_peak_col = [](const Eigen::MatrixXcd& img) {
        int max_col = 0;
        double max_val = 0.0;
        for (int row = 0; row < img.rows(); ++row) {
            for (int col = 0; col < img.cols(); ++col) {
                double val = std::abs(img(row, col));
                if (val > max_val) {
                    max_val = val;
                    max_col = col;
                }
            }
        }
        return max_col;
    };

    int peak0 = find_peak_col(img0);
    int peak1 = find_peak_col(img1);
    int peak2 = find_peak_col(img2);

    EXPECT_NE(peak0, peak1)
        << "Peaks for range bin 0 and 1 should be at different positions";
    EXPECT_NE(peak0, peak2)
        << "Peaks for range bin 0 and 2 should be at different positions";
    EXPECT_NE(peak1, peak2)
        << "Peaks for range bin 1 and 2 should be at different positions";
}

// Test full reconstruct() produces correct number of images per range bin
TEST_F(ReconstructorTest, FullReconstructCreatesPerRangeBinImages) {
    config_.set_algorithm(ALGORITHM_DIRECT_ADJOINT);

    Reconstructor reconstructor(config_, forward_model());

    const int num_range_bins = 5;
    auto tensor = create_test_tensor(
        kDefaultNumDetectors, num_range_bins, kDefaultWavelength);

    add_point_target_with_forward_model(tensor, forward_model(), 1, 0.0, 0.0, {1.0, 0.0});
    add_point_target_with_forward_model(tensor, forward_model(), 3, 0.002, 0.001, {1.0, 0.0});

    auto scene = reconstructor.reconstruct(tensor);

    ASSERT_EQ(scene.wavelengths_size(), 1);
    EXPECT_EQ(scene.wavelengths(0).images_size(), num_range_bins)
        << "Should produce one image per range bin";

    const auto& img_bin1 = scene.wavelengths(0).images(1);
    const auto& img_bin3 = scene.wavelengths(0).images(3);
    const auto& img_bin0 = scene.wavelengths(0).images(0);

    EXPECT_GT(img_bin1.peak_value(), img_bin0.peak_value())
        << "Bin 1 with target should have higher peak than empty bin 0";
    EXPECT_GT(img_bin3.peak_value(), img_bin0.peak_value())
        << "Bin 3 with target should have higher peak than empty bin 0";
}

}  // namespace aurora::reconstruction
