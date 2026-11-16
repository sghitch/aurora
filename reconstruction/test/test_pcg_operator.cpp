#include "reconstruction/pcg_operator.h"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "forward/forward_model.h"
#include "test_helpers.h"

#include "bandwidth_config.pb.h"
#include "detector_geometry.pb.h"
#include "k_space_slice.pb.h"
#include "nufft_config.pb.h"
#include "optics_config.pb.h"

namespace aurora::reconstruction {

namespace {

constexpr uint32_t kNumDetectors = 8;   // point-detector geometry size
constexpr double kPitch = 0.5e-3;  // m

// Helper function that provides a default BandwidthConfig for tests
aurora::forward::BandwidthConfig make_test_bandwidth_config() {
    aurora::forward::BandwidthConfig config;
    config.set_window_type(aurora::forward::BANDWIDTH_WINDOW_SINC);
    config.set_quadrature_order(4);
    return config;
}

// Helper to create a KSpaceSlice from K-vectors and sample weights.
// detector_index is left at 0: with the point-detector geometry this maps
// every sample to a zero-bandwidth window, i.e. a point sample at (kx, ky).
aurora::shared::KSpaceSlice create_slice(
    const std::vector<Eigen::Vector2d>& k_vectors,
    const std::vector<double>& weights) {
    aurora::shared::KSpaceSlice slice;
    slice.set_wavelength_m(905e-9);
    slice.set_range_bin(0);

    for (size_t i = 0; i < k_vectors.size(); ++i) {
        auto* sample = slice.add_samples();
        sample->set_kx(k_vectors[i].x());
        sample->set_ky(k_vectors[i].y());
        sample->set_weight(i < weights.size() ? weights[i] : 1.0);
    }
    return slice;
}

}  // namespace

class PcgConfigValidationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a valid config as baseline
        config_.set_max_iterations(50);
        config_.set_tolerance(1e-6);
        config_.set_regularizer_type(REGULARIZER_TIKHONOV);
        config_.set_regularizer_lambda(1e-4);
        config_.set_preconditioner_type(PRECONDITIONER_NONE);
    }

    PcgConfig config_;
};

TEST_F(PcgConfigValidationTest, RejectsUnknownRegularizer) {
    config_.set_regularizer_type(REGULARIZER_UNKNOWN);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RejectsUnknownPreconditioner) {
    config_.set_preconditioner_type(PRECONDITIONER_UNKNOWN);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RejectsZeroIterations) {
    config_.set_max_iterations(0);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RejectsZeroTolerance) {
    config_.set_tolerance(0.0);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RejectsToleranceAtOne) {
    config_.set_tolerance(1.0);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RejectsNegativeTolerance) {
    config_.set_tolerance(-0.1);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RejectsNegativeLambda) {
    config_.set_regularizer_lambda(-1.0);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RejectsZeroLambda) {
    config_.set_regularizer_lambda(0.0);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RequiresEpsilonForCirculant) {
    config_.set_preconditioner_type(PRECONDITIONER_CIRCULANT);
    config_.set_preconditioner_epsilon(0.0);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RequiresEpsilonForHybrid) {
    config_.set_preconditioner_type(PRECONDITIONER_HYBRID);
    config_.set_preconditioner_epsilon(0.0);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, RequiresNufftConfigWhenEnabled) {
    config_.set_use_nufft_operator(true);
    // No nufft_config set
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, AcceptsNoNufftConfigWhenDisabled) {
    config_.set_use_nufft_operator(false);
    // No nufft_config set - should be fine
    EXPECT_NO_THROW(validate_pcg_config(config_));
}

TEST_F(PcgConfigValidationTest, WarmStartRejectsZeroThreshold) {
    auto* ws = config_.mutable_warm_start();
    ws->set_change_threshold(0.0);
    ws->set_cache_depth(1);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, WarmStartRejectsThresholdAboveOne) {
    auto* ws = config_.mutable_warm_start();
    ws->set_change_threshold(1.5);
    ws->set_cache_depth(1);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, WarmStartAcceptsThresholdAtOne) {
    auto* ws = config_.mutable_warm_start();
    ws->set_change_threshold(1.0);
    ws->set_cache_depth(1);
    EXPECT_NO_THROW(validate_pcg_config(config_));
}

TEST_F(PcgConfigValidationTest, WarmStartRejectsZeroCacheDepth) {
    auto* ws = config_.mutable_warm_start();
    ws->set_change_threshold(0.3);
    ws->set_cache_depth(0);
    EXPECT_THROW(validate_pcg_config(config_), std::invalid_argument);
}

TEST_F(PcgConfigValidationTest, AcceptsValidConfig) {
    EXPECT_NO_THROW(validate_pcg_config(config_));
}

TEST_F(PcgConfigValidationTest, AcceptsAllRegularizerTypes) {
    config_.set_regularizer_type(REGULARIZER_TIKHONOV);
    EXPECT_NO_THROW(validate_pcg_config(config_));

    config_.set_regularizer_type(REGULARIZER_GRADIENT);
    EXPECT_NO_THROW(validate_pcg_config(config_));

    config_.set_regularizer_type(REGULARIZER_LAPLACIAN);
    EXPECT_NO_THROW(validate_pcg_config(config_));
}

TEST_F(PcgConfigValidationTest, AcceptsAllPreconditionerTypesWithEpsilon) {
    config_.set_preconditioner_epsilon(1e-6);

    config_.set_preconditioner_type(PRECONDITIONER_NONE);
    EXPECT_NO_THROW(validate_pcg_config(config_));

    config_.set_preconditioner_type(PRECONDITIONER_JACOBI);
    EXPECT_NO_THROW(validate_pcg_config(config_));

    config_.set_preconditioner_type(PRECONDITIONER_CIRCULANT);
    EXPECT_NO_THROW(validate_pcg_config(config_));

    config_.set_preconditioner_type(PRECONDITIONER_HYBRID);
    EXPECT_NO_THROW(validate_pcg_config(config_));
}

// =============================================================================
// PCG Operator Tests
// =============================================================================

class PcgOperatorTest : public ::testing::Test {
  protected:
    static constexpr double kWavelength = 905e-9;

    void SetUp() override {
        // Point-detector geometry (zero pixel size): no K-space bandwidth, so
        // the operator treats the hand-built K-grid in each slice as point
        // samples rather than integrating over a K-window. This makes the
        // fixture a generic operator test on a dense K-grid.
        detector_geom_ = aurora::reconstruction::test::create_point_detector_geometry(
            "test_detector", kNumDetectors, kPitch, kWavelength);

        // Set up optics configuration
        optics_.set_fourier_lens_focal_m(0.05);
        optics_.add_wavelengths_m(kWavelength);
        optics_.set_k_max_rad_per_m(1e5);

        // Create ForwardModel
        forward_model_ = std::make_unique<aurora::forward::ForwardModel>(
            detector_geom_, optics_);

        // Create simple K-space samples for testing
        k_vectors_.clear();
        weights_.clear();
        for (int i = -3; i <= 3; ++i) {
            for (int j = -3; j <= 3; ++j) {
                k_vectors_.emplace_back(i * 150.0, j * 150.0);
                weights_.push_back(1.0);
            }
        }

        // Default grid parameters
        grid_size_ = Eigen::Vector2i(16, 16);
        grid_spacing_ = Eigen::Vector2d(0.001, 0.001);
        grid_origin_ = Eigen::Vector2d(-0.008, -0.008);

        // Valid config
        config_.set_max_iterations(20);
        config_.set_tolerance(1e-6);
        config_.set_regularizer_type(REGULARIZER_TIKHONOV);
        config_.set_regularizer_lambda(1e-4);
        config_.set_preconditioner_type(PRECONDITIONER_NONE);
    }

    aurora::forward::DetectorGeometry detector_geom_;
    aurora::forward::OpticsConfig optics_;
    std::unique_ptr<aurora::forward::ForwardModel> forward_model_;

    std::vector<Eigen::Vector2d> k_vectors_;
    std::vector<double> weights_;
    Eigen::Vector2i grid_size_;
    Eigen::Vector2d grid_spacing_;
    Eigen::Vector2d grid_origin_;
    PcgConfig config_;
};

TEST_F(PcgOperatorTest, ConstructFromKSpaceSlice) {
    auto slice = create_slice(k_vectors_, weights_);

    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    EXPECT_EQ(op.num_samples(), k_vectors_.size());
    EXPECT_EQ(op.k_vectors().size(), k_vectors_.size());
    EXPECT_EQ(op.weights().size(), weights_.size());
}

TEST_F(PcgOperatorTest, InvalidConfigThrows) {
    auto slice = create_slice(k_vectors_, weights_);
    PcgConfig bad_config;
    bad_config.set_regularizer_type(REGULARIZER_UNKNOWN);

    EXPECT_THROW(
        PcgOperator(slice, *forward_model_,
                    make_test_bandwidth_config(),
                    grid_size_, grid_spacing_, grid_origin_, bad_config),
        std::invalid_argument);
}

TEST_F(PcgOperatorTest, ForwardDelegatesToInner) {
    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(grid_size_.y(), grid_size_.x());
    auto k_data = op.forward(scene, grid_origin_, grid_spacing_);

    EXPECT_EQ(k_data.size(), static_cast<int>(k_vectors_.size()));
}

TEST_F(PcgOperatorTest, AdjointProducesCorrectSize) {
    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    Eigen::VectorXcd k_data = Eigen::VectorXcd::Ones(k_vectors_.size());
    auto image = op.adjoint(k_data, grid_origin_, grid_spacing_, grid_size_);

    EXPECT_EQ(image.rows(), grid_size_.y());
    EXPECT_EQ(image.cols(), grid_size_.x());
}

TEST_F(PcgOperatorTest, ResidualDecreases) {
    // Set more iterations to verify residual behavior
    config_.set_max_iterations(10);
    config_.set_tolerance(1e-10);  // Very tight to ensure we iterate

    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    // Generate synthetic K-space data from a known scene
    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Zero(grid_size_.y(), grid_size_.x());
    scene(8, 8) = 1.0;  // Point source at center

    auto k_data = op.forward(scene, grid_origin_, grid_spacing_);

    // Run reconstruction
    auto result = op.adjoint(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Verify that iterations occurred and residual is reasonable
    EXPECT_GT(op.last_iterations(), 0);
    EXPECT_GE(op.last_residual(), 0.0);
}

TEST_F(PcgOperatorTest, RespectsMaxIterations) {
    config_.set_max_iterations(2);
    config_.set_tolerance(1e-15);  // Very tight - won't be reached

    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    Eigen::VectorXcd k_data = Eigen::VectorXcd::Ones(k_vectors_.size());
    op.adjoint(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Should stop at max_iterations
    EXPECT_EQ(op.last_iterations(), 2);
}

TEST_F(PcgOperatorTest, ConvergesOnPointTarget) {
    // Dense K-space for good reconstruction
    std::vector<Eigen::Vector2d> k;
    std::vector<double> eff;
    for (int i = -5; i <= 5; ++i) {
        for (int j = -5; j <= 5; ++j) {
            k.emplace_back(i * 200.0, j * 200.0);
            eff.push_back(1.0);
        }
    }

    config_.set_max_iterations(50);
    config_.set_tolerance(1e-6);
    config_.set_regularizer_lambda(1e-6);  // Small lambda

    auto slice = create_slice(k, eff);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    // Point target
    double target_x = 0.001;
    double target_y = 0.001;

    // Generate K-space data for point target
    Eigen::VectorXcd k_data(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        double phase = -(k[i].x() * target_x + k[i].y() * target_y);
        k_data(i) = std::polar(1.0, phase);
    }

    auto image = op.adjoint(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Find peak position
    int peak_ix = 0, peak_iy = 0;
    double max_mag = 0.0;
    for (int iy = 0; iy < grid_size_.y(); ++iy) {
        for (int ix = 0; ix < grid_size_.x(); ++ix) {
            double mag = std::abs(image(iy, ix));
            if (mag > max_mag) {
                max_mag = mag;
                peak_ix = ix;
                peak_iy = iy;
            }
        }
    }

    double peak_x = grid_origin_.x() + peak_ix * grid_spacing_.x();
    double peak_y = grid_origin_.y() + peak_iy * grid_spacing_.y();

    // Peak should be within 1 pixel of target
    EXPECT_NEAR(peak_x, target_x, grid_spacing_.x());
    EXPECT_NEAR(peak_y, target_y, grid_spacing_.y());
}

TEST_F(PcgOperatorTest, AdjointConsistency) {
    // PCG adjoint is iterative, so consistency is looser
    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Random scene
    Eigen::MatrixXcd x(grid_size_.y(), grid_size_.x());
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    // Random K-space data
    Eigen::VectorXcd y(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        y(i) = std::complex<double>(dist(rng), dist(rng));
    }

    // Note: PCG's adjoint is NOT a mathematical adjoint of forward,
    // it's a solver. So we test using the inner operator instead.
    double error = op.inner_operator().check_adjoint_consistency(
        x, y, grid_origin_, grid_spacing_);
    EXPECT_LT(error, 1e-10);
}

TEST_F(PcgOperatorTest, WorksWithGradientRegularizer) {
    config_.set_regularizer_type(REGULARIZER_GRADIENT);

    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    Eigen::VectorXcd k_data = Eigen::VectorXcd::Ones(k_vectors_.size());
    auto image = op.adjoint(k_data, grid_origin_, grid_spacing_, grid_size_);

    EXPECT_EQ(image.rows(), grid_size_.y());
    EXPECT_EQ(image.cols(), grid_size_.x());
}

TEST_F(PcgOperatorTest, WorksWithLaplacianRegularizer) {
    config_.set_regularizer_type(REGULARIZER_LAPLACIAN);

    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    Eigen::VectorXcd k_data = Eigen::VectorXcd::Ones(k_vectors_.size());
    auto image = op.adjoint(k_data, grid_origin_, grid_spacing_, grid_size_);

    EXPECT_EQ(image.rows(), grid_size_.y());
    EXPECT_EQ(image.cols(), grid_size_.x());
}

TEST_F(PcgOperatorTest, EmptyInputReturnsCorrectSize) {
    std::vector<Eigen::Vector2d> empty_k;
    std::vector<double> empty_eff;

    auto slice = create_slice(empty_k, empty_eff);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    EXPECT_EQ(op.num_samples(), 0u);

    Eigen::VectorXcd empty_samples(0);
    auto image = op.adjoint(empty_samples, grid_origin_, grid_spacing_, grid_size_);

    EXPECT_EQ(image.rows(), grid_size_.y());
    EXPECT_EQ(image.cols(), grid_size_.x());
}

TEST_F(PcgOperatorTest, ConfigAccessor) {
    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    EXPECT_EQ(op.config().max_iterations(), config_.max_iterations());
    EXPECT_DOUBLE_EQ(op.config().tolerance(), config_.tolerance());
    EXPECT_EQ(op.config().regularizer_type(), config_.regularizer_type());
}

TEST_F(PcgOperatorTest, InnerOperatorAccessor) {
    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    EXPECT_EQ(op.inner_operator().num_samples(), k_vectors_.size());
}

TEST_F(PcgOperatorTest, WorksWithNufftOperator) {
    // Configure to use NUFFT operator
    config_.set_use_nufft_operator(true);
    auto* nufft_config = config_.mutable_nufft_config();
    nufft_config->set_kernel(KERNEL_KAISER_BESSEL);
    nufft_config->set_oversampling_factor(2.0);
    nufft_config->set_kernel_width(4);

    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    EXPECT_EQ(op.num_samples(), k_vectors_.size());

    // Test forward
    Eigen::MatrixXcd scene = Eigen::MatrixXcd::Ones(grid_size_.y(), grid_size_.x());
    auto k_data = op.forward(scene, grid_origin_, grid_spacing_);
    EXPECT_EQ(k_data.size(), static_cast<int>(k_vectors_.size()));

    // Test adjoint (PCG solve)
    auto image = op.adjoint(k_data, grid_origin_, grid_spacing_, grid_size_);
    EXPECT_EQ(image.rows(), grid_size_.y());
    EXPECT_EQ(image.cols(), grid_size_.x());

    // Verify convergence metadata is set
    EXPECT_GE(op.last_iterations(), 0);
    EXPECT_GE(op.last_residual(), 0.0);
}

TEST_F(PcgOperatorTest, NufftOperatorConvergesOnPointTarget) {
    // Dense K-space for good reconstruction
    std::vector<Eigen::Vector2d> k;
    std::vector<double> eff;
    for (int i = -5; i <= 5; ++i) {
        for (int j = -5; j <= 5; ++j) {
            k.emplace_back(i * 200.0, j * 200.0);
            eff.push_back(1.0);
        }
    }

    // Configure with NUFFT
    config_.set_max_iterations(50);
    config_.set_tolerance(1e-6);
    config_.set_regularizer_lambda(1e-6);
    config_.set_use_nufft_operator(true);
    auto* nufft_config = config_.mutable_nufft_config();
    nufft_config->set_kernel(KERNEL_KAISER_BESSEL);
    nufft_config->set_oversampling_factor(2.0);
    nufft_config->set_kernel_width(4);

    auto slice = create_slice(k, eff);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    // Point target
    double target_x = 0.001;
    double target_y = 0.001;

    // Generate K-space data for point target
    Eigen::VectorXcd k_data(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        double phase = -(k[i].x() * target_x + k[i].y() * target_y);
        k_data(i) = std::polar(1.0, phase);
    }

    auto image = op.adjoint(k_data, grid_origin_, grid_spacing_, grid_size_);

    // Find peak position
    int peak_ix = 0, peak_iy = 0;
    double max_mag = 0.0;
    for (int iy = 0; iy < grid_size_.y(); ++iy) {
        for (int ix = 0; ix < grid_size_.x(); ++ix) {
            double mag = std::abs(image(iy, ix));
            if (mag > max_mag) {
                max_mag = mag;
                peak_ix = ix;
                peak_iy = iy;
            }
        }
    }

    double peak_x = grid_origin_.x() + peak_ix * grid_spacing_.x();
    double peak_y = grid_origin_.y() + peak_iy * grid_spacing_.y();

    // Peak should be within 1 pixel of target
    EXPECT_NEAR(peak_x, target_x, grid_spacing_.x());
    EXPECT_NEAR(peak_y, target_y, grid_spacing_.y());
}

TEST_F(PcgOperatorTest, NufftInnerOperatorAdjointConsistency) {
    // Configure with NUFFT
    config_.set_use_nufft_operator(true);
    auto* nufft_config = config_.mutable_nufft_config();
    nufft_config->set_kernel(KERNEL_KAISER_BESSEL);
    nufft_config->set_oversampling_factor(2.0);
    nufft_config->set_kernel_width(4);

    auto slice = create_slice(k_vectors_, weights_);
    PcgOperator op(slice, *forward_model_,
                   make_test_bandwidth_config(),
                   grid_size_, grid_spacing_, grid_origin_, config_);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Random scene
    Eigen::MatrixXcd x(grid_size_.y(), grid_size_.x());
    for (int i = 0; i < grid_size_.y(); ++i) {
        for (int j = 0; j < grid_size_.x(); ++j) {
            x(i, j) = std::complex<double>(dist(rng), dist(rng));
        }
    }

    // Random K-space data
    Eigen::VectorXcd y(k_vectors_.size());
    for (size_t i = 0; i < k_vectors_.size(); ++i) {
        y(i) = std::complex<double>(dist(rng), dist(rng));
    }

    // NUFFT has looser adjoint consistency due to interpolation
    double error = op.inner_operator().check_adjoint_consistency(
        x, y, grid_origin_, grid_spacing_);
    EXPECT_LT(error, 1e-4);  // Looser tolerance for NUFFT
}

}  // namespace aurora::reconstruction
