/**
 * @file test_detector.cpp
 * @brief Unit tests for Detector with phase-gradient angle estimation.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "detection/detector.h"
#include "detection/detection_error.h"

#include "detection_config.pb.h"
#include "forward_config.pb.h"
#include "measurement_tensor.pb.h"
#include "sequence_schedule.pb.h"

using namespace aurora::detection;

namespace {

constexpr double kFocalLength = 0.2;      // meters
constexpr double kWavelength = 1550e-9;   // meters
constexpr double kPitch = 0.5e-3;         // meters
constexpr uint32_t kNumDetectors = 8;
constexpr uint32_t kNumRangeBins = 64;
constexpr double kRangeStep = 0.15;       // meters per bin
constexpr double kChirpDuration = 10e-6;  // seconds
constexpr double kTwoPi = 2.0 * M_PI;

double detector_k_step() {
    return kTwoPi * kPitch / (kFocalLength * kWavelength);
}

/// Unambiguous azimuth period of the uniformly spaced detector array.
double azimuth_period() {
    return kTwoPi / (kFocalLength * detector_k_step());
}

DetectionConfig create_detection_config() {
    DetectionConfig config;
    config.set_magnitude_threshold_db(-30.0);
    config.set_min_pixels_for_gradient(3);
    config.set_max_phase_residual_rad(0.5);
    config.set_max_condition_number(1e12);
    config.set_enable_doppler(false);
    config.set_noise_floor_db(-80.0);
    return config;
}

aurora::forward::ForwardConfig create_forward_config() {
    aurora::forward::ForwardConfig config;

    auto* optics = config.mutable_optics();
    optics->set_fourier_lens_focal_m(kFocalLength);
    optics->add_wavelengths_m(kWavelength);
    optics->set_system_fov_x_rad(0.2);
    optics->set_system_fov_y_rad(0.2);

    auto* array = config.mutable_detector()->add_arrays();
    array->set_array_id("test");
    array->set_nominal_pitch_m(kPitch);
    array->add_wavelengths_m(kWavelength);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_POINT);
    for (uint32_t i = 0; i < kNumDetectors; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((static_cast<double>(i) - 3.5) * kPitch);
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.4e-3);
        pixel->set_height_m(0.4e-3);
    }
    return config;
}

aurora::acquisition::SequenceSchedule create_sequence_schedule() {
    aurora::acquisition::SequenceSchedule schedule;
    auto* chirp = schedule.mutable_chirp_schedule();
    chirp->set_chirp_duration_s(kChirpDuration);
    chirp->set_chirp_bandwidth_hz(1e9);
    chirp->set_chirps_per_burst(2);
    chirp->set_mode(aurora::acquisition::ChirpSchedule::CHIRP_MODE_UP_DOWN);
    return schedule;
}

struct SyntheticTarget {
    uint32_t range_bin;
    double theta_x;        // radians
    double amplitude;
    double velocity_mps = 0.0;
};

/// Fill one WavelengthChannel record with the responses of the 8 detector
/// pixels to the targets for a single burst. `direction_sign` is +1 for up
/// chirps and -1 for down chirps: the Doppler phase advances in opposite
/// directions, matching DopplerExtractor's model.
void fill_wavelength(aurora::shared::WavelengthChannel* wl,
                     const std::vector<SyntheticTarget>& targets,
                     double direction_sign) {
    wl->set_wavelength_m(kWavelength);
    const double step = detector_k_step();

    for (uint32_t d = 0; d < kNumDetectors; ++d) {
        auto* det = wl->add_detectors();
        det->set_detector_index(d);
        Eigen::Vector2d k((static_cast<double>(d) - 3.5) * step, 0.0);
        det->mutable_k_vector()->set_kx(k.x());
        det->mutable_k_vector()->set_ky(k.y());

        for (uint32_t bin = 0; bin < kNumRangeBins; ++bin) {
            std::complex<double> value(0.0, 0.0);
            for (const auto& target : targets) {
                if (target.range_bin != bin) continue;
                Eigen::Vector2d r(kFocalLength * target.theta_x, 0.0);
                double doppler_phase = direction_sign * kTwoPi * target.velocity_mps *
                                       kChirpDuration / kWavelength;
                value += std::polar(target.amplitude, -k.dot(r) + doppler_phase);
            }
            auto* sample = det->add_range_data();
            sample->set_real(value.real());
            sample->set_imag(value.imag());
        }
    }
}

/// Build a one-burst MeasurementTensor for the targets. With `with_doppler`,
/// the up- and down-chirp range profiles are included.
aurora::shared::MeasurementTensor create_tensor(
    const std::vector<SyntheticTarget>& targets,
    bool with_doppler = false) {
    aurora::shared::MeasurementTensor tensor;
    tensor.set_burst_id(1);
    tensor.set_timestamp_s(0.0);

    auto* axis = tensor.mutable_range_axis();
    axis->set_unit("m");
    axis->set_start(0.0);
    axis->set_step(kRangeStep);
    axis->set_count(kNumRangeBins);

    fill_wavelength(tensor.add_data(), targets, 0.0);
    if (with_doppler) {
        fill_wavelength(tensor.add_up_chirp_data(), targets, 1.0);
        fill_wavelength(tensor.add_down_chirp_data(), targets, -1.0);
    }
    return tensor;
}

class DetectorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_ = create_detection_config();
        forward_config_ = create_forward_config();
        sequence_schedule_ = create_sequence_schedule();
    }

    DetectionConfig config_;
    aurora::forward::ForwardConfig forward_config_;
    aurora::acquisition::SequenceSchedule sequence_schedule_;
};

}  // namespace

TEST_F(DetectorTest, ConstructsWithValidConfig) {
    EXPECT_NO_THROW({
        Detector detector(config_, forward_config_, sequence_schedule_);
    });
}

TEST_F(DetectorTest, ThrowsOnMissingOptics) {
    aurora::forward::ForwardConfig bad = forward_config_;
    bad.clear_optics();
    EXPECT_THROW(Detector(config_, bad, sequence_schedule_), DetectionError);
}

TEST_F(DetectorTest, ThrowsOnMissingDetectorGeometry) {
    aurora::forward::ForwardConfig bad = forward_config_;
    bad.clear_detector();
    EXPECT_THROW(Detector(config_, bad, sequence_schedule_), DetectionError);
}

TEST_F(DetectorTest, ThrowsOnMissingFieldOfView) {
    aurora::forward::ForwardConfig bad = forward_config_;
    bad.mutable_optics()->set_system_fov_x_rad(0.0);
    bad.mutable_optics()->clear_pupil();
    EXPECT_THROW(Detector(config_, bad, sequence_schedule_), DetectionError);
}

TEST_F(DetectorTest, FallsBackToPupilFieldOfView) {
    aurora::forward::ForwardConfig cfg = forward_config_;
    cfg.mutable_optics()->set_system_fov_x_rad(0.0);
    cfg.mutable_optics()->set_system_fov_y_rad(0.0);
    cfg.mutable_optics()->mutable_pupil()->set_fov_half_angle_rad(0.1);
    EXPECT_NO_THROW({
        Detector detector(config_, cfg, sequence_schedule_);
    });
}

TEST_F(DetectorTest, ThrowsOnMissingChirpSchedule) {
    aurora::acquisition::SequenceSchedule bad;
    EXPECT_THROW(Detector(config_, forward_config_, bad), DetectionError);
}

TEST_F(DetectorTest, EmptyTensorProducesEmptyFrame) {
    Detector detector(config_, forward_config_, sequence_schedule_);
    aurora::shared::MeasurementTensor empty;
    auto frame = detector.detect(empty);

    EXPECT_EQ(frame.detections_size(), 0);
    EXPECT_EQ(frame.num_active_range_bins(), 0u);
}

TEST_F(DetectorTest, FrameIdIncrements) {
    Detector detector(config_, forward_config_, sequence_schedule_);
    aurora::shared::MeasurementTensor empty;
    auto f0 = detector.detect(empty);
    auto f1 = detector.detect(empty);
    EXPECT_EQ(f0.frame_id(), 0u);
    EXPECT_EQ(f1.frame_id(), 1u);
    EXPECT_EQ(detector.next_frame_id(), 2u);
}

TEST_F(DetectorTest, DetectsSinglePointTarget) {
    SyntheticTarget target{20, 0.3 * azimuth_period(), 1.0};
    auto tensor = create_tensor({target});

    Detector detector(config_, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);

    ASSERT_EQ(frame.detections_size(), 1);
    const auto& det = frame.detections(0);
    EXPECT_NEAR(det.range_m(), 20 * kRangeStep, 1e-9);
    EXPECT_NEAR(det.azimuth_rad(), target.theta_x, 1e-6);
    EXPECT_LT(det.phase_residual_rad(), 1e-6);
    EXPECT_TRUE(std::isfinite(det.azimuth_uncertainty_rad()));
    EXPECT_EQ(det.detector_indices_size(), static_cast<int>(kNumDetectors));
    EXPECT_GT(det.snr_db(), 0.0);
    EXPECT_NEAR(det.position_m().z(), det.range_m() * std::cos(det.azimuth_rad()), 1e-9);
}

TEST_F(DetectorTest, LinearArrayReportsElevationUnobservable) {
    SyntheticTarget target{20, 0.1 * azimuth_period(), 1.0};
    auto tensor = create_tensor({target});

    Detector detector(config_, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);

    ASSERT_EQ(frame.detections_size(), 1);
    EXPECT_DOUBLE_EQ(frame.detections(0).elevation_rad(), 0.0);
    EXPECT_TRUE(std::isinf(frame.detections(0).elevation_uncertainty_rad()));
}

TEST_F(DetectorTest, DetectsTargetsAtDifferentRanges) {
    std::vector<SyntheticTarget> targets = {
        {10, 0.1 * azimuth_period(), 1.0},
        {40, -0.2 * azimuth_period(), 0.5},
    };
    auto tensor = create_tensor(targets);

    Detector detector(config_, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);

    ASSERT_EQ(frame.detections_size(), 2);
    EXPECT_EQ(frame.num_active_range_bins(), 2u);
    for (const auto& det : frame.detections()) {
        int bin = static_cast<int>(std::lround(det.range_m() / kRangeStep));
        bool matched = false;
        for (const auto& target : targets) {
            if (static_cast<int>(target.range_bin) == bin) {
                EXPECT_NEAR(det.azimuth_rad(), target.theta_x, 1e-6);
                matched = true;
            }
        }
        EXPECT_TRUE(matched) << "Unexpected detection at bin " << bin;
    }
}

TEST_F(DetectorTest, AdjacentRangeBinsCollapseToOnePeak) {
    // Energy spread over three adjacent bins (as a windowed range FFT does)
    // must produce a single detection at the strongest bin.
    double theta = 0.1 * azimuth_period();
    std::vector<SyntheticTarget> targets = {
        {19, theta, 0.5},
        {20, theta, 1.0},
        {21, theta, 0.5},
    };
    auto tensor = create_tensor(targets);

    Detector detector(config_, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);

    ASSERT_EQ(frame.detections_size(), 1);
    EXPECT_NEAR(frame.detections(0).range_m(), 20 * kRangeStep, 1e-9);
    EXPECT_EQ(frame.num_active_range_bins(), 1u);
}

TEST_F(DetectorTest, DopplerFromUpDownChirps) {
    DetectionConfig config = config_;
    config.set_enable_doppler(true);
    config.set_min_velocity_mps(0.01);

    // Phase-based Doppler is unambiguous for |v| < λ/(4T) ≈ 0.039 m/s here.
    const double velocity = 0.02;  // m/s
    SyntheticTarget target{25, 0.05 * azimuth_period(), 1.0, velocity};
    auto tensor = create_tensor({target}, /*with_doppler=*/true);

    Detector detector(config, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);

    ASSERT_EQ(frame.detections_size(), 1);
    const auto& det = frame.detections(0);
    ASSERT_TRUE(det.has_doppler());
    EXPECT_NEAR(det.doppler().velocity_mps(), velocity, 1e-6);
}

TEST_F(DetectorTest, NoDopplerWithoutUpDownChirps) {
    DetectionConfig config = config_;
    config.set_enable_doppler(true);
    config.set_min_velocity_mps(0.01);

    SyntheticTarget target{25, 0.05 * azimuth_period(), 1.0, 0.02};
    auto tensor = create_tensor({target}, /*with_doppler=*/false);

    Detector detector(config, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);

    ASSERT_EQ(frame.detections_size(), 1);
    EXPECT_FALSE(frame.detections(0).has_doppler());
}

TEST_F(DetectorTest, BelowThresholdIsNotDetected) {
    SyntheticTarget target{20, 0.0, 1e-4};  // energy ~ 8e-8 < 1e-3
    auto tensor = create_tensor({target});

    Detector detector(config_, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);
    EXPECT_EQ(frame.detections_size(), 0);
}

TEST_F(DetectorTest, MissingKVectorIsSkipped) {
    SyntheticTarget target{20, 0.0, 1.0};
    auto tensor = create_tensor({target});
    for (auto& det : *tensor.mutable_data(0)->mutable_detectors()) {
        det.clear_k_vector();
    }

    Detector detector(config_, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);
    EXPECT_EQ(frame.num_active_range_bins(), 1u);
    EXPECT_EQ(frame.detections_size(), 0);
}

TEST_F(DetectorTest, CfarThresholdingDetectsTarget) {
    DetectionConfig config = config_;
    auto* cfar = config.mutable_cfar();
    cfar->set_mode(CFAR_MODE_CA);
    cfar->set_guard_cells(2);
    cfar->set_reference_cells(8);
    cfar->set_threshold_factor(5.0);
    cfar->set_min_threshold(1e-6);
    cfar->set_skip_bins(2);

    SyntheticTarget target{30, 0.1 * azimuth_period(), 1.0};
    auto tensor = create_tensor({target});

    Detector detector(config, forward_config_, sequence_schedule_);
    auto frame = detector.detect(tensor);

    ASSERT_EQ(frame.detections_size(), 1);
    EXPECT_NEAR(frame.detections(0).azimuth_rad(), target.theta_x, 1e-6);
}

TEST_F(DetectorTest, UpdateConfigRebuildsPipeline) {
    Detector detector(config_, forward_config_, sequence_schedule_);

    DetectionConfig strict = config_;
    strict.set_magnitude_threshold_db(100.0);  // Nothing passes
    detector.update_config(strict);

    SyntheticTarget target{20, 0.0, 1.0};
    auto tensor = create_tensor({target});
    EXPECT_EQ(detector.detect(tensor).detections_size(), 0);
}
