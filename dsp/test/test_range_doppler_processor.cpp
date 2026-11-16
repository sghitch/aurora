#include "dsp/range_doppler_processor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <memory>
#include <optional>

#include "forward/forward_model.h"
#include "test_forward_model_builder.h"

#include "detector_geometry.pb.h"
#include "optics_config.pb.h"

namespace aurora::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kSpeedOfLight = 299792458.0;

/// Create test detector geometry with 8 pixels.
aurora::forward::DetectorGeometry create_test_detector_geometry() {
    aurora::forward::DetectorGeometry detector;
    auto* array = detector.add_arrays();
    array->set_array_id("test_array");
    array->set_nominal_pitch_m(0.5e-3);
    array->set_bandwidth_hz(150e6);
    array->add_wavelengths_m(1550e-9);
    array->set_sensitivity_type(aurora::forward::DETECTOR_SENSITIVITY_SINC);

    for (uint32_t i = 0; i < 8; ++i) {
        auto* pixel = array->add_pixels();
        pixel->set_index(i);
        pixel->set_x_m((static_cast<double>(i) - 3.5) * 0.5e-3);
        pixel->set_y_m(0.0);
        pixel->set_width_m(0.4e-3);
        pixel->set_height_m(0.4e-3);
    }
    return detector;
}

/// Create test optics configuration.
aurora::forward::OpticsConfig create_test_optics_config() {
    aurora::forward::OpticsConfig optics;
    optics.set_fourier_lens_focal_m(0.05);
    optics.set_fourier_lens_diameter_m(0.025);
    optics.add_wavelengths_m(1550e-9);
    return optics;
}

/// Create a test DSP config.
DspConfig create_test_dsp_config() {
    DspConfig config;
    config.set_range_fft_size(2048);
    config.mutable_range_window()->set_type(WindowConfig::WINDOW_HANN);
    config.set_enable_dc_removal(true);

    auto* adc = config.mutable_adc_config();
    adc->set_sample_rate_hz(125e6);
    adc->set_bits(14);
    adc->set_input_range_v(2.0);

    auto* iq = config.mutable_iq_demod_config();
    iq->set_lpf_cutoff_hz(10e6);
    iq->set_lpf_order(4);

    return config;
}

/// Generate simulated FMCW beat signal for a target at given range.
std::vector<int32_t> generate_fmcw_signal(
    double target_range_m,
    double sample_rate_hz,
    double chirp_bandwidth_hz,
    double chirp_duration_s,
    size_t num_samples,
    double amplitude = 0.5) {

    double beat_freq = 2.0 * chirp_bandwidth_hz * target_range_m /
                       (kSpeedOfLight * chirp_duration_s);

    std::vector<int32_t> samples(num_samples);
    const double adc_scale = (1 << 13);

    for (size_t i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / sample_rate_hz;
        double value = amplitude * std::cos(kTwoPi * beat_freq * t);
        samples[i] = static_cast<int32_t>(value * adc_scale);
    }

    return samples;
}

/// Create a test burst with specified parameters.
aurora::simulation::Burst create_test_burst(
    double target_range_m,
    double sample_rate_hz = 125e6,
    double chirp_bandwidth_hz = 200e6,
    double chirp_duration_s = 12.5e-6) {

    size_t num_samples = static_cast<size_t>(sample_rate_hz * chirp_duration_s);

    aurora::simulation::Burst burst;
    burst.set_burst_id(1);
    burst.set_timestamp_s(0.0);

    auto* plan = burst.mutable_plan();
    plan->set_plan_id(1);
    plan->set_start_timestamp_s(0.0);

    auto* chirp_plan = plan->add_chirps();
    chirp_plan->set_chirp_index(0);
    chirp_plan->set_chirp_start_s(0.0);
    chirp_plan->set_chirp_duration_s(chirp_duration_s);
    chirp_plan->set_chirp_start_hz(0.0);
    chirp_plan->set_chirp_bandwidth_hz(chirp_bandwidth_hz);
    chirp_plan->set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);
    chirp_plan->add_wavelength_m(1550e-9);

    for (uint32_t det_idx = 0; det_idx < 8; ++det_idx) {
        auto* det_burst = burst.add_detectors();
        det_burst->set_detector_index(det_idx);

        auto* chirp_samples = det_burst->add_chirps();
        chirp_samples->set_chirp_index(0);
        chirp_samples->add_wavelengths_m(1550e-9);

        auto samples = generate_fmcw_signal(
            target_range_m, sample_rate_hz, chirp_bandwidth_hz,
            chirp_duration_s, num_samples);

        for (int32_t s : samples) {
            chirp_samples->add_samples(s);
        }
    }

    return burst;
}

/// Test fixture that sets up ForwardModel once per test suite.
class RangeDopplerProcessorTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        detector_ = create_test_detector_geometry();
        optics_ = create_test_optics_config();
        dsp_config_ = create_test_dsp_config();

        // Create ForwardModel from geometry
        forward_model_.emplace(detector_, optics_);
    }

    static void TearDownTestSuite() {
        forward_model_.reset();
    }

    const DspConfig& dsp_config() const { return dsp_config_; }
    const aurora::forward::ForwardModel& forward_model() const { return *forward_model_; }

    static aurora::forward::DetectorGeometry detector_;
    static aurora::forward::OpticsConfig optics_;
    static DspConfig dsp_config_;
    static std::optional<aurora::forward::ForwardModel> forward_model_;
};

// Static member definitions
aurora::forward::DetectorGeometry RangeDopplerProcessorTest::detector_;
aurora::forward::OpticsConfig RangeDopplerProcessorTest::optics_;
DspConfig RangeDopplerProcessorTest::dsp_config_;
std::optional<aurora::forward::ForwardModel> RangeDopplerProcessorTest::forward_model_;

TEST_F(RangeDopplerProcessorTest, ConstructorValidatesConfig) {
    // Zero FFT size should throw
    DspConfig bad_config = dsp_config();
    bad_config.set_range_fft_size(0);
    EXPECT_THROW(
        RangeDopplerProcessor(bad_config, forward_model()),
        std::invalid_argument);

    // Non-power-of-2 FFT size should throw
    bad_config = dsp_config();
    bad_config.set_range_fft_size(1000);
    EXPECT_THROW(
        RangeDopplerProcessor(bad_config, forward_model()),
        std::invalid_argument);

    // Missing ADC config should throw
    bad_config = dsp_config();
    bad_config.clear_adc_config();
    EXPECT_THROW(
        RangeDopplerProcessor(bad_config, forward_model()),
        std::invalid_argument);

    // Invalid ADC sample rate should throw
    bad_config = dsp_config();
    bad_config.mutable_adc_config()->set_sample_rate_hz(0);
    EXPECT_THROW(
        RangeDopplerProcessor(bad_config, forward_model()),
        std::invalid_argument);

    // Invalid ADC bits should throw
    bad_config = dsp_config();
    bad_config.mutable_adc_config()->set_bits(0);
    EXPECT_THROW(
        RangeDopplerProcessor(bad_config, forward_model()),
        std::invalid_argument);

    // Missing IQ demod config should throw
    bad_config = dsp_config();
    bad_config.clear_iq_demod_config();
    EXPECT_THROW(
        RangeDopplerProcessor(bad_config, forward_model()),
        std::invalid_argument);

    // Invalid LPF cutoff should throw
    bad_config = dsp_config();
    bad_config.mutable_iq_demod_config()->set_lpf_cutoff_hz(0);
    EXPECT_THROW(
        RangeDopplerProcessor(bad_config, forward_model()),
        std::invalid_argument);
}

TEST_F(RangeDopplerProcessorTest, EmptyBurstReturnsEmptyCube) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    aurora::simulation::Burst empty_burst;
    empty_burst.set_burst_id(42);
    empty_burst.set_timestamp_s(1.5);

    RangeCube cube = processor.process_range(empty_burst);

    EXPECT_EQ(cube.burst_id(), 42u);
    EXPECT_DOUBLE_EQ(cube.timestamp_s(), 1.5);
    EXPECT_EQ(cube.wavelengths_size(), 0);
}

TEST_F(RangeDopplerProcessorTest, MissingWavelengthThrows) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    aurora::simulation::Burst burst;
    burst.set_burst_id(1);
    auto* plan = burst.mutable_plan();
    plan->set_plan_id(1);

    auto* det_burst = burst.add_detectors();
    det_burst->set_detector_index(0);
    auto* chirp = det_burst->add_chirps();
    chirp->set_chirp_index(0);
    chirp->add_samples(1000);

    EXPECT_THROW(processor.process_range(burst), std::invalid_argument);
}

TEST_F(RangeDopplerProcessorTest, RangeAccuracy) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    double target_range_m = 50.0;
    auto burst = create_test_burst(target_range_m);

    RangeCube cube = processor.process_range(burst);

    ASSERT_GT(cube.wavelengths_size(), 0);
    ASSERT_GT(cube.wavelengths_size(), 0);
    ASSERT_GT(cube.wavelengths(0).detectors_size(), 0);

    const auto& detector = cube.wavelengths(0).detectors(0);
    ASSERT_GT(detector.bins_size(), 0);

    int peak_bin = 0;
    double peak_mag = 0.0;
    for (int i = 0; i < detector.bins_size(); ++i) {
        double mag = std::sqrt(
            detector.bins(i).real() * detector.bins(i).real() +
            detector.bins(i).imag() * detector.bins(i).imag());
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_bin = i;
        }
    }

    double sample_rate_hz = 125e6;
    double chirp_bandwidth_hz = 200e6;
    double chirp_duration_s = 12.5e-6;
    double fft_size = static_cast<double>(dsp_config().range_fft_size());
    double beat_freq = 2.0 * chirp_bandwidth_hz * target_range_m /
                       (kSpeedOfLight * chirp_duration_s);
    double freq_resolution = sample_rate_hz / fft_size;
    double expected_bin = beat_freq / freq_resolution;

    EXPECT_NEAR(static_cast<double>(peak_bin), expected_bin, 2.0)
        << "Peak at bin " << peak_bin << ", expected ~" << expected_bin;
}

TEST_F(RangeDopplerProcessorTest, RangeAxisConfiguration) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    double target_range_m = 30.0;
    auto burst = create_test_burst(target_range_m);

    RangeCube cube = processor.process_range(burst);

    EXPECT_EQ(cube.range_axis().unit(), "m");
    EXPECT_DOUBLE_EQ(cube.range_axis().start(), 0.0);
    EXPECT_EQ(cube.range_axis().count(), dsp_config().range_fft_size() / 2);

    double chirp_duration_s = 12.5e-6;
    double sample_rate_hz = 125e6;
    double chirp_bandwidth_hz = 200e6;
    uint32_t fft_size = dsp_config().range_fft_size();
    double expected_range_per_bin = kSpeedOfLight * chirp_duration_s * sample_rate_hz /
                                    (2.0 * chirp_bandwidth_hz * fft_size);
    EXPECT_NEAR(cube.range_axis().step(), expected_range_per_bin, 0.01)
        << "Range per bin should be ~" << expected_range_per_bin << " m";
}

TEST_F(RangeDopplerProcessorTest, BurstMetadataPreserved) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_test_burst(25.0);
    burst.set_burst_id(12345);
    burst.set_timestamp_s(42.5);

    RangeCube cube = processor.process_range(burst);

    EXPECT_EQ(cube.burst_id(), 12345u);
    EXPECT_DOUBLE_EQ(cube.timestamp_s(), 42.5);
}

TEST_F(RangeDopplerProcessorTest, PlanPreserved) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_test_burst(40.0);
    RangeCube cube = processor.process_range(burst);

    ASSERT_EQ(cube.wavelengths_size(), 1);
    ASSERT_TRUE(cube.has_plan());
    EXPECT_EQ(cube.plan().plan_id(), burst.plan().plan_id());
    ASSERT_EQ(cube.plan().chirps_size(), 1);
    EXPECT_DOUBLE_EQ(cube.plan().chirps(0).chirp_bandwidth_hz(), 200e6);
}

TEST_F(RangeDopplerProcessorTest, AllDetectorsProcessed) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_test_burst(35.0);
    RangeCube cube = processor.process_range(burst);

    ASSERT_EQ(cube.wavelengths_size(), 1);
    ASSERT_GT(cube.wavelengths_size(), 0);

    const auto& wavelength = cube.wavelengths(0);
    EXPECT_DOUBLE_EQ(wavelength.wavelength_m(), 1550e-9);
    EXPECT_EQ(wavelength.detectors_size(), 8);

    for (int i = 0; i < wavelength.detectors_size(); ++i) {
        EXPECT_EQ(wavelength.detectors(i).detector_index(),
                  static_cast<uint32_t>(i));
        EXPECT_EQ(wavelength.detectors(i).bins_size(),
                  static_cast<int>(dsp_config().range_fft_size() / 2));
    }
}

TEST_F(RangeDopplerProcessorTest, MeasurementTensorHasKVectors) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_test_burst(20.0);
    RangeCube cube = processor.process_range(burst);
    auto tensor = processor.to_measurement_tensor(cube);

    ASSERT_EQ(tensor.data_size(), 1);
    ASSERT_GT(tensor.data(0).detectors_size(), 0);
    ASSERT_GT(tensor.data(0).detectors_size(), 0);

    for (const auto& detector : tensor.data(0).detectors()) {
        EXPECT_TRUE(detector.has_k_vector());
        EXPECT_TRUE(std::isfinite(detector.k_vector().kx()));
        EXPECT_TRUE(std::isfinite(detector.k_vector().ky()));
    }
}

TEST_F(RangeDopplerProcessorTest, MeasurementTensorPreservesMetadata) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_test_burst(15.0);
    burst.set_burst_id(999);
    burst.set_timestamp_s(123.456);

    RangeCube cube = processor.process_range(burst);
    auto tensor = processor.to_measurement_tensor(cube);

    EXPECT_EQ(tensor.burst_id(), 999u);
    EXPECT_DOUBLE_EQ(tensor.timestamp_s(), 123.456);
    EXPECT_EQ(tensor.range_axis().unit(), cube.range_axis().unit());
    EXPECT_DOUBLE_EQ(tensor.range_axis().start(), cube.range_axis().start());
    EXPECT_DOUBLE_EQ(tensor.range_axis().step(), cube.range_axis().step());
    EXPECT_EQ(tensor.range_axis().count(), cube.range_axis().count());
}

TEST_F(RangeDopplerProcessorTest, RangeDataPreservedInTensor) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_test_burst(60.0);
    RangeCube cube = processor.process_range(burst);
    auto tensor = processor.to_measurement_tensor(cube);

    const auto& cube_det = cube.wavelengths(0).detectors(0);
    const auto& tensor_det = tensor.data(0).detectors(0);

    ASSERT_EQ(cube_det.bins_size(), tensor_det.range_data_size());

    for (int i = 0; i < std::min(10, cube_det.bins_size()); ++i) {
        EXPECT_DOUBLE_EQ(cube_det.bins(i).real(), tensor_det.range_data(i).real());
        EXPECT_DOUBLE_EQ(cube_det.bins(i).imag(), tensor_det.range_data(i).imag());
    }
}

TEST_F(RangeDopplerProcessorTest, LatencyStatsPopulated) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_test_burst(45.0);
    processor.process_range(burst);

    auto stats = processor.get_latency_stats();

    EXPECT_GT(stats.total_us, 0.0);
    EXPECT_GE(stats.preprocess_us, 0.0);
    EXPECT_GE(stats.iq_demod_us, 0.0);
    EXPECT_GE(stats.window_us, 0.0);
    EXPECT_GE(stats.fft_us, 0.0);

    double sum = stats.preprocess_us + stats.iq_demod_us +
                 stats.window_us + stats.fft_us;
    EXPECT_GE(stats.total_us, sum * 0.9);
}

TEST_F(RangeDopplerProcessorTest, DcRemovalConfigurable) {
    DspConfig config_with_dc = dsp_config();
    config_with_dc.set_enable_dc_removal(true);
    RangeDopplerProcessor proc_with_dc(config_with_dc, forward_model());

    DspConfig config_no_dc = dsp_config();
    config_no_dc.set_enable_dc_removal(false);
    RangeDopplerProcessor proc_no_dc(config_no_dc, forward_model());

    auto burst = create_test_burst(55.0);

    auto cube_with_dc = proc_with_dc.process_range(burst);
    auto cube_no_dc = proc_no_dc.process_range(burst);

    ASSERT_GT(cube_with_dc.wavelengths(0).detectors(0).bins_size(), 0);
    ASSERT_GT(cube_no_dc.wavelengths(0).detectors(0).bins_size(), 0);
}

TEST_F(RangeDopplerProcessorTest, WindowTypeApplied) {
    DspConfig hann_config = dsp_config();
    hann_config.mutable_range_window()->set_type(WindowConfig::WINDOW_HANN);
    RangeDopplerProcessor proc_hann(hann_config, forward_model());

    DspConfig rect_config = dsp_config();
    rect_config.mutable_range_window()->set_type(WindowConfig::WINDOW_RECTANGULAR);
    RangeDopplerProcessor proc_rect(rect_config, forward_model());

    auto burst = create_test_burst(70.0);

    auto cube_hann = proc_hann.process_range(burst);
    auto cube_rect = proc_rect.process_range(burst);

    ASSERT_GT(cube_hann.wavelengths(0).detectors(0).bins_size(), 0);
    ASSERT_GT(cube_rect.wavelengths(0).detectors(0).bins_size(), 0);

    double peak_hann = 0.0, peak_rect = 0.0;
    for (const auto& bin : cube_hann.wavelengths(0).detectors(0).bins()) {
        double mag = std::sqrt(bin.real() * bin.real() + bin.imag() * bin.imag());
        peak_hann = std::max(peak_hann, mag);
    }
    for (const auto& bin : cube_rect.wavelengths(0).detectors(0).bins()) {
        double mag = std::sqrt(bin.real() * bin.real() + bin.imag() * bin.imag());
        peak_rect = std::max(peak_rect, mag);
    }

    EXPECT_GT(peak_rect, 0.0);
    EXPECT_GT(peak_hann, 0.0);
}

TEST_F(RangeDopplerProcessorTest, DifferentFftSizes) {
    std::vector<uint32_t> fft_sizes = {512, 1024, 2048, 4096};

    for (uint32_t fft_size : fft_sizes) {
        DspConfig config = dsp_config();
        config.set_range_fft_size(fft_size);

        RangeDopplerProcessor proc(config, forward_model());
        auto burst = create_test_burst(30.0);
        RangeCube cube = proc.process_range(burst);

        EXPECT_EQ(cube.range_axis().count(), fft_size / 2)
            << "FFT size " << fft_size << " should produce " << (fft_size / 2) << " range bins";
    }
}

// ============================================================================
// Up/Down Doppler Processing Tests
// ============================================================================

std::vector<int32_t> generate_fmcw_signal_with_doppler(
    double target_range_m,
    double target_velocity_mps,
    double wavelength_m,
    bool is_up_chirp,
    double sample_rate_hz,
    double chirp_bandwidth_hz,
    double chirp_duration_s,
    size_t num_samples,
    double amplitude = 0.5) {

    double f_beat_range = 2.0 * chirp_bandwidth_hz * target_range_m /
                          (kSpeedOfLight * chirp_duration_s);
    double f_doppler = 2.0 * target_velocity_mps / wavelength_m;
    double f_beat = is_up_chirp ? (f_beat_range + f_doppler)
                                : (f_beat_range - f_doppler);

    std::vector<int32_t> samples(num_samples);
    const double adc_scale = (1 << 13);

    for (size_t i = 0; i < num_samples; ++i) {
        double t = static_cast<double>(i) / sample_rate_hz;
        double value = amplitude * std::cos(kTwoPi * f_beat * t);
        samples[i] = static_cast<int32_t>(value * adc_scale);
    }

    return samples;
}

aurora::simulation::Burst create_up_down_burst(
    double target_range_m,
    double target_velocity_mps = 0.0,
    double sample_rate_hz = 125e6,
    double chirp_bandwidth_hz = 200e6,
    double chirp_duration_s = 12.5e-6,
    double wavelength_m = 1550e-9) {

    size_t num_samples = static_cast<size_t>(sample_rate_hz * chirp_duration_s);

    aurora::simulation::Burst burst;
    burst.set_burst_id(1);
    burst.set_timestamp_s(0.0);

    auto* plan = burst.mutable_plan();
    plan->set_plan_id(1);
    plan->set_start_timestamp_s(0.0);

    auto* up_chirp_plan = plan->add_chirps();
    up_chirp_plan->set_chirp_index(0);
    up_chirp_plan->set_chirp_start_s(0.0);
    up_chirp_plan->set_chirp_duration_s(chirp_duration_s);
    up_chirp_plan->set_chirp_start_hz(0.0);
    up_chirp_plan->set_chirp_bandwidth_hz(chirp_bandwidth_hz);
    up_chirp_plan->set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);
    up_chirp_plan->add_wavelength_m(wavelength_m);

    auto* down_chirp_plan = plan->add_chirps();
    down_chirp_plan->set_chirp_index(1);
    down_chirp_plan->set_chirp_start_s(chirp_duration_s);
    down_chirp_plan->set_chirp_duration_s(chirp_duration_s);
    down_chirp_plan->set_chirp_start_hz(chirp_bandwidth_hz);
    down_chirp_plan->set_chirp_bandwidth_hz(chirp_bandwidth_hz);
    down_chirp_plan->set_direction(aurora::acquisition::ChirpPlan::DIRECTION_DOWN);
    down_chirp_plan->add_wavelength_m(wavelength_m);

    for (uint32_t det_idx = 0; det_idx < 8; ++det_idx) {
        auto* det_burst = burst.add_detectors();
        det_burst->set_detector_index(det_idx);

        auto* up_chirp_samples = det_burst->add_chirps();
        up_chirp_samples->set_chirp_index(0);
        up_chirp_samples->add_wavelengths_m(wavelength_m);
        auto up_samples = generate_fmcw_signal_with_doppler(
            target_range_m, target_velocity_mps, wavelength_m, true,
            sample_rate_hz, chirp_bandwidth_hz, chirp_duration_s, num_samples);
        for (int32_t s : up_samples) {
            up_chirp_samples->add_samples(s);
        }

        auto* down_chirp_samples = det_burst->add_chirps();
        down_chirp_samples->set_chirp_index(1);
        down_chirp_samples->add_wavelengths_m(wavelength_m);
        auto down_samples = generate_fmcw_signal_with_doppler(
            target_range_m, target_velocity_mps, wavelength_m, false,
            sample_rate_hz, chirp_bandwidth_hz, chirp_duration_s, num_samples);
        for (int32_t s : down_samples) {
            down_chirp_samples->add_samples(s);
        }
    }

    return burst;
}

TEST_F(RangeDopplerProcessorTest, VelocityResolutionComputation) {
    double wavelength_m = 1550e-9;
    double chirp_duration_s = 12.5e-6;

    double v_res = RangeDopplerProcessor::compute_velocity_resolution(
        wavelength_m, chirp_duration_s);

    double expected = wavelength_m / (4.0 * chirp_duration_s);
    EXPECT_NEAR(v_res, expected, 1e-6);
    EXPECT_NEAR(v_res, 0.031, 0.001);
}

TEST_F(RangeDopplerProcessorTest, ProcessDopplerWithValidBurst) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    double target_range_m = 50.0;
    auto burst = create_up_down_burst(target_range_m);

    RangeCube cube = processor.process_doppler(burst);

    ASSERT_GT(cube.wavelengths_size(), 0);
    ASSERT_GT(cube.wavelengths_size(), 0);
    ASSERT_GT(cube.wavelengths(0).detectors_size(), 0);
}

TEST_F(RangeDopplerProcessorTest, ProcessDopplerThrowsWithoutUpChirps) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    aurora::simulation::Burst burst;
    burst.set_burst_id(1);
    auto* plan = burst.mutable_plan();
    plan->set_plan_id(1);

    auto* chirp_plan = plan->add_chirps();
    chirp_plan->set_chirp_index(0);
    chirp_plan->set_chirp_duration_s(12.5e-6);
    chirp_plan->set_chirp_bandwidth_hz(200e6);
    chirp_plan->set_direction(aurora::acquisition::ChirpPlan::DIRECTION_DOWN);
    chirp_plan->add_wavelength_m(1550e-9);

    auto* det_burst = burst.add_detectors();
    det_burst->set_detector_index(0);
    auto* chirp = det_burst->add_chirps();
    chirp->set_chirp_index(0);
    chirp->add_wavelengths_m(1550e-9);
    chirp->add_samples(1000);

    EXPECT_THROW(processor.process_doppler(burst), std::invalid_argument);
}

TEST_F(RangeDopplerProcessorTest, ProcessDopplerThrowsWithoutDownChirps) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    aurora::simulation::Burst burst;
    burst.set_burst_id(1);
    auto* plan = burst.mutable_plan();
    plan->set_plan_id(1);

    auto* chirp_plan = plan->add_chirps();
    chirp_plan->set_chirp_index(0);
    chirp_plan->set_chirp_duration_s(12.5e-6);
    chirp_plan->set_chirp_bandwidth_hz(200e6);
    chirp_plan->set_direction(aurora::acquisition::ChirpPlan::DIRECTION_UP);
    chirp_plan->add_wavelength_m(1550e-9);

    auto* det_burst = burst.add_detectors();
    det_burst->set_detector_index(0);
    auto* chirp = det_burst->add_chirps();
    chirp->set_chirp_index(0);
    chirp->add_wavelengths_m(1550e-9);
    chirp->add_samples(1000);

    EXPECT_THROW(processor.process_doppler(burst), std::invalid_argument);
}

TEST_F(RangeDopplerProcessorTest, ProcessDopplerEmptyBurst) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    aurora::simulation::Burst empty_burst;
    empty_burst.set_burst_id(42);
    empty_burst.set_timestamp_s(1.5);

    RangeCube cube = processor.process_doppler(empty_burst);

    EXPECT_EQ(cube.burst_id(), 42u);
    EXPECT_DOUBLE_EQ(cube.timestamp_s(), 1.5);
    EXPECT_EQ(cube.wavelengths_size(), 0);
}

TEST_F(RangeDopplerProcessorTest, DopplerCorrectionStationaryTarget) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    double target_range_m = 50.0;
    double target_velocity_mps = 0.0;

    auto burst = create_up_down_burst(target_range_m, target_velocity_mps);
    RangeCube cube = processor.process_doppler(burst);

    const auto& detector = cube.wavelengths(0).detectors(0);

    int peak_bin = 0;
    double peak_mag = 0.0;
    for (int i = 0; i < detector.bins_size(); ++i) {
        double mag = std::sqrt(
            detector.bins(i).real() * detector.bins(i).real() +
            detector.bins(i).imag() * detector.bins(i).imag());
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_bin = i;
        }
    }

    double beat_freq = 2.0 * 200e6 * target_range_m / (kSpeedOfLight * 12.5e-6);
    double freq_resolution = 125e6 / static_cast<double>(dsp_config().range_fft_size());
    double expected_bin = beat_freq / freq_resolution;

    EXPECT_NEAR(static_cast<double>(peak_bin), expected_bin, 2.0);
}

TEST_F(RangeDopplerProcessorTest, DopplerCorrectionRemovesVelocityBias) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    double target_range_m = 50.0;
    double target_velocity_mps = 0.05;

    auto burst_updown = create_up_down_burst(target_range_m, target_velocity_mps);
    RangeCube cube = processor.process_doppler(burst_updown);

    const auto& det_corrected = cube.wavelengths(0).detectors(0);
    int peak_corrected = 0;
    double peak_mag_corrected = 0.0;
    for (int i = 0; i < det_corrected.bins_size(); ++i) {
        double mag = std::sqrt(
            det_corrected.bins(i).real() * det_corrected.bins(i).real() +
            det_corrected.bins(i).imag() * det_corrected.bins(i).imag());
        if (mag > peak_mag_corrected) {
            peak_mag_corrected = mag;
            peak_corrected = i;
        }
    }

    double beat_freq_true = 2.0 * 200e6 * target_range_m / (kSpeedOfLight * 12.5e-6);
    double freq_resolution = 125e6 / static_cast<double>(dsp_config().range_fft_size());
    double expected_bin = beat_freq_true / freq_resolution;

    EXPECT_NEAR(static_cast<double>(peak_corrected), expected_bin, 2.0)
        << "Doppler-corrected peak should be at true range bin";
}

TEST_F(RangeDopplerProcessorTest, AllDetectorsProcessedInDopplerMode) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_up_down_burst(35.0, 5.0);
    RangeCube cube = processor.process_doppler(burst);

    ASSERT_EQ(cube.wavelengths_size(), 1);
    ASSERT_GT(cube.wavelengths_size(), 0);

    const auto& wavelength = cube.wavelengths(0);
    EXPECT_EQ(wavelength.detectors_size(), 8);

    for (int i = 0; i < wavelength.detectors_size(); ++i) {
        EXPECT_EQ(wavelength.detectors(i).detector_index(),
                  static_cast<uint32_t>(i));
        EXPECT_GT(wavelength.detectors(i).bins_size(), 0);
    }
}

TEST_F(RangeDopplerProcessorTest, MetadataPreservedInDopplerProcessing) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_up_down_burst(40.0, 2.0);
    burst.set_burst_id(12345);
    burst.set_timestamp_s(42.5);

    RangeCube cube = processor.process_doppler(burst);

    EXPECT_EQ(cube.burst_id(), 12345u);
    EXPECT_DOUBLE_EQ(cube.timestamp_s(), 42.5);
}

TEST_F(RangeDopplerProcessorTest, RangeAxisCorrectInDopplerProcessing) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_up_down_burst(30.0);
    RangeCube cube = processor.process_doppler(burst);

    EXPECT_EQ(cube.range_axis().unit(), "m");
    EXPECT_DOUBLE_EQ(cube.range_axis().start(), 0.0);
    EXPECT_EQ(cube.range_axis().count(),
              dsp_config().range_fft_size() / 2);

    double chirp_duration_s = 12.5e-6;
    double sample_rate_hz = 125e6;
    double chirp_bandwidth_hz = 200e6;
    uint32_t fft_size = dsp_config().range_fft_size();
    double expected_range_per_bin = kSpeedOfLight * chirp_duration_s * sample_rate_hz /
                                    (2.0 * chirp_bandwidth_hz * fft_size);
    EXPECT_NEAR(cube.range_axis().step(), expected_range_per_bin, 0.01);
}

// ============================================================================
// Up/Down Chirp Profiles in RangeCube and MeasurementTensor
// ============================================================================

/// Index of the strongest range bin in a detector's profile.
int find_peak_bin(const DetectorRangeData& detector) {
    int peak_bin = 0;
    double peak_mag = 0.0;
    for (int i = 0; i < detector.bins_size(); ++i) {
        double mag = std::sqrt(
            detector.bins(i).real() * detector.bins(i).real() +
            detector.bins(i).imag() * detector.bins(i).imag());
        if (mag > peak_mag) {
            peak_mag = mag;
            peak_bin = i;
        }
    }
    return peak_bin;
}

TEST_F(RangeDopplerProcessorTest, ProcessDopplerFillsUpDownChirps) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    const double target_range_m = 50.0;
    const double target_velocity_mps = 1.0;
    const double wavelength_m = 1550e-9;
    const double chirp_bandwidth_hz = 200e6;
    const double chirp_duration_s = 12.5e-6;
    const double sample_rate_hz = 125e6;

    auto burst = create_up_down_burst(target_range_m, target_velocity_mps);
    RangeCube cube = processor.process_doppler(burst);

    // Per-direction profiles are populated alongside the averaged data
    ASSERT_EQ(cube.wavelengths_size(), 1);
    ASSERT_EQ(cube.up_chirps_size(), 1);
    ASSERT_EQ(cube.down_chirps_size(), 1);
    EXPECT_DOUBLE_EQ(cube.up_chirps(0).wavelength_m(), wavelength_m);
    EXPECT_DOUBLE_EQ(cube.down_chirps(0).wavelength_m(), wavelength_m);
    ASSERT_EQ(cube.up_chirps(0).detectors_size(), 8);
    ASSERT_EQ(cube.down_chirps(0).detectors_size(), 8);

    const int num_bins = static_cast<int>(dsp_config().range_fft_size() / 2);
    for (int d = 0; d < 8; ++d) {
        EXPECT_EQ(cube.up_chirps(0).detectors(d).detector_index(), static_cast<uint32_t>(d));
        EXPECT_EQ(cube.down_chirps(0).detectors(d).detector_index(), static_cast<uint32_t>(d));
        EXPECT_EQ(cube.up_chirps(0).detectors(d).bins_size(), num_bins);
        EXPECT_EQ(cube.down_chirps(0).detectors(d).bins_size(), num_bins);
    }

    // Up-chirp sees f_beat + f_doppler, down-chirp sees f_beat - f_doppler
    const double freq_resolution = sample_rate_hz / static_cast<double>(dsp_config().range_fft_size());
    const double f_beat = 2.0 * chirp_bandwidth_hz * target_range_m /
                          (kSpeedOfLight * chirp_duration_s);
    const double f_doppler = 2.0 * target_velocity_mps / wavelength_m;
    const double expected_up_bin = (f_beat + f_doppler) / freq_resolution;
    const double expected_down_bin = (f_beat - f_doppler) / freq_resolution;

    const int up_peak = find_peak_bin(cube.up_chirps(0).detectors(0));
    const int down_peak = find_peak_bin(cube.down_chirps(0).detectors(0));

    EXPECT_NEAR(static_cast<double>(up_peak), expected_up_bin, 2.0);
    EXPECT_NEAR(static_cast<double>(down_peak), expected_down_bin, 2.0);
    EXPECT_GT(up_peak, down_peak) << "Approaching target: up-chirp beat must exceed down-chirp beat";

    // Velocity recovered from the up/down peak separation: v = (f_up - f_down)/2 * lambda/2
    const double f_dop_est = (up_peak - down_peak) * freq_resolution / 2.0;
    const double v_est = f_dop_est * wavelength_m / 2.0;
    const double v_tolerance = 2.0 * freq_resolution * wavelength_m / 2.0;  // +/- 2 bins
    EXPECT_NEAR(v_est, target_velocity_mps, v_tolerance);

    // The averaged data is the bin-wise mean of the up and down profiles
    const auto& avg_det = cube.wavelengths(0).detectors(0);
    const auto& up_det = cube.up_chirps(0).detectors(0);
    const auto& down_det = cube.down_chirps(0).detectors(0);
    for (int b = 0; b < num_bins; ++b) {
        EXPECT_NEAR(avg_det.bins(b).real(),
                    (up_det.bins(b).real() + down_det.bins(b).real()) / 2.0, 1e-9);
        EXPECT_NEAR(avg_det.bins(b).imag(),
                    (up_det.bins(b).imag() + down_det.bins(b).imag()) / 2.0, 1e-9);
    }
}

TEST_F(RangeDopplerProcessorTest, ProcessRangeLeavesUpDownChirpsEmpty) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_up_down_burst(50.0, 1.0);
    RangeCube cube = processor.process_range(burst);

    ASSERT_EQ(cube.wavelengths_size(), 1);
    EXPECT_EQ(cube.up_chirps_size(), 0);
    EXPECT_EQ(cube.down_chirps_size(), 0);
}

TEST_F(RangeDopplerProcessorTest, MeasurementTensorCarriesUpDownChirpData) {
    RangeDopplerProcessor processor(dsp_config(), forward_model());

    auto burst = create_up_down_burst(50.0, 1.0);
    RangeCube cube = processor.process_doppler(burst);
    auto tensor = processor.to_measurement_tensor(cube);

    ASSERT_EQ(tensor.data_size(), 1);
    ASSERT_EQ(tensor.up_chirp_data_size(), 1);
    ASSERT_EQ(tensor.down_chirp_data_size(), 1);
    EXPECT_DOUBLE_EQ(tensor.up_chirp_data(0).wavelength_m(), 1550e-9);
    EXPECT_DOUBLE_EQ(tensor.down_chirp_data(0).wavelength_m(), 1550e-9);
    ASSERT_EQ(tensor.up_chirp_data(0).detectors_size(), 8);
    ASSERT_EQ(tensor.down_chirp_data(0).detectors_size(), 8);

    for (int d = 0; d < 8; ++d) {
        const auto& cube_up = cube.up_chirps(0).detectors(d);
        const auto& cube_down = cube.down_chirps(0).detectors(d);
        const auto& tensor_up = tensor.up_chirp_data(0).detectors(d);
        const auto& tensor_down = tensor.down_chirp_data(0).detectors(d);

        EXPECT_EQ(tensor_up.detector_index(), cube_up.detector_index());
        EXPECT_EQ(tensor_down.detector_index(), cube_down.detector_index());
        EXPECT_TRUE(tensor_up.has_k_vector());
        EXPECT_TRUE(tensor_down.has_k_vector());

        // Up/down K-vectors match the averaged channel's K-vector
        const auto& tensor_avg = tensor.data(0).detectors(d);
        EXPECT_DOUBLE_EQ(tensor_up.k_vector().kx(), tensor_avg.k_vector().kx());
        EXPECT_DOUBLE_EQ(tensor_down.k_vector().kx(), tensor_avg.k_vector().kx());

        ASSERT_EQ(tensor_up.range_data_size(), cube_up.bins_size());
        ASSERT_EQ(tensor_down.range_data_size(), cube_down.bins_size());
        for (int b = 0; b < cube_up.bins_size(); ++b) {
            EXPECT_DOUBLE_EQ(tensor_up.range_data(b).real(), cube_up.bins(b).real());
            EXPECT_DOUBLE_EQ(tensor_up.range_data(b).imag(), cube_up.bins(b).imag());
            EXPECT_DOUBLE_EQ(tensor_down.range_data(b).real(), cube_down.bins(b).real());
            EXPECT_DOUBLE_EQ(tensor_down.range_data(b).imag(), cube_down.bins(b).imag());
        }
    }
}

}  // namespace
}  // namespace aurora::dsp
