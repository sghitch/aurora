#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>

#include "forward/k_space_slice_utils.h"

namespace aurora::forward {
namespace {

constexpr double kTwoPi = 2.0 * M_PI;

class KSpaceSliceUtilsTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Create a sample KSpaceSlice with 4 samples at corners of a square
        slice_.set_range_bin(10);
        slice_.set_wavelength_m(1550e-9);
        slice_.set_burst_id(12345);
        slice_.set_timestamp_s(1.5);

        // Sample 1: lower-left
        auto* s1 = slice_.add_samples();
        s1->mutable_value()->set_real(1.0);
        s1->mutable_value()->set_imag(0.0);
        s1->set_kx(-100.0);
        s1->set_ky(-100.0);
        s1->set_weight(0.8);
        s1->set_detector_index(0);

        // Sample 2: lower-right
        auto* s2 = slice_.add_samples();
        s2->mutable_value()->set_real(0.0);
        s2->mutable_value()->set_imag(1.0);
        s2->set_kx(100.0);
        s2->set_ky(-100.0);
        s2->set_weight(0.7);
        s2->set_detector_index(1);

        // Sample 3: upper-left
        auto* s3 = slice_.add_samples();
        s3->mutable_value()->set_real(-1.0);
        s3->mutable_value()->set_imag(0.0);
        s3->set_kx(-100.0);
        s3->set_ky(100.0);
        s3->set_weight(0.9);
        s3->set_detector_index(0);

        // Sample 4: upper-right
        auto* s4 = slice_.add_samples();
        s4->mutable_value()->set_real(0.5);
        s4->mutable_value()->set_imag(0.5);
        s4->set_kx(100.0);
        s4->set_ky(100.0);
        s4->set_weight(0.85);
        s4->set_detector_index(1);
    }

    aurora::shared::KSpaceSlice slice_;
};

TEST_F(KSpaceSliceUtilsTest, ToEigenSamples) {
    auto samples = to_eigen_samples(slice_);

    ASSERT_EQ(samples.size(), 4);
    EXPECT_DOUBLE_EQ(samples(0).real(), 1.0);
    EXPECT_DOUBLE_EQ(samples(0).imag(), 0.0);
    EXPECT_DOUBLE_EQ(samples(1).real(), 0.0);
    EXPECT_DOUBLE_EQ(samples(1).imag(), 1.0);
    EXPECT_DOUBLE_EQ(samples(2).real(), -1.0);
    EXPECT_DOUBLE_EQ(samples(2).imag(), 0.0);
    EXPECT_DOUBLE_EQ(samples(3).real(), 0.5);
    EXPECT_DOUBLE_EQ(samples(3).imag(), 0.5);
}

TEST_F(KSpaceSliceUtilsTest, ToKVectors) {
    auto k_vectors = to_k_vectors(slice_);

    ASSERT_EQ(k_vectors.size(), 4);
    EXPECT_DOUBLE_EQ(k_vectors[0].x(), -100.0);
    EXPECT_DOUBLE_EQ(k_vectors[0].y(), -100.0);
    EXPECT_DOUBLE_EQ(k_vectors[1].x(), 100.0);
    EXPECT_DOUBLE_EQ(k_vectors[1].y(), -100.0);
    EXPECT_DOUBLE_EQ(k_vectors[2].x(), -100.0);
    EXPECT_DOUBLE_EQ(k_vectors[2].y(), 100.0);
    EXPECT_DOUBLE_EQ(k_vectors[3].x(), 100.0);
    EXPECT_DOUBLE_EQ(k_vectors[3].y(), 100.0);
}

TEST_F(KSpaceSliceUtilsTest, ToWeights) {
    auto weights = to_weights(slice_);

    ASSERT_EQ(weights.size(), 4);
    EXPECT_DOUBLE_EQ(weights[0], 0.8);
    EXPECT_DOUBLE_EQ(weights[1], 0.7);
    EXPECT_DOUBLE_EQ(weights[2], 0.9);
    EXPECT_DOUBLE_EQ(weights[3], 0.85);
}

TEST_F(KSpaceSliceUtilsTest, ComputeKBounds) {
    auto [kx_min, kx_max, ky_min, ky_max] = compute_k_bounds(slice_);

    EXPECT_DOUBLE_EQ(kx_min, -100.0);
    EXPECT_DOUBLE_EQ(kx_max, 100.0);
    EXPECT_DOUBLE_EQ(ky_min, -100.0);
    EXPECT_DOUBLE_EQ(ky_max, 100.0);
}

TEST_F(KSpaceSliceUtilsTest, SuggestGridSize) {
    auto grid_size = suggest_grid_size(slice_, 2.0);

    // K-extent is 200 rad/m in both dimensions
    // n = ceil(200 / (2*pi) * 2.0) = ceil(63.66) = 64
    int expected = static_cast<int>(std::ceil(200.0 / kTwoPi * 2.0));
    EXPECT_EQ(grid_size.x(), expected);
    EXPECT_EQ(grid_size.y(), expected);
}

TEST_F(KSpaceSliceUtilsTest, ComputeGridHints) {
    auto hints = compute_grid_hints(slice_, 2.0);

    EXPECT_DOUBLE_EQ(hints.kx_min(), -100.0);
    EXPECT_DOUBLE_EQ(hints.kx_max(), 100.0);
    EXPECT_DOUBLE_EQ(hints.ky_min(), -100.0);
    EXPECT_DOUBLE_EQ(hints.ky_max(), 100.0);

    int expected = static_cast<int>(std::ceil(200.0 / kTwoPi * 2.0));
    EXPECT_EQ(hints.suggested_nx(), static_cast<uint32_t>(expected));
    EXPECT_EQ(hints.suggested_ny(), static_cast<uint32_t>(expected));
}

TEST_F(KSpaceSliceUtilsTest, EmptySlice) {
    aurora::shared::KSpaceSlice empty_slice;

    EXPECT_TRUE(is_empty(empty_slice));
    EXPECT_EQ(num_samples(empty_slice), 0);

    auto samples = to_eigen_samples(empty_slice);
    EXPECT_EQ(samples.size(), 0);

    auto k_vectors = to_k_vectors(empty_slice);
    EXPECT_TRUE(k_vectors.empty());

    auto weights = to_weights(empty_slice);
    EXPECT_TRUE(weights.empty());

    auto [kx_min, kx_max, ky_min, ky_max] = compute_k_bounds(empty_slice);
    EXPECT_DOUBLE_EQ(kx_min, 0.0);
    EXPECT_DOUBLE_EQ(kx_max, 0.0);
    EXPECT_DOUBLE_EQ(ky_min, 0.0);
    EXPECT_DOUBLE_EQ(ky_max, 0.0);

    auto grid_size = suggest_grid_size(empty_slice);
    EXPECT_EQ(grid_size.x(), 0);
    EXPECT_EQ(grid_size.y(), 0);
}

TEST_F(KSpaceSliceUtilsTest, NonEmptySlice) {
    EXPECT_FALSE(is_empty(slice_));
    EXPECT_EQ(num_samples(slice_), 4);
}

TEST_F(KSpaceSliceUtilsTest, SerializationRoundtrip) {
    // Serialize the slice to string
    std::string serialized;
    ASSERT_TRUE(slice_.SerializeToString(&serialized));

    // Deserialize into a new slice
    aurora::shared::KSpaceSlice restored;
    ASSERT_TRUE(restored.ParseFromString(serialized));

    // Verify all fields are preserved
    EXPECT_EQ(restored.range_bin(), slice_.range_bin());
    EXPECT_DOUBLE_EQ(restored.wavelength_m(), slice_.wavelength_m());
    EXPECT_EQ(restored.burst_id(), slice_.burst_id());
    EXPECT_DOUBLE_EQ(restored.timestamp_s(), slice_.timestamp_s());
    ASSERT_EQ(restored.samples_size(), slice_.samples_size());

    // Verify sample data
    for (int i = 0; i < slice_.samples_size(); ++i) {
        const auto& orig = slice_.samples(i);
        const auto& rest = restored.samples(i);

        EXPECT_DOUBLE_EQ(rest.value().real(), orig.value().real());
        EXPECT_DOUBLE_EQ(rest.value().imag(), orig.value().imag());
        EXPECT_DOUBLE_EQ(rest.kx(), orig.kx());
        EXPECT_DOUBLE_EQ(rest.ky(), orig.ky());
        EXPECT_DOUBLE_EQ(rest.weight(), orig.weight());
        EXPECT_EQ(rest.detector_index(), orig.detector_index());
    }
}

TEST_F(KSpaceSliceUtilsTest, SingleSample) {
    aurora::shared::KSpaceSlice single;
    single.set_range_bin(0);
    single.set_wavelength_m(905e-9);

    auto* s = single.add_samples();
    s->mutable_value()->set_real(2.0);
    s->mutable_value()->set_imag(-1.0);
    s->set_kx(50.0);
    s->set_ky(-25.0);
    s->set_weight(1.0);

    auto samples = to_eigen_samples(single);
    ASSERT_EQ(samples.size(), 1);
    EXPECT_DOUBLE_EQ(samples(0).real(), 2.0);
    EXPECT_DOUBLE_EQ(samples(0).imag(), -1.0);

    auto k_vectors = to_k_vectors(single);
    ASSERT_EQ(k_vectors.size(), 1);
    EXPECT_DOUBLE_EQ(k_vectors[0].x(), 50.0);
    EXPECT_DOUBLE_EQ(k_vectors[0].y(), -25.0);

    auto [kx_min, kx_max, ky_min, ky_max] = compute_k_bounds(single);
    EXPECT_DOUBLE_EQ(kx_min, 50.0);
    EXPECT_DOUBLE_EQ(kx_max, 50.0);
    EXPECT_DOUBLE_EQ(ky_min, -25.0);
    EXPECT_DOUBLE_EQ(ky_max, -25.0);

    // Single sample should give grid size of 1 (minimum)
    auto grid_size = suggest_grid_size(single);
    EXPECT_EQ(grid_size.x(), 1);
    EXPECT_EQ(grid_size.y(), 1);
}

TEST_F(KSpaceSliceUtilsTest, OptionalBandwidth) {
    auto* sample = slice_.mutable_samples(0);
    sample->set_kx_bandwidth(10.0);
    sample->set_ky_bandwidth(15.0);

    EXPECT_TRUE(sample->has_kx_bandwidth());
    EXPECT_TRUE(sample->has_ky_bandwidth());
    EXPECT_DOUBLE_EQ(sample->kx_bandwidth(), 10.0);
    EXPECT_DOUBLE_EQ(sample->ky_bandwidth(), 15.0);

    // Second sample should not have bandwidth set
    EXPECT_FALSE(slice_.samples(1).has_kx_bandwidth());
    EXPECT_FALSE(slice_.samples(1).has_ky_bandwidth());
}

TEST_F(KSpaceSliceUtilsTest, KSpaceSliceCollection) {
    aurora::shared::KSpaceSliceCollection collection;
    collection.set_wavelength_m(1550e-9);

    // Add slices for multiple range bins
    for (uint32_t bin = 0; bin < 3; ++bin) {
        auto* new_slice = collection.add_slices();
        new_slice->set_range_bin(bin);
        new_slice->set_wavelength_m(1550e-9);

        auto* sample = new_slice->add_samples();
        sample->mutable_value()->set_real(static_cast<double>(bin));
        sample->mutable_value()->set_imag(0.0);
        sample->set_kx(0.0);
        sample->set_ky(0.0);
        sample->set_weight(1.0);
    }

    EXPECT_EQ(collection.slices_size(), 3);
    EXPECT_DOUBLE_EQ(collection.wavelength_m(), 1550e-9);

    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(collection.slices(i).range_bin(), static_cast<uint32_t>(i));
        EXPECT_DOUBLE_EQ(collection.slices(i).samples(0).value().real(),
                        static_cast<double>(i));
    }
}

}  // namespace
}  // namespace aurora::forward
