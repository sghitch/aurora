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

#include "reconstruction/warm_start_cache.h"

#include <gtest/gtest.h>

namespace aurora::reconstruction {

// =============================================================================
// Configuration Validation Tests
// =============================================================================

TEST(WarmStartConfigValidation, RejectsZeroThreshold) {
    WarmStartConfig config;
    config.set_change_threshold(0.0);
    config.set_cache_depth(1);
    EXPECT_THROW(validate_warm_start_config(config), std::invalid_argument);
}

TEST(WarmStartConfigValidation, RejectsNegativeThreshold) {
    WarmStartConfig config;
    config.set_change_threshold(-0.5);
    config.set_cache_depth(1);
    EXPECT_THROW(validate_warm_start_config(config), std::invalid_argument);
}

TEST(WarmStartConfigValidation, RejectsThresholdAboveOne) {
    WarmStartConfig config;
    config.set_change_threshold(1.5);
    config.set_cache_depth(1);
    EXPECT_THROW(validate_warm_start_config(config), std::invalid_argument);
}

TEST(WarmStartConfigValidation, AcceptsThresholdAtOne) {
    WarmStartConfig config;
    config.set_change_threshold(1.0);
    config.set_cache_depth(1);
    EXPECT_NO_THROW(validate_warm_start_config(config));
}

TEST(WarmStartConfigValidation, AcceptsThresholdJustAboveZero) {
    WarmStartConfig config;
    config.set_change_threshold(0.001);
    config.set_cache_depth(1);
    EXPECT_NO_THROW(validate_warm_start_config(config));
}

TEST(WarmStartConfigValidation, RejectsZeroCacheDepth) {
    WarmStartConfig config;
    config.set_change_threshold(0.3);
    config.set_cache_depth(0);
    EXPECT_THROW(validate_warm_start_config(config), std::invalid_argument);
}

TEST(WarmStartConfigValidation, AcceptsCacheDepthOne) {
    WarmStartConfig config;
    config.set_change_threshold(0.3);
    config.set_cache_depth(1);
    EXPECT_NO_THROW(validate_warm_start_config(config));
}

TEST(WarmStartConfigValidation, AcceptsLargeCacheDepth) {
    WarmStartConfig config;
    config.set_change_threshold(0.3);
    config.set_cache_depth(100);
    EXPECT_NO_THROW(validate_warm_start_config(config));
}

// =============================================================================
// Basic Operation Tests
// =============================================================================

class WarmStartCacheTest : public ::testing::Test {
  protected:
    void SetUp() override {
        config_.set_change_threshold(0.5);
        config_.set_cache_depth(2);
    }

    WarmStartConfig config_;

    WarmStartCache make_cache() const {
        return WarmStartCache(config_);
    }
};

TEST_F(WarmStartCacheTest, ConstructorValidatesConfig) {
    WarmStartConfig bad_config;
    bad_config.set_change_threshold(0.0);
    bad_config.set_cache_depth(1);
    EXPECT_THROW({ WarmStartCache cache(bad_config); }, std::invalid_argument);
}

TEST_F(WarmStartCacheTest, InitialMiss) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);
    auto result = cache.get(0, dirty, 1);

    EXPECT_FALSE(result.initial_guess.has_value());
    EXPECT_FALSE(result.use_previous_solution);
    EXPECT_EQ(cache.misses(), 1u);
    EXPECT_EQ(cache.hits(), 0u);
}

TEST_F(WarmStartCacheTest, StoreAndRetrieveStableScene) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    // Store solution for frame 1
    cache.store(0, solution, dirty, 1);

    // Get for frame 2 with same dirty image (no change)
    auto result = cache.get(0, dirty, 2);

    EXPECT_TRUE(result.initial_guess.has_value());
    EXPECT_TRUE(result.use_previous_solution);
    EXPECT_EQ(cache.hits(), 1u);
    EXPECT_EQ(cache.previous_solution_uses(), 1u);
}

TEST_F(WarmStartCacheTest, StoreAndRetrieveChangedScene) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8);
    Eigen::MatrixXcd dirty2 = Eigen::MatrixXcd::Ones(8, 8) * 10.0;  // Big change

    // Store solution for frame 1
    cache.store(0, solution, dirty1, 1);

    // Get for frame 2 with different dirty image
    auto result = cache.get(0, dirty2, 2);

    // Scene changed significantly, should indicate dirty image usage
    EXPECT_FALSE(result.initial_guess.has_value());
    EXPECT_FALSE(result.use_previous_solution);
    EXPECT_EQ(cache.hits(), 1u);  // Still a hit (used for comparison)
    EXPECT_EQ(cache.dirty_image_uses(), 1u);
}

TEST_F(WarmStartCacheTest, SeparateRangeBins) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution1 = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd solution2 = Eigen::MatrixXcd::Random(8, 8) * 2.0;
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    // Store different solutions for different range bins
    cache.store(0, solution1, dirty, 1);
    cache.store(5, solution2, dirty, 1);

    // Verify they are independent
    auto result0 = cache.get(0, dirty, 2);
    auto result5 = cache.get(5, dirty, 2);

    EXPECT_TRUE(result0.initial_guess.has_value());
    EXPECT_TRUE(result5.initial_guess.has_value());

    // Solutions should be different
    EXPECT_FALSE(result0.initial_guess->isApprox(*result5.initial_guess));

    EXPECT_EQ(cache.size(), 2u);
}

// =============================================================================
// Change Detection Tests
// =============================================================================

TEST_F(WarmStartCacheTest, DetectsSmallChange) {
    config_.set_change_threshold(0.5);
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8);
    Eigen::MatrixXcd dirty2 = dirty1 * 1.05;  // 5% change

    cache.store(0, solution, dirty1, 1);
    auto result = cache.get(0, dirty2, 2);

    // Small change should use previous solution
    EXPECT_TRUE(result.use_previous_solution);
    EXPECT_TRUE(result.initial_guess.has_value());
}

TEST_F(WarmStartCacheTest, DetectsLargeChange) {
    config_.set_change_threshold(0.5);
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8);
    // For ratio = ||d2-d1||/||d2|| > 0.5:
    // If d2 = k*d1, then ratio = |k-1|/k. For k=3: ratio = 2/3 ≈ 0.67 > 0.5
    Eigen::MatrixXcd dirty2 = dirty1 * 3.0;

    cache.store(0, solution, dirty1, 1);
    auto result = cache.get(0, dirty2, 2);

    // Large change should use dirty image
    EXPECT_FALSE(result.use_previous_solution);
}

TEST_F(WarmStartCacheTest, ThresholdBoundaryBelowUsePrevious) {
    config_.set_change_threshold(0.5);
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    // Create dirty images where change_ratio is just below threshold
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8);
    // For change_ratio = ||d2-d1||/||d2|| = 0.49 < 0.5
    // If d1 = 1, d2 = k*d1 = k, then |d2-d1| = |k-1|, |d2| = k
    // ratio = |k-1|/k = 1 - 1/k = 0.49 => k = 1/0.51 ≈ 1.96
    Eigen::MatrixXcd dirty2 = dirty1 * 1.9;  // ratio ≈ 0.47 < 0.5

    cache.store(0, solution, dirty1, 1);
    auto result = cache.get(0, dirty2, 2);

    EXPECT_TRUE(result.use_previous_solution);
}

TEST_F(WarmStartCacheTest, ThresholdBoundaryAboveUseDirty) {
    config_.set_change_threshold(0.5);
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8);
    // ratio = |k-1|/k > 0.5 => k > 2
    Eigen::MatrixXcd dirty2 = dirty1 * 2.1;  // ratio ≈ 0.52 > 0.5

    cache.store(0, solution, dirty1, 1);
    auto result = cache.get(0, dirty2, 2);

    EXPECT_FALSE(result.use_previous_solution);
}

TEST_F(WarmStartCacheTest, HandlesZeroSignal) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8);
    Eigen::MatrixXcd dirty2 = Eigen::MatrixXcd::Zero(8, 8);  // Zero signal

    cache.store(0, solution, dirty1, 1);
    auto result = cache.get(0, dirty2, 2);

    // Zero signal should safely use dirty image
    EXPECT_FALSE(result.use_previous_solution);
}

TEST_F(WarmStartCacheTest, HandlesNearZeroSignal) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8) * 1e-15;
    Eigen::MatrixXcd dirty2 = Eigen::MatrixXcd::Ones(8, 8) * 1e-15;

    cache.store(0, solution, dirty1, 1);
    auto result = cache.get(0, dirty2, 2);

    // Near-zero signal should safely use dirty image
    EXPECT_FALSE(result.use_previous_solution);
}

// =============================================================================
// Cache Depth Tests
// =============================================================================

TEST_F(WarmStartCacheTest, RespectsDepthOne) {
    config_.set_cache_depth(1);
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    // Store for frame 1
    cache.store(0, solution, dirty, 1);

    // Frame 2 should hit
    auto result2 = cache.get(0, dirty, 2);
    EXPECT_TRUE(result2.initial_guess.has_value());

    // Frame 3 is beyond depth=1 from frame 1
    auto result3 = cache.get(0, dirty, 3);
    EXPECT_FALSE(result3.initial_guess.has_value());
}

TEST_F(WarmStartCacheTest, RespectsDepthTwo) {
    config_.set_cache_depth(2);
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    // Store for frame 1
    cache.store(0, solution, dirty, 1);

    // Frame 2 and 3 should hit (within depth=2)
    auto result2 = cache.get(0, dirty, 2);
    EXPECT_TRUE(result2.initial_guess.has_value());

    auto result3 = cache.get(0, dirty, 3);
    EXPECT_TRUE(result3.initial_guess.has_value());

    // Frame 4 is beyond depth=2 from frame 1
    auto result4 = cache.get(0, dirty, 4);
    EXPECT_FALSE(result4.initial_guess.has_value());
}

TEST_F(WarmStartCacheTest, EvictsOldEntries) {
    config_.set_cache_depth(2);
    WarmStartCache cache(config_);

    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    // Store 3 entries for same range bin
    cache.store(0, Eigen::MatrixXcd::Random(8, 8), dirty, 1);
    cache.store(0, Eigen::MatrixXcd::Random(8, 8), dirty, 2);
    cache.store(0, Eigen::MatrixXcd::Random(8, 8), dirty, 3);

    // Should still only have 1 range bin
    EXPECT_EQ(cache.size(), 1u);

    // The most recent entry (frame 3) should be used
    auto result = cache.get(0, dirty, 4);
    EXPECT_TRUE(result.initial_guess.has_value());
}

TEST_F(WarmStartCacheTest, UpdatesExistingEntry) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution1 = Eigen::MatrixXcd::Ones(8, 8);
    Eigen::MatrixXcd solution2 = Eigen::MatrixXcd::Ones(8, 8) * 2.0;
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    // Store first solution
    cache.store(0, solution1, dirty, 1);

    // Store second solution (same range bin)
    cache.store(0, solution2, dirty, 2);

    // Get should return most recent
    auto result = cache.get(0, dirty, 3);
    ASSERT_TRUE(result.initial_guess.has_value());
    EXPECT_TRUE(result.initial_guess->isApprox(solution2));
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(WarmStartCacheTest, TracksHits) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    cache.store(0, solution, dirty, 1);

    cache.get(0, dirty, 2);
    cache.get(0, dirty, 2);
    cache.get(0, dirty, 2);

    EXPECT_EQ(cache.hits(), 3u);
}

TEST_F(WarmStartCacheTest, TracksMisses) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    // Empty cache misses
    cache.get(0, dirty, 1);
    cache.get(1, dirty, 1);
    cache.get(2, dirty, 1);

    EXPECT_EQ(cache.misses(), 3u);
}

TEST_F(WarmStartCacheTest, TracksPreviousSolutionUses) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    cache.store(0, solution, dirty, 1);

    // Same dirty = use previous solution
    cache.get(0, dirty, 2);
    cache.get(0, dirty, 2);

    EXPECT_EQ(cache.previous_solution_uses(), 2u);
}

TEST_F(WarmStartCacheTest, TracksDirtyImageUses) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8);
    Eigen::MatrixXcd dirty2 = dirty1 * 10.0;  // Big change

    cache.store(0, solution, dirty1, 1);

    // Big change = use dirty image
    cache.get(0, dirty2, 2);

    EXPECT_EQ(cache.dirty_image_uses(), 1u);
}

// =============================================================================
// Clear Tests
// =============================================================================

TEST_F(WarmStartCacheTest, ClearRemovesAll) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    cache.store(0, solution, dirty, 1);
    cache.store(1, solution, dirty, 1);
    cache.store(2, solution, dirty, 1);

    EXPECT_EQ(cache.size(), 3u);

    cache.clear();

    EXPECT_EQ(cache.size(), 0u);

    // Should miss after clear
    auto result = cache.get(0, dirty, 2);
    EXPECT_FALSE(result.initial_guess.has_value());
}

TEST_F(WarmStartCacheTest, ClearResetsStats) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    cache.store(0, solution, dirty, 1);
    cache.get(0, dirty, 2);  // hit
    cache.get(5, dirty, 2);  // miss

    EXPECT_EQ(cache.hits(), 1u);
    EXPECT_EQ(cache.misses(), 1u);

    cache.clear();

    EXPECT_EQ(cache.hits(), 0u);
    EXPECT_EQ(cache.misses(), 0u);
    EXPECT_EQ(cache.previous_solution_uses(), 0u);
    EXPECT_EQ(cache.dirty_image_uses(), 0u);
}

TEST_F(WarmStartCacheTest, ResetStatisticsOnly) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    cache.store(0, solution, dirty, 1);
    cache.get(0, dirty, 2);

    EXPECT_EQ(cache.hits(), 1u);
    EXPECT_EQ(cache.size(), 1u);

    cache.reset_statistics();

    // Stats reset but data remains
    EXPECT_EQ(cache.hits(), 0u);
    EXPECT_EQ(cache.misses(), 0u);
    EXPECT_EQ(cache.size(), 1u);

    // Can still get cached data
    auto result = cache.get(0, dirty, 2);
    EXPECT_TRUE(result.initial_guess.has_value());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(WarmStartCacheTest, SameFrameIdStore) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution1 = Eigen::MatrixXcd::Ones(8, 8);
    Eigen::MatrixXcd solution2 = Eigen::MatrixXcd::Ones(8, 8) * 2.0;
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    // Store twice with same frame_id
    cache.store(0, solution1, dirty, 1);
    cache.store(0, solution2, dirty, 1);

    // Should get the latest one stored
    auto result = cache.get(0, dirty, 2);
    ASSERT_TRUE(result.initial_guess.has_value());
    EXPECT_TRUE(result.initial_guess->isApprox(solution2));
}

TEST_F(WarmStartCacheTest, LargeRangeBinIndex) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd dirty = Eigen::MatrixXcd::Ones(8, 8);

    uint32_t large_bin = 1000000;
    cache.store(large_bin, solution, dirty, 1);

    auto result = cache.get(large_bin, dirty, 2);
    EXPECT_TRUE(result.initial_guess.has_value());
}

TEST_F(WarmStartCacheTest, ConfigAccessor) {
    WarmStartCache cache(config_);

    EXPECT_DOUBLE_EQ(cache.config().change_threshold(), config_.change_threshold());
    EXPECT_EQ(cache.config().cache_depth(), config_.cache_depth());
}

TEST_F(WarmStartCacheTest, DifferentGridSizes) {
    WarmStartCache cache(config_);

    Eigen::MatrixXcd solution1 = Eigen::MatrixXcd::Random(8, 8);
    Eigen::MatrixXcd solution2 = Eigen::MatrixXcd::Random(16, 16);
    Eigen::MatrixXcd dirty1 = Eigen::MatrixXcd::Ones(8, 8);
    Eigen::MatrixXcd dirty2 = Eigen::MatrixXcd::Ones(16, 16);

    // Different range bins can have different grid sizes
    cache.store(0, solution1, dirty1, 1);
    cache.store(1, solution2, dirty2, 1);

    auto result0 = cache.get(0, dirty1, 2);
    auto result1 = cache.get(1, dirty2, 2);

    ASSERT_TRUE(result0.initial_guess.has_value());
    ASSERT_TRUE(result1.initial_guess.has_value());

    EXPECT_EQ(result0.initial_guess->rows(), 8);
    EXPECT_EQ(result1.initial_guess->rows(), 16);
}

}  // namespace aurora::reconstruction
