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

#ifndef AURORA_RECONSTRUCTION_WARM_START_CACHE_H_
#define AURORA_RECONSTRUCTION_WARM_START_CACHE_H_

/**
 * @file warm_start_cache.h
 * @brief Cache for warm-starting PCG from previous frame solutions.
 *
 * When scene content changes slowly between frames, warm-starting PCG from
 * the previous solution can significantly reduce iteration count (from 5-6
 * down to 2-3 iterations).
 *
 * The cache stores per-range-bin solutions and implements change detection
 * to decide between using the previous solution (stable scene) or the dirty
 * image (scene changed significantly).
 *
 * ### Thread Safety
 *
 * This class is NOT thread-safe. External synchronization is required if
 * multiple threads access the same cache instance. Statistics counters use
 * std::atomic for safe concurrent reads, but the cache data structures
 * themselves are not protected.
 */

#include <Eigen/Core>
#include <atomic>
#include <optional>
#include <unordered_map>

#include "pcg_config.pb.h"

namespace aurora::reconstruction {

/**
 * @brief Cache entry for one range bin.
 */
struct WarmStartEntry {
    Eigen::MatrixXcd solution;     ///< Previous PCG solution
    Eigen::MatrixXcd dirty_image;  ///< Previous b = A†Wy (for change detection)
    uint64_t frame_id;             ///< Frame this was computed for
};

/**
 * @brief Result from get() indicating what initial guess to use.
 *
 * The initial_guess is returned by value (via std::optional) to avoid
 * lifetime issues with pointers to internal cache data.
 */
struct WarmStartResult {
    std::optional<Eigen::MatrixXcd> initial_guess;  ///< Initial guess if available
    bool use_previous_solution;                     ///< true = previous solution, false = use dirty image
};

/**
 * @brief Cache for warm-starting PCG from previous solutions.
 *
 * Stores previous frame solutions per-range-bin and implements change detection
 * to decide the best initialization strategy:
 *
 * | Initialization           | When to Use              | Expected Iterations |
 * |--------------------------|--------------------------|---------------------|
 * | Previous frame solution  | Scene changes slowly     | 2-3                 |
 * | Dirty image (A†Wy)       | Scene changed significantly | 4-5              |
 * | Zero                     | First frame only         | 5-6                 |
 *
 * ### Change Detection
 *
 * Scene change is detected by comparing normalized backprojection difference:
 * @code
 * change_ratio = ||b_current - b_previous|| / ||b_current||
 * if (change_ratio > threshold) use dirty image else use previous solution
 * @endcode
 *
 * ### Memory Usage
 *
 * For 64×64 grid with cache_depth=1:
 * - Solution: 64×64×16 bytes = 64 KB per range bin
 * - Dirty image: 64 KB per range bin
 * - Total: ~128 KB per range bin (~12.5 MB for 100 range bins)
 *
 * @see PcgOperator for usage within PCG reconstruction
 * @see WarmStartConfig for configuration parameters
 */
class WarmStartCache {
  public:
    /**
     * @brief Construct cache with given configuration.
     *
     * @param config WarmStartConfig specifying change_threshold and cache_depth
     * @throws std::invalid_argument if config parameters are out of valid range
     */
    explicit WarmStartCache(const WarmStartConfig& config);

    /**
     * @brief Get warm start for a range bin.
     *
     * Returns the best available initial guess based on change detection:
     * - If no cache entry exists: initial_guess is empty (use zero initialization)
     * - If entry too old: initial_guess is empty
     * - If scene changed significantly: initial_guess is empty, use_previous_solution=false
     *   (caller should use current_dirty_image as initial guess)
     * - If scene stable: initial_guess contains previous solution, use_previous_solution=true
     *
     * @param range_bin Range bin index
     * @param current_dirty_image Current frame's dirty image (A†Wy)
     * @param current_frame_id Current frame identifier
     * @return WarmStartResult with initial guess (by value) and usage flag
     */
    WarmStartResult get(
        uint32_t range_bin,
        const Eigen::MatrixXcd& current_dirty_image,
        uint64_t current_frame_id) const;

    /**
     * @brief Store solution after PCG completes.
     *
     * @param range_bin Range bin index
     * @param solution PCG solution to cache
     * @param dirty_image Dirty image used (for future change detection)
     * @param frame_id Frame identifier
     */
    void store(
        uint32_t range_bin,
        const Eigen::MatrixXcd& solution,
        const Eigen::MatrixXcd& dirty_image,
        uint64_t frame_id);

    /**
     * @brief Clear all cached entries.
     *
     * Call this when the receiver geometry changes or calibration is updated,
     * as cached solutions are no longer valid.
     */
    void clear();

    /** @brief Get number of range bins with cached entries. */
    size_t size() const { return cache_.size(); }

    /** @brief Get total cache hit count. */
    size_t hits() const { return hits_.load(std::memory_order_relaxed); }

    /** @brief Get total cache miss count. */
    size_t misses() const { return misses_.load(std::memory_order_relaxed); }

    /** @brief Get count of times previous solution was used. */
    size_t previous_solution_uses() const { return previous_solution_uses_.load(std::memory_order_relaxed); }

    /** @brief Get count of times dirty image was used instead. */
    size_t dirty_image_uses() const { return dirty_image_uses_.load(std::memory_order_relaxed); }

    /** @brief Reset statistics counters. */
    void reset_statistics();

    /** @brief Get configuration. */
    const WarmStartConfig& config() const { return config_; }

  private:
    WarmStartConfig config_;
    std::unordered_map<uint32_t, WarmStartEntry> cache_;

    // Statistics (mutable + atomic for safe concurrent reads in const get())
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
    mutable std::atomic<size_t> previous_solution_uses_{0};
    mutable std::atomic<size_t> dirty_image_uses_{0};

    /**
     * @brief Detect if scene changed significantly.
     *
     * @param current_dirty Current frame's dirty image
     * @param previous_dirty Previous frame's dirty image
     * @return true if scene changed beyond threshold (use dirty image)
     */
    bool should_use_dirty_image(
        const Eigen::MatrixXcd& current_dirty,
        const Eigen::MatrixXcd& previous_dirty) const;
};

/**
 * @brief Validate WarmStartConfig fields.
 *
 * @param config WarmStartConfig to validate
 * @throws std::invalid_argument if any field is invalid
 */
void validate_warm_start_config(const WarmStartConfig& config);

}  // namespace aurora::reconstruction

#endif  // AURORA_RECONSTRUCTION_WARM_START_CACHE_H_
