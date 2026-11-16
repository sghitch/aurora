#include "reconstruction/warm_start_cache.h"

#include <stdexcept>

namespace aurora::reconstruction {

void validate_warm_start_config(const WarmStartConfig& config) {
    if (config.change_threshold() <= 0.0 || config.change_threshold() > 1.0) {
        throw std::invalid_argument(
            "WarmStartConfig: change_threshold must be in (0, 1], got " +
            std::to_string(config.change_threshold()));
    }
    if (config.cache_depth() < 1) {
        throw std::invalid_argument(
            "WarmStartConfig: cache_depth must be >= 1, got " +
            std::to_string(config.cache_depth()));
    }
}

WarmStartCache::WarmStartCache(const WarmStartConfig& config)
    : config_(config) {
    validate_warm_start_config(config_);
}

bool WarmStartCache::should_use_dirty_image(
    const Eigen::MatrixXcd& current_dirty,
    const Eigen::MatrixXcd& previous_dirty) const {

    double change = (current_dirty - previous_dirty).norm();
    double scale = current_dirty.norm();

    // Handle near-zero signal - use dirty image for safety
    if (scale < 1e-10) {
        return true;
    }

    double ratio = change / scale;
    return ratio > config_.change_threshold();
}

WarmStartResult WarmStartCache::get(
    uint32_t range_bin,
    const Eigen::MatrixXcd& current_dirty_image,
    uint64_t current_frame_id) const {

    WarmStartResult result{std::nullopt, false};

    auto it = cache_.find(range_bin);
    if (it == cache_.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    const auto& entry = it->second;

    // Check if entry is too old (beyond cache_depth frames)
    if (current_frame_id > entry.frame_id &&
        (current_frame_id - entry.frame_id) > config_.cache_depth()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    // Check if scene changed significantly
    if (should_use_dirty_image(current_dirty_image, entry.dirty_image)) {
        // Scene changed - caller should use current_dirty_image as initial guess
        result.initial_guess = std::nullopt;
        result.use_previous_solution = false;
        hits_.fetch_add(1, std::memory_order_relaxed);
        dirty_image_uses_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Scene stable - return copy of previous solution
        result.initial_guess = entry.solution;
        result.use_previous_solution = true;
        hits_.fetch_add(1, std::memory_order_relaxed);
        previous_solution_uses_.fetch_add(1, std::memory_order_relaxed);
    }

    return result;
}

void WarmStartCache::store(
    uint32_t range_bin,
    const Eigen::MatrixXcd& solution,
    const Eigen::MatrixXcd& dirty_image,
    uint64_t frame_id) {

    cache_[range_bin] = WarmStartEntry{solution, dirty_image, frame_id};
}

void WarmStartCache::clear() {
    cache_.clear();
    reset_statistics();
}

void WarmStartCache::reset_statistics() {
    hits_.store(0, std::memory_order_relaxed);
    misses_.store(0, std::memory_order_relaxed);
    previous_solution_uses_.store(0, std::memory_order_relaxed);
    dirty_image_uses_.store(0, std::memory_order_relaxed);
}

}  // namespace aurora::reconstruction
