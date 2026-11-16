#ifndef AURORA_ACQUISITION_TIMING_UTILS_H_
#define AURORA_ACQUISITION_TIMING_UTILS_H_

/**
 * @file timing_utils.h
 * @brief Timing utilities for sequence planning.
 *
 * Provides helper functions for validating chirp and burst timing.
 */

#include <cstdint>

namespace aurora::acquisition {

/**
 * @brief Validate that chirp timing fits within the burst period.
 *
 * @param num_chirps Number of chirps per burst
 * @param chirp_duration_s Duration of each chirp in seconds
 * @param burst_period_s Burst period in seconds
 * @param num_wavelengths Number of wavelengths (for switching time)
 * @param switch_time_s Time to switch between wavelengths
 * @return true if timing is valid, false if chirps exceed the burst period
 */
inline bool validate_timing(uint32_t num_chirps,
                            double chirp_duration_s,
                            double burst_period_s,
                            uint32_t num_wavelengths = 1,
                            double switch_time_s = 0.0) {
    double total_chirp_time = num_chirps * chirp_duration_s;
    double total_switch_time = (num_wavelengths > 1) ?
        (num_wavelengths - 1) * switch_time_s : 0.0;
    return (total_chirp_time + total_switch_time) <= burst_period_s;
}

}  // namespace aurora::acquisition

#endif  // AURORA_ACQUISITION_TIMING_UTILS_H_
