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

#ifndef AURORA_ACQUISITION_SEQUENCE_PLANNER_H_
#define AURORA_ACQUISITION_SEQUENCE_PLANNER_H_

/**
 * @file sequence_planner.h
 * @brief Plans chirp timing and wavelength interleaving for each burst.
 *
 * The SequencePlanner expands a SequenceSchedule into the concrete chirp
 * list of one acquisition burst: chirp start times, durations, up/down
 * directions, and which laser wavelength(s) are active during each chirp.
 *
 * ### Wavelength Interleaving Modes
 *
 * **Alternating Mode (e.g. 905nm + 1550nm)**:
 * ```
 * Burst N:
 *   t=0:      Switch to λ₁ (905nm)
 *   t=0-50μs: 4 chirps at λ₁
 *   t=50μs:   Switch to λ₂ (1550nm)
 *   t=50-100μs: 4 chirps at λ₂
 *   t=100μs:  Next burst
 * ```
 *
 * **Simultaneous Mode**:
 * Both wavelengths active for all chirps of the burst.
 *
 * @see SequenceSchedule for configuration
 */

#include <cstdint>
#include <vector>

#include "sequence_plan.pb.h"
#include "sequence_schedule.pb.h"

namespace aurora::acquisition {

/**
 * @brief Plans chirp timing and wavelength interleaving for each burst.
 *
 * ### Example Usage
 * @code
 * SequenceSchedule schedule;
 * schedule.set_burst_period_s(100e-6);  // burst period in seconds
 *
 * auto* chirp = schedule.mutable_chirp_schedule();
 * chirp->set_chirp_duration_s(12.5e-6);
 * chirp->set_chirps_per_burst(8);
 *
 * auto* wavelength = schedule.mutable_wavelength_schedule();
 * wavelength->set_mode(WavelengthSchedule::INTERLEAVING_ALTERNATING);
 * wavelength->add_wavelengths_m(905e-9);
 * wavelength->add_wavelengths_m(1550e-9);
 *
 * SequencePlanner planner(schedule);
 * SequencePlan plan = planner.plan_burst(0, 0.0);
 * @endcode
 */
class SequencePlanner {
  public:
    /**
     * @brief Construct a sequence planner.
     * @param schedule Sequence configuration (timing, wavelengths)
     */
    explicit SequencePlanner(const SequenceSchedule& schedule);

    /**
     * @brief Generate the chirp plan for one burst.
     *
     * @param burst_id Unique burst identifier (becomes plan_id)
     * @param timestamp_s Absolute start time of the burst in seconds
     * @return Complete sequence plan for this burst
     * @throws std::invalid_argument if the schedule has no chirp_schedule
     */
    SequencePlan plan_burst(uint64_t burst_id, double timestamp_s) const;

    double get_burst_period_s() const;
    double get_chirp_duration_s() const;
    uint32_t get_chirps_per_burst() const;

  private:
    SequenceSchedule schedule_;

    std::vector<ChirpPlan> generate_alternating_chirps(
        uint32_t total_chirps,
        double chirp_duration,
        double bandwidth,
        ChirpSchedule::ChirpMode chirp_mode) const;

    std::vector<ChirpPlan> generate_simultaneous_chirps(
        uint32_t total_chirps,
        double chirp_duration,
        double bandwidth,
        ChirpSchedule::ChirpMode chirp_mode) const;

    ChirpPlan create_chirp(uint32_t index,
                           double start_time_s,
                           double duration_s,
                           double bandwidth_hz,
                           ChirpPlan::Direction direction,
                           const std::vector<double>& wavelengths_m) const;
};

}  // namespace aurora::acquisition

#endif  // AURORA_ACQUISITION_SEQUENCE_PLANNER_H_
