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

#include "acquisition/sequence_planner.h"

#include <stdexcept>
#include <utility>

namespace aurora::acquisition {

namespace {

/// Time to switch laser wavelength between alternating blocks.
constexpr double kWavelengthSwitchTimeS = 1e-6;

}  // namespace

SequencePlanner::SequencePlanner(const SequenceSchedule& schedule)
    : schedule_(schedule) {}

double SequencePlanner::get_burst_period_s() const {
    return schedule_.burst_period_s();
}

double SequencePlanner::get_chirp_duration_s() const {
    if (!schedule_.has_chirp_schedule()) {
        throw std::invalid_argument("SequenceSchedule missing chirp_schedule");
    }
    return schedule_.chirp_schedule().chirp_duration_s();
}

uint32_t SequencePlanner::get_chirps_per_burst() const {
    if (!schedule_.has_chirp_schedule()) {
        throw std::invalid_argument("SequenceSchedule missing chirp_schedule");
    }
    return schedule_.chirp_schedule().chirps_per_burst();
}

ChirpPlan SequencePlanner::create_chirp(
    uint32_t index,
    double start_time_s,
    double duration_s,
    double bandwidth_hz,
    ChirpPlan::Direction direction,
    const std::vector<double>& wavelengths_m) const {
    ChirpPlan chirp;
    chirp.set_chirp_index(index);
    chirp.set_chirp_start_s(start_time_s);
    chirp.set_chirp_duration_s(duration_s);
    chirp.set_chirp_bandwidth_hz(bandwidth_hz);
    chirp.set_direction(direction);
    for (double wl : wavelengths_m) {
        chirp.add_wavelength_m(wl);
    }
    return chirp;
}

std::vector<ChirpPlan> SequencePlanner::generate_alternating_chirps(
    uint32_t total_chirps,
    double chirp_duration,
    double bandwidth,
    ChirpSchedule::ChirpMode chirp_mode) const {
    std::vector<ChirpPlan> chirps;
    chirps.reserve(total_chirps);

    const auto& wl_schedule = schedule_.wavelength_schedule();
    const int num_wavelengths = wl_schedule.wavelengths_m_size();
    if (num_wavelengths == 0) return chirps;

    const uint32_t chirps_per_wavelength = total_chirps / num_wavelengths;
    const double wavelength_block_time = chirps_per_wavelength * chirp_duration;

    uint32_t chirp_index = 0;
    for (int wl_idx = 0; wl_idx < num_wavelengths; ++wl_idx) {
        double wavelength_start = wl_idx * (wavelength_block_time + kWavelengthSwitchTimeS);
        std::vector<double> wavelengths = {wl_schedule.wavelengths_m(wl_idx)};

        for (uint32_t i = 0; i < chirps_per_wavelength; ++i) {
            auto direction = (chirp_mode == ChirpSchedule::CHIRP_MODE_UP_DOWN && i % 2 == 1)
                ? ChirpPlan::DIRECTION_DOWN : ChirpPlan::DIRECTION_UP;
            chirps.push_back(create_chirp(
                chirp_index,
                wavelength_start + i * chirp_duration,
                chirp_duration,
                bandwidth,
                direction,
                wavelengths));
            ++chirp_index;
        }
    }
    return chirps;
}

std::vector<ChirpPlan> SequencePlanner::generate_simultaneous_chirps(
    uint32_t total_chirps,
    double chirp_duration,
    double bandwidth,
    ChirpSchedule::ChirpMode chirp_mode) const {
    std::vector<ChirpPlan> chirps;
    chirps.reserve(total_chirps);

    const auto& wl_schedule = schedule_.wavelength_schedule();
    std::vector<double> wavelengths;
    for (int i = 0; i < wl_schedule.wavelengths_m_size(); ++i) {
        wavelengths.push_back(wl_schedule.wavelengths_m(i));
    }

    for (uint32_t i = 0; i < total_chirps; ++i) {
        auto direction = (chirp_mode == ChirpSchedule::CHIRP_MODE_UP_DOWN && i % 2 == 1)
            ? ChirpPlan::DIRECTION_DOWN : ChirpPlan::DIRECTION_UP;
        chirps.push_back(create_chirp(
            i,
            i * chirp_duration,
            chirp_duration,
            bandwidth,
            direction,
            wavelengths));
    }
    return chirps;
}

SequencePlan SequencePlanner::plan_burst(uint64_t burst_id, double timestamp_s) const {
    SequencePlan plan;
    plan.set_plan_id(burst_id);
    plan.set_start_timestamp_s(timestamp_s);

    if (!schedule_.has_chirp_schedule()) {
        throw std::invalid_argument("SequenceSchedule missing chirp_schedule");
    }

    const auto& chirp_schedule = schedule_.chirp_schedule();
    const uint32_t total_chirps = chirp_schedule.chirps_per_burst();
    const double chirp_duration = chirp_schedule.chirp_duration_s();
    const double bandwidth = chirp_schedule.chirp_bandwidth_hz();
    const auto chirp_mode = chirp_schedule.mode();

    const auto& wl_schedule = schedule_.wavelength_schedule();
    const bool use_alternating =
        wl_schedule.mode() == WavelengthSchedule::INTERLEAVING_ALTERNATING &&
        wl_schedule.wavelengths_m_size() > 1;

    auto chirp_plans = use_alternating
        ? generate_alternating_chirps(total_chirps, chirp_duration, bandwidth, chirp_mode)
        : generate_simultaneous_chirps(total_chirps, chirp_duration, bandwidth, chirp_mode);
    for (auto& chirp : chirp_plans) {
        *plan.add_chirps() = std::move(chirp);
    }

    return plan;
}

}  // namespace aurora::acquisition
