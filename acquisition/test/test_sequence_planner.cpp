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

/**
 * @file test_sequence_planner.cpp
 * @brief Unit tests for SequencePlanner and timing utilities.
 */

#include <gtest/gtest.h>

#include "acquisition/sequence_planner.h"
#include "acquisition/timing_utils.h"

using namespace aurora::acquisition;

namespace {

constexpr double kBurstPeriod = 100e-6;   // burst period in seconds
constexpr double kChirpDuration = 10e-6;
constexpr double kChirpBandwidth = 1e9;
constexpr uint32_t kChirpsPerBurst = 8;

SequenceSchedule create_basic_schedule() {
    SequenceSchedule schedule;
    schedule.set_burst_period_s(kBurstPeriod);

    auto* chirp = schedule.mutable_chirp_schedule();
    chirp->set_chirp_duration_s(kChirpDuration);
    chirp->set_chirp_bandwidth_hz(kChirpBandwidth);
    chirp->set_chirps_per_burst(kChirpsPerBurst);
    chirp->set_mode(ChirpSchedule::CHIRP_MODE_UP_ONLY);

    auto* wavelength = schedule.mutable_wavelength_schedule();
    wavelength->set_mode(WavelengthSchedule::INTERLEAVING_SIMULTANEOUS);
    wavelength->add_wavelengths_m(1550e-9);

    return schedule;
}

class SequencePlannerTest : public ::testing::Test {
  protected:
    void SetUp() override {
        schedule_ = create_basic_schedule();
    }

    SequenceSchedule schedule_;
};

}  // namespace

TEST_F(SequencePlannerTest, ScheduleAccessors) {
    SequencePlanner planner(schedule_);
    EXPECT_DOUBLE_EQ(planner.get_burst_period_s(), kBurstPeriod);
    EXPECT_DOUBLE_EQ(planner.get_chirp_duration_s(), kChirpDuration);
    EXPECT_EQ(planner.get_chirps_per_burst(), kChirpsPerBurst);
}

TEST_F(SequencePlannerTest, ChirpCount) {
    SequencePlanner planner(schedule_);
    auto plan = planner.plan_burst(0, 0.0);
    EXPECT_EQ(plan.chirps_size(), static_cast<int>(kChirpsPerBurst));
}

TEST_F(SequencePlannerTest, WavelengthInterleavingAlternating) {
    SequenceSchedule schedule = create_basic_schedule();
    auto* wavelength = schedule.mutable_wavelength_schedule();
    wavelength->set_mode(WavelengthSchedule::INTERLEAVING_ALTERNATING);
    wavelength->clear_wavelengths_m();
    wavelength->add_wavelengths_m(905e-9);
    wavelength->add_wavelengths_m(1550e-9);

    SequencePlanner planner(schedule);
    auto plan = planner.plan_burst(0, 0.0);
    ASSERT_EQ(plan.chirps_size(), 8);

    // First half at 905 nm, second half at 1550 nm, one wavelength per chirp
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(plan.chirps(i).wavelength_m_size(), 1);
        EXPECT_DOUBLE_EQ(plan.chirps(i).wavelength_m(0), 905e-9);
    }
    for (int i = 4; i < 8; ++i) {
        ASSERT_EQ(plan.chirps(i).wavelength_m_size(), 1);
        EXPECT_DOUBLE_EQ(plan.chirps(i).wavelength_m(0), 1550e-9);
    }
    // Second block starts after the first block plus the switch time
    EXPECT_GT(plan.chirps(4).chirp_start_s(), plan.chirps(3).chirp_start_s() + kChirpDuration);
}

TEST_F(SequencePlannerTest, WavelengthInterleavingSimultaneous) {
    SequenceSchedule schedule = create_basic_schedule();
    auto* wavelength = schedule.mutable_wavelength_schedule();
    wavelength->set_mode(WavelengthSchedule::INTERLEAVING_SIMULTANEOUS);
    wavelength->clear_wavelengths_m();
    wavelength->add_wavelengths_m(905e-9);
    wavelength->add_wavelengths_m(1550e-9);

    SequencePlanner planner(schedule);
    auto plan = planner.plan_burst(0, 0.0);

    ASSERT_EQ(plan.chirps_size(), 8);
    for (const auto& chirp : plan.chirps()) {
        EXPECT_EQ(chirp.wavelength_m_size(), 2);
    }
}

TEST_F(SequencePlannerTest, ChirpTimingSequential) {
    SequencePlanner planner(schedule_);
    auto plan = planner.plan_burst(0, 0.0);

    for (int i = 0; i < plan.chirps_size(); ++i) {
        EXPECT_DOUBLE_EQ(plan.chirps(i).chirp_start_s(), i * kChirpDuration);
        EXPECT_DOUBLE_EQ(plan.chirps(i).chirp_duration_s(), kChirpDuration);
        EXPECT_DOUBLE_EQ(plan.chirps(i).chirp_bandwidth_hz(), kChirpBandwidth);
        EXPECT_EQ(plan.chirps(i).chirp_index(), static_cast<uint32_t>(i));
    }
}

TEST_F(SequencePlannerTest, BurstMetadata) {
    SequencePlanner planner(schedule_);
    auto plan = planner.plan_burst(42, 1.5);
    EXPECT_EQ(plan.plan_id(), 42u);
    EXPECT_DOUBLE_EQ(plan.start_timestamp_s(), 1.5);
}

TEST_F(SequencePlannerTest, UpDownChirpMode) {
    SequenceSchedule schedule = create_basic_schedule();
    schedule.mutable_chirp_schedule()->set_mode(ChirpSchedule::CHIRP_MODE_UP_DOWN);

    SequencePlanner planner(schedule);
    auto plan = planner.plan_burst(0, 0.0);

    for (int i = 0; i < plan.chirps_size(); ++i) {
        auto expected = (i % 2 == 0) ? ChirpPlan::DIRECTION_UP : ChirpPlan::DIRECTION_DOWN;
        EXPECT_EQ(plan.chirps(i).direction(), expected);
    }
}

TEST_F(SequencePlannerTest, UpOnlyChirpMode) {
    SequencePlanner planner(schedule_);
    auto plan = planner.plan_burst(0, 0.0);
    for (const auto& chirp : plan.chirps()) {
        EXPECT_EQ(chirp.direction(), ChirpPlan::DIRECTION_UP);
    }
}

TEST_F(SequencePlannerTest, ZeroChirpsProducesEmptyPlan) {
    SequenceSchedule schedule = create_basic_schedule();
    schedule.mutable_chirp_schedule()->set_chirps_per_burst(0);
    SequencePlanner planner(schedule);
    EXPECT_EQ(planner.plan_burst(0, 0.0).chirps_size(), 0);
}

TEST_F(SequencePlannerTest, MissingChirpScheduleThrows) {
    SequenceSchedule schedule = create_basic_schedule();
    schedule.clear_chirp_schedule();

    SequencePlanner planner(schedule);
    EXPECT_THROW(planner.plan_burst(0, 0.0), std::invalid_argument);
    EXPECT_THROW(planner.get_chirp_duration_s(), std::invalid_argument);
    EXPECT_THROW(planner.get_chirps_per_burst(), std::invalid_argument);
}

TEST_F(SequencePlannerTest, PlansAreIndependentOfBurstId) {
    SequencePlanner planner(schedule_);
    auto a = planner.plan_burst(0, 0.0);
    auto b = planner.plan_burst(7, 0.7);
    ASSERT_EQ(a.chirps_size(), b.chirps_size());
    for (int i = 0; i < a.chirps_size(); ++i) {
        EXPECT_DOUBLE_EQ(a.chirps(i).chirp_start_s(), b.chirps(i).chirp_start_s());
        EXPECT_EQ(a.chirps(i).direction(), b.chirps(i).direction());
    }
}

TEST(TimingUtilsTest, ValidateTiming) {
    EXPECT_TRUE(validate_timing(8, 10e-6, 100e-6));
    EXPECT_FALSE(validate_timing(12, 10e-6, 100e-6));
    EXPECT_TRUE(validate_timing(8, 10e-6, 100e-6, 2, 1e-6));
    EXPECT_FALSE(validate_timing(8, 12.4e-6, 100e-6, 2, 1e-6));
}
