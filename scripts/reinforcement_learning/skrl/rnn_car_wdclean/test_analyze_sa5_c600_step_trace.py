"""Synthetic event tests for c600 lateral and random-2D timing analysis."""

from __future__ import annotations

import numpy as np
import pytest

from rnn_car_wdclean.analyze_sa5_c600_step_trace import (
    analyze_lateral_trace,
    analyze_random2d_trace,
)


def _trace(steps: int = 12) -> dict[str, np.ndarray]:
    shape = (steps, 1)
    data = {
        "episode_step": np.arange(steps, dtype=np.int32)[:, None],
        "policy_action_indices": np.tile(
            np.array([[[9, 9]]], dtype=np.int8), (steps, 1, 1)
        ),
        "pre_delay_command_mps_rad_s": np.zeros((*shape, 2), np.float32),
        "post_delay_command_mps_rad_s": np.zeros((*shape, 2), np.float32),
        "robot_xy_m": np.zeros((*shape, 2), np.float32),
        "robot_yaw_rad": np.zeros(shape, np.float32),
        "dynamic_positions_m": np.zeros((*shape, 2, 2), np.float32),
        "dynamic_velocities_mps": np.zeros((*shape, 2, 2), np.float32),
        "dynamic_valid": np.ones((*shape, 2), bool),
        "patrol_waypoint_index": np.zeros((*shape, 2), np.int16),
        "patrol_pause_remaining": np.zeros((*shape, 2), np.int16),
        "patrol_target_m": np.zeros((*shape, 2, 2), np.float32),
        "patrol_active": np.ones((*shape, 2), bool),
        "done": np.zeros(shape, bool),
        "termination_cause": np.zeros(shape, np.int8),
    }
    data["dynamic_positions_m"][:, 0, 1, 0] = 2.0
    return data


def test_lateral_crossing_compares_actual_departure_with_vacated_side():
    data = _trace()
    # Slot 0 crosses +x -> -x at t=4, vacating +x. The robot subsequently
    # moves toward +x, so this is a match.
    data["dynamic_positions_m"][:, 0, 0, 0] = np.array(
        [0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    )
    data["dynamic_positions_m"][:, 0, 0, 1] = 1.0
    data["dynamic_velocities_mps"][:, 0, 0, 0] = -0.5
    data["robot_xy_m"][7:, 0, 0] = 0.25
    data["post_delay_command_mps_rad_s"][6:, 0, 0] = 0.3

    report = analyze_lateral_trace(data, dt_s=0.2)
    assert report["crossing_events"] == 1
    assert report["departure_events"] == 1
    assert report["vacated_side_matches"] == 1
    assert report["departure_latency_s"]["p50"] == pytest.approx(0.6)


def test_random2d_waypoint_switch_reports_reaction_after_reversal():
    data = _trace()
    data["patrol_waypoint_index"][6:, 0, 0] = 1
    data["dynamic_velocities_mps"][:6, 0, 0, 0] = 0.4
    data["dynamic_velocities_mps"][6:, 0, 0, 0] = -0.4
    data["post_delay_command_mps_rad_s"][:8, 0, 0] = 0.4
    data["post_delay_command_mps_rad_s"][8:, 0, 0] = 0.0

    report = analyze_random2d_trace(data, dt_s=0.2)
    assert report["waypoint_switch_events"] == 1
    assert report["direction_reversal_events"] == 1
    assert report["stop_latency_s"]["p50"] == 0.4
    assert report["stopped_at_reversal_fraction"] == 0.0


def test_random2d_reversal_detection_spans_waypoint_pause():
    data = _trace()
    data["patrol_waypoint_index"][6:, 0, 0] = 1
    data["dynamic_velocities_mps"][:5, 0, 0, 0] = 0.4
    data["dynamic_velocities_mps"][5:8, 0, 0, 0] = 0.0
    data["dynamic_velocities_mps"][8:, 0, 0, 0] = -0.4
    report = analyze_random2d_trace(data, dt_s=0.2)
    assert report["waypoint_switch_events"] == 1
    assert report["direction_reversal_events"] == 1
