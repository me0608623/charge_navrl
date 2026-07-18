"""Unit contract tests for swept-arc LiDAR distance semantics."""

from pathlib import Path
import sys

import torch


SKRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKRL_ROOT))

from rnn_car_wdclean import swept_arc  # noqa: E402


BODY_RADIUS = 0.35
MAX_RANGE = 20.0


def _policy_obs(sensor_range_m: torch.Tensor) -> torch.Tensor:
    return (sensor_range_m - BODY_RADIUS).clamp(min=0.0, max=MAX_RANGE) / MAX_RANGE


def test_policy_lidar_round_trip_including_r_min_and_no_return():
    sensor_range = torch.tensor([[0.5, 0.85, 2.0, MAX_RANGE]])
    recovered = swept_arc.policy_lidar_to_sensor_range(
        _policy_obs(sensor_range), max_range=MAX_RANGE, body_radius=BODY_RADIUS
    )
    torch.testing.assert_close(recovered, sensor_range)


def test_arc_clearance_subtracts_body_radius_once():
    sensor_range = torch.full((1, 72), MAX_RANGE)
    sensor_range[0, 36] = 0.85
    clearance = swept_arc.arc_clearance(
        sensor_range,
        v=torch.zeros(1),
        omega=torch.zeros(1),
        body_radius=BODY_RADIUS,
        max_range=MAX_RANGE,
    )
    torch.testing.assert_close(clearance, torch.tensor([0.5]))


def test_no_return_is_not_treated_as_obstacle():
    policy_obs = _policy_obs(torch.full((1, 72), MAX_RANGE))
    sensor_range = swept_arc.policy_lidar_to_sensor_range(
        policy_obs, max_range=MAX_RANGE, body_radius=BODY_RADIUS
    )
    clearance = swept_arc.arc_clearance(
        sensor_range,
        v=torch.tensor([0.75]),
        omega=torch.zeros(1),
        body_radius=BODY_RADIUS,
        max_range=MAX_RANGE,
    )
    assert torch.isinf(clearance).all()


def test_c_safe_is_body_edge_clearance():
    c_arc = torch.tensor([0.51, 0.50, 0.49])
    reward = swept_arc.r_arc_from_clearance(c_arc, c_safe=0.5)
    assert reward[0] == 0
    assert reward[1] == 0
    assert reward[2] < 0
