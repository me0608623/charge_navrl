from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from d7_lidar_residual_origin import (
    LidarResidualOriginAudit,
    ResidualOriginSpec,
    _validate_trace_contract,
    residual_origin_protocol,
    select_matching_lidar_trace,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OBS_FUNCTIONS = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py"
)
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
RUNNER = HERE / "run_sa4_d7_lidar_residual_origin.py"
FREEZE = REPO / "docs/freeze/sa4_d7_lidar_residual_origin_v2.json"


def _noise_contract() -> dict:
    return {
        "displacement_std_soft": 0.008672,
        "hole_rate": 0.194859,
        "distractor_rate": 0.002515,
        "per_ring_bias": True,
        "human_dynamic_dropout": False,
        "block_dropout_prob": 0.0,
        "r_robot": 0.35,
        "r_max": 20.0,
        "num_bins": 72,
        "num_rays": 5760,
    }


def _synthetic_trace() -> dict:
    ranges = torch.full((1, 5, 72), 20.0)
    ranges[0, 4, 36] = 1.0
    normalized = torch.clamp(ranges[:, -1] - 0.35, min=0.0, max=20.0) / 20.0
    angles = (torch.arange(72).float() * 5.0 - 180.0) * torch.pi / 180.0
    distractor = torch.zeros((1, 72), dtype=torch.bool)
    distractor[0, 36] = True
    return {
        "schema": "vlp16_d7_realized_trace/v1",
        "sweep_sensor_ranges_m": ranges,
        "sweep_normalized": normalized,
        "winner_ray_index": torch.zeros((1, 72), dtype=torch.long),
        "winner_valid": torch.ones((1, 72), dtype=torch.bool),
        "winner_raw_valid": torch.zeros((1, 72), dtype=torch.bool),
        "winner_raw_range_m": torch.full((1, 72), 20.0),
        "winner_bias_range_m": torch.full((1, 72), 20.0),
        "winner_sigma_range_m": torch.full((1, 72), 20.0),
        "winner_dropout_range_m": torch.full((1, 72), 20.0),
        "winner_final_range_m": ranges[:, -1].clone(),
        "winner_hole": torch.zeros((1, 72), dtype=torch.bool),
        "winner_distractor": distractor,
        "winner_hit_body_xyz_m": torch.zeros((1, 72, 3)),
        "winner_actual_angle_rad": angles[None, :],
        "winner_ring_index": torch.zeros((1, 72), dtype=torch.long),
        "winner_horizontal_index": torch.zeros((1, 72), dtype=torch.long),
        "noise_contract": _noise_contract(),
    }


def _context() -> dict:
    return {
        "dynamic_positions_body_m": torch.tensor([[[100.0, 100.0]]]),
        "dynamic_radii_m": torch.tensor([[0.3]]),
        "dynamic_valid": torch.tensor([[True]]),
        "robot_position_local_m": torch.zeros((1, 2)),
        "robot_yaw_rad": torch.zeros(1),
        "static_positions_local_m": torch.zeros((1, 1, 2)),
        "static_radii_m": torch.tensor([[0.3]]),
        "static_obstacle_valid": torch.tensor([[False]]),
        "wall_centers_local_m": torch.zeros((1, 1, 2)),
        "wall_sizes_m": torch.ones((1, 1, 2)),
        "wall_valid": torch.tensor([[False]]),
    }


def test_protocol_is_baseline_only_and_content_addressed():
    protocol = residual_origin_protocol()
    assert protocol["schema"] == "sa4_d7_lidar_residual_origin/v2"
    assert len(protocol["sha256"]) == 64
    assert "never draw additional random numbers" in protocol["trace_rule"]
    assert "single checkpoint" in protocol["interpretation_limits"][-1]


def test_frozen_protocol_matches_runtime_protocol():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == residual_origin_protocol()


def test_exact_policy_trace_is_selected():
    trace = _synthetic_trace()
    wrong = _synthetic_trace()
    wrong["sweep_normalized"] = wrong["sweep_normalized"].clone()
    wrong["sweep_normalized"][0, 0] += 1.0e-6
    env = SimpleNamespace(_d7_lidar_trace_candidates=[wrong, trace])
    selected, count = select_matching_lidar_trace(
        env, trace["sweep_normalized"]
    )
    assert selected is trace
    assert count == 1


def test_ambiguous_different_trace_fails_closed():
    first = _synthetic_trace()
    second = _synthetic_trace()
    second["winner_distractor"] = second["winner_distractor"].clone()
    second["winner_distractor"][0, 35] = True
    env = SimpleNamespace(_d7_lidar_trace_candidates=[first, second])
    with pytest.raises(RuntimeError, match="ambiguous"):
        select_matching_lidar_trace(env, first["sweep_normalized"])


def test_ambiguous_winner_angle_fails_closed():
    first = _synthetic_trace()
    second = _synthetic_trace()
    second["winner_actual_angle_rad"] = second[
        "winner_actual_angle_rad"
    ].clone()
    second["winner_actual_angle_rad"][0, 35] += 0.01
    env = SimpleNamespace(_d7_lidar_trace_candidates=[first, second])

    with pytest.raises(RuntimeError, match="ambiguous"):
        select_matching_lidar_trace(env, first["sweep_normalized"])


def test_noise_contract_drift_fails_closed():
    trace = _synthetic_trace()
    _validate_trace_contract(trace, ResidualOriginSpec())
    trace["noise_contract"]["distractor_rate"] = 0.0
    with pytest.raises(RuntimeError, match="distractor_rate"):
        _validate_trace_contract(trace, ResidualOriginSpec())


def test_realized_distractor_is_partitioned_and_reconciles():
    trace = _synthetic_trace()
    audit = LidarResidualOriginAudit()
    actions = torch.tensor([[9.0, 9.0]])
    derived = audit.observe(
        actions,
        policy_lidar=trace["sweep_normalized"],
        trace=trace,
        trace_match_count=1,
        context=_context(),
        d6_snapshot={"frozen_static_blocked": torch.tensor([True])},
    )
    audit.record_transition(derived, torch.tensor([False]))
    report = audit.report(metadata={"expected_records": 1})
    all_scope = report["scopes"]["all_evaluated"]
    assert all_scope["final_residual_points"] == 1
    assert all_scope["final_mechanism"]["distractor_winner"] == 1
    assert all_scope["final_creation_stage"]["distractor"] == 1
    assert all_scope["final_winner_raw_hit_source"]["invalid"] == 1
    assert report["self_check"]["reconciliation_ok"] is True
    assert torch.equal(actions, torch.tensor([[9.0, 9.0]]))


def test_observation_trace_is_opt_in_and_snapshots_realized_stages():
    source = OBS_FUNCTIONS.read_text(encoding="utf-8")
    assert 'CHARGE_D7_LIDAR_TRACE' in source
    for name in (
        "d7_raw_ranges",
        "d7_bias_ranges",
        "d7_sigma_ranges",
        "d7_dropout_ranges",
        "d7_final_ranges",
    ):
        assert name in source
    assert "torch.randn" not in source[source.index('if d7_trace_enabled:'):source.index('d7_raw_ranges')]


def test_winner_angle_comes_from_emitted_ray_direction_not_hit_point():
    source = OBS_FUNCTIONS.read_text(encoding="utf-8")

    assert 'getattr(sensor, "_ray_directions_w", None)' in source
    assert "d7_ray_angles = torch.atan2(" in source
    assert "winner_angle = _d7_gather(d7_ray_angles)" in source
    assert "winner_angle = _d7_gather(angles)" not in source


def test_play_enables_trace_before_environment_creation():
    source = PLAY.read_text(encoding="utf-8")
    enable = source.index('os.environ["CHARGE_D7_LIDAR_TRACE"] = "1"')
    make = source.index("env = gym.make")
    assert enable < make
    assert "select_matching_lidar_trace" in source
    assert "D7 shadow modified policy actions" in source


def test_runner_pins_full_noise_and_all_identity_audits():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"vlp16_noise_mode": "full"' in source
    for flag in (
        "--d3_yield_audit",
        "--d6_static_sensitivity",
        "--d7_lidar_residual_origin",
    ):
        assert flag in source
    assert '"policy_actions_modified": False' in source
    assert '"training_started": False' in source
    assert '"next_stage_started": False' in source
    assert "D7 refuses to share a busy GPU" in source
