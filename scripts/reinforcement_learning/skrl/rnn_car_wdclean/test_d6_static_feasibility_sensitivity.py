"""CPU contracts for the SA4-D6 static-feasibility sensitivity audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import torch

from rnn_car_wdclean import d6_static_feasibility_sensitivity as d6
from rnn_car_wdclean.d4_geometry_selector import (
    GeometrySelectorSpec,
    _point_obb_clearance_grid,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
RUNNER = HERE / "run_sa4_d6_static_sensitivity.py"
FREEZE = REPO / "docs/freeze/sa4_d6_static_feasibility_sensitivity_v2.json"


def _policy_lidar(sensor_range_m: torch.Tensor) -> torch.Tensor:
    return (sensor_range_m - 0.35).clamp(min=0.0) / 20.0


def _context() -> dict:
    return {
        "obstacle_distances_m": torch.tensor([[2.0]]),
        "current_velocity_mps": torch.tensor([0.4]),
        "current_omega_rad_s": torch.tensor([0.0]),
        "pending_command": torch.tensor([[0.4, 0.0]]),
        "policy_lidar_clearance": _policy_lidar(
            torch.full((1, 72), 20.0)
        ),
        "dynamic_positions_body_m": torch.tensor([[[2.0, 0.0]]]),
        "dynamic_velocities_body_mps": torch.tensor([[[0.0, 0.2]]]),
        "dynamic_radii_m": torch.tensor([[0.35]]),
        "dynamic_valid": torch.tensor([[True]]),
        "robot_position_local_m": torch.tensor([[0.0, -2.0]]),
        "robot_yaw_rad": torch.tensor([0.0]),
        "static_positions_local_m": torch.tensor([[[1.25, 0.0]]]),
        "static_radii_m": torch.tensor([[0.35]]),
        "static_obstacle_valid": torch.tensor([[True]]),
        "wall_centers_local_m": torch.tensor([[[-2.5, 0.0], [2.5, 0.0]]]),
        "wall_sizes_m": torch.tensor([[[1.0, 10.0], [1.0, 10.0]]]),
        "wall_valid": torch.tensor([[True, True]]),
        "num_bins": 19,
        "max_linear_velocity": 1.0,
        "reverse_velocity_scale": 0.2,
        "max_linear_accel": 0.5,
        "max_angular_velocity": 1.2,
        "max_angular_accel": 3.0,
    }


def test_protocol_freezes_factor_grid_and_identity_scope():
    protocol = d6.static_sensitivity_protocol()
    assert protocol["schema"] == "sa4_d6_static_feasibility_sensitivity/v2"
    assert protocol["spec"]["horizons_s"] == [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
    assert protocol["spec"]["clearances_m"] == [
        0.0,
        0.01,
        0.02,
        0.05,
        0.08,
        0.10,
        0.15,
    ]
    assert "arena boundary walls" in protocol["source_attribution"]["wall"]
    assert "policy action is never modified" in protocol["scope"]
    assert len(protocol["sha256"]) == 64


def test_protocol_matches_checked_in_freeze():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert d6.static_sensitivity_protocol() == frozen


def test_variant_keys_have_one_frozen_baseline_and_source_ablations():
    keys = d6.variant_keys()
    assert len(keys) == len(set(keys)) == 51
    assert d6.frozen_variant_key() == "lidar_all_h2.4_c0.10"
    assert "lidar_without_wall_assigned_h2.4_c0.10" in keys
    assert "lidar_without_static_assigned_h2.4_c0.10" in keys
    assert "lidar_without_residual_h2.4_c0.10" in keys
    assert "no_static_lidar_dynamic_ceiling" in keys


def test_source_attribution_is_exclusive_and_exhaustive():
    # Wall point, static-circle surface point, and unmatched interior point.
    points = torch.tensor([[[-2.0, 0.0], [0.65, 0.0], [0.0, 0.0]]])
    result = d6.attribute_static_lidar_sources(
        points_body_m=points,
        static_valid=torch.ones(1, 3, dtype=torch.bool),
        robot_position_local_m=torch.zeros(1, 2),
        robot_yaw_rad=torch.zeros(1),
        static_positions_local_m=torch.tensor([[[1.0, 0.0]]]),
        static_radii_m=torch.tensor([[0.35]]),
        static_obstacle_valid=torch.tensor([[True]]),
        wall_centers_local_m=torch.tensor([[[-2.5, 0.0], [2.5, 0.0]]]),
        wall_sizes_m=torch.tensor([[[1.0, 10.0], [1.0, 10.0]]]),
        wall_valid=torch.tensor([[True, True]]),
    )
    assert result["wall"].tolist() == [[True, False, False]]
    assert result["static_obstacle"].tolist() == [[False, True, False]]
    assert result["residual"].tolist() == [[False, False, True]]
    partition = sum(result[name].long() for name in d6.SOURCE_NAMES)
    assert torch.equal(partition, torch.ones_like(partition))


def test_clearance_timeline_preserves_time_axis():
    path = torch.tensor([[[[[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]]]]])
    yaw = torch.zeros(1, 1, 1, 3)
    points = torch.tensor([[[1.0, 0.0]]])
    timeline = d6.point_obb_clearance_timelines(
        path_m=path,
        yaw_rad=yaw,
        points_m=points,
        source_masks={"wall": torch.tensor([[True]])},
    )["wall"]
    assert timeline.shape == (1, 1, 1, 3)
    assert torch.allclose(
        timeline[0, 0, 0], torch.tensor([0.778, 0.378, 0.0]), atol=1.0e-5
    )


def test_frozen_timeline_min_matches_d4_static_clearance_exactly():
    generator = torch.Generator().manual_seed(818)
    path = torch.randn(2, 3, 4, 12, 2, generator=generator) * 0.5
    yaw = torch.randn(2, 3, 4, 12, generator=generator) * 0.4
    points = torch.randn(2, 9, 2, generator=generator) * 2.0
    valid = torch.rand(2, 9, generator=generator) > 0.25
    geometry_spec = GeometrySelectorSpec(action_chunk_size=5)
    timeline = d6.point_obb_clearance_timelines(
        path_m=path,
        yaw_rad=yaw,
        points_m=points,
        source_masks={"all": valid},
        geometry_spec=geometry_spec,
    )["all"]
    frozen = _point_obb_clearance_grid(
        path, yaw, points, valid, geometry_spec
    )
    assert torch.equal(timeline.amin(dim=-1), frozen)


def test_shadow_keeps_actions_and_reconciles_transition():
    shadow = d6.StaticFeasibilitySensitivity()
    action = torch.tensor([[9.0, 9.0]])
    before = action.clone()
    snapshot = shadow.observe(action, _context())
    assert torch.equal(action, before)
    shadow.record_transition(snapshot, torch.tensor([False]))
    report = shadow.report(metadata={"expected_records": 1})
    assert report["runtime"]["environment_frames"] == 1
    assert report["self_check"]["reconciliation_ok"] is True
    assert (
        report["matrix"]["all_evaluated"]["frames"] == 1
    )


def test_factor_matrix_is_monotone_for_static_feasibility():
    shadow = d6.StaticFeasibilitySensitivity()
    snapshot = shadow.observe(torch.tensor([[9.0, 9.0]]), _context())
    shadow.record_transition(snapshot, torch.tensor([False]))
    variants = shadow.report(metadata={"expected_records": 1})["matrix"][
        "all_evaluated"
    ]["variants"]
    for clearance in (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15):
        rates = [
            variants[f"lidar_all_h{horizon:.1f}_c{clearance:.2f}"][
                "static_any_feasible_rate"
            ]
            for horizon in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
        ]
        assert rates == sorted(rates, reverse=True)
    for horizon in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4):
        rates = [
            variants[f"lidar_all_h{horizon:.1f}_c{clearance:.2f}"][
                "static_any_feasible_rate"
            ]
            for clearance in (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15)
        ]
        assert rates == sorted(rates, reverse=True)


def test_play_wires_d6_before_env_step_and_records_after_transition():
    source = PLAY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    observe = source.index("_d6_step_context = _d6_shadow.observe(")
    env_step = source.index("next_obs, reward, terminated, truncated, info = env.step(")
    transition = source.index("_d6_shadow.record_transition(")
    assert observe < env_step < transition
    assert 'if not torch.equal(actions, _d6_policy_before)' in source


def test_d6_privileged_context_is_added_before_d3_capture_returns():
    source = PLAY.read_text(encoding="utf-8")
    function_start = source.index("    def _d3_capture_pre_step(")
    function_end = source.index("\n    # --- 回合統計計數器 ---", function_start)
    body = source[function_start:function_end]
    context = body.index("_d3_context = {")
    combined_walls = body.index("_d6_get_combined_wall_data(raw_env)")
    privileged = body.index('"robot_position_local"')
    returned = body.index("return _d3_context")
    assert context < combined_walls < privileged < returned
    assert '"wall_centers_local": (\n                            _d6_wall_centers' in body


def test_runner_is_fail_closed_and_forbids_training_or_sa5():
    source = RUNNER.read_text(encoding="utf-8")
    assert "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE" in source
    assert "reconciliation_ok" in source
    assert "source_fingerprint" in source
    assert "training_started" in source
    assert "next_stage_started" in source
    assert '"--d6_static_sensitivity"' in source
    assert '"--d6_static_sensitivity_output"' in source
