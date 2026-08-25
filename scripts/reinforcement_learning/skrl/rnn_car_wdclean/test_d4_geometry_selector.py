"""CPU contracts for the delay-aware SA4-D4 geometry selector."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from rnn_car_wdclean import d4_geometry_selector as d4
from rnn_car_wdclean.reward_diagnostics import (
    decode_discrete_drive_action_grid,
)


NUM_BINS = 19
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
RUNNER = HERE / "run_sa4_d4_geometry_suite.py"
ARGMIN_RUNNER = HERE / "run_sa4_d4_argmin_suite.py"
FREEZE = REPO / "docs/freeze/sa4_d4_geometry_selector_v1.json"
ARGMIN_FREEZE = REPO / "docs/freeze/sa4_d4_geometry_selector_argmin_v1.json"


def _policy_lidar_from_sensor_range(sensor_range_m: torch.Tensor) -> torch.Tensor:
    return (sensor_range_m - 0.35).clamp(min=0.0) / 20.0


def _clear_lidar(envs: int = 1) -> torch.Tensor:
    return _policy_lidar_from_sensor_range(torch.full((envs, 72), 20.0))


def _dynamic_crossing(envs: int = 1):
    # Straight motion intersects the pedestrian, while a turning candidate
    # still has more than the frozen 0.10 m surface-clearance margin.
    positions = torch.tensor([[[1.8, -1.0]]]).repeat(envs, 1, 1)
    velocities = torch.tensor([[[0.0, 0.4]]]).repeat(envs, 1, 1)
    radii = torch.full((envs, 1), 0.35)
    valid = torch.ones(envs, 1, dtype=torch.bool)
    return positions, velocities, radii, valid


def _grid_inputs(**overrides):
    positions, velocities, radii, valid = _dynamic_crossing()
    values = {
        "policy_actions": torch.tensor([[18.0, 9.0]]),
        "current_velocity_mps": torch.tensor([0.8]),
        "current_omega_rad_s": torch.tensor([0.0]),
        "pending_command": torch.tensor([[0.8, 0.0]]),
        "policy_lidar_clearance": _clear_lidar(),
        "dynamic_positions_body_m": positions,
        "dynamic_velocities_body_mps": velocities,
        "dynamic_radii_m": radii,
        "dynamic_valid": valid,
        "num_bins": NUM_BINS,
        "max_linear_velocity": 1.0,
        "reverse_velocity_scale": 0.2,
        "max_linear_accel": 0.5,
        "max_angular_velocity": 1.2,
        "max_angular_accel": 3.0,
    }
    values.update(overrides)
    return values


def test_protocol_is_content_addressed_and_explicitly_diagnostic():
    protocol = d4.geometry_selector_protocol()
    assert protocol["schema"] == "sa4_d4_geometry_selector/v1"
    assert protocol["spec"]["delay_steps"] == 1
    assert protocol["spec"]["samples"] == 12
    assert protocol["spec"]["num_action_bins"] == 19
    assert protocol["spec"]["max_linear_velocity_mps"] == 1.0
    assert protocol["spec"]["max_angular_velocity_rad_s"] == 1.2
    assert "privileged" in protocol["scope"]
    assert "keep the policy action" in " ".join(protocol["selection"])
    assert protocol["sha256"] == (
        "1729fe3a878b1a1c64f172bb7b3bb0a23f6553385c8b8ca9e982e0cf97170e9d"
    )
    assert protocol == d4.geometry_selector_protocol()


def test_action_dynamics_drift_is_rejected_before_geometry_selection():
    with pytest.raises(ValueError, match="action contract drift"):
        d4.geometry_feasible_action_grid(
            **_grid_inputs(max_angular_velocity=0.785398)
        )


def test_speed_scaled_geometry_spec_scales_four_action_limits_only():
    base = d4.GeometrySelectorSpec()
    scaled = d4.speed_scaled_geometry_spec(0.7, base)
    assert scaled.max_linear_velocity_mps == pytest.approx(0.7)
    assert scaled.max_linear_accel_mps2 == pytest.approx(0.35)
    assert scaled.max_angular_velocity_rad_s == pytest.approx(0.84)
    assert scaled.max_angular_accel_rad_s2 == pytest.approx(2.1)
    assert scaled.reverse_velocity_scale == base.reverse_velocity_scale
    assert scaled.horizon_s == base.horizon_s
    d4.validate_action_contract(
        num_bins=19,
        max_linear_velocity=0.7,
        reverse_velocity_scale=0.2,
        max_linear_accel=0.35,
        max_angular_velocity=0.84,
        max_angular_accel=2.1,
        spec=scaled,
    )
    with pytest.raises(ValueError, match="action contract drift"):
        d4.validate_action_contract(
            num_bins=19,
            max_linear_velocity=1.0,
            reverse_velocity_scale=0.2,
            max_linear_accel=0.5,
            max_angular_velocity=1.2,
            max_angular_accel=3.0,
            spec=scaled,
        )


def test_checked_in_protocol_freeze_matches_generated_contract():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen == d4.geometry_selector_protocol()


def test_argmin_protocol_is_frozen_and_forbids_center_fallback():
    protocol = d4.geometry_argmin_selector_protocol()
    frozen = json.loads(ARGMIN_FREEZE.read_text(encoding="utf-8"))

    assert protocol == frozen
    assert protocol["schema"] == "sa4_d4_geometry_selector_argmin/v1"
    assert protocol["mode"] == d4.ARGMIN_MODE
    assert protocol["lidar_angle_contract"]["fallback"] is None
    assert "bitwise identical" in protocol["lidar_angle_contract"]["match"]
    assert protocol["sha256"] == (
        "aca15fbdc2a2a7545dbc3d4b6dfba86549e1fac9e39ae0e4c24549abe6facdcd"
    )


def test_d1_candidate_cannot_change_the_first_path_sample():
    candidate_v = torch.tensor([[[0.0, 1.0]]])
    candidate_w = torch.zeros_like(candidate_v)
    pending = torch.tensor([[0.4, 0.0]])

    path, _, times = d4.delayed_unicycle_paths(
        candidate_v, candidate_w, pending
    )

    assert times[0].item() == pytest.approx(0.2)
    assert path[0, 0, 0, 0, 0].item() == pytest.approx(0.08)
    assert path[0, 0, 1, 0, 0].item() == pytest.approx(0.08)
    assert path[0, 0, 0, 1, 0].item() == pytest.approx(0.08)
    assert path[0, 0, 1, 1, 0].item() == pytest.approx(0.28)


def test_pending_d1_command_uses_previous_decode_and_zeros_reset_rows():
    previous = torch.tensor([[0.6, -0.2], [0.7, 0.3]])
    pending = d4.pending_d1_command(
        [torch.zeros_like(previous), previous],
        torch.tensor([False, True]),
        reference=torch.full_like(previous, 9.0),
    )

    torch.testing.assert_close(
        pending, torch.tensor([[0.6, -0.2], [0.0, 0.0]])
    )


def test_dynamic_lidar_return_is_removed_but_static_return_remains():
    ranges = torch.full((1, 72), 20.0)
    ranges[0, 36] = 2.0  # +x, coincident with the dynamic obstacle
    ranges[0, 54] = 1.5  # +y, independent static/wall return
    points, static_valid, dynamic_return = d4.lidar_static_points(
        ranges,
        dynamic_positions_body_m=torch.tensor([[[2.0, 0.0]]]),
        dynamic_radii_m=torch.tensor([[0.35]]),
        dynamic_valid=torch.tensor([[True]]),
    )

    assert points.shape == (1, 72, 2)
    assert bool(dynamic_return[0, 36])
    assert not bool(static_valid[0, 36])
    assert not bool(dynamic_return[0, 54])
    assert bool(static_valid[0, 54])


def test_static_point_uses_realized_winner_angle_when_supplied():
    ranges = torch.full((1, 72), 20.0)
    ranges[0, 36] = 2.0
    actual_angles = d4.beam_angles(
        72, device=ranges.device, dtype=ranges.dtype
    )[None, :].clone()
    actual_angles[0, 36] = 0.04
    points, static_valid, _ = d4.lidar_static_points(
        ranges,
        dynamic_positions_body_m=torch.zeros(1, 1, 2),
        dynamic_radii_m=torch.full((1, 1), 0.35),
        dynamic_valid=torch.zeros(1, 1, dtype=torch.bool),
        beam_angles_rad=actual_angles,
    )

    torch.testing.assert_close(
        points[0, 36],
        torch.tensor([2.0 * torch.cos(torch.tensor(0.04)),
                      2.0 * torch.sin(torch.tensor(0.04))]),
    )
    assert bool(static_valid[0, 36])
    assert points[0, 36, 1].item() != pytest.approx(0.0)


def test_realized_angle_shape_and_range_are_fail_closed():
    ranges = torch.full((1, 72), 20.0)
    common = {
        "dynamic_positions_body_m": torch.zeros(1, 1, 2),
        "dynamic_radii_m": torch.full((1, 1), 0.35),
        "dynamic_valid": torch.zeros(1, 1, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="shape"):
        d4.lidar_static_points(
            ranges, beam_angles_rad=torch.zeros(1, 71), **common
        )
    bad = torch.zeros_like(ranges)
    bad[0, 0] = torch.pi + 0.01
    with pytest.raises(ValueError, match=r"\[-pi, pi\]"):
        d4.lidar_static_points(ranges, beam_angles_rad=bad, **common)


def test_policy_action_is_preserved_when_jointly_feasible():
    positions = torch.zeros(1, 1, 2)
    result = d4.geometry_feasible_action_grid(
        **_grid_inputs(
            dynamic_positions_body_m=positions,
            dynamic_velocities_body_mps=torch.zeros_like(positions),
            dynamic_valid=torch.zeros(1, 1, dtype=torch.bool),
        )
    )

    assert bool(result["policy_feasible"][0])
    assert result["actions"][0].tolist() == [18, 9]


def test_crossing_rejects_straight_policy_and_selects_dynamic_safe_action():
    result = d4.geometry_feasible_action_grid(**_grid_inputs())

    assert not bool(result["policy_feasible"][0])
    assert bool(result["any_feasible"][0])
    assert result["actions"][0].tolist() != [18, 9]
    assert result["selected_dynamic_clearance_m"][0] >= 0.10
    assert result["selected_static_clearance_m"][0] >= 0.10


def test_static_lidar_geometry_can_veto_the_dynamic_only_choice():
    clear = d4.geometry_feasible_action_grid(**_grid_inputs())
    first_action = clear["actions"][0]

    linear, angular = decode_discrete_drive_action_grid(
        torch.tensor([0.8]),
        torch.tensor([0.0]),
        num_bins=19,
        dt=0.2,
        max_linear_velocity=1.0,
        reverse_velocity_scale=0.2,
        max_linear_accel=0.5,
        max_angular_velocity=1.2,
        max_angular_accel=3.0,
    )
    path, _, _ = d4.delayed_unicycle_paths(
        linear, angular, torch.tensor([[0.8, 0.0]])
    )
    point = path[
        0, int(first_action[0]), int(first_action[1]), 5
    ]
    angle = torch.atan2(point[1], point[0])
    beam = int(round((float(angle) * 180.0 / torch.pi + 180.0) / 5.0)) % 72
    distance = float(point.norm().item())
    ranges = torch.full((1, 72), 20.0)
    ranges[0, beam] = max(distance, 0.41)

    blocked = d4.geometry_feasible_action_grid(
        **_grid_inputs(
            policy_lidar_clearance=_policy_lidar_from_sensor_range(ranges)
        )
    )

    old_static = blocked["static_clearance_grid_m"][
        0, int(first_action[0]), int(first_action[1])
    ]
    assert old_static < 0.10
    if bool(blocked["any_feasible"][0]):
        assert blocked["actions"][0].tolist() != first_action.tolist()
        assert blocked["selected_dynamic_clearance_m"][0] >= 0.10
        assert blocked["selected_static_clearance_m"][0] >= 0.10


def test_no_jointly_feasible_action_fails_closed_to_policy():
    near_ring = torch.full((1, 72), 0.41)
    positions = torch.zeros(1, 1, 2)
    result = d4.geometry_feasible_action_grid(
        **_grid_inputs(
            policy_lidar_clearance=_policy_lidar_from_sensor_range(near_ring),
            dynamic_positions_body_m=positions,
            dynamic_velocities_body_mps=torch.zeros_like(positions),
            dynamic_valid=torch.zeros(1, 1, dtype=torch.bool),
        )
    )

    assert not bool(result["any_feasible"][0])
    assert result["actions"][0].tolist() == [18, 9]


def _selector_context(**overrides):
    positions, velocities, radii, valid = _dynamic_crossing()
    delta = positions
    distances = delta.norm(dim=-1)
    robot_velocity = torch.tensor([[[0.8, 0.0]]])
    radial = delta / distances[..., None]
    closing = -((velocities - robot_velocity) * radial).sum(dim=-1)
    values = {
        "obstacle_distances_m": distances,
        "relative_closing_speeds_mps": closing,
        "current_velocity_mps": torch.tensor([0.8]),
        "current_omega_rad_s": torch.tensor([0.0]),
        "pending_command": torch.tensor([[0.8, 0.0]]),
        "policy_lidar_clearance": _clear_lidar(),
        "dynamic_positions_body_m": positions,
        "dynamic_velocities_body_mps": velocities,
        "dynamic_radii_m": radii,
        "dynamic_valid": valid,
        "num_bins": 19,
        "max_linear_velocity": 1.0,
        "reverse_velocity_scale": 0.2,
        "max_linear_accel": 0.5,
        "max_angular_velocity": 1.2,
        "max_angular_accel": 3.0,
    }
    values.update(overrides)
    return values


def test_two_frame_trigger_and_inactive_path_preserve_policy_object():
    selector = d4.build_geometry_selector()
    policy = torch.tensor([[18.0, 9.0]])

    first = selector(policy, _selector_context())
    second = selector(policy, _selector_context())

    assert first is policy
    assert torch.equal(first, policy)
    assert second[0].tolist() != policy[0].tolist()
    report = selector.report()
    assert report["trigger_activations"] == 1
    assert report["override_environment_frames"] == 1
    assert report["no_feasible_active_frames"] == 0


def test_reset_clears_active_selector_state():
    selector = d4.build_geometry_selector()
    policy = torch.tensor([[18.0, 9.0]])
    selector(policy, _selector_context())
    selector(policy, _selector_context())
    selector.reset(torch.tensor([True]))
    far = _selector_context(
        obstacle_distances_m=torch.tensor([[5.0]]),
        relative_closing_speeds_mps=torch.tensor([[0.0]]),
    )

    after = selector(policy, far)

    assert after is policy
    assert selector.report()["active_at_end"] == 0


def test_argmin_selector_requires_angles_and_reconciles_paired_shadow():
    selector = d4.build_argmin_geometry_selector()
    policy = torch.tensor([[18.0, 9.0]])
    context = _selector_context()
    selector(policy, context)
    with pytest.raises(ValueError, match="lidar_beam_angles_rad"):
        selector(policy, context)

    selector = d4.build_argmin_geometry_selector()
    angles = d4.beam_angles(
        72, device=policy.device, dtype=policy.dtype
    )[None, :]
    context = _selector_context(lidar_beam_angles_rad=angles)
    selector(policy, context)
    selector(policy, context)
    report = selector.report()

    assert report["mode"] == d4.ARGMIN_MODE
    assert report["lidar_angle_source"] == "winning_raw_ray"
    assert report["paired_center_shadow_active_frames"] == 1
    assert (
        report["both_jointly_feasible_active_frames"]
        + report["argmin_only_feasible_active_frames"]
        + report["center_only_feasible_active_frames"]
        + report["both_no_feasible_active_frames"]
        == 1
    )
    assert (
        report["jointly_feasible_active_frames"]
        == report["both_jointly_feasible_active_frames"]
        + report["argmin_only_feasible_active_frames"]
    )


def test_play_wires_geometry_context_before_decode_and_env_step():
    source = PLAY.read_text(encoding="utf-8")
    preflight = source.index("validate_d4_action_contract(")
    rollout_loop = source.index("while simulation_app.is_running()")
    capture = source.index("_d3_capture_pre_step(actions, obs_tensor)")
    hook = source.index("_d3_effective_actions = _d3_shield(", capture)
    step = source.index("env.step(actions.float())", hook)
    assert preflight < rollout_loop < capture < hook < step
    for key in (
        '"pending_command"',
        '"policy_lidar_clearance"',
        '"dynamic_positions_body_m"',
        '"dynamic_velocities_body_mps"',
        '"dynamic_radii_m"',
        '"dynamic_valid"',
    ):
        assert source.index(key, hook) < step
    assert source.index("pending_d1_command(") < capture
    assert '"geometry_feasible"' in source
    assert '"shield_protocol": _d3_protocol' in source


def test_play_wires_bitwise_winner_angles_before_argmin_selector():
    source = PLAY.read_text(encoding="utf-8")
    env_create = source.index("env = gym.make(")
    trace_enable = source.index('os.environ["CHARGE_D7_LIDAR_TRACE"] = "1"')
    capture = source.index("_d3_capture_pre_step(actions, obs_tensor)")
    trace_match = source.index("select_matching_lidar_trace(", capture)
    angle_publish = source.index(
        '_d3_step_context["lidar_beam_angles_rad"] = _d4_winner_angles',
        trace_match,
    )
    hook = source.index("_d3_effective_actions = _d3_shield(", angle_publish)
    step = source.index("env.step(actions.float())", hook)

    assert trace_enable < env_create
    assert capture < trace_match < angle_publish < hook < step
    assert '"geometry_feasible_argmin"' in source
    assert '"lidar_beam_angles_rad": _d3_step_context.get(' in source[hook:step]


def test_runner_preregisters_fresh_baseline_and_geometry_before_rollout():
    source = RUNNER.read_text(encoding="utf-8")
    protocol_write = source.index("protocol_path.write_text(")
    rollout = source.index("arms = [", protocol_write)
    assert protocol_write < rollout
    assert 'ARMS = ("baseline", GEOMETRY_MODE)' in source
    assert '"obstacle_CR"' in source
    assert '"wall_CR"' in source
    assert '"TO"' in source
    assert "refuses to overwrite evidence" in source
    for forbidden in (
        "train_rnn_car_wdclip.py",
        "systemctl start",
        "sa5-sim",
    ):
        assert forbidden not in source


def test_argmin_runner_freezes_same_cell_and_paired_recovery_before_rollout():
    source = ARGMIN_RUNNER.read_text(encoding="utf-8")
    protocol_write = source.index("targets[0].write_text(")
    rollout = source.index("arms = [", protocol_write)

    assert protocol_write < rollout
    assert 'ARMS = ("baseline", ARGMIN_MODE)' in source
    assert '"noise_model_change_in_this_suite": None' in source
    assert '"paired_net_feasible_fraction_min"' in source
    assert '"paired_net_feasible_frames_min"' in source
    assert '"training_started": False' in source
    assert '"next_stage_started": False' in source


def test_selector_module_starts_no_processes():
    tree = ast.parse(Path(d4.__file__).read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "subprocess" not in imports
