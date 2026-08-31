import pytest
import torch
from pathlib import Path

from rnn_car_wdclean.privileged_corridor_teacher import (
    CorridorTeacherSpec,
    _d1_unicycle_paths,
    corridor_teacher_action_grid,
    predict_patrol_obstacle_paths,
    select_teacher_rollout_actions,
)


def _teacher_inputs(num_envs: int = 1, num_obstacle_slots: int = 1):
    samples = 10
    return {
        "current_velocity": torch.zeros(num_envs),
        "current_omega": torch.zeros(num_envs),
        "robot_xy_m": torch.zeros(num_envs, 2),
        "robot_yaw_rad": torch.zeros(num_envs),
        "goal_xy_m": torch.tensor([[5.0, 0.0]]).repeat(num_envs, 1),
        "obstacle_paths_m": torch.zeros(
            num_envs, num_obstacle_slots, samples, 2
        ),
        "obstacle_radii_m": torch.full((num_envs, num_obstacle_slots), 0.3),
        "obstacle_valid": torch.zeros(
            num_envs, num_obstacle_slots, dtype=torch.bool
        ),
        "wall_centers_m": torch.zeros(num_envs, 1, 2),
        "wall_sizes_m": torch.ones(num_envs, 1, 2),
        "wall_valid": torch.zeros(num_envs, 1, dtype=torch.bool),
        "num_bins": 19,
        "dt": 0.2,
        "max_linear_velocity": 1.0,
        "reverse_velocity_scale": 0.2,
        "max_linear_accel": 0.5,
        "max_angular_velocity": 1.2,
        "max_angular_accel": 3.0,
    }


def test_empty_corridor_teacher_accelerates_straight_to_goal():
    result = corridor_teacher_action_grid(**_teacher_inputs())

    assert bool(result["any_feasible"][0])
    assert result["actions"][0].tolist() == [18, 9]


def test_d1_candidate_cannot_change_the_first_path_sample():
    linear = torch.tensor([[[0.5]]])
    angular = torch.zeros_like(linear)
    path, yaw = _d1_unicycle_paths(
        linear,
        angular,
        pending_command=torch.tensor([[0.0, 0.0]]),
        robot_xy_m=torch.zeros(1, 2),
        robot_yaw_rad=torch.zeros(1),
        horizon_s=0.4,
        samples=2,
    )

    assert path[0, 0, 0, 0, 0] == pytest.approx(0.0)
    assert path[0, 0, 0, 1, 0] == pytest.approx(0.1)
    torch.testing.assert_close(yaw, torch.zeros_like(yaw))


def test_d1_teacher_executes_pending_command_before_candidate():
    inputs = _teacher_inputs()
    result = corridor_teacher_action_grid(
        **inputs,
        pending_d1_command=torch.tensor([[0.25, 0.0]]),
    )

    # The selected 0.1 m/s candidate runs for 1.8 s after the queued 0.25 m/s
    # command runs for the first 0.2 s: 0.25*0.2 + 0.1*1.8 = 0.23 m.
    assert result["actions"][0].tolist() == [18, 9]
    assert result["endpoint_grid"][0, 18, 9, 0] == pytest.approx(0.23)


def test_pending_d1_command_shape_is_fail_closed():
    with pytest.raises(ValueError, match="pending d1 command"):
        corridor_teacher_action_grid(
            **_teacher_inputs(),
            pending_d1_command=torch.zeros(1, 1, 2),
        )


def test_goal_denominator_default_preserves_explicit_historical_r1():
    implicit = corridor_teacher_action_grid(**_teacher_inputs())
    explicit = corridor_teacher_action_grid(
        **_teacher_inputs(),
        spec=CorridorTeacherSpec(goal_denominator_floor_m=1.0),
    )

    assert CorridorTeacherSpec().goal_denominator_floor_m == 1.0
    for key in ("actions", "selected_cost", "cost_grid"):
        torch.testing.assert_close(implicit[key], explicit[key])


@pytest.mark.parametrize("floor", [0.0, -1.0, float("nan"), float("inf")])
def test_goal_denominator_floor_must_be_finite_and_positive(floor):
    with pytest.raises(ValueError, match="goal denominator floor"):
        corridor_teacher_action_grid(
            **_teacher_inputs(),
            spec=CorridorTeacherSpec(goal_denominator_floor_m=floor),
        )


def test_r10_restores_clearance_in_the_synthetic_near_goal_band():
    inputs = _teacher_inputs()
    inputs["current_velocity"][:] = 1.0
    inputs["goal_xy_m"][:] = torch.tensor([2.0, 0.0])
    inputs["obstacle_paths_m"][0, 0, :, 0] = 2.0
    inputs["obstacle_paths_m"][0, 0, :, 1] = 0.3
    inputs["obstacle_valid"][:] = True

    r1 = corridor_teacher_action_grid(
        **inputs,
        spec=CorridorTeacherSpec(goal_denominator_floor_m=1.0),
    )
    r10 = corridor_teacher_action_grid(
        **inputs,
        spec=CorridorTeacherSpec(goal_denominator_floor_m=10.0),
    )

    def selected_clearance(result):
        linear_index, angular_index = result["actions"][0]
        return result["min_obstacle_clearance_grid"][
            0, linear_index, angular_index
        ]

    assert selected_clearance(r1) == pytest.approx(0.104259, abs=1e-5)
    assert selected_clearance(r10) == pytest.approx(0.381992, abs=1e-5)
    assert selected_clearance(r10) > selected_clearance(r1) + 0.25


def test_patrol_prediction_turns_at_current_waypoint():
    positions = torch.tensor([[[1.0, 0.0]]])
    behavior = torch.tensor([[2]])
    waypoints = torch.tensor([[[[-1.0, 0.0], [1.0, 0.0]]]])
    waypoint_index = torch.tensor([[1]])
    num_waypoints = torch.tensor([[2]])
    speed = torch.tensor([[0.5]])
    pause = torch.tensor([[0]])

    paths, valid = predict_patrol_obstacle_paths(
        positions,
        behavior,
        waypoints,
        waypoint_index,
        num_waypoints,
        speed,
        pause,
        dt=0.2,
        samples=3,
    )

    assert bool(valid[0, 0])
    assert torch.isclose(paths[0, 0, 0, 0], torch.tensor(1.0))
    assert paths[0, 0, 1, 0] < paths[0, 0, 0, 0]
    assert paths[0, 0, 2, 0] < paths[0, 0, 1, 0]


def test_patrol_prediction_can_hold_at_future_waypoint():
    positions = torch.tensor([[[1.0, 0.0]]])
    behavior = torch.tensor([[2]])
    waypoints = torch.tensor([[[[-1.0, 0.0], [1.0, 0.0]]]])

    paths, _ = predict_patrol_obstacle_paths(
        positions,
        behavior,
        waypoints,
        patrol_wp_index=torch.tensor([[1]]),
        patrol_num_waypoints=torch.tensor([[2]]),
        patrol_speed_mps=torch.tensor([[0.5]]),
        patrol_pause_remaining=torch.tensor([[0]]),
        dt=0.2,
        samples=3,
        new_waypoint_pause_steps=5,
    )

    torch.testing.assert_close(
        paths[0, 0, :, 0], torch.ones(3)
    )


def test_mirrored_crossing_produces_mirrored_teacher_turn():
    inputs = _teacher_inputs(num_envs=2)
    inputs["current_velocity"][:] = 0.8
    crossing_y = torch.linspace(-0.8, 0.6, 10)
    inputs["obstacle_paths_m"][0, 0, :, 0] = 2.0
    inputs["obstacle_paths_m"][0, 0, :, 1] = crossing_y
    inputs["obstacle_paths_m"][1, 0, :, 0] = 2.0
    inputs["obstacle_paths_m"][1, 0, :, 1] = -crossing_y
    inputs["obstacle_valid"][:] = True

    result = corridor_teacher_action_grid(
        **inputs,
        spec=CorridorTeacherSpec(action_smoothness_weight=0.01),
    )

    assert bool(result["any_feasible"].all())
    first = result["actions"][0]
    second = result["actions"][1]
    assert first[0] == second[0]
    assert first[1] + second[1] == 18
    assert first[1] != 9


def test_teacher_rejects_straight_collision_with_static_obstacle():
    inputs = _teacher_inputs()
    inputs["current_velocity"][:] = 0.8
    inputs["obstacle_paths_m"][0, 0, :, 0] = 1.8
    inputs["obstacle_valid"][:] = True

    result = corridor_teacher_action_grid(**inputs)
    selected = result["actions"][0]

    assert bool(result["any_feasible"][0])
    assert bool(result["obstacle_collision_grid"][0, 9, 9])
    assert selected[1] != 9 or selected[0] < 9


def test_teacher_shadow_keeps_student_actions():
    policy = torch.tensor([[18.0, 9.0], [9.0, 4.0]])
    teacher = torch.tensor([[4, 12], [3, 15]])
    feasible = torch.tensor([True, False])

    executed = select_teacher_rollout_actions(
        policy,
        teacher,
        feasible,
        override=False,
    )

    torch.testing.assert_close(executed, policy)


def test_teacher_override_changes_only_feasible_actions():
    policy = torch.tensor([[18.0, 9.0], [9.0, 4.0]])
    teacher = torch.tensor([[4, 12], [3, 15]])
    feasible = torch.tensor([True, False])

    executed = select_teacher_rollout_actions(
        policy,
        teacher,
        feasible,
        override=True,
    )

    torch.testing.assert_close(
        executed,
        torch.tensor([[4.0, 12.0], [9.0, 4.0]]),
    )


def test_play_teacher_models_pending_d1_before_env_step():
    play = (
        Path(__file__).resolve().parents[1]
        / "play_eval"
        / "play_rnn_car.py"
    ).read_text(encoding="utf-8")

    pending = play.index(
        "_teacher_pending_d1_command = pending_d1_command("
    )
    teacher = play.index("_teacher_result = corridor_teacher_action_grid(", pending)
    passed = play.index(
        "pending_d1_command=_teacher_pending_d1_command", teacher
    )
    override = play.index("actions = select_teacher_rollout_actions(", passed)
    env_step = play.index("env.step(actions.float())", override)

    assert pending < teacher < passed < override < env_step


def test_obstacle_dynamic_mask_splits_static_and_dynamic_collision():
    """A per-slot dynamic flag must attribute collision to the right group.

    env0 collides only with its static (slot 0) obstacle; env1 collides only
    with its dynamic (slot 1) obstacle. The aggregate ``obstacle_collision_grid``
    cannot tell the two apart -- this is exactly what the diagnostic needs.
    """
    inputs = _teacher_inputs(num_envs=2, num_obstacle_slots=2)
    inputs["current_velocity"][:] = 0.8
    # env0: static (slot 0) blocks the straight-ahead candidate; dynamic
    # (slot 1) is far away and irrelevant.
    inputs["obstacle_paths_m"][0, 0, :, 0] = 1.8
    inputs["obstacle_paths_m"][0, 1, :, 0] = 50.0
    # env1: dynamic (slot 1) blocks; static (slot 0) is far away.
    inputs["obstacle_paths_m"][1, 0, :, 0] = 50.0
    inputs["obstacle_paths_m"][1, 1, :, 0] = 1.8
    inputs["obstacle_valid"][:] = True
    obstacle_dynamic_mask = torch.tensor(
        [[False, True], [False, True]]
    )

    result = corridor_teacher_action_grid(
        **inputs, obstacle_dynamic_mask=obstacle_dynamic_mask
    )

    assert bool(result["static_obstacle_collision_grid"][0, 9, 9])
    assert not bool(result["dynamic_obstacle_collision_grid"][0, 9, 9])
    assert not bool(result["static_obstacle_collision_grid"][1, 9, 9])
    assert bool(result["dynamic_obstacle_collision_grid"][1, 9, 9])
    # Reconciliation: the split must union back to the aggregate exactly,
    # everywhere in the grid, not just at the probed cell.
    union = (
        result["static_obstacle_collision_grid"]
        | result["dynamic_obstacle_collision_grid"]
    )
    assert torch.equal(union, result["obstacle_collision_grid"])


def test_obstacle_dynamic_mask_reconciles_when_both_groups_collide():
    """Both a static and a dynamic slot colliding at the same cell.

    The union identity must still hold when static and dynamic collisions
    overlap, not just when they are mutually exclusive.
    """
    inputs = _teacher_inputs(num_envs=1, num_obstacle_slots=2)
    inputs["current_velocity"][:] = 0.8
    inputs["obstacle_paths_m"][0, 0, :, 0] = 1.8
    inputs["obstacle_paths_m"][0, 1, :, 0] = 1.8
    inputs["obstacle_paths_m"][0, 1, :, 1] = 0.05
    inputs["obstacle_valid"][:] = True
    obstacle_dynamic_mask = torch.tensor([[False, True]])

    result = corridor_teacher_action_grid(
        **inputs, obstacle_dynamic_mask=obstacle_dynamic_mask
    )

    assert bool(result["static_obstacle_collision_grid"][0, 9, 9])
    assert bool(result["dynamic_obstacle_collision_grid"][0, 9, 9])
    union = (
        result["static_obstacle_collision_grid"]
        | result["dynamic_obstacle_collision_grid"]
    )
    assert torch.equal(union, result["obstacle_collision_grid"])


def test_obstacle_dynamic_mask_omitted_keeps_existing_keys_and_values_unchanged():
    inputs = _teacher_inputs(num_envs=1, num_obstacle_slots=2)
    inputs["current_velocity"][:] = 0.8
    inputs["obstacle_paths_m"][0, 0, :, 0] = 1.8
    inputs["obstacle_valid"][:] = True

    baseline = corridor_teacher_action_grid(**inputs)
    with_mask = corridor_teacher_action_grid(
        **inputs,
        obstacle_dynamic_mask=torch.tensor([[False, True]]),
    )

    assert "static_obstacle_collision_grid" not in baseline
    assert "dynamic_obstacle_collision_grid" not in baseline
    assert "static_obstacle_collision_grid" in with_mask
    for key in baseline:
        torch.testing.assert_close(
            with_mask[key], baseline[key], equal_nan=True
        )


def test_obstacle_dynamic_mask_shape_mismatch_raises():
    inputs = _teacher_inputs(num_envs=1, num_obstacle_slots=2)

    with pytest.raises(ValueError, match="obstacle_dynamic_mask"):
        corridor_teacher_action_grid(
            **inputs,
            obstacle_dynamic_mask=torch.zeros(1, 3, dtype=torch.bool),
        )
