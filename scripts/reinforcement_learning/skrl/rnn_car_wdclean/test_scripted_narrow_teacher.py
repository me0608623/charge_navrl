"""Scripted direct-crossing narrow-gap teacher — unit tests.

07-27 verdict branch 4 -> N1: every RL checkpoint in the lineage (and the SA5
narrow teacher itself) crosses the 1.2 m gap only after detouring to
|y|~3.8 m, so imitation needs a teacher that actually goes straight.

The first version of this teacher only handled the fixed Gate5 scene
(barrier x=0, gap y=0, goal at +3, west->east only). The production narrow
replay randomises gap center y in [-1,+1], barrier x in [-0.5,+0.5], and
picks left->right or right->left with equal probability, so the teacher is
now driven entirely by per-env geometry. These tests pin that
generalisation: mirrored direction, translated gap/barrier, and mixed
batches where neighbouring envs disagree on everything.
"""

import math

import pytest
import torch

from rnn_car_wdclean.reward_diagnostics import (
    decode_discrete_drive_action_grid,
)
from rnn_car_wdclean.scripted_narrow_teacher import (
    ScriptedNarrowTeacherSpec,
    narrow_bridge_teacher_geometry,
    scripted_narrow_gap_action_indices,
)


GRID = dict(
    num_bins=19,
    dt=0.2,
    max_linear_velocity=1.0,
    reverse_velocity_scale=1.0,
    max_linear_accel=0.5,
    max_angular_velocity=0.25 * math.pi,
    max_angular_accel=4.0 * math.pi,
)
CENTER = GRID["num_bins"] // 2


def _act(
    xy,
    yaw,
    v=0.0,
    omega=0.0,
    barrier_x=0.0,
    gap_y=0.0,
    goal=(3.0, 0.0),
    spec=ScriptedNarrowTeacherSpec(),
):
    return scripted_narrow_gap_action_indices(
        torch.tensor([xy], dtype=torch.float32),
        torch.tensor([yaw], dtype=torch.float32),
        torch.tensor([v], dtype=torch.float32),
        torch.tensor([omega], dtype=torch.float32),
        barrier_x_m=torch.tensor([barrier_x], dtype=torch.float32),
        gap_center_y_m=torch.tensor([gap_y], dtype=torch.float32),
        goal_xy_m=torch.tensor([goal], dtype=torch.float32),
        spec=spec,
        **GRID,
    )[0]


# ---------------------------------------------------------------------------
# Fixed Gate5 geometry (west -> east, gap on the axis)
# ---------------------------------------------------------------------------


def test_start_pose_drives_straight_at_full_accel():
    action = _act([-3.0, 0.0], 0.0)
    assert action[1].item() == CENTER
    assert action[0].item() == GRID["num_bins"] - 1


def test_lateral_offset_steers_back_toward_gap_center():
    action = _act([-2.0, 1.0], 0.0)
    assert action[1].item() < CENTER


def test_large_heading_error_slows_down():
    fast = _act([-3.0, 0.0], 0.0, v=1.0)
    skew = _act([-3.0, 0.0], math.pi / 2.0, v=1.0)
    assert skew[0].item() < fast[0].item()
    assert skew[1].item() < CENTER


def test_after_crossing_targets_goal():
    action = _act([1.0, 0.0], 0.0, v=1.0)
    assert action[1].item() == CENTER
    assert action[0].item() >= CENTER


def test_backward_facing_robot_turns_hard():
    action = _act([-3.0, 0.0], math.pi)
    assert action[1].item() in (0, GRID["num_bins"] - 1)


# ---------------------------------------------------------------------------
# Production replay randomisation: mirrored direction, translated geometry
# ---------------------------------------------------------------------------


def test_mirrored_direction_drives_straight_west():
    """Goal west of the barrier: facing pi is the aligned heading, not pi/2."""
    action = _act([3.0, 0.0], math.pi, goal=(-3.0, 0.0))
    assert action[1].item() == CENTER
    assert action[0].item() == GRID["num_bins"] - 1


def test_mirrored_direction_lateral_offset_steers_toward_gap():
    """Heading west at +y: the gap is to the robot's left -> positive omega."""
    action = _act([2.0, 1.0], math.pi, goal=(-3.0, 0.0))
    assert action[1].item() > CENTER


def test_mirrored_crossing_switches_to_goal_after_barrier():
    """Travelling west, x below barrier-0.6 counts as crossed."""
    east = _act([1.0, 0.0], math.pi, v=1.0, goal=(-3.0, 0.0))
    west = _act([-1.0, 0.0], math.pi, v=1.0, goal=(-3.0, 0.0))
    # Both are aligned with the -x axis, so neither should steer.
    assert east[1].item() == CENTER
    assert west[1].item() == CENTER


def test_translated_gap_center_steers_to_the_offset_gap():
    """Robot on the axis, gap at y=+0.8: teacher must aim up, not straight."""
    action = _act([-3.0, 0.0], 0.0, gap_y=0.8)
    assert action[1].item() > CENTER


def test_translated_barrier_shifts_the_crossing_test():
    """barrier_x=+0.5 -> x=1.0 is only 0.5 past it, still pre-cross."""
    spec = ScriptedNarrowTeacherSpec()
    pre = _act([1.0, 0.4], 0.0, barrier_x=0.5, gap_y=0.0, spec=spec)
    post = _act([1.2, 0.4], 0.0, barrier_x=0.5, gap_y=0.0, spec=spec)
    # Pre-cross aims at the throat point (1.1, 0.0) -> steer down hard.
    # Post-cross aims at the goal (3.0, 0.0) -> much gentler correction.
    assert pre[1].item() < CENTER
    assert post[1].item() < CENTER
    assert pre[1].item() < post[1].item()


def test_gap_center_beats_goal_y_before_the_barrier():
    """Goal on the axis but gap at +0.8: pre-cross must track the gap."""
    action = _act([-3.0, 0.8], 0.0, gap_y=0.8, goal=(3.0, 0.0))
    assert action[1].item() == CENTER  # already aligned with the gap


def test_mixed_batch_directions_and_offsets_are_independent():
    """Neighbouring envs disagreeing on direction/gap/barrier must not leak."""
    xy = torch.tensor(
        [[-3.0, 0.0], [3.0, 0.0], [-3.0, 0.0], [-2.0, 0.5]],
        dtype=torch.float32,
    )
    yaw = torch.tensor([0.0, math.pi, 0.0, 0.0], dtype=torch.float32)
    zeros = torch.zeros(4)
    barrier = torch.tensor([0.0, 0.0, 0.0, 0.5], dtype=torch.float32)
    gap = torch.tensor([0.0, 0.0, 0.8, -0.5], dtype=torch.float32)
    goal = torch.tensor(
        [[3.0, 0.0], [-3.0, 0.0], [3.0, 0.8], [3.0, -0.5]],
        dtype=torch.float32,
    )
    batch = scripted_narrow_gap_action_indices(
        xy, yaw, zeros, zeros,
        barrier_x_m=barrier, gap_center_y_m=gap, goal_xy_m=goal, **GRID,
    )
    for i in range(4):
        single = scripted_narrow_gap_action_indices(
            xy[i : i + 1], yaw[i : i + 1], zeros[:1], zeros[:1],
            barrier_x_m=barrier[i : i + 1],
            gap_center_y_m=gap[i : i + 1],
            goal_xy_m=goal[i : i + 1],
            **GRID,
        )
        assert torch.equal(batch[i : i + 1], single), f"env {i} leaked"
    # env0 aligned east, env1 aligned west: both straight; env2 aims up.
    assert batch[0, 1].item() == CENTER
    assert batch[1, 1].item() == CENTER
    assert batch[2, 1].item() > CENTER


def test_mirror_symmetry_of_the_whole_scene():
    """Reflecting x and yaw about the barrier mirrors only the omega index."""
    east = _act([-2.0, 0.3], 0.2, v=0.5, gap_y=0.0, goal=(3.0, 0.0))
    west = _act([2.0, 0.3], math.pi - 0.2, v=0.5, gap_y=0.0, goal=(-3.0, 0.0))
    assert east[0].item() == west[0].item()
    # x-mirroring flips the sign of the bearing, so omega mirrors about center.
    assert east[1].item() + west[1].item() == 2 * CENTER


# ---------------------------------------------------------------------------
# Closed-loop kinematic rollout
#
# The single-step tests above cannot catch sign errors that only compound
# over a trajectory (a mirrored run that slowly spirals, say). This rolls
# the teacher forward through the same unicycle model the env integrates,
# on randomised geometry, and asserts the *trajectory* is a direct
# crossing — the property the GPU check will later measure for real.
# ---------------------------------------------------------------------------


GOAL_TOLERANCE_M = 0.5


def _rollout(barrier_x, gap_y, goal_xy, start_xy, start_yaw, steps=120):
    """Integrate the teacher's own actions; return the xy trajectory.

    Stops on goal arrival like the env does — without that the robot keeps
    driving past the goal and the path-length ratio measures the harness,
    not the teacher.
    """
    dt = GRID["dt"]
    xy = torch.tensor([start_xy], dtype=torch.float32)
    yaw = torch.tensor([start_yaw], dtype=torch.float32)
    v = torch.zeros(1)
    omega = torch.zeros(1)
    bx = torch.tensor([barrier_x], dtype=torch.float32)
    gy = torch.tensor([gap_y], dtype=torch.float32)
    goal = torch.tensor([goal_xy], dtype=torch.float32)

    path = [xy.clone()]
    for _ in range(steps):
        action = scripted_narrow_gap_action_indices(
            xy, yaw, v, omega,
            barrier_x_m=bx, gap_center_y_m=gy, goal_xy_m=goal, **GRID,
        )
        lin_grid, ang_grid = decode_discrete_drive_action_grid(
            v, omega,
            num_bins=GRID["num_bins"], dt=dt,
            max_linear_velocity=GRID["max_linear_velocity"],
            reverse_velocity_scale=GRID["reverse_velocity_scale"],
            max_linear_accel=GRID["max_linear_accel"],
            max_angular_velocity=GRID["max_angular_velocity"],
            max_angular_accel=GRID["max_angular_accel"],
        )
        v = lin_grid[:, action[0, 0], 0]
        omega = ang_grid[:, 0, action[0, 1]]
        yaw = yaw + omega * dt
        xy = xy + torch.stack(
            [v * torch.cos(yaw), v * torch.sin(yaw)], dim=1
        ) * dt
        path.append(xy.clone())
        if float(torch.linalg.vector_norm(xy - goal)) <= GOAL_TOLERANCE_M:
            break
    return torch.cat(path, dim=0)


def _assert_direct(path, barrier_x, gap_y, goal_xy, start_xy):
    """Same four criteria the Gate5 / replay evaluators apply."""
    direction = math.copysign(1.0, goal_xy[0] - barrier_x)
    signed = direction * (path[:, 0] - barrier_x)
    crossed_at = (signed >= 0).nonzero()
    assert crossed_at.numel() > 0, "teacher never crossed the barrier"
    first = int(crossed_at[0])

    pre = path[: first + 1]
    lateral = (pre[:, 1] - gap_y).abs().max()
    assert lateral <= 1.0, f"pre-cross lateral excursion {float(lateral):.3f} m"

    steps = torch.linalg.vector_norm(path[1:] - path[:-1], dim=1).sum()
    straight = math.dist(start_xy, goal_xy)
    assert steps / straight <= 1.35, f"path ratio {float(steps / straight):.3f}"

    assert first * GRID["dt"] <= 10.0, f"crossed at {first * GRID['dt']:.1f} s"

    backtrack = (signed[:first] - signed[1 : first + 1]).clamp_min(0.0).sum()
    assert backtrack <= 0.5, f"backtrack {float(backtrack):.3f} m"


def test_closed_loop_direct_crossing_west_to_east():
    path = _rollout(0.0, 0.0, (3.0, 0.0), (-3.0, 0.0), 0.0)
    _assert_direct(path, 0.0, 0.0, (3.0, 0.0), (-3.0, 0.0))


def test_closed_loop_direct_crossing_east_to_west():
    path = _rollout(0.0, 0.0, (-3.0, 0.0), (3.0, 0.0), math.pi)
    _assert_direct(path, 0.0, 0.0, (-3.0, 0.0), (3.0, 0.0))


def test_closed_loop_over_randomised_replay_geometry():
    """gap y in [-1,1], barrier x in [-0.5,0.5], both directions, yaw noise."""
    cases = [
        # (barrier_x, gap_y, direction, start_lateral_offset, yaw_error_deg)
        (0.0, 0.9, +1, 0.10, 4.0),
        (0.5, -0.8, +1, -0.12, -4.0),
        (-0.5, 0.4, -1, 0.08, 3.0),
        (0.3, -1.0, -1, -0.05, -3.5),
        (-0.2, 0.0, +1, 0.00, 0.0),
        (0.45, 1.0, -1, 0.10, 2.0),
    ]
    for barrier_x, gap_y, direction, offset, yaw_err_deg in cases:
        goal = (barrier_x + direction * 3.0, gap_y)
        start = (barrier_x - direction * 3.0, gap_y + offset)
        base_yaw = 0.0 if direction > 0 else math.pi
        path = _rollout(
            barrier_x, gap_y, goal, start,
            base_yaw + math.radians(yaw_err_deg),
        )
        _assert_direct(path, barrier_x, gap_y, goal, start)


def test_closed_loop_over_n1_randomised_goal_extremes():
    """N1 must aim through the gap before turning toward an off-axis goal."""
    cases = [
        # barrier_x, gap_y, direction, goal_distance, goal_offset, start_offset, yaw_deg
        (0.0, 0.0, +1, 2.0, +1.5, +0.10, +4.0),
        (+0.5, -1.0, +1, 4.0, -1.5, -0.12, -4.0),
        (-0.5, +1.0, -1, 2.0, +1.5, +0.08, +3.0),
        (+0.3, -0.8, -1, 4.0, -1.5, -0.05, -3.5),
    ]
    for barrier_x, gap_y, direction, distance, goal_offset, start_offset, yaw_deg in cases:
        goal = (barrier_x + direction * distance, gap_y + goal_offset)
        start = (barrier_x - direction * 3.0, gap_y + start_offset)
        base_yaw = 0.0 if direction > 0 else math.pi
        path = _rollout(
            barrier_x,
            gap_y,
            goal,
            start,
            base_yaw + math.radians(yaw_deg),
        )
        _assert_direct(path, barrier_x, gap_y, goal, start)


def test_closed_loop_recovers_from_a_bad_initial_heading():
    """Spawned facing 60 deg off: it may cost time, but never a detour."""
    path = _rollout(0.0, 0.5, (3.0, 0.5), (-3.0, 0.5), math.radians(60.0))
    direction = 1.0
    signed = direction * path[:, 0]
    first = int((signed >= 0).nonzero()[0])
    lateral = (path[: first + 1, 1] - 0.5).abs().max()
    assert lateral <= 1.0, f"turned into a detour: {float(lateral):.3f} m"


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_batch_shapes_and_validation():
    xy = torch.zeros(4, 2)
    yaw = torch.zeros(4)
    v = torch.zeros(4)
    om = torch.zeros(4)
    geom = dict(
        barrier_x_m=torch.zeros(4),
        gap_center_y_m=torch.zeros(4),
        goal_xy_m=torch.tensor([[3.0, 0.0]]).repeat(4, 1),
    )
    out = scripted_narrow_gap_action_indices(xy, yaw, v, om, **geom, **GRID)
    assert out.shape == (4, 2)
    assert out.dtype == torch.long
    with pytest.raises(ValueError):
        scripted_narrow_gap_action_indices(
            xy[:, :1], yaw, v, om, **geom, **GRID
        )
    with pytest.raises(ValueError):
        scripted_narrow_gap_action_indices(
            xy, yaw[:2], v, om, **geom, **GRID
        )
    with pytest.raises(ValueError):
        bad = dict(geom, barrier_x_m=torch.zeros(2))
        scripted_narrow_gap_action_indices(xy, yaw, v, om, **bad, **GRID)


def test_spec_rejects_nonpositive_gain():
    with pytest.raises(ValueError):
        _act([-3.0, 0.0], 0.0, spec=ScriptedNarrowTeacherSpec(heading_gain=0.0))


def test_zero_length_batch_is_allowed():
    empty = scripted_narrow_gap_action_indices(
        torch.zeros(0, 2), torch.zeros(0), torch.zeros(0), torch.zeros(0),
        barrier_x_m=torch.zeros(0),
        gap_center_y_m=torch.zeros(0),
        goal_xy_m=torch.zeros(0, 2),
        **GRID,
    )
    assert empty.shape == (0, 2)


class _FakeEnv:
    """Minimal stand-in for the per-env narrow-bridge state on the env."""

    def __init__(self):
        self.num_envs = 3
        self.device = "cpu"
        seg = 9.0
        # env0: gap at +0.4 width 1.3 barrier +0.2; env1: inactive; env2: gap -0.6
        gap_center = torch.tensor([0.4, 0.0, -0.6])
        gap_width = torch.tensor([1.3, 1.2, 1.2])
        barrier_x = torch.tensor([0.2, 0.0, -0.3])
        hi = gap_center + 0.5 * gap_width + 0.5 * seg
        lo = gap_center - 0.5 * gap_width - 0.5 * seg
        self._narrow_bridge_wall_centers = torch.stack(
            [
                torch.stack([barrier_x, hi], dim=1),
                torch.stack([barrier_x, lo], dim=1),
            ],
            dim=1,
        )
        self._narrow_bridge_active = torch.tensor([True, False, True])
        origins = torch.tensor([[10.0, 20.0, 0.0], [0.0, 0.0, 0.0], [-5.0, 5.0, 0.0]])
        goal_local = torch.tensor([[3.2, 0.4, 0.0], [3.0, 0.0, 0.0], [-3.3, -0.6, 0.0]])
        self._narrow_bridge_goal_w = origins + goal_local

        class _Scene:
            env_origins = origins

        self.scene = _Scene()


def test_geometry_reader_recovers_barrier_gap_and_goal():
    env = _FakeEnv()
    geom = narrow_bridge_teacher_geometry(env)
    assert torch.equal(geom["active"], torch.tensor([True, False, True]))
    torch.testing.assert_close(
        geom["barrier_x_m"], torch.tensor([0.2, 0.0, -0.3])
    )
    torch.testing.assert_close(
        geom["gap_center_y_m"], torch.tensor([0.4, 0.0, -0.6])
    )
    # Goal must come back in env-local coordinates.
    torch.testing.assert_close(
        geom["goal_xy_m"],
        torch.tensor([[3.2, 0.4], [3.0, 0.0], [-3.3, -0.6]]),
    )


def test_geometry_reader_on_env_without_narrow_state():
    class _Bare:
        num_envs = 2
        device = "cpu"

    geom = narrow_bridge_teacher_geometry(_Bare())
    assert geom["active"].shape == (2,)
    assert not bool(geom["active"].any())
