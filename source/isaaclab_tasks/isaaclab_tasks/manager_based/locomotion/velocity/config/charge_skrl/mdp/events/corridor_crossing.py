"""Reset-mode event: inject a corridor + front-crossing-pedestrian trap.

Runs LAST in the reset event order (after reset_base). For a random `fraction`
of the envs being reset, overrides walls/robot/goal/one pedestrian slot to
build the deployment near-wall herding scenario that the flat SA4 arena never
elicits.

Corridor along world-Y (see corridor_crossing_geometry). Robot at local
(0, -corridor_half_len) facing +y; goal at (0, +corridor_half_len); walls
(slots 0 and 1) at local x = -/+ half_width running along y; pedestrian on a
native horizontal_crossing starting at the left wall (x=-half_width) crossing
+x at a fixed y `crossing_ahead` metres in front of the robot.

Baseline safety: when fraction=0.0 the function returns immediately without
touching any env state — byte-for-byte identical to baseline training.
"""

from __future__ import annotations

import os
import sys

import torch

# Resolve the obstacle_agent package (lives in scripts/…/skrl/, not installed).
# Match the same pattern used by rule_behaviors.py and behavior_scheduler.py.
_skrl_dir = os.path.join(os.path.dirname(__file__), "../../../../../../scripts/reinforcement_learning/skrl")
_skrl_dir = os.path.abspath(_skrl_dir)
if _skrl_dir not in sys.path:
    sys.path.insert(0, _skrl_dir)

from obstacle_agent.behavior_config import BEHAVIOR_HORIZONTAL_CROSSING  # noqa: E402

from . import corridor_crossing_geometry as g

# Native mesh lengths of wall slots 0 and 1 (WALL_SLOT_SPECS in wall_layout.py).
# Slot 0 length=4.0m, slot 1 length=3.0m — match the cuboid sizes at scene init.
_WALL_SLOTS = (0, 1)
_WALL_MESH_LEN = {0: 4.0, 1: 3.0}


def setup_corridor_crossing(
    env,
    env_ids,
    fraction: float = 0.12,
    half_width: float = 2.0,
    corridor_half_len: float = 3.0,
    crossing_ahead: float = 2.4,
    ped_speed: float = 0.65,
    ped_slot: int = 0,
    wall_z: float = 1.5,
) -> None:
    """Inject corridor trap for a random fraction of the envs being reset.

    Args:
        env: The ManagerBasedRLEnv instance.
        env_ids: Indices of environments being reset (tensor or list).
        fraction: Fraction of env_ids that receive corridor injection.
            0.0 = baseline (no injection, instant return).
        half_width: Half-width of the corridor in metres (wall offset from centre).
        corridor_half_len: Half-length of the corridor; robot spawns at -corridor_half_len,
            goal at +corridor_half_len (local Y).
        crossing_ahead: Y offset (local) in front of the robot spawn where the
            pedestrian crosses, i.e. cross_y = -corridor_half_len + crossing_ahead.
        ped_speed: Pedestrian crossing speed in m/s (+x direction).
        ped_slot: BehaviorScheduler obstacle slot index to overwrite.
        wall_z: Z centre of the internal wall meshes in world coordinates.
    """
    # Baseline guard — return immediately so training is byte-for-byte identical.
    if fraction <= 0.0:
        return

    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    rand = torch.rand(env_ids.shape[0], device=env.device)
    sel = g.select_corridor_envs(env_ids, fraction, rand)
    if sel.numel() == 0:
        return

    origins = env.scene.env_origins[sel]  # [K, 3] world

    # -------------------------------------------------------------------------
    # 1. Corridor walls (slots 0 and 1): update local AABB tensors + physical pose
    # -------------------------------------------------------------------------
    wall_x = {0: -half_width, 1: +half_width}
    for slot in _WALL_SLOTS:
        size_x, size_y = g.wall_corridor_aabb(_WALL_MESH_LEN[slot])
        env._maze_wall_centers[sel, slot, 0] = wall_x[slot]
        env._maze_wall_centers[sel, slot, 1] = 0.0
        env._maze_wall_sizes[sel, slot, 0] = size_x
        env._maze_wall_sizes[sel, slot, 1] = size_y
        env._maze_wall_mask[sel, slot] = True
        wall = env.scene[f"wall_internal_{slot}"]
        wall.write_root_pose_to_sim(
            g.wall_corridor_pose(origins, wall_x[slot], wall_z),
            env_ids=sel,
        )

    # Disable all other internal wall slots for corridor envs.
    # Use explicit per-slot indexing to avoid the advanced-index copy trap
    # (env._maze_wall_mask[sel][:, slot] writes to a temporary copy).
    num_slots = env._maze_wall_mask.shape[1]
    other_slots = [s for s in range(num_slots) if s not in _WALL_SLOTS]
    for slot in other_slots:
        env._maze_wall_mask[sel, slot] = False

    # -------------------------------------------------------------------------
    # 2. Robot at corridor mouth, facing +y (yaw=+90 deg)
    # -------------------------------------------------------------------------
    env.scene["robot"].write_root_pose_to_sim(
        g.robot_corridor_pose(origins, -corridor_half_len),
        env_ids=sel,
    )
    env.scene["robot"].write_root_velocity_to_sim(
        torch.zeros(sel.shape[0], 6, device=env.device),
        env_ids=sel,
    )

    # -------------------------------------------------------------------------
    # 3. Goal at corridor end (+y)
    # -------------------------------------------------------------------------
    goal_cmd = env.command_manager.get_command("goal_command")
    goal_cmd.goal_pos_w[sel] = g.goal_corridor_pos(origins, corridor_half_len)

    # -------------------------------------------------------------------------
    # 4. One pedestrian slot: native horizontal_crossing, left → right
    # -------------------------------------------------------------------------
    sched = getattr(env.unwrapped, "_behavior_scheduler", None)
    if sched is not None:
        # Local Y in front of robot: robot is at -corridor_half_len, so add crossing_ahead.
        cross_y = -corridor_half_len + crossing_ahead
        sched.behavior_type[sel, ped_slot] = BEHAVIOR_HORIZONTAL_CROSSING
        sched.positions[sel, ped_slot, 0] = -half_width
        sched.positions[sel, ped_slot, 1] = cross_y
        sched.hc_velocity[sel, ped_slot, 0] = ped_speed
        sched.hc_velocity[sel, ped_slot, 1] = 0.0
        sched.hc_cross_y[sel, ped_slot] = cross_y
        sched.hc_spawn_pos[sel, ped_slot, 0] = -half_width
        sched.hc_spawn_pos[sel, ped_slot, 1] = cross_y
        sched.hc_cooldown[sel, ped_slot] = 0
