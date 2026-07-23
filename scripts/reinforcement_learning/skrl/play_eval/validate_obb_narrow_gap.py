"""Four-environment Isaac Sim smoke test for the frozen 0.85 m OBB corridor."""

from __future__ import annotations

from isaaclab.app import AppLauncher


app = AppLauncher(headless=True).app

import math

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 - registers environments
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.terminations.robot_state import (
    wall_collision_termination,
)


TASK = "Isaac-Navigation-Charge-VLP16-Curriculum-WD"
NUM_ENVS = 4
WALL_SLOTS = (0, 6)  # Both are physical 4.0 x 1.0 m wall assets.


def _place_corridor(raw_env) -> None:
    device = raw_env.device
    env_ids = torch.arange(NUM_ENVS, device=device)
    origins = raw_env.scene.env_origins[env_ids]

    # Robot cases: aligned, 2 deg, 3 deg, and true wheel contact at +13 cm.
    yaw = torch.tensor(
        [math.pi / 2, math.pi / 2 + math.radians(2), math.pi / 2 + math.radians(3), math.pi / 2],
        device=device,
    )
    lateral_x = torch.tensor([0.0, 0.0, 0.0, 0.13], device=device)
    robot = raw_env.scene["robot"]
    robot_pose = robot.data.default_root_state[env_ids, :7].clone()
    robot_pose[:, :2] = origins[:, :2]
    robot_pose[:, 0] += lateral_x
    robot_pose[:, 3:7] = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(NUM_ENVS, 6, device=device), env_ids=env_ids)

    # Hide every randomized wall, then place two 4 m walls with a 0.85 m inner gap.
    inner_half_gap = 0.85 / 2
    wall_half_thickness = 1.0 / 2
    wall_x = inner_half_gap + wall_half_thickness
    centers = torch.tensor([[-wall_x, 0.0], [wall_x, 0.0]], device=device)
    wall_yaw = torch.full((NUM_ENVS,), math.pi / 2, device=device)
    wall_quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(wall_yaw), torch.zeros_like(wall_yaw), wall_yaw
    )

    for slot in range(8):
        name = f"wall_internal_{slot}"
        if name not in raw_env.scene.keys():
            continue
        wall = raw_env.scene[name]
        pose = torch.zeros(NUM_ENVS, 7, device=device)
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = -10.0
        pose[:, 3] = 1.0
        if slot in WALL_SLOTS:
            idx = WALL_SLOTS.index(slot)
            pose[:, :2] += centers[idx]
            pose[:, 2] = 1.5
            pose[:, 3:7] = wall_quat
        wall.write_root_pose_to_sim(pose, env_ids=env_ids)

    raw_env._maze_wall_mask[env_ids] = False
    for idx, slot in enumerate(WALL_SLOTS):
        raw_env._maze_wall_mask[env_ids, slot] = True
        raw_env._maze_wall_centers[env_ids, slot] = centers[idx]
        raw_env._maze_wall_sizes[env_ids, slot] = torch.tensor([1.0, 4.0], device=device)

    # Keep obstacle termination quiet so the smoke isolates wall semantics.
    for i in range(50):
        name = f"obstacle_{i}"
        if name not in raw_env.scene.keys():
            continue
        obstacle = raw_env.scene[name]
        pose = obstacle.data.default_root_state[env_ids, :7].clone()
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = -10.0
        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(torch.zeros(NUM_ENVS, 6, device=device), env_ids=env_ids)


def main() -> int:
    env = None
    try:
        env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM_ENVS, use_fabric=True)
        env_cfg.terminations.wall_collision.params["use_obb"] = True
        env_cfg.terminations.obstacle_collision.params["use_obb"] = True
        env = gym.make(TASK, cfg=env_cfg)
        raw_env = env.unwrapped
        print("[OBB-SMOKE] environment constructed", flush=True)
        _place_corridor(raw_env)
        print("[OBB-SMOKE] corridor placed", flush=True)

        robot_cfg = SceneEntityCfg("robot")
        obb_hit = wall_collision_termination(raw_env, robot_cfg, use_obb=True)
        circle_hit = wall_collision_termination(raw_env, robot_cfg, threshold=0.45, use_obb=False)

        # Evaluate before stepping physics. The contact sensor is intentionally
        # not an oracle here: teleported kinematic walls do not reliably produce
        # contact forces in this environment. Env 3 is nevertheless a true
        # physical-footprint overlap (0.60 m body in a 0.85 m gap, offset 0.13 m).
        raw_env.termination_manager.compute()
        managed_obb_hit = raw_env.termination_manager.get_term("wall_collision")

        expected_obb = torch.tensor([False, False, True, True], device=raw_env.device)
        expected_circle = torch.ones(NUM_ENVS, dtype=torch.bool, device=raw_env.device)
        print(f"[OBB-SMOKE] direct_obb={obb_hit.int().tolist()} expected={expected_obb.int().tolist()}")
        print(f"[OBB-SMOKE] managed_obb={managed_obb_hit.int().tolist()}")
        print(f"[OBB-SMOKE] old_circle={circle_hit.int().tolist()} expected={expected_circle.int().tolist()}")

        if not torch.equal(obb_hit, expected_obb):
            raise AssertionError("direct OBB termination did not match the frozen corridor boundary")
        if not torch.equal(managed_obb_hit, expected_obb):
            raise AssertionError("TerminationManager is not using OBB wall collision")
        if not torch.equal(circle_hit, expected_circle):
            raise AssertionError("legacy circle baseline no longer rejects all 0.85 m cases")

        print("[OBB-SMOKE] PASS")
        return 0
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
