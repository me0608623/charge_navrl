"""Smoke test: corridor-crossing reset injector.

Launches a tiny headless env with corridor injection forced ON (fraction=1.0)
and asserts that the selected envs actually get 2 corridor walls + a crossing
pedestrian.

Run ONLY when the GPU is not saturated by training (check nvidia-smi first):

    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

Expected output: SMOKE PASS: corridor walls + crossing pedestrian confirmed

Usage:
    cd /home/aa/IsaacLab
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p \\
        scripts/reinforcement_learning/skrl/rnn_car_wdclean/smoke_corridor_crossing.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for corridor_crossing injector")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import torch  # noqa: E402

_skrl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _skrl_dir not in sys.path:
    sys.path.insert(0, _skrl_dir)

import isaaclab_tasks  # noqa: F401, E402  (registers tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Navigation-Charge-VLP16-Curriculum-WD"  # e2e curriculum task
NUM_ENVS = 8

env_cfg = parse_env_cfg(TASK, num_envs=NUM_ENVS)

# Use the production curriculum/stage. Without these explicit params the task's
# legacy default curriculum is incomplete and fails before reset events run.
curriculum_term = env_cfg.curriculum.goal_obstacle_curriculum
curriculum_term.params["curriculum_version"] = "warp_drive_e2e_final20_v1"
curriculum_term.params["initial_stage"] = 4
curriculum_term.params["fixed_stage"] = True

env = gym.make(TASK, cfg=env_cfg)
env.reset()

u = env.unwrapped

# Apply the production injector directly to all envs after the normal reset
# ordering has completed. The curriculum's real fraction remains 0.12; forcing
# 1.0 here only makes this small smoke deterministic.
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.events.corridor_crossing import (  # noqa: E402
    setup_corridor_crossing,
)

setup_corridor_crossing(
    u,
    torch.arange(NUM_ENVS, device=u.device),
    fraction=1.0,
    half_width=2.0,
)

# All 8 envs selected (fraction=1.0) -> slots 0 and 1 must be active as walls.
assert u._maze_wall_mask[:, 0].all(), "SMOKE FAIL: wall slot 0 not active in corridor envs"
assert u._maze_wall_mask[:, 1].all(), "SMOKE FAIL: wall slot 1 not active in corridor envs"

# Verify the pedestrian uses the one-shot path-crossing state machine.
sched = u._behavior_scheduler
assert sched is not None, "SMOKE FAIL: _behavior_scheduler is None (obstacle_mode may not be rule_based)"

from obstacle_agent.behavior_config import BEHAVIOR_PATH_CROSSING  # noqa: E402

assert (sched.behavior_type[:, 0] == BEHAVIOR_PATH_CROSSING).all(), (
    "SMOKE FAIL: ped slot 0 not set to BEHAVIOR_PATH_CROSSING"
)

# Check the complete production path. Calling sched.step() directly is not a
# valid smoke test: it bypasses collision termination and auto-reset, which can
# silently replace an injected pedestrian before its first movement step.
robot_local = u.scene["robot"].data.root_pos_w[:, :2] - u.scene.env_origins[:, :2]
other_active = sched.behavior_type[:, 1:] != 0
other_dist = torch.linalg.vector_norm(sched.positions[:, 1:] - robot_local[:, None, :], dim=-1)
other_dist = torch.where(other_active, other_dist, torch.full_like(other_dist, float("inf")))
min_other_clearance = other_dist.amin(dim=1)

x0 = sched.positions[:, 0, 0].clone()
alive = torch.ones(NUM_ENVS, dtype=torch.bool, device=u.device)
x_history = [x0.clone()]
actions = torch.zeros(NUM_ENVS, 2, device=u.device)
for _ in range(20):
    _, _, terminated, truncated, _ = env.step(actions)
    alive &= ~(terminated | truncated)
    x_history.append(sched.positions[:, 0, 0].clone())

x_history = torch.stack(x_history)
print(
    "[SMOKE-CORRIDOR] "
    f"survived={int(alive.sum())}/{NUM_ENVS} "
    f"min_other_clearance={min_other_clearance.min().item():.3f}m "
    f"x0={x_history[0].tolist()} x5={x_history[5].tolist()} x20={x_history[-1].tolist()}",
    flush=True,
)

assert alive.all(), (
    "SMOKE FAIL: an injected corridor env terminated/reset before the pedestrian completed; "
    "check robot clearance from untouched obstacle slots"
)
assert (x_history[5] > x0).all(), "SMOKE FAIL: pedestrian did not move +x through env.step()"
assert sched.pc_done[:, 0].all(), "SMOKE FAIL: corridor pedestrian never completed its crossing"
assert (sched.positions[:, 0, 0] > 0.9).all(), "SMOKE FAIL: pedestrian did not cross robot path"
assert (sched.velocities[:, 0].norm(dim=-1) == 0.0).all(), (
    "SMOKE FAIL: completed pedestrian retained a false scheduler velocity"
)

print("SMOKE PASS: production env.step corridor crossing + survival confirmed")
env.close()
app.close()
