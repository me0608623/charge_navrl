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
import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402  (registers tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Navigation-Charge-VLP16-Curriculum-WD"  # e2e curriculum task
NUM_ENVS = 8

env_cfg = parse_env_cfg(TASK, num_envs=NUM_ENVS)

# Force corridor injection ON for every reset so the smoke test is deterministic.
env_cfg.events.corridor_crossing.params["fraction"] = 1.0

env = gym.make(TASK, cfg=env_cfg)
env.reset()

u = env.unwrapped

# All 8 envs selected (fraction=1.0) -> slots 0 and 1 must be active as walls.
assert u._maze_wall_mask[:, 0].all(), "SMOKE FAIL: wall slot 0 not active in corridor envs"
assert u._maze_wall_mask[:, 1].all(), "SMOKE FAIL: wall slot 1 not active in corridor envs"

# Verify BehaviorScheduler pedestrian slot is set to horizontal crossing.
sched = u._behavior_scheduler
assert sched is not None, "SMOKE FAIL: _behavior_scheduler is None (obstacle_mode may not be rule_based)"

import sys, os  # noqa: E401, E402
_skrl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _skrl_dir not in sys.path:
    sys.path.insert(0, _skrl_dir)
from obstacle_agent.behavior_config import BEHAVIOR_HORIZONTAL_CROSSING  # noqa: E402

assert (sched.behavior_type[:, 0] == BEHAVIOR_HORIZONTAL_CROSSING).all(), (
    "SMOKE FAIL: ped slot 0 not set to BEHAVIOR_HORIZONTAL_CROSSING"
)

# Pedestrian should move +x after a few steps.
x0 = sched.positions[:, 0, 0].clone()
for _ in range(5):
    env.step(torch.zeros(env.action_space.shape, device=u.device))

assert (sched.positions[:, 0, 0] > x0).all(), (
    "SMOKE FAIL: pedestrian did not move +x (BehaviorScheduler may reset before injector)"
)

print("SMOKE PASS: corridor walls + crossing pedestrian confirmed")
env.close()
app.close()
