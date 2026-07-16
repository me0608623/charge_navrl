# Corridor-Crossing Curriculum Injector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject a "corridor + front-crossing pedestrian" trap scenario into a fraction of SA3–SA5 training envs so the e2e clean-PPO policy actually experiences (and gets collision gradient on) the near-wall herding failure it currently exhibits only at deployment.

**Architecture:** A new reset-mode `EventTerm` (`setup_corridor_crossing`) runs LAST in the reset event order. For a random `fraction` (0.12) of the envs being reset, it coherently overrides four things AFTER default placement: (1) two internal wall slots become corridor walls at local x=±2 running along y, (2) robot pose to the corridor mouth (local (0,−3), facing +y), (3) the active goal to the corridor end (local (0,+3)), (4) one BehaviorScheduler obstacle slot becomes a native horizontal_crossing pedestrian starting at the left wall (x=−2) crossing +x across the robot's path. Because all observations are robot-frame relative, orienting the corridor along world-Y (so the pedestrian crosses along the native world-X axis of `horizontal_crossing`) reuses existing spawn/step machinery without modification. Pure geometry math is extracted into a sim-free helper module for unit testing; the integration is smoke-tested with a short headless Isaac Sim run and validated by the existing D-suite `near_wall_crossing_eval`.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab (ManagerBasedRLEnv), pytest. Conda env `env_isaaclab`.

## Global Constraints

- **K8 rebase (2026-07-16):** Codex's `crossing_observability_probe.py` proved the K4 (4-frame) LiDAR stack CANNOT perceive crossing direction well enough (88.75% < 90% gate); K8 (8-frame) = 96.15%. Therefore the corridor curriculum targets the **K8 stack** (`lidar_frame_stack=8`, `end_to_end_frame_stack=True`) — otherwise the policy cannot see the crossing it is being asked to escape. The injector CODE (Tasks 1–3) is frame-depth-agnostic (it only places scene geometry); only the training config (Task 4) selects K8. Base K8 configs already exist: `scripts/reinforcement_learning/skrl/rnn_car_modular/configs/e2e_sa4_fs8_clean_antispin_control.py` (Codex).
- The e2e task id is **`Isaac-Navigation-Charge-VLP16-Curriculum-WD`** (confirmed from the live K8 configs), `curriculum_version="warp_drive_e2e_final20_v1"`.
- e2e training uses **ManagerBasedRLEnv** via `gym.make(--task)` in `scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py` (NOT DirectMARL `charge_marl_env.py` — ignore that file entirely).
- `obstacle_mode: "rule_based"` → obstacles are controlled by `BehaviorScheduler`, accessible at `env.unwrapped._behavior_scheduler` (may be `None` before init — guard it).
- Corridor is oriented along **world-Y**: robot travels +y, walls at local x=±2, pedestrian crosses along world-X (native `horizontal_crossing` axis). World orientation is invisible to the policy (obs are robot-frame relative).
- Wall mesh sizes are FIXED at spawn (`WALL_SLOT_SPECS` in `mdp/wall_layout.py`): slot 0 = (4.0, 1.0, 3.0), slot 1 = (3.0, 1.0, 3.0). Runtime can only move/rotate/hide walls, not resize the mesh. A 90° Z rotation uses quaternion `(0.7071, 0, 0, 0.7071)` (qw, qx, qy, qz).
- Collision termination reads `env._maze_wall_centers [N,8,2]` / `_maze_wall_sizes [N,8,2]` (env-LOCAL frame) AND requires the physical mesh moved via `wall_internal_{i}.write_root_pose_to_sim(pose7_world, env_ids)` (WORLD frame) — BOTH must be updated.
- Default values (confirmed against `near_wall_crossing_eval.py`): `half_width=2.0`, corridor `fraction=0.12`, applied to stages **SA3_walls_crossing, SA4_spatial_plan, SA5_endurance**, pedestrian `speed ∈ (0.6, 0.7)`, crossing point ~2.4 m ahead of robot.
- Keep a few static obstacles in corridor envs (do NOT clear the other obstacle slots) — density is intentional.
- All new CLI/config knobs default to baseline behavior (`fraction=0.0` → no corridor injection, exactly current behavior).
- Pose tensor layout everywhere: `[x, y, z, qw, qx, qy, qz]`.
- Do NOT run the smoke-test Isaac Sim commands while two training runs already saturate the GPU. Check `nvidia-smi` first; if >2 heavy processes, defer the smoke step and say so.

---

### Task 1: Sim-free corridor geometry helpers

Pure tensor math, no Isaac Sim import — fully unit-testable with pytest.

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/corridor_crossing_geometry.py`
- Test: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/tests/test_corridor_crossing_geometry.py`

**Interfaces:**
- Produces:
  - `YAW90_QUAT: tuple[float, float, float, float]` = `(0.7071067811865476, 0.0, 0.0, 0.7071067811865476)`
  - `select_corridor_envs(env_ids: Tensor, fraction: float, rand: Tensor) -> Tensor` — returns the subset of `env_ids` where `rand < fraction`; `rand` is a `[len(env_ids)]` float tensor supplied by caller (so tests are deterministic).
  - `robot_corridor_pose(origins: Tensor, spawn_y: float) -> Tensor` — `origins` is `[K,3]` world env origins; returns `[K,7]` world pose at `origin + (0, spawn_y, 0)` facing +y (yaw +90°).
  - `goal_corridor_pos(origins: Tensor, goal_y: float) -> Tensor` — returns `[K,3]` world goal at `origin + (0, goal_y, 0)`.
  - `wall_corridor_pose(origins: Tensor, wall_x: float, wall_z: float) -> Tensor` — returns `[K,7]` world pose at `origin + (wall_x, 0, wall_z)` rotated 90° about Z.
  - `wall_corridor_aabb(mesh_length: float) -> tuple[float, float]` — returns `(1.0, mesh_length)`: the local AABB (size_x, size_y) of a 90°-rotated wall whose native mesh is `(mesh_length, 1.0)`.

- [ ] **Step 1: Write the failing test**

```python
# test_corridor_crossing_geometry.py
import math
import torch
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.events import (
    corridor_crossing_geometry as g,
)


def test_select_corridor_envs_uses_rand_threshold():
    env_ids = torch.tensor([10, 11, 12, 13])
    rand = torch.tensor([0.05, 0.5, 0.11, 0.9])
    sel = g.select_corridor_envs(env_ids, fraction=0.12, rand=rand)
    assert sel.tolist() == [10, 12]


def test_select_corridor_envs_zero_fraction_empty():
    env_ids = torch.tensor([1, 2, 3])
    rand = torch.zeros(3)
    sel = g.select_corridor_envs(env_ids, fraction=0.0, rand=rand)
    assert sel.numel() == 0


def test_robot_pose_faces_plus_y():
    origins = torch.zeros(2, 3)
    pose = g.robot_corridor_pose(origins, spawn_y=-3.0)
    assert pose.shape == (2, 7)
    assert torch.allclose(pose[:, :3], torch.tensor([0.0, -3.0, 0.0]).expand(2, 3))
    # yaw +90deg about Z -> qw=qz=cos/sin(45deg), qx=qy=0
    expected = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    assert torch.allclose(pose[:, 3:], expected.expand(2, 4), atol=1e-6)


def test_goal_pos_ahead_in_y():
    origins = torch.tensor([[5.0, 7.0, 0.0]])
    goal = g.goal_corridor_pos(origins, goal_y=3.0)
    assert torch.allclose(goal, torch.tensor([[5.0, 10.0, 0.0]]))


def test_wall_pose_rotated_90():
    origins = torch.zeros(1, 3)
    pose = g.wall_corridor_pose(origins, wall_x=-2.0, wall_z=1.5)
    assert torch.allclose(pose[:, :3], torch.tensor([[-2.0, 0.0, 1.5]]))
    assert torch.allclose(pose[:, 3:], torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]]), atol=1e-6)


def test_wall_aabb_swaps_length_into_y():
    assert g.wall_corridor_aabb(4.0) == (1.0, 4.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aa/IsaacLab && /home/aa/miniconda3/envs/env_isaaclab/bin/python -m pytest source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/tests/test_corridor_crossing_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module ... has no attribute 'select_corridor_envs'`.

- [ ] **Step 3: Write minimal implementation**

```python
# corridor_crossing_geometry.py
"""Pure, sim-free geometry helpers for the corridor-crossing injector.

Corridor is oriented along world-Y: robot travels +y, walls run along y at
local x = +/- half_width, pedestrian crosses along world-X (native
horizontal_crossing axis). No Isaac Sim import here so these stay unit-testable.
"""

from __future__ import annotations

import math

import torch

# Quaternion (qw, qx, qy, qz) for a +90 degree rotation about Z.
YAW90_QUAT: tuple[float, float, float, float] = (
    math.cos(math.pi / 4),
    0.0,
    0.0,
    math.sin(math.pi / 4),
)


def select_corridor_envs(env_ids: torch.Tensor, fraction: float, rand: torch.Tensor) -> torch.Tensor:
    """Return the subset of env_ids selected for corridor injection.

    rand is a [len(env_ids)] float tensor in [0, 1) supplied by the caller so
    that selection is deterministic under test.
    """
    if fraction <= 0.0:
        return env_ids[:0]
    return env_ids[rand < fraction]


def robot_corridor_pose(origins: torch.Tensor, spawn_y: float) -> torch.Tensor:
    """[K,7] world pose at origin + (0, spawn_y, 0), facing +y (yaw +90 deg)."""
    k = origins.shape[0]
    pose = torch.zeros(k, 7, device=origins.device, dtype=origins.dtype)
    pose[:, 0] = origins[:, 0]
    pose[:, 1] = origins[:, 1] + spawn_y
    pose[:, 2] = origins[:, 2]
    qw, qx, qy, qz = YAW90_QUAT
    pose[:, 3] = qw
    pose[:, 4] = qx
    pose[:, 5] = qy
    pose[:, 6] = qz
    return pose


def goal_corridor_pos(origins: torch.Tensor, goal_y: float) -> torch.Tensor:
    """[K,3] world goal at origin + (0, goal_y, 0)."""
    goal = origins.clone()
    goal[:, 1] = origins[:, 1] + goal_y
    goal[:, 2] = 0.0
    return goal


def wall_corridor_pose(origins: torch.Tensor, wall_x: float, wall_z: float) -> torch.Tensor:
    """[K,7] world pose at origin + (wall_x, 0, wall_z), rotated 90 deg about Z."""
    k = origins.shape[0]
    pose = torch.zeros(k, 7, device=origins.device, dtype=origins.dtype)
    pose[:, 0] = origins[:, 0] + wall_x
    pose[:, 1] = origins[:, 1]
    pose[:, 2] = wall_z
    qw, qx, qy, qz = YAW90_QUAT
    pose[:, 3] = qw
    pose[:, 4] = qx
    pose[:, 5] = qy
    pose[:, 6] = qz
    return pose


def wall_corridor_aabb(mesh_length: float) -> tuple[float, float]:
    """Local AABB (size_x, size_y) of a 90-deg-rotated wall.

    A native wall mesh is (length, 1.0) along (x, y). Rotating 90 deg about Z
    swaps them: the long axis now runs along y.
    """
    return (1.0, mesh_length)
```

Also create an empty `tests/__init__.py` next to the test if the events dir lacks a test package: `source/.../mdp/events/tests/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aa/IsaacLab && /home/aa/miniconda3/envs/env_isaaclab/bin/python -m pytest source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/tests/test_corridor_crossing_geometry.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/corridor_crossing_geometry.py source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/tests/
git commit -m "feat(curriculum): sim-free corridor-crossing geometry helpers"
```

---

### Task 2: Corridor-crossing injector event + EventCfg wiring

The reset-mode event that consumes Task 1 helpers and coherently overrides walls, robot, goal, and one pedestrian slot for the selected fraction of envs. Requires a live sim to verify → smoke-tested, not pure-unit-tested.

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/corridor_crossing.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/cfg/charge_env_cfg_vlp16_curriculum.py` (add EventTerm AFTER `reset_base`, around line 331–353)
- Smoke harness: `scripts/reinforcement_learning/skrl/rnn_car_wdclean/smoke_corridor_crossing.py`

**Interfaces:**
- Consumes (from Task 1): `select_corridor_envs`, `robot_corridor_pose`, `goal_corridor_pos`, `wall_corridor_pose`, `wall_corridor_aabb`.
- Consumes (existing, verified):
  - `env.scene["robot"].write_root_pose_to_sim(pose7_world, env_ids)`
  - `env.scene[f"wall_internal_{i}"].write_root_pose_to_sim(pose7_world, env_ids)`
  - `env._maze_wall_centers [N,8,2]` / `env._maze_wall_sizes [N,8,2]` / `env._maze_wall_mask [N,8]` (env-local)
  - `env.command_manager.get_command("goal_command").goal_pos_w [N,3]` (world)
  - `env.unwrapped._behavior_scheduler` with `.behavior_type [E,N]`, `.positions [E,N,2]` (local), `.hc_velocity [E,N,2]`, `.hc_cross_y [E,N]`, `.hc_spawn_pos [E,N,2]`, `.hc_cooldown [E,N]`
  - `from ...obstacle_agent.behavior_config import BEHAVIOR_HORIZONTAL_CROSSING` (import path per existing `rule_behaviors.py:33-34`)
- Produces: `setup_corridor_crossing(env, env_ids, fraction, half_width, corridor_half_len, crossing_ahead, ped_speed, ped_slot, wall_z) -> None`

- [ ] **Step 1: Write the injector**

```python
# corridor_crossing.py
"""Reset-mode event: inject a corridor + front-crossing-pedestrian trap.

Runs LAST in the reset event order. For a random `fraction` of the envs being
reset, overrides walls/robot/goal/one pedestrian slot to build the deployment
near-wall herding scenario the flat SA4 arena never elicits.

Corridor along world-Y (see corridor_crossing_geometry). Robot at local
(0, -corridor_half_len) facing +y; goal at (0, +corridor_half_len); walls
(slots 0 and 1) at local x = -/+ half_width running along y; pedestrian on a
native horizontal_crossing starting at the left wall (x=-half_width) crossing
+x at a fixed y `crossing_ahead` metres in front of the robot.
"""

from __future__ import annotations

import torch

from ...obstacle_agent.behavior_config import BEHAVIOR_HORIZONTAL_CROSSING
from . import corridor_crossing_geometry as g

# Native mesh lengths of wall slots 0 and 1 (WALL_SLOT_SPECS in wall_layout.py).
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

    origins = env.scene.env_origins[sel]  # [K,3] world

    # --- 1. Corridor walls (slots 0,1): local tensors + physical mesh pose ---
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
            g.wall_corridor_pose(origins, wall_x[slot], wall_z), env_ids=sel
        )
    # Disable all other internal wall slots for corridor envs.
    other = [s for s in range(env._maze_wall_mask.shape[1]) if s not in _WALL_SLOTS]
    env._maze_wall_mask[sel][:, other] = False  # note: see Step 3 for correct masked write

    # --- 2. Robot at corridor mouth, facing +y ---
    env.scene["robot"].write_root_pose_to_sim(
        g.robot_corridor_pose(origins, -corridor_half_len), env_ids=sel
    )
    env.scene["robot"].write_root_velocity_to_sim(
        torch.zeros(sel.shape[0], 6, device=env.device), env_ids=sel
    )

    # --- 3. Goal at corridor end ---
    goal_cmd = env.command_manager.get_command("goal_command")
    goal_cmd.goal_pos_w[sel] = g.goal_corridor_pos(origins, corridor_half_len)

    # --- 4. One pedestrian slot: native horizontal_crossing, left->right ---
    sched = getattr(env.unwrapped, "_behavior_scheduler", None)
    if sched is not None:
        cross_y = -corridor_half_len + crossing_ahead  # local y in front of robot
        sched.behavior_type[sel, ped_slot] = BEHAVIOR_HORIZONTAL_CROSSING
        sched.positions[sel, ped_slot, 0] = -half_width
        sched.positions[sel, ped_slot, 1] = cross_y
        sched.hc_velocity[sel, ped_slot, 0] = ped_speed
        sched.hc_velocity[sel, ped_slot, 1] = 0.0
        sched.hc_cross_y[sel, ped_slot] = cross_y
        sched.hc_spawn_pos[sel, ped_slot, 0] = -half_width
        sched.hc_spawn_pos[sel, ped_slot, 1] = cross_y
        sched.hc_cooldown[sel, ped_slot] = 0
```

- [ ] **Step 2: Fix the masked-write correctness bug from Step 1**

`env._maze_wall_mask[sel][:, other] = False` writes to a COPY (advanced indexing) and is a no-op. Replace with an explicit loop over `other` slots:

```python
    # --- Disable all other internal wall slots for corridor envs (correct write) ---
    for slot in other:
        env._maze_wall_mask[sel, slot] = False
```

Replace the buggy line in `corridor_crossing.py` accordingly.

- [ ] **Step 3: Register the EventTerm after reset_base**

In `charge_env_cfg_vlp16_curriculum.py`, inside `class EventCfgVLP16Curriculum`, immediately AFTER the `reset_base` EventTerm (around line 331–352) and BEFORE `domain_randomization`, add:

```python
    corridor_crossing = EventTerm(
        func=corridor_crossing.setup_corridor_crossing,
        mode="reset",
        params={
            "fraction": 0.0,          # 0.0 = baseline (no injection); curriculum overrides per stage
            "half_width": 2.0,
            "corridor_half_len": 3.0,
            "crossing_ahead": 2.4,
            "ped_speed": 0.65,
            "ped_slot": 0,
            "wall_z": 1.5,
        },
    )
```

Add the import at the top of the file alongside the other event imports (match the existing relative-import style, e.g. `from ..mdp.events import corridor_crossing` — verify the exact prefix used by neighboring event imports in this file and match it).

- [ ] **Step 4: Write the smoke harness**

```python
# smoke_corridor_crossing.py
"""Launch a tiny headless env with corridor injection forced ON and assert the
selected envs actually get 2 corridor walls + a crossing pedestrian.

Run ONLY when the GPU is not saturated by training (check nvidia-smi first).
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401  (registers tasks)
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Navigation-Charge-VLP16-Curriculum-WD"  # e2e curriculum task (matches Codex K8 configs)
env_cfg = parse_env_cfg(TASK, num_envs=8)
# Force corridor injection ON for every reset env so the smoke test is deterministic.
env_cfg.events.corridor_crossing.params["fraction"] = 1.0
env = gym.make(TASK, cfg=env_cfg)
env.reset()

u = env.unwrapped
# All 8 envs selected (fraction=1.0) -> slots 0,1 active as walls.
assert u._maze_wall_mask[:, 0].all(), "wall slot 0 not active in corridor envs"
assert u._maze_wall_mask[:, 1].all(), "wall slot 1 not active in corridor envs"
sched = u._behavior_scheduler
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.obstacle_agent.behavior_config import (
    BEHAVIOR_HORIZONTAL_CROSSING,
)
assert (sched.behavior_type[:, 0] == BEHAVIOR_HORIZONTAL_CROSSING).all(), "ped slot not crossing"
# Pedestrian moves +x after a few steps.
x0 = sched.positions[:, 0, 0].clone()
for _ in range(5):
    env.step(torch.zeros(env.action_space.shape, device=u.device))
assert (sched.positions[:, 0, 0] > x0).all(), "pedestrian did not move +x"
print("SMOKE PASS: corridor walls + crossing pedestrian confirmed")
env.close()
app.close()
```

- [ ] **Step 5: Run the smoke test (guarded by GPU check)**

```bash
cd /home/aa/IsaacLab
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
# If GPU is saturated by 2 training runs, DEFER this step and note it. Otherwise:
source /home/aa/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/rnn_car_wdclean/smoke_corridor_crossing.py
```
Expected: `SMOKE PASS: corridor walls + crossing pedestrian confirmed`. If it fails on "pedestrian did not move" or "ped slot not crossing", the `BehaviorScheduler.reset()` runs AFTER this event and overwrites slot 0 — resolve by confirming event order and, if needed, moving the injector to run after the scheduler reset (see Task 2 notes: the scheduler reset is triggered from the training loop / an event; the injector must be strictly later).

- [ ] **Step 6: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/corridor_crossing.py source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/cfg/charge_env_cfg_vlp16_curriculum.py scripts/reinforcement_learning/skrl/rnn_car_wdclean/smoke_corridor_crossing.py
git commit -m "feat(curriculum): corridor-crossing reset injector + EventCfg wiring + smoke test"
```

---

### Task 3: Curriculum config knob for SA3–SA5

Wire a per-stage `corridor_crossing` config into `e2e_final20_v1.py` and consume it in the curriculum's stage-application so the event's `fraction` param is set per stage.

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/e2e_final20_v1.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/goal_obstacle_curriculum.py` (`_apply_stage`, around line 1888–1949 where other event params are written)
- Test: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/tests/test_e2e_final20_corridor.py`

**Interfaces:**
- Consumes: existing `_build_stages()` structure in `e2e_final20_v1.py`; the `CONFIG["stages"]` list of flattened phase dicts.
- Produces: each SA3/SA4/SA5 flattened stage dict contains key `corridor_crossing_fraction: 0.12`; SA1/SA2/SA6/SA7/SA8 contain `corridor_crossing_fraction: 0.0`.

- [ ] **Step 1: Write the failing test**

```python
# test_e2e_final20_corridor.py
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.phases import (
    e2e_final20_v1 as m,
)


def _stage(name):
    return next(s for s in m.CONFIG["stages"] if s["name"] == name)


def test_corridor_fraction_on_for_sa3_sa5():
    for name in ("SA3_walls_crossing", "SA4_spatial_plan", "SA5_endurance"):
        assert _stage(name)["corridor_crossing_fraction"] == 0.12


def test_corridor_fraction_off_elsewhere():
    for name in ("SA1_nav_bootstrap", "SA2_nav_static", "SA6_dense_avoid",
                 "SA7_high_pressure", "SA8_final"):
        assert _stage(name)["corridor_crossing_fraction"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aa/IsaacLab && /home/aa/miniconda3/envs/env_isaaclab/bin/python -m pytest source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/tests/test_e2e_final20_corridor.py -v`
Expected: FAIL with `KeyError: 'corridor_crossing_fraction'`.

- [ ] **Step 3: Add the fraction into `_build_stages()`**

In `e2e_final20_v1.py`, add a module-level map and set the key inside the `for stage in stages:` loop of `_build_stages()`:

```python
# Corridor-crossing injection fraction per stage (deployment near-wall trap).
# ON for the stages that first introduce crossing pedestrians (SA3-5).
_CORRIDOR_FRACTION = {
    "SA1_nav_bootstrap": 0.0,
    "SA2_nav_static": 0.0,
    "SA3_walls_crossing": 0.12,
    "SA4_spatial_plan": 0.12,
    "SA5_endurance": 0.12,
    "SA6_dense_avoid": 0.0,
    "SA7_high_pressure": 0.0,
    "SA8_final": 0.0,
}
```

Then inside `_build_stages()` where `name = stage["name"]` is already available:

```python
        stage["scene"]["corridor_crossing_fraction"] = _CORRIDOR_FRACTION[name]
```

Confirm `_flatten_phase` promotes `scene` keys to the top level of each flattened stage dict (it already does for `static_obstacles` etc.). If `_flatten_phase` only copies a whitelist of keys, add `corridor_crossing_fraction` to that whitelist.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aa/IsaacLab && /home/aa/miniconda3/envs/env_isaaclab/bin/python -m pytest source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/tests/test_e2e_final20_corridor.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Consume the fraction in `_apply_stage`**

In `goal_obstacle_curriculum.py._apply_stage`, near where `randomize_wall_positions` / `behavior_mix` event params are written (around line 1888–1949), add (guard for envs/configs without the corridor event so non-e2e curricula are unaffected):

```python
        corridor_frac = cfg.get("corridor_crossing_fraction", 0.0)
        corridor_term = getattr(getattr(env, "event_manager", None), "cfg", None)
        cc = getattr(getattr(env, "cfg", None), "events", None)
        cc_term = getattr(cc, "corridor_crossing", None)
        if cc_term is not None:
            cc_term.params["fraction"] = float(corridor_frac)
```

Match the exact env/cfg access pattern used by the neighboring `randomize_wall_positions` param write in the same function (they already reach the event term cfg — copy that idiom rather than the illustrative getattr chain above).

- [ ] **Step 6: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/e2e_final20_v1.py source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/goal_obstacle_curriculum.py source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/tests/
git commit -m "feat(curriculum): corridor_crossing_fraction=0.12 for SA3-5, wired into stage apply"
```

---

### Task 4: End-to-end smoke train + D-suite validation baseline

Confirm a short real training run with corridor injection does not crash and that the pedestrian actually herds; establish the D-suite numbers to compare against later.

**Files:**
- Use existing: `scripts/reinforcement_learning/skrl/play_eval/near_wall_crossing_eval.py` (no change)
- Modify (docs only): append a run-recipe note to `docs/training_design_dossier_for_codex.md`

**Interfaces:**
- Consumes: Tasks 1–3 (event + config). No new code.

- [ ] **Step 1: Short headless training smoke (GPU-guarded)**

```bash
cd /home/aa/IsaacLab
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
# Only if GPU has headroom (Codex's K8 A/B may be running — do NOT contend). 200 iters confirms no crash + reward flows.
# K8 stack (lidar_frame_stack=8): use Codex's control config as the base; SA4 has corridor_crossing_fraction=0.12 (Task 3),
# so corridor injection fires under this run.
source /home/aa/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
  scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config scripts/reinforcement_learning/skrl/rnn_car_modular/configs/e2e_sa4_fs8_clean_antispin_control.py \
  --headless --run_name smoke_corridor_k8_sa4_$(date +%m%d) --max_iterations 200 2>&1 | tail -40
```
Expected: no traceback; `reward/term/*` logs present; `charge/obstacle_collision_rate` finite. If a KeyError about `corridor_crossing` appears, the event import/registration (Task 2 Step 3) is wrong — fix and re-run.

- [ ] **Step 2: Record the pre-fix D-suite baseline on SA4 c89600**

```bash
cd /home/aa/IsaacLab
source /home/aa/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_eval/near_wall_crossing_eval.py \
  --checkpoint logs/rnn_car/<sa4_c89600_run>/checkpoint_89600.pt --headless \
  --near_wall_crossing_eval 2>&1 | tail -30
```
Expected: reproduces the known SA4 failure (spin_360 ≈ 7/8, first_turn_correct low). Record these numbers in the dossier as the "before" column. (Locate the exact SA4 c89600 run dir with `ls -dt logs/rnn_car/*4y41ym4w* 2>/dev/null || ls -dt logs/rnn_car/sa4_*`.)

- [ ] **Step 3: Document the full retrain + validation recipe**

Append to `docs/training_design_dossier_for_codex.md` a short section: (a) the SA1→SA5 retrain command chain with corridor injection now active in SA3–5; (b) the acceptance test = run `near_wall_crossing_eval` on the retrained SA5 checkpoint and confirm `spin_360` drops well below 7/8 and `first_turn_correct` rises materially vs the Step-2 baseline; (c) note that per the anti-spin A ablation (run 6lfj72tt), the standard `collision −15` gradient — now that the trap is IN the scene — is expected to teach escape without a symptom-level anti-spin term.

- [ ] **Step 4: Commit**

```bash
git add docs/training_design_dossier_for_codex.md
git commit -m "docs(curriculum): corridor-crossing retrain recipe + D-suite before/after protocol"
```

---

## Notes for the executor

- **Testing reality:** only Task 1 and Task 3 are pure-pytest. Tasks 2 and 4 need Isaac Sim — run their sim steps ONLY when `nvidia-smi` shows headroom (two training runs currently saturate the GPU). If deferred, mark the step and say so explicitly rather than skipping silently.
- **The one real integration risk** is event ordering vs `BehaviorScheduler.reset()`: the injector MUST run after the scheduler assigns behaviors, or slot 0 gets overwritten. Task 2 Step 5's smoke assertion (`pedestrian did not move +x`) is the tripwire for this — do not mark Task 2 complete until that assertion passes.
- **Baseline safety:** with `fraction=0.0` default everywhere except SA3–5, every non-e2e curriculum and every other stage behaves exactly as today.
