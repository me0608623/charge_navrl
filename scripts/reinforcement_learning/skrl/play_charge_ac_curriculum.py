# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
播放訓練好的 Charge VLP16 Curriculum AC agent。

預設 6 envs：2 空曠 + 2 靜態障礙 + 2 動態障礙。

使用方法:
    # 使用最新 checkpoint（自動偵測）
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py

    # 指定 checkpoint
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --checkpoint logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-AC/2026-03-17_21-41-14/best_model/best_agent.pt

    # 自訂 env 數量和場景比例
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --empty 4 --static 4 --dynamic 4

    # 跟隨視角
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py --camera follow
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import glob
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from isaaclab.app import AppLauncher

# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Play trained Charge VLP16 Curriculum AC agent.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum",
                    help="Task name (used for gym.make).")
parser.add_argument("--num_envs", type=int, default=6, help="Number of environments.")
parser.add_argument("--empty", type=int, default=2, help="Number of empty (no obstacle) envs.")
parser.add_argument("--static", type=int, default=2, help="Number of static obstacle envs.")
parser.add_argument("--dynamic", type=int, default=2, help="Number of dynamic obstacle envs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--num_goals", type=int, default=1, help="Number of goals per env.")
parser.add_argument("--num_obstacles_static", type=int, default=5, help="Static obstacles per env.")
parser.add_argument("--num_obstacles_dynamic", type=int, default=3, help="Dynamic obstacles per env.")
parser.add_argument("--stage", type=int, default=None,
                    help="Use a specific curriculum stage config (1-8). Overrides obstacle/goal/wall settings.")
parser.add_argument("--max_steps", type=int, default=5000, help="Max total steps before exit.")
parser.add_argument("--camera", type=str, default="top", choices=["top", "follow", "side"],
                    help="Camera view.")

# AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# If user provides --num_envs, use it; otherwise derive from empty+static+dynamic
if args_cli.num_envs != (args_cli.empty + args_cli.static + args_cli.dynamic):
    # User explicitly set --num_envs, redistribute evenly
    n = args_cli.num_envs
    args_cli.empty = n // 3
    args_cli.static = n // 3
    args_cli.dynamic = n - args_cli.empty - args_cli.static

# Launch sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import skrl
from packaging import version
from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Custom model modules
sys.path.insert(0, str(Path(__file__).parent))
import vlp16_models as _vlp16_models_module
import charge_models as _charge_models_module
_CUSTOM_MODEL_MODULES = {
    "charge_models": _charge_models_module,
    "vlp16_models": _vlp16_models_module,
}

import isaaclab_tasks  # noqa: F401

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    print(f"[ERROR] skrl >= {SKRL_VERSION} required, got {skrl.__version__}")
    exit()


# ============================================================================
# Patch Runner for custom model classes
# ============================================================================
def patch_runner_for_custom_models():
    original_generate_models = Runner._generate_models

    def patched_generate_models(self, env, cfg):
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' defined in cfg")
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True

            if separate:
                for role in models_cfg:
                    model_class_name = models_cfg[role].get("class")
                    if not model_class_name:
                        raise ValueError(f"No 'class' in 'models:{role}'")
                    del models_cfg[role]["class"]

                    if "." in model_class_name:
                        parts = model_class_name.split(".")
                        module = _CUSTOM_MODEL_MODULES.get(parts[0])
                        if module is not None:
                            model_class = getattr(module, parts[1])
                        else:
                            raise ValueError(f"Module '{parts[0]}' not found")
                    else:
                        model_class = self._component(model_class_name)

                    models[agent_id][role] = model_class(
                        observation_space=observation_spaces[agent_id],
                        action_space=action_spaces[agent_id],
                        device=device,
                        **self._process_cfg(models_cfg[role]),
                    )
            else:
                raise ValueError("Shared models not supported")
        return models

    Runner._generate_models = patched_generate_models


patch_runner_for_custom_models()


# ============================================================================
# Checkpoint finder
# ============================================================================
def find_latest_checkpoint():
    """自動找到最新的 best_agent.pt。"""
    task_name = args_cli.task or ""
    if "NavRL" in task_name:
        log_base = "logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-AC"
    else:
        log_base = "logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-AC"
    if not os.path.exists(log_base):
        print(f"[CHECKPOINT] Directory not found: {log_base}")
        return None

    for run_dir in sorted(os.listdir(log_base), reverse=True):
        # best_model/best_agent.pt
        best = os.path.join(log_base, run_dir, "best_model", "best_agent.pt")
        if os.path.exists(best):
            print(f"[CHECKPOINT] Found: {best}")
            return best
        # checkpoints/best_agent.pt
        ckpt_best = os.path.join(log_base, run_dir, "checkpoints", "best_agent.pt")
        if os.path.exists(ckpt_best):
            print(f"[CHECKPOINT] Found: {ckpt_best}")
            return ckpt_best
        # highest agent_NNNN.pt
        ckpt_dir = os.path.join(log_base, run_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            pts = sorted(glob.glob(os.path.join(ckpt_dir, "agent_*.pt")))
            if pts:
                print(f"[CHECKPOINT] Found: {pts[-1]}")
                return pts[-1]

    print("[CHECKPOINT] No checkpoint found")
    return None


# ============================================================================
# Main
# ============================================================================
def main():
    n_empty = args_cli.empty
    n_static = args_cli.static
    n_dynamic = args_cli.dynamic
    total_envs = n_empty + n_static + n_dynamic

    print(f"\n{'='*70}")
    print(f"  Charge VLP16 Curriculum — Play Mode")
    print(f"{'='*70}")
    print(f"  Envs:       {total_envs} ({n_empty} empty + {n_static} static + {n_dynamic} dynamic)")
    print(f"  Goals:      {args_cli.num_goals}")
    print(f"  Obstacles:  {args_cli.num_obstacles_static}s + {args_cli.num_obstacles_dynamic}d")
    print(f"  Checkpoint: {args_cli.checkpoint or 'Auto-detect'}")
    print(f"  Camera:     {args_cli.camera}")
    print(f"{'='*70}\n")

    # --- Load env config directly (no Hydra) ---
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
        ChargeNavigationEnvCfgVLP16Curriculum,
        ChargeNavigationEnvCfgVLP16CurriculumNavRL,
    )
    if "NavRL" in args_cli.task:
        env_cfg = ChargeNavigationEnvCfgVLP16CurriculumNavRL()
        print("[INFO] Using NavRL reward config")
    else:
        env_cfg = ChargeNavigationEnvCfgVLP16Curriculum()
    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Camera
    from isaaclab.envs import ViewerCfg
    if args_cli.camera == "top":
        env_cfg.viewer = ViewerCfg(
            eye=(0.0, 0.0, 30.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )
    elif args_cli.camera == "follow":
        env_cfg.viewer = ViewerCfg(
            eye=(-3.0, 0.0, 3.0), lookat=(2.0, 0.0, 0.0),
            origin_type="asset_root", env_index=0, asset_name="robot",
            resolution=(1920, 1080),
        )
    elif args_cli.camera == "side":
        env_cfg.viewer = ViewerCfg(
            eye=(15.0, -15.0, 15.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )

    # --- Load agent config directly from YAML ---
    agent_yaml = Path(__file__).parent.parent.parent.parent / (
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
        "locomotion/velocity/config/charge_skrl/agents/skrl_ppo_cfg_vlp16.yaml"
    )
    if not agent_yaml.exists():
        # Fallback: relative to CWD
        agent_yaml = Path(
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
            "locomotion/velocity/config/charge_skrl/agents/skrl_ppo_cfg_vlp16.yaml"
        )
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)
    print(f"[INFO] Agent config loaded from: {agent_yaml}")

    # Play mode: no training
    agent_cfg["agent"]["experiment"]["directory"] = "logs/skrl/play"
    agent_cfg["agent"]["experiment"]["experiment_name"] = "play"
    agent_cfg["trainer"]["timesteps"] = 0

    # --- Disable curriculum (play mode: fixed obstacle config, no stage transitions) ---
    env_cfg.curriculum = None
    print("[INFO] Curriculum disabled for play mode")

    # --- Stage override: 使用指定課程階段的完整配置 ---
    if args_cli.stage is not None:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import STAGES, MAX_STAGE
        stage = max(1, min(args_cli.stage, MAX_STAGE))
        s = STAGES[stage]
        # 覆蓋 obstacle 參數
        args_cli.num_obstacles_static = s["num_obstacles_static"]
        args_cli.num_obstacles_dynamic = s["num_obstacles_dynamic"]
        args_cli.num_goals = s["num_goals"]
        # 覆蓋 env mix (有障礙物時: 0% empty)
        if s["num_obstacles_static"] + s["num_obstacles_dynamic"] > 0:
            n_empty = 0
            n_static = args_cli.num_envs // 2
            n_dynamic = args_cli.num_envs - n_static
        else:
            n_empty = args_cli.num_envs
            n_static = 0
            n_dynamic = 0
        # 覆蓋牆壁參數
        wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
        if wall_evt is not None:
            wall_evt.params["min_walls"] = s["min_walls"]
            wall_evt.params["max_walls"] = s["max_walls"]
        print(
            f"[INFO] Stage {stage} override: "
            f"{s['num_goals']}G / {s['num_obstacles_static']}S+{s['num_obstacles_dynamic']}D / "
            f"walls {s['min_walls']}-{s['max_walls']} / "
            f"episode {s['episode_length_s']}s"
        )

    # --- Override event obstacle ratios in env_cfg BEFORE env creation ---
    total_envs = n_empty + n_static + n_dynamic
    obstacle_params = {
        "empty_ratio": n_empty / total_envs,
        "static_ratio": n_static / total_envs,
        "dynamic_ratio": n_dynamic / total_envs,
        "num_obstacles_static": args_cli.num_obstacles_static,
        "num_obstacles_dynamic": args_cli.num_obstacles_dynamic,
    }
    # Override both startup and reset event params
    for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
        evt_term = getattr(env_cfg.events, evt_attr, None)
        if evt_term is not None:
            evt_term.params.update(obstacle_params)
    print(f"[INFO] Obstacles: {n_empty} empty + {n_static} static + {n_dynamic} dynamic "
          f"({args_cli.num_obstacles_static}s + {args_cli.num_obstacles_dynamic}d per env)")

    # --- Override goal config: uniform across 20×20m scene ---
    import math
    env_cfg.commands.goal_command.num_goals = args_cli.num_goals
    env_cfg.commands.goal_command.ranges.distance = (2.0, 14.0)  # 覆蓋整個 20×20 場景（對角線 ~14m）
    env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)
    env_cfg.commands.goal_command.wall_boundary = 9.5  # 20×20 場景邊界
    print(f"[INFO] Goals: {args_cli.num_goals}, distance=(2.0, 14.0)m, full 360°")

    # --- Create env ---
    task_name = args_cli.task
    env = gym.make(task_name, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # --- Wrap for SKRL ---
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    print(f"[INFO] observation_space: {env.observation_space}")
    print(f"[INFO] action_space: {env.action_space}")

    # --- Setup agent ---
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(env, agent_cfg)

    # Load checkpoint
    checkpoint_path = args_cli.checkpoint
    if checkpoint_path:
        checkpoint_path = retrieve_file_path(checkpoint_path)
    else:
        checkpoint_path = find_latest_checkpoint()

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            runner.agent.load(checkpoint_path)
            print(f"[OK] Loaded: {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[WARN] No checkpoint — running random policy!")

    runner.agent.set_running_mode("eval")

    # --- Play loop ---
    print(f"\n{'='*70}")
    print(f"  Running... (max {args_cli.max_steps} steps, Ctrl+C to stop)")
    print(f"{'='*70}\n")

    episode_counts = {"success": 0, "collision": 0, "timeout": 0}
    start_time = time.time()
    step = 0

    try:
        obs, _ = env.reset()
        for step in range(args_cli.max_steps):
            with torch.no_grad():
                actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
            obs, reward, terminated, truncated, infos = env.step(actions)

            # Count episodes
            if step % 50 == 0:
                try:
                    tm = env.unwrapped.unwrapped.termination_manager
                    for name in tm._term_names:
                        buf = tm.get_term(name)
                        if buf is not None:
                            count = int(buf.sum().item())
                            if "goal_reached" in name:
                                episode_counts["success"] += count
                            elif "collision" in name:
                                episode_counts["collision"] += count
                            elif "time_out" in name:
                                episode_counts["timeout"] += count
                except Exception:
                    pass

            if step % 500 == 0 and step > 0:
                elapsed = time.time() - start_time
                total_ep = sum(episode_counts.values())
                if total_ep > 0:
                    sr = episode_counts["success"] / total_ep * 100
                    cr = episode_counts["collision"] / total_ep * 100
                    tr = episode_counts["timeout"] / total_ep * 100
                    print(f"  Step {step:>5} | {elapsed:.0f}s | "
                          f"Ep={total_ep} SR={sr:.1f}% CR={cr:.1f}% TO={tr:.1f}%")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    # --- Summary ---
    elapsed = time.time() - start_time
    total_ep = sum(episode_counts.values())
    print(f"\n{'='*70}")
    print(f"  PLAY SUMMARY")
    print(f"{'='*70}")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Steps:      {step + 1}")
    print(f"  Episodes:   {total_ep}")
    if total_ep > 0:
        print(f"  Success:    {episode_counts['success']} ({episode_counts['success']/total_ep*100:.1f}%)")
        print(f"  Collision:  {episode_counts['collision']} ({episode_counts['collision']/total_ep*100:.1f}%)")
        print(f"  Timeout:    {episode_counts['timeout']} ({episode_counts['timeout']/total_ep*100:.1f}%)")
    print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
