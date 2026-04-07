# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
播放訓練好的 Charge VLP16 Curriculum AC agent — 自動逐階段展示。

預設 6 envs，從 Stage 1 開始，每階段跑 200 steps 後自動升階到 Stage 8。

使用方法:
    # 預設：6 envs, stage 1→8, 每階 200 steps
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py

    # 指定 checkpoint
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --checkpoint logs/skrl/.../best_agent.pt

    # 自訂起始階段和每階步數
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --start_stage 3 --steps_per_stage 500

    # 只跑單一階段（不自動升階）
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --start_stage 5 --end_stage 5

    # 跟隨視角
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py --camera follow

    # 不使用課程學習，自訂障礙物與牆壁數量
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ac_curriculum.py \
        --no_curriculum --num_static 5 --num_dynamic 4 --num_walls 6 --max_steps 2000
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
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
                    help="Task name (used for gym.make). 預設 NavRL 與訓練一致。")
parser.add_argument("--num_envs", type=int, default=6, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--start_stage", type=int, default=1, help="Starting curriculum stage (1-8).")
parser.add_argument("--end_stage", type=int, default=8, help="Final curriculum stage (1-8).")
parser.add_argument("--steps_per_stage", type=int, default=200, help="Steps per stage before advancing.")
parser.add_argument("--curriculum_version", type=str, default="baseline_v1",
                    help="Curriculum version to use (baseline_v1, goal_first_v1).")
parser.add_argument("--no_curriculum", action="store_true", default=False,
                    help="Disable stage cycling; use fixed obstacle/wall counts from CLI.")
parser.add_argument("--num_static", type=int, default=3, help="Static obstacles per env (--no_curriculum mode).")
parser.add_argument("--num_dynamic", type=int, default=3, help="Dynamic obstacles per env (--no_curriculum mode).")
parser.add_argument("--num_walls", type=int, default=None, help="Number of internal walls (overrides curriculum stage walls if set).")
parser.add_argument("--max_steps", type=int, default=1600, help="Max steps in --no_curriculum mode.")
parser.add_argument("--camera", type=str, default="top", choices=["top", "follow", "side"],
                    help="Camera view.")
parser.add_argument("--use_cadn", action="store_true", default=False,
                    help="Use PerBranchCADN normalizer (required when checkpoint was trained with CADN).")
parser.add_argument("--diagnostic", action="store_true", default=False,
                    help="每步印出 env[0] 的 obs/action 詳細診斷（除錯用）。")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="使用 argmax（確定性動作）取代 sample，排除隨機抽樣造成的振盪。")
parser.add_argument("--cadn_online", action="store_true", default=False,
                    help="Play 時讓 CADN 持續更新 stats（train=True），避免凍結 stats 導致速度振盪。")
parser.add_argument("--match_level", type=int, default=None,
                    help="自動套用 open_ended 指定 level 的完整訓練參數（goal_distance, boundary, "
                         "obstacle ratios, episode_length 等）。用法: --match_level 30 = 對齊 L30 訓練環境。")
parser.add_argument("--episode_length", type=float, default=None,
                    help="覆寫 episode 長度 (秒)。不指定時: --match_level 自動設定，否則用 env 預設。")
parser.add_argument("--no_domain_randomization", action="store_true", default=False,
                    help="關閉 domain_randomization event (physics/sensor_noise/external_force)。"
                         "對齊訓練時的 --no_domain_randomization。")
parser.add_argument("--lidar_no_noise", action="store_true", default=False,
                    help="關閉 LiDAR 觀測函數內建合成噪聲 (displacement/hole/distractor/Unoise)。"
                         "對齊訓練時的 --lidar_no_noise。")

# AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- Play 模式渲染最佳化 ---
if not args_cli.headless:
    if args_cli.rendering_mode is None or args_cli.rendering_mode == "balanced":
        args_cli.rendering_mode = "performance"
    if not getattr(args_cli, "enable_cameras", False):
        args_cli.enable_cameras = True
    extra_kit = " ".join([
        "--/rtx/indirectDiffuse/enabled=true",
        "--/rtx/shadows/enabled=false",
        "--/app/asyncRendering=true",
        "--/app/asyncRenderingLowLatency=true",
    ])
    args_cli.kit_args = f"{args_cli.kit_args} {extra_kit}" if args_cli.kit_args else extra_kit

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
    import math

    use_curriculum = not args_cli.no_curriculum and args_cli.match_level is None

    # --- Load env config ---
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
        ChargeNavigationEnvCfgVLP16Curriculum,
        ChargeNavigationEnvCfgVLP16CurriculumNavRL,
    )
    if "NavRL" in args_cli.task:
        env_cfg = ChargeNavigationEnvCfgVLP16CurriculumNavRL()
    else:
        env_cfg = ChargeNavigationEnvCfgVLP16Curriculum()
    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Disable training curriculum (we control stage transitions or use fixed config)
    env_cfg.curriculum = None

    # Play mode: goal/robot 離牆壁/邊界至少 1.0m
    env_cfg.commands.goal_command.wall_safe_margin = 1.0

    # Play 模式: render_interval 折衷（每 action 渲染 5 幀 ~25 FPS）
    if not args_cli.headless:
        env_cfg.sim.render_interval = 4

    # ========================================================================
    # 對齊訓練 flag — Bug A/B 修復後的乾淨環境
    # ========================================================================
    # --- no_domain_randomization: 關閉 physics/sensor_noise/external_force ---
    if args_cli.no_domain_randomization:
        events = getattr(env_cfg, "events", None)
        if events is not None:
            dr = getattr(events, "domain_randomization", None)
            if dr is not None:
                dr.params["enable_physics"] = False
                dr.params["enable_sensor_noise"] = False
                dr.params["enable_external_force"] = False
                print("[NO_DR] domain_randomization event: physics/sensor_noise/external_force = False")

    # --- lidar_no_noise: 關閉 LiDAR 觀測函數內建噪聲 ---
    if args_cli.lidar_no_noise:
        obs_root = getattr(env_cfg, "observations", None)
        if obs_root is not None:
            cleared = []
            for group_name in dir(obs_root):
                if group_name.startswith("_"):
                    continue
                group = getattr(obs_root, group_name, None)
                if group is None or not hasattr(group, "__dict__"):
                    continue
                for term_name in dir(group):
                    if term_name.startswith("_") or "lidar" not in term_name.lower():
                        continue
                    term = getattr(group, term_name, None)
                    if term is None or not hasattr(term, "params"):
                        continue
                    params = term.params
                    if "displacement_std" in params:
                        params["displacement_std"] = 0.0
                    if "hole_rate" in params:
                        params["hole_rate"] = 0.0
                    if "distractor_rate" in params:
                        params["distractor_rate"] = 0.0
                    if hasattr(term, "noise"):
                        term.noise = None
                    cleared.append(f"{group_name}.{term_name}")
            print(f"[NO_LIDAR_NOISE] LiDAR observation noise disabled: {cleared}")
            print(f"[NO_LIDAR_NOISE] displacement_std=0, hole_rate=0, distractor_rate=0, Unoise=None")

    if use_curriculum:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            _apply_stage, _load_stages,
        )
        stages, max_stage, _ = _load_stages(args_cli.curriculum_version)
        start_stage = max(1, min(args_cli.start_stage, max_stage))
        end_stage = max(start_stage, min(args_cli.end_stage, max_stage))
        steps_per_stage = args_cli.steps_per_stage

        print(f"\n{'='*70}")
        print(f"  Charge VLP16 Curriculum — Play Mode")
        print(f"{'='*70}")
        print(f"  Envs:            {args_cli.num_envs}")
        print(f"  Curriculum:      {args_cli.curriculum_version}")
        print(f"  Stages:          {start_stage} → {end_stage}  ({end_stage - start_stage + 1} stages)")
        print(f"  Steps/stage:     {steps_per_stage}")
        print(f"  Checkpoint:      {args_cli.checkpoint or 'Auto-detect'}")
        print(f"  Camera:          {args_cli.camera}")
        print(f"{'='*70}\n")

        # Apply initial stage config BEFORE env creation
        s1 = stages[start_stage]
        for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
            evt_term = getattr(env_cfg.events, evt_attr, None)
            if evt_term is not None:
                evt_term.params.update({
                    "empty_ratio": s1["empty_ratio"],
                    "static_ratio": s1["static_ratio"],
                    "dynamic_ratio": s1["dynamic_ratio"],
                    "num_obstacles_static": s1["num_obstacles_static"],
                    "num_obstacles_dynamic": s1["num_obstacles_dynamic"],
                })
        wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
        if wall_evt is not None:
            if args_cli.num_walls is not None:
                wall_evt.params["min_walls"] = args_cli.num_walls
                wall_evt.params["max_walls"] = args_cli.num_walls
            else:
                wall_evt.params["min_walls"] = s1["min_walls"]
                wall_evt.params["max_walls"] = s1["max_walls"]

        env_cfg.commands.goal_command.num_goals = s1["num_goals"]
        env_cfg.commands.goal_command.num_obstacles = s1["num_obstacles_static"] + s1["num_obstacles_dynamic"]
        env_cfg.commands.goal_command.ranges.distance = s1["goal_distance"]
        env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)

    elif args_cli.match_level is not None:
        # --- Match Training Level mode: replicate exact open_ended L{N} params ---
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            _open_ended_params,
        )
        ml = args_cli.match_level
        oe = _open_ended_params(ml)

        # Apply obstacle ratios exactly as training curriculum does
        obs_params = {
            "empty_ratio": oe["empty_ratio"],
            "static_ratio": oe["static_ratio"],
            "dynamic_ratio": oe["dynamic_ratio"],
            "num_obstacles_static": oe["num_obstacles_static"],
            "num_obstacles_dynamic": oe["num_obstacles_dynamic"],
            "boundary": oe["boundary"],
            # active_obstacle_ratio stays at config default (0.25) — same as training
        }
        for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
            evt_term = getattr(env_cfg.events, evt_attr, None)
            if evt_term is not None:
                evt_term.params.update(obs_params)

        # Dynamic obstacle speed
        move_evt = getattr(env_cfg.events, "move_dynamic_obstacles", None)
        if move_evt is not None:
            move_evt.params["speed_max"] = oe["speed_range"]

        # Walls (open_ended = no walls)
        wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
        if wall_evt is not None:
            n_w = args_cli.num_walls if args_cli.num_walls is not None else oe["max_walls"]
            wall_evt.params["min_walls"] = oe["min_walls"] if args_cli.num_walls is None else n_w
            wall_evt.params["max_walls"] = n_w

        # Goal command
        env_cfg.commands.goal_command.num_goals = oe["num_goals"]
        env_cfg.commands.goal_command.num_obstacles = (
            oe["num_obstacles_static"] + oe["num_obstacles_dynamic"]
        )
        env_cfg.commands.goal_command.ranges.distance = oe["goal_distance"]
        env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)

        # Episode length (match training unless explicitly overridden)
        ep_len = args_cli.episode_length if args_cli.episode_length is not None else oe["episode_length_s"]
        env_cfg.episode_length_s = ep_len

        # Compute actual visible obstacle count for display
        n_s_cfg = oe["num_obstacles_static"]
        n_d_cfg = oe["num_obstacles_dynamic"]
        if oe["dynamic_ratio"] > 0:
            # active_obstacle_ratio=0.25 (config default), only applies to dynamic mode
            est_visible = max(1, int(n_d_cfg * 0.25))
            vis_note = f"~{est_visible} visible (active_ratio=0.25)"
        else:
            est_visible = n_s_cfg
            vis_note = f"{est_visible} visible"

        print(f"\n{'='*70}")
        print(f"  Charge VLP16 — Match Training Level L{ml}")
        print(f"{'='*70}")
        print(f"  Task:            {args_cli.task}")
        print(f"  Envs:            {args_cli.num_envs}")
        print(f"  Config S/D:      {n_s_cfg}S + {n_d_cfg}D = {n_s_cfg + n_d_cfg} total")
        print(f"  Actual visible:  {vis_note}")
        print(f"  Goal distance:   {oe['goal_distance']}")
        print(f"  Boundary:        {oe['boundary']}m")
        print(f"  Episode length:  {ep_len}s ({int(ep_len / (env_cfg.sim.dt * env_cfg.decimation))} steps)")
        print(f"  Walls:           {oe['min_walls']}~{n_w}")
        print(f"  Speed range:     {oe['speed_range']:.1f}")
        print(f"  Checkpoint:      {args_cli.checkpoint or 'Auto-detect'}")
        print(f"  CADN:            {args_cli.use_cadn}")
        print(f"{'='*70}\n")

    else:
        # --- Fixed mode: user-specified obstacles/walls ---
        n_s = args_cli.num_static
        n_d = args_cli.num_dynamic
        n_w = args_cli.num_walls if args_cli.num_walls is not None else 4
        has_obs = (n_s + n_d) > 0

        # 障礙物分配: 同時有 static + dynamic → mixed 模式
        if n_s > 0 and n_d > 0:
            obs_params = {
                "empty_ratio": 0.0, "static_ratio": 0.0, "dynamic_ratio": 0.0,
                "mixed_ratio": 1.0,
                "num_obstacles_static": n_s, "num_obstacles_dynamic": n_d,
            }
        elif n_d > 0:
            obs_params = {
                "empty_ratio": 0.0, "static_ratio": 0.0, "dynamic_ratio": 1.0,
                "num_obstacles_static": 0, "num_obstacles_dynamic": n_d,
            }
        elif n_s > 0:
            obs_params = {
                "empty_ratio": 0.0, "static_ratio": 1.0, "dynamic_ratio": 0.0,
                "num_obstacles_static": n_s, "num_obstacles_dynamic": 0,
            }
        else:
            obs_params = {
                "empty_ratio": 1.0, "static_ratio": 0.0, "dynamic_ratio": 0.0,
                "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
            }

        max_obs = max(n_s + n_d, 10)
        obs_params["max_obstacles"] = max_obs
        for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
            evt_term = getattr(env_cfg.events, evt_attr, None)
            if evt_term is not None:
                evt_term.params.update(obs_params)
        # 同步 move_dynamic_obstacles 的 max_obstacles
        move_evt = getattr(env_cfg.events, "move_dynamic_obstacles", None)
        if move_evt is not None:
            move_evt.params["max_obstacles"] = max_obs
        wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
        if wall_evt is not None:
            wall_evt.params["min_walls"] = n_w
            wall_evt.params["max_walls"] = n_w

        env_cfg.commands.goal_command.num_goals = 1
        env_cfg.commands.goal_command.num_obstacles = n_s + n_d
        env_cfg.commands.goal_command.ranges.distance = (2.0, 14.0)
        env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)

        # Episode length override
        if args_cli.episode_length is not None:
            env_cfg.episode_length_s = args_cli.episode_length

        print(f"\n{'='*70}")
        print(f"  Charge VLP16 — Fixed Play Mode (no curriculum)")
        print(f"{'='*70}")
        print(f"  Task:            {args_cli.task}")
        print(f"  Envs:            {args_cli.num_envs}")
        print(f"  Static obs:      {n_s}")
        print(f"  Dynamic obs:     {n_d}")
        print(f"  Walls:           {n_w}")
        print(f"  Max steps:       {args_cli.max_steps}")
        print(f"  Checkpoint:      {args_cli.checkpoint or 'Auto-detect'}")
        print(f"  Camera:          {args_cli.camera}")
        print(f"  CADN:            {args_cli.use_cadn}")
        print(f"  Diagnostic:      {args_cli.diagnostic}")
        # OOD 警告
        total_obs = n_s + n_d
        if total_obs > 30:
            print(f"  {'─'*56}")
            print(f"  [WARN] {total_obs} obstacles 可能超出訓練分布")
            print(f"  [HINT] 建議先用 --num_static 8 --num_dynamic 3 驗證基本功能")
        print(f"{'='*70}\n")

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

    # --- Load agent config from YAML ---
    agent_yaml = Path(__file__).parent.parent.parent.parent / (
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
        "locomotion/velocity/config/charge_skrl/agents/skrl_ppo_cfg_vlp16.yaml"
    )
    if not agent_yaml.exists():
        agent_yaml = Path(
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
            "locomotion/velocity/config/charge_skrl/agents/skrl_ppo_cfg_vlp16.yaml"
        )
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["agent"]["experiment"]["directory"] = "logs/skrl/play"
    agent_cfg["agent"]["experiment"]["experiment_name"] = "play"
    agent_cfg["trainer"]["timesteps"] = 0

    # --- Create env ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # --- Setup agent ---
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(env, agent_cfg)

    # CADN: replace state_preprocessor before loading checkpoint
    if args_cli.use_cadn:
        try:
            from cadn import PerBranchCADN
            cadn = PerBranchCADN(device=env.device)
        except (ImportError, ModuleNotFoundError):
            from cadn_preprocessor import CurriculumAwareDualRateNormalizer
            cadn = CurriculumAwareDualRateNormalizer(size=139, device=env.device)
        runner.agent._state_preprocessor = cadn
        runner.agent.checkpoint_modules["state_preprocessor"] = cadn
        print(f"[INFO] CADN normalizer enabled: {type(cadn).__name__}")

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

    # --- CADN Verification ---
    if args_cli.use_cadn:
        cadn_mod = runner.agent._state_preprocessor
        if hasattr(cadn_mod, '_per_branch') and cadn_mod._per_branch:
            for bname, branch in [("state", cadn_mod.branch_state),
                                   ("lidar", cadn_mod.branch_lidar),
                                   ("obstacle", cadn_mod.branch_obstacle)]:
                init = branch._initialized.item()
                mu_rng = f"[{branch.mu_f.min():.3f}, {branch.mu_f.max():.3f}]"
                var_rng = f"[{branch.var_f.min():.4f}, {branch.var_f.max():.4f}]"
                drift = branch.get_drift()
                print(f"  CADN {bname:>8}: init={init} mu_f={mu_rng} var_f={var_rng} drift={drift:.4f}")

    # --- Deterministic action monkey-patch ---
    if args_cli.deterministic:
        print("[INFO] Deterministic mode: using argmax instead of sample")
        import types
        _orig_policy_act = runner.agent.policy.act

        def _deterministic_act(self_policy, inputs, role=""):
            net_output, outputs = self_policy.compute(inputs, role)
            # Split logits and take argmax per category
            nvec = self_policy.action_space.nvec.tolist()
            splits = torch.split(net_output, nvec, dim=-1)
            actions = torch.stack([s.argmax(dim=-1) for s in splits], dim=-1)
            # Compute log_prob for diagnostics
            dists = [torch.distributions.Categorical(logits=s) for s in splits]
            log_prob = torch.stack(
                [d.log_prob(a) for d, a in zip(dists, torch.unbind(actions, dim=-1))],
                dim=-1,
            ).sum(dim=-1, keepdim=True)
            outputs["net_output"] = net_output
            return actions, log_prob, outputs
        runner.agent.policy.act = types.MethodType(_deterministic_act, runner.agent.policy)

    # --- CADN online mode: let stats adapt during play ---
    if args_cli.use_cadn and args_cli.cadn_online:
        _orig_preproc = runner.agent._state_preprocessor
        def _online_preproc(states, **kwargs):
            return _orig_preproc(states, train=True, **kwargs)
        runner.agent._state_preprocessor = _online_preproc
        print("[INFO] CADN online mode: stats will update during play (train=True)")

    # --- Unwrap to get the real ManagerBasedRLEnv for _apply_stage ---
    raw_env = env.unwrapped
    while hasattr(raw_env, "unwrapped") and raw_env is not raw_env.unwrapped:
        raw_env = raw_env.unwrapped

    # --- Play loop ---
    global_start = time.time()
    global_step = 0

    def _count_episodes(raw_env, counts):
        """Accumulate termination counts from current step."""
        try:
            tm = raw_env.termination_manager
            for name in tm._term_names:
                buf = tm.get_term(name)
                if buf is not None:
                    c = int(buf.sum().item())
                    if c > 0:
                        if "goal_reached" in name:
                            counts["success"] += c
                        elif "collision" in name:
                            counts["collision"] += c
                        elif "time_out" in name:
                            counts["timeout"] += c
        except Exception:
            pass

    if use_curriculum:
        # ── Curriculum mode: cycle through stages ──
        stage_results = []

        try:
            for current_stage in range(start_stage, end_stage + 1):
                s_cfg = stages[current_stage]
                s_name = s_cfg.get("name", "")
                s_goals = s_cfg["num_goals"]
                s_static = s_cfg["num_obstacles_static"]
                s_dynamic = s_cfg["num_obstacles_dynamic"]
                if args_cli.num_walls is not None:
                    s_walls = str(args_cli.num_walls)
                else:
                    s_walls = f"{s_cfg['min_walls']}-{s_cfg['max_walls']}"

                print(f"\n{'─'*70}")
                print(f"  Stage {current_stage}/{max_stage}: {s_name}")
                print(f"  {s_goals}G / {s_static}S+{s_dynamic}D / walls {s_walls} / "
                      f"γ={s_cfg['gamma']} / ep={s_cfg['episode_length_s']}s")
                print(f"  Running {steps_per_stage} steps...")
                print(f"{'─'*70}")

                _apply_stage(raw_env, current_stage)

                # Override walls after _apply_stage if CLI specified
                if args_cli.num_walls is not None:
                    try:
                        evt = raw_env.event_manager
                        ec = evt.get_term_cfg("randomize_wall_positions")
                        ec.params["min_walls"] = args_cli.num_walls
                        ec.params["max_walls"] = args_cli.num_walls
                        evt.set_term_cfg("randomize_wall_positions", ec)
                    except Exception:
                        pass

                obs, _ = env.reset()

                counts = {"success": 0, "collision": 0, "timeout": 0}
                stage_start = time.time()

                for _ in range(steps_per_stage):
                    with torch.no_grad():
                        actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                    obs, reward, terminated, truncated, infos = env.step(actions)
                    global_step += 1
                    _count_episodes(raw_env, counts)

                stage_elapsed = time.time() - stage_start
                total_ep = sum(counts.values())
                sr = counts["success"] / total_ep * 100 if total_ep > 0 else 0
                cr = counts["collision"] / total_ep * 100 if total_ep > 0 else 0
                tr = counts["timeout"] / total_ep * 100 if total_ep > 0 else 0

                stage_results.append({
                    "stage": current_stage, "name": s_name,
                    "episodes": total_ep, "sr": sr, "cr": cr, "tr": tr,
                    "elapsed": stage_elapsed,
                })
                print(f"  Stage {current_stage} done: {stage_elapsed:.1f}s | "
                      f"Ep={total_ep} SR={sr:.0f}% CR={cr:.0f}% TO={tr:.0f}%")

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")

        # Curriculum summary
        total_elapsed = time.time() - global_start
        print(f"\n{'='*70}")
        print(f"  PLAY SUMMARY — {args_cli.curriculum_version}")
        print(f"{'='*70}")
        print(f"  {'Stage':<8} {'Name':<20} {'Ep':>4} {'SR':>6} {'CR':>6} {'TO':>6} {'Time':>6}")
        print(f"  {'─'*56}")
        total_ep_all = 0
        total_sr_all = 0
        total_cr_all = 0
        for r in stage_results:
            print(f"  {r['stage']:<8} {r['name']:<20} {r['episodes']:>4} "
                  f"{r['sr']:>5.0f}% {r['cr']:>5.0f}% {r['tr']:>5.0f}% {r['elapsed']:>5.1f}s")
            total_ep_all += r["episodes"]
            total_sr_all += r["episodes"] * r["sr"] / 100
            total_cr_all += r["episodes"] * r["cr"] / 100
        print(f"  {'─'*56}")
        avg_sr = total_sr_all / total_ep_all * 100 if total_ep_all > 0 else 0
        avg_cr = total_cr_all / total_ep_all * 100 if total_ep_all > 0 else 0
        print(f"  {'Total':<8} {'':<20} {total_ep_all:>4} {avg_sr:>5.0f}% {avg_cr:>5.0f}%")
        print(f"\n  Duration: {total_elapsed:.1f}s | Steps: {global_step}")
        print(f"{'='*70}\n")

    else:
        # ── Fixed mode: single configuration, run max_steps ──
        counts = {"success": 0, "collision": 0, "timeout": 0}
        diag = args_cli.diagnostic
        # 診斷用累積器
        _diag_zero_act_count = 0  # center(9,9) action 次數
        _diag_speed_sum = 0.0
        _diag_step_count = 0

        print(f"  Running {args_cli.max_steps} steps... (Ctrl+C to stop)\n")

        try:
            obs, _ = env.reset()

            # ── 一次性診斷: Step 0 obs pipeline 比對 ──
            if diag:
                try:
                    o = obs[0]
                    print(f"  [DIAG] ═══ Step 0 Obs Pipeline (env[0]) ═══")
                    print(f"  [DIAG] Raw obs shape: {obs.shape} dtype: {obs.dtype}")
                    print(f"  [DIAG] ego[0:4]   = [{o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}, {o[3]:.3f}]")
                    print(f"  [DIAG] goal[4:6]  = [{o[4]:.3f}, {o[5]:.3f}]")
                    print(f"  [DIAG] lidar[6:78]  min={o[6:78].min():.4f} mean={o[6:78].mean():.4f} max={o[6:78].max():.4f}")
                    print(f"  [DIAG] obs[78:138]  non-zero slots={int((o[78:138].reshape(10,6).abs().sum(1) > 0.01).sum())}/10")
                    print(f"  [DIAG] time[138]  = {o[138]:.4f}")
                    if args_cli.use_cadn:
                        cadn_mod = runner.agent._state_preprocessor
                        n = cadn_mod(obs[:1])[0]
                        print(f"  [DIAG] CADN ego   = [{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}, {n[3]:.2f}]")
                        print(f"  [DIAG] CADN goal  = [{n[4]:.2f}, {n[5]:.2f}]")
                        print(f"  [DIAG] CADN lidar min={n[6:78].min():.2f} mean={n[6:78].mean():.2f} max={n[6:78].max():.2f}")
                        print(f"  [DIAG] CADN obs   min={n[78:138].min():.2f} mean={n[78:138].mean():.2f} max={n[78:138].max():.2f}")
                        print(f"  [DIAG] CADN time  = {n[138]:.2f}")
                    print(f"  [DIAG] ════════��═════════════════════════════")
                except Exception as e:
                    print(f"  [DIAG] Step 0 pipeline error: {e}")

            # ── LOGIT 診斷函數 ──
            def _logit_diagnostic(obs_t, label=""):
                """直接呼叫 policy.compute() 檢視 raw logits + action distribution."""
                try:
                    import torch.nn.functional as F
                    cadn_mod = runner.agent._state_preprocessor
                    preprocessed = cadn_mod(obs_t)
                    net_output, _ = runner.agent.policy.compute(
                        {"states": preprocessed}, role="policy"
                    )
                    # net_output: [N, 38] = [19 linear, 19 angular]
                    logits_lin = net_output[0, :19]
                    logits_ang = net_output[0, 19:]
                    probs_lin = F.softmax(logits_lin, dim=0)
                    probs_ang = F.softmax(logits_ang, dim=0)

                    # 分區統計: reverse(0-8), stop(9), forward(10-18)
                    p_rev = probs_lin[:9].sum().item()
                    p_stop = probs_lin[9].item()
                    p_fwd = probs_lin[10:].sum().item()
                    p_left = probs_ang[:9].sum().item()
                    p_str = probs_ang[9].item()
                    p_right = probs_ang[10:].sum().item()

                    argmax_lin = logits_lin.argmax().item()
                    argmax_ang = logits_ang.argmax().item()

                    h_lin = -(probs_lin * probs_lin.log()).sum().item()
                    h_ang = -(probs_ang * probs_ang.log()).sum().item()
                    h_max = torch.log(torch.tensor(19.0)).item()

                    print(f"  [LOGIT] {label}")
                    print(f"  [LOGIT]   lin logits: min={logits_lin.min():.3f} max={logits_lin.max():.3f} "
                          f"range={logits_lin.max()-logits_lin.min():.3f} | "
                          f"argmax={argmax_lin} ({'REV' if argmax_lin < 9 else 'STOP' if argmax_lin == 9 else 'FWD'})")
                    print(f"  [LOGIT]   ang logits: min={logits_ang.min():.3f} max={logits_ang.max():.3f} "
                          f"range={logits_ang.max()-logits_ang.min():.3f} | "
                          f"argmax={argmax_ang} ({'L' if argmax_ang < 9 else 'STR' if argmax_ang == 9 else 'R'})")
                    print(f"  [LOGIT]   lin probs: REV={p_rev:.1%} STOP={p_stop:.1%} FWD={p_fwd:.1%}")
                    print(f"  [LOGIT]   ang probs: LEFT={p_left:.1%} STR={p_str:.1%} RIGHT={p_right:.1%}")
                    print(f"  [LOGIT]   H_lin={h_lin:.3f}/{h_max:.3f}({h_lin/h_max*100:.0f}%) "
                          f"H_ang={h_ang:.3f}/{h_max:.3f}({h_ang/h_max*100:.0f}%) "
                          f"H_total={h_lin+h_ang:.3f}/{2*h_max:.3f}({(h_lin+h_ang)/(2*h_max)*100:.0f}%)")

                    # 6 envs 的 argmax 一覽
                    all_logits = net_output  # [N, 38]
                    all_argmax_lin = all_logits[:, :19].argmax(dim=1)
                    all_argmax_ang = all_logits[:, 19:].argmax(dim=1)
                    print(f"  [LOGIT]   all envs argmax: "
                          f"lin={all_argmax_lin.tolist()} ang={all_argmax_ang.tolist()}")
                except Exception as e:
                    print(f"  [LOGIT] error: {e}")
                    import traceback; traceback.print_exc()

            if diag:
                _logit_diagnostic(obs, label="Step 0 (after reset)")

            for step in range(args_cli.max_steps):
                with torch.no_grad():
                    actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                obs, reward, terminated, truncated, infos = env.step(actions)
                global_step += 1
                _count_episodes(raw_env, counts)

                # ── 診斷數據收集 ──
                try:
                    robot = raw_env.scene["robot"]
                    robot_vel = robot.data.root_lin_vel_w[:, :2]
                    speed = torch.norm(robot_vel, dim=-1)
                    _diag_speed_sum += speed.mean().item()
                    _diag_step_count += 1

                    # Action 分析: MultiDiscrete [19,19], center = [9,9]
                    if actions.dim() == 1:
                        # flat index → [lin, ang]
                        act_lin = actions // 19
                        act_ang = actions % 19
                    else:
                        act_lin = actions[:, 0] if actions.shape[-1] >= 2 else actions[:, 0]
                        act_ang = actions[:, 1] if actions.shape[-1] >= 2 else actions[:, 0]
                    is_center = (act_lin == 9) & (act_ang == 9)
                    _diag_zero_act_count += is_center.sum().item()

                    # LiDAR d_safe
                    sensor = raw_env.scene.sensors["lidar"]
                    sensor_pos = sensor.data.pos_w[:, :2]
                    hits = sensor.data.ray_hits_w[:, :, :2]
                    dists = torch.norm(hits - sensor_pos.unsqueeze(1), dim=-1)
                    dists = torch.nan_to_num(dists, nan=20.0, posinf=20.0)
                    d_safe_min = dists.min(dim=1).values.mean().item()
                except Exception:
                    d_safe_min = -1.0

                # ── --diagnostic: 每步 env[0] 詳細 ──
                if diag and step < 200:
                    try:
                        o = obs[0] if obs.dim() > 1 else obs
                        lidar_slice = o[6:78]
                        obs_slice = o[78:138].reshape(10, 6)
                        obs_nonzero = (obs_slice.abs().sum(dim=1) > 0.01).sum().item()
                        lin_i = act_lin[0].item() if act_lin.dim() > 0 else act_lin.item()
                        ang_i = act_ang[0].item() if act_ang.dim() > 0 else act_ang.item()

                        # CADN-normalized obs for env[0]
                        cadn_info = ""
                        if args_cli.use_cadn:
                            try:
                                cadn_mod = runner.agent._state_preprocessor
                                norm_obs = cadn_mod(obs[:1])
                                n = norm_obs[0]
                                cadn_info = (
                                    f" | CADN: state=[{n[0]:.1f},{n[1]:.1f},{n[2]:.1f},{n[3]:.1f}]"
                                    f" goal=[{n[4]:.1f},{n[5]:.1f}]"
                                    f" lidar=[{n[6:78].min():.1f},{n[6:78].mean():.1f},{n[6:78].max():.1f}]"
                                )
                            except Exception:
                                pass

                        # Policy entropy (confidence measure)
                        entropy_info = ""
                        try:
                            ent = runner.agent.policy.get_entropy()
                            if ent.numel() > 0:
                                entropy_info = f" | H={ent[0].item():.2f}"
                        except Exception:
                            pass

                        # Per-env0 d_safe
                        d0 = dists[0].min().item() if dists.shape[0] > 0 else -1.0

                        print(
                            f"  [DIAG] step={step:>4} "
                            f"lidar: min={lidar_slice.min():.3f} mean={lidar_slice.mean():.3f} | "
                            f"obs_slots={obs_nonzero}/10 | "
                            f"act=[{lin_i},{ang_i}] | "
                            f"speed={speed[0]:.3f} d0_safe={d0:.2f}"
                            f"{cadn_info}{entropy_info}"
                        )
                        # 每 10 步: 障礙物世界位置 vs obs 中位置 (驗證 obs 正確性)
                        if step % 10 == 0:
                            try:
                                rob = raw_env.scene["robot"]
                                rob_pos = rob.data.root_pos_w[0, :2]
                                rob_quat = rob.data.root_quat_w[0]
                                env_orig = raw_env.scene.env_origins[0, :2]
                                rob_local = rob_pos - env_orig

                                # Top-3 nearest obstacle world pos
                                near_obs_info = []
                                for oi in range(min(25, 100)):
                                    try:
                                        oe = raw_env.scene[f"obstacle_{oi}"]
                                        op = oe.data.root_pos_w[0]
                                        if op[2] > 0:  # visible
                                            dist_to_rob = torch.norm(op[:2] - rob_pos).item()
                                            near_obs_info.append((oi, dist_to_rob, op[:2].tolist()))
                                    except (KeyError, IndexError):
                                        break
                                near_obs_info.sort(key=lambda x: x[1])

                                # obs 中前 3 slot 的 body-frame pos
                                obs_bf = obs_slice[:3]  # [3, 6]: px,py,vx,vy,r,m
                                obs_str = " | ".join(
                                    f"({obs_bf[j,0]:.2f},{obs_bf[j,1]:.2f} r={obs_bf[j,4]:.2f} m={obs_bf[j,5]:.0f})"
                                    for j in range(3)
                                )

                                near_str = " | ".join(
                                    f"obs_{n[0]}:d={n[1]:.1f}m@({n[2][0]:.1f},{n[2][1]:.1f})"
                                    for n in near_obs_info[:3]
                                )

                                # Termination check
                                term_info = ""
                                try:
                                    tm = raw_env.termination_manager
                                    for tn in tm._term_names:
                                        buf = tm.get_term(tn)
                                        if buf is not None and buf[0].item():
                                            term_info = f" TERM={tn}"
                                except Exception:
                                    pass

                                print(
                                    f"  [OBS-CHECK] step={step} "
                                    f"robot=({rob_local[0]:.1f},{rob_local[1]:.1f}) "
                                    f"near=[{near_str}] "
                                    f"obs_bf=[{obs_str}]{term_info}"
                                )
                            except Exception as e2:
                                if step == 0:
                                    print(f"  [OBS-CHECK] error: {e2}")

                    except Exception as e:
                        if step == 0:
                            print(f"  [DIAG] error: {e}")

                # ── 每 50 步摘要 ──
                if (step + 1) % 50 == 0:
                    elapsed = time.time() - global_start
                    total_ep = sum(counts.values())
                    sr = counts["success"] / total_ep * 100 if total_ep > 0 else 0
                    cr = counts["collision"] / total_ep * 100 if total_ep > 0 else 0
                    tr = counts["timeout"] / total_ep * 100 if total_ep > 0 else 0
                    avg_spd = _diag_speed_sum / max(_diag_step_count, 1)
                    total_acts = _diag_step_count * args_cli.num_envs
                    zero_pct = _diag_zero_act_count / max(total_acts, 1) * 100
                    print(
                        f"  Step {step+1:>5} | {elapsed:.0f}s | "
                        f"Ep={total_ep} SR={sr:.0f}% CR={cr:.0f}% TO={tr:.0f}% | "
                        f"speed={avg_spd:.2f} d_safe={d_safe_min:.2f} | "
                        f"zero_act={zero_pct:.0f}%"
                    )
                    if diag:
                        _logit_diagnostic(obs, label=f"Step {step+1}")

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")

        # Fixed mode summary
        total_elapsed = time.time() - global_start
        total_ep = sum(counts.values())
        print(f"\n{'='*70}")
        print(f"  PLAY SUMMARY — Fixed Mode")
        print(f"{'='*70}")
        print(f"  Static obs:  {args_cli.num_static}")
        print(f"  Dynamic obs: {args_cli.num_dynamic}")
        print(f"  Walls:       {args_cli.num_walls}")
        print(f"  Duration:    {total_elapsed:.1f}s")
        print(f"  Steps:       {global_step}")
        print(f"  Episodes:    {total_ep}")
        if total_ep > 0:
            print(f"  Success:     {counts['success']} ({counts['success']/total_ep*100:.1f}%)")
            print(f"  Collision:   {counts['collision']} ({counts['collision']/total_ep*100:.1f}%)")
            print(f"  Timeout:     {counts['timeout']} ({counts['timeout']/total_ep*100:.1f}%)")
        print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
