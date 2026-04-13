# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train Charge navigation RL agent with SKRL — Standard AC (Actor-Critic).

與 train_charge.py (AAC) 的差異：
  - 使用標準 SkrlVecEnvWrapper（不使用 AACIsaacLabWrapper）
  - 不做 PPO monkey-patching（不注入 shared_states）
  - Critic 直接使用 policy observation（不經過 AAC wrapper 的 state_space）
  - 用於與 AAC 版本做 ablation 比較

觀測設計（v2 139D）：
  - Policy: 139D = ego(4) + goal(2) + static(72) + obs(60) + time(1)
  - Critic: 139D（與 Policy 相同，標準 AC 對稱設計）

使用方法:
    PYTHONUNBUFFERED=1 /home/aa/miniconda3/envs/env_isaaclab/bin/python \\
        scripts/reinforcement_learning/skrl/train_charge_ac.py \\
        --task Isaac-Navigation-Charge-VLP16 \\
        --num_envs 256 \\
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import AppLauncher first
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train Charge navigation RL agent with SKRL (Standard AC).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="skrl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations (legacy, prefer --timesteps).")
parser.add_argument("--timesteps", type=int, default=None, help="Override YAML trainer.timesteps directly. 1 timestep = 1 env.step() call.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--print_summary_every",
    type=int,
    default=0,
    help="Print training summary every N timesteps. Default: auto (min(1000, total_timesteps/10)). Set to 0 to disable.",
)
parser.add_argument(
    "--phase",
    type=int,
    default=1,
    choices=[1, 2],
    help="Training phase: 1=from scratch, 2=fine-tune from checkpoint (requires --checkpoint).",
)

# --- 實驗命名 ---
parser.add_argument("--run_name", type=str, default=None,
                    help="Custom run name for log dir / wandb (e.g. abl1_vgate-floor_s1_ne512)")

# --- 消融實驗 CLI ---
parser.add_argument("--v_gate_mode", type=str, default="baseline",
                    choices=["baseline", "floor", "softer"],
                    help="v_gate ablation: baseline / floor(min=0.2) / softer(d_att=0.6)")
parser.add_argument("--progress_gate_mode", type=str, default="baseline",
                    choices=["baseline", "delayed_negative", "weaken_negative"],
                    help="safe_progress gate: baseline / delayed(d_danger=0.55) / weaken(neg=-0.2)")
parser.add_argument("--use_gap_reward", action="store_true", default=False,
                    help="Enable gap-seeking rewards (heading_to_gap / forward_clearance)")
parser.add_argument("--gap_reward_type", type=str, default="heading",
                    choices=["heading", "clearance", "both"],
                    help="Gap reward type")
parser.add_argument("--gap_reward_weight", type=float, default=5.0,
                    help="Gap reward weight")
# --- Directional Gate (v16) ---
parser.add_argument("--directional_gate", action="store_true", default=False,
                    help="v16: 方向性 gate — 只看目標方向 cone 的 LiDAR，側面/後方障礙不壓制 goal attraction")
parser.add_argument("--gate_cone_half_bins", type=int, default=6,
                    help="Directional gate cone 半寬 (bins)。6=±30° (5°/bin)")
parser.add_argument("--gate_cone_bottom_k", type=int, default=3,
                    help="Directional gate cone 內取最近 k 條 ray 平均")
parser.add_argument("--gate_omni_blend", type=float, default=0.2,
                    help="Directional gate: 混合 omnidirectional 信號比例 (0=純方向性, 1=退化為全局)")

parser.add_argument("--use_safety_shield", action="store_true", default=False,
                    help="Enable safety shield on actions (speed limiting near obstacles)")
parser.add_argument("--shield_mode", type=str, default="soft",
                    choices=["soft", "hard"],
                    help="Shield mode: soft(linear reduction) / hard(force stop)")

# --- CADN observation preprocessor ---
parser.add_argument("--use_cadn", action="store_true", default=False,
                    help="Replace state preprocessor with CADN (per-branch dual-rate EMA normalizer)")

# --- NavRL-Ground v1 reward mode ---
parser.add_argument("--reward_mode", type=str, default="current",
                    choices=["current", "navrl_ground_v1", "navrl_ground_v2", "navrl_ground_v3", "navrl_ground_v4", "navrl_ground_v5", "navrl_ground_v6", "navrl_ground_v7", "navrl_ground_v8"],
                    help="Reward mode: current / v1-v7 / v8(v7+safety降低+vel提高)")
parser.add_argument("--dynamic_safety_mode", type=str, default="log_distance",
                    choices=["log_distance", "closing_risk"],
                    help="Dynamic safety reward mode")
parser.add_argument("--goal_vel_gate_beta", type=float, default=0.2,
                    help="Goal velocity soft gate floor (beta)")
parser.add_argument("--progress_scale_gamma", type=float, default=0.3,
                    help="Progress soft scale floor (gamma)")
parser.add_argument("--w_goal", type=float, default=500.0, help="Goal terminal reward weight")
parser.add_argument("--w_vel", type=float, default=10.0, help="Goal velocity reward weight")
parser.add_argument("--w_prog", type=float, default=12.0, help="Progress reward weight")
parser.add_argument("--w_ss", type=float, default=3.0, help="Static safety reward weight")
parser.add_argument("--w_ds", type=float, default=4.0, help="Dynamic safety reward weight")
parser.add_argument("--w_smooth", type=float, default=-0.05, help="Smoothness penalty weight")
parser.add_argument("--w_time", type=float, default=-0.1, help="Time penalty weight")
parser.add_argument("--w_collision", type=float, default=-100.0, help="Collision penalty weight")
parser.add_argument("--w_alive", type=float, default=0.2, help="Alive reward weight (v2 only)")
parser.add_argument("--goal_vel_use_soft_gate", action="store_true", default=False,
                    help="Enable soft gate on goal_velocity (v2 default=off, v1 style=on)")

# --- Curriculum version ---
parser.add_argument("--curriculum_version", type=str, default=None,
                    choices=["baseline_v1", "goal_first_v1", "goal_first_v2", "goal_first_v3", "open_ended_v1", "navrl_hybrid", "navrl_hybrid_dense"],
                    help="Curriculum version (default: use task config's baseline_v1)")
parser.add_argument("--no_walls", action="store_true", default=False,
                    help="移除所有內牆（保留外牆），所有 stage 的 min/max_walls=0")
parser.add_argument("--no_domain_randomization", action="store_true", default=False,
                    help="關閉所有 domain randomization (physics, sensor noise, external force, "
                         "robot 初始位置/速度隨機化)。Obstacle/goal layout 不受影響。")
parser.add_argument("--lidar_no_noise", action="store_true", default=False,
                    help="關閉 LiDAR 觀測函數內建的所有合成噪聲 (displacement_std, hole_rate, "
                         "distractor_rate, Unoise)。用於訓練「真正無噪聲」對照組。"
                         "搭配 --no_domain_randomization 使用以獲得 zero-noise baseline。")
parser.add_argument("--reward_speed_v05", action="store_true", default=False,
                    help="調整 reward 鼓勵 speed=0.5 m/s 目標 (方案 A): "
                         "goal_velocity ×3, static_safety ×0.375, time_penalty -0.2→-0.6。"
                         "解決 v20 出現的 over-cautious slow 行為 (speed=0.06)。")

# --- M1.4 ds rebalance: 修 v05 monkey-patch 漏洞 + 提高 closing_risk 訊號強度 ---
parser.add_argument("--ds_weight_boost", type=float, default=1.0,
                    help="OE 階段 dynamic_safety reward weight 倍率 (M1.4)。"
                         "預設 1.0=不變。建議 3.0 補回 v05 monkey-patch 漏掉的 ds 同步 boost。"
                         "v21 後實測：v05 把 goal_velocity ×3 但沒調 ds → ds effective 只佔 0.31% of goal_velocity。")
parser.add_argument("--risk_sigma", type=float, default=None,
                    help="dynamic_safety closing_risk 的 sigma (m)。預設 None = 使用 config 值。"
                         "建議 1.2 (vs V3 的 2.0) → 1m clearance risk 從 60% → 43%，更聚焦近距離。")
parser.add_argument("--b_risk", type=float, default=None,
                    help="dynamic_safety closing_risk 內部的 b_risk 倍率。預設 None = 使用 config 值。"
                         "建議 2.0 (vs V3 的 1.0) → closing_rate 訊號 2x。")

# --- Ablation Family 5: v8 safety balance ---
parser.add_argument("--ss_lower_mode", type=str, default=None,
                    choices=["aggressive", "moderate"],
                    help="Ablation 5: 進一步降低 safety 權重 (aggressive: ss-40%, moderate: ss-20%)")
parser.add_argument("--ss_raise_mode", action="store_true", default=False,
                    help="Ablation 5: 提高 safety 權重 (ss+40%)")
parser.add_argument("--model_variant", type=str, default="maxpool",
                    choices=["maxpool", "attention", "heterogeneous"],
                    help="obstacle encoder variant. 'maxpool' (v2), 'attention' (v24 dynamic-only), 'heterogeneous' (v25 HEIGHT full: LiDAR tokens + dynamic + type/angular emb)")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Phase 2 validation: require --checkpoint
if args_cli.phase == 2:
    if args_cli.checkpoint is None:
        parser.error("--phase 2 requires --checkpoint to specify Phase 1 weights")
    print(f"\n{'='*70}")
    print(f"  Phase 2: Obstacle-Avoidance Fine-Tuning")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Task: {args_cli.task}")
    print(f"{'='*70}\n")

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Determine headless mode
headless_mode = "--headless" in sys.argv or getattr(args_cli, "headless", False)

# Configure WandB based on mode
if headless_mode:
    os.environ["WANDB_PROJECT"] = "charge_skrl"
    print("[INFO] WandB logging enabled: project = charge_skrl")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print("[INFO] GUI mode detected: WandB logging disabled")

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import logging

import gymnasium as gym
import numpy as np
import torch
from packaging import version

import skrl

# Import SKRL utilities
from skrl.utils.runner.torch import Runner

# Import IsaacLab utilities
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Import custom WandB-enabled trainer and console summary logger
import sys
sys.path.insert(0, str(Path(__file__).parent))
from wandb_trainer import WandBSequentialTrainer
from console_summary import ConsoleSummaryLogger
from training_params_logger import dump_training_params

# Registry for custom SKRL model modules (resolved in patched _generate_models)
import vlp16_models as _vlp16_models_module
import charge_models as _charge_models_module
_CUSTOM_MODEL_MODULES = {
    "charge_models": _charge_models_module,
    "vlp16_models": _vlp16_models_module,
}

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import logger
logger = logging.getLogger(__name__)


# ============================================================================
# Patch Runner._generate_models for custom model class resolution ONLY
# (NO AAC patches — this is the key difference from train_charge.py)
# ============================================================================

def patch_runner_for_custom_models():
    """Patch Runner._generate_models to support custom model classes.

    This is a MINIMAL patch — only resolves dotted model class names
    (e.g., "vlp16_models.VLP16DiscretePolicy") to actual Python classes.
    No shared_states, no AAC, no PPO patches.
    """
    from skrl.utils.runner.torch import Runner
    from skrl.envs.wrappers.torch import MultiAgentEnvWrapper
    import copy

    original_generate_models = Runner._generate_models

    def patched_generate_models(self, env, cfg):
        """Patched _generate_models that resolves custom model classes."""
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]

        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' are defined in cfg")
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True

            if separate:
                for role in models_cfg:
                    model_class_name = models_cfg[role].get("class")
                    if not model_class_name:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]

                    # Resolve model class: built-in or custom (dotted name)
                    is_custom_model = False
                    if "." in model_class_name:
                        parts = model_class_name.split(".")
                        module = _CUSTOM_MODEL_MODULES.get(parts[0])
                        if module is not None:
                            model_class = getattr(module, parts[1])
                            is_custom_model = True
                        else:
                            raise ValueError(
                                f"Custom model module '{parts[0]}' not found in _CUSTOM_MODEL_MODULES. "
                                f"Available: {list(_CUSTOM_MODEL_MODULES.keys())}"
                            )
                    else:
                        model_class = self._component(model_class_name)

                    # Standard AC: both policy and value use observation_space
                    observation_space = observation_spaces[agent_id]

                    if is_custom_model:
                        models[agent_id][role] = model_class(
                            observation_space=observation_space,
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                        )
                    else:
                        source = model_class(
                            observation_space=observation_space,
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                            return_source=True,
                        )
                        models[agent_id][role] = model_class(
                            observation_space=observation_space,
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                        )
            else:
                roles = list(models_cfg.keys())
                if len(roles) != 2:
                    raise ValueError(
                        "Runner currently only supports shared models, made up of exactly two models. "
                        "Set 'separate' field to True to create non-shared models for the given cfg"
                    )
                structure = []
                parameters = []
                for role in roles:
                    model_structure = models_cfg[role].get("class")
                    if not model_structure:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]
                    structure.append(model_structure)
                    parameters.append(self._process_cfg(models_cfg[role]))
                model_class = self._component("Shared")

                source = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                    return_source=True,
                )
                models[agent_id] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                )

        return models

    Runner._generate_models = patched_generate_models
    print("[INFO] Runner patched for custom model resolution (Standard AC — no AAC patches)")

# Apply the minimal patch
patch_runner_for_custom_models()

# Check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()


# ============================================================================
# Helper Functions
# ============================================================================

def cleanup_pbar(*args):
    """Helper to stop training and cleanup progress bar properly on ctrl+c."""
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, cleanup_pbar)


def disable_debug_vis(cfg) -> None:
    """Recursively disable all debug_vis options in configuration."""
    if hasattr(cfg, "debug_vis"):
        cfg.debug_vis = False

    for attr_name in dir(cfg):
        if not attr_name.startswith("_"):
            try:
                attr = getattr(cfg, attr_name)
                if hasattr(attr, "__dataclass_fields__") or hasattr(attr, "__attrs_attrs__"):
                    disable_debug_vis(attr)
            except Exception:
                pass


# ============================================================================
# 消融實驗 CLI → env_cfg 覆蓋
# ============================================================================

def _apply_ablation_overrides(env_cfg, args_cli):
    """根據 CLI 消融參數覆蓋 env_cfg 的 reward/action 設定。

    所有參數預設值 = baseline 行為，不帶參數時不做任何修改。
    """
    rewards = getattr(env_cfg, 'rewards', None)
    if rewards is None:
        return

    changed = False

    # --- v_gate_mode ---
    if args_cli.v_gate_mode != "baseline":
        vt = getattr(rewards, 'velocity_to_goal', None)
        if vt is not None:
            if args_cli.v_gate_mode == "floor":
                vt.params["v_gate_floor"] = 0.2
            elif args_cli.v_gate_mode == "softer":
                vt.params["d_attenuate"] = 0.6
            print(f"[ABLATION] v_gate_mode={args_cli.v_gate_mode}: {vt.params}")
            changed = True

    # --- progress_gate_mode ---
    if args_cli.progress_gate_mode != "baseline":
        sp = getattr(rewards, 'safe_progress', None)
        if sp is not None:
            if args_cli.progress_gate_mode == "delayed_negative":
                sp.params["d_danger"] = 0.55
            elif args_cli.progress_gate_mode == "weaken_negative":
                sp.params["negative_scale"] = 0.2
            print(f"[ABLATION] progress_gate_mode={args_cli.progress_gate_mode}: {sp.params}")
            changed = True

    # --- gap reward ---
    if args_cli.use_gap_reward:
        from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.rewards.gap_rewards import (
            heading_to_gap_reward, forward_clearance_improvement_reward,
        )
        BODY_R = 0.35
        w = args_cli.gap_reward_weight
        if args_cli.gap_reward_type in ("heading", "both"):
            rewards.heading_to_gap = RewTerm(
                func=heading_to_gap_reward,
                params={"robot_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("lidar"),
                        "body_radius": BODY_R, "min_gap_width": 0.9,
                        "activation_d_safe": 2.0, "speed_threshold": 0.05},
                weight=w,
            )
        if args_cli.gap_reward_type in ("clearance", "both"):
            rewards.forward_clearance = RewTerm(
                func=forward_clearance_improvement_reward,
                params={"robot_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("lidar"),
                        "body_radius": BODY_R, "front_arc_bins": 12, "activation_d_safe": 2.0},
                weight=w,
            )
        print(f"[ABLATION] gap_reward: type={args_cli.gap_reward_type} weight={w}")
        changed = True

    # --- safety shield ---
    if args_cli.use_safety_shield:
        import math as _math
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.actions.safety_shield import (
            ShieldedDiscreteDifferentialDriveActionCfg,
        )
        from isaaclab.utils import configclass as _configclass
        BODY_R = 0.35
        mode = args_cli.shield_mode

        @_configclass
        class _ShieldedActions:
            diff_drive = ShieldedDiscreteDifferentialDriveActionCfg(
                asset_name="robot", debug_vis=True, num_bins=19,
                max_linear_velocity=1.0, max_linear_accel=0.5,
                max_angular_vel=0.25 * _math.pi,
                shield_mode=mode,
                shield_d_danger=0.55, shield_d_safe=1.2,
                sensor_name="lidar", body_radius=BODY_R,
            )
        env_cfg.actions = _ShieldedActions()
        print(f"[ABLATION] safety_shield: mode={mode}")
        changed = True

    # --- directional gate (通用：不依賴 reward_mode) ---
    if args_cli.directional_gate:
        _dir_params = {
            "cone_half_bins": args_cli.gate_cone_half_bins,
            "cone_bottom_k": args_cli.gate_cone_bottom_k,
            "omni_blend": args_cli.gate_omni_blend,
        }
        gv = getattr(rewards, 'goal_velocity', None)
        if gv is not None:
            gv.params["directional_gate"] = True
            gv.params.update(_dir_params)
        gp = getattr(rewards, 'goal_progress', None)
        if gp is not None:
            gp.params["directional_scale"] = True
            gp.params.update(_dir_params)
        if gv or gp:
            print(f"[ABLATION] directional_gate: cone=±{args_cli.gate_cone_half_bins}bins "
                  f"bottom_k={args_cli.gate_cone_bottom_k} blend={args_cli.gate_omni_blend}")
            changed = True

    # --- reward_mode: navrl_ground_v1 ---
    if getattr(args_cli, 'reward_mode', 'current') == "navrl_ground_v1":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGround,
        )
        env_cfg.rewards = RewardsCfgVLP16NavRLGround()

        # 套用 CLI 權重覆蓋
        r = env_cfg.rewards
        r.reaching_goal.weight = args_cli.w_goal
        r.goal_velocity.weight = args_cli.w_vel
        r.goal_progress.weight = args_cli.w_prog
        r.static_safety.weight = args_cli.w_ss
        r.dynamic_safety.weight = args_cli.w_ds
        r.smoothness.weight = args_cli.w_smooth
        r.time_penalty.weight = args_cli.w_time
        r.collision_ground.weight = args_cli.w_collision

        # 套用 CLI 參數覆蓋
        r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
        r.goal_progress.params["scale_gamma"] = args_cli.progress_scale_gamma
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode

        # 方向性 gate 覆蓋
        if args_cli.directional_gate:
            for term_name, dir_key in [("goal_velocity", "directional_gate"), ("goal_progress", "directional_scale")]:
                term = getattr(r, term_name, None)
                if term is not None:
                    term.params[dir_key] = True
                    term.params["cone_half_bins"] = args_cli.gate_cone_half_bins
                    term.params["cone_bottom_k"] = args_cli.gate_cone_bottom_k
                    term.params["omni_blend"] = args_cli.gate_omni_blend

        dir_tag = ""
        if args_cli.directional_gate:
            dir_tag = (f" | dir_gate: cone=±{args_cli.gate_cone_half_bins}bins "
                       f"bottom_k={args_cli.gate_cone_bottom_k} blend={args_cli.gate_omni_blend}")

        print(
            f"[REWARD_MODE] navrl_ground_v1 | "
            f"w: goal={args_cli.w_goal} vel={args_cli.w_vel} prog={args_cli.w_prog} "
            f"ss={args_cli.w_ss} ds={args_cli.w_ds} smooth={args_cli.w_smooth} "
            f"time={args_cli.w_time} collision={args_cli.w_collision} | "
            f"beta={args_cli.goal_vel_gate_beta} gamma={args_cli.progress_scale_gamma} "
            f"ds_mode={args_cli.dynamic_safety_mode}{dir_tag}"
        )
        changed = True

    # --- reward_mode: navrl_ground_v2 (no gate + alive reward) ---
    if getattr(args_cli, 'reward_mode', 'current') == "navrl_ground_v2":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV2,
        )
        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV2()

        # 套用 CLI 覆蓋
        r = env_cfg.rewards
        r.alive.weight = args_cli.w_alive
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode

        # gate 消融：v2 預設 False，--goal_vel_use_soft_gate 可恢復 gate
        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
            print(f"[ABLATION] goal_velocity: soft_gate ON, beta={args_cli.goal_vel_gate_beta}")

        print(
            f"[REWARD_MODE] navrl_ground_v2 (no gate + alive) | "
            f"w: alive={args_cli.w_alive} vel=2.0 prog=3.0 ss=2.0 ds=2.0 "
            f"smooth=-0.1 goal=100 collision=-50 | "
            f"ds_mode={args_cli.dynamic_safety_mode}"
        )
        changed = True

    # --- reward_mode: navrl_ground_v3 (v2 + 20 obstacles + density weights) ---
    if getattr(args_cli, 'reward_mode', 'current') == "navrl_ground_v3":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV3,
        )
        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV3()

        r = env_cfg.rewards
        r.alive.weight = args_cli.w_alive
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode

        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta

        print(
            f"[REWARD_MODE] navrl_ground_v3 (v2 + 20obs + density weights) | "
            f"w: alive={args_cli.w_alive} | "
            f"ds_mode={args_cli.dynamic_safety_mode} | "
            f"weights controlled by curriculum stage"
        )
        changed = True

    # --- reward_mode: navrl_ground_v4 (v3 + goal=500 + alive→alignment) ---
    if getattr(args_cli, 'reward_mode', 'current') == "navrl_ground_v4":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV3,
        )
        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV3()

        r = env_cfg.rewards
        # v4 核心修正:
        # 1. reaching_goal=500 (已在 V3 class 中設定)
        # 2. alive=0 (移除存活獎勵，r_vel + γ折扣已足夠)
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode

        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta

        print(
            f"[REWARD_MODE] navrl_ground_v4 (v3 + goal=500 + no alive) | "
            f"w: reaching_goal={r.reaching_goal.weight} alive={r.alive.weight} | "
            f"ds_mode={args_cli.dynamic_safety_mode} | "
            f"weights controlled by curriculum stage"
        )
        changed = True

    # --- reward_mode: navrl_ground_v5 (collision=-100 + goal_first_v3 ss_boost Stage 3-6) ---
    if getattr(args_cli, 'reward_mode', 'current') == "navrl_ground_v5":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV5,
        )
        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV5()

        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode

        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta

        print(
            f"[REWARD_MODE] navrl_ground_v5 (collision=-100 + ss_boost Stage 3-6) | "
            f"w: reaching_goal={r.reaching_goal.weight} "
            f"collision={r.collision_ground.weight} alive={r.alive.weight} | "
            f"ds_mode={args_cli.dynamic_safety_mode} | "
            f"搭配 goal_first_v3 使用"
        )
        changed = True

    # --- reward_mode: navrl_ground_v6 (open-ended, front_block=0.4, collision=-50) ---
    if getattr(args_cli, 'reward_mode', 'current') == "navrl_ground_v6":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV6,
        )
        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV6()

        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode

        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta

        print(
            f"[REWARD_MODE] navrl_ground_v6 (open-ended, front_block=0.4) | "
            f"w: goal={r.reaching_goal.weight} coll={r.collision_ground.weight} "
            f"alive={r.alive.weight} ss_front={r.static_safety.params['a_front_block']} | "
            f"ds_mode={args_cli.dynamic_safety_mode}"
        )
        changed = True

    # --- reward_mode: navrl_ground_v7 (v6 + 碰撞成本遞增, collision=-5 基礎) ---
    if getattr(args_cli, 'reward_mode', 'current') == "navrl_ground_v7":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV7,
        )
        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV7()

        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode

        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta

        print(
            f"[REWARD_MODE] navrl_ground_v7 (碰撞遞增: -5→-80) | "
            f"w: goal={r.reaching_goal.weight} coll_base={r.collision_ground.weight} "
            f"alive={r.alive.weight} ss_front={r.static_safety.params['a_front_block']} | "
            f"ds_mode={args_cli.dynamic_safety_mode} | "
            f"collision 由 curriculum 動態調整"
        )
        changed = True

    # --- reward_mode: navrl_ground_v8 (v7 + safety降低 + vel提高) ---
    if getattr(args_cli, 'reward_mode', 'current') == "navrl_ground_v8":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV8,
        )
        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV8()

        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode

        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta

        print(
            f"[REWARD_MODE] navrl_ground_v8 (safety↓ vel↑ 修正比例失衡) | "
            f"w: goal={r.reaching_goal.weight} coll_base={r.collision_ground.weight} | "
            f"ss/ds weight 由 curriculum 控制 (上限 ss=0.5 ds=0.4) | "
            f"vel 下限=3.5"
        )
        changed = True

    # --- curriculum_version ---
    cv = getattr(args_cli, 'curriculum_version', None)
    if cv is not None:
        cur = getattr(env_cfg, 'curriculum', None)
        if cur is not None:
            term = getattr(cur, 'goal_obstacle_curriculum', None)
            if term is not None:
                term.params["curriculum_version"] = cv
                print(f"[CURRICULUM] version={cv}")
                changed = True

    # --- no_walls: 強制所有 stage 的內牆為 0 ---
    if getattr(args_cli, 'no_walls', False):
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            CURRICULUM_CONFIGS,
        )
        cv_key = cv or "baseline_v1"
        if cv_key in CURRICULUM_CONFIGS:
            for stage_cfg in CURRICULUM_CONFIGS[cv_key]["stages"]:
                stage_cfg["min_walls"] = 0
                stage_cfg["max_walls"] = 0
            print(f"[NO_WALLS] 所有 stage 的 min/max_walls 已設為 0（保留外牆）")
            changed = True

    # --- no_domain_randomization: 只關閉 physics/sensor/force DR event ---
    # 注意: 不動 reset_base 的 pose/velocity range —— 那是 scene initialization 必須的
    # spread，把它收成 (0,0) 會讓所有 robot 在同一點 spawn 並導致物理問題（v18 教訓）
    if getattr(args_cli, 'no_domain_randomization', False):
        events = getattr(env_cfg, 'events', None)
        if events is not None:
            dr = getattr(events, 'domain_randomization', None)
            if dr is not None:
                dr.params["enable_physics"] = False
                dr.params["enable_sensor_noise"] = False
                dr.params["enable_external_force"] = False
                print(f"[NO_DR] domain_randomization event: physics/sensor_noise/external_force = False")
                print(f"[NO_DR] reset_base 保留原樣（pose/velocity range 不動）")
        changed = True

    # --- reward_speed_v05: 方案 A reward 調整鼓勵 v=0.5 m/s ---
    # v20 訓練後發現 agent settled at speed=0.06 m/s，原因:
    # 1. goal_velocity weight 太低 (4.5)
    # 2. static_safety log clearance 太強 (0.4) — 慢速 + 遠離障礙物 = 大正獎勵
    # 3. time_penalty 太弱 (-0.2) — 慢速沒有顯著代價
    # 修復: ×3 goal_velocity, ×0.375 static_safety, time_penalty -0.6
    if getattr(args_cli, 'reward_speed_v05', False):
        # 1) 修改全域 time_penalty
        rewards = getattr(env_cfg, 'rewards', None)
        if rewards is not None and hasattr(rewards, 'time_penalty'):
            rewards.time_penalty.weight = -0.6
            print("[REWARD_SPEED_V05] time_penalty weight: -0.2 → -0.6")

        # 2) 修改 bootstrap stages 的 reward_weights
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            CURRICULUM_CONFIGS,
        )
        cv_key = cv or "open_ended_v1"
        if cv_key in CURRICULUM_CONFIGS:
            stages = CURRICULUM_CONFIGS[cv_key]["stages"]
            for stage_cfg in stages:
                rw = stage_cfg.get("reward_weights")
                if rw is None:
                    continue
                if "goal_velocity" in rw:
                    rw["goal_velocity"] = round(rw["goal_velocity"] * 3.0, 2)
                if "static_safety" in rw:
                    rw["static_safety"] = round(rw["static_safety"] * 0.375, 3)
            print(f"[REWARD_SPEED_V05] {len(stages)} bootstrap stages: "
                  f"goal_velocity ×3, static_safety ×0.375")

        # 3) Monkey-patch _open_ended_reward_weights for OE stages
        # 注意: 不能用 `import ... as _curr_mod`，因為 module 與內部函數同名衝突
        # 改用 importlib.import_module() 強制取得 module 物件
        import importlib
        _curr_mod = importlib.import_module(
            "isaaclab_tasks.manager_based.locomotion.velocity.config."
            "charge_skrl.curriculum.goal_obstacle_curriculum"
        )
        _orig_oe_weights = _curr_mod._open_ended_reward_weights

        def _patched_oe_weights(level: int) -> dict:
            rw = _orig_oe_weights(level)
            rw["goal_velocity"] = round(rw.get("goal_velocity", 4.0) * 3.0, 2)
            rw["static_safety"] = round(rw.get("static_safety", 0.4) * 0.375, 3)
            return rw

        _curr_mod._open_ended_reward_weights = _patched_oe_weights
        print(f"[REWARD_SPEED_V05] OE stages: _open_ended_reward_weights monkey-patched")
        print(f"[REWARD_SPEED_V05] 目標 speed ≈ 0.4-0.6 m/s, 預期 SR ≈ 88-92%")
        changed = True

    # --- M1.4: ds rebalance (修 v05 monkey-patch 漏洞 + 提高 closing_risk 訊號) ---
    # 背景: v21 實測證實 closing_risk mode 已啟用 (line 731)，但 effective ds weight
    # 只佔 goal_velocity 的 0.31%，因為 v05 把 goal_velocity ×3 卻沒同步調 ds。
    # M1.4 修復: ds_weight_boost (OE weight scale) + risk_sigma (sharper field) +
    #            b_risk (stronger closing signal)。
    # 預期 effective ds weight 從 0.31% → ~3-5%，足以讓 PPO 學到 predictive avoidance。
    _need_ds_patch = (
        args_cli.ds_weight_boost != 1.0
        or args_cli.risk_sigma is not None
        or args_cli.b_risk is not None
    )
    if _need_ds_patch:
        rewards = getattr(env_cfg, 'rewards', None)
        if rewards is not None and hasattr(rewards, 'dynamic_safety'):
            r = rewards
            ds_params = r.dynamic_safety.params
            # 1) 套用 risk_sigma override
            if args_cli.risk_sigma is not None:
                old_sigma = ds_params.get("risk_sigma", "?")
                ds_params["risk_sigma"] = args_cli.risk_sigma
                print(f"[M1.4_DS] risk_sigma: {old_sigma} → {args_cli.risk_sigma}")
            # 2) 套用 b_risk override
            if args_cli.b_risk is not None:
                old_b = ds_params.get("b_risk", "?")
                ds_params["b_risk"] = args_cli.b_risk
                print(f"[M1.4_DS] b_risk: {old_b} → {args_cli.b_risk}")
            # 3) 套用 OE weight boost (chained on top of v05 patch if exists)
            if args_cli.ds_weight_boost != 1.0:
                import importlib as _il
                _curr_mod_m14 = _il.import_module(
                    "isaaclab_tasks.manager_based.locomotion.velocity.config."
                    "charge_skrl.curriculum.goal_obstacle_curriculum"
                )
                _orig_oe_for_ds = _curr_mod_m14._open_ended_reward_weights
                _ds_boost = args_cli.ds_weight_boost

                def _patched_oe_with_ds(level: int) -> dict:
                    rw = _orig_oe_for_ds(level)
                    rw["dynamic_safety"] = round(
                        rw.get("dynamic_safety", 0.4) * _ds_boost, 3
                    )
                    return rw

                _curr_mod_m14._open_ended_reward_weights = _patched_oe_with_ds
                print(f"[M1.4_DS] OE dynamic_safety weight boost ×{_ds_boost} (chained)")

                # 4) 也套用到 bootstrap stages (B5+, dynamic_safety > 0)
                from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
                    CURRICULUM_CONFIGS,
                )
                _cv_key = cv or "open_ended_v1"
                if _cv_key in CURRICULUM_CONFIGS:
                    _stages = CURRICULUM_CONFIGS[_cv_key]["stages"]
                    _patched_count = 0
                    for stage_cfg in _stages:
                        rw = stage_cfg.get("reward_weights")
                        if rw is None or "dynamic_safety" not in rw:
                            continue
                        if rw["dynamic_safety"] > 0:
                            rw["dynamic_safety"] = round(rw["dynamic_safety"] * _ds_boost, 3)
                            _patched_count += 1
                    print(f"[M1.4_DS] {_patched_count} bootstrap stages: dynamic_safety ×{_ds_boost}")
            print(f"[M1.4_DS] M1.4 ds rebalance applied. 預期 ds effective weight ↑ 4-10x")
            changed = True

    # --- lidar_no_noise: 關閉 LiDAR 觀測函數內建的所有 noise ---
    # 修復 Bug B: lidar_vlp16_to_2d_bins 預設帶 displacement_std=0.02 + hole_rate=0.005
    # + distractor_rate=0.002 + Unoise(±0.02)，導致 lidar.min 永遠 ≈ 0（因為每幀都有
    # ~11 個假近距離 distractor），policy 學不到正確的近距離 obstacle 訊號。
    if getattr(args_cli, 'lidar_no_noise', False):
        obs_root = getattr(env_cfg, 'observations', None)
        if obs_root is not None:
            cleared_terms = []
            # 遍歷所有 obs group (policy / critic / shared)
            for group_name in dir(obs_root):
                if group_name.startswith('_'):
                    continue
                group = getattr(obs_root, group_name, None)
                if group is None or not hasattr(group, '__dict__'):
                    continue
                for term_name in dir(group):
                    if term_name.startswith('_'):
                        continue
                    term = getattr(group, term_name, None)
                    if term is None or not hasattr(term, 'params'):
                        continue
                    # 只處理 LiDAR observation terms
                    if 'lidar' not in term_name.lower():
                        continue
                    params = term.params
                    if 'displacement_std' in params:
                        params['displacement_std'] = 0.0
                    if 'hole_rate' in params:
                        params['hole_rate'] = 0.0
                    if 'distractor_rate' in params:
                        params['distractor_rate'] = 0.0
                    # 移除 ObsTerm 上的 Unoise 包裝
                    if hasattr(term, 'noise'):
                        term.noise = None
                    cleared_terms.append(f"{group_name}.{term_name}")
            print(f"[NO_LIDAR_NOISE] LiDAR observation noise disabled: {cleared_terms}")
            print(f"[NO_LIDAR_NOISE] displacement_std=0, hole_rate=0, distractor_rate=0, Unoise=None")
        changed = True

    # --- Ablation 5: safety 權重調整 ---
    ss_lower = getattr(args_cli, 'ss_lower_mode', None)
    ss_raise = getattr(args_cli, 'ss_raise_mode', False)
    if ss_lower or ss_raise:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            set_ablation_params,
        )
        set_ablation_params(ss_lower_mode=ss_lower, ss_raise=ss_raise)
        mode_str = f"ss_lower={ss_lower}" if ss_lower else f"ss_raise={ss_raise}"
        print(f"[ABLATION_5] Safety 權重調整: {mode_str}")
        changed = True

    if not changed:
        print("[ABLATION] baseline (no overrides)")


# ============================================================================
# Main Training Function
# ============================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with SKRL agent for Charge navigation (Standard AC — no AAC)."""
    # Randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # Override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Set seed
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Override trainer timesteps (--timesteps takes priority over --max_iterations)
    if args_cli.timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.timesteps
        ppo_updates = args_cli.timesteps // agent_cfg["agent"]["rollouts"]
        total_env_steps = args_cli.timesteps * env_cfg.scene.num_envs
        print(f"[INFO] --timesteps={args_cli.timesteps} → {ppo_updates} PPO updates, {total_env_steps:,} total env steps")
    elif args_cli.max_iterations is not None:
        print(f"[WARNING] max_iterations={args_cli.max_iterations} is set, overriding timesteps!")
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"] * env_cfg.scene.num_envs

    # Set environment seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Directory for logging — use "ac" subdirectory to separate from AAC logs
    if args_cli.run_name:
        run_info = args_cli.run_name
    else:
        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", args_cli.task + "-AC"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"[INFO] Run name: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    # --- v24/v25: model variant override ---
    _mv = getattr(args_cli, "model_variant", "maxpool")
    if _mv in ("attention", "heterogeneous"):
        try:
            models_cfg = agent_cfg["models"]
            if _mv == "attention":
                policy_cls = "vlp16_models.VLP16AttentionPolicy"
                value_cls = "vlp16_models.VLP16AttentionValue"
                tag = "v24"
            else:  # heterogeneous
                policy_cls = "vlp16_models.VLP16HeterogeneousPolicy"
                value_cls = "vlp16_models.VLP16HeterogeneousValue"
                tag = "v25"
            for role in ("policy", "value"):
                if role in models_cfg and "class" in models_cfg[role]:
                    old = models_cfg[role]["class"]
                    if "VLP16DiscretePolicy" in old or "VLP16AttentionPolicy" in old:
                        models_cfg[role]["class"] = policy_cls
                    elif "VLP16Value" in old or "VLP16AttentionValue" in old:
                        models_cfg[role]["class"] = value_cls
                    print(f"[{tag}] models.{role}.class: {old} → {models_cfg[role]['class']}")
        except Exception as e:
            print(f"[model_variant] WARN: failed to override model class: {e}")

    # --- 消融實驗: CLI → env_cfg 覆蓋 (在 config dump 前，gym.make 前) ---
    _apply_ablation_overrides(env_cfg, args_cli)

    # Dump configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # Dump comprehensive training parameters (rewards, hyperparams, etc.)
    dump_training_params(
        log_dir=log_dir,
        run_name=run_info,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        args_cli=args_cli,
    )

    # Set IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. "
            "No IO descriptors will be exported."
        )

    # Set log directory for environment
    env_cfg.log_dir = log_dir

    # Disable debug visualization in headless mode
    if headless_mode:
        disable_debug_vis(env_cfg)
        print("[INFO] Headless mode: All debug_visualization disabled")

    # ========================================================================
    # Create Isaac Lab Environment
    # ========================================================================
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None
    )

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(Path(log_dir) / "videos" / "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ========================================================================
    # Wrap Environment for SKRL — Standard AC (SkrlVecEnvWrapper)
    # ========================================================================
    # 關鍵差異：使用標準 SkrlVecEnvWrapper 而非 AACIsaacLabWrapper
    # - 不注入 shared_states
    # - Critic 直接使用 policy 的 observation_space
    # - 不做 frame stacking（SkrlVecEnvWrapper 本身不提供）
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    print(f"[INFO] Environment wrapped with standard SkrlVecEnvWrapper (AC mode)")
    print(f"[INFO] observation_space: {env.observation_space}")
    print(f"[INFO] action_space: {env.action_space}")

    # ========================================================================
    # Configure SKRL Agent and Trainer
    # ========================================================================
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_info

    # Configure trainer
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # ========================================================================
    # Initialize WandB (only in headless mode)
    # ========================================================================
    wandb_run = None
    if headless_mode:
        try:
            import wandb
            wandb_name = args_cli.run_name if args_cli.run_name else f"AC_{run_info}"
            wandb.init(
                project="charge_skrl",
                name=wandb_name,
                config={
                    "task": args_cli.task,
                    "num_envs": env_cfg.scene.num_envs,
                    "agent": "PPO",
                    "architecture": "AC",
                    "seed": agent_cfg["seed"],
                    "ml_framework": args_cli.ml_framework,
                    "run_name": wandb_name,
                    # 消融參數
                    "v_gate_mode": args_cli.v_gate_mode,
                    "progress_gate_mode": args_cli.progress_gate_mode,
                    "use_gap_reward": args_cli.use_gap_reward,
                    "gap_reward_type": args_cli.gap_reward_type,
                    "gap_reward_weight": args_cli.gap_reward_weight,
                    "use_safety_shield": args_cli.use_safety_shield,
                    "shield_mode": args_cli.shield_mode,
                    # NavRL-Ground v1
                    "reward_mode": args_cli.reward_mode,
                    "dynamic_safety_mode": args_cli.dynamic_safety_mode,
                    "goal_vel_gate_beta": args_cli.goal_vel_gate_beta,
                    "progress_scale_gamma": args_cli.progress_scale_gamma,
                    "w_goal": args_cli.w_goal,
                    "w_vel": args_cli.w_vel,
                    "w_prog": args_cli.w_prog,
                    "w_ss": args_cli.w_ss,
                    "w_ds": args_cli.w_ds,
                    "w_smooth": args_cli.w_smooth,
                    "w_time": args_cli.w_time,
                    "w_collision": args_cli.w_collision,
                    "curriculum_version": args_cli.curriculum_version or "baseline_v1",
                    "w_alive": args_cli.w_alive,
                },
                tags=["AC", "ablation"],
            )
            wandb_run = wandb.run
            print(f"[INFO] WandB initialized: {wandb.run.name}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")

    # ========================================================================
    # Create SKRL Trainer and Train
    # ========================================================================
    start_time = time.time()

    # Console Summary Logger
    console_summary_logger = None
    print_summary_every = args_cli.print_summary_every

    if print_summary_every == 0:
        total_timesteps = agent_cfg['trainer']['timesteps']
        auto_interval = max(100, min(1000, total_timesteps // 10))
        print_summary_every = auto_interval
        print(f"[INFO] Console summary auto-enabled: printing every {print_summary_every} timesteps (total: {total_timesteps})")

    if print_summary_every > 0:
        console_summary_logger = ConsoleSummaryLogger(
            print_every=print_summary_every,
            task_name=args_cli.task,
            num_envs=env_cfg.scene.num_envs,
            headless=headless_mode,
            device=env_cfg.sim.device,
            seed=agent_cfg["seed"],
            algo="PPO-AC",
        )

    if wandb_run is not None:
        print("[INFO] Using WandB-enabled trainer for metrics logging")
        import copy

        trainer_cfg = copy.deepcopy(agent_cfg.get("trainer", {}))
        trainer_cfg["close_environment_at_exit"] = False

        # 使用 Runner 來建立 agent 和 models
        runner = Runner(env, agent_cfg)
        agents_scope = runner.trainer.agents_scope if hasattr(runner.trainer, 'agents_scope') else None

        # TrainingDebugLogger
        from training_debug_logger import TrainingDebugLogger
        debug_logger = TrainingDebugLogger(
            env=env,
            log_interval=agent_cfg.get("agent", {}).get("rollouts", 128),
        )
        print("[INFO] TrainingDebugLogger initialized")

        # WandB-enabled trainer
        trainer = WandBSequentialTrainer(
            env=env,
            agents=runner.agent,
            agents_scope=agents_scope,
            cfg=trainer_cfg,
            wandb_run=wandb_run,
            console_summary_logger=console_summary_logger,
            debug_logger=debug_logger,
            log_dir=log_dir,
        )
        print("[INFO] WandB-enabled trainer configured (Standard AC)")

        # Ablation metrics logger (NavRL0* tasks)
        task_name = args_cli.task
        if "NavRL0" in task_name or "NavRL" in task_name:
            try:
                from diagnostics.ablation_metrics import AblationMetricsLogger
                trainer.ablation_logger = AblationMetricsLogger(env)
                print(f"[INFO] AblationMetricsLogger enabled for task: {task_name}")
            except Exception as e:
                print(f"[WARN] AblationMetricsLogger failed to init: {e}")

            # Reward space logger (always enabled with ablation_logger)
            try:
                from diagnostics import RewardSpaceLogger
                # 使用訓練 run 的 logs 目錄
                log_dir = os.path.join(os.path.dirname(env.cfg.log_dir), "reward_space") if hasattr(env.cfg, "log_dir") else "logs/reward_space"
                trainer.reward_space_logger = RewardSpaceLogger(env, output_dir=log_dir)
                print(f"[INFO] RewardSpaceLogger enabled (CSV: {trainer.reward_space_logger._csv_path})")
            except Exception as e:
                print(f"[WARN] RewardSpaceLogger failed to init: {e}")
    else:
        runner = Runner(env, agent_cfg)
        trainer = None

    # --- CADN injection (after agent creation, before checkpoint load) ---
    if args_cli.use_cadn:
        from cadn_preprocessor import CurriculumAwareDualRateNormalizer
        obs_dim = env.observation_space.shape[-1] if hasattr(env.observation_space, "shape") else 139
        cadn = CurriculumAwareDualRateNormalizer(
            size=obs_dim,
            device=runner.agent.device,
        )
        runner.agent._state_preprocessor = cadn
        runner.agent.checkpoint_modules["state_preprocessor"] = cadn
        print(f"[INFO] CADN enabled: state_preprocessor replaced (obs_dim={obs_dim})")

    # Load checkpoint if specified
    checkpoint_agent = runner.agent
    if args_cli.checkpoint is not None:
        print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")
        if getattr(args_cli, "model_variant", "maxpool") == "attention":
            # v24: partial warm-start — skip attention_encoder.* keys (randomly init)
            import torch as _torch
            ckpt = _torch.load(args_cli.checkpoint, map_location=runner.agent.device, weights_only=False)
            loaded_modules = []
            for mod_name, mod in runner.agent.checkpoint_modules.items():
                if mod_name not in ckpt:
                    print(f"[v24 warm-start] skip '{mod_name}' (not in checkpoint)")
                    continue
                state = ckpt[mod_name]
                if not hasattr(mod, "load_state_dict"):
                    print(f"[v24 warm-start] skip '{mod_name}' (no load_state_dict)")
                    continue
                try:
                    result = mod.load_state_dict(state, strict=False)
                    print(
                        f"[v24 warm-start] '{mod_name}': loaded "
                        f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
                    )
                    loaded_modules.append(mod_name)
                except Exception as e:
                    print(f"[v24 warm-start] WARN '{mod_name}' load failed: {e}")
            print(f"[v24 warm-start] partial load complete. modules loaded: {loaded_modules}")
        else:
            checkpoint_agent.load(args_cli.checkpoint)

    # ── 訓練啟動摘要 ──
    r = env_cfg.rewards
    reward_terms = []
    for attr_name in dir(r):
        attr = getattr(r, attr_name, None)
        if hasattr(attr, 'weight') and hasattr(attr, 'func'):
            reward_terms.append((attr_name, attr.weight))
    active_terms = [(n, w) for n, w in reward_terms if w != 0.0]
    inactive_terms = [n for n, w in reward_terms if w == 0.0]

    cv = getattr(args_cli, 'curriculum_version', None) or 'default'
    print("\n" + "=" * 70)
    print("  訓練配置摘要")
    print("=" * 70)
    print(f"  Task:       {args_cli.task}")
    print(f"  Reward:     {args_cli.reward_mode}")
    print(f"  Curriculum: {cv}")
    print(f"  Seed: {args_cli.seed}  |  Envs: {env_cfg.scene.num_envs}  |  dt: {env_cfg.decimation * env_cfg.sim.dt:.2f}s")
    print(f"  Timesteps:  {agent_cfg['trainer']['timesteps']}  |  Rollouts: {agent_cfg['agent']['rollouts']}  |  Epochs: {agent_cfg['agent']['learning_epochs']}  |  Batches: {agent_cfg['agent']['mini_batches']}")
    print(f"  γ: {agent_cfg['agent']['discount_factor']}  |  LR: {agent_cfg['agent']['learning_rate']}  |  Clip: {agent_cfg['agent']['ratio_clip']}")
    print(f"\n  活躍 Reward Terms ({len(active_terms)}):")
    for name, weight in sorted(active_terms, key=lambda x: -abs(x[1])):
        print(f"    {name:30s} w={weight:+.1f}")
    if inactive_terms:
        print(f"  已關閉 ({len(inactive_terms)}): {', '.join(sorted(inactive_terms))}")
    print("=" * 70 + "\n")

    try:
        if wandb_run is not None:
            trainer.train()
        else:
            runner.run()
        print(f"[INFO] Training completed successfully")
    except KeyboardInterrupt:
        print(f"[INFO] Training interrupted by user")
    except Exception as e:
        print(f"[ERROR] Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if wandb_run is not None:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
