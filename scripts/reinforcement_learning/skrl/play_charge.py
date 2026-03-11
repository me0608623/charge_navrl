# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play trained SKRL agent for Charge navigation with Frame Stacking support.

使用訓練好的 SKRL 模型進行演示/評估，支援 Frame Stacking 架構。

使用方法:
    # 使用預設模型路徑
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge.py \\
        --task Isaac-Navigation-Charge-Phase0-Play

    # 指定模型路徑
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge.py \\
        --task Isaac-Navigation-Charge-Phase0-Play \\
        --checkpoint logs/skrl/Isaac-Navigation-Charge-Phase0/2024-01-01_12-00-00/checkpoints/agent.pt

    # 自定義 Frame Stacking 數量
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge.py \\
        --task Isaac-Navigation-Charge-Phase0-Play \\
        --num_stack 3
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from isaaclab.app import AppLauncher

# ============================================================================
# 評估統計數據類
# ============================================================================

@dataclass
class EpisodeStats:
    """單回合統計數據"""
    episode_id: int
    total_reward: float
    episode_length: int
    success: bool
    collision: bool
    timeout: bool
    final_distance_to_goal: float
    max_speed: float
    min_distance_to_obstacle: float

    def __str__(self):
        status = "✓ Success" if self.success else ("✗ Collision" if self.collision else "✗ Timeout")
        return (f"Ep {self.episode_id:3d} | {status} | "
                f"Reward: {self.total_reward:7.2f} | "
                f"Length: {self.episode_length:3d} | "
                f"Final Dist: {self.final_distance_to_goal:.2f}m")


@dataclass
class EvaluationSummary:
    """評估匯總統計"""
    total_episodes: int
    success_count: int
    collision_count: int
    timeout_count: int
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    success_rate: float
    mean_final_distance: float

    def print_summary(self):
        """打印評估摘要"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Episodes:       {self.total_episodes}")
        print(f"Success Rate:         {self.success_rate:.2f}%")
        print(f"  - Success:          {self.success_count} ({self.success_count/self.total_episodes*100:.1f}%)")
        print(f"  - Collision:        {self.collision_count} ({self.collision_count/self.total_episodes*100:.1f}%)")
        print(f"  - Timeout:          {self.timeout_count} ({self.timeout_count/self.total_episodes*100:.1f}%)")
        print("-" * 70)
        print(f"Mean Reward:          {self.mean_reward:.2f} ± {self.std_reward:.2f}")
        print(f"Mean Episode Length:  {self.mean_episode_length:.2f}")
        print(f"Mean Final Distance:  {self.mean_final_distance:.3f} m")
        print("=" * 70 + "\n")


# ============================================================================
# 評估器類
# ============================================================================

class Phase0Evaluator:
    """Phase 0 導航任務評估器"""

    def __init__(self, num_episodes: int = 100, print_interval: int = 10):
        self.num_episodes = num_episodes
        self.print_interval = print_interval
        self.reset()

    def reset(self):
        """重置評估狀態"""
        self.episodes = []
        self.current_episode = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.success = False
        self.collision = False
        self.timeout = False
        self.episode_max_speed = 0.0
        self.episode_min_obstacle_dist = float('inf')

    def step(self, reward: float, infos: dict, env):
        """處理單步，更新統計"""
        self.episode_reward += reward
        self.episode_length += 1

        # 追蹤最大速度
        if "robot_velocity" in infos:
            speed = torch.norm(infos["robot_velocity"], dim=-1).mean().item()
            self.episode_max_speed = max(self.episode_max_speed, speed)

        # 追蹤最小障礙物距離
        if "min_obstacle_distance" in infos:
            min_dist = infos["min_obstacle_distance"].mean().item()
            self.episode_min_obstacle_dist = min(self.episode_min_obstacle_dist, min_dist)

        # 檢查回合結束條件
        dones = infos.get("dones", torch.zeros(env.unwrapped.num_envs, dtype=torch.bool))
        truncated = infos.get("time_outs", torch.zeros(env.unwrapped.num_envs, dtype=torch.bool))

        # 只關注第一個環境（單環境評估）
        if len(dones) > 0 and (dones[0] or truncated[0]):
            self._end_episode(infos, env)
            return True
        return False

    def _end_episode(self, infos: dict, env):
        """結束當前回合，記錄統計"""
        self.current_episode += 1

        # 檢查成功條件
        self.success = infos.get("is_success", torch.tensor([False]))[0].item()
        self.collision = infos.get("collision_occurred", torch.tensor([False]))[0].item()
        self.timeout = infos.get("episode_timeout", torch.tensor([False]))[0].item()

        # 獲取最終距離
        if "goal_distance" in infos:
            self.final_distance = infos["goal_distance"][0].item()
        else:
            self.final_distance = 0.0

        # 創建統計記錄
        episode_stat = EpisodeStats(
            episode_id=self.current_episode,
            total_reward=self.episode_reward,
            episode_length=self.episode_length,
            success=self.success,
            collision=self.collision,
            timeout=self.timeout,
            final_distance_to_goal=self.final_distance,
            max_speed=self.episode_max_speed,
            min_distance_to_obstacle=self.episode_min_obstacle_dist
        )
        self.episodes.append(episode_stat)

        # 打印進度
        if self.current_episode % self.print_interval == 0:
            print(f"[{self.current_episode}/{self.num_episodes}] {episode_stat}")

        # 重置當前回合統計
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_max_speed = 0.0
        self.episode_min_obstacle_dist = float('inf')

    def get_summary(self) -> EvaluationSummary:
        """獲取評估摘要"""
        if not self.episodes:
            return None

        rewards = [ep.total_reward for ep in self.episodes]
        lengths = [ep.episode_length for ep in self.episodes]
        final_dists = [ep.final_distance_to_goal for ep in self.episodes]

        success_count = sum(1 for ep in self.episodes if ep.success)
        collision_count = sum(1 for ep in self.episodes if ep.collision)
        timeout_count = sum(1 for ep in self.episodes if ep.timeout)

        return EvaluationSummary(
            total_episodes=len(self.episodes),
            success_count=success_count,
            collision_count=collision_count,
            timeout_count=timeout_count,
            mean_reward=np.mean(rewards),
            std_reward=np.std(rewards),
            mean_episode_length=np.mean(lengths),
            success_rate=success_count / len(self.episodes) * 100,
            mean_final_distance=np.mean(final_dists)
        )


# ============================================================================
# 命令行參數
# ============================================================================

parser = argparse.ArgumentParser(description="Play trained SKRL agent for Charge navigation with Frame Stacking.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="skrl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to load.")
parser.add_argument(
    "--num_stack",
    type=int,
    default=3,
    help="Number of frames to stack (default: 3, must match training)."
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for the skrl agent.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric logging.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=10,
    help="Number of episodes to evaluate."
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=10000,
    help="Maximum steps per episode."
)
parser.add_argument(
    "--save_stats",
    action="store_true",
    help="Save evaluation statistics to file."
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record video of evaluation."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of recorded video in steps."
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=1,
    help="Print evaluation progress every N episodes."
)
parser.add_argument(
    "--camera",
    type=str,
    default="top",
    choices=["top", "follow", "side"],
    help="Camera view: top (bird's eye), follow (track robot), side (45-degree angle)."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Store task and agent before AppLauncher consumes them
task_name = args_cli.task
agent_name = args_cli.agent

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import skrl
from packaging import version

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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Import AAC wrapper for asymmetric actor-critic support with frame stacking
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aac_wrapper import AACIsaacLabWrapper, wrap_env_for_aac

# Registry for custom SKRL model modules (resolved in patched _generate_models)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import charge_models as _charge_models_module
import vlp16_models as _vlp16_models_module
_CUSTOM_MODEL_MODULES = {
    "charge_models": _charge_models_module,
    "vlp16_models": _vlp16_models_module,
}

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)


# ============================================================================
# Patch SKRL Runner for AAC support
# ============================================================================

def patch_skrl_for_aac():
    """Patch SKRL Runner to support AAC with state_space for Critic model"""

    from skrl.utils.runner.torch import Runner
    from skrl.envs.wrappers.torch import MultiAgentEnvWrapper
    import copy

    # Store original _generate_models
    original_generate_models = Runner._generate_models

    def patched_generate_models(self, env, cfg):
        """Patched _generate_models that uses state_space for value model in AAC"""
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]

        # Get spaces - AAC: use state_space for Critic if available
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        agent_class = cfg.get("agent", {}).get("class", "").lower()

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' are defined in cfg")
            # get separate (non-shared) configuration and remove 'separate' key
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True

            # non-shared models
            if separate:
                for role in models_cfg:
                    # get instantiator function and remove 'class' key
                    model_class_name = models_cfg[role].get("class")
                    if not model_class_name:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]

                    # Resolve model class: built-in or custom (dotted name)
                    if "." in model_class_name:
                        parts = model_class_name.split(".")
                        module = _CUSTOM_MODEL_MODULES.get(parts[0])
                        if module is not None:
                            model_class = getattr(module, parts[1])
                        else:
                            raise ValueError(
                                f"Custom model module '{parts[0]}' not found. "
                                f"Available: {list(_CUSTOM_MODEL_MODULES.keys())}"
                            )
                    else:
                        model_class = self._component(model_class_name)

                    # KEY MODIFICATION: Use state_space for Critic in AAC
                    observation_space = observation_spaces[agent_id]
                    if role == "value" and hasattr(env, 'state_space') and env.state_space is not None:
                        observation_space = state_spaces[agent_id]
                        print(f"[PATCH] Using state_space {observation_space.shape} for Critic model")

                    # instantiate model
                    models[agent_id][role] = model_class(
                        observation_space=observation_space,
                        action_space=action_spaces[agent_id],
                        device=device,
                        **self._process_cfg(models_cfg[role]),
                    )
            else:
                raise ValueError("Shared models not supported in AAC mode")

        return models

    # Apply patch
    Runner._generate_models = patched_generate_models
    print("[PATCH] SKRL Runner patched for AAC support")


# Apply the patch
patch_skrl_for_aac()


# Check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()


def find_latest_checkpoint(task_name: str) -> Optional[str]:
    """Find the latest checkpoint for a given task.

    Search strategy:
        1. Search logs/skrl/<task_name>/ (exact match)
        2. Search logs/skrl/<base_task_name>/ (with -Play suffix removed)
        3. Within the most recent run directory, prefer best_agent.pt
        4. If no best_agent.pt, use the highest-numbered agent_NNNN.pt

    Args:
        task_name: Name of the task (e.g., Isaac-Navigation-Charge-VLP16)

    Returns:
        Path to the latest checkpoint file, or None if not found.
    """
    base_task_name = task_name.replace("-Play", "")

    # Search directories (exact name + base name without -Play)
    search_dirs = []
    for name in [task_name, base_task_name]:
        log_base = os.path.join("logs", "skrl", name)
        if os.path.exists(log_base):
            search_dirs.append(log_base)
    # Deduplicate while preserving order
    search_dirs = list(dict.fromkeys(search_dirs))

    if not search_dirs:
        print(f"[CHECKPOINT] No log directories found for task '{task_name}'")
        print(f"[CHECKPOINT] Searched: logs/skrl/{task_name}/, logs/skrl/{base_task_name}/")
        return None

    # Collect all run directories sorted by name (timestamp-based names sort chronologically)
    all_runs = []
    for log_base in search_dirs:
        for run_dir in sorted(os.listdir(log_base), reverse=True):
            full_path = os.path.join(log_base, run_dir)
            checkpoint_dir = os.path.join(full_path, "checkpoints")
            if os.path.isdir(checkpoint_dir):
                all_runs.append((run_dir, checkpoint_dir, log_base))

    if not all_runs:
        print(f"[CHECKPOINT] No checkpoint directories found in: {search_dirs}")
        return None

    # Use the most recent run (first after reverse sort)
    latest_run, latest_ckpt_dir, latest_log_base = all_runs[0]

    # List all .pt files in latest run
    pt_files = sorted(glob.glob(os.path.join(latest_ckpt_dir, "*.pt")))

    if not pt_files:
        print(f"[CHECKPOINT] No .pt files in: {latest_ckpt_dir}")
        return None

    # Print all available checkpoints in this run
    print(f"\n{'=' * 70}")
    print(f"  CHECKPOINT AUTO-DETECTION")
    print(f"{'=' * 70}")
    print(f"  Task:       {task_name}")
    print(f"  Log dir:    {latest_log_base}")
    print(f"  Run:        {latest_run}")
    print(f"  Available checkpoints ({len(pt_files)}):")
    for f in pt_files:
        fname = os.path.basename(f)
        fsize = os.path.getsize(f) / (1024 * 1024)  # MB
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        print(f"    {fname:<30s}  {fsize:6.1f} MB  ({mtime:%Y-%m-%d %H:%M:%S})")

    # Selection priority: best_agent.pt > highest agent_NNNN.pt > latest by mtime
    best_agent = os.path.join(latest_ckpt_dir, "best_agent.pt")
    if os.path.exists(best_agent):
        selected = best_agent
        reason = "best_agent.pt (preferred)"
    else:
        # Find highest numbered agent_NNNN.pt
        import re
        numbered = []
        for f in pt_files:
            m = re.search(r'agent_(\d+)\.pt$', f)
            if m:
                numbered.append((int(m.group(1)), f))
        if numbered:
            numbered.sort(reverse=True)
            selected = numbered[0][1]
            reason = f"highest step ({numbered[0][0]})"
        else:
            # Fallback: newest by modification time
            selected = max(pt_files, key=os.path.getmtime)
            reason = "latest modified"

    fsize = os.path.getsize(selected) / (1024 * 1024)
    mtime = datetime.fromtimestamp(os.path.getmtime(selected))
    print(f"  {'─' * 66}")
    print(f"  >>> SELECTED: {os.path.basename(selected)}")
    print(f"      Reason:   {reason}")
    print(f"      Size:     {fsize:.1f} MB")
    print(f"      Modified: {mtime:%Y-%m-%d %H:%M:%S}")
    print(f"      Path:     {selected}")
    print(f"{'=' * 70}\n")

    return selected


# ============================================================================
# Main Play Function
# ============================================================================

@hydra_task_config(task_name, agent_name)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play trained SKRL agent for Charge navigation."""

    print("\n" + "=" * 70)
    print("🚀 Charge Navigation - Play Script (with Frame Stacking)")
    print("=" * 70)
    print(f"Task:              {task_name}")
    print(f"Frame Stacking:    {args_cli.num_stack}")
    print(f"Episodes:          {args_cli.num_episodes}")
    print(f"Max Steps/Episode: {args_cli.max_steps}")
    print(f"Checkpoint:        {args_cli.checkpoint or 'Auto-detect'}")
    print("=" * 70 + "\n")

    # Override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ========================================================================
    # Configure viewer camera for play mode
    # ========================================================================
    from isaaclab.envs import ViewerCfg
    if args_cli.camera == "top":
        # Bird's eye view — see entire 16x16m arena
        env_cfg.viewer = ViewerCfg(
            eye=(0.0, 0.0, 25.0),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            env_index=0,
            resolution=(1920, 1080),
        )
    elif args_cli.camera == "follow":
        # Third-person follow — track robot from behind
        env_cfg.viewer = ViewerCfg(
            eye=(-3.0, 0.0, 3.0),
            lookat=(2.0, 0.0, 0.0),
            origin_type="asset_root",
            env_index=0,
            asset_name="robot",
            resolution=(1920, 1080),
        )
    elif args_cli.camera == "side":
        # 45-degree angle — good for seeing obstacles and depth
        env_cfg.viewer = ViewerCfg(
            eye=(15.0, -15.0, 15.0),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            env_index=0,
            resolution=(1920, 1080),
        )
    print(f"[INFO] Camera mode: {args_cli.camera}")

    # For play mode, reduce logging
    agent_cfg["agent"]["experiment"]["directory"] = "logs/skrl/play"
    agent_cfg["agent"]["experiment"]["experiment_name"] = "play"
    agent_cfg["trainer"]["timesteps"] = 0  # No training, just evaluation

    # Set the log directory for the environment
    env_cfg.log_dir = os.path.join("logs", "skrl", "play")

    # ========================================================================
    # Create Isaac Lab Environment
    # ========================================================================
    env = gym.make(
        task_name,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None
    )

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(Path(env_cfg.log_dir) / "videos" / "play"),
            "step_trigger": lambda step: step == 0,  # Record first episode
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ========================================================================
    # Wrap Environment for SKRL (with AAC + Frame Stacking support)
    # ========================================================================
    env = wrap_env_for_aac(env, ml_framework=args_cli.ml_framework, num_stack=args_cli.num_stack)
    print(f"[INFO] Environment wrapped for SKRL with AAC + Frame Stacking (num_stack={args_cli.num_stack})")

    # Print observation dimensions
    print(f"[INFO] Actor observation shape: {env.observation_space.shape}")
    print(f"[INFO] Critic state shape: {env.state_space.shape}")

    # ========================================================================
    # Configure SKRL Runner
    # ========================================================================
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Configure and instantiate the SKRL runner
    runner = Runner(env, agent_cfg)

    # Load checkpoint
    checkpoint_path = None
    if args_cli.checkpoint is not None:
        # User specified checkpoint
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
        print(f"\n[CHECKPOINT] User-specified: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Specified checkpoint does not exist: {checkpoint_path}")
            checkpoint_path = None
    else:
        # Auto-find latest checkpoint
        checkpoint_path = find_latest_checkpoint(task_name)

    if checkpoint_path:
        try:
            runner.agent.load(checkpoint_path)
            print(f"[OK] Checkpoint loaded successfully: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Continuing with untrained model (random actions)")
    else:
        print("\n[WARNING] No checkpoint found. Using untrained model (random actions).")
        print("[HINT] Train first with:")
        print(f"  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge.py \\")
        print(f"      --task {task_name} --num_envs 1024 --headless")

    # Set agent to evaluation mode
    runner.agent.set_running_mode("eval")
    print("[INFO] Agent set to evaluation mode")

    # ========================================================================
    # Create Evaluator
    # ========================================================================
    evaluator = Phase0Evaluator(
        num_episodes=args_cli.num_episodes,
        print_interval=args_cli.print_interval
    )

    # ========================================================================
    # Run Evaluation Loop
    # ========================================================================
    print("\n" + "=" * 70)
    print("🎮 Starting Evaluation...")
    print("=" * 70 + "\n")

    import time
    start_time = time.time()
    total_steps = 0

    try:
        # Reset environment
        obs, _ = env.reset()

        for step in range(args_cli.max_steps * args_cli.num_episodes):
            with torch.no_grad():
                # Get actions from agent
                actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]

            # Step environment
            obs, reward, terminated, truncated, infos = env.step(actions)

            # Update evaluator (use mean reward across envs)
            if len(reward) > 0:
                mean_reward = reward.mean().item()
            else:
                mean_reward = 0.0

            episode_done = evaluator.step(mean_reward, infos, env)
            total_steps += 1

            # Check if all episodes completed
            if evaluator.current_episode >= evaluator.num_episodes:
                break

            # Note: IsaacLab auto-resets terminated envs inside step(),
            # so obs already contains fresh observations for reset envs.

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user")

    finally:
        eval_time = time.time() - start_time

        # ====================================================================
        # Print Summary
        # ====================================================================
        summary = evaluator.get_summary()
        if summary:
            summary.print_summary()
            print(f"Total Evaluation Time: {eval_time:.2f} seconds")
            print(f"Average Time per Episode: {eval_time / len(evaluator.episodes):.2f} seconds")
            print(f"Total Steps: {total_steps}")
            print(f"Steps per Second: {total_steps / eval_time:.2f}")

            # Save statistics if requested
            if args_cli.save_stats:
                stats_dir = Path(env_cfg.log_dir) / "statistics"
                stats_dir.mkdir(parents=True, exist_ok=True)
                stats_file = stats_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

                with open(stats_file, 'w') as f:
                    f.write(f"Charge Navigation Evaluation Results\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Task: {task_name}\n")
                    f.write(f"Frame Stacking: {args_cli.num_stack}\n")
                    f.write(f"Checkpoint: {checkpoint_path}\n")
                    f.write(f"=" * 70 + "\n\n")

                    f.write(f"Total Episodes: {summary.total_episodes}\n")
                    f.write(f"Success Rate: {summary.success_rate:.2f}%\n")
                    f.write(f"  - Success: {summary.success_count}\n")
                    f.write(f"  - Collision: {summary.collision_count}\n")
                    f.write(f"  - Timeout: {summary.timeout_count}\n")
                    f.write(f"Mean Reward: {summary.mean_reward:.2f} ± {summary.std_reward:.2f}\n")
                    f.write(f"Mean Episode Length: {summary.mean_episode_length:.2f}\n")
                    f.write(f"Mean Final Distance: {summary.mean_final_distance:.3f} m\n")
                    f.write(f"\nEvaluation Time: {eval_time:.2f} seconds\n\n")

                    f.write("Episode Details:\n")
                    f.write("-" * 70 + "\n")
                    for ep in evaluator.episodes:
                        f.write(f"{ep}\n")

                print(f"\n[INFO] Statistics saved to: {stats_file}")

        env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
