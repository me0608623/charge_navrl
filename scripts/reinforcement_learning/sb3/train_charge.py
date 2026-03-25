# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3 for Charge navigation."""

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
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the training info.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Determine headless mode
headless_mode = "--headless" in sys.argv or getattr(args_cli, "headless", False)

# Configure WandB based on mode
if headless_mode:
    os.environ["WANDB_PROJECT"] = "charge_sb3"
    print("[INFO] WandB logging enabled: project = charge_sb3")
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import charge training modules
sys.path.insert(0, str(Path(__file__).parent / "charge"))
from charge import (
    ENABLE_LR_DECAY,
    DEFAULT_LR_FINAL_VALUE,
    DEFAULT_LR_INITIAL_VALUE,
    DEFAULT_WARMUP_RATIO,
    IsaacLabMetricsWrapper,
    NanProtectionCallback,
    SanitizeObservationsWrapper,
    WandBCallback,
    linear_schedule_with_warmup,
)

# Import logger
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def cleanup_pbar(*args):
    """Helper to stop training and cleanup progress bar properly on ctrl+c.

    Isaac Lab and SB3 will automatically handle resource cleanup.
    """
    raise KeyboardInterrupt


# Disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)


def disable_debug_vis(cfg) -> None:
    """Recursively disable all debug_vis options in configuration.

    Args:
        cfg: Configuration object (dataclass or attrs)
    """
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
# Environment Setup
# ============================================================================

def setup_environment(env_cfg, args_cli) -> tuple:
    """Create and configure the training environment.

    Args:
        env_cfg: Environment configuration
        args_cli: Command line arguments

    Returns:
        Tuple of (environment, metrics_wrapper)
    """
    # Disable debug visualization in headless mode
    if headless_mode:
        disable_debug_vis(env_cfg)
        print("[INFO] Headless mode: All debug_visualization disabled")

    # Create isaac environment
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
        from pathlib import Path as Pathlib
        log_dir = getattr(env_cfg, 'log_dir', 'logs/sb3/tmp')
        video_kwargs = {
            "video_folder": str(Pathlib(log_dir) / "videos" / "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for stable baselines (fast_variant=False to support episode statistics)
    env = Sb3VecEnvWrapper(env, fast_variant=False)
    print("[INFO] Sb3VecEnvWrapper with fast_variant=False (for episode statistics extraction)")

    # Add Isaac Lab metrics collection wrapper
    metrics_wrapper = IsaacLabMetricsWrapper(env)
    env = metrics_wrapper
    print("[INFO] Added IsaacLabMetricsWrapper (Gymnasium -> SB3 API adapter)")

    # Add observation sanitization wrapper
    env = SanitizeObservationsWrapper(env)
    print("[INFO] Added SanitizeObservationsWrapper (cleans nan/inf from observations)")

    return env, metrics_wrapper


def setup_normalization(env, agent_cfg: dict, env_cfg) -> VecNormalize:
    """Setup observation and reward normalization.

    Args:
        env: The environment to wrap
        agent_cfg: Agent configuration dict
        env_cfg: Environment configuration

    Returns:
        Normalized environment or original environment
    """
    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=10.0,
        )

    return env


def setup_learning_rate_schedule(agent_cfg: dict) -> dict:
    """Setup learning rate schedule with warmup and decay.

    Args:
        agent_cfg: Agent configuration dict

    Returns:
        Modified agent_cfg with learning_rate as a schedule function
    """
    if ENABLE_LR_DECAY and "learning_rate" in agent_cfg:
        lr_value = agent_cfg["learning_rate"]
        # Check if learning_rate is already a function (schedule)
        if callable(lr_value):
            print(f"[INFO] Learning rate is already a schedule function, skipping LR decay setup")
        else:
            initial_lr = float(agent_cfg.pop("learning_rate"))
            lr_schedule = linear_schedule_with_warmup(
                initial_value=initial_lr,
                final_value=DEFAULT_LR_FINAL_VALUE,
                warmup_ratio=DEFAULT_WARMUP_RATIO,
            )
            agent_cfg["learning_rate"] = lr_schedule
            print(f"[INFO] Linear learning rate decay enabled:")
            print(f"       Initial LR: {initial_lr:.2e}")
            print(f"       Final LR: {DEFAULT_LR_FINAL_VALUE:.2e}")
            print(f"       Warmup ratio: {DEFAULT_WARMUP_RATIO}")

    return agent_cfg


# ============================================================================
# Callback Setup
# ============================================================================

def setup_callbacks(
    log_dir: str,
    log_interval: int,
    metrics_wrapper: Optional[IsaacLabMetricsWrapper],
) -> list:
    """Setup all training callbacks.

    Args:
        log_dir: Directory for logging
        log_interval: Interval for logging
        metrics_wrapper: Metrics wrapper for WandB callback

    Returns:
        List of callbacks
    """
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=log_dir,
        name_prefix="model",
        verbose=2
    )
    callbacks.append(checkpoint_callback)

    # Log callback
    callbacks.append(LogEveryNTimesteps(n_steps=log_interval))

    # NaN protection callback
    nan_protection = NanProtectionCallback(verbose=1)
    callbacks.append(nan_protection)
    print("[INFO] Added NanProtectionCallback (monitors policy network weights)")

    # WandB callback (only in headless mode)
    if headless_mode:
        try:
            import wandb
            if wandb.run is not None:
                wandb_callback = WandBCallback(verbose=1, metrics_wrapper=metrics_wrapper)
                callbacks.append(wandb_callback)
                print("[INFO] WandB callback added for automatic metric logging")
        except Exception as e:
            print(f"[WARNING] Failed to add WandB callback: {e}")

    return callbacks


# ============================================================================
# Main Training Function
# ============================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # Randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # Override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    # Max iterations for training
    if args_cli.max_iterations is not None:
        print(f"[WARNING] max_iterations={args_cli.max_iterations} is set, overriding n_timesteps!")
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # Set environment seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Directory for logging
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    # Dump configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # Post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

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

    # Setup environment
    env, metrics_wrapper = setup_environment(env_cfg, args_cli)

    # Setup normalization
    env = setup_normalization(env, agent_cfg, env_cfg)

    # Setup learning rate schedule
    agent_cfg = setup_learning_rate_schedule(agent_cfg)

    # Create agent from stable baselines
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)

    # Load checkpoint if specified
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # Initialize WandB (only in headless mode)
    wandb_run = None
    if headless_mode:
        try:
            import wandb
            wandb.init(
                project="charge_sb3",
                name=f"Phase0_{run_info}",
                config={
                    "task": args_cli.task,
                    "num_envs": env_cfg.scene.num_envs,
                    "agent": "PPO",
                    "seed": agent_cfg["seed"],
                },
                sync_tensorboard=True,
                save_code=True,
            )
            wandb_run = wandb.run
            print(f"[INFO] WandB initialized: {wandb.run.name}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")

    # Setup callbacks
    callbacks = setup_callbacks(log_dir, args_cli.log_interval, metrics_wrapper)

    # Train the agent
    print(f"[INFO] Starting training: {n_timesteps} timesteps...")
    print(f"[INFO] Callbacks: {[type(c).__name__ for c in callbacks]}")

    start_time = time.time()
    try:
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=None,
        )
        print(f"[INFO] Training completed successfully")
    except KeyboardInterrupt:
        print(f"[INFO] Training interrupted by user")
    except Exception as e:
        print(f"[ERROR] Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Ensure WandB is properly closed
        if wandb_run is not None:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass

    # Save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
