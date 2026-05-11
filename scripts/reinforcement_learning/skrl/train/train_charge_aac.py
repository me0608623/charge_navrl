# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train Charge navigation RL agent with SKRL.

從 Stable Baselines3 遷移到 SKRL 的訓練腳本。
適用於 Phase 0 (車輛動力學校準) 訓練。

環境規格:
    - 觀測空間: 131 維 (Box)
    - 動作空間: 2 維 (Box[-1, 1]) - [前進速度, 旋轉速度]

使用方法:
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge.py \\
        --task Isaac-Navigation-Charge-Phase0 \\
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
parser = argparse.ArgumentParser(description="Train Charge navigation RL agent with SKRL.")
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
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
    default=0,  # 0表示自动计算（总timesteps的10%或1000，取较小值）
    help="Print training summary every N timesteps. Default: auto (min(1000, total_timesteps/10)). Set to 0 to disable.",
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

# Import custom AAC wrapper
from aac_wrapper import AACIsaacLabWrapper, wrap_env_for_aac

# Registry for custom SKRL model modules (resolved in patched _generate_models)
import charge_models as _charge_models_module
import vlp16_models as _vlp16_models_module
_CUSTOM_MODEL_MODULES = {
    "charge_models": _charge_models_module,
    "vlp16_models": _vlp16_models_module,
}

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import logger
logger = logging.getLogger(__name__)


# ============================================================================
# Patch PPO Agent and Runner for AAC (shared_states support)
# ============================================================================

def patch_skrl_for_aac():
    """Patch SKRL Runner and PPO for AAC support

    1. Patch Runner to treat PPO like MAPPO for Critic model
    2. Patch PPO.init to add shared_states to _tensors_names (Memory allocation)
    3. Patch PPO.record_transition to store shared_states in Memory
    4. Patch PPO._update to use shared_states for Value Loss computation
    """

    from skrl.utils.runner.torch import Runner
    from skrl.agents.torch.ppo import PPO
    from skrl.envs.wrappers.torch import MultiAgentEnvWrapper
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    import copy
    import torch.nn.functional as F
    import itertools

    # Patch Runner._generate_models to use state_space for Critic (like MAPPO)
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
                    is_custom_model = False
                    if "." in model_class_name:
                        # Custom model: e.g. "charge_models.ChargePolicy"
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

                    # KEY MODIFICATION: Use state_space for Critic in AAC (like MAPPO)
                    observation_space = observation_spaces[agent_id]
                    if role == "value" and hasattr(env, 'state_space') and env.state_space is not None:
                        observation_space = state_spaces[agent_id]

                    if is_custom_model:
                        # Custom models: instantiate directly (no return_source)
                        models[agent_id][role] = model_class(
                            observation_space=observation_space,
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                        )
                    else:
                        # Built-in models: use return_source for printing
                        source = model_class(
                            observation_space=observation_space,
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                            return_source=True,
                        )
                        # instantiate model
                        models[agent_id][role] = model_class(
                            observation_space=observation_space,
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                        )
            # shared models
            else:
                roles = list(models_cfg.keys())
                if len(roles) != 2:
                    raise ValueError(
                        "Runner currently only supports shared models, made up of exactly two models. "
                        "Set 'separate' field to True to create non-shared models for the given cfg"
                    )
                # get shared model structure and parameters
                structure = []
                parameters = []
                for role in roles:
                    # get instantiator function and remove 'class' key
                    model_structure = models_cfg[role].get("class")
                    if not model_structure:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]
                    structure.append(model_structure)
                    parameters.append(self._process_cfg(models_cfg[role]))
                model_class = self._component("Shared")

                # print model source
                source = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                    return_source=True,
                )
                # instantiate model
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

    # Patch PPO's record_transition to use shared_states
    original_record_transition = PPO.record_transition

    def aac_record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        """Record transition with shared_states support for AAC

        Key changes:
        1. Use CACHED shared_states from time t (not infos which is time t+1)
        2. Compute values using shared_states (critic-dim privileged info)
        3. Store shared_states in Memory for _update phase
        4. Cache next shared_states for the next record_transition call
        """
        super(PPO, self).record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            self._current_next_states = next_states

            # Extract next_shared_states from infos (corresponds to time t+1)
            next_shared_states = infos.get("shared_states", None)

            # AAC: Store next_shared_states for GAE bootstrapping
            if next_shared_states is not None:
                self._current_next_shared_states = next_shared_states
            else:
                self._current_next_shared_states = None

            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # AAC: Use CACHED shared_states from time t (matching states)
            # infos["shared_states"] is from time t+1 (after env.step), but we need
            # the shared_states that corresponds to the current states (time t).
            # On the first call, fall back to infos since no cache exists yet.
            shared_states = getattr(self, '_cached_shared_states', None)
            if shared_states is None:
                shared_states = infos.get("shared_states", states)

            # Cache the NEXT shared_states for the next record_transition call
            if next_shared_states is not None:
                self._cached_shared_states = next_shared_states.clone()
            else:
                self._cached_shared_states = None

            # Sanitize shared_states before preprocessing (NaN poisons RunningStandardScaler)
            if shared_states is not None and torch.isnan(shared_states).any():
                shared_states = torch.nan_to_num(shared_states, nan=0.0)

            # Compute values using shared_states with its preprocessor
            if hasattr(self, '_shared_states_preprocessor') and shared_states is not None:
                states_for_value = self._shared_states_preprocessor(shared_states)
            else:
                # Fallback to regular states with preprocessor
                states_for_value = self._state_preprocessor(states) if self._state_preprocessor else states

            with torch.cuda.amp.autocast(enabled=self._mixed_precision):
                values, _, _ = self.value.act(
                    {"states": states_for_value}, role="value"
                )
                values = self._value_preprocessor(values, inverse=True) if self._value_preprocessor else values

            # ── NaN detection: check critic output and inputs ──
            if not hasattr(self, '_nan_record_diag_printed'):
                _has_nan = torch.isnan(values).any().item()
                _input_nan = torch.isnan(states_for_value).any().item()
                _shared_nan = torch.isnan(shared_states).any().item()
                _states_nan = torch.isnan(states).any().item()
                _rewards_nan = torch.isnan(rewards).any().item()
                if _has_nan or _input_nan or _shared_nan or _states_nan or _rewards_nan:
                    self._nan_record_diag_printed = True
                    print("\n" + "!" * 70)
                    print("[NaN DETECT] NaN found in record_transition!")
                    print("!" * 70)
                    print(f"  states (policy obs):    nan={_states_nan}, shape={states.shape}")
                    print(f"  shared_states:          nan={_shared_nan}, shape={shared_states.shape}")
                    if _shared_nan:
                        _ss_nan_per_dim = torch.isnan(shared_states).float().mean(dim=0)
                        _top_nan_dims = torch.where(_ss_nan_per_dim > 0)[0]
                        print(f"    NaN dims: {_top_nan_dims[:20].tolist()}")
                    print(f"  states_for_value:       nan={_input_nan}, shape={states_for_value.shape}")
                    print(f"  values (critic output): nan={_has_nan}, shape={values.shape}")
                    print(f"  rewards:                nan={_rewards_nan}")
                    print("!" * 70 + "\n")

            # ── One-time diagnostic: verify critic output is not constant/degenerate ──
            if not hasattr(self, '_debug_record_diag_printed'):
                self._debug_record_diag_printed = True
                print("\n" + "=" * 70)
                print("[AAC_RECORD] === ONE-TIME DIAGNOSTIC (first record_transition) ===")
                print("=" * 70)
                print(f"  states (policy obs):       shape={states.shape}, dtype={states.dtype}")
                print(f"  shared_states (critic in): shape={shared_states.shape}, dtype={shared_states.dtype}")
                print(f"  states_for_value (after preproc): shape={states_for_value.shape}")
                print(f"  values (critic output):    shape={values.shape}, dtype={values.dtype}")
                print(f"    mean={values.mean().item():.6f}, std={values.std().item():.6f}")
                print(f"    min={values.min().item():.6f}, max={values.max().item():.6f}")
                print(f"    any_nan={torch.isnan(values).any().item()}")
                print(f"  rewards:                   shape={rewards.shape}")
                print(f"    mean={rewards.mean().item():.6f}, min={rewards.min().item():.6f}, max={rewards.max().item():.6f}")
                val_std = values.std().item()
                if val_std < 1e-4:
                    print(f"  WARNING: Critic output std={val_std:.8f} — nearly CONSTANT at mean={values.mean().item():.4f}")
                    print(f"    This will produce near-zero advantages -> policy won't learn!")
                print("=" * 70 + "\n")

            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # 🔥 AAC FIX: Store shared_states in Memory for _update phase
            # Check if shared_states is in _tensors_names (Memory has space for it)
            if "shared_states" in self._tensors_names:
                self.memory.add_samples(
                    states=states, actions=actions, rewards=rewards,
                    next_states=next_states, terminated=terminated, truncated=truncated,
                    log_prob=self._current_log_prob, values=values,
                    shared_states=shared_states,  # 🔥 Store privileged info!
                )
                for memory in self.secondary_memories:
                    memory.add_samples(
                        states=states, actions=actions, rewards=rewards,
                        next_states=next_states, terminated=terminated, truncated=truncated,
                        log_prob=self._current_log_prob, values=values,
                        shared_states=shared_states,  # 🔥 Store privileged info!
                    )
            else:
                # Fallback: Memory doesn't have shared_states space
                self.memory.add_samples(
                    states=states, actions=actions, rewards=rewards,
                    next_states=next_states, terminated=terminated, truncated=truncated,
                    log_prob=self._current_log_prob, values=values,
                )
                for memory in self.secondary_memories:
                    memory.add_samples(
                        states=states, actions=actions, rewards=rewards,
                        next_states=next_states, terminated=terminated, truncated=truncated,
                        log_prob=self._current_log_prob, values=values,
                    )

    PPO.record_transition = aac_record_transition

    # ========================================================================
    # Patch 3: PPO.__init__ to detect AAC and store state_size
    # ========================================================================
    # We patch __init__ to detect AAC before calling original init
    # This allows us to know the shared_states size before Memory is created in init()
    original_ppo_dunder_init = PPO.__init__

    def aac_ppo_dunder_init(self, models, memory=None, observation_space=None, action_space=None, device=None, cfg=None):
        """Patched PPO.__init__ that detects AAC and stores state_size for later use"""
        # Detect AAC by checking observation_space vs state_space from env in models
        self._aac_state_size = None
        self._aac_obs_size = None

        # Try to find the actual environment with state_space
        # The env might be in the memory object or we can look at the value model's observation_space
        # Since we patched Runner._generate_models to use state_space for value model,
        # we can check if the value model has a different observation_space than the policy

        try:
            policy = models.get("policy", None)
            value = models.get("value", None)

            if policy is not None and value is not None:
                # Check if value model has different observation space (indicates AAC)
                policy_obs_space = policy.observation_space if hasattr(policy, 'observation_space') else observation_space
                value_obs_space = value.observation_space if hasattr(value, 'observation_space') else observation_space

                if policy_obs_space is not None and value_obs_space is not None:
                    try:
                        policy_size = policy_obs_space.shape[0] if hasattr(policy_obs_space, 'shape') else policy_obs_space
                        value_size = value_obs_space.shape[0] if hasattr(value_obs_space, 'shape') else value_obs_space

                        if value_size != policy_size:
                            self._aac_state_size = value_size
                            self._aac_obs_size = policy_size
                            print(f"[DEBUG] AAC: Detected in __init__ - policy_obs={policy_size}, value_obs={value_size}")
                    except (AttributeError, IndexError, TypeError) as e:
                        print(f"[DEBUG] AAC: Could not extract sizes: {e}")
        except Exception as e:
            print(f"[DEBUG] AAC: Detection error in __init__: {e}")

        # Call original __init__
        original_ppo_dunder_init(self, models, memory, observation_space, action_space, device, cfg)

    PPO.__init__ = aac_ppo_dunder_init

    # ========================================================================
    # Patch 3b: PPO.init to create shared_states tensor and preprocessor
    # ========================================================================
    original_ppo_init = PPO.init

    def aac_ppo_init(self, trainer_cfg=None):
        """Patched PPO.init that creates shared_states tensor in Memory for AAC"""
        # Call original init first (this creates Memory tensors and sets _tensors_names)
        original_ppo_init(self, trainer_cfg=trainer_cfg)

        # If AAC was detected in __init__, create the shared_states tensor and preprocessor
        if hasattr(self, '_aac_state_size') and self._aac_state_size is not None:
            from skrl.resources.preprocessors.torch import RunningStandardScaler

            state_size = self._aac_state_size

            # Add shared_states to _tensors_names
            if "shared_states" not in self._tensors_names:
                self._tensors_names.append("shared_states")
                print(f"[DEBUG] AAC: Added 'shared_states' to _tensors_names")

            # Create shared_states tensor in Memory
            if self.memory is not None:
                self.memory.create_tensor(name="shared_states", size=(state_size,), dtype=torch.float32)
                print(f"[DEBUG] AAC: Created shared_states tensor in Memory with size=({state_size},)")

                # Also create in secondary memories
                for memory in self.secondary_memories:
                    memory.create_tensor(name="shared_states", size=(state_size,), dtype=torch.float32)

            # Create separate preprocessor for shared_states (privileged info)
            self._shared_states_preprocessor = RunningStandardScaler(size=state_size)
            self.checkpoint_modules["shared_states_preprocessor"] = self._shared_states_preprocessor
            print(f"[DEBUG] AAC: Created shared_states_preprocessor with size={state_size}")
        else:
            print(f"[DEBUG] AAC: No AAC detected (no shared_states needed)")

    PPO.init = aac_ppo_init

    # ========================================================================
    # Patch 5: PPO._update to use shared_states for Value Loss computation
    # ========================================================================
    original_update = PPO._update

    def aac_update(self, timestep: int, timesteps: int):
        """Patched _update that uses shared_states (privileged info) for Value Loss

        This is the CORE FIX for AAC:
        - Policy Loss uses sampled_states (81-dim, non-privileged)
        - Value Loss uses sampled_shared_states (131-dim, privileged)
        """

        # 🔥 AAC FIX: Compute last_values using privileged info (shared_states)
        # We use the shared_states that was stored in record_transition
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)

            # Try to get shared_states for GAE bootstrapping
            last_values = None

            # Option 1: Use _current_next_shared_states if available (from last record_transition)
            if hasattr(self, '_current_next_shared_states') and self._current_next_shared_states is not None:
                try:
                    # Sanitize NaN before preprocessing
                    _next_ss = torch.nan_to_num(self._current_next_shared_states.float(), nan=0.0)
                    # Use shared_states_preprocessor for critic-dim input
                    if hasattr(self, '_shared_states_preprocessor'):
                        preprocessed_states = self._shared_states_preprocessor(_next_ss)
                    else:
                        preprocessed_states = _next_ss
                    last_values, _, _ = self.value.act(
                        {"states": preprocessed_states}, role="value"
                    )
                    print(f"[DEBUG] AAC: Computed last_values from _current_next_shared_states, shape={last_values.shape}")
                except RuntimeError as e:
                    print(f"[WARNING] AAC: Failed to compute last_values with _current_next_shared_states: {e}")
                    last_values = None

            # Option 2: Try to get last shared_states from memory
            if last_values is None and "shared_states" in self._tensors_names:
                try:
                    all_shared_states = self.memory.get_tensor_by_name("shared_states")
                    if all_shared_states is not None and all_shared_states.numel() > 0:
                        # Get the last row (most recent across all envs)
                        # Memory shape: [memory_size * num_envs, shared_states_dim]
                        # We need the last num_envs rows
                        num_envs = self._current_next_states.shape[0]

                        # Reshape if needed
                        if all_shared_states.dim() == 1:
                            # Single dimension, assume it's one sample
                            last_shared_states = all_shared_states.unsqueeze(0)
                        else:
                            # Get the last num_envs rows
                            last_shared_states = all_shared_states[-num_envs:]

                        # Use shared_states_preprocessor for 131-dim input
                        if hasattr(self, '_shared_states_preprocessor'):
                            preprocessed_states = self._shared_states_preprocessor(last_shared_states.float())
                        else:
                            preprocessed_states = last_shared_states.float()

                        last_values, _, _ = self.value.act(
                            {"states": preprocessed_states}, role="value"
                        )
                        print(f"[DEBUG] AAC: Computed last_values from memory shared_states, shape={last_values.shape}")
                except (KeyError, AttributeError, RuntimeError) as e:
                    print(f"[WARNING] AAC: Failed to get shared_states from memory: {e}")
                    last_values = None

            # Option 3: Final fallback - use zeros
            if last_values is None:
                num_envs = self._current_next_states.shape[0]
                last_values = torch.zeros(num_envs, 1, device=self.device, dtype=torch.float32)
                print(f"[WARNING] AAC: Using zeros for last_values (num_envs={num_envs})")

            self.value.train(True)
        last_values = self._value_preprocessor(last_values, inverse=True) if self._value_preprocessor else last_values

        def compute_gae(rewards, dones, values, last_values_boot, discount_factor=0.99, lambda_coefficient=0.95):
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                next_val = values[i + 1] if i < memory_size - 1 else last_values_boot
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_val + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            return returns, advantages

        values = self.memory.get_tensor_by_name("values")
        rewards_mem = self.memory.get_tensor_by_name("rewards")
        terminated_mem = self.memory.get_tensor_by_name("terminated")
        truncated_mem = self.memory.get_tensor_by_name("truncated")

        # ── NaN early detection: check memory tensors before GAE ──
        if not hasattr(self, '_nan_memory_diag_printed'):
            _mem_nans = {}
            for _name, _t in [("values", values), ("rewards", rewards_mem),
                               ("terminated", terminated_mem), ("truncated", truncated_mem),
                               ("last_values", last_values)]:
                if torch.isnan(_t).any():
                    _mem_nans[_name] = torch.isnan(_t).sum().item()
            if _mem_nans:
                self._nan_memory_diag_printed = True
                print("\n" + "!" * 70)
                print("[NaN DETECT] NaN found in memory tensors BEFORE GAE computation!")
                print("!" * 70)
                for _name, _count in _mem_nans.items():
                    print(f"  {_name}: {_count} NaN values")
                print(f"  values: shape={values.shape}, mean={torch.nanmean(values).item():.6f}")
                print(f"  rewards: shape={rewards_mem.shape}, mean={torch.nanmean(rewards_mem).item():.6f}")
                print(f"  last_values: shape={last_values.shape}, mean={torch.nanmean(last_values).item():.6f}")
                # Check shared_states too
                if "shared_states" in self._tensors_names:
                    try:
                        _ss = self.memory.get_tensor_by_name("shared_states")
                        _ss_nan = torch.isnan(_ss).sum().item()
                        print(f"  shared_states: shape={_ss.shape}, nan_count={_ss_nan}")
                    except Exception:
                        pass
                # Check states (policy obs)
                try:
                    _st = self.memory.get_tensor_by_name("states")
                    _st_nan = torch.isnan(_st).sum().item()
                    print(f"  states (policy obs): shape={_st.shape}, nan_count={_st_nan}")
                except Exception:
                    pass
                print("!" * 70 + "\n")

        returns, advantages = compute_gae(
            rewards=rewards_mem,
            dones=terminated_mem | truncated_mem,
            values=values,
            last_values_boot=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        # ── NaN check after GAE ──
        if not hasattr(self, '_nan_gae_diag_printed'):
            if torch.isnan(returns).any() or torch.isnan(advantages).any():
                self._nan_gae_diag_printed = True
                print(f"[NaN DETECT] GAE output contains NaN! returns_nan={torch.isnan(returns).sum().item()}, advantages_nan={torch.isnan(advantages).sum().item()}")

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # Check if shared_states exists in memory
        has_shared_states = "shared_states" in self._tensors_names
        if has_shared_states:
            try:
                _ = self.memory.get_tensor_by_name("shared_states")
            except KeyError:
                has_shared_states = False
                print("[WARNING] AAC: shared_states in _tensors_names but not in Memory!")

        # ── 除錯：首次 _update 的一次性診斷 ──
        if not hasattr(self, '_debug_update_diag_printed'):
            self._debug_update_diag_printed = True
            rewards_mem = self.memory.get_tensor_by_name("rewards")
            print("\n" + "=" * 70)
            print("[AAC 更新] === 首次更新診斷 ===")
            print("=" * 70)
            print(f"  價值函數 (記憶體): shape={values.shape}, dtype={values.dtype}, device={values.device}")
            print(f"    均值={values.mean().item():.6f}, 標準差={values.std().item():.6f}, 最小={values.min().item():.6f}, 最大={values.max().item():.6f}")
            print(f"    全有限={torch.isfinite(values).all().item()}, 含NaN={torch.isnan(values).any().item()}")
            print(f"  回報 (GAE 輸出): shape={returns.shape}")
            print(f"    均值={returns.mean().item():.6f}, 標準差={returns.std().item():.6f}, 最小={returns.min().item():.6f}, 最大={returns.max().item():.6f}")
            print(f"  優勢函數 (GAE): shape={advantages.shape}")
            print(f"    均值={advantages.mean().item():.6f}, 標準差={advantages.std().item():.6f}, 最小={advantages.min().item():.6f}, 最大={advantages.max().item():.6f}")
            print(f"  獎勵 (記憶體): shape={rewards_mem.shape}")
            print(f"    均值={rewards_mem.mean().item():.6f}, 標準差={rewards_mem.std().item():.6f}, 最小={rewards_mem.min().item():.6f}, 最大={rewards_mem.max().item():.6f}")
            print(f"  自舉末值: shape={last_values.shape}")
            print(f"    均值={last_values.mean().item():.6f}")
            print(f"  折扣因子={self._discount_factor}, GAE λ={self._lambda}")
            # 檢查價值函數是否異常恆定
            val_std = values.std().item()
            if val_std < 1e-4:
                print(f"  ⚠ 警告: 價值函數標準差={val_std:.8f} — Critic 輸出幾乎恆定！")
                print(f"    這代表 TD 誤差 ≈ 獎勵，優勢值會很小，策略無法學習。")
            adv_std = advantages.std().item()
            if adv_std < 1e-3:
                print(f"  ⚠ WARNING: advantages std={adv_std:.8f} — learning signal is extremely weak!")
            print("=" * 70 + "\n")

        # ── Debug: Advantage statistics (before normalization) ──
        adv_raw = advantages.clone()
        adv_raw_mean = adv_raw.mean().item()
        adv_raw_std = adv_raw.std().item() if adv_raw.numel() > 1 else 0.0
        adv_raw_min = adv_raw.min().item()
        adv_raw_max = adv_raw.max().item()

        # Normalize advantages (standard PPO practice)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.memory.set_tensor_by_name("advantages", advantages)

        adv_norm_mean = advantages.mean().item()
        adv_norm_std = advantages.std().item() if advantages.numel() > 1 else 0.0

        # ── Debug: Value prediction statistics (with full min/max) ──
        val_pred_mean = values.mean().item()
        val_pred_std = values.std().item() if values.numel() > 1 else 0.0
        val_pred_min = values.min().item()
        val_pred_max = values.max().item()
        returns_mean = returns.mean().item()
        returns_std = returns.std().item() if returns.numel() > 1 else 0.0
        returns_min = returns.min().item()
        returns_max = returns.max().item()
        value_prediction_error_abs = (returns - values).abs().mean().item()

        # ── Debug: Reward statistics from memory ──
        rewards_in_memory = self.memory.get_tensor_by_name("rewards")
        reward_mem_mean = rewards_in_memory.mean().item()
        reward_mem_std = rewards_in_memory.std().item() if rewards_in_memory.numel() > 1 else 0.0
        reward_mem_min = rewards_in_memory.min().item()
        reward_mem_max = rewards_in_memory.max().item()

        # ── Debug: Explained variance ──
        # EV = 1 - Var(returns - values) / Var(returns)
        var_returns = returns.var().item() if returns.numel() > 1 else 1e-8
        var_residual = (returns - values).var().item() if returns.numel() > 1 else 0.0
        explained_variance = 1.0 - var_residual / max(var_returns, 1e-8)

        # ── Debug: Parameter snapshot (sample first policy layer) ──
        param_snapshot = None
        for p in self.policy.parameters():
            if p.requires_grad and p.numel() > 10:
                param_snapshot = p.data.clone()
                break

        # Sample mini-batches
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_clip_fraction = 0
        cumulative_ratio_mean = 0
        num_minibatch_updates = 0
        policy_grad_norm = 0.0
        value_grad_norm = 0.0
        all_kl_divergences = []  # 跨所有 epoch 累計，給 KLAdaptiveLR 用

        # Learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # Mini-batches loop
            for batch_data in sampled_batches:
                # Unpack batch data
                sampled_states = batch_data[0]
                sampled_actions = batch_data[1]
                sampled_log_prob = batch_data[2]
                sampled_values = batch_data[3]
                sampled_returns = batch_data[4]
                sampled_advantages = batch_data[5]

                # 🔥 AAC: Get shared_states for Value Loss computation
                if has_shared_states and len(batch_data) > 6:
                    sampled_shared_states = batch_data[6]
                else:
                    sampled_shared_states = sampled_states  # Fallback

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # Sanitize NaN before preprocessing (prevents scaler poisoning)
                    sampled_states = torch.nan_to_num(sampled_states, nan=0.0)
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    # Safety: clamp discrete actions to valid range before log_prob
                    # (prevents CUDA device-side assert from out-of-bounds gather)
                    _taken_actions = sampled_actions
                    if hasattr(self.policy, '_c_distribution'):
                        # CategoricalMixin — clamp flat index to [0, n-1]
                        n = self.policy.action_space.n
                        _taken_actions = sampled_actions.clone()
                        _taken_actions[:, 0] = _taken_actions[:, 0].clamp(0, n - 1)
                    elif hasattr(self.policy, '_mc_distributions'):
                        # MultiCategoricalMixin — clamp each action dim to [0, nvec_i - 1]
                        nvec = self.policy.action_space.nvec
                        _taken_actions = sampled_actions.clone()
                        for i, n in enumerate(nvec):
                            _taken_actions[:, i] = _taken_actions[:, i].clamp(0, n - 1)

                    # Policy forward pass (uses 81-dim states)
                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": _taken_actions}, role="policy"
                    )

                    # Compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # Early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # Compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # Compute policy loss (uses 81-dim states)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # ── Debug: clip fraction and ratio stats ──
                    with torch.no_grad():
                        clip_frac = ((ratio - 1.0).abs() > self._ratio_clip).float().mean().item()
                        cumulative_clip_fraction += clip_frac
                        cumulative_ratio_mean += ratio.mean().item()
                        num_minibatch_updates += 1

                    # AAC: Value Loss uses critic-dim shared_states
                    if has_shared_states:
                        # Sanitize NaN before preprocessing
                        _ss_clean = torch.nan_to_num(sampled_shared_states, nan=0.0)
                        if hasattr(self, '_shared_states_preprocessor'):
                            sampled_shared_states_for_value = self._shared_states_preprocessor(_ss_clean, train=not epoch)
                        else:
                            # No separate preprocessor - use raw shared_states
                            # (Value model was created to expect raw 131-dim input)
                            sampled_shared_states_for_value = sampled_shared_states

                        predicted_values, _, _ = self.value.act(
                            {"states": sampled_shared_states_for_value}, role="value"
                        )
                    else:
                        # Fallback: Use regular states with preprocessor (functionally symmetric)
                        predicted_values, _, _ = self.value.act(
                            {"states": sampled_states}, role="value"
                        )

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # ── NaN guard: skip optimizer step if any loss is NaN/Inf ──
                total_loss = policy_loss + entropy_loss + value_loss
                _loss_is_bad = torch.isnan(total_loss) or torch.isinf(total_loss)

                if _loss_is_bad:
                    # One-time detailed NaN diagnostic
                    if not hasattr(self, '_nan_diag_printed'):
                        self._nan_diag_printed = True
                        print("\n" + "!" * 70)
                        print("[NaN GUARD] Loss is NaN/Inf — SKIPPING optimizer step")
                        print("!" * 70)
                        print(f"  policy_loss  = {policy_loss.item()}")
                        print(f"  entropy_loss = {entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss}")
                        print(f"  value_loss   = {value_loss.item()}")
                        print(f"  epoch={epoch}, mini-batch update #{num_minibatch_updates}")
                        # Trace NaN source
                        print(f"  ratio: nan={torch.isnan(ratio).any()}, inf={torch.isinf(ratio).any()}, mean={ratio.mean().item():.6f}")
                        print(f"  next_log_prob: nan={torch.isnan(next_log_prob).any()}, min={next_log_prob.min().item():.4f}, max={next_log_prob.max().item():.4f}")
                        print(f"  sampled_log_prob: nan={torch.isnan(sampled_log_prob).any()}, min={sampled_log_prob.min().item():.4f}, max={sampled_log_prob.max().item():.4f}")
                        print(f"  sampled_advantages: nan={torch.isnan(sampled_advantages).any()}, mean={sampled_advantages.mean().item():.6f}")
                        print(f"  sampled_returns: nan={torch.isnan(sampled_returns).any()}, mean={sampled_returns.mean().item():.6f}")
                        print(f"  sampled_values: nan={torch.isnan(sampled_values).any()}, mean={sampled_values.mean().item():.6f}")
                        print(f"  predicted_values: nan={torch.isnan(predicted_values).any()}, mean={predicted_values.mean().item():.6f}")
                        print(f"  sampled_states: nan={torch.isnan(sampled_states).any()}, shape={sampled_states.shape}")
                        if has_shared_states and len(batch_data) > 6:
                            print(f"  sampled_shared_states: nan={torch.isnan(sampled_shared_states).any()}, shape={sampled_shared_states.shape}")
                        print("!" * 70 + "\n")
                    # Do NOT call backward/optimizer — skip this mini-batch entirely
                    num_minibatch_updates += 1
                    continue

                # Optimization step (loss is finite)
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                if hasattr(self, '_grad_norm_clip') and self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)

                    # ── NaN guard: check gradients before clipping ──
                    _has_nan_grad = False
                    with torch.no_grad():
                        for p in itertools.chain(self.policy.parameters(), self.value.parameters()):
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                _has_nan_grad = True
                                break

                    if _has_nan_grad:
                        if not hasattr(self, '_nan_grad_diag_printed'):
                            self._nan_grad_diag_printed = True
                            print("[NaN GUARD] NaN/Inf in gradients after backward — zeroing grads and skipping step")
                            print(f"  total_loss={total_loss.item():.6f}, policy={policy_loss.item():.6f}, value={value_loss.item():.6f}")
                        self.optimizer.zero_grad()
                        num_minibatch_updates += 1
                        continue

                    # ── Debug: compute grad norms BEFORE clipping (last epoch only) ──
                    if epoch == self._learning_epochs - 1:
                        with torch.no_grad():
                            _pnorm = 0.0
                            for p in self.policy.parameters():
                                if p.grad is not None:
                                    _pnorm += p.grad.data.norm(2).item() ** 2
                            policy_grad_norm = _pnorm ** 0.5

                            _vnorm = 0.0
                            for p in self.value.parameters():
                                if p.grad is not None:
                                    _vnorm += p.grad.data.norm(2).item() ** 2
                            value_grad_norm = _vnorm ** 0.5

                    if self.policy is self.value:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # 累計本 epoch 的 KL（供 scheduler 使用）
            all_kl_divergences.extend(kl_divergences)

        # Update learning rate ONCE after all epochs (not per-epoch)
        # 必須傳 KL 給 KLAdaptiveLR，否則 scheduler 不會調整 LR
        if self._learning_rate_scheduler:
            if isinstance(self.scheduler, KLAdaptiveLR):
                if all_kl_divergences:
                    kl_mean = torch.tensor(all_kl_divergences, device=self.device).mean()
                    self.scheduler.step(kl_mean.item())
            elif hasattr(self.scheduler, 'step'):
                self.scheduler.step()

        # Record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        # Track policy distribution metric (stddev for continuous, entropy for discrete)
        if hasattr(self.policy, '_c_distribution') and self.policy._c_distribution is not None:
            # CategoricalMixin — track entropy
            entropy_val = self.policy.get_entropy(role="policy").mean().item()
            self.track_data("Policy / Entropy", entropy_val)
        elif hasattr(self.policy, '_mc_distributions') and self.policy._mc_distributions:
            # MultiCategoricalMixin — track mean entropy across sub-distributions
            entropy_val = self.policy.get_entropy(role="policy").mean().item()
            self.track_data("Policy / Entropy", entropy_val)
        else:
            # GaussianMixin — track standard deviation
            self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        # ── Debug: Parameter diff check (max|w_after - w_before|) ──
        param_max_diff = 0.0
        if param_snapshot is not None:
            for p in self.policy.parameters():
                if p.requires_grad and p.numel() > 10:
                    param_max_diff = (p.data - param_snapshot).abs().max().item()
                    break

        # ── Debug: Collect all PPO internal metrics into _debug_ppo_metrics ──
        total_mb = max(num_minibatch_updates, 1)
        self._debug_ppo_metrics = {
            # Advantage (before normalization)
            "advantage_raw_mean": adv_raw_mean,
            "advantage_raw_std": adv_raw_std,
            "advantage_raw_min": adv_raw_min,
            "advantage_raw_max": adv_raw_max,
            # Advantage (after normalization)
            "advantage_norm_mean": adv_norm_mean,
            "advantage_norm_std": adv_norm_std,
            # Value predictions (full stats)
            "value_pred_mean": val_pred_mean,
            "value_pred_std": val_pred_std,
            "value_pred_min": val_pred_min,
            "value_pred_max": val_pred_max,
            # Returns (full stats)
            "returns_mean": returns_mean,
            "returns_std": returns_std,
            "returns_min": returns_min,
            "returns_max": returns_max,
            # Rewards in memory (per-step reward stats)
            "reward_mem_mean": reward_mem_mean,
            "reward_mem_std": reward_mem_std,
            "reward_mem_min": reward_mem_min,
            "reward_mem_max": reward_mem_max,
            "explained_variance": explained_variance,
            # Value prediction error (|returns - values|)
            "value_prediction_error_abs": value_prediction_error_abs,
            # KL divergence（跨所有 epoch 的均值，與 scheduler 看到的一致）
            "kl_divergence": torch.tensor(all_kl_divergences, device=self.device).mean().item() if all_kl_divergences else 0.0,
            # Ratio and clip
            "policy_ratio_mean": cumulative_ratio_mean / total_mb,
            "clip_fraction": cumulative_clip_fraction / total_mb,
            # Gradient norms (from last epoch)
            "grad_norm_policy": policy_grad_norm,
            "grad_norm_value": value_grad_norm,
            # Parameter update magnitude
            "param_max_diff": param_max_diff,
            # Learning rate
            "learning_rate": self.scheduler.get_last_lr()[0] if self._learning_rate_scheduler else 0.0,
        }

        # Also push to SKRL's TensorBoard logger via track_data
        # (ensures metrics appear even without WandBSequentialTrainer)
        for k, v in self._debug_ppo_metrics.items():
            if not (v != v):  # skip NaN
                self.track_data(f"ppo/{k}", v)

    PPO._update = aac_update

    print("[INFO] SKRL patched for AAC (shared_states support in Memory + _update)")

# Apply the patches
patch_skrl_for_aac()

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
    """Helper to stop training and cleanup progress bar properly on ctrl+c.

    Isaac Lab and SKRL will automatically handle resource cleanup.
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
# Main Training Function
# ============================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with SKRL agent for Charge navigation."""
    # Randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # Override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Set seed
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Max iterations for training
    if args_cli.max_iterations is not None:
        print(f"[WARNING] max_iterations={args_cli.max_iterations} is set, overriding timesteps!")
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"] * env_cfg.scene.num_envs

    # Set environment seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Directory for logging
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    # Dump configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

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
    # Wrap Environment for SKRL with AAC support
    # ========================================================================
    # Use custom wrapper that supports asymmetric actor-critic (shared_states)
    #
    # VLP16 v2: 79D single-frame observation, no frame stacking (num_stack=1).
    # Other tasks (Phase0, Maze, etc.): wrapper does frame stacking (num_stack=3).
    is_vlp16_task = "VLP16" in (args_cli.task or "")
    aac_num_stack = 1 if is_vlp16_task else 3
    env = wrap_env_for_aac(env, ml_framework=args_cli.ml_framework, num_stack=aac_num_stack)
    print(f"[INFO] Environment wrapped for SKRL with AAC support (num_stack={aac_num_stack})")

    # ========================================================================
    # Configure SKRL Agent and Trainer
    # ========================================================================
    # Set experiment directory in agent config
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
            wandb.init(
                project="charge_skrl",
                name=f"Phase0_{run_info}",
                config={
                    "task": args_cli.task,
                    "num_envs": env_cfg.scene.num_envs,
                    "agent": "PPO",
                    "seed": agent_cfg["seed"],
                    "ml_framework": args_cli.ml_framework,
                },
            )
            wandb_run = wandb.run
            print(f"[INFO] WandB initialized: {wandb.run.name}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize WandB: {e}")

    # ========================================================================
    # Create SKRL Trainer and Train
    # ========================================================================
    start_time = time.time()

    # 创建 Console Summary Logger（根据命令行参数）
    console_summary_logger = None
    print_summary_every = args_cli.print_summary_every

    # 自动计算打印间隔（如果为0）
    if print_summary_every == 0:
        total_timesteps = agent_cfg['trainer']['timesteps']
        # 使用总timesteps的10%或1000，取较小值，最小为100
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
            algo="PPO",
        )

    # 根据WandB状态选择trainer类型
    if wandb_run is not None:
        print("[INFO] Using WandB-enabled trainer for metrics logging")
        # 直接使用 WandBSequentialTrainer
        # 使用 SKRL 的内部组件创建自定义 trainer
        import copy

        # 生成trainer配置
        trainer_cfg = copy.deepcopy(agent_cfg.get("trainer", {}))
        trainer_cfg["close_environment_at_exit"] = False

        # 先创建Runner来获取agent和models
        runner = Runner(env, agent_cfg)

        # 获取原始trainer的agents_scope
        agents_scope = runner.trainer.agents_scope if hasattr(runner.trainer, 'agents_scope') else None

        # 创建 TrainingDebugLogger（5 大类 debug 指标）
        from training_debug_logger import TrainingDebugLogger
        debug_logger = TrainingDebugLogger(
            env=env,
            log_interval=agent_cfg.get("agent", {}).get("rollouts", 128),
        )
        print("[INFO] TrainingDebugLogger initialized (5 categories: Reward/Episode/Action/Motion/PPO)")

        # 创建WandB-enabled trainer（同时传入console_summary_logger + debug_logger + log_dir）
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
        print("[INFO] WandB-enabled trainer configured")
    else:
        # 使用标准 SKRL Runner（即使没有WandB，也可以使用console summary）
        runner = Runner(env, agent_cfg)
        trainer = None

    # Load checkpoint if specified
    checkpoint_agent = runner.agent if wandb_run is not None else runner.agent
    if args_cli.checkpoint is not None:
        print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")
        checkpoint_agent.load(args_cli.checkpoint)

    # Print training info
    print(f"[INFO] Starting training: {agent_cfg['trainer']['timesteps']} timesteps...")
    print(f"[INFO] Rollouts per update: {agent_cfg['agent']['rollouts']}")
    print(f"[INFO] Learning epochs: {agent_cfg['agent']['learning_epochs']}")
    print(f"[INFO] Mini batches: {agent_cfg['agent']['mini_batches']}")

    try:
        # Run training
        if wandb_run is not None:
            # 使用自定义trainer
            trainer.train()
        else:
            # 使用标准runner
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
        # Ensure WandB is properly closed
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
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
