"""Experiment config -- LEGO-style composition of training runs.

An ExperimentConfig bundles scene/phase/reward/algorithm/aux/encoder/budget
into a single named config. CLI args override config fields when explicitly provided.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, asdict, fields
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment specification.

    All fields have defaults matching the canonical train_rnn_car_wdclip.py parser
    defaults so that omitting a field keeps current behavior unchanged.
    """

    name: str = "custom"
    description: str = ""

    # IsaacLab task / scene
    task: str = "Isaac-Navigation-Charge-VLP16-Curriculum-WD"
    curriculum_version: str = "warp_drive_single_agent_v1"
    initial_stage: int = 1
    fixed_stage: bool = True
    scene_profile: str = "warp_drive_single_agent_v1"
    obstacle_mode: str = "rule_based"

    # profiles (metadata, Phase 0)
    reward_profile: str = "wd_sparse"
    algorithm_profile: str = "a2c_wd"
    aux_profile: str = "wd_7d_geometry"
    encoder_profile: str = "wd_exact_rnn"
    critic_profile: str = "symmetric"
    budget_profile: str = "custom"

    # training budget
    num_envs: int = 1024
    rollout_length: int = 300
    timesteps: int = 180000
    seed: int = 42

    # RL hyperparams
    lr: float = 2e-4
    rnn_lr: float = 5e-4
    vf_coeff: float = 0.5
    gamma: float = 0.991
    gae_lambda: float = 0.95
    normalize_return: bool = True
    value_init_bias: float | None = 0.0
    max_grad_norm: float = 1.0

    # A2C/PPO
    use_a2c: bool = True
    ppo_epochs: int = 2
    mini_batches: int = 16
    clip_eps: float = 0.1

    # entropy / WD update caps
    ent_coeff_linear: float = 0.0
    ent_coeff_angular: float = 0.0
    wd_update_clip: bool = True
    wd_actor_update_clip: float = 8.0
    wd_critic_update_clip: float = 30.0

    # aux
    disable_aux_training: bool = False
    aux_seq_len: int = 15
    aux_burn_in: int = 0
    aux_seq_batch_size: int = 256
    aux_grad_clip: float | None = 0.5

    # safety/logging
    lidar_no_noise: bool = True
    action_table_sample_size: int = 0
    log_interval: int = 10
    save_interval: int = 100

    # checkpoint / resume
    checkpoint: str | None = None
    no_resume_optimizer: bool = True

    # metadata
    tags: tuple[str, ...] = ()
    notes: str = ""


# ── Field name → CLI flag mapping ──────────────────────────────────────────
# Most fields map 1:1 to --<field_name>. Exceptions listed here.
_FIELD_TO_FLAGS: dict[str, list[str]] = {
    "use_a2c": ["--use_a2c", "--a2c", "--use_ppo", "--ppo"],
    "normalize_return": ["--normalize_return", "--normalize_returns"],
    "wd_update_clip": ["--wd_update_clip", "--no_wd_update_clip"],
    "fixed_stage": ["--fixed_stage"],
    "disable_aux_training": ["--disable_aux_training"],
    "lidar_no_noise": ["--lidar_no_noise"],
    "no_resume_optimizer": ["--no_resume_optimizer"],
}

# Fields that are metadata-only (not applied to args_cli)
_METADATA_ONLY_FIELDS = frozenset({
    "name", "description", "tags", "notes",
    "critic_profile", "budget_profile",
})

# Fields that map to a different args_cli attribute name
_FIELD_TO_ATTR: dict[str, str] = {
    # all field names match argparse dest names, so no remapping needed currently
}


def _was_cli_provided(field_name: str, argv: list[str]) -> bool:
    """Check if the user explicitly passed a CLI flag for this field."""
    flags = _FIELD_TO_FLAGS.get(field_name)
    if flags is None:
        flags = [f"--{field_name}"]
    for flag in flags:
        if flag in argv:
            return True
        if any(a.startswith(flag + "=") for a in argv):
            return True
    return False


def apply_experiment_config(
    args_cli: Any,
    cfg: ExperimentConfig,
    argv: list[str] | None = None,
) -> list[str]:
    """Apply experiment config defaults to args_cli, respecting CLI overrides.

    Args:
        args_cli: Parsed argparse namespace (mutated in-place).
        cfg: The experiment config providing defaults.
        argv: sys.argv for detecting explicit CLI flags. If None, uses sys.argv.

    Returns:
        List of field names that were applied (not overridden by CLI).
    """
    if argv is None:
        argv = sys.argv

    applied: list[str] = []

    for f in fields(cfg):
        if f.name in _METADATA_ONLY_FIELDS:
            continue

        attr_name = _FIELD_TO_ATTR.get(f.name, f.name)

        # If user explicitly provided this flag on CLI, keep their value
        if _was_cli_provided(f.name, argv):
            continue

        cfg_value = getattr(cfg, f.name)
        setattr(args_cli, attr_name, cfg_value)
        applied.append(f.name)

    return applied


def experiment_config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Convert to dict for WandB config / run_metadata. Handles tuples."""
    d = asdict(cfg)
    # Convert tuples to lists for JSON serialization
    if isinstance(d.get("tags"), tuple):
        d["tags"] = list(d["tags"])
    return d


def load_experiment_config_from_file(path: str) -> ExperimentConfig:
    """Load ExperimentConfig from a Python file that defines CONFIG."""
    spec = importlib.util.spec_from_file_location("_exp_config", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load experiment config from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "CONFIG", None)
    if cfg is None:
        raise ValueError(f"Experiment config file {path} must define CONFIG")
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError(
            f"CONFIG in {path} must be ExperimentConfig, got {type(cfg).__name__}"
        )
    return cfg
