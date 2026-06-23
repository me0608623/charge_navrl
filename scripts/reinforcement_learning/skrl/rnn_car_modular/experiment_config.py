"""Experiment config -- LEGO-style composition of training runs.

An ExperimentConfig bundles scene/phase/reward/algorithm/aux/encoder/budget
into a single named config. CLI args override config fields when explicitly provided.

Supports both YAML (recommended) and Python config files.
"""

from __future__ import annotations

import importlib.util
import os
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

    # profiles
    reward_profile: str = "wd_sparse"
    algorithm: str = "a2c"          # "a2c" | "ppo"
    aux_profile: str = "wd_7d_geometry"
    encoder_profile: str = "wd_exact_rnn"
    critic_profile: str = "symmetric"

    # training budget
    num_envs: int = 1024
    rollout_length: int = 300
    timesteps: int = 180000
    seed: int = 42

    # RL hyperparams
    lr: float = 2e-4
    rnn_lr: float = 5e-4
    aux_lr: float = 0.0
    aux_lr_predict_head: float = 0.0
    aux_lr_fc_middle: float = 0.0
    aux_lr_fc_front: float = 0.0
    aux_lr_extractor: float = 0.0
    vf_coeff: float = 0.5
    gamma: float = 0.991
    gae_lambda: float = 0.95
    normalize_return: bool = True
    adv_norm_mode: str = "mean_only"     # mean_only | partial | full
    value_init_bias: float | None = 0.0
    max_grad_norm: float = 0.5
    tbptt_len: int = 0
    lr_decay: float = 0.0

    # PPO-specific (ignored when algorithm="a2c")
    ppo_epochs: int = 2
    mini_batches: int = 16
    clip_eps: float = 0.1

    # entropy / WD update caps
    ent_coeff: float = 0.0
    ent_coeff_linear: float = 0.0
    ent_coeff_angular: float = 0.0
    wd_update_clip: bool = True
    wd_update_monitor_only: bool = False
    wd_actor_update_clip: float = 8.0
    wd_critic_update_clip: float = 30.0
    wd_module_entropy_eps: float = 1e-12
    policy_loss_clamp: float = 20.0
    vf_term_clamp: float = 8.0

    # model / encoder
    charge_encoder_mode: str = "extractor_rnn"
    hidden_dim: int = 30
    preprocess_dim: int = 12
    fc_dim: int = 48
    wd_middle_dim: int = 32
    rnn_type: str = "RNN"
    predict_dim: int = 7                  # aux target dims (7=WD original, 13=+velocity)
    aux_velocity_topk: int = 0            # 0=no velocity target; 3=nearest 3 obstacles

    # obstacle / scene controls
    obs_lr: float = 3e-4
    obs_ent_coeff: float = 0.1
    obs_speed_limit: float = 0.8
    train_goal_rate: int = 3
    obs_reward_mode: str = "zero"
    max_active_obstacles: int = 10
    obs_size_rand: float = 0.0
    obs_collision_base: float = 0.9
    scene_bound_rand: float = 0.0
    scene_bound_base: float = 7.0
    room_size: float | None = None

    # v3d: act_hist dropout（訓練時隨機 mask 4D 動作歷史，弱化 "copy 上一步" shortcut）
    # 0.0 = 不啟用（v3c 行為）。配合 CHARGE_ACT_HIST_MODE=delta 一起斷 sin 波抽動。
    act_hist_dropout: float = 0.0

    # aux
    aux_seq_len: int = 15
    aux_burn_in: int = 0
    aux_seq_batch_size: int = 256
    aux_grad_clip: float | None = 0.5
    zero_preprocess_feature_for_rl: bool = False

    # safety/logging/runtime switches
    lidar_no_noise: bool = True
    no_domain_randomization: bool = False
    reward_speed_v05: bool = False
    reward_mode: str = "current"
    action_table_sample_size: int = 0
    log_interval: int = 10
    save_interval: int = 100

    # ── Sim-to-Real Domain Randomization (TLNI + Physics + Disturbance) ──

    # LiDAR Layer 1: Per-Ray noise (before min-pool)
    # distance-dependent σ(r) = lidar_displacement_std_per_meter × r  [preferred]
    lidar_displacement_std_per_meter: float = 0.00036  # fitted from VLP-16 measurement: σ(r)≈0.00036·r
    lidar_displacement_std_soft: float = 0.0          # fixed-σ for soft targets (human/clothing); RSS-combined with per_meter
    lidar_displacement_std: float = 0.0               # legacy fixed-σ; use per_meter instead
    lidar_hole_rate: float = 0.20                     # ray dropout rate, VLP-16 measured ~21%
    lidar_distractor_rate: float = 0.002              # ghost/mixed-pixel rate
    lidar_distance_bias: bool = False                 # distance-dependent bias k=0.021, b=-0.030
    lidar_per_ring_bias: bool = False                 # 16ch per-ring calibration offset

    # LiDAR Layer 2: Per-Bin noise (after min-pool)
    lidar_obs_noise_std: float = 0.005           # ObsTerm Gaussian noise σ (replaces Unoise)
    lidar_block_dropout_prob: float = 0.0        # block dropout probability per step
    lidar_block_dropout_width: tuple[int, int] = (3, 8)  # contiguous bin width range

    # LiDAR Layer 3: Per-Episode DR ranges (reset-time sampling)
    # per-meter scale: covers between-robot calibration variation
    lidar_displacement_std_per_meter_dr: tuple[float, float] | None = None  # e.g. (0.00020, 0.00060)
    lidar_displacement_std_soft_dr: tuple[float, float] | None = None       # e.g. (0.002, 0.020) for human targets
    lidar_displacement_std_dr: tuple[float, float] | None = None  # legacy, use per_meter_dr
    lidar_hole_rate_dr: tuple[float, float] | None = None         # e.g. (0.15, 0.30)
    # distance bias DR: per-episode sample (k, b), covers measured ±36mm without assuming model shape
    lidar_distance_bias_k_dr: tuple[float, float] | None = None   # e.g. (-0.010, +0.010) slope range
    lidar_distance_bias_b_dr: tuple[float, float] | None = None   # e.g. (-0.040, +0.040) offset range

    # Physics DR
    physics_mass_dr: tuple[float, float] = (0.85, 1.15)     # mass scale range
    physics_friction_dr: tuple[float, float] = (0.7, 1.3)   # friction scale range
    physics_com_offset: float = 0.05                          # CoM offset ±(m)

    # External disturbance DR
    disturbance_wind_force: tuple[float, float] = (0.0, 3.0)    # continuous wind (N)
    disturbance_push_force: tuple[float, float] = (5.0, 20.0)   # random push (N)
    disturbance_push_ratio: float = 0.10                          # push env fraction

    # Actuator DR (experimental, default off)
    enable_actuator_dr: bool = False
    actuator_delay_range: tuple[int, int] = (0, 2)       # action delay (steps)
    actuator_velocity_scale: tuple[float, float] = (0.9, 1.1)  # velocity scaling
    actuator_motor_lag: float = 0.3                        # first-order lag α

    # Observation latency DR (simulate sensor pipeline delay)
    obs_delay_steps: tuple[int, int] = (0, 0)            # per-env random delay [lo, hi] steps

    # Heading stability reward (penalize angular oscillation)
    heading_stability_weight: float = 0.0                 # 0 = disabled; negative = penalize sign flips

    # RGDR reward-guided loss weighting (focus training on failing envs)
    rgdr_enabled: bool = False                            # enable per-env advantage weighting
    rgdr_weight_clamp: tuple[float, float] = (0.5, 2.0)  # env_weight clamp range

    # DORAEMON auto DR (entropy-maximizing DR expansion)
    doraemon_enabled: bool = False
    doraemon_sr_threshold: float = 0.85                   # SR >= τ → expand DR ranges
    doraemon_check_interval: int = 50                     # iterations between expansion checks
    doraemon_expansion_rate: float = 0.1                  # fraction of remaining range per step

    # checkpoint / resume
    checkpoint: str | None = None
    no_resume_optimizer: bool = True

    # metadata
    tags: tuple[str, ...] = ()
    notes: str = ""

    # --- Derived properties ---

    @property
    def use_a2c(self) -> bool:
        return self.algorithm == "a2c"

    @property
    def disable_aux_training(self) -> bool:
        return self.aux_profile == "none"

    @property
    def algorithm_profile(self) -> str:
        return "a2c_wd" if self.algorithm == "a2c" else "ppo_clip"


# ── Field name → CLI flag mapping ──────────────────────────────────────────
# Most fields map 1:1 to --<field_name>. Exceptions listed here.
_FIELD_TO_FLAGS: dict[str, list[str]] = {
    "algorithm": ["--use_a2c", "--a2c", "--use_ppo", "--ppo", "--algorithm"],
    "normalize_return": ["--normalize_return", "--normalize_returns"],
    "wd_update_clip": ["--wd_update_clip", "--no_wd_update_clip"],
    "fixed_stage": ["--fixed_stage"],
    "lidar_no_noise": ["--lidar_no_noise"],
    "no_resume_optimizer": ["--no_resume_optimizer"],
    "wd_update_monitor_only": ["--wd_update_monitor_only"],
    "zero_preprocess_feature_for_rl": ["--zero_preprocess_feature_for_rl"],
    "no_domain_randomization": ["--no_domain_randomization"],
    "reward_speed_v05": ["--reward_speed_v05"],
    "lidar_distance_bias": ["--lidar_distance_bias"],
    "lidar_per_ring_bias": ["--lidar_per_ring_bias"],
    "enable_actuator_dr": ["--enable_actuator_dr"],
    "rgdr_enabled": ["--rgdr_enabled"],
    "doraemon_enabled": ["--doraemon_enabled"],
}

# Fields that are metadata-only (not applied to args_cli)
_METADATA_ONLY_FIELDS = frozenset({
    "name", "description", "tags", "notes",
})

# Fields that map to a different args_cli attribute name
_FIELD_TO_ATTR: dict[str, str] = {
    "algorithm": "use_a2c",
}

# Derived properties to also apply to args_cli
_DERIVED_PROPERTIES = {
    "use_a2c": ["--use_a2c", "--a2c", "--use_ppo", "--ppo"],
    "disable_aux_training": ["--disable_aux_training"],
    "algorithm_profile": ["--algorithm_profile"],
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

        # If user explicitly provided this flag on CLI, keep their value
        if _was_cli_provided(f.name, argv):
            continue

        attr_name = _FIELD_TO_ATTR.get(f.name, f.name)
        cfg_value = getattr(cfg, f.name)

        # algorithm -> use_a2c conversion
        if f.name == "algorithm":
            setattr(args_cli, "use_a2c", cfg.use_a2c)
        else:
            setattr(args_cli, attr_name, cfg_value)
        applied.append(f.name)

    # Apply derived properties
    for prop_name, flags in _DERIVED_PROPERTIES.items():
        cli_provided = any(
            f in argv or any(a.startswith(f + "=") for a in argv)
            for f in flags
        )
        if not cli_provided and hasattr(args_cli, prop_name):
            setattr(args_cli, prop_name, getattr(cfg, prop_name))

    return applied


def experiment_config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Convert to dict for WandB config / run_metadata. Handles tuples."""
    d = asdict(cfg)
    # Convert tuples to lists for JSON serialization
    if isinstance(d.get("tags"), tuple):
        d["tags"] = list(d["tags"])
    # Add derived fields
    d["use_a2c"] = cfg.use_a2c
    d["disable_aux_training"] = cfg.disable_aux_training
    d["algorithm_profile"] = cfg.algorithm_profile
    return d


def load_experiment_config_from_yaml(path: str) -> ExperimentConfig:
    """Load ExperimentConfig from a YAML file."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML config {path} must be a mapping, got {type(data).__name__}")

    # Convert YAML lists to tuples for all tuple-typed fields
    _tuple_fields = {
        f.name for f in fields(ExperimentConfig)
        if "tuple" in str(f.type)
    }
    for key in _tuple_fields:
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])

    # Remove fields not in ExperimentConfig (comments become None, etc.)
    valid_fields = {f.name for f in fields(ExperimentConfig)}
    unknown = set(data.keys()) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown fields in {path}: {unknown}. "
            f"Valid fields: {sorted(valid_fields)}"
        )

    return ExperimentConfig(**data)


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
