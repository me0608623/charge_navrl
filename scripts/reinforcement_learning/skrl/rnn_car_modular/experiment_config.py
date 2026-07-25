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
    critic_detach_encoder: bool = False

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
    target_kl: float = 0.0
    value_clip_eps: float = 0.0

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
    lidar_frame_stack: int = 1
    end_to_end_frame_stack: bool = False
    predict_dim: int = 7                  # aux target dims (7=WD original, 13=+velocity)
    aux_velocity_topk: int = 0            # 0=no velocity target; 3=nearest 3 obstacles
    aux_loss_type: str = "log"           # log=WD original (grad ∝ 1/|e|); huber=smooth-L1 (大誤差大梯度)
    aux_huber_delta: float = 1.0          # Huber 轉折點 δ
    aux_reinit_frozen: bool = False      # 載入時跳過 fc_middle/fc_front/predict_head(隨機 readout)
    aux_skip_input: bool = False         # predict_head 直接 concat extractor 輸入(繞過 RNN)
    aux_target_pos_scale: float = 1.0    # WD-diff #2:位置 target 縮放(0.33→std 3m 縮到 ~unit,配 huber 抗常數陷阱)
    aux_cpc: bool = False                # CPC/InfoNCE contrastive aux(常數 hidden 打不贏對比任務)
    aux_cpc_dim: int = 64                # CPC proj 維度
    aux_cpc_temp: float = 0.1            # InfoNCE 溫度 τ
    aux_cpc_lr: float = 5e-4             # CPC optimizer lr(訓 fc_front+rnn+extractor+proj)
    aux_cpc_max_samples: int = 2048      # CPC 每步最大樣本數(logits 矩陣)
    aux_epochs: int = 1                  # 每 iter aux 更新次數(>1 複製離線多 epoch 梯度密度)
    feat_norm: bool = False              # extractor 輸出 per-dim running 正規化再進 RNN(離線證關鍵缺件)
    aux_zero_h0: bool = False            # aux 用 h0=0(對標離線 fresh hidden,逼 GRU 從特徵萃取)

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
    # Fixed SA5 general-scene replay for later stages. This changes only the
    # selected envs' obstacle/wall layout; current-stage reward and horizon stay.
    previous_stage_replay_fraction: float = 0.0
    previous_stage_replay_static_obstacles: int = 10
    previous_stage_replay_dynamic_obstacles: int = 3
    previous_stage_replay_min_walls: int = 2
    previous_stage_replay_max_walls: int = 3
    previous_stage_replay_wall_length: float = 4.0
    previous_stage_replay_obstacle_boundary: float = 5.5
    # Optional fixed-stage narrow-passage bridge. A zero fraction is a strict
    # no-op and does not add bridge assets or alter reset behavior.
    narrow_passage_fraction: float = 0.0
    narrow_passage_schedule_steps: int = 19200
    narrow_passage_segment_length: float = 9.0
    narrow_passage_final_stress_ratio: float = 0.25
    narrow_passage_fixed_width_range: tuple[float, float] | None = None
    narrow_passage_fixed_yaw_limit_deg: float | None = None
    narrow_passage_exact_width: float | None = None
    narrow_passage_exact_width_ratio: float = 0.0
    # Deployment corridor replay is disjoint from narrow-passage replay.
    long_corridor_fraction: float = 0.0
    long_corridor_free_width: float = 4.0
    long_corridor_length: float = 10.0
    long_corridor_static_obstacles: int = 4
    long_corridor_dynamic_obstacles: int = 2
    long_corridor_dynamic_speed_range: tuple[float, float] = (0.30, 0.60)
    # Frozen successful policy used only on narrow-passage replay frames.
    teacher_retention_checkpoint: str | None = None
    teacher_retention_weight: float = 0.0
    teacher_retention_margin_weight: float = 0.0
    teacher_retention_action_ce_weight: float = 0.0
    teacher_retention_argmax_margin: float = 0.2
    teacher_retention_post_kl_epochs: int = 0
    teacher_retention_post_kl_lr: float = 1e-3
    teacher_retention_post_kl_batch_size: int = 4096
    teacher_retention_post_kl_max_grad_norm: float = 0.5
    teacher_retention_post_margin_weight: float = 0.0
    teacher_retention_post_action_ce_weight: float = 0.0
    teacher_retention_post_policy_head_only: bool = False
    teacher_retention_post_anchor_weight: float = 0.0
    teacher_retention_rollout_override: bool = False
    # Frozen SA5 Pareto teacher used only on previous-stage replay frames.
    # This is independent of the narrow-passage c20 teacher above.
    previous_stage_teacher_checkpoint: str | None = None
    previous_stage_teacher_retention_weight: float = 0.0
    previous_stage_teacher_scope: str = "previous_stage"
    # Privileged teacher labels are generated only on deployment-corridor
    # replay frames and applied as a post-PPO action projection.
    corridor_teacher_distill_epochs: int = 0
    corridor_teacher_distill_lr: float = 5e-4
    corridor_teacher_distill_batch_size: int = 4096
    corridor_teacher_distill_max_grad_norm: float = 0.5
    corridor_teacher_distill_neighbor_mass: float = 0.20
    corridor_teacher_distill_stride: int = 2
    corridor_teacher_distill_chunk_size: int = 32
    corridor_teacher_intervention_only: bool = False
    corridor_teacher_intervention_clearance_m: float = 0.20
    # Deployable observation-gated residual policy. The corridor scene label
    # supervises the gate only; inference consumes the current policy obs.
    corridor_adapter_enabled: bool = False
    corridor_adapter_hidden_dim: int = 64
    corridor_adapter_gate_loss_weight: float = 0.05
    corridor_adapter_gate_init_probability: float = 0.01
    corridor_adapter_max_logit_delta: float = 2.0
    corridor_adapter_freeze_base: bool = False
    corridor_adapter_gate_checkpoint: str | None = None
    corridor_adapter_residual_features: str = "current_obs"

    # Observation layout must be fixed before isaaclab_tasks is imported.
    # None preserves the process environment for legacy configs.
    use_action_history: bool | None = None

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
    use_obb_collision: bool = False
    reward_speed_v05: bool = False
    reward_mode: str = "current"
    # react clearance-gated 減速稅 (距離稅). -1.0 = 不覆寫 curriculum(維持舊行為);
    # >=0.0 = 覆寫 curriculum 的 spot_penalty_speed_near_obs. 對應 --penalty_speed_near_obs
    # (CLI 仍可覆寫此 YAML 值). ★寫進 config 消除 CLI-only footgun(忘帶=稅靜默關閉).
    penalty_speed_near_obs: float = -1.0
    anti_spin_weight: float = 0.0
    anti_spin_hazard_distance: float = 1.5
    anti_spin_omega_threshold: float = 0.8
    anti_spin_progress_threshold: float = 0.02
    anti_spin_grace_steps: int = 5
    anti_spin_ramp_steps: int = 5
    anti_spin_yaw_grace_deg: float = 180.0
    anti_spin_yaw_ramp_deg: float = 180.0
    anti_spin_dt: float = 0.2
    future_occupancy_weight: float = 0.0
    future_occupancy_horizon_s: float = 1.5
    future_occupancy_samples: int = 8
    future_occupancy_safe_distance_m: float = 1.0
    future_occupancy_near_distance_m: float = 3.0
    future_occupancy_move_threshold_mps: float = 0.1
    action_table_sample_size: int = 0
    log_interval: int = 10
    save_interval: int = 100

    # ── Sim-to-Real Domain Randomization (TLNI + Physics + Disturbance) ──

    # LiDAR Layer 1: Per-Ray noise (before min-pool)
    # ── Defaults calibrated to measured VLP-16 white_wall data (2026-07-01) ──
    #    source: vlp16_noise/isaac_lab_noise_params.py. σ is FIXED (R²=0.078 → distance-independent),
    #    so per_meter=0 and the fixed σ lives in displacement_std_soft.
    lidar_displacement_std_per_meter: float = 0.0     # measured slope≈0 (σ not distance-dependent)
    lidar_displacement_std_soft: float = 0.008672     # measured fixed σ = 8.67mm (point-to-plane residual std)
    lidar_displacement_std: float = 0.0               # legacy fixed-σ; use soft instead
    lidar_hole_rate: float = 0.194859                 # measured dropout rate (intensity<thr), ~19.5%
    lidar_distractor_rate: float = 0.002515           # measured mixed-pixel / ghost rate
    lidar_distance_bias: bool = False                 # keep OFF: per_ring_bias carries systematic bias (no double-count)
    lidar_per_ring_bias: bool = False                 # 16ch measured per-ring calibration offset (mean ~+12.9mm)

    # LiDAR Layer 2: Per-Bin noise (after min-pool)
    # ⚠ Removed the old 6cm blanket (0.005 norm × 20m): not in the measured model and it buried the
    #    real 8.67mm L1 σ (~7×). Default 0 → measured L1 σ is the honest range noise.
    lidar_obs_noise_std: float = 0.0             # ObsTerm Gaussian noise σ (0 = off; measured model has none)
    lidar_block_dropout_prob: float = 0.0        # block dropout probability per step (scene occlusion aug, not sensor)
    lidar_block_dropout_width: tuple[int, int] = (3, 8)  # contiguous bin width range

    # VLP-16 empirical-noise ablation switch (README §5): overrides the fine-grained lidar_* params
    # above with the measured preset for the chosen component. None = use fine-grained params as-is.
    #   ideal   → clean rays (no noise, no bias)
    #   sigma   → only measured N(0, 8.67mm)
    #   bias    → only measured per-ring systematic bias
    #   dropout → only measured hole (19.5%) + mixed-pixel (0.25%)
    #   full    → sigma + bias + dropout (deployment / sim-to-real)
    #   full_material → full + measured HUMAN dropout(d) on dynamic-obstacle rays (Phase 2)
    vlp16_noise_mode: str | None = None

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
    "no_resume_optimizer": ["--no_resume_optimizer", "--resume_optimizer"],
    "wd_update_monitor_only": ["--wd_update_monitor_only"],
    "zero_preprocess_feature_for_rl": ["--zero_preprocess_feature_for_rl"],
    "no_domain_randomization": ["--no_domain_randomization"],
    "use_obb_collision": ["--use_obb_collision"],
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
