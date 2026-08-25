"""SA5-R2 c250 continuation pilot at the deployed 0.7 speed rate.

This is a learnability pilot, not a new SA5 graduation claim. It resumes the
exact c250 model and optimizer and changes only the vehicle speed-rate contract
plus the declared 50-iteration continuation budget.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from pathlib import Path

import torch

from rnn_car_modular.experiment_config import ExperimentConfig


RUN_NAME = "sa5_r2_c250_speed0p7_adapt_ne1024_s42_p50_r1"
PARENT_CONCEPTUAL_ITERATION = 250
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_r2_sealed_ldmix_from_sa4r3_it125_actd12_ne1024_s42_p300_r1/"
    "checkpoint_32000.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "df14c45d8dd27b0e327f9447a9a613bf62eaa850af542b474aad682dcd937a4a"
)
RATE_1P0_CONTINUATION_CONTROL = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_r2_sealed_ldmix_from_sa4r3_it125_actd12_ne1024_s42_p300_r1/"
    "checkpoint_38400.pt"
)
RATE_1P0_CONTINUATION_CONTROL_SHA256 = (
    "97a4ccdf929a0fcab9adff5d80ae7a1c7475067e3049f11571e04fd164a524db"
)
ROLLOUT_LENGTH = 128
TRAINING_ITERATIONS = 50
SAVE_INTERVAL = 25
CHECKPOINTS = (
    (25, "checkpoint_3200.pt"),
    (50, "checkpoint_6400.pt"),
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
CONTINUATION_FIELDS = frozenset(
    {
        "checkpoint",
        "no_resume_optimizer",
        "timesteps",
        "save_interval",
    }
)
INTERVENTION_FIELDS = frozenset({"speed_rate"})
EXPECTED_DIFFS = METADATA_FIELDS | CONTINUATION_FIELDS | INTERVENTION_FIELDS


def _sha256(path: str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _config_saved_in_parent(path: str) -> ExperimentConfig:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = payload.get("args")
    if not isinstance(saved_args, dict):
        raise RuntimeError("c250 parent checkpoint has no saved args contract")
    field_names = {field.name for field in fields(ExperimentConfig)}
    resolved = {
        name: value
        for name, value in saved_args.items()
        if name in field_names
    }
    # ``algorithm`` is represented by the derived CLI flag ``use_a2c`` in
    # checkpoint args, so it must be restored explicitly instead of falling
    # back to ExperimentConfig's historical A2C default.
    if "use_a2c" not in saved_args:
        raise RuntimeError("c250 parent checkpoint has no use_a2c algorithm flag")
    resolved["algorithm"] = "a2c" if saved_args["use_a2c"] else "ppo"
    return ExperimentConfig(**resolved)


assert Path(PARENT_CHECKPOINT).is_file()
assert _sha256(PARENT_CHECKPOINT) == PARENT_CHECKPOINT_SHA256
assert Path(RATE_1P0_CONTINUATION_CONTROL).is_file()
assert (
    _sha256(RATE_1P0_CONTINUATION_CONTROL)
    == RATE_1P0_CONTINUATION_CONTROL_SHA256
)

# Reconstruct the exact resolved SA5-R2 contract saved inside c250. Importing
# the historical Python config here would correctly fail its frozen trainer
# source hash after adding this new speed-rate training feature.
_SA5_R2 = _config_saved_in_parent(PARENT_CHECKPOINT)

CONFIG = replace(
    _SA5_R2,
    name="e2e_sa5_r2_c250_speed0p7_adapt_p50",
    description=(
        "SA5-R2 c250 exact model+optimizer continuation for 50 iterations at "
        "the deployed speed_rate=0.7 ego-observation semantics."
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=False,
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    speed_rate=0.7,
    speed_rate_obs="ego",
    tags=(
        "c250_exact_optimizer_resume",
        "speed_rate_0p7",
        "speed_rate_learnability_pilot",
        "no_sa6_auto_advance",
    ),
    notes=(
        "Only causal intervention versus an equal-budget rate=1 continuation "
        "is vehicle speed_rate=0.7 with deployed ego observation scaling and "
        "unscaled LiDAR. The already-completed same-lineage c300 checkpoint is "
        "the natural +50-iteration rate=1 control. Save adaptation it25/it50; "
        "validate lateral, longitudinal, native, and narrow before any longer "
        "training or SA6 decision."
    ),
)


assert PARENT_CONCEPTUAL_ITERATION == 250
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.algorithm == "ppo"
assert CONFIG.use_a2c is False
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.initial_stage == 5
assert CONFIG.rollout_length == ROLLOUT_LENGTH == 128
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH == 6_400
assert CONFIG.save_interval == SAVE_INTERVAL == 25
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.enable_actuator_dr is True
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
assert CONFIG.actuator_motor_lag == 1.0
assert CHECKPOINTS == (
    (25, "checkpoint_3200.pt"),
    (50, "checkpoint_6400.pt"),
)

_drifted = {
    field.name
    for field in fields(CONFIG)
    if getattr(CONFIG, field.name) != getattr(_SA5_R2, field.name)
}
if _drifted != EXPECTED_DIFFS:
    raise RuntimeError(
        "SA5-R2 c250 speed0p7 pilot drifted outside metadata, continuation "
        f"budget, parent/resume, and speed_rate; got {sorted(_drifted)}"
    )
