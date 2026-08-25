"""Stage-2 paired A2 arm: equal-allocation continuation from control it25.

This bounded development arm resumes the exact matched-control checkpoint and
embedded PPO optimizer for 25 more iterations. It cannot graduate SA5 or start
SA6.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c50_equalweight_control_p25 import (
    CONFIG as _PARENT,
)


REPO = Path(__file__).resolve().parents[5]

RUN_NAME = "sa5_v3_c50_stage2_equal_ne1024_s42_p25_r1"
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_v3_c50_equalweight_control_ne1024_s42_p25_r1/"
    "checkpoint_3200.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "d9095e7df21633056ef3f491d460520b99dbd72d9924355821c1e48783b3419a"
)

TRAINING_ITERATIONS = 25
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINTS = ((25, "checkpoint_3200.pt"),)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage2_paired_p25_authorization_20260824.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "ae5bc7b00311a61a9c461f2441a6bd32002de23b03e48396cd3a86180d180950"
)
PARENT_COMPLETION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_equalweight_control_p25_completion_20260824.json"
)
PARENT_COMPLETION_SHA256 = (
    "81771f0d7c5d7b10ec90494f8590b4c5a5bf77bcea3ab40a1d1efa1c56807e50"
)
THREE_ARM_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_c50_equalweight_control_screen/"
    "screen_20260824_r1/SUMMARY.json"
)
THREE_ARM_SUMMARY_SHA256 = (
    "db3cb240e6627d6a481eda1aec76952c2c59d6f39c0af0a408402187175514a6"
)
ALLOCATOR_CALIBRATION = (
    REPO
    / "logs/gates/sa5_v3_motion_allocator_calibration/"
    "calibration_20260824_r1/CALIBRATION.json"
)
ALLOCATOR_CALIBRATION_SHA256 = (
    "f32180c16c457993f0553c9e2dcf41d3481aaf7c3fbeebc3f0d6bad6a0529c4e"
)
PARENT_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_equalweight_control_p25.py"
)
PARENT_CONFIG_SHA256 = (
    "210c5e073f2d3433b3b43e9066e3e176428253408125711ff23fed70df9bd64e"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_PARENT_DIFFS = frozenset({"checkpoint"})


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )


for _path, _expected, _label in (
    (Path(PARENT_CHECKPOINT), PARENT_CHECKPOINT_SHA256, "stage-2 parent"),
    (AUTHORIZATION_RECORD, AUTHORIZATION_RECORD_SHA256, "authorization"),
    (PARENT_COMPLETION, PARENT_COMPLETION_SHA256, "parent completion"),
    (THREE_ARM_SUMMARY, THREE_ARM_SUMMARY_SHA256, "three-arm summary"),
    (
        ALLOCATOR_CALIBRATION,
        ALLOCATOR_CALIBRATION_SHA256,
        "allocator calibration",
    ),
    (PARENT_CONFIG, PARENT_CONFIG_SHA256, "parent config"),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_completion = json.loads(PARENT_COMPLETION.read_text(encoding="utf-8"))
_summary = json.loads(THREE_ARM_SUMMARY.read_text(encoding="utf-8"))
_calibration = json.loads(ALLOCATOR_CALIBRATION.read_text(encoding="utf-8"))
if (
    _authorization.get("decision")
    != "RUN_PAIRED_STAGE2_EQUAL_VS_MILD_030_030_040_P25"
    or (_authorization.get("lineage") or {}).get("sa6")
    != "HOLD_NOT_AUTHORIZED"
    or ((_authorization.get("arms") or {}).get("a2_equal") or {}).get(
        "motion_weights"
    )
    is not None
    or ((_authorization.get("arms") or {}).get("a2_equal") or {}).get(
        "timesteps"
    )
    != TRAINING_ITERATIONS * ROLLOUT_LENGTH
    or (_authorization.get("post_training_screen") or {}).get(
        "extension_auto_authorized"
    )
    is not False
    or _completion.get("status")
    != "COMPLETE_VALID_MATCHED_CONTROL_AWAITING_FIXED_SCREEN"
    or (_completion.get("checkpoint") or {}).get("sha256")
    != PARENT_CHECKPOINT_SHA256
    or _summary.get("next_action")
    != "WEIGHT_SPECIFIC_CAPABILITY_REDISTRIBUTION_OBSERVED_REJECT_CURRENT_WEIGHTS"
    or _summary.get("sa6_started") is not False
    or _calibration.get("status") != "COMPLETE_VALID_CPU_CALIBRATION"
    or tuple(_calibration.get("selected_candidate_weights") or ())
    != (0.30, 0.30, 0.40)
):
    raise RuntimeError("frozen evidence does not authorize paired stage-2 A2")

CONFIG = replace(
    _PARENT,
    name="e2e_sa5_v3_c50_stage2_equal_p25",
    description=(
        "Paired stage-2 equal-allocation 25-iteration continuation from the "
        "matched-control it25 checkpoint."
    ),
    checkpoint=PARENT_CHECKPOINT,
    tags=tuple(
        tag
        for tag in _PARENT.tags
        if tag not in {"matched_equalweight_continuation_control"}
    )
    + (
        "paired_stage2_a2_equal",
        "matched_development_anchor_not_parent",
        "bounded_p25",
        "sa6_hold",
    ),
    notes=(
        f"{_PARENT.notes} Authorization {AUTHORIZATION_RECORD_SHA256} resumes "
        "the exact matched-control it25 policy and PPO optimizer for another "
        "25 iterations with the same implicit equal motion allocation. This "
        "is paired A2 only; do not extend, graduate SA5, or launch SA6."
    ),
)

_actual_parent_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_PARENT, field.name)
)
if _actual_parent_diffs != EXPECTED_PARENT_DIFFS:
    raise RuntimeError(
        "paired A2 drifted from parent: expected "
        f"{sorted(EXPECTED_PARENT_DIFFS)}, got {sorted(_actual_parent_diffs)}"
    )

assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 5
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.long_corridor_dynamic_motion_mode == "env_stratified"
assert CONFIG.long_corridor_dynamic_motion_weights is None
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (1, 2)
assert CHECKPOINTS == ((25, "checkpoint_3200.pt"),)

