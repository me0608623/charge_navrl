"""Stage-3 A3 matched continuation control from the rejected B2 anchor."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage2_mild040_p25 import (
    CONFIG as _B2,
)


REPO = Path(__file__).resolve().parents[5]
RUN_NAME = "sa5_v3_c50_stage3_a3_control_ne1024_s42_p25_r1"
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_v3_c50_stage2_mild040_ne1024_s42_p25_r1/checkpoint_3200.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "45c972f804f1e9b24ade9b8164568cd8ac93f33e4f8b8685ee7347723b1aee30"
)
TRAINING_ITERATIONS = 25
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINTS = ((25, "checkpoint_3200.pt"),)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage3_p060_alignment_p25_authorization_20260824.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "f72e9afcc5e840de52ab8571edb0ddfde943c6bbad60334f41fc174a52ccb1dc"
)
PARENT_COMPLETION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage2_b2_mild040_p25_completion_20260824.json"
)
PARENT_COMPLETION_SHA256 = (
    "452926026b680709b91df22cd14938c2dc2be9cc5541b6dae27820d4d4fc45d7"
)
RETENTION_COMPLETION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage2_retention_completion_20260824.json"
)
RETENTION_COMPLETION_SHA256 = (
    "1822e7a350a6112057552c29bdeee39a1d3a93cdd420b26ff3cc6a38c95ab4c1"
)
ANCHOR_AWARE_ANALYSIS = (
    REPO
    / "logs/gates/sa5_v3_c50_stage2_retention_screen/"
    "screen_20260824_r1/ANCHOR_AWARE_ANALYSIS.json"
)
ANCHOR_AWARE_ANALYSIS_SHA256 = (
    "f27624e56df78c369e2118fa64bf9e93e27637a804fcdfab8e1845c6a7c24a60"
)
PARENT_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_stage2_mild040_p25.py"
)
PARENT_CONFIG_SHA256 = (
    "0e4b648c6b80426f746e0eb852ba38991bb4c3b01c37303047413bcb24ee16a7"
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
    (Path(PARENT_CHECKPOINT), PARENT_CHECKPOINT_SHA256, "stage-3 parent"),
    (AUTHORIZATION_RECORD, AUTHORIZATION_RECORD_SHA256, "authorization"),
    (PARENT_COMPLETION, PARENT_COMPLETION_SHA256, "parent completion"),
    (RETENTION_COMPLETION, RETENTION_COMPLETION_SHA256, "retention completion"),
    (
        ANCHOR_AWARE_ANALYSIS,
        ANCHOR_AWARE_ANALYSIS_SHA256,
        "anchor-aware analysis",
    ),
    (PARENT_CONFIG, PARENT_CONFIG_SHA256, "parent config"),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_parent_completion = json.loads(PARENT_COMPLETION.read_text(encoding="utf-8"))
_retention_completion = json.loads(
    RETENTION_COMPLETION.read_text(encoding="utf-8")
)
_analysis = json.loads(ANCHOR_AWARE_ANALYSIS.read_text(encoding="utf-8"))
if (
    _authorization.get("decision")
    != "RUN_PAIRED_STAGE3_CONTROL_VS_P060_ALIGNED_P25"
    or ((_authorization.get("parent") or {}).get("sha256"))
    != PARENT_CHECKPOINT_SHA256
    or ((_authorization.get("post_training_screen") or {}).get("sa6_auto_start"))
    is not False
    or _parent_completion.get("status")
    != "COMPLETE_VALID_PAIRED_B2_AWAITING_FIXED_SCREEN"
    or ((_parent_completion.get("checkpoint") or {}).get("sha256"))
    != PARENT_CHECKPOINT_SHA256
    or _retention_completion.get("status")
    != "COMPLETE_VALID_REJECT_MILD_WEIGHT_FOR_RETENTION_FAILURE"
    or _analysis.get("status")
    != "COMPLETE_VALID_OFFLINE_THREE_GENERATION_ANALYSIS"
    or ((_analysis.get("decision") or {}).get("b2_parent_accepted")) is not False
):
    raise RuntimeError("frozen evidence does not authorize paired stage-3 A3")

CONFIG = replace(
    _B2,
    name="e2e_sa5_v3_c50_stage3_a3_control_p25",
    description=(
        "Stage-3 A3 matched 25-iteration continuation control from the rejected "
        "B2 diagnostic anchor."
    ),
    checkpoint=PARENT_CHECKPOINT,
    tags=tuple(
        tag
        for tag in _B2.tags
        if tag not in {"matched_development_candidate_not_parent"}
    )
    + (
        "paired_stage3_a3_control",
        "rejected_b2_diagnostic_anchor",
        "bounded_p25",
        "sa6_hold",
    ),
    notes=(
        f"{_B2.notes} Authorization {AUTHORIZATION_RECORD_SHA256} resumes the "
        "exact rejected B2 policy and PPO optimizer for a matched 25-iteration "
        "control. No parent promotion, extension, graduation, or SA6 start."
    ),
)

_actual_parent_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_B2, field.name)
)
if _actual_parent_diffs != EXPECTED_PARENT_DIFFS:
    raise RuntimeError(
        "paired A3 drifted from B2: expected "
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
assert CONFIG.long_corridor_dynamic_motion_weights == (0.30, 0.30, 0.40)
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (1, 2)
assert CHECKPOINTS == ((25, "checkpoint_3200.pt"),)
