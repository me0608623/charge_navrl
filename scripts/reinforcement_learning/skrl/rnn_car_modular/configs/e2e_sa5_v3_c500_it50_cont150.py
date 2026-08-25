"""Bounded SA5-v3 continuation from c500 pilot it50 to conceptual it150.

The prior c500 screen verdict remains unchanged. This human-authorized run
resumes the exact it50 checkpoint and optimizer, changes only the parent and
training budget, and preserves every behavioral field from the 50-iteration
c500 pilot.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c500_parent_control_from_sa4v3_p50 import (
    CONFIG as _C500_PILOT,
)


REPO = Path(__file__).resolve().parents[5]

RUN_NAME = "sa5_v3_c500_it50_cont150_ne1024_s42_p100_r1"
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1/"
    "checkpoint_6400.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c"
)
PARENT_CONCEPTUAL_ITERATION = 50

ADDITIONAL_TRAINING_ITERATIONS = 100
CONCEPTUAL_END_ITERATION = 150
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINTS = (
    (75, "checkpoint_3200.pt"),
    (100, "checkpoint_6400.pt"),
    (125, "checkpoint_9600.pt"),
    (150, "checkpoint_12800.pt"),
)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/sa5_v3_c500_it50_cont150_authorization_20260823.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "837227255310974afa7fd855c6f81c75119a632b57ece32b032c5a6473781709"
)
C500_SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c500_parent_control_from_sa4v3_p50.py"
)
C500_SOURCE_CONFIG_SHA256 = (
    "ee7990c162da118571910d47d3d73c530b63595c0ef14a404703ab57bc2005f1"
)
PRIOR_SCREEN_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_c500_parent_control_screen/"
    "screen_20260823_r1/SUMMARY.json"
)
PRIOR_SCREEN_SUMMARY_SHA256 = (
    "35417943f31191496b56a24b5d0ba939d3e40ca754ed597d6643ced40f8a79e7"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_BEHAVIORAL_DIFFS = frozenset({"checkpoint", "timesteps"})


def _verify_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )


_parent_path = Path(PARENT_CHECKPOINT)
for _path, _expected, _label in (
    (_parent_path, PARENT_CHECKPOINT_SHA256, "c500 it50 parent"),
    (
        AUTHORIZATION_RECORD,
        AUTHORIZATION_RECORD_SHA256,
        "bounded-continuation authorization",
    ),
    (C500_SOURCE_CONFIG, C500_SOURCE_CONFIG_SHA256, "c500 pilot recipe"),
    (
        PRIOR_SCREEN_SUMMARY,
        PRIOR_SCREEN_SUMMARY_SHA256,
        "prior c500 screen summary",
    ),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_lineage = _authorization.get("lineage_status") or {}
_parent = _authorization.get("parent") or {}
_identity = _authorization.get("identity_contract") or {}
_continuation = _authorization.get("continuation") or {}
if (
    _authorization.get("decision")
    != "HUMAN_AUTHORIZED_BOUNDED_CONTINUATION_C500_IT50_TO_IT150"
    or _lineage.get("sa4") != "SA4_NOT_GRADUATED"
    or _lineage.get("sa6") != "HOLD_NOT_AUTHORIZED"
    or _parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or (REPO / str(_parent.get("checkpoint"))).resolve()
    != _parent_path.resolve()
    or _parent.get("optimizer_resume") is not True
    or _identity.get("source_config_sha256") != C500_SOURCE_CONFIG_SHA256
    or set(_identity.get("allowed_behavioral_differences") or ())
    != EXPECTED_BEHAVIORAL_DIFFS
    or _continuation.get("additional_training_iterations")
    != ADDITIONAL_TRAINING_ITERATIONS
    or _continuation.get("conceptual_end_iteration")
    != CONCEPTUAL_END_ITERATION
    or _continuation.get("conceptual_save_iterations")
    != [row[0] for row in CHECKPOINTS]
):
    raise RuntimeError("authorization does not permit c500 it50 continuation")

_prior_summary = json.loads(PRIOR_SCREEN_SUMMARY.read_text(encoding="utf-8"))
_amendment = _authorization.get("protocol_amendment") or {}
if (
    _amendment.get("prior_summary_sha256") != PRIOR_SCREEN_SUMMARY_SHA256
    or _prior_summary.get("status") != _amendment.get("prior_status")
    or bool(_prior_summary.get("extension_authorized"))
    is not _amendment.get("prior_extension_authorized")
    or _prior_summary.get("next_action")
    != _amendment.get("prior_next_action")
):
    raise RuntimeError("prior c500 verdict does not match the amendment trigger")

CONFIG = replace(
    _C500_PILOT,
    name="e2e_sa5_v3_c500_it50_cont150",
    description=(
        "Human-authorized bounded SA5-v3 continuation from c500 pilot it50 "
        "to conceptual it150 with an unchanged behavioral contract."
    ),
    checkpoint=PARENT_CHECKPOINT,
    timesteps=ADDITIONAL_TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    tags=tuple(
        tag
        for tag in _C500_PILOT.tags
        if tag
        not in {
            "provisional_sa5_control_parent_c500",
            "preregistered_c550_fallback",
        }
    )
    + (
        "bounded_continuation_c500_it50_to_it150",
        "prior_fail_preserved",
        "sa6_hold",
    ),
    notes=(
        f"{_C500_PILOT.notes} New authorization "
        f"{AUTHORIZATION_RECORD_SHA256} resumes c500 pilot it50 "
        f"({PARENT_CHECKPOINT_SHA256}) with optimizer state. Only checkpoint "
        "and timesteps differ; action, observation, reward, scene mix, LiDAR "
        "noise, actuator delay, model, PPO, seed, rollout, and save interval "
        "remain identical. Stop at conceptual it150; do not launch SA6."
    ),
)

_actual_behavioral_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_C500_PILOT, field.name)
)
if _actual_behavioral_diffs != EXPECTED_BEHAVIORAL_DIFFS:
    raise RuntimeError(
        "bounded continuation drifted from c500 pilot: expected "
        f"{sorted(EXPECTED_BEHAVIORAL_DIFFS)}, got "
        f"{sorted(_actual_behavioral_diffs)}"
    )

assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 5
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 12800
assert CONFIG.save_interval == SAVE_INTERVAL
assert CHECKPOINTS == (
    (75, "checkpoint_3200.pt"),
    (100, "checkpoint_6400.pt"),
    (125, "checkpoint_9600.pt"),
    (150, "checkpoint_12800.pt"),
)
