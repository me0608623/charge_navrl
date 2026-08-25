"""Identical 50-iteration SA5-v3 parent control from SA4-v3 c500.

This fallback is activated only because the preregistered c550 pilot screen did
not authorize extension. It inherits the complete c550 SA5 recipe and changes
only the parent checkpoint plus descriptive metadata.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_provisional_from_sa4v3_c550_p50 import (
    CHECKPOINTS,
    CONFIG as _C550_PILOT,
    ROLLOUT_LENGTH,
    SA5_SPEED_DENSITY_MIX,
    TRAINING_ITERATIONS,
)


REPO = Path(__file__).resolve().parents[5]

RUN_NAME = "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1"
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/checkpoint_25600.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "e27677349eba8f37e2e6937f7d92400a1550e58d89d4430e556932697e9cb9f4"
)
PARENT_CONCEPTUAL_ITERATION = 500

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/sa5_v3_c500_parent_control_authorization_20260823.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "87201f68f7c9c082f7cfd2125bf0b4fda9731f7e2b60ad59439b14609560d3be"
)
C550_SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_provisional_from_sa4v3_c550_p50.py"
)
C550_SOURCE_CONFIG_SHA256 = (
    "64e3c5ed66ef1d2f3bb17394b8e3b1ad6e283e83d428a88455fa4089d3c9c520"
)
C550_SCREEN_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_provisional_pilot_screen/"
    "screen_20260822_r1/SUMMARY.json"
)
C550_SCREEN_SUMMARY_SHA256 = (
    "4cd82cc3912b7ff70a053ca9203594899d5871e3f72fa65cbbecd6065069f744"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_BEHAVIORAL_DIFFS = frozenset({"checkpoint"})


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
    (_parent_path, PARENT_CHECKPOINT_SHA256, "SA4-v3 c500 parent"),
    (
        AUTHORIZATION_RECORD,
        AUTHORIZATION_RECORD_SHA256,
        "c500 control authorization",
    ),
    (C550_SOURCE_CONFIG, C550_SOURCE_CONFIG_SHA256, "c550 pilot recipe"),
    (
        C550_SCREEN_SUMMARY,
        C550_SCREEN_SUMMARY_SHA256,
        "c550 pilot screen trigger",
    ),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_trigger = _authorization.get("trigger") or {}
_parent = _authorization.get("parent") or {}
_recipe = _authorization.get("identical_recipe") or {}
_pilot = _authorization.get("pilot") or {}
if (
    _authorization.get("decision")
    != "RUN_IDENTICAL_C500_PARENT_CONTROL_P50"
    or (_authorization.get("lineage_status") or {}).get("sa4")
    != "SA4_NOT_GRADUATED"
    or (_authorization.get("lineage_status") or {}).get("control_parent")
    != "PROVISIONAL_SA5_CONTROL_PARENT_C500"
    or _parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or (REPO / str(_parent.get("checkpoint"))).resolve()
    != _parent_path.resolve()
    or _parent.get("optimizer_resume") is not True
    or _recipe.get("source_config_sha256") != C550_SOURCE_CONFIG_SHA256
    or _recipe.get("allowed_behavioral_differences") != ["checkpoint"]
    or _pilot.get("training_iterations") != TRAINING_ITERATIONS
    or _pilot.get("save_iterations") != [25, 50]
):
    raise RuntimeError("authorization does not permit the c500 parent control")

_c550_summary = json.loads(C550_SCREEN_SUMMARY.read_text(encoding="utf-8"))
if (
    _trigger.get("summary_sha256") != C550_SCREEN_SUMMARY_SHA256
    or _c550_summary.get("status") != _trigger.get("required_status")
    or bool(_c550_summary.get("extension_authorized"))
    is not _trigger.get("required_extension_authorized")
    or _c550_summary.get("next_action") != _trigger.get("required_next_action")
):
    raise RuntimeError("c550 screen does not authorize the c500 fallback")

_retained_tags = tuple(
    tag
    for tag in _C550_PILOT.tags
    if tag not in {"provisional_sa5_parent_c550", "human_authorized_bounded_pilot"}
)

CONFIG = replace(
    _C550_PILOT,
    name="e2e_sa5_v3_c500_parent_control_from_sa4v3_p50",
    description=(
        "Identical bounded SA5-v3 parent control from non-graduated SA4-v3 "
        "c500, triggered by the failed c550 pilot screen."
    ),
    checkpoint=PARENT_CHECKPOINT,
    tags=_retained_tags
    + (
        "provisional_sa5_control_parent_c500",
        "preregistered_c550_fallback",
        "identical_recipe_control",
    ),
    notes=(
        f"{_C550_PILOT.notes} This control substitutes only the parent with "
        f"SA4-v3 c500 ({PARENT_CHECKPOINT_SHA256}); every behavioral and "
        "training field otherwise remains identical to the frozen c550 "
        "50-iteration pilot. Do not auto-extend and do not launch SA6."
    ),
)

_actual_behavioral_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_C550_PILOT, field.name)
)
if _actual_behavioral_diffs != EXPECTED_BEHAVIORAL_DIFFS:
    raise RuntimeError(
        "c500 control is not identical to c550 recipe: expected only "
        f"checkpoint, got {sorted(_actual_behavioral_diffs)}"
    )

assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 5
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH == 128
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH == 6400
assert CONFIG.save_interval == 25
assert CONFIG.long_corridor_speed_density_mix == SA5_SPEED_DENSITY_MIX
assert CHECKPOINTS == ((25, "checkpoint_3200.pt"), (50, "checkpoint_6400.pt"))
