"""Exact B3 continuation from conceptual it25 to it50.

Only the checkpoint, training budget, and save interval differ from B3. The
checkpoint includes the original PPO optimizer and ``no_resume_optimizer``
remains false, so this is a continuation rather than a warm restart.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_p060aligned_p25 import (
    CONFIG as _B3,
)


REPO = Path(__file__).resolve().parents[5]
RUN_NAME = "sa5_v3_c50_stage3_b3_cont25_from_it25_ne1024_s42_p25_r1"
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_v3_c50_stage3_b3_p060aligned_ne1024_s42_p25_r1/"
    "checkpoint_3200.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "5645f84625649a6050e862eb17b8e81e2e02225973fd371f29e663785104f2b8"
)
PARENT_CONCEPTUAL_ITERATION = 25
ADDITIONAL_TRAINING_ITERATIONS = 25
CONCEPTUAL_END_ITERATION = 50
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 5
CHECKPOINTS = (
    (30, "checkpoint_640.pt"),
    (35, "checkpoint_1280.pt"),
    (40, "checkpoint_1920.pt"),
    (45, "checkpoint_2560.pt"),
    (50, "checkpoint_3200.pt"),
)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/sa5_v3_c50_stage3_b3_cont25_authorization_20260824.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "664ecb32476ccffdf6bbd337e27c39f72222ebe2aa35ec43d27065005c052f13"
)
B3_SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_stage3_b3_p060aligned_p25.py"
)
B3_SOURCE_CONFIG_SHA256 = (
    "2040a9d9e154d447756763d95a6351c68816f941dc42dcee07c3cf4387740498"
)
PRIOR_PHASE_A_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_c50_stage3_phase_a_screen/"
    "screen_20260824_r1/SUMMARY.json"
)
PRIOR_PHASE_A_SUMMARY_SHA256 = (
    "cf7076561df1c22f288e0958232627dd12ab2774e074db329a8a6284b2aa7b71"
)
TRAINER = REPO / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"
TRAINER_SHA256 = (
    "4209725a07954631a2490cedfd925633e5adb95aec8373c4a2160516b3349090"
)
CORRIDOR_FAMILY_METRICS = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean/"
    "corridor_family_metrics.py"
)
CORRIDOR_FAMILY_METRICS_SHA256 = (
    "76da80bbd5d1b782631f16474ff1b5d0df3b7711cf290ef720ae7e93967db8c3"
)
CORRIDOR_PROFILE_FAMILY_METRICS = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_wdclean/"
    "corridor_profile_family_metrics.py"
)
CORRIDOR_PROFILE_FAMILY_METRICS_SHA256 = (
    "96c1d7248e5bf88eb10186f9ecfef8aa4c50298bdf321e3ee4e1fb2e7b330eed"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
AUTHORIZED_BEHAVIORAL_DIFFS = frozenset(
    {"checkpoint", "timesteps", "save_interval"}
)
EXPECTED_ACTUAL_DIFFS = frozenset({"checkpoint", "save_interval"})


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
    (Path(PARENT_CHECKPOINT), PARENT_CHECKPOINT_SHA256, "B3 it25 checkpoint"),
    (AUTHORIZATION_RECORD, AUTHORIZATION_RECORD_SHA256, "authorization"),
    (B3_SOURCE_CONFIG, B3_SOURCE_CONFIG_SHA256, "B3 source config"),
    (PRIOR_PHASE_A_SUMMARY, PRIOR_PHASE_A_SUMMARY_SHA256, "prior Phase-A summary"),
    (TRAINER, TRAINER_SHA256, "trainer with profile accounting"),
    (
        CORRIDOR_FAMILY_METRICS,
        CORRIDOR_FAMILY_METRICS_SHA256,
        "corridor-family accounting",
    ),
    (
        CORRIDOR_PROFILE_FAMILY_METRICS,
        CORRIDOR_PROFILE_FAMILY_METRICS_SHA256,
        "corridor profile-family accounting",
    ),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_parent = _authorization.get("parent") or {}
_identity = _authorization.get("identity_contract") or {}
_monitoring = _authorization.get("monitoring_contract") or {}
_continuation = _authorization.get("continuation") or {}
_lineage = _authorization.get("lineage") or {}
if (
    _authorization.get("decision")
    != "HUMAN_AUTHORIZED_B3_EXACT_CONTINUATION_IT25_TO_IT50"
    or _parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or (REPO / str(_parent.get("checkpoint"))).resolve()
    != Path(PARENT_CHECKPOINT).resolve()
    or _parent.get("optimizer_resume") is not True
    or _identity.get("source_config_sha256") != B3_SOURCE_CONFIG_SHA256
    or set(_identity.get("allowed_behavioral_differences") or ())
    != AUTHORIZED_BEHAVIORAL_DIFFS
    or _monitoring.get("fail_closed") is not True
    or _monitoring.get("trainer_sha256") != TRAINER_SHA256
    or _monitoring.get("corridor_family_metrics_sha256")
    != CORRIDOR_FAMILY_METRICS_SHA256
    or _monitoring.get("corridor_profile_family_metrics_sha256")
    != CORRIDOR_PROFILE_FAMILY_METRICS_SHA256
    or _continuation.get("additional_training_iterations")
    != ADDITIONAL_TRAINING_ITERATIONS
    or _continuation.get("conceptual_end_iteration") != CONCEPTUAL_END_ITERATION
    or _continuation.get("conceptual_checkpoints")
    != [list(row) for row in CHECKPOINTS]
    or _lineage.get("phase_b_auto_start") is not False
    or _lineage.get("sa6_auto_start") is not False
):
    raise RuntimeError("authorization does not permit the B3 it25 continuation")

CONFIG = replace(
    _B3,
    name="e2e_sa5_v3_c50_stage3_b3_cont25_from_it25",
    description=(
        "Human-authorized exact B3 continuation from conceptual it25 to it50 "
        "with profile-by-family accounting."
    ),
    checkpoint=PARENT_CHECKPOINT,
    timesteps=ADDITIONAL_TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_B3.tags
    + (
        "exact_optimizer_continuation_it25_to_it50",
        "profile_family_accounting_only",
        "phase_b_hold",
        "sa6_hold",
    ),
    notes=(
        f"{_B3.notes} Authorization {AUTHORIZATION_RECORD_SHA256} resumes B3 "
        f"it25 ({PARENT_CHECKPOINT_SHA256}) with its PPO optimizer. Monitoring "
        "adds exact profile-by-family accounting but cannot modify policy "
        "behavior. Stop at conceptual it50; do not launch Phase B or SA6."
    ),
)

_actual_behavioral_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_B3, field.name)
)
if _actual_behavioral_diffs != EXPECTED_ACTUAL_DIFFS:
    raise RuntimeError(
        "B3 continuation drifted: expected "
        f"{sorted(EXPECTED_ACTUAL_DIFFS)}, got "
        f"{sorted(_actual_behavioral_diffs)}"
    )

assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 5 and CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024 and CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 3200
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.speed_rate == 0.7 and CONFIG.speed_rate_obs == "ego"
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.long_corridor_speed_density_mix == _B3.long_corridor_speed_density_mix
