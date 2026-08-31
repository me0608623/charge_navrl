"""Exact B3 continuation from conceptual it100 to it600."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_cont50_from_it50 import (
    CONFIG as _IT100,
)


REPO = Path(__file__).resolve().parents[5]
RUN_NAME = "sa5_v3_c50_stage3_b3_cont500_from_it100_ne1024_s42_p500_r1"
PARENT_CHECKPOINT = (
    REPO
    / "logs/rnn_car/"
    "sa5_v3_c50_stage3_b3_cont50_from_it50_ne1024_s42_p50_r1/"
    "checkpoint_6400.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "c163f70e17412ed1a8954e5896587e09b0047ae599e30649f0daa54e77a15244"
)
PARENT_CONCEPTUAL_ITERATION = 100
ADDITIONAL_TRAINING_ITERATIONS = 500
CONCEPTUAL_END_ITERATION = 600
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 50
CHECKPOINTS = (
    (150, "checkpoint_6400.pt"),
    (200, "checkpoint_12800.pt"),
    (250, "checkpoint_19200.pt"),
    (300, "checkpoint_25600.pt"),
    (350, "checkpoint_32000.pt"),
    (400, "checkpoint_38400.pt"),
    (450, "checkpoint_44800.pt"),
    (500, "checkpoint_51200.pt"),
    (550, "checkpoint_57600.pt"),
    (600, "checkpoint_64000.pt"),
)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage3_b3_cont500_from_it100_authorization_20260825.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "865b9f5d367a51914643c64c34e0e1a37c9a1b197708dc966a885113d5252589"
)
SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_stage3_b3_cont50_from_it50.py"
)
SOURCE_CONFIG_SHA256 = (
    "79a6ec38ea4f296d2b50fc131b11db33742eda16ce083ed524b8bf5e2f223f41"
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
LONG_CORRIDOR_REPLAY = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/"
    "velocity/config/charge_skrl/mdp/events/long_corridor_replay.py"
)
LONG_CORRIDOR_REPLAY_SHA256 = (
    "9e6e1f6f922daede1c58343f93e12b279ff3798f55449287ca13eb99094c0067"
)
LONG_CORRIDOR_GEOMETRY = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/"
    "velocity/config/charge_skrl/mdp/events/long_corridor_replay_geometry.py"
)
LONG_CORRIDOR_GEOMETRY_SHA256 = (
    "f57f4351573af93cd3e42d60554ba9d77956a49ed5614a9df5d2b40b646466a4"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_DIFFS = frozenset({"checkpoint", "timesteps", "save_interval"})


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
    (PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, "B3 conceptual it100 checkpoint"),
    (AUTHORIZATION_RECORD, AUTHORIZATION_RECORD_SHA256, "authorization"),
    (SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "it100 source config"),
    (TRAINER, TRAINER_SHA256, "trainer"),
    (CORRIDOR_FAMILY_METRICS, CORRIDOR_FAMILY_METRICS_SHA256, "family metrics"),
    (
        CORRIDOR_PROFILE_FAMILY_METRICS,
        CORRIDOR_PROFILE_FAMILY_METRICS_SHA256,
        "profile-family metrics",
    ),
    (LONG_CORRIDOR_REPLAY, LONG_CORRIDOR_REPLAY_SHA256, "corridor replay"),
    (LONG_CORRIDOR_GEOMETRY, LONG_CORRIDOR_GEOMETRY_SHA256, "corridor geometry"),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_parent = _authorization.get("parent") or {}
_identity = _authorization.get("identity_contract") or {}
_source = _authorization.get("source_contract") or {}
_continuation = _authorization.get("continuation") or {}
_automatic = _authorization.get("automatic_actions") or {}
if (
    _authorization.get("decision")
    != "HUMAN_AUTHORIZED_B3_EXACT_CONTINUATION_IT100_TO_IT600"
    or (REPO / str(_parent.get("checkpoint"))).resolve()
    != PARENT_CHECKPOINT.resolve()
    or _parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or _parent.get("optimizer_resume") is not True
    or _parent.get("rl_optimizer_state_entries") != 38
    or _identity.get("source_config_sha256") != SOURCE_CONFIG_SHA256
    or set(_identity.get("allowed_behavioral_differences") or ())
    != EXPECTED_DIFFS
    or _source.get("trainer_sha256") != TRAINER_SHA256
    or _source.get("corridor_family_metrics_sha256")
    != CORRIDOR_FAMILY_METRICS_SHA256
    or _source.get("corridor_profile_family_metrics_sha256")
    != CORRIDOR_PROFILE_FAMILY_METRICS_SHA256
    or _source.get("long_corridor_replay_sha256")
    != LONG_CORRIDOR_REPLAY_SHA256
    or _source.get("long_corridor_geometry_sha256")
    != LONG_CORRIDOR_GEOMETRY_SHA256
    or _continuation.get("additional_training_iterations")
    != ADDITIONAL_TRAINING_ITERATIONS
    or _continuation.get("conceptual_end_iteration")
    != CONCEPTUAL_END_ITERATION
    or _continuation.get("conceptual_checkpoints")
    != [list(row) for row in CHECKPOINTS]
    or _automatic.get("sa6_auto_start") is not False
):
    raise RuntimeError("authorization does not permit the B3 it100 continuation")

CONFIG = replace(
    _IT100,
    name="e2e_sa5_v3_c50_stage3_b3_cont500_from_it100",
    description=(
        "User-authorized exact B3 optimizer continuation from conceptual "
        "it100 to it600."
    ),
    checkpoint=str(PARENT_CHECKPOINT),
    timesteps=ADDITIONAL_TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_IT100.tags
    + (
        "exact_optimizer_continuation_it100_to_it600",
        "user_authorized_500_iterations",
        "exploratory_not_graduated",
        "sa6_hold",
    ),
    notes=(
        f"{_IT100.notes} Authorization {AUTHORIZATION_RECORD_SHA256} resumes "
        f"conceptual it100 ({PARENT_CHECKPOINT_SHA256}) with its PPO optimizer. "
        "This long continuation produces checkpoint candidates only; fixed "
        "validation is still required for graduation."
    ),
)

_actual_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_IT100, field.name)
)
if _actual_diffs != EXPECTED_DIFFS:
    raise RuntimeError(
        "B3 it100 continuation drifted: expected "
        f"{sorted(EXPECTED_DIFFS)}, got {sorted(_actual_diffs)}"
    )

assert CONFIG.checkpoint == str(PARENT_CHECKPOINT)
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 5 and CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024 and CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 64000
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.speed_rate == 0.7 and CONFIG.speed_rate_obs == "ego"
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
