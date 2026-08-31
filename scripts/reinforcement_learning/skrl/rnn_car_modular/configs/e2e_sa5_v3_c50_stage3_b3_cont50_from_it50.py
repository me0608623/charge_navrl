"""Exact B3 continuation from conceptual it50 to it100."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_cont25_from_it25 import (
    CONFIG as _IT50,
)


REPO = Path(__file__).resolve().parents[5]
RUN_NAME = "sa5_v3_c50_stage3_b3_cont50_from_it50_ne1024_s42_p50_r1"
PARENT_CHECKPOINT = (
    REPO
    / "logs/rnn_car/"
    "sa5_v3_c50_stage3_b3_cont25_from_it25_ne1024_s42_p25_r1/"
    "checkpoint_3200.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "2cf4b6beddb74cb13ad474b684e4d5ce1b3887ff0052aba343809b2b64fffff6"
)
PARENT_CONCEPTUAL_ITERATION = 50
ADDITIONAL_TRAINING_ITERATIONS = 50
CONCEPTUAL_END_ITERATION = 100
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 10
CHECKPOINTS = (
    (60, "checkpoint_1280.pt"),
    (70, "checkpoint_2560.pt"),
    (80, "checkpoint_3840.pt"),
    (90, "checkpoint_5120.pt"),
    (100, "checkpoint_6400.pt"),
)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage3_b3_cont50_from_it50_authorization_20260825.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "3a0cdc42ea36ccc07674101ab1c0323ece9033651e17d3af079b9a21221c7140"
)
SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_stage3_b3_cont25_from_it25.py"
)
SOURCE_CONFIG_SHA256 = (
    "3af0b9e6bb9f3683d4b3edbf30365fe2e5a9750e4942ef28500baa1e0c43b952"
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
    (PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, "B3 conceptual it50 checkpoint"),
    (AUTHORIZATION_RECORD, AUTHORIZATION_RECORD_SHA256, "authorization"),
    (SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "it50 source config"),
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
    != "HUMAN_AUTHORIZED_B3_EXACT_CONTINUATION_IT50_TO_IT100"
    or (REPO / str(_parent.get("checkpoint"))).resolve()
    != PARENT_CHECKPOINT.resolve()
    or _parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or _parent.get("optimizer_resume") is not True
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
    or _automatic.get("phase_b_auto_start") is not False
    or _automatic.get("sa6_auto_start") is not False
):
    raise RuntimeError("authorization does not permit the B3 it50 continuation")

CONFIG = replace(
    _IT50,
    name="e2e_sa5_v3_c50_stage3_b3_cont50_from_it50",
    description=(
        "User-authorized exact B3 optimizer continuation from conceptual "
        "it50 to it100."
    ),
    checkpoint=str(PARENT_CHECKPOINT),
    timesteps=ADDITIONAL_TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_IT50.tags
    + (
        "exact_optimizer_continuation_it50_to_it100",
        "exploratory_not_graduated",
        "teacher_diagnostics_hold",
        "sa6_hold",
    ),
    notes=(
        f"{_IT50.notes} Authorization {AUTHORIZATION_RECORD_SHA256} resumes "
        f"conceptual it50 ({PARENT_CHECKPOINT_SHA256}) with its PPO optimizer. "
        "The prior 24-cell screen was incomplete and supplies no formal parent "
        "ranking. This run may produce checkpoint candidates only."
    ),
)

_actual_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_IT50, field.name)
)
if _actual_diffs != EXPECTED_DIFFS:
    raise RuntimeError(
        "B3 it50 continuation drifted: expected "
        f"{sorted(EXPECTED_DIFFS)}, got {sorted(_actual_diffs)}"
    )

assert CONFIG.checkpoint == str(PARENT_CHECKPOINT)
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 5 and CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024 and CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 6400
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.speed_rate == 0.7 and CONFIG.speed_rate_obs == "ego"
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"

