"""Exact SA6 continuation from conceptual it50 to it350."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa6_v3_from_sa5_c600_p50 import CONFIG as _IT50


REPO = Path(__file__).resolve().parents[5]
RUN_NAME = "sa6_v3_cont300_from_it50_ne1024_s42_p300_r1"
PARENT_CHECKPOINT = (
    REPO
    / "logs/rnn_car/sa6_v3_from_sa5_c600_ne1024_s42_p50_r1/"
    "checkpoint_6400.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "096a953132a4ec9a4208aa331448e83189ede6b35f40d77d5552d4e94909d65b"
)
PARENT_CONCEPTUAL_ITERATION = 50
ADDITIONAL_TRAINING_ITERATIONS = 300
CONCEPTUAL_END_ITERATION = 350
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 50
CHECKPOINTS = (
    (100, "checkpoint_6400.pt"),
    (150, "checkpoint_12800.pt"),
    (200, "checkpoint_19200.pt"),
    (250, "checkpoint_25600.pt"),
    (300, "checkpoint_32000.pt"),
    (350, "checkpoint_38400.pt"),
)
TRAINING_STARTED = False

AUTHORIZATION_RECORD = (
    REPO / "docs/freeze/sa6_v3_cont300_from_it50_authorization_20260827.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "a800ff50a298f06fef3bcfa10fe58c9e8d8dabd3de59d1ed7b77071cc9e043c8"
)
SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa6_v3_from_sa5_c600_p50.py"
)
SOURCE_CONFIG_SHA256 = (
    "c0130ec73d1465dcbb2505eee25de1d85007b958acba4c863e23c703669ab09c"
)
COMPLETION_RECORD = (
    REPO / "docs/freeze/sa6_v3_from_sa5_c600_p50_launch_20260826.json"
)
COMPLETION_RECORD_SHA256 = (
    "ede96653824189443a96384ca969cd8d18270a1600dfd3ddab10ff6e469ac5b9"
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
    (PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, "SA6 conceptual it50 checkpoint"),
    (AUTHORIZATION_RECORD, AUTHORIZATION_RECORD_SHA256, "continuation authorization"),
    (SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "completed SA6 P50 source config"),
    (COMPLETION_RECORD, COMPLETION_RECORD_SHA256, "SA6 P50 completion record"),
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

_completion = json.loads(COMPLETION_RECORD.read_text(encoding="utf-8"))
_completion_verification = _completion.get("verification") or {}
_completion_result = _completion_verification.get("completion") or {}
if (
    _completion.get("status") != "COMPLETE_HEALTHY_AWAITING_FIXED_CHECKPOINT_SCREEN"
    or _completion_result.get("iterations") != 50
    or _completion_result.get("iterations_target") != 50
    or _completion_result.get("trainer_steps") != 6_400
    or _completion_result.get("checkpoint_it50_sha256")
    != PARENT_CHECKPOINT_SHA256
    or _completion_result.get("strict_error_scan_count") != 0
    or _completion_result.get("family_reconciliation_failures") != 0
    or _completion_result.get("profile_family_reconciliation_failures") != 0
):
    raise RuntimeError("completion record does not identify a healthy SA6 it50 parent")

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_parent = _authorization.get("parent") or {}
_identity = _authorization.get("identity_contract") or {}
_source = _authorization.get("source_contract") or {}
_continuation = _authorization.get("continuation") or {}
_automatic = _authorization.get("automatic_actions") or {}
if (
    _authorization.get("schema") != "sa6_v3_cont300_from_it50_authorization/v1"
    or _authorization.get("decision")
    != "HUMAN_AUTHORIZED_EXACT_CONTINUATION_IT50_PLUS_300"
    or (REPO / str(_parent.get("checkpoint"))).resolve()
    != PARENT_CHECKPOINT.resolve()
    or _parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or _parent.get("conceptual_iteration") != PARENT_CONCEPTUAL_ITERATION
    or _parent.get("checkpoint_total_steps") != 6_400
    or _parent.get("checkpoint_iteration_zero_based") != 49
    or _parent.get("rl_optimizer_state_entries") != 38
    or _parent.get("resume_ppo_optimizer") is not True
    or _identity.get("source_config_sha256") != SOURCE_CONFIG_SHA256
    or _identity.get("completion_record_sha256") != COMPLETION_RECORD_SHA256
    or set(_identity.get("allowed_behavioral_differences") or ()) != EXPECTED_DIFFS
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
    or _continuation.get("additional_timesteps")
    != ADDITIONAL_TRAINING_ITERATIONS * ROLLOUT_LENGTH
    or _continuation.get("conceptual_end_iteration") != CONCEPTUAL_END_ITERATION
    or _continuation.get("conceptual_checkpoints")
    != [list(row) for row in CHECKPOINTS]
    or _automatic.get("config_build_authorized") is not True
    or _automatic.get("gpu_smoke_authorized") is not True
    or _automatic.get("training_launch_authorized") is not True
    or _automatic.get("extend_beyond_conceptual_it350") is not False
    or _automatic.get("start_sa7") is not False
    or _automatic.get("enable_teacher_distillation_or_override") is not False
):
    raise RuntimeError("authorization does not permit this exact SA6 continuation")

CONFIG = replace(
    _IT50,
    name="e2e_sa6_v3_cont300_from_it50",
    description=(
        "User-authorized exact SA6 PPO continuation from conceptual it50 "
        "through it350."
    ),
    checkpoint=str(PARENT_CHECKPOINT),
    timesteps=ADDITIONAL_TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=tuple(tag for tag in _IT50.tags if tag != "pilot50_ready_not_run")
    + (
        "exact_optimizer_continuation_it50_to_it350",
        "user_authorized_300_additional_iterations",
        "fixed_comparison_required",
    ),
    notes=(
        f"{_IT50.notes} Authorization {AUTHORIZATION_RECORD_SHA256} resumes "
        f"conceptual it50 checkpoint {PARENT_CHECKPOINT_SHA256} and its PPO "
        "optimizer for exactly 300 additional iterations. Checkpoints are "
        "candidates only until compared under one fixed evaluator protocol."
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
        "SA6 it50 continuation drifted: expected "
        f"{sorted(EXPECTED_DIFFS)}, got {sorted(_actual_diffs)}"
    )

assert CONFIG.checkpoint == str(PARENT_CHECKPOINT)
assert CONFIG.no_resume_optimizer is False
assert CONFIG.initial_stage == 6 and CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024 and CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 38_400 and CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.lidar_frame_stack == 8 and CONFIG.end_to_end_frame_stack is True
assert CONFIG.lidar_no_noise is False and CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.enable_actuator_dr is True
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.speed_rate == 0.7 and CONFIG.speed_rate_obs == "ego"
assert CONFIG.teacher_retention_checkpoint is None
assert CONFIG.teacher_retention_rollout_override is False
assert CONFIG.corridor_teacher_distill_epochs == 0

