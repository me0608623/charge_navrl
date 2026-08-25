"""Bounded provisional SA5-v3 pilot from the non-graduated SA4-v3 c550.

The human authorization permits training progression without rewriting the
failed SA4 verdict. Only canonical Stage-5 scene geometry, a preregistered
speed-density mix, run budget, lineage, and metadata may differ from SA4-v3.
The embedded Adam state is resumed because action, observation, reward, model,
and PPO contracts remain unchanged.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa4_k8_obb_speed_density_v3_from_sa3 import (
    CONFIG as _SA4_V3,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (
    P035,
    P060,
    P080,
)
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    make_sim2real_curriculum_config,
)


REPO = Path(__file__).resolve().parents[5]

RUN_NAME = "sa5_v3_provisional_from_sa4v3_c550_ne1024_s42_p50_r1"
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/checkpoint_32000.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "235a79a41b1d83510bbbb314c965a7ef5ba73e898aec121310a92965092c4ba8"
)
PARENT_CONCEPTUAL_ITERATION = 550

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_provisional_c550_parent_authorization_20260822.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "867517dbf279f7d1be7105be8e4496d4cb879c406e796a24d7e18b867dd7fa27"
)
SA4_SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa4_k8_obb_speed_density_v3_from_sa3.py"
)
SA4_SOURCE_CONFIG_SHA256 = (
    "30dee032487d786393030d7e4c436a10226338fa765ba4d72bef396b8c1764d3"
)
STAGE_CURRICULUM_SOURCE = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_stage_curriculum_v2.py"
)
STAGE_CURRICULUM_SOURCE_SHA256 = (
    "f594920f61361db99b943cb82d51d2763bc4670c244d26a4cc82041f03a3f7da"
)
SA4_GATE_SUMMARY = (
    REPO
    / "logs/gates/sa4_v3_c300_c600_gate/screen_20260822_r1/SUMMARY.json"
)
SA4_GATE_SUMMARY_SHA256 = (
    "ab0e41e148fd82582b15e9e7c473297d5ec5b8a3a050411801b639e1214e3df5"
)
RETENTION_SUMMARY = (
    REPO
    / "logs/gates/sa4_v3_c550_c600_retention/screen_20260822_r1/SUMMARY.json"
)
RETENTION_SUMMARY_SHA256 = (
    "620b777dc1fa611b3e24cef8cc0a6f88f14783cd2a9235418615b58a6d15f25a"
)

TRAINING_ITERATIONS = 50
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINTS = (
    (25, "checkpoint_3200.pt"),
    (50, "checkpoint_6400.pt"),
)

# Retain the successful SA5-R2 density ladder while assigning speed bands from
# the rebuilt v3 contract. P100 remains excluded because SA4 P080 did not pass.
SA5_SPEED_DENSITY_MIX = (
    ((0, 1), P080, 0.10),
    ((1, 1), P080, 0.10),
    ((2, 1), P060, 0.15),
    ((3, 2), P035, 0.30),
    ((4, 2), P035, 0.35),
)
SPEED_ENVELOPE = (P035[0], P080[1])

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
STAGE_SCENE_FIELDS = frozenset(
    {
        "initial_stage",
        "room_size",
        "narrow_passage_fraction",
        "narrow_passage_segment_length",
        "narrow_passage_fixed_width_range",
        "narrow_passage_fixed_yaw_limit_deg",
        "narrow_passage_exact_width",
        "narrow_passage_exact_width_ratio",
        "long_corridor_fraction",
        "long_corridor_free_width",
        "long_corridor_length",
        "long_corridor_static_obstacles",
        "long_corridor_dynamic_obstacles",
        "long_corridor_obstacle_count_mix",
        "long_corridor_speed_density_mix",
        "long_corridor_dynamic_speed_range",
        "long_corridor_dynamic_motion_mode",
        "long_corridor_dynamic_motion_weights",
        "long_corridor_random_2d_kinematics",
    }
)
RUN_FIELDS = frozenset(
    {"checkpoint", "no_resume_optimizer", "timesteps", "save_interval"}
)
ALLOWED_DIFFS = METADATA_FIELDS | STAGE_SCENE_FIELDS | RUN_FIELDS
EXPECTED_CHANGED_FIELDS = frozenset(
    {
        "checkpoint",
        "initial_stage",
        "long_corridor_dynamic_motion_weights",
        "long_corridor_fraction",
        "long_corridor_free_width",
        "long_corridor_speed_density_mix",
        "long_corridor_static_obstacles",
        "narrow_passage_fixed_width_range",
        "narrow_passage_fraction",
        "narrow_passage_segment_length",
        "no_resume_optimizer",
        "room_size",
        "save_interval",
        "timesteps",
    }
)


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
    (_parent_path, PARENT_CHECKPOINT_SHA256, "provisional c550 parent"),
    (
        AUTHORIZATION_RECORD,
        AUTHORIZATION_RECORD_SHA256,
        "provisional parent authorization",
    ),
    (SA4_SOURCE_CONFIG, SA4_SOURCE_CONFIG_SHA256, "SA4-v3 source config"),
    (
        STAGE_CURRICULUM_SOURCE,
        STAGE_CURRICULUM_SOURCE_SHA256,
        "canonical stage curriculum",
    ),
    (SA4_GATE_SUMMARY, SA4_GATE_SUMMARY_SHA256, "SA4-v3 Gate summary"),
    (RETENTION_SUMMARY, RETENTION_SUMMARY_SHA256, "c550 retention summary"),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_authorized_parent = _authorization.get("parent") or {}
_authorized_pilot = _authorization.get("pilot") or {}
if (
    _authorization.get("decision")
    != "HUMAN_AUTHORIZED_PROVISIONAL_SA5_PARENT_C550"
    or (_authorization.get("lineage_status") or {}).get("sa4")
    != "SA4_NOT_GRADUATED"
    or (_authorization.get("lineage_status") or {}).get("sa5_parent")
    != "PROVISIONAL_SA5_PARENT_C550"
    or _authorized_parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or (REPO / str(_authorized_parent.get("checkpoint"))).resolve()
    != _parent_path.resolve()
    or _authorized_parent.get("optimizer_resume") is not True
    or _authorized_pilot.get("training_iterations") != TRAINING_ITERATIONS
    or _authorized_pilot.get("save_iterations") != [25, 50]
):
    raise RuntimeError("authorization does not permit the c550 SA5-v3 pilot")

_gate = json.loads(SA4_GATE_SUMMARY.read_text(encoding="utf-8"))
_gate_c550 = next(
    (
        row
        for row in _gate.get("ranked_candidates", [])
        if row.get("checkpoint_name") == "c550"
    ),
    None,
)
if (
    not _gate_c550
    or _gate.get("status")
    != "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_GATE_COMPARISON"
    or not bool(_gate.get("source_fingerprint_stable"))
    or bool(_gate_c550.get("hard_gate_pass"))
    or (_gate.get("ranked_candidates") or [{}])[0].get("checkpoint_name")
    != "c550"
):
    raise RuntimeError("source Gate must rank c550 first while retaining FAIL")

_retention = json.loads(RETENTION_SUMMARY.read_text(encoding="utf-8"))
_retention_c550 = next(
    (
        row
        for row in _retention.get("candidate_results", [])
        if row.get("checkpoint_name") == "c550"
    ),
    None,
)
if (
    not _retention_c550
    or _retention.get("status")
    != "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_RETENTION_SCREEN"
    or not bool(_retention.get("source_fingerprint_stable"))
    or bool(_retention_c550.get("retention_pass"))
    or bool((_retention_c550.get("cells") or {}).get("nav_native", {}).get("gate_pass"))
):
    raise RuntimeError("source retention must preserve c550 native FAIL")

_SA5_CANONICAL = make_sim2real_curriculum_config(
    5, checkpoint=PARENT_CHECKPOINT
)

CONFIG = replace(
    _SA4_V3,
    name="e2e_sa5_v3_provisional_from_sa4v3_c550_p50",
    description=(
        "Bounded 50-iteration provisional SA5-v3 pilot from the non-graduated "
        "SA4-v3 c550 with retained low-density replay."
    ),
    initial_stage=_SA5_CANONICAL.initial_stage,
    room_size=_SA5_CANONICAL.room_size,
    narrow_passage_fraction=_SA5_CANONICAL.narrow_passage_fraction,
    narrow_passage_segment_length=(
        _SA5_CANONICAL.narrow_passage_segment_length
    ),
    narrow_passage_fixed_width_range=(
        _SA5_CANONICAL.narrow_passage_fixed_width_range
    ),
    narrow_passage_fixed_yaw_limit_deg=(
        _SA5_CANONICAL.narrow_passage_fixed_yaw_limit_deg
    ),
    narrow_passage_exact_width=_SA5_CANONICAL.narrow_passage_exact_width,
    narrow_passage_exact_width_ratio=(
        _SA5_CANONICAL.narrow_passage_exact_width_ratio
    ),
    long_corridor_fraction=_SA5_CANONICAL.long_corridor_fraction,
    long_corridor_free_width=_SA5_CANONICAL.long_corridor_free_width,
    long_corridor_length=_SA5_CANONICAL.long_corridor_length,
    long_corridor_static_obstacles=(
        _SA5_CANONICAL.long_corridor_static_obstacles
    ),
    long_corridor_dynamic_obstacles=(
        _SA5_CANONICAL.long_corridor_dynamic_obstacles
    ),
    long_corridor_obstacle_count_mix=None,
    long_corridor_speed_density_mix=SA5_SPEED_DENSITY_MIX,
    long_corridor_dynamic_speed_range=SPEED_ENVELOPE,
    long_corridor_dynamic_motion_mode=(
        _SA5_CANONICAL.long_corridor_dynamic_motion_mode
    ),
    long_corridor_dynamic_motion_weights=(
        _SA5_CANONICAL.long_corridor_dynamic_motion_weights
    ),
    long_corridor_random_2d_kinematics=(
        _SA5_CANONICAL.long_corridor_random_2d_kinematics
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=False,
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_SA4_V3.tags
    + (
        "sa4_not_graduated",
        "provisional_sa5_parent_c550",
        "human_authorized_bounded_pilot",
        "resume_embedded_optimizer",
        "stage5_native78_narrow12_corridor10",
        "stage5_low_density_retained",
        "p100_excluded",
        "p50",
    ),
    notes=(
        f"{_SA4_V3.notes} Human-authorized provisional progression from SA4-v3 "
        "c550 without rewriting its failed SA4 Gate. Parent SHA-256 is "
        f"{PARENT_CHECKPOINT_SHA256}. Only Stage-5 scene geometry and reset "
        "distribution change. Native/narrow/corridor reset shares are "
        "0.78/0.12/0.10. Corridor profiles are 0S1D-P080 10%, 1S1D-P080 "
        "10%, 2S1D-P060 15%, 3S2D-P035 30%, and 4S2D-P035 35%. Resume the "
        "embedded optimizer because reward, action, observation, model, PPO, "
        "sensor, and actuator contracts are unchanged. Stop after 50 "
        "iterations for the frozen parent/it25/it50 screen. Do not auto-launch "
        "SA6."
    ),
)

for _field_name in STAGE_SCENE_FIELDS - {"long_corridor_speed_density_mix"}:
    _expected = getattr(_SA5_CANONICAL, _field_name)
    if _field_name == "long_corridor_obstacle_count_mix":
        _expected = None
    elif _field_name == "long_corridor_dynamic_speed_range":
        _expected = SPEED_ENVELOPE
    if getattr(CONFIG, _field_name) != _expected:
        raise RuntimeError(f"SA5-v3 scene field {_field_name} drifted")

_unexpected = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in ALLOWED_DIFFS
    and getattr(CONFIG, field.name) != getattr(_SA4_V3, field.name)
)
if _unexpected:
    raise RuntimeError(
        "SA5-v3 pilot changed reward/action/observation/model/PPO contract: "
        f"{_unexpected}"
    )

_actual_changed = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_SA4_V3, field.name)
)
if _actual_changed != EXPECTED_CHANGED_FIELDS:
    raise RuntimeError(
        "SA5-v3 pilot changed-field set mismatch: expected "
        f"{sorted(EXPECTED_CHANGED_FIELDS)}, got {sorted(_actual_changed)}"
    )

assert CONFIG.initial_stage == 5
assert CONFIG.fixed_stage is True
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 6400
assert CONFIG.save_interval == 25
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
assert CONFIG.actuator_motor_lag == 1.0
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.long_corridor_obstacle_count_mix is None
assert CONFIG.long_corridor_speed_density_mix == SA5_SPEED_DENSITY_MIX
assert abs(sum(row[2] for row in SA5_SPEED_DENSITY_MIX) - 1.0) < 1e-12
assert 1.0 - CONFIG.narrow_passage_fraction - CONFIG.long_corridor_fraction == 0.78
assert CHECKPOINTS == ((25, "checkpoint_3200.pt"), (50, "checkpoint_6400.pt"))
