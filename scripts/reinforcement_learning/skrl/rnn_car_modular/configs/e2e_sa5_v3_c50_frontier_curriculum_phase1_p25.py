"""Phase 1 of the bounded SA5-v3 c50 difficulty-frontier curriculum.

The c50 anchor is not graduated. This diagnostic pilot resumes its exact PPO
optimizer for 25 iterations while changing only the joint corridor profile
mix. The first phase retains the prior low-density speed replay, trains the
1S -> 2S -> 4S single-pedestrian ladder, and introduces 1S2D as the first
measured two-pedestrian boundary.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c500_parent_control_from_sa4v3_p50 import (
    CONFIG as _C50_SOURCE,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (
    P035,
    P060,
    P080,
)


REPO = Path(__file__).resolve().parents[5]

RUN_NAME = "sa5_v3_c50_frontier_curriculum_phase1_ne1024_s42_p25_r1"
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1/"
    "checkpoint_6400.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c"
)

TRAINING_ITERATIONS = 25
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINTS = ((25, "checkpoint_3200.pt"),)

PHASE_1_SPEED_DENSITY_MIX = (
    ((0, 1), P080, 0.10),
    ((1, 1), P080, 0.10),
    ((2, 1), P060, 0.15),
    ((1, 1), P035, 0.20),
    ((2, 1), P035, 0.15),
    ((4, 1), P035, 0.10),
    ((1, 2), P035, 0.20),
)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_frontier_curriculum_p50_authorization_20260823.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "9ddfbf16a0492e4ec1dc528b75b69607cb9b98c53c9b0ab4caa903de6af3ac54"
)
SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c500_parent_control_from_sa4v3_p50.py"
)
SOURCE_CONFIG_SHA256 = (
    "ee7990c162da118571910d47d3d73c530b63595c0ef14a404703ab57bc2005f1"
)
FRONTIER_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_c50_difficulty_frontier/"
    "screen_20260823_r1/SUMMARY.json"
)
FRONTIER_SUMMARY_SHA256 = (
    "2eb683551ddeb4d9e33cb56994cdd1eb6bbe3dc7196385088768cb818f859bdf"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_BEHAVIORAL_DIFFS = frozenset(
    {"checkpoint", "long_corridor_speed_density_mix", "timesteps"}
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
    (_parent_path, PARENT_CHECKPOINT_SHA256, "SA5-v3 c50 diagnostic anchor"),
    (
        AUTHORIZATION_RECORD,
        AUTHORIZATION_RECORD_SHA256,
        "difficulty-frontier pilot authorization",
    ),
    (SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "c50 source recipe"),
    (FRONTIER_SUMMARY, FRONTIER_SUMMARY_SHA256, "difficulty-frontier summary"),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_lineage = _authorization.get("lineage_status") or {}
_evidence = _authorization.get("evidence") or {}
_parent = _authorization.get("parent") or {}
_identity = _authorization.get("identity_contract") or {}
_pilot = _authorization.get("pilot") or {}
_phase_1 = _pilot.get("phase_1") or {}
if (
    _authorization.get("decision")
    != "HUMAN_AUTHORIZED_SA5_V3_C50_JOINT_FRONTIER_CURRICULUM_P50"
    or _lineage.get("anchor")
    != "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT"
    or _lineage.get("sa6") != "HOLD_NOT_AUTHORIZED"
    or _parent.get("sha256") != PARENT_CHECKPOINT_SHA256
    or (REPO / str(_parent.get("checkpoint"))).resolve()
    != _parent_path.resolve()
    or _parent.get("optimizer_resume") is not True
    or _identity.get("source_config_sha256") != SOURCE_CONFIG_SHA256
    or set(_identity.get("allowed_behavioral_differences") or ())
    != EXPECTED_BEHAVIORAL_DIFFS
    or _pilot.get("total_iterations") != 50
    or _phase_1.get("conceptual_iterations") != [1, 25]
):
    raise RuntimeError("authorization does not permit frontier curriculum phase 1")

_summary = json.loads(FRONTIER_SUMMARY.read_text(encoding="utf-8"))
if (
    _evidence.get("summary_sha256") != FRONTIER_SUMMARY_SHA256
    or _summary.get("status") != _evidence.get("required_status")
    or _summary.get("recommended_next_action")
    != _evidence.get("required_next_action")
    or not bool(_summary.get("source_fingerprint_stable"))
    or bool(_summary.get("pilot_auto_started"))
    or bool(_summary.get("sa6_started"))
):
    raise RuntimeError("difficulty-frontier evidence does not authorize phase 1")

CONFIG = replace(
    _C50_SOURCE,
    name="e2e_sa5_v3_c50_frontier_curriculum_phase1_p25",
    description=(
        "First 25 iterations of the bounded joint 1D-to-2D and 1S-to-2S-to-4S "
        "SA5-v3 difficulty-frontier curriculum from the ungraduated c50 anchor."
    ),
    checkpoint=PARENT_CHECKPOINT,
    long_corridor_speed_density_mix=PHASE_1_SPEED_DENSITY_MIX,
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    tags=tuple(
        tag
        for tag in _C50_SOURCE.tags
        if tag
        not in {
            "provisional_sa5_control_parent_c500",
            "preregistered_c550_fallback",
            "identical_recipe_control",
        }
    )
    + (
        "ungraduated_diagnostic_anchor_c50",
        "joint_density_frontier_curriculum",
        "phase1_1d_ladder_then_1s2d",
        "sa6_hold",
    ),
    notes=(
        f"{_C50_SOURCE.notes} Authorization {AUTHORIZATION_RECORD_SHA256} "
        f"resumes the c50 anchor ({PARENT_CHECKPOINT_SHA256}) and its optimizer. "
        "Phase 1 changes only the corridor speed-density profile proportions: "
        "35% low-density P080/P060 retention, 45% P035 single-pedestrian "
        "1S/2S/4S ladder, and 20% P035 1S2D boundary exposure. Reward, action, "
        "observation, network, PPO, LiDAR noise, actuator delay, speed rate, "
        "geometry, outer scene shares, env count, and seed remain unchanged. "
        "Stop at conceptual pilot it25; do not launch SA6."
    ),
)

_actual_behavioral_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_C50_SOURCE, field.name)
)
if _actual_behavioral_diffs != EXPECTED_BEHAVIORAL_DIFFS:
    raise RuntimeError(
        "frontier curriculum phase 1 drifted from c50 source: expected "
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
assert CONFIG.timesteps == 3200
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.long_corridor_speed_density_mix == PHASE_1_SPEED_DENSITY_MIX
assert abs(sum(row[2] for row in PHASE_1_SPEED_DENSITY_MIX) - 1.0) < 1e-12
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (1, 2)
assert 1.0 - CONFIG.narrow_passage_fraction - CONFIG.long_corridor_fraction == 0.78
assert CHECKPOINTS == ((25, "checkpoint_3200.pt"),)
