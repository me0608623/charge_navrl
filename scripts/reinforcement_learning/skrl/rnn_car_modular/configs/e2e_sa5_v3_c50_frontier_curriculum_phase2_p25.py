"""Phase 2 of the bounded SA5-v3 c50 difficulty-frontier curriculum.

This phase resumes the exact phase-1 it25 optimizer state. It retains
low-density and single-pedestrian replay, then introduces the measured
1S2D -> 2S2D -> 4S2D ladder. No non-curriculum behavioral field may change.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c50_frontier_curriculum_phase1_p25 import (
    CONFIG as _PHASE_1,
)
from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (
    P035,
    P060,
    P080,
)


REPO = Path(__file__).resolve().parents[5]

RUN_NAME = "sa5_v3_c50_frontier_curriculum_phase2_ne1024_s42_p25_r1"
PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_v3_c50_frontier_curriculum_phase1_ne1024_s42_p25_r1/"
    "checkpoint_3200.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "c120ef3c892030019af377f2932b1ddf6c8253071b7c98513c3af7f28cd7cb9f"
)

TRAINING_ITERATIONS = 25
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
CHECKPOINTS = ((50, "checkpoint_3200.pt"),)

PHASE_2_SPEED_DENSITY_MIX = (
    ((0, 1), P080, 0.05),
    ((1, 1), P080, 0.05),
    ((2, 1), P060, 0.10),
    ((1, 1), P035, 0.05),
    ((2, 1), P035, 0.05),
    ((4, 1), P035, 0.10),
    ((1, 2), P035, 0.15),
    ((2, 2), P035, 0.20),
    ((4, 2), P035, 0.25),
)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_frontier_curriculum_p50_authorization_20260823.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "9ddfbf16a0492e4ec1dc528b75b69607cb9b98c53c9b0ab4caa903de6af3ac54"
)
PHASE_1_RESULT = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_frontier_curriculum_phase1_complete_v1.json"
)
PHASE_1_RESULT_SHA256 = (
    "13c74e682a5958181d63e23e2261f6747aaf13c29c4f351a22ef554d3e7af651"
)
PHASE_1_SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_frontier_curriculum_phase1_p25.py"
)
PHASE_1_SOURCE_CONFIG_SHA256 = (
    "181bcb01354f352df28cf49a5c146a69d6e0ebd7c704d509eeddaf47e7427bcc"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_BEHAVIORAL_DIFFS = frozenset(
    {"checkpoint", "long_corridor_speed_density_mix"}
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
    (_parent_path, PARENT_CHECKPOINT_SHA256, "phase-1 it25 checkpoint"),
    (
        AUTHORIZATION_RECORD,
        AUTHORIZATION_RECORD_SHA256,
        "difficulty-frontier pilot authorization",
    ),
    (PHASE_1_RESULT, PHASE_1_RESULT_SHA256, "phase-1 completion evidence"),
    (
        PHASE_1_SOURCE_CONFIG,
        PHASE_1_SOURCE_CONFIG_SHA256,
        "phase-1 source config",
    ),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_lineage = _authorization.get("lineage_status") or {}
_pilot = _authorization.get("pilot") or {}
_phase_2 = _pilot.get("phase_2") or {}
if (
    _authorization.get("decision")
    != "HUMAN_AUTHORIZED_SA5_V3_C50_JOINT_FRONTIER_CURRICULUM_P50"
    or _lineage.get("anchor")
    != "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT"
    or _lineage.get("sa6") != "HOLD_NOT_AUTHORIZED"
    or _pilot.get("total_iterations") != 50
    or _phase_2.get("conceptual_iterations") != [26, 50]
):
    raise RuntimeError("authorization does not permit frontier curriculum phase 2")

_result = json.loads(PHASE_1_RESULT.read_text(encoding="utf-8"))
_result_checkpoint = _result.get("checkpoint") or {}
if (
    _result.get("status") != "COMPLETE_VALID_PHASE1_DIAGNOSTIC_CURRICULUM"
    or (_result.get("lineage") or {}).get("phase_2")
    != "READY_FOR_LOCK_AND_SMOKE_NOT_AUTO_STARTED"
    or (_result.get("lineage") or {}).get("sa6") != "HOLD_NOT_AUTHORIZED"
    or _result_checkpoint.get("sha256") != PARENT_CHECKPOINT_SHA256
    or (REPO / str(_result_checkpoint.get("path"))).resolve()
    != _parent_path.resolve()
    or _result_checkpoint.get("rl_optimizer_state_entries") != 38
    or not bool((_result.get("integrity") or {}).get("training_complete_marker"))
    or (_result.get("integrity") or {}).get("strict_errors") != 0
):
    raise RuntimeError("phase-1 result does not authorize phase 2")

CONFIG = replace(
    _PHASE_1,
    name="e2e_sa5_v3_c50_frontier_curriculum_phase2_p25",
    description=(
        "Final 25 iterations of the bounded joint 1D-to-2D and 1S-to-2S-to-4S "
        "SA5-v3 difficulty-frontier curriculum, resumed from phase-1 it25."
    ),
    checkpoint=PARENT_CHECKPOINT,
    long_corridor_speed_density_mix=PHASE_2_SPEED_DENSITY_MIX,
    tags=tuple(
        tag
        for tag in _PHASE_1.tags
        if tag != "phase1_1d_ladder_then_1s2d"
    )
    + (
        "phase2_1s2d_2s2d_4s2d_ladder",
        "exact_phase1_optimizer_resume",
    ),
    notes=(
        f"{_PHASE_1.notes} Phase-1 completion evidence "
        f"{PHASE_1_RESULT_SHA256} freezes it25 as {PARENT_CHECKPOINT_SHA256}. "
        "Phase 2 changes only the corridor profile proportions: 20% P080/P060 "
        "low-density retention, 20% P035 single-pedestrian replay, and a "
        "15%/20%/25% P035 1S2D/2S2D/4S2D ladder. Resume the embedded "
        "optimizer and stop at conceptual pilot it50. Do not auto-extend or "
        "launch SA6."
    ),
)

_actual_behavioral_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_PHASE_1, field.name)
)
if _actual_behavioral_diffs != EXPECTED_BEHAVIORAL_DIFFS:
    raise RuntimeError(
        "frontier curriculum phase 2 drifted from phase 1: expected "
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
assert CONFIG.long_corridor_speed_density_mix == PHASE_2_SPEED_DENSITY_MIX
assert abs(sum(row[2] for row in PHASE_2_SPEED_DENSITY_MIX) - 1.0) < 1e-12
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (1, 2)
assert 1.0 - CONFIG.narrow_passage_fraction - CONFIG.long_corridor_fraction == 0.78
assert CHECKPOINTS == ((50, "checkpoint_3200.pt"),)
