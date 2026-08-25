"""Bounded SA5-v3 c50 random-2D exposure pilot.

The formal D5 audit found short-horizon jointly feasible action candidates in
every recorded 4S2D random-2D collision window. This pilot therefore changes
only the corridor motion-family proportions. It resumes the exact c50 PPO
optimizer for 25 iterations and does not authorize SA5 graduation or SA6.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c500_parent_control_from_sa4v3_p50 import (
    CONFIG as _C50_SOURCE,
)


REPO = Path(__file__).resolve().parents[5]

RUN_NAME = "sa5_v3_c50_random2d_weighted_ne1024_s42_p25_r1"
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
MOTION_WEIGHTS = (0.15, 0.15, 0.70)

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_random2d_weighted_p25_authorization_20260824.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "1d4aa5e87cf0290ee622239b87496963a443364e2ae8d90a9f4d9dcb0d84c75f"
)
SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c500_parent_control_from_sa4v3_p50.py"
)
SOURCE_CONFIG_SHA256 = (
    "ee7990c162da118571910d47d3d73c530b63595c0ef14a404703ab57bc2005f1"
)
FEASIBILITY_ANALYSIS = (
    REPO
    / "logs/gates/sa5_v3_c50_random2d_feasibility_shadow/"
    "audit_20260824_r1/ANALYSIS.json"
)
FEASIBILITY_ANALYSIS_SHA256 = (
    "016ae0731b2bdb6fe3c0cbdf60c7b383a2dd18ee14d85a3131f2948a341b647f"
)
FEASIBILITY_MANIFEST = (
    REPO
    / "logs/gates/sa5_v3_c50_random2d_feasibility_shadow/"
    "audit_20260824_r1/MANIFEST.json"
)
FEASIBILITY_MANIFEST_SHA256 = (
    "dff1fff697faa88f5cdb3f2a0196ed237ae2903783c7fc3d9de46f94d88a992c"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_BEHAVIORAL_DIFFS = frozenset(
    {"checkpoint", "long_corridor_dynamic_motion_weights", "timesteps"}
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


for _path, _expected, _label in (
    (
        Path(PARENT_CHECKPOINT),
        PARENT_CHECKPOINT_SHA256,
        "SA5-v3 c50 diagnostic anchor",
    ),
    (
        AUTHORIZATION_RECORD,
        AUTHORIZATION_RECORD_SHA256,
        "random-2D pilot authorization",
    ),
    (SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "c50 source recipe"),
    (
        FEASIBILITY_ANALYSIS,
        FEASIBILITY_ANALYSIS_SHA256,
        "random-2D feasibility analysis",
    ),
    (
        FEASIBILITY_MANIFEST,
        FEASIBILITY_MANIFEST_SHA256,
        "random-2D feasibility manifest",
    ),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_analysis = json.loads(FEASIBILITY_ANALYSIS.read_text(encoding="utf-8"))
_manifest = json.loads(FEASIBILITY_MANIFEST.read_text(encoding="utf-8"))
if (
    _authorization.get("decision")
    != "ACTIVATE_SHORT_HORIZON_ACTION_OPPORTUNITY_BRANCH"
    or (_authorization.get("lineage") or {}).get("anchor")
    != "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT"
    or (_authorization.get("lineage") or {}).get("sa6")
    != "HOLD_NOT_AUTHORIZED"
    or tuple((_authorization.get("pilot") or {}).get(
        "intervention_motion_weights", ()
    ))
    != MOTION_WEIGHTS
    or set(
        (_authorization.get("identity_contract") or {}).get(
            "allowed_behavioral_differences", ()
        )
    )
    != EXPECTED_BEHAVIORAL_DIFFS
):
    raise RuntimeError("authorization does not permit this bounded pilot")

if (
    _analysis.get("status")
    != "COMPLETE_VALID_MODEL_BOUNDED_DIAGNOSTIC_EVIDENCE"
    or _analysis.get("decision")
    != "SHORT_HORIZON_ACTION_OPPORTUNITY_OBSERVED"
    or _analysis.get("training_started") is not False
    or _analysis.get("sa6_started") is not False
    or _manifest.get("source_fingerprint_stable") is not True
    or _manifest.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
    or _manifest.get("training_started") is not False
    or _manifest.get("sa6_started") is not False
):
    raise RuntimeError("feasibility evidence does not authorize training")

CONFIG = replace(
    _C50_SOURCE,
    name="e2e_sa5_v3_c50_random2d_weighted_p25",
    description=(
        "Bounded 25-iteration SA5-v3 pilot that increases only random-2D "
        "corridor exposure from the ungraduated c50 anchor."
    ),
    checkpoint=PARENT_CHECKPOINT,
    long_corridor_dynamic_motion_weights=MOTION_WEIGHTS,
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
        "random2d_weighted_70pct_within_corridor",
        "bounded_p25",
        "sa6_hold",
    ),
    notes=(
        f"{_C50_SOURCE.notes} Conditional authorization "
        f"{AUTHORIZATION_RECORD_SHA256} activates the opportunity-observed "
        "branch from c50. Resume its exact PPO optimizer and change only the "
        "corridor family weights from implicit equal allocation to lateral "
        "0.15, longitudinal 0.15, random_2d 0.70. The outer scene shares and "
        "the complete speed-density mix remain unchanged. Stop at it25 for "
        "the frozen 8-cell direction screen; do not launch SA6."
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
        "random-2D pilot drifted from c50 source: expected "
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
assert CONFIG.long_corridor_dynamic_motion_mode == "env_stratified"
assert CONFIG.long_corridor_dynamic_motion_weights == MOTION_WEIGHTS
assert CONFIG.long_corridor_speed_density_mix == _C50_SOURCE.long_corridor_speed_density_mix
assert CONFIG.long_corridor_fraction == 0.10
assert CONFIG.narrow_passage_fraction == 0.12
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (1, 2)
assert abs(sum(MOTION_WEIGHTS) - 1.0) < 1e-12
assert CHECKPOINTS == ((25, "checkpoint_3200.pt"),)
