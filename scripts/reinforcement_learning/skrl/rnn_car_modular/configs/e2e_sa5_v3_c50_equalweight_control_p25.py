"""Matched 25-iteration continuation control for the SA5-v3 c50 anchor.

This arm resumes the exact c50 policy and PPO optimizer without changing the
original corridor motion-family allocation. It isolates generic continuation
from the random-2D weight intervention and cannot graduate SA5 or launch SA6.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c500_parent_control_from_sa4v3_p50 import (
    CONFIG as _C50_SOURCE,
)
from rnn_car_modular.configs.e2e_sa5_v3_c50_random2d_weighted_p25 import (
    CONFIG as _WEIGHTED,
)


REPO = Path(__file__).resolve().parents[5]

RUN_NAME = "sa5_v3_c50_equalweight_control_ne1024_s42_p25_r1"
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

AUTHORIZATION_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_equalweight_control_p25_authorization_20260824.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "baa8249688e919356a88589744e0e5acc3fc338556d5200338d950b445f6439f"
)
SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c500_parent_control_from_sa4v3_p50.py"
)
SOURCE_CONFIG_SHA256 = (
    "ee7990c162da118571910d47d3d73c530b63595c0ef14a404703ab57bc2005f1"
)
WEIGHTED_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_random2d_weighted_p25.py"
)
WEIGHTED_CONFIG_SHA256 = (
    "5dd4309cb93f2f584151a3f9752e91d6ba39425e2c24e02b44455b34c873758a"
)
WEIGHTED_SCREEN_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_c50_random2d_weighted_screen/"
    "screen_20260824_r1/SUMMARY.json"
)
WEIGHTED_SCREEN_SUMMARY_SHA256 = (
    "2f428c4c2ce25a9c27d0fc15b5f705be824e9789e72cabcd04496dcefd9ad3b4"
)
WEIGHTED_COMPLETION = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_random2d_weighted_p25_completion_20260824.json"
)
WEIGHTED_COMPLETION_SHA256 = (
    "b580363d653c1c560d52a3ab89a567f95ff6cade0fd777005f834426dfbd7ee7"
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_SOURCE_DIFFS = frozenset({"checkpoint", "timesteps"})
EXPECTED_WEIGHTED_DIFFS = frozenset(
    {"long_corridor_dynamic_motion_weights"}
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
    (Path(PARENT_CHECKPOINT), PARENT_CHECKPOINT_SHA256, "SA5-v3 c50 anchor"),
    (
        AUTHORIZATION_RECORD,
        AUTHORIZATION_RECORD_SHA256,
        "matched-control authorization",
    ),
    (SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "c50 source recipe"),
    (WEIGHTED_CONFIG, WEIGHTED_CONFIG_SHA256, "weighted intervention recipe"),
    (
        WEIGHTED_SCREEN_SUMMARY,
        WEIGHTED_SCREEN_SUMMARY_SHA256,
        "weighted fixed-screen result",
    ),
    (
        WEIGHTED_COMPLETION,
        WEIGHTED_COMPLETION_SHA256,
        "weighted training completion",
    ),
):
    _verify_sha256(_path, _expected, _label)

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
_summary = json.loads(WEIGHTED_SCREEN_SUMMARY.read_text(encoding="utf-8"))
_completion = json.loads(WEIGHTED_COMPLETION.read_text(encoding="utf-8"))
if (
    _authorization.get("decision") != "RUN_MATCHED_CONTINUATION_CONTROL_P25"
    or (_authorization.get("lineage") or {}).get("anchor")
    != "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT"
    or (_authorization.get("lineage") or {}).get("sa6")
    != "HOLD_NOT_AUTHORIZED"
    or (_authorization.get("control") or {}).get("motion_weights") is not None
    or set(
        (_authorization.get("identity_contract") or {}).get(
            "allowed_behavioral_differences_from_source", ()
        )
    )
    != EXPECTED_SOURCE_DIFFS
    or set(
        (_authorization.get("identity_contract") or {}).get(
            "required_behavioral_differences_from_weighted", ()
        )
    )
    != EXPECTED_WEIGHTED_DIFFS
):
    raise RuntimeError("authorization does not permit this matched control")

if (
    _summary.get("status")
    != "COMPLETE_VALID_SINGLE_SEED_EVALUATION_ONLY_SCREEN"
    or _summary.get("next_action")
    != "REJECT_WEIGHT_ONLY_HYPOTHESIS_DO_NOT_EXTEND"
    or _summary.get("target_pass") is not False
    or _summary.get("retention_pass") is not False
    or _summary.get("extension_authorized") is not False
    or _summary.get("sa6_started") is not False
    or _completion.get("status")
    != "COMPLETE_VALID_BOUNDED_PILOT_NOT_ACCEPTED_NOT_REJECTED"
    or (_completion.get("interpretation") or {}).get("sa6_authorized")
    is not False
):
    raise RuntimeError("weighted evidence does not justify the matched control")

CONFIG = replace(
    _C50_SOURCE,
    name="e2e_sa5_v3_c50_equalweight_control_p25",
    description=(
        "Matched 25-iteration continuation control from the ungraduated "
        "SA5-v3 c50 anchor with the original motion-family allocation."
    ),
    checkpoint=PARENT_CHECKPOINT,
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
        "matched_equalweight_continuation_control",
        "bounded_p25",
        "sa6_hold",
    ),
    notes=(
        f"{_C50_SOURCE.notes} Conditional authorization "
        f"{AUTHORIZATION_RECORD_SHA256} resumes the exact c50 PPO optimizer "
        "for 25 iterations without changing the original implicit motion-"
        "family allocation. Compare only against the frozen anchor and "
        "weighted-it25 arms; do not extend, graduate SA5, or launch SA6."
    ),
)

_actual_source_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_C50_SOURCE, field.name)
)
if _actual_source_diffs != EXPECTED_SOURCE_DIFFS:
    raise RuntimeError(
        "matched control drifted from c50 source: expected "
        f"{sorted(EXPECTED_SOURCE_DIFFS)}, got {sorted(_actual_source_diffs)}"
    )

_actual_weighted_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_WEIGHTED, field.name)
)
if _actual_weighted_diffs != EXPECTED_WEIGHTED_DIFFS:
    raise RuntimeError(
        "matched control is not paired with weighted arm: expected "
        f"{sorted(EXPECTED_WEIGHTED_DIFFS)}, got "
        f"{sorted(_actual_weighted_diffs)}"
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
assert CONFIG.long_corridor_dynamic_motion_weights is None
assert CONFIG.long_corridor_speed_density_mix == _C50_SOURCE.long_corridor_speed_density_mix
assert CONFIG.long_corridor_fraction == 0.10
assert CONFIG.narrow_passage_fraction == 0.12
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (1, 2)
assert CHECKPOINTS == ((25, "checkpoint_3200.pt"),)
