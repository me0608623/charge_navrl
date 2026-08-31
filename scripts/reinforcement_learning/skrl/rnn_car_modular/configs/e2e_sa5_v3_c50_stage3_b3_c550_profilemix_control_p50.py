"""Fresh c550 control arm for the READY_NOT_RUN profile-ratio pilot."""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_v3_c50_stage3_b3_cont500_from_it100 import (
    CONFIG as _SOURCE,
)


REPO = Path(__file__).resolve().parents[5]
RUN_NAME = "sa5_v3_c50_stage3_b3_c550_profilemix_control_ne1024_s42_p50_r1"
PARENT_CHECKPOINT = (
    REPO
    / "logs/rnn_car/"
    "sa5_v3_c50_stage3_b3_cont500_from_it100_ne1024_s42_p500_r1/"
    "checkpoint_57600.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "501c79199b9c0a73d556d5ac1c1345cde347675a213197414af1183c5a2be4ed"
)
SOURCE_CONFIG = (
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_stage3_b3_cont500_from_it100.py"
)
SOURCE_CONFIG_SHA256 = (
    "cfb5c2cd7827696def1ff93f463a7f8defa6cff2644e918b24c2a90c86cee022"
)
CORRIDOR_DENSITY = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/corridor_density.py"
)
CORRIDOR_DENSITY_SHA256 = (
    "f6877a92699406809675f4013a86d4be6ba6d9db7cf93564106d19ad8cf91028"
)
DESIGN_RECORD = (
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_stage3_b3_c550_profile_ratio_matched_pilot_design_20260826.json"
)
DESIGN_RECORD_SHA256 = (
    "58461b7e1c6fe163bf70e5a6a8559148e54d8afa6c7278e10f3ec539495ba71c"
)

TRAINING_ITERATIONS = 50
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 25
METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
EXPECTED_SOURCE_DIFFS = frozenset({"checkpoint", "timesteps", "save_interval"})


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
    (PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256, "c550 parent checkpoint"),
    (SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "source continuation config"),
    (CORRIDOR_DENSITY, CORRIDOR_DENSITY_SHA256, "corridor density sampler"),
    (DESIGN_RECORD, DESIGN_RECORD_SHA256, "matched pilot design"),
):
    _verify_sha256(_path, _expected, _label)

_design = json.loads(DESIGN_RECORD.read_text(encoding="utf-8"))
_automatic = _design.get("automatic_actions") or {}
if (
    _design.get("decision") != "ESTABLISH_READY_NOT_RUN"
    or _design.get("parent", {}).get("sha256") != PARENT_CHECKPOINT_SHA256
    or _design.get("parent", {}).get("resume_rl_optimizer") is not True
    or _automatic.get("gpu_launch_authorized") is not False
    or _automatic.get("sa6_start_authorized") is not False
):
    raise RuntimeError("design record does not permit this READY_NOT_RUN control")

CONFIG = replace(
    _SOURCE,
    name="e2e_sa5_v3_c50_stage3_b3_c550_profilemix_control_p50",
    description=(
        "Fresh c550 process-matched control for the P080 profile-ratio pilot; "
        "READY_NOT_RUN."
    ),
    checkpoint=str(PARENT_CHECKPOINT),
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_SOURCE.tags
    + (
        "fresh_c550_profilemix_control",
        "matched_pilot_ready_not_run",
        "sa6_hold",
    ),
    notes=(
        f"{_SOURCE.notes} Fresh process-matched control from c550. The design "
        f"record {DESIGN_RECORD_SHA256} explicitly forbids automatic launch."
    ),
)

_actual_source_diffs = frozenset(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_SOURCE, field.name)
)
if _actual_source_diffs != EXPECTED_SOURCE_DIFFS:
    raise RuntimeError(
        "c550 control drifted: expected "
        f"{sorted(EXPECTED_SOURCE_DIFFS)}, got {sorted(_actual_source_diffs)}"
    )

assert CONFIG.checkpoint == str(PARENT_CHECKPOINT)
assert CONFIG.no_resume_optimizer is False
assert CONFIG.timesteps == 6400 and CONFIG.save_interval == 25
assert CONFIG.initial_stage == 5 and CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024 and CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.speed_rate == 0.7 and CONFIG.speed_rate_obs == "ego"
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
