"""Ready-to-run SA4-v3 from the formally accepted SA3-v3 c300 parent."""

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (
    ACTUATOR_DELAY_RANGE,
    SA4_SPEED_DENSITY_MIX,
    VEHICLE_SPEED_RATE,
    make_speed_density_curriculum_config,
)


_REPO = Path(__file__).resolve().parents[5]

PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa3_speed_density_v3_from_sa2r1_c100_ne1024_s42_p300_r1/"
    "checkpoint_38400.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "6888d4c4759413ddc81f1e1eb13f6fb02244823ea1977a97705f4ee2e5c79122"
)
PARENT_ACCEPTANCE_RECORD = (
    _REPO / "docs/freeze/sa3_v3_c300_parent_acceptance_20260821.json"
)
PARENT_ACCEPTANCE_RECORD_SHA256 = (
    "b941ef990651eb2102be418ec2cbb7a32e9b76a2b9ec3acf258ceb2890ae7272"
)
CURRICULUM_FREEZE = (
    _REPO / "docs/freeze/sim2real_speed_density_curriculum_v3_20260821.json"
)
CURRICULUM_FREEZE_SHA256 = (
    "a2d40f443fb151fca1b0e1f178e5adad600fddb9f90eacd128d6a8b4d363a24a"
)
P060_SUMMARY = (
    _REPO
    / "logs/gates/sa3_v3_p060_checkpoint_screen/screen_20260821_r1/"
    "SUMMARY.json"
)
P060_SUMMARY_SHA256 = (
    "2b95ab43c93aa141c796999966865e1eedf924fb98fd6b6c14797dc581f80e23"
)
RETENTION_SUMMARY = (
    _REPO
    / "logs/gates/sa3_v3_retention_screen/screen_20260821_r1/"
    "SUMMARY.json"
)
RETENTION_SUMMARY_SHA256 = (
    "82b2b99e7ddf305500ae94f6717eb289a3b77d8dcf551bd1fdcea94ba4fcd79a"
)

TRAINING_ITERATIONS = 300
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 50
TRAINING_STARTED = False


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
_verify_sha256(_parent_path, PARENT_CHECKPOINT_SHA256, "SA4-v3 parent")
_verify_sha256(
    PARENT_ACCEPTANCE_RECORD,
    PARENT_ACCEPTANCE_RECORD_SHA256,
    "SA3-v3 parent acceptance record",
)
_verify_sha256(
    CURRICULUM_FREEZE,
    CURRICULUM_FREEZE_SHA256,
    "speed-density v3 curriculum freeze",
)
_verify_sha256(P060_SUMMARY, P060_SUMMARY_SHA256, "SA3-v3 P060 summary")
_verify_sha256(
    RETENTION_SUMMARY,
    RETENTION_SUMMARY_SHA256,
    "SA3-v3 retention summary",
)

_acceptance = json.loads(PARENT_ACCEPTANCE_RECORD.read_text(encoding="utf-8"))
_accepted_checkpoint = _acceptance.get("accepted_checkpoint") or {}
_accepted_curriculum = _acceptance.get("curriculum_contract") or {}
if (
    _acceptance.get("schema") != "sa3_v3_parent_acceptance/v1"
    or _acceptance.get("decision") != "ACCEPT_C300_AS_SA4_V3_PARENT"
    or _accepted_checkpoint.get("name") != "c300"
    or _accepted_checkpoint.get("sha256") != PARENT_CHECKPOINT_SHA256
    or (_REPO / str(_accepted_checkpoint.get("path"))).resolve()
    != _parent_path.resolve()
    or (_REPO / str(_accepted_curriculum.get("path"))).resolve()
    != CURRICULUM_FREEZE.resolve()
    or _accepted_curriculum.get("sha256") != CURRICULUM_FREEZE_SHA256
):
    raise RuntimeError("SA4-v3 parent acceptance record does not authorize c300")

_p060 = json.loads(P060_SUMMARY.read_text(encoding="utf-8"))
if (
    _p060.get("status") != "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_SCREEN"
    or (_p060.get("promoted_for_retention") or [None])[0] != "c300"
):
    raise RuntimeError("SA4-v3 parent lacks the frozen P060 promotion")

_retention = json.loads(RETENTION_SUMMARY.read_text(encoding="utf-8"))
_c300_retention = next(
    (
        row
        for row in _retention.get("candidate_results", [])
        if row.get("checkpoint_name") == "c300"
    ),
    None,
)
if (
    _retention.get("status")
    != "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_RETENTION_SCREEN"
    or not bool(_retention.get("source_fingerprint_stable"))
    or _retention.get("recommended_parent_candidate") != "c300"
    or not _c300_retention
    or not bool(_c300_retention.get("retention_pass"))
):
    raise RuntimeError("SA4-v3 parent lacks the frozen retention pass")

_BASE = make_speed_density_curriculum_config(
    4,
    checkpoint=PARENT_CHECKPOINT,
)

CONFIG = replace(
    _BASE,
    name="e2e_sa4_k8_obb_speed_density_v3_from_sa3",
    description=(
        "SA4-v3 from formally accepted SA3-v3 c300 with low-density P080 "
        "introduction and the frozen speed-density curriculum."
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=True,
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_BASE.tags
    + (
        "from_formally_accepted_sa3_v3_c300",
        "parent_sha256_locked",
        "pilot300_fail_closed",
    ),
    notes=(
        f"{_BASE.notes} Parent c300 passed the frozen P060 screen and all "
        "native/narrow/P035 retention cells. The optimizer is reset at the "
        "SA3-to-SA4 boundary. This config is ready for an explicit launch "
        "preflight, but importing it never starts training."
    ),
)

_ALLOWED_OVERRIDES = {
    "name",
    "description",
    "no_resume_optimizer",
    "timesteps",
    "save_interval",
    "tags",
    "notes",
}
_drifted = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in _ALLOWED_OVERRIDES
    and getattr(CONFIG, field.name) != getattr(_BASE, field.name)
)
if _drifted:
    raise RuntimeError(f"SA4-v3 config drift outside allowed overrides: {_drifted}")

assert CONFIG.initial_stage == 4
assert CONFIG.fixed_stage is True
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True
assert CONFIG.timesteps == 38_400
assert CONFIG.save_interval == 50
assert CONFIG.speed_rate == VEHICLE_SPEED_RATE == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.long_corridor_speed_density_mix == SA4_SPEED_DENSITY_MIX
assert CONFIG.long_corridor_obstacle_count_mix is None
assert CONFIG.actuator_delay_range == ACTUATOR_DELAY_RANGE == (1, 2)
assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
assert CONFIG.actuator_motor_lag == 1.0
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert "not_ready_to_run" not in CONFIG.tags
assert TRAINING_STARTED is False
