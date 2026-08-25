"""Exact SA4-v3 continuation from conceptual c300 to c600.

The continuation restores the c300 optimizer and changes no behavioral,
sensor, actuator, scene, reward, network, or PPO setting. New checkpoint file
names count steps within this run; ``CONCEPTUAL_CHECKPOINTS`` maps them back to
the total SA4-v3 iteration count.
"""

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

from rnn_car_modular.configs.e2e_sa4_k8_obb_speed_density_v3_from_sa3 import (
    CONFIG as _SA4_V3,
)


_REPO = Path(__file__).resolve().parents[5]

PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa4_speed_density_v3_from_sa3r1_c300_ne1024_s42_p300_r1/"
    "checkpoint_38400.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "ccb95b1667e0f3f9330ffaed62ea8ad728221d270bf9b8a30b6135f421998e73"
)
SOURCE_CONFIG = (
    _REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa4_k8_obb_speed_density_v3_from_sa3.py"
)
SOURCE_CONFIG_SHA256 = (
    "30dee032487d786393030d7e4c436a10226338fa765ba4d72bef396b8c1764d3"
)
SELECTION_SUMMARY = (
    _REPO
    / "logs/gates/sa4_v3_checkpoint_screen/screen_20260822_r1/SUMMARY.json"
)
SELECTION_SUMMARY_SHA256 = (
    "046e2f390aec12dd7439109768977c4a085b2f368a08f17610f17ed04d32aa0e"
)
AUTHORIZATION_RECORD = (
    _REPO / "docs/freeze/sa4_v3_cont300_from_c300_v1.json"
)
AUTHORIZATION_RECORD_SHA256 = (
    "acdc9b1c4f775b28d80c7a376e3cfdfbe190ab60675c2ab4b4aadfea224bfe99"
)

PARENT_TOTAL_ITERATION = 300
CONTINUATION_ITERATIONS = 300
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 50
CONCEPTUAL_CHECKPOINTS = (
    (350, "checkpoint_6400.pt"),
    (400, "checkpoint_12800.pt"),
    (450, "checkpoint_19200.pt"),
    (500, "checkpoint_25600.pt"),
    (550, "checkpoint_32000.pt"),
    (600, "checkpoint_38400.pt"),
)

METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})
CONTINUATION_FIELDS = frozenset({"checkpoint", "no_resume_optimizer"})
ALLOWED_DIFFS = METADATA_FIELDS | CONTINUATION_FIELDS


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
_verify_sha256(_parent_path, PARENT_CHECKPOINT_SHA256, "SA4-v3 c300 parent")
_verify_sha256(SOURCE_CONFIG, SOURCE_CONFIG_SHA256, "SA4-v3 source config")
_verify_sha256(
    SELECTION_SUMMARY,
    SELECTION_SUMMARY_SHA256,
    "SA4-v3 c50-c300 Gate summary",
)
_verify_sha256(
    AUTHORIZATION_RECORD,
    AUTHORIZATION_RECORD_SHA256,
    "SA4-v3 c300-to-c600 authorization",
)

_selection = json.loads(SELECTION_SUMMARY.read_text(encoding="utf-8"))
_ranked = _selection.get("ranked_candidates") or []
if (
    _selection.get("schema") != "sa4_v3_checkpoint_screen_summary/v1"
    or _selection.get("status")
    != "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_SCREEN"
    or not bool(_selection.get("source_fingerprint_stable"))
    or not _ranked
    or _ranked[0].get("checkpoint_name") != "c300"
    or bool(_ranked[0].get("hard_gate_pass"))
):
    raise RuntimeError(
        "selection evidence must show c300 ranked first but still below the "
        "absolute SA4 Gate"
    )

_authorization = json.loads(AUTHORIZATION_RECORD.read_text(encoding="utf-8"))
if (
    _authorization.get("schema") != "sa4_v3_cont300_authorization/v1"
    or _authorization.get("decision")
    != "HUMAN_AUTHORIZED_C300_TO_C600_EXACT_CONTINUATION"
    or (_authorization.get("parent") or {}).get("sha256")
    != PARENT_CHECKPOINT_SHA256
    or (_authorization.get("continuation") or {}).get("additional_iterations")
    != CONTINUATION_ITERATIONS
):
    raise RuntimeError("authorization record does not permit this continuation")


CONFIG = replace(
    _SA4_V3,
    name="e2e_sa4_v3_cont300_from_c300",
    description=(
        "Exact same-recipe SA4-v3 continuation from conceptual c300 to c600, "
        "restoring optimizer state and saving every 50 iterations."
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=False,
    tags=_SA4_V3.tags
    + (
        "human_authorized_c300_to_c600",
        "same_recipe_continuation",
        "resume_sa4_v3_optimizer",
        "parent_sha256_locked",
    ),
    notes=(
        f"{_SA4_V3.notes} Human-authorized exact continuation from SA4-v3 "
        "conceptual c300. Restore the checkpoint optimizer and preserve all "
        "training fields. New checkpoint_6400 through checkpoint_38400 map "
        "to conceptual c350 through c600. After completion, compare c300-c600 "
        "using the unchanged frozen three-cell Gate. Do not auto-accept a "
        "parent and do not auto-launch SA5."
    ),
)

_drifted = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in ALLOWED_DIFFS
    and getattr(CONFIG, field.name) != getattr(_SA4_V3, field.name)
)
if _drifted:
    raise RuntimeError(
        "SA4-v3 continuation changed frozen training fields: "
        f"{_drifted}"
    )

assert CONFIG.initial_stage == 4
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == CONTINUATION_ITERATIONS * ROLLOUT_LENGTH == 38_400
assert CONFIG.save_interval == SAVE_INTERVAL
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is False
assert CONFIG.speed_rate == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert len(CONCEPTUAL_CHECKPOINTS) == 6

